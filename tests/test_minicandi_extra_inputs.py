"""miniCANDI extra-input modules (Amendment 3, chunk XM; pinned interface E): the RNA-seq track
stem, the cell-state FiLM from an expression vector and/or a cell id, and `extra_kwargs`."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

from minicandi import extra_inputs as X  # noqa: E402

B, L, A, D_COV, N_GENES, N_CELLS = 4, 16, 5, 8, 20, 6


def _n(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def _cell_batch(seed=0, avail=(True, False, True, False), ids=(1, 0, 3, 6)) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {"x_cell_vec": torch.randn(B, N_GENES, generator=g),
            "x_cell_vec_avail": torch.tensor(avail),
            "cell_id": torch.tensor(ids, dtype=torch.int64)}


def _perturb(m: torch.nn.Module, seed=1) -> None:
    """Move every parameter away from its initial value (a trained-like module)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.3 * torch.randn(p.shape, generator=g))


# ---- identity at initialisation --------------------------------------------------------------

@pytest.mark.parametrize("n_genes,n_cells", [(N_GENES, 0), (0, N_CELLS), (N_GENES, N_CELLS)])
def test_cell_state_identity_at_init(n_genes, n_cells):
    m = X.CellState(D_COV, n_genes=n_genes, n_cells=n_cells).eval()
    gamma, beta = m(_cell_batch())
    assert gamma.shape == beta.shape == (B, D_COV)
    assert torch.equal(gamma, torch.ones(B, D_COV)) and torch.equal(beta, torch.zeros(B, D_COV))
    cov = torch.randn(B, A, D_COV)
    assert torch.equal(X.apply_cell_state(cov, gamma, beta), cov)


def test_stem_starts_at_zero():
    s = X.ExtraTrackStem(2, 12)
    out = s(torch.randn(B, L, 2), torch.tensor([True, False, True, True]))
    assert out.shape == (B, L, 12) and torch.equal(out, torch.zeros(B, L, 12))


def test_cell_state_trains_from_identity():
    """The zero last layer still passes gradient, so the FiLM can leave the identity."""
    m = X.CellState(D_COV, n_genes=N_GENES, n_cells=N_CELLS).train()
    gamma, beta = m(_cell_batch())
    (gamma.sum() + beta.sum()).backward()
    assert m.g.net[-1].weight.grad.abs().sum() > 0


# ---- an unavailable input does not move the output -------------------------------------------

def test_stem_unavailable_ignores_values():
    s = X.ExtraTrackStem(2, 12)
    _perturb(s)
    avail = torch.tensor([True, False, True, False])
    x1 = torch.randn(B, L, 2)
    x2 = x1.clone()
    x2[~avail] = torch.randn(int((~avail).sum()), L, 2) * 100
    x2[1, 3, 0] = float("nan")
    o1, o2 = s(x1, avail), s(x2, avail)
    assert torch.equal(o1, o2)
    assert torch.equal(o1[1], s.null.detach().expand(L, 12))
    assert not torch.equal(o1[0], o1[2])          # available samples do read their values
    # no NaN gradient from an unavailable NaN input
    x3 = x2.clone().requires_grad_(True)
    s(x3, avail).sum().backward()
    assert torch.isfinite(x3.grad).all() and torch.equal(x3.grad[~avail], torch.zeros(2, L, 2))


def test_cell_state_unavailable_ignores_values():
    m = X.CellState(D_COV, n_genes=N_GENES, n_cells=N_CELLS).eval()
    _perturb(m)
    b1 = _cell_batch(seed=0)
    b2 = _cell_batch(seed=5)
    b2["x_cell_vec"][[0, 2]] = b1["x_cell_vec"][[0, 2]]     # same values where available
    g1, be1 = m(b1)
    g2, be2 = m(b2)
    assert torch.equal(g1, g2) and torch.equal(be1, be2)
    # a missing vector equals the null; a missing key equals "not available" everywhere
    none = {"cell_id": torch.zeros(B, dtype=torch.int64)}
    gz, bz = m(none)
    assert torch.allclose(gz, gz[0].expand(B, D_COV)) and torch.allclose(bz, bz[0].expand(B, D_COV))
    assert torch.equal(m(dict(none, x_cell_vec_avail=torch.zeros(B, dtype=torch.bool),
                              x_cell_vec=torch.randn(B, N_GENES)))[0], gz)
    # an id out of range reads as unknown
    bad = dict(none, cell_id=torch.tensor([0, -3, N_CELLS + 1, 99]))
    assert torch.equal(m(bad)[0], gz)
    # available inputs do move it
    b3 = _cell_batch(seed=0)
    b3["x_cell_vec"][0] += 1.0
    assert not torch.equal(m(b3)[0][0], g1[0])


def test_cell_state_train_dropout_goes_to_null():
    m = X.CellState(D_COV, n_genes=N_GENES, n_cells=N_CELLS, dropout=0.5)
    _perturb(m)
    m.eval()
    null_g, _ = m({"cell_id": torch.zeros(1, dtype=torch.int64)})
    m.train()
    big = 400
    b = {"x_cell_vec": torch.randn(big, N_GENES), "x_cell_vec_avail": torch.ones(big, dtype=torch.bool),
         "cell_id": torch.full((big,), 2, dtype=torch.int64)}
    torch.manual_seed(0)
    g, _ = m(b)
    at_null = torch.isclose(g, null_g.expand(big, D_COV), atol=1e-6).all(1).float().mean().item()
    # both parts dropped: probability 0.25; binomial SD at n=400 is 0.022
    assert 0.15 < at_null < 0.35
    m.dropout = 0.0
    assert torch.isclose(m(b)[0], null_g.expand(big, D_COV), atol=1e-6).all(1).sum() == 0


# ---- apply, extra_kwargs, parameter counts ---------------------------------------------------

def test_apply_cell_state_per_sample():
    cov = torch.randn(B, A, D_COV)
    gamma, beta = torch.randn(B, D_COV), torch.randn(B, D_COV)
    out = X.apply_cell_state(cov, gamma, beta)
    for b in range(B):
        assert torch.allclose(out[b], gamma[b] * cov[b] + beta[b])
    with pytest.raises(ValueError):
        X.apply_cell_state(cov, gamma[:, :3], beta)


def test_extra_kwargs():
    assert X.extra_kwargs({}) == {"rna_track": None, "cell_state": None}
    k = X.extra_kwargs({"rna_track": True, "cell_vector": True})
    assert k == {"rna_track": {"n_channels": 2},
                 "cell_state": {"n_genes": 2000, "n_cells": 0, "dropout": 0.1}}
    k = X.extra_kwargs({"cell_embedding": True, "n_cells": 40, "cell_state_dropout": 0.2})
    assert k["rna_track"] is None and k["cell_state"] == {"n_genes": 0, "n_cells": 40, "dropout": 0.2}
    for bad in ({"cell_embedding": True}, {"cell_state_dropout": 1.0}, {"rna_track": 1},
                {"cell_embedding": True, "n_cells": 0}):
        with pytest.raises(ValueError):
            X.extra_kwargs(bad)
    with pytest.raises(ValueError):
        X.CellState(D_COV)
    # the kwargs build the modules
    k = X.extra_kwargs({"rna_track": True, "cell_vector": True, "cell_embedding": True, "n_cells": 3})
    X.ExtraTrackStem(d=16, **k["rna_track"])
    X.CellState(32, **k["cell_state"])


def test_parameter_counts():
    H = X.HIDDEN
    g = H * H + H + H * H + H + H * 2 * 32 + 2 * 32           # ladder.model.G(64 -> 64 -> 64 -> 64)
    counts = {
        "rna_stem_d32": (_n(X.ExtraTrackStem(2, 32)), 2 * 32 + 32 + 32),
        "cell_vector_2000": (_n(X.CellState(32, n_genes=2000)),
                             2000 * H + H + H * H + H + H + g),
        "cell_embedding_40": (_n(X.CellState(32, n_cells=40)), 41 * H + g),
    }
    print("\nextra-input parameter counts (d=32, d_cov=32):",
          {k: v[0] for k, v in counts.items()})
    for k, (got, want) in counts.items():
        assert got == want, k
