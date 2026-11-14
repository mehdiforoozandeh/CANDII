"""miniCANDI rung 6 (Perceiver IO): shapes, budget, copy-leak part 1, learning, gradients, padding."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "tools",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from minicandi import components as M  # noqa: E402
from minicandi import contract as C  # noqa: E402
from minicandi import synth  # noqa: E402
from minicandi.dataset import OnionDataset  # noqa: E402
from minicandi.rungs import perceiver as R6  # noqa: E402
from onion_eval import guardrail  # noqa: E402

L = 64                                   # 4 patches of 16 bins
BASE = json.loads((_ROOT / "tools/minicandi/configs/rung6/base.json").read_text())

#: every arm differs from base in one key (agent extras included)
ARMS = {
    "base": {},
    "dna_off": {"dna": "off"},
    "dna_shuffled": {"dna": "shuffled"},
    "film": {"covariate_injection": "film"},
    "concat": {"covariate_injection": "concat"},
    "mask_token": {"mask_mode": "token"},
    "scrambled_covariates": {"scramble_covariates": True},
    "qc_on_input": {"qc_on_input": True},
    "average_input": {"average_input": True},
    "latents_small": {"n_latents": 32},
    "latents_large": {"n_latents": 512},
    **{f"transform_{t}": {"count_transform": t} for t in M.TRANSFORMS if t != "offset"},
}


def _cfg(arm: str) -> dict:
    return {**BASE, **ARMS[arm], "context_bins": L}


def _model(arm: str, seed: int = 0, bite: bool = True) -> R6.Model:
    torch.manual_seed(seed)
    m = R6.Model(_cfg(arm))
    if bite:      # the NB head starts at zero, so an untrained model ignores its inputs and a
        g = torch.Generator().manual_seed(seed)  # wiring check would pass vacuously: add noise
        with torch.no_grad():                     # to every parameter
            for p in m.parameters():
                p.add_(0.05 * torch.randn(p.shape, generator=g))
    return m


@pytest.fixture(scope="module")
def tables(tmp_path_factory) -> Path:
    return synth.make_synthetic_tables(tmp_path_factory.mktemp("onion6"))


def _batch(tables, cfg: dict, n: int = 3, seed: int = 0) -> dict:
    ds = OnionDataset(tables, "train", {**cfg, "context_bins": L}, seed=seed)
    b = ds.batch(n)
    C.validate_batch(b)
    return b


@pytest.fixture(scope="module")
def batch(tables) -> dict:
    return _batch(tables, BASE)


@pytest.fixture(scope="module")
def avg_batch(tables) -> dict:
    return _batch(tables, {**BASE, "average_input": True})


@pytest.fixture(scope="module")
def qc_batch(tables) -> dict:
    return _batch(tables, {**BASE, "qc_on_input": True})


def _batch_for(arm, batch, avg_batch, qc_batch):
    return {"average_input": avg_batch, "qc_on_input": qc_batch}.get(arm, batch)


# ------------------------------------------------------------------------------------ config


def test_base_config_pins_the_plan_defaults():
    assert BASE["rung"] == 6 and BASE["mask_mode"] == "drop"
    assert BASE["covariate_injection"] == "tokens" and BASE["simpler_than_base"] is None
    assert R6.RUNG == 6 and R6.NAME == "perceiver"


def test_two_step_is_not_applicable():
    with pytest.raises(M.NotApplicable, match="later ablation on the winning design"):
        R6.Model({**BASE, "two_step": True})


# ------------------------------------------------------------------------------------ budget


@pytest.mark.parametrize("arm", list(ARMS))
def test_param_count_within_budget(arm):
    m = R6.Model(_cfg(arm))
    assert abs(m.param_count() - BASE["param_budget"]) <= 0.10 * BASE["param_budget"], \
        (arm, m.param_count())


def test_dna_off_has_no_dna_parameters():
    m = R6.Model(_cfg("dna_off"))
    assert not any(n.startswith(("dna.", "dna_patch", "dna_type")) for n, _ in m.named_parameters())


# ------------------------------------------------------------------------------------ outputs


@pytest.mark.parametrize("arm", list(ARMS))
def test_shapes_positivity_and_copy_leak_part1(arm, batch, avg_batch, qc_batch):
    b = _batch_for(arm, batch, avg_batch, qc_batch)
    m = _model(arm).eval()
    out = m(b)
    B = b["x_counts"].shape[0]
    assert out["n"].shape == out["mu"].shape == (B, L, C.A)
    assert out["n"].dtype == out["mu"].dtype == torch.float32
    assert (out["n"] > 0).all() and (out["mu"] > 0).all()
    assert torch.isfinite(out["n"]).all() and torch.isfinite(out["mu"]).all()
    r = C.wiring_check(m, b)
    assert r["pass"] and r["n_perturbed"] > 0, (arm, r)
    g = guardrail.copy_leak(m, [b])
    assert g["part1_wiring_pass"], (arm, g)
    assert g["hidden_input_max_abs_change"] <= guardrail.REL_TOL
    assert g["unavailable_input_max_abs_change"] <= guardrail.REL_TOL
    assert g["n_hidden_bins"] > 0 and g["n_unavailable_bins"] > 0


@pytest.mark.parametrize("arm", ["base", "mask_token"])
def test_wiring_check_fails_with_the_mask_bypassed(arm, batch, monkeypatch):
    """Planted negative control: with every bin and track called visible, hidden and missing
    counts reach the perturbed model and the check must fail (so the passes above are real)."""
    m = _model(arm).eval()
    assert C.wiring_check(m, batch)["pass"]
    monkeypatch.setattr(M.MaskHandler, "visible",
                        staticmethod(lambda a, mk: torch.ones_like(mk)))
    assert not C.wiring_check(m, batch)["pass"]
    assert not guardrail.copy_leak(m, [batch])["part1_wiring_pass"]


def test_output_depends_on_visible_input(batch):
    """Positive control for the wiring check: visible counts do move the output."""
    m = _model("base").eval()
    vis = M.MaskHandler.visible(batch["x_avail"], batch["x_bin_mask"])
    pert = dict(batch, x_counts=torch.where(vis, batch["x_counts"] + 7, batch["x_counts"]))
    with torch.no_grad():
        assert not torch.allclose(m(batch)["mu"], m(pert)["mu"])


# ------------------------------------------------------------------------------------ learning


@pytest.mark.parametrize("arm", ["base", "mask_token"])
def test_a_few_optimizer_steps_lower_the_nb_loss(arm, batch):
    m = _model(arm, bite=False).train()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    mask = C.loss_mask(batch)
    assert mask.any()

    def loss():
        o = m(batch)
        return C.nb_nll(o["n"], o["mu"], batch["y_counts"], mask)

    with torch.no_grad():
        first = loss().item()
    for _ in range(25):
        opt.zero_grad()
        lo = loss()
        lo.backward()
        opt.step()
    with torch.no_grad():
        last = loss().item()
    assert last < 0.9 * first, (first, last)


@pytest.mark.parametrize("arm", ["base", "film", "concat"])
def test_gradients_reach_covariate_encoders_and_dna_tower(arm, batch):
    m = _model(arm).train()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(2):          # FiLM's g starts at zero (identity); one step opens its path
        opt.zero_grad()
        o = m(batch)
        C.nb_nll(o["n"], o["mu"], batch["y_counts"], C.loss_mask(batch)).backward()
        if _ == 0:
            opt.step()

    def nz(mod):
        return any(p.grad is not None and p.grad.abs().sum() > 0 for p in mod.parameters())

    assert nz(m.cin.embed), "input covariate encoder gets no gradient"
    assert nz(m.cout.embed), "output covariate encoder gets no gradient"
    assert nz(m.dna), "DNA tower gets no gradient"
    assert nz(m.encode) and m.latents.grad.abs().sum() > 0


# ------------------------------------------------------------------------------------ padding


def _ragged_batch(seed: int = 0) -> dict:
    """Two windows with very different element counts: 3 tracks with hidden bins vs 1 track."""
    g = torch.Generator().manual_seed(seed)
    B = 2
    x_avail = torch.zeros(B, C.A, dtype=torch.bool)
    x_avail[0, [1, 15, 21]] = True
    x_avail[1, [16]] = True
    y_kind = torch.zeros(B, C.A, dtype=torch.uint8)
    y_kind[0, 1], y_kind[0, 16], y_kind[0, 15] = C.KIND_SAME, C.KIND_IMPUTE, C.KIND_CORRECT
    y_kind[1, 16], y_kind[1, 1] = C.KIND_SAME, C.KIND_IMPUTE
    y_avail = y_kind > 0
    bm = torch.zeros(B, L, C.A, dtype=torch.bool)
    bm[0, ::3, 1] = True
    bm[0, 16:32, 15] = True                      # one whole patch of track 15 hidden: dropped
    bm[1, 5:40, 16] = True
    x = torch.randint(0, 60, (B, L, C.A), generator=g, dtype=torch.int32)
    x[bm | ~x_avail[:, None, :]] = 0
    c_in = torch.zeros(B, C.A, C.K_META)
    c_in[x_avail] = 1.0
    c_in[..., 0][x_avail] = 23.0 + torch.rand(int(x_avail.sum()), generator=g) * 3
    c_out = torch.zeros(B, C.A, C.K_META)
    c_out[y_avail] = 1.0
    c_out[..., 0][y_avail] = 24.0
    b = {"x_counts": x, "x_avail": x_avail, "x_bin_mask": bm,
         "x_dna": torch.randint(0, 5, (B, L * C.RES), generator=g).to(torch.uint8),
         "c_in": c_in, "c_out": c_out,
         "y_counts": torch.randint(0, 60, (B, L, C.A), generator=g, dtype=torch.int32),
         "y_avail": y_avail, "y_kind": y_kind,
         "meta": [{"cell": f"c{i}", "chrom": "chr1", "start_bin": 0} for i in range(B)]}
    C.validate_batch(b)
    return b


def _item(b: dict, i: int) -> dict:
    return {k: (v[i:i + 1] if torch.is_tensor(v) else v[i:i + 1]) for k, v in b.items()}


@pytest.mark.parametrize("arm", ["base", "mask_token", "film", "dna_off"])
def test_ragged_items_padding_is_masked(arm):
    b = _ragged_batch()
    m = _model(arm).eval()
    with torch.no_grad():
        full = m(b)
        for i in range(2):
            one = m(_item(b, i))
            for k in ("n", "mu"):
                torch.testing.assert_close(full[k][i:i + 1], one[k], rtol=1e-5, atol=1e-6)
    r = C.wiring_check(m, b)
    assert r["pass"], r


def test_drop_mode_element_counts():
    """Drop mode: item 0 has 3 tracks x 4 patches minus 1 wholly hidden patch; item 1 has 1 track
    x 4 patches minus 1 wholly hidden patch (bins 16-31 inside its hidden run 5-39)."""
    b = _ragged_batch()
    vis = M.MaskHandler.visible(b["x_avail"], b["x_bin_mask"])
    valid = vis.view(2, L // 16, 16, C.A).any(2).reshape(2, -1)
    assert valid.sum(1).tolist() == [11, 3]
    tok = torch.randn(2, valid.shape[1], 5)
    out, keep = R6._gather_valid(tok, valid)
    assert out.shape == (2, 11, 5) and keep.sum(1).tolist() == [11, 3]
    assert torch.equal(out[1, :3], tok[1][valid[1]]) and (out[1, 3:] == 0).all()
