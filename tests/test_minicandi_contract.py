"""miniCANDI contract: the NB loss, the loss mask, batch validation, the shared components."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.stats import nbinom

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "tools",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from minicandi import components as M  # noqa: E402
from minicandi import contract as C  # noqa: E402
from minicandi.covariates import CovariateEmbedding, CovariateEncoder  # noqa: E402


# ------------------------------------------------------------------------------- constants


def test_track_order_is_the_sorted_eic_vocabulary_then_the_control():
    panel = json.loads((_ROOT / "configs" / "panel.eic_full.json").read_text())["assays"]
    assert list(C.ASSAYS) == sorted(panel) == panel
    assert C.A == 36 and C.TRACKS[35] == "chipseq-control" == C.TRACKS[C.CONTROL_INDEX]
    assert C.K_META == 11 and C.META_COLUMNS[0] == "log2_depth"
    assert C.assay_class("H3K27me3") == "broad_histone"
    assert C.assay_class("H3K27ac") == "narrow_histone"
    assert C.assay_class("chipseq-control") == "control"


# ------------------------------------------------------------------------------- NB loss


def test_nb_nll_matches_scipy_and_closed_form():
    n = torch.tensor([[0.5, 2.0, 7.0], [1.0, 30.0, 3.0]])
    mu = torch.tensor([[0.3, 4.0, 12.0], [2.5, 0.1, 50.0]])
    y = torch.tensor([[0, 3, 20], [1, 0, 41]])
    p = (n.double() / (n.double() + mu.double())).numpy()
    ref = -nbinom.logpmf(y.numpy(), n.double().numpy(), p)
    # closed form: -[lgamma(y+n) - lgamma(n) - lgamma(y+1) + n log(n/(n+mu)) + y log(mu/(n+mu))]
    from math import lgamma, log
    cf = np.array([[-(lgamma(yy + nn) - lgamma(nn) - lgamma(yy + 1) + nn * log(nn / (nn + mm))
                      + yy * log(mm / (nn + mm)))
                    for nn, mm, yy in zip(nr, mr, yr)]
                   for nr, mr, yr in zip(n.tolist(), mu.tolist(), y.tolist())])
    np.testing.assert_allclose(ref, cf, rtol=1e-10)
    assert C.nb_nll(n, mu, y).item() == pytest.approx(ref.mean(), rel=1e-6)
    mask = torch.tensor([[True, False, True], [False, False, True]])
    assert C.nb_nll(n, mu, y, mask).item() == pytest.approx(ref[mask.numpy()].mean(), rel=1e-6)


def test_nb_nll_empty_mask_is_zero_with_a_graph():
    n = torch.ones(2, 3, requires_grad=True)
    mu = torch.ones(2, 3, requires_grad=True)
    out = C.nb_nll(n, mu, torch.zeros(2, 3), torch.zeros(2, 3, dtype=torch.bool))
    assert out.item() == 0.0
    out.backward()
    assert torch.isfinite(mu.grad).all()


# ------------------------------------------------------------------------------- batches


def _toy_batch(B=2, L=16, K=C.K_META, seed=0):
    g = torch.Generator().manual_seed(seed)
    x_avail = torch.zeros(B, C.A, dtype=torch.bool)
    x_avail[:, [1, 15, 21]] = True
    y_kind = torch.zeros(B, C.A, dtype=torch.uint8)
    y_kind[:, 1] = C.KIND_SAME
    y_kind[:, 15] = C.KIND_CORRECT
    y_kind[:, 16] = C.KIND_IMPUTE
    y_avail = y_kind > 0
    bm = torch.zeros(B, L, C.A, dtype=torch.bool)
    bm[:, ::3, 1] = True
    x = torch.randint(0, 50, (B, L, C.A), generator=g, dtype=torch.int32)
    x[bm | ~x_avail[:, None, :]] = 0
    c_in = torch.zeros(B, C.A, K)
    c_in[x_avail] = 1.0
    c_in[..., 0][x_avail] = 24.0
    c_out = torch.zeros(B, C.A, C.K_META)
    c_out[y_avail] = 1.0
    c_out[..., 0][y_avail] = 24.0
    return {"x_counts": x, "x_avail": x_avail, "x_bin_mask": bm,
            "x_dna": torch.randint(0, 5, (B, L * C.RES), generator=g).to(torch.uint8),
            "c_in": c_in, "c_out": c_out,
            "y_counts": torch.randint(0, 50, (B, L, C.A), generator=g, dtype=torch.int32),
            "y_avail": y_avail, "y_kind": y_kind,
            "meta": [{"cell": "x", "chrom": "chr1", "start_bin": 0}] * B}


def test_loss_mask_kinds():
    b = _toy_batch()
    m = C.loss_mask(b)
    assert m.shape == b["x_counts"].shape and m.dtype == torch.bool
    assert m[:, :, 15].all() and m[:, :, 16].all()                  # kinds 2, 1: every bin
    assert torch.equal(m[:, :, 1], b["x_bin_mask"][:, :, 1])         # kind 3: hidden bins only
    assert not (m[..., 1] & ~b["x_bin_mask"][..., 1]).any()          # never a visible bin
    assert not m[:, :, 21].any() and not m[:, :, 0].any()            # kind 0
    assert C.validate_batch(b) == (2, 16)


def test_validate_batch_rejects_off_contract():
    b = _toy_batch()
    bad = dict(b, x_counts=b["x_counts"].long())
    with pytest.raises(ValueError, match="dtype"):
        C.validate_batch(bad)
    bad = dict(b, c_out=torch.zeros(2, C.A, 18))
    with pytest.raises(ValueError, match="shape"):
        C.validate_batch(bad)
    x = b["x_counts"].clone()
    x[0, 0, 1] = 5                                                   # a hidden bin with counts
    with pytest.raises(ValueError, match="hidden"):
        C.validate_batch(dict(b, x_counts=x))
    k = b["y_kind"].clone()
    k[0, 1] = C.KIND_IMPUTE                                          # imputing a given track
    with pytest.raises(ValueError):
        C.validate_batch(dict(b, y_kind=k))


# ------------------------------------------------------------------------------- components


class _Toy(C.MiniCandi):
    """A tiny model built only from the shared components (per-bin MLP over tracks)."""

    def __init__(self, injection="film", mask_mode="token", dna="own", transform="offset",
                 K=C.K_META, leak=False):
        super().__init__()
        d = 8
        self.leak = leak
        self.tr = M.CountTransform(transform)
        self.inp = torch.nn.Linear(1, d)
        self.mask = M.MaskHandler(mask_mode, d)
        self.cin = M.CovariateInjector(injection, d, K)
        self.cout = M.CovariateInjector("film", d, C.K_META)
        self.dna = M.DnaTower(dna, d)
        self.mix = torch.nn.Linear(C.A * d, C.A * d)
        self.head = M.NBHead(d, self.tr)

    def forward(self, batch):
        B, L, Ac = batch["x_counts"].shape
        x = self.tr(batch["x_counts"], batch["c_in"][..., C.LOG2_DEPTH_COL])
        if self.leak:                       # deliberately reads the raw counts at hidden bins
            h = self.inp(x[..., None])
        else:
            h = self.inp(x[..., None])
            h, _vis = self.mask(h, batch["x_avail"], batch["x_bin_mask"])
        h = self.cin(h, batch["c_in"][:, None])
        h = h + self.dna(batch["x_dna"])[:, :, None, :]
        h = self.mix(h.reshape(B, L, -1)).reshape(B, L, Ac, -1)
        h = self.cout(h, batch["c_out"][:, None])
        return self.head(h, batch["c_out"])


@pytest.mark.parametrize("injection", ["concat", "film"])
@pytest.mark.parametrize("mask_mode", ["token", "drop"])
@pytest.mark.parametrize("dna", ["own", "off", "shuffled"])
def test_components_pass_the_wiring_check(injection, mask_mode, dna):
    torch.manual_seed(0)
    model = _Toy(injection, mask_mode, dna).eval()
    for p in model.head.lin.parameters():                   # leave the zero init: make it bite
        torch.nn.init.normal_(p, std=0.5)
    b = _toy_batch()
    out = model(b)
    assert out["n"].shape == out["mu"].shape == (2, 16, C.A)
    assert (out["n"] > 0).all() and (out["mu"] > 0).all()
    r = C.wiring_check(model, b)
    assert r["pass"] and r["n_perturbed"] > 0, r


def test_wiring_check_catches_a_leaky_model():
    torch.manual_seed(0)
    model = _Toy(leak=True).eval()
    for p in model.head.lin.parameters():
        torch.nn.init.normal_(p, std=0.5)
    assert not C.wiring_check(model, _toy_batch())["pass"]


def test_film_starts_as_identity_and_tokens_mode():
    inj = M.CovariateInjector("film", 8, C.K_META)
    h = torch.randn(2, 4, C.A, 8)
    c = torch.rand(2, 1, C.A, C.K_META) * 3
    assert torch.allclose(inj(h, c), h)
    tok = M.CovariateInjector("tokens", 8, C.K_META)
    assert torch.equal(tok(h, c), h)
    assert tok.tokens(torch.rand(2, C.A, C.K_META)).shape == (2, C.A, 8)
    with pytest.raises(M.NotApplicable):
        inj.tokens(torch.rand(2, C.A, C.K_META))
    with pytest.raises(ValueError):
        M.CovariateInjector("cross", 8, C.K_META)


def test_dna_tower_modes():
    x = torch.randint(0, 5, (2, 10 * C.RES)).to(torch.uint8)
    assert M.DnaTower("own", 6)(x).shape == (2, 10, 6)
    assert torch.equal(M.DnaTower("off", 6)(x), torch.zeros(2, 10, 6))
    assert sum(p.numel() for p in M.DnaTower("off", 6).parameters()) == 0
    sh = M.DnaTower("shuffled", 6).eval()
    assert torch.equal(sh(x), sh(x))                            # deterministic in eval mode
    own = M.DnaTower("own", 6)
    own.load_state_dict(sh.state_dict())
    assert not torch.allclose(own(x), sh(x))


def test_mask_handler_hides_exactly():
    h = torch.randn(1, 5, C.A, 4)
    avail = torch.zeros(1, C.A, dtype=torch.bool)
    avail[0, :3] = True
    bm = torch.zeros(1, 5, C.A, dtype=torch.bool)
    bm[0, 2, 0] = True
    for mode in ("token", "drop"):
        mh = M.MaskHandler(mode, 4)
        out, vis = mh(h, avail, bm)
        out2, _ = mh(h + 100 * (~vis)[..., None], avail, bm)
        assert torch.equal(out, out2)
        assert vis[0, 1, 0] and not vis[0, 2, 0] and not vis[0, 0, 5]
        if mode == "drop":
            assert (out[~vis] == 0).all()
        else:
            assert not torch.equal(out[0, 2, 0], out[0, 0, 5])     # hidden != missing token


@pytest.mark.parametrize("name", ["offset", "log1p", "asinh", "anscombe", "pearson", "raw"])
def test_count_transform_inverts(name):
    tr = M.CountTransform(name)
    y = torch.tensor([[[0, 1, 5, 40, 300]]]).permute(0, 2, 1).expand(1, 5, C.A).contiguous()
    d = torch.full((1, C.A), 25.0)
    mu = tr.add_back(tr(y, d), d)
    expect = y.float() + (1.0 if name == "offset" else 0.0)     # offset carries its +1
    big = (y >= 5) & (y <= 40) if name == "pearson" else y >= 5   # pearson clips at +-30 sd
    assert torch.allclose(mu[big], expect[big], rtol=1e-3)
    assert (mu > 0).all()


def test_depth_add_back_scales_with_target_depth():
    tr = M.CountTransform("offset")
    eta = torch.randn(2, 7, C.A)
    a = tr.add_back(eta, torch.full((2, C.A), 24.0))
    b = tr.add_back(eta, torch.full((2, C.A), 25.0))
    assert torch.allclose(b, 2 * a, rtol=1e-5)
    for name in M.INPUT_ONLY:
        x = M.CountTransform(name)(torch.randint(0, 9, (2, 64, C.A)), torch.full((2, C.A), 24.0))
        assert x.shape == (2, 64, C.A) and torch.isfinite(x).all()


def test_quantile_ties_share_a_score():
    y = torch.tensor([0, 0, 0, 5, 9, 9]).reshape(1, 6, 1).expand(1, 6, C.A).contiguous()
    x = M.CountTransform("quantile")(y, torch.full((1, C.A), 24.0))[0, :, 0]
    assert x[0] == x[1] == x[2] and x[4] == x[5] and x[2] < x[3] < x[4]


def test_qc_never_enters_c_out():
    enc = CovariateEncoder({"vocab": {"lab": ["L1"]}})
    row = {"depth_reads": "16777216", "run_type": "paired-ended", "read_length": "100",
           "dedup": "1", "mapq_threshold": "30", "lab": "L1", "nsc": "1.2", "frip": "0.3"}
    assert enc.c_out(row).shape == (C.K_META,)
    assert enc.c_in(row, qc_on_input=True).shape == (C.K_META + C.K_QC,)
    assert enc.c_out(row)[0] == 24.0 and enc.c_out(row, depth_scale=0.5)[0] == 23.0
    assert enc.c_out(row)[C.META_COLUMNS.index("lab_id")] == 1.0
    assert enc.c_out({"lab": "unseen"})[C.META_COLUMNS.index("lab_id")] == 0.0
    emb = CovariateEmbedding(C.K_META + C.K_QC, 5)
    assert emb(torch.from_numpy(enc.c_in(row, True))[None]).shape == (1, 5)


def test_ckpt_format_round_trips(tmp_path):
    model = _Toy()
    ck = C.make_ckpt(model, {"rung": 1}, 10, "abc", "fa02c35")
    C.check_ckpt(ck)
    torch.save(ck, tmp_path / "ckpt.pt")
    back = torch.load(tmp_path / "ckpt.pt", weights_only=False)
    assert set(back) == set(C.CKPT_KEYS) and back["param_count"] == model.param_count()
    m2 = _Toy()
    m2.load_state_dict(back["state_dict"])
