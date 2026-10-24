"""t128 rungs DS / DSBOX — the symmetric encoder-decoder D (`fforms/form_ds.py`, `model.GSym`).

  * one FormD is both encoder and decoder (shared weights); context 60; identity at init;
  * log sigma_z = head[1], so sigma_z = 1 at init (PI ruling (i));
  * theta[:256] = g(c, C_clean) reads c only, theta[256:] = g(C_clean, c') reads c' and c's assay;
  * C_clean is per assay, assay entries copied from c; form A init copies (zeros by PI ruling (ii)),
    the form D box keeps knobs in [lo, hi] and reproduces the init value;
  * the predictor path, a checkpoint round trip, and the old rungs' g classes are unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "tools" / "t118", REPO / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ladder import encoding, model  # noqa: E402
from ladder.fforms import form_d, form_ds  # noqa: E402

STATS = {"knots_x": np.linspace(0, 5, 12), "n0": 5.0, "sigma0": 0.5}
N = encoding.N_COV
C = form_ds.CONTEXT


def _covs(k, seed, assays=None):
    """k covariate vectors: random knobs, a one-hot assay (given indices, or random)."""
    g = torch.Generator().manual_seed(seed)
    c = torch.randn(k, N, generator=g)
    a = (torch.randint(0, form_ds.N_ASSAY, (k,), generator=g) if assays is None
         else torch.as_tensor(assays))
    c[:, form_ds.ASSAY_SLICE] = 0.0
    c[torch.arange(k), form_ds.N_KNOB + a] = 1.0
    return c


def _x(k, w, seed):
    return torch.log1p(torch.poisson(torch.full((k, w), 3.0),
                                     generator=torch.Generator().manual_seed(seed)))


def _ladder(rung="DS", seed=0) -> model.Ladder:
    """A Ladder with random g last layer, head and C_clean, so nothing is trivially the identity."""
    torch.manual_seed(seed)
    lad = model.Ladder(rung, "counts", STATS)
    with torch.no_grad():
        lad.g.mlp[-1].weight.normal_(0.0, 0.3)
        lad.g.mlp[-1].bias.normal_(0.5, 0.3)
        lad.form.f.head.weight.normal_(0.0, 0.3)
        lad.form.f.head.bias.normal_(0.0, 0.3)
        lad.g.cclean.normal_(0.0, 1.0)
        if lad.g.box:
            lad.g.box_lo.fill_(-1.0)
            lad.g.box_hi.fill_(2.0)
    return lad


def test_constants_and_shapes():
    assert form_ds.FILM == 256 and form_ds.CTX1 == 30 and form_ds.CONTEXT == 60
    assert form_ds.N_KNOB == 13 and form_ds.N_ASSAY == 7
    lad = _ladder("DS", 1)
    assert isinstance(lad.g, model.GSym) and lad.g.box is False
    assert lad.context == 60 and lad.n_theta == 512 and lad.per_bin is False
    x = _x(3, 200 + 2 * C, 0)
    loc, disp = lad(torch.cat([_covs(3, 1), _covs(3, 2)], 1), x)
    assert loc.shape == (3, 200) and disp.shape == (3, 200)
    th = lad.theta(torch.cat([_covs(3, 1), _covs(3, 2)], 1))
    mu, ls = lad.form.encode(x, th[:, :256])
    assert mu.shape == ls.shape == (3, 200 + C)
    with pytest.raises(ValueError):
        lad.form(_x(3, 2 * C, 0), th)
    with pytest.raises(ValueError):
        lad.form(x, th[:, :256])


@pytest.mark.parametrize("rung", ["DS", "DSBOX"])
def test_identity_at_init(rung):
    lad = model.Ladder(rung, "counts", STATS)
    cov = torch.cat([_covs(4, 1), _covs(4, 2)], 1)
    th = lad.theta(cov)
    assert torch.equal(th, lad.form.init_theta().expand(4, -1))
    x = _x(4, 300, 0)
    loc, disp = lad(cov, x)
    assert torch.equal(loc, x[:, C:-C])
    assert torch.allclose(disp, torch.full_like(disp, float(np.log(5.0))))
    _, log_sigma_z = lad.form.encode(x, th[:, :256])
    assert torch.equal(log_sigma_z, torch.zeros_like(log_sigma_z))


def test_encode_decode_share_one_formd():
    lad = _ladder("DS", 2)
    form = lad.form
    ref = form_d.FormD("counts", STATS)
    assert [n for n, _ in form.named_children()] == ["f"]
    assert sum(p.numel() for p in form.parameters()) == sum(p.numel() for p in ref.parameters())
    assert all(k.startswith(("f.convs.", "f.head.")) for k in form.state_dict())
    cov = torch.cat([_covs(5, 3), _covs(5, 4)], 1)
    x = _x(5, 260, 1)
    th = lad.theta(cov)
    # encoder and decoder are the same module: running form.f by hand reproduces both
    mu, _ = form.f(x, th[:, :256])
    loc, disp = form.f(mu, th[:, 256:])
    l2, d2 = lad(cov, x)
    assert torch.equal(loc, l2) and torch.equal(disp, d2)
    # log sigma_z is head[1]
    _, ls = form.encode(x, th[:, :256])
    assert torch.allclose(ls, form.f(x, th[:, :256])[1] - form.f.disp0)


def test_each_half_reads_one_side():
    lad = _ladder("DS", 3)
    c, ct = _covs(6, 5), _covs(6, 6)
    th = lad.theta(torch.cat([c, ct], 1))
    # change c' -> the encoder half is unchanged, the decoder half moves
    th2 = lad.theta(torch.cat([c, _covs(6, 7)], 1))
    assert torch.equal(th[:, :256], th2[:, :256])
    assert not torch.allclose(th[:, 256:], th2[:, 256:])
    # change c's knobs (same assay) -> the decoder half is unchanged, the encoder half moves
    c3 = c.clone()
    c3[:, form_ds.KNOB_SLICE] = torch.randn(6, form_ds.N_KNOB, generator=torch.Generator().manual_seed(8))
    th3 = lad.theta(torch.cat([c3, ct], 1))
    assert torch.equal(th[:, 256:], th3[:, 256:])
    assert not torch.allclose(th[:, :256], th3[:, :256])


def test_cclean_of_selects_by_assay_and_copies_assay():
    lad = model.Ladder("DS", "counts", STATS)
    g = lad.g
    vals = torch.randn(7, 13, generator=torch.Generator().manual_seed(9))
    g.set_cclean_init(vals)
    assert torch.equal(g.cclean.data, vals)
    c = _covs(4, 10, assays=[2, 2, 5, 0])
    cc = g.cclean_of(c)
    assert torch.equal(cc[:, form_ds.ASSAY_SLICE], c[:, form_ds.ASSAY_SLICE])
    assert torch.equal(cc[0], cc[1])
    assert not torch.allclose(cc[0, :13], cc[2, :13])
    assert torch.equal(cc[2, :13], vals[5]) and torch.equal(cc[3, :13], vals[0])
    tab = g.cclean_table()
    assert list(tab) == list(encoding.ASSAYS) and np.allclose(tab["H3K27me3"], vals[2].numpy())


def test_form_a_zero_init_and_set_box_noop():
    """PI ruling (ii): form A initialises C_clean at zeros (the fit-pid mean in z units)."""
    g = model.Ladder("DS", "counts", STATS).g
    assert torch.equal(g.cclean_knobs(), torch.zeros(7, 13))
    g.set_cclean_init(torch.ones(7, 13))
    g.set_cclean_init(torch.zeros(7, 13))
    assert torch.equal(g.cclean_knobs(), torch.zeros(7, 13))
    with pytest.warns(UserWarning):
        g.set_box(torch.full((7, 13), -1.0), torch.full((7, 13), 1.0))
    assert torch.equal(g.box_lo, torch.zeros(7, 13)) and torch.equal(g.box_hi, torch.zeros(7, 13))


def test_box_bounds_and_init():
    """Form D: knobs stay in [lo, hi]; the init reproduces the base's value inside the box."""
    lad = model.Ladder("DSBOX", "counts", STATS)
    g = lad.g
    assert g.box is True and lad.form.cclean_box is True
    gen = torch.Generator().manual_seed(11)
    lo = torch.randn(7, 13, generator=gen)
    hi = lo + torch.rand(7, 13, generator=gen) * 3.0
    hi[0, 0] = lo[0, 0]                                     # a flat knob
    g.set_box(lo, hi)
    with torch.no_grad():
        g.cclean.copy_(torch.randn(7, 13, generator=gen) * 10.0)
    k = g.cclean_knobs()
    assert bool((k >= lo - 1e-6).all()) and bool((k <= hi + 1e-6).all())
    # interior values are reproduced
    v = lo + (hi - lo) * (0.1 + 0.8 * torch.rand(7, 13, generator=gen))
    g.set_cclean_init(v)
    assert torch.allclose(g.cclean_knobs(), v, atol=1e-5)
    assert float(g.cclean[0, 0]) == 0.0
    # edge values land within 0.02 of the range (u clamped at +-4)
    edge = torch.where(torch.rand(7, 13, generator=torch.Generator().manual_seed(12)) < 0.5, lo, hi)
    g.set_cclean_init(edge)
    assert bool(((g.cclean_knobs() - edge).abs() <= 0.02 * (hi - lo) + 1e-6).all())
    assert float(g.cclean.abs().max()) <= 4.0


def test_predict_chrom_matches_forward():
    lad = _ladder("DSBOX", 4)
    c1, c2 = _covs(2, 13), _covs(2, 14)
    x1 = _x(2, 400, 3)
    loc, _ = lad(torch.cat([c1, c2], 1), x1)
    theta = lad.theta(torch.cat([c1[:1], c2[:1]], 1))[0]
    X = torch.expm1(x1[0]).numpy()
    lp, _ = model.predict_chrom(lad, X, theta, "cpu", chunk=100)
    assert np.allclose(lp[C:-C], loc[0].detach().numpy(), atol=1e-5)


def test_checkpoint_round_trip(tmp_path):
    lad = _ladder("DSBOX", 5)
    path = tmp_path / "ckpt.pt"
    torch.save({"g": lad.g.state_dict(), "form": lad.form.state_dict()}, path)
    ck = torch.load(path)
    new = model.Ladder("DSBOX", "counts", STATS)
    new.g.load_state_dict(ck["g"])
    new.form.load_state_dict(ck["form"])
    cov = torch.cat([_covs(3, 15), _covs(3, 16)], 1)
    x = _x(3, 300, 4)
    a, b = lad(cov, x), new(cov, x)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert torch.equal(new.g.box_lo, lad.g.box_lo) and torch.equal(new.g.box_hi, lad.g.box_hi)


def test_describe_keys():
    lad = _ladder("DS", 6)
    th = lad.theta(torch.cat([_covs(1, 17), _covs(1, 18)], 1))[0]
    d = lad.form.describe(th)
    assert set(d) == {"film_gamma_enc", "film_beta_enc", "film_gamma_dec", "film_beta_dec"}
    assert np.array(d["film_gamma_enc"]).shape == (4, 32)


def test_old_rungs_build_their_old_g():
    assert type(model.Ladder("D", "counts", STATS).g) is model.G
    assert type(model.Ladder("DED", "counts", STATS).g) is model.GEncDecD
    assert type(model.Ladder("A", "counts", STATS).g) is model.G
