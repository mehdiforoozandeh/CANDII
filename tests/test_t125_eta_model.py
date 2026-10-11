"""t125 part 1 — `tools/t125/eta/model.py`: g + f with the depth offset, prediction, the run loader.

Written from plan/ETA_OFFSET_AND_H_DESIGN.md "Part 1" and the plan's pinned interfaces (chunk
t125-P1c): eta_in = log(1 + X) - log d (d in millions of reads), log mu' = log d' + eta_out.
"""
from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.stats import nbinom

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "tools" / "t125", ROOT / "tools" / "t118"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ladder import base, encoding, synth  # noqa: E402
from ladder import model as lmodel  # noqa: E402
from ladder import pairs as lpairs  # noqa: E402
from eta import offset  # noqa: E402
from eta import model as em  # noqa: E402
from eta import pairs as ep  # noqa: E402

DEV = torch.device("cpu")
N0, SIGMA0 = 5.0, 0.5
ROW1_ALL = ("A", "Aold", "B", "C", "D")
ROW2_ALL = ("A2", "B2", "C2", "D2")


# ---------------------------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synth_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth_eta")
    manifest, cov = synth.make_products(root, seed=0)
    rows = lpairs.read_manifest(manifest)
    return {"root": root, "manifest": manifest, "cov": cov, "rows": rows,
            "log_d": offset.depth_table(rows)}


@pytest.fixture(scope="module")
def stats(synth_dir):
    """Knots from an eta sample of the synthetic chr1 bins (every product, its own depth)."""
    from ladder.data import Corpus
    corpus = Corpus(synth_dir["root"])
    xs = np.concatenate([offset.transform_eta(corpus.get(pid, "counts", "chr1"), ld)
                         for pid, ld in synth_dir["log_d"].items()])
    return {"knots_x": base.knot_stats(xs), "n0": N0, "sigma0": SIGMA0}


@pytest.fixture(scope="module")
def encoder(synth_dir):
    rows = encoding.read_covariates(synth_dir["cov"])
    return encoding.Encoder.fit(rows, lpairs.fit_pids(synth_dir["rows"], "all"))


def _corpus(synth_dir):
    from ladder.data import Corpus
    return Corpus(synth_dir["root"])


def _perturb(lad: em.EtaLadder, seed: int = 0, scale: float = 0.05) -> em.EtaLadder:
    """Move g (and the form's own weights) off the identity, so a test is not trivially passed."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in list(lad.g.parameters()) + list(lad.form.parameters()):
            p.add_(scale * torch.randn(p.shape, generator=gen))
    return lad.eval()


def _cov(encoder, src, tgt) -> torch.Tensor:
    return torch.from_numpy(encoder.encode_pair(src, tgt)).reshape(1, -1)


def _haloed_eta(X: np.ndarray, c: int, log_d_src: float) -> np.ndarray:
    Xp = np.zeros(X.shape[0] + 2 * c, dtype=np.float64)
    Xp[c:c + X.shape[0]] = X
    return offset.transform_eta(Xp, log_d_src)


T1B, T1D = "T1__base__base", "T1__depth__7.5M"
T2B, T2P = "T2__base__base", "T2__pe__pe"


# ---------------------------------------------------------------------------------------------
# load_form and EtaLadder construction
# ---------------------------------------------------------------------------------------------


def test_load_form_every_rung(stats):
    for rung in ep.RUNGS:
        form = em.load_form(rung, stats)
        assert isinstance(form, base.FForm)
        assert form.space == "counts"
        assert bool(form.per_bin) == (ep.row_of(rung) == 2)
    assert em.load_form("A", stats).n_theta == 14          # t125 A: spread set by level
    assert em.load_form("Aold", stats).n_theta == 3        # t118 A: one spread per pair
    assert type(em.load_form("A", stats)).__module__ == "eta.fforms.form_a"
    assert type(em.load_form("Aold", stats)).__module__ == "ladder.fforms.form_a"
    with pytest.raises(KeyError):
        em.load_form("E", stats)


@pytest.mark.parametrize("rung", ep.RUNGS)
def test_eta_ladder_attributes(rung, stats):
    lad = em.EtaLadder(rung, stats)
    assert lad.rung == rung and lad.space == "counts"
    assert lad.context == lad.form.context and lad.n_theta == lad.form.n_theta
    assert lad.per_bin == (ep.row_of(rung) == 2)
    assert isinstance(lad.g, lmodel.GBin if lad.per_bin else lmodel.G)
    assert lad.g.net[0].in_features == 2 * encoding.N_COV + (1 if lad.per_bin else 0)


# ---------------------------------------------------------------------------------------------
# the "do nothing" map is the depth scale d'/d
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("rung", ("A", "Aold", "B", "A2", "D2", "C", "D"))
def test_identity_at_init_scales_by_depth_ratio(rung, synth_dir, stats, encoder):
    """With f's output equal to its input (eta_out = eta_in), mu' = (1 + X) d'/d."""
    lad = em.EtaLadder(rung, stats).eval()
    c = lad.context
    corpus = _corpus(synth_dir)
    ld_s, ld_t = synth_dir["log_d"][T1B], synth_dir["log_d"][T1D]
    assert math.isclose(math.exp(ld_t - ld_s), 0.25)
    start, stop = 100, 400
    Xw = corpus.window(T1B, "counts", "chr19", start - c, stop + c)
    cov = _cov(encoder, T1B, T1D)
    cond = cov if lad.per_bin else lad.theta(cov)[0]
    loc, disp = em.predict_window_eta(lad, Xw, ld_s, ld_t, cond, DEV)
    assert loc.shape == (stop - start,) and loc.dtype == np.float32
    X = corpus.get(T1B, "counts", "chr19")[start:stop].astype(np.float64)
    mu = np.exp(np.clip(loc.astype(np.float64), math.log(lmodel.MU_MIN), math.log(lmodel.MU_MAX)))
    np.testing.assert_allclose(mu, (1.0 + X) * 0.25, rtol=2e-5)
    np.testing.assert_allclose(disp, math.log(N0), rtol=1e-6)


def test_identity_forward_is_offset_eta(stats):
    """forward at init: loc = log1p(X) - log d + log d', per sample of the batch."""
    lad = em.EtaLadder("A", stats).eval()
    X = np.array([[0, 1, 5, 20], [3, 0, 0, 7]], dtype=np.float64)
    ld_s = np.array([math.log(30.0), math.log(7.5)])
    ld_t = np.array([math.log(15.0), math.log(60.0)])
    x = torch.from_numpy(np.stack([offset.transform_eta(X[i], ld_s[i]) for i in range(2)]))
    cov = torch.zeros(2, 2 * encoding.N_COV)
    with torch.no_grad():
        loc, disp = lad(cov, x, torch.tensor(ld_t, dtype=torch.float32))
        loc_eta, disp_eta = lad.form_out(cov, x)
    want = np.log1p(X) - ld_s[:, None] + ld_t[:, None]
    np.testing.assert_allclose(loc.numpy(), want, atol=1e-5)
    np.testing.assert_allclose(loc_eta.numpy(), np.log1p(X) - ld_s[:, None], atol=1e-5)
    assert torch.equal(disp, disp_eta)
    # zero count and pad map to -log d
    assert offset.transform_eta(np.zeros(1), math.log(30.0))[0] == pytest.approx(-math.log(30.0))


@pytest.mark.parametrize("rung", ("A", "C", "A2", "D2"))
def test_forward_equals_form_out_plus_offset(rung, stats, encoder):
    lad = _perturb(em.EtaLadder(rung, stats), seed=3)
    c = lad.context
    rng = np.random.default_rng(1)
    X = rng.poisson(3.0, (2, 50 + 2 * c))
    x = torch.from_numpy(offset.transform_eta(X, math.log(30.0)))
    cov = torch.cat([_cov(encoder, T1B, T1D), _cov(encoder, T2B, T2P)])
    ld_t = torch.tensor([math.log(7.5), math.log(30.0)], dtype=torch.float32)
    with torch.no_grad():
        loc, disp = lad(cov, x, ld_t)
        loc_eta, disp_eta = lad.form_out(cov, x)
    assert loc.shape == (2, 50) and disp.shape == (2, 50)
    torch.testing.assert_close(loc, loc_eta + ld_t[:, None])
    assert torch.equal(disp, disp_eta)


# ---------------------------------------------------------------------------------------------
# whole-chromosome prediction
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("rung", ("A", "Aold", "C", "D"))
@pytest.mark.parametrize("chunk", (lmodel.CHUNK, 300))
def test_predict_chrom_eta_equals_form_out(rung, chunk, synth_dir, stats, encoder):
    """Whole chromosome, halo included (X = 0 outside: eta pad = -log d_src)."""
    lad = _perturb(em.EtaLadder(rung, stats), seed=5)
    c = lad.context
    X = _corpus(synth_dir).get(T1B, "counts", "chr22")
    ld_s, ld_t = synth_dir["log_d"][T1B], synth_dir["log_d"][T1D]
    theta = lad.theta(_cov(encoder, T1B, T1D))[0].detach()
    loc, disp = em.predict_chrom_eta(lad, X, ld_s, ld_t, theta, DEV, chunk=chunk)
    assert loc.shape == disp.shape == X.shape and loc.dtype == disp.dtype == np.float32
    x = torch.from_numpy(_haloed_eta(X, c, ld_s))[None]
    with torch.no_grad():
        want_loc, want_disp = lad.form_out(_cov(encoder, T1B, T1D), x)
    np.testing.assert_allclose(loc, want_loc[0].numpy() + ld_t, atol=1e-4, rtol=1e-5)
    np.testing.assert_allclose(disp, want_disp[0].numpy(), atol=1e-4, rtol=1e-5)


@pytest.mark.parametrize("rung", ("A2", "D2"))
@pytest.mark.parametrize("chunk", (lmodel.CHUNK, 300))
def test_predict_chrom_bins_eta_equals_per_bin(rung, chunk, synth_dir, stats, encoder):
    lad = _perturb(em.EtaLadder(rung, stats), seed=7)
    c = lad.context
    X = _corpus(synth_dir).get(T1B, "counts", "chr22")
    ld_s, ld_t = synth_dir["log_d"][T1B], synth_dir["log_d"][T1D]
    cov = _cov(encoder, T1B, T1D)
    loc, disp = em.predict_chrom_bins_eta(lad, X, ld_s, ld_t, cov, DEV, chunk=chunk)
    assert loc.shape == disp.shape == X.shape and loc.dtype == np.float32
    x = torch.from_numpy(_haloed_eta(X, c, ld_s))[None]
    with torch.no_grad():
        want_loc, want_disp = lad.form_out(cov, x)            # g evaluated at every bin directly
    np.testing.assert_allclose(loc, want_loc[0].numpy() + ld_t, atol=1e-4, rtol=1e-5)
    np.testing.assert_allclose(disp, want_disp[0].numpy(), atol=1e-4, rtol=1e-5)


def test_predict_chrom_bins_eta_level_cache(synth_dir, stats, encoder):
    lad = _perturb(em.EtaLadder("B2", stats), seed=8)
    X = _corpus(synth_dir).get(T2B, "counts", "chr22")
    ld_s, ld_t = synth_dir["log_d"][T2B], synth_dir["log_d"][T2P]
    cov = _cov(encoder, T2B, T2P)
    cache: dict = {}
    a = em.predict_chrom_bins_eta(lad, X, ld_s, ld_t, cov, DEV, level_cache=cache)
    assert cache and "uniq" in cache and "inv" in cache
    b = em.predict_chrom_bins_eta(lad, X, ld_s, ld_t, cov, DEV, level_cache=cache)
    c = em.predict_chrom_bins_eta(lad, X, ld_s, ld_t, cov, DEV)
    for u, v, w in zip(a, b, c):
        assert np.array_equal(u, v) and np.array_equal(u, w)
    with pytest.raises(ValueError):
        em.predict_chrom_bins_eta(lad, X[:-5], ld_s, ld_t, cov, DEV, level_cache=cache)


def test_nocov_averaged_map_equals_explicit_mean(synth_dir, stats, encoder):
    """Row 2 nocov: theta_bar(x) = mean_j g(x, C_j, C'_j), then f."""
    lad = _perturb(em.EtaLadder("A2", stats), seed=9)
    tp = lpairs.train_pairs(synth_dir["rows"], "all")
    covs = torch.from_numpy(np.stack([encoder.encode_pair(p["source_pid"], p["target_pid"])
                                      for p in tp]))
    X = _corpus(synth_dir).get(T1B, "counts", "chr22")
    ld_s, ld_t = synth_dir["log_d"][T1B], synth_dir["log_d"][T1D]
    loc, disp = em.predict_chrom_bins_eta(lad, X, ld_s, ld_t, covs, DEV, average=True)
    x = torch.from_numpy(_haloed_eta(X, 0, ld_s))[None]
    with torch.no_grad():
        theta_bar = torch.stack([lad.theta(covs[j:j + 1], x) for j in range(len(tp))]).mean(0)
        want_loc, want_disp = lad.form(x, theta_bar)
    np.testing.assert_allclose(loc, want_loc[0].numpy() + ld_t, atol=1e-4, rtol=1e-5)
    np.testing.assert_allclose(disp, want_disp[0].numpy(), atol=1e-4, rtol=1e-5)
    # row 1 nocov: one theta, the mean over the training pairs (t118's nocov_theta, reused)
    lad1 = _perturb(em.EtaLadder("B", stats), seed=10)
    th = lmodel.nocov_theta(lad1, covs)
    with torch.no_grad():
        assert torch.allclose(th, lad1.theta(covs).mean(0))


# ---------------------------------------------------------------------------------------------
# window prediction
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("rung", ("A", "C", "D", "A2", "C2", "D2"))
def test_predict_window_eta_matches_chrom(rung, synth_dir, stats, encoder):
    lad = _perturb(em.EtaLadder(rung, stats), seed=11)
    c = lad.context
    corpus = _corpus(synth_dir)
    X = corpus.get(T1B, "counts", "chr22")
    n = X.shape[0]
    ld_s, ld_t = synth_dir["log_d"][T1B], synth_dir["log_d"][T1D]
    cov = _cov(encoder, T1B, T1D)
    if lad.per_bin:
        full = em.predict_chrom_bins_eta(lad, X, ld_s, ld_t, cov, DEV)
        cond = cov
    else:
        cond = lad.theta(cov)[0].detach()
        full = em.predict_chrom_eta(lad, X, ld_s, ld_t, cond, DEV)
    for start, stop in ((200, 456), (0, 64), (n - 100, n)):        # interior, both chrom ends
        Xw = corpus.window(T1B, "counts", "chr22", start - c, stop + c)
        loc, disp = em.predict_window_eta(lad, Xw, ld_s, ld_t, cond, DEV)
        assert loc.shape == disp.shape == (stop - start,) and loc.dtype == np.float32
        np.testing.assert_allclose(loc, full[0][start:stop], atol=1e-4, rtol=1e-5)
        np.testing.assert_allclose(disp, full[1][start:stop], atol=1e-4, rtol=1e-5)


def test_predict_window_eta_nocov_average(synth_dir, stats, encoder):
    lad = _perturb(em.EtaLadder("D2", stats), seed=12)
    tp = lpairs.train_pairs(synth_dir["rows"], "T1")
    covs = torch.from_numpy(np.stack([encoder.encode_pair(p["source_pid"], p["target_pid"])
                                      for p in tp]))
    corpus = _corpus(synth_dir)
    X = corpus.get(T1B, "counts", "chr22")
    ld_s, ld_t = synth_dir["log_d"][T1B], synth_dir["log_d"][T1D]
    full = em.predict_chrom_bins_eta(lad, X, ld_s, ld_t, covs, DEV, average=True)
    c = lad.context
    Xw = corpus.window(T1B, "counts", "chr22", 300 - c, 500 + c)
    loc, disp = em.predict_window_eta(lad, Xw, ld_s, ld_t, covs, DEV, average=True)
    np.testing.assert_allclose(loc, full[0][300:500], atol=1e-4, rtol=1e-5)
    np.testing.assert_allclose(disp, full[1][300:500], atol=1e-4, rtol=1e-5)


def test_predict_window_eta_bad_shapes(stats):
    """A haloed window must hold at least one bin beyond the 2c halo."""
    lad = em.EtaLadder("C", stats)
    with pytest.raises(ValueError):
        em.predict_window_eta(lad, np.zeros(2 * lad.context), 0.0, 0.0,
                              lad.form.init_theta(), DEV)


# ---------------------------------------------------------------------------------------------
# NLL with the offset
# ---------------------------------------------------------------------------------------------


def test_nll_with_offset_on_constant_window(stats):
    lad = em.EtaLadder("A", stats).eval()
    L, Xv, yv = 64, 3.0, 5.0
    ld_s, ld_t = math.log(30.0), math.log(7.5)
    x = torch.from_numpy(offset.transform_eta(np.full((1, L), Xv), ld_s))
    cov = torch.zeros(1, 2 * encoding.N_COV)
    y = torch.full((1, L), yv)
    with torch.no_grad():
        loc, disp = lad(cov, x, torch.tensor([ld_t], dtype=torch.float32))
        got = float(lad.nll(loc, disp, y))
        got_m = float(lad.nll(loc, disp, y, torch.ones(1, L, dtype=torch.bool)))
    mu = (1.0 + Xv) * math.exp(ld_t - ld_s)
    want = -nbinom.logpmf(yv, N0, N0 / (N0 + mu))
    assert got == pytest.approx(want, rel=1e-5)
    assert got_m == pytest.approx(want, rel=1e-5)
    assert float(lmodel.nb_nll(loc, disp, y).mean()) == pytest.approx(got, rel=1e-6)
    # a mask selects bins
    y2 = y.clone()
    y2[0, :10] = 50.0
    m = torch.ones(1, L, dtype=torch.bool)
    m[0, :10] = False
    with torch.no_grad():
        assert float(lad.nll(loc, disp, y2, m)) == pytest.approx(want, rel=1e-5)


# ---------------------------------------------------------------------------------------------
# the run loader and the predictor
# ---------------------------------------------------------------------------------------------


def _write_ckpt(run_dir: Path, synth_dir, stats, encoder, rung: str, model: str, g: str = "all",
                seed: int = 0, log_depth: dict | None = None) -> em.EtaLadder:
    lad = _perturb(em.EtaLadder(rung, stats), seed=21)
    rows = synth_dir["rows"]
    tp = lpairs.train_pairs(rows, g)
    config = {
        "run_name": ep.run_name(rung, g, model, seed), "rung": rung, "g": g, "space": "counts",
        "model": model, "seed": seed, "n_cov": encoding.N_COV,
        "fit_pids": lpairs.fit_pids(rows, g), "train_pairs": tp,
        "manifest_md5": lmodel.md5_path(synth_dir["manifest"]),
        "covariates_md5": lmodel.md5_path(synth_dir["cov"]),
        "log_depth": dict(log_depth if log_depth is not None else synth_dir["log_d"]),
        "t125": {"offset": "log1p(X) - log(d/1e6)", "d_unit": 1e6},
        **({"row": 2} if lad.per_bin else {}),
    }
    ckpt = {"g": {k: v.clone() for k, v in lad.g.state_dict().items()},
            "form": {k: v.clone() for k, v in lad.form.state_dict().items()},
            "stats": {"knots_x": [float(v) for v in stats["knots_x"]], "n0": N0, "sigma0": SIGMA0},
            "encoder": encoder.state(), "config": config, "step": 7, "val_nll": 1.25}
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, run_dir / "ckpt.pt")
    (run_dir / "config.json").write_text(json.dumps(config))
    return lad


@pytest.mark.parametrize("rung,model", [("A", "real"), ("Aold", "nocov"), ("C", "real"),
                                        ("A2", "real"), ("B2", "nocov")])
def test_load_run_round_trip(rung, model, tmp_path, synth_dir, stats, encoder):
    # a planted log depth for T1D, distinct from the manifest's: predict must read the config's
    log_depth = dict(synth_dir["log_d"])
    log_depth[T1D] = log_depth[T1D] + 0.3
    run_dir = tmp_path / "run"
    lad = _write_ckpt(run_dir, synth_dir, stats, encoder, rung, model, log_depth=log_depth)
    pred = em.load_run(run_dir, synth_dir["root"], synth_dir["cov"], synth_dir["manifest"],
                       device="cpu")
    assert isinstance(pred, em.EtaPredictor)
    assert pred.rung == rung and pred.space == "counts" and pred.model == model
    assert pred.g_id == "all" and pred.seed == 0 and pred.step == 7 and pred.val_nll == 1.25
    assert pred.run_name == ep.run_name(rung, "all", model, 0)
    assert pred.context == lad.context and pred.n_theta == lad.n_theta
    assert pred.per_bin == lad.per_bin
    assert pred.log_d == log_depth
    assert pred.fit_pids == lpairs.fit_pids(synth_dir["rows"], "all")
    assert pred.cov_pids(T1B, T1D) == (T1B, T1D)
    for k, v in lad.g.state_dict().items():
        assert torch.equal(pred.ladder.g.state_dict()[k], v)

    X = _corpus(synth_dir).get(T1B, "counts", "chr22")
    tp = lpairs.train_pairs(synth_dir["rows"], "all")
    covs = torch.from_numpy(np.stack([encoder.encode_pair(p["source_pid"], p["target_pid"])
                                      for p in tp]))
    cov = _cov(encoder, T1B, T1D)
    for cov_tgt in (T1D, T2P):        # trained pair, then a shuffle (the wrong target's depth)
        loc, disp = pred.predict(T1B, T1B, cov_tgt, "chr22")
        ld_s, ld_t = log_depth[T1B], log_depth[cov_tgt]
        cov = _cov(encoder, T1B, cov_tgt)
        if lad.per_bin:
            want = em.predict_chrom_bins_eta(lad, X, ld_s, ld_t,
                                             covs if model == "nocov" else cov, DEV,
                                             average=model == "nocov")
        else:
            th = lmodel.nocov_theta(lad, covs) if model == "nocov" else lad.theta(cov)[0]
            want = em.predict_chrom_eta(lad, X, ld_s, ld_t, th.detach(), DEV)
        np.testing.assert_allclose(loc, want[0], atol=1e-4, rtol=1e-5)
        np.testing.assert_allclose(disp, want[1], atol=1e-4, rtol=1e-5)
    # swap: C' = C so d' = d
    loc_sw, _ = pred.predict(T1B, T1B, T1B, "chr22")
    loc_tr, _ = pred.predict(T1B, T1B, T1D, "chr22")
    if model == "nocov":       # the map does not read C: only the offset moves
        np.testing.assert_allclose(loc_tr - loc_sw, log_depth[T1D] - log_depth[T1B], atol=1e-4)

    # window prediction: the matching slice of the chromosome
    n = X.shape[0]
    for start, stop in ((100, 300), (n - 40, n)):
        lw, dw = pred.predict_window(T1B, T1B, T1D, "chr22", start, stop)
        assert lw.shape == (stop - start,)
        np.testing.assert_allclose(lw, loc_tr[start:stop], atol=1e-4, rtol=1e-5)

    # theta / describe as Predictor's
    th = pred.theta(T1B, T1D)
    if pred.per_bin:
        assert th.shape == (12, lad.n_theta)
        d = pred.describe(T1B, T1D)
        assert d["levels_x"] == [float(v) for v in stats["knots_x"]]
        assert len(d["response"]["loc"]) == 12 and len(d["response"]["disp"]) == 12
    else:
        assert th.shape == (lad.n_theta,)
        assert isinstance(pred.describe(T1B, T1D), dict)


def test_describe_response_has_no_offset(tmp_path, synth_dir, stats, encoder):
    """Row 2 at the identity: the response at level x is loc = x (eta out, no log d')."""
    run_dir = tmp_path / "run"
    _write_ckpt(run_dir, synth_dir, stats, encoder, "A2", "real")
    ck = torch.load(run_dir / "ckpt.pt", weights_only=True)
    lad0 = em.EtaLadder("A2", stats)
    ck["g"] = {k: v.clone() for k, v in lad0.g.state_dict().items()}
    torch.save(ck, run_dir / "ckpt.pt")
    pred = em.load_run(run_dir, synth_dir["root"], None, None, device="cpu")
    d = pred.describe(T1B, T1D)
    np.testing.assert_allclose(d["response"]["loc"], stats["knots_x"], atol=1e-5)


def test_mean_and_quantiles_nb(tmp_path, synth_dir, stats, encoder):
    run_dir = tmp_path / "run"
    _write_ckpt(run_dir, synth_dir, stats, encoder, "A", "real")
    pred = em.load_run(run_dir, synth_dir["root"], None, None, device="cpu")
    loc = np.array([math.log(1e-6), 0.0, math.log(4.0), math.log(1e8)])
    disp = np.full(4, math.log(N0))
    mu = pred.mean(loc)
    np.testing.assert_allclose(mu, [1e-4, 1.0, 4.0, 1e6])
    q = pred.quantiles(loc, disp, 0.5)
    assert q.shape == (4,)
    np.testing.assert_allclose(q, nbinom.ppf(0.5, N0, N0 / (N0 + mu)))
    qs = pred.quantiles(loc, disp, [0.1, 0.9])
    assert qs.shape == (2, 4) and np.all(qs[0] <= qs[1])


def test_load_run_md5_warning_and_bad_model(tmp_path, synth_dir, stats, encoder):
    run_dir = tmp_path / "run"
    _write_ckpt(run_dir, synth_dir, stats, encoder, "A", "real")
    other = tmp_path / "other.tsv"
    other.write_text("not the covariates\n")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        em.load_run(run_dir, synth_dir["root"], other, synth_dir["manifest"], device="cpu")
    assert any("md5" in str(x.message) for x in w)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        em.load_run(run_dir, synth_dir["root"], synth_dir["cov"], synth_dir["manifest"],
                    device="cpu")
    # t125 has no ids twin
    ck = torch.load(run_dir / "ckpt.pt", weights_only=True)
    ck["config"]["model"] = "ids"
    torch.save(ck, run_dir / "ckpt.pt")
    with pytest.raises(ValueError):
        em.load_run(run_dir, synth_dir["root"], None, None, device="cpu")
