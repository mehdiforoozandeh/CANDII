"""t396 C9: tools/t396/calib_cov.py on synthetic arrays in the C6 contract format."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from t396 import calib_cov as CV  # noqa: E402
from t396 import covconv as CC  # noqa: E402

R, RB, L, B = 8, 1024, 64, 4           # regions, region bins, window bins, infer batch
REACH = CC.REACH


def _kernel(rng):
    k = np.zeros((2, 17))
    k[0, REACH:REACH + 4] = [1.0, 0.9, 0.8, 0.3]      # plus strand covers bins i .. i + 3
    k[1, REACH - 3:REACH + 1] = [0.3, 0.8, 0.9, 1.0]
    return k


def _store_convert(mu_sub, n_sub, w, stranded, segs):
    mu = np.empty(mu_sub.shape[:2])
    n = np.empty_like(mu)
    for r in range(len(mu_sub)):
        for lo, hi in segs[r]:
            mu[r, lo:hi], n[r, lo:hi] = CC.convert_nb_cov(
                mu_sub[r, lo:hi], n_sub[r, lo:hi], w, stranded, CC.identity_rho())
    return mu.astype(np.float32), n.astype(np.float32)


def _make(tmp, targets, corr: bool, seed=0, ref_factor=3.0):
    """targets: list of (id, assay, stranded). Writes truth/, arrays/, ref/ under tmp."""
    rng = np.random.default_rng(seed)
    truth, arrays, ref = (tmp / x for x in ("truth", "arrays", "ref"))
    for d in (truth, arrays, ref):
        d.mkdir(parents=True, exist_ok=True)
    segs = CV.segments(R, RB, L, B)
    for tid, assay, stranded in targets:
        w = _kernel(rng)
        nrs = np.array([1000.0, 1200.0])
        base = rng.gamma(4.0, 5.0, size=(R, RB // 32, 1, 1)).repeat(32, 1)   # smooth mean
        mu_sub = np.broadcast_to(base, (R, RB, 2, 1))[..., 0].copy()
        shape = 2.0
        if corr:   # one Gamma rate shared by 5 neighbouring bins
            g = rng.gamma(shape, 1.0 / shape, size=(R, RB // 5 + 1, 2)).repeat(5, 1)[:, :RB]
        else:
            g = rng.gamma(shape, 1.0 / shape, size=(R, RB, 2))
        if not stranded:                         # one channel holds the whole track
            mu_sub[..., 1] = 0.0
        s = rng.poisson(mu_sub * g)
        n_sub = np.full((R, RB, 2), shape)
        if not stranded:
            n_sub[..., 1] = 1e4
        wk = CV.kernel_row(w, nrs, stranded)
        wk2 = wk if wk.ndim == 2 else wk[None]
        y = np.zeros((R, RB), np.int64)
        for k in range(-REACH, REACH + 1):
            for st in ((0, 1) if stranded else (0,)):
                p = wk2[st, REACH + k]
                if p <= 0:
                    continue
                src = s[:, :, st]
                th = rng.binomial(src, min(p, 1.0))
                sh = np.zeros_like(th)
                if k >= 0:
                    sh[:, k:] = th[:, :RB - k]
                else:
                    sh[:, :RB + k] = th[:, -k:]
                y += sh
        mu, n = _store_convert(mu_sub, n_sub, wk, stranded, segs)
        np.savez_compressed(truth / f"{tid}.npz", mid=np.array(tid), assay=np.array(assay),
                            stranded=np.array(stranded), kernel=w, n_reads_strand=nrs,
                            y=y.astype(np.int32), s=s.astype(np.uint16))
        np.savez_compressed(arrays / f"{tid}.npz", mu_sub=mu_sub.astype(np.float32),
                            n_sub=n_sub.astype(np.float32), mu=mu, n=n,
                            region_indices=np.arange(R, dtype=np.int64))
        key = f"item/{tid}/{CV.REGIME}"
        np.savez(ref / (hashlib.sha1(key.encode()).hexdigest() + ".npz"), key=np.array(key),
                 mu=mu, n=(n * ref_factor).astype(np.float32))
    return truth, arrays, ref


def _run(tmp, truth, arrays, ref, seed=0, infer_batch=B):
    common = ["--truth", str(truth), "--arrays", str(arrays), "--ref-pred-store", str(ref),
              "--seed", str(seed), "--window-bins", str(L), "--infer-batch", str(infer_batch)]
    fit = tmp / f"fit{seed}.npz"
    rec = tmp / f"rec{seed}.jsonl"
    assert CV.main(["fit", *common, "--out", str(fit)]) == 0
    assert CV.main(["score", *common, "--fit", str(fit), "--out", str(rec)]) == 0
    rows = [json.loads(x) for x in rec.read_text().splitlines()]
    return fit, rec, rows


def _macro(rows, variant):
    v, _ = CV.macro_ratio([r for r in rows if r.get("variant") == variant])
    return v


TARGETS = [(f"T{i}", "H3K4me3", True) for i in range(4)] + \
          [(f"U{i}", "DNase", False) for i in range(3)] + [("S0", "H3K27ac", True)]


@pytest.fixture(scope="module")
def indep(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("indep")
    truth, arrays, ref = _make(tmp, TARGETS, corr=False)
    return (tmp, truth, arrays, ref) + _run(tmp, truth, arrays, ref)


@pytest.fixture(scope="module")
def corr(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("corr")
    truth, arrays, ref = _make(tmp, TARGETS, corr=True, seed=1)
    return (tmp, truth, arrays, ref) + _run(tmp, truth, arrays, ref)


def test_segments_per_forward_batch():
    s = CV.segments(3, 16384, 1024, 8)
    assert s[0] == [(0, 8192), (8192, 16384)] and len(s) == 3
    s = CV.segments(2, 4096, 1024, 3)        # chunks of 3 windows, the last short
    assert s[1] == [(0, 3072), (3072, 4096)]
    s = CV.segments(2, 2048, 1024, 3)        # B > windows per region: flat batches
    assert s == [[(0, 2048)], [(0, 1024), (1024, 2048)]]


def test_records_shape_and_identity_reproduction(indep):
    rows = indep[-1]
    var = [r for r in rows if "variant" in r]
    assert len(var) == 6 * len(TARGETS)
    assert {r["variant"] for r in var} == set(CV.VARIANTS)
    assert all(r["regions"] == "even" and r["n_regions"] == R // 2 for r in var)
    for r in var:
        if r["variant"] == "rs_independent":
            assert r["n_rel_diff_identity"] < 1e-5      # per-batch conversion reproduces n
    assert {r["assay"] for r in rows if r.get("kind") == "rho"} == {"H3K4me3", "DNase", "H3K27ac"}


def test_wrong_batching_does_not_reproduce(tmp_path, indep):
    _, truth, arrays, ref = indep[:4]
    _, _, rows = _run(tmp_path, truth, arrays, ref, infer_batch=2)
    d = [r["n_rel_diff_identity"] for r in rows if r.get("variant") == "rs_independent"]
    assert max(d) > 1e-3


def test_independent_starts_cov_matches_independent(indep):
    rows = indep[-1]
    ri, rii = _macro(rows, "rs_independent"), _macro(rows, "rs_cov_loo")
    assert abs(rii / ri - 1) < 0.02, (ri, rii)


def test_correlated_starts_cov_moves_toward_one(corr):
    rows = corr[-1]
    ri, rii = _macro(rows, "rs_independent"), _macro(rows, "rs_cov_loo")
    assert ri > 1.1, ri                          # the independent sum understates the variance
    assert abs(np.log(rii)) < 0.5 * abs(np.log(ri)), (ri, rii)


def test_scored_target_absent_from_its_rho(corr):
    fit_path, rows = corr[4], corr[-1]
    for r in rows:
        if r.get("variant") == "rs_cov_loo":
            assert r["target"] not in r["rho_from"]
            if not r["fallback"]:
                assert all(t.startswith(r["target"][0]) for t in r["rho_from"])
        if r.get("variant") == "rs_cov_insample":
            assert r["target"] in r["rho_from"]
    f = CV.Fit(fit_path)
    before, ids, _ = f.rho_for("T0", "loo")
    f.num = f.num.copy()
    f.num[f.pos["T0"]] *= 50.0                   # the scored target's own parts do not enter
    after, _, _ = f.rho_for("T0", "loo")
    assert "T0" not in ids and np.array_equal(before, after)
    assert not np.array_equal(f.rho_for("T0", "insample")[0], before)


def test_single_target_assay_falls_back(corr):
    rows = corr[-1]
    s0 = [r for r in rows if r.get("target") == "S0"]
    loo = next(r for r in s0 if r["variant"] == "rs_cov_loo")
    assert loo["fallback"] and set(loo["rho_from"]) == {f"T{i}" for i in range(4)}  # stranded
    assert all(r["fallback"] for r in s0 if r["variant"] in ("rs_scaled_loo", "ref_scaled_loo"))
    assert all(np.isfinite(r["c"]) for r in s0 if "c" in r)


def test_equal_recalibration_of_reference(corr):
    rows = corr[-1]
    assert _macro(rows, "ref_stored") > 1.5      # n inflated 3x: overconfident
    assert abs(np.log(_macro(rows, "ref_scaled_loo"))) < 0.1
    assert abs(np.log(_macro(rows, "rs_scaled_loo"))) < 0.1


def test_table(tmp_path, corr):
    tmp, truth, arrays, ref = corr[:4]
    rec0 = corr[5]
    _, rec1, _ = _run(tmp, truth, arrays, ref, seed=1)
    sj = tmp_path / "scores.json"
    sj.write_text(json.dumps({"scorable": {"imputation": ["H3K4me3", "DNase"]}}))
    legb = tmp_path / "legB.tsv"
    legb.write_text("space\tmodel\tregions\tcalibration_ratio\n"
                    "start_sum\trs_halves\tall\t1.2\noverlap_count\trs_halves\tall\t1.3\n"
                    "start_sum\trcp_halves\tall\t0.9\n")
    out = tmp_path / "tab"
    assert CV.main(["table", "--records", str(rec0), str(rec1), "--scores-json", str(sj),
                    "--legB-table", str(legb), "--out", str(out)]) == 0
    with open(out / "table.tsv") as fh:
        t = list(csv.DictReader(fh, delimiter="\t"))
    assert list(t[0]) == ["variant", "macro", "calibration_ratio", "seed_range", "per_seed",
                          "n_targets", "flag"]
    assert {r["variant"] for r in t} == set(CV.VARIANTS) and len(t) == 12
    assert {r["macro"] for r in t} == {"all_assays", "scorable"}
    for r in t:
        assert float(r["seed_range"]) == 0.0      # the same arrays scored as seeds 0 and 1
        assert r["per_seed"].count("seed") == 2
    assert "in_sample_upper_bound" in next(r["flag"] for r in t if r["variant"] == "rs_cov_insample")
    assert "fallback_assays=1" in next(r["flag"] for r in t if r["variant"] == "rs_cov_loo")
    with open(out / "rho.tsv") as fh:
        rho = list(csv.DictReader(fh, delimiter="\t"))
    lag0 = [r for r in rho if r["lag"] == "0" and r["strand_pair"] in ("00", "11")]
    assert all(float(r["rho"]) == 1.0 for r in lag0)
    lag1 = [float(r["rho"]) for r in rho if r["lag"] == "1" and r["strand_pair"] == "00"
            and r["seed"] == "0" and r["assay"] == "H3K4me3"]
    assert lag1 and lag1[0] > 0.3
    nat = (out / "native_start.tsv").read_text().splitlines()
    assert len(nat) == 2 and "rs_halves" in nat[1]


def test_ref_key_matches_predict_store():
    ps = pytest.importorskip("onion_eval.predict_store")
    assert CV.ref_key_file("X") == ps.key_file(f"item/X/{CV.REGIME}")


# ------------------------------------------------------------------ iteration 2 (A4)
def _fit(tmp, truth, arrays, ref, half, name, seed=0):
    out = tmp / f"{name}.npz"
    args = ["fit", "--truth", str(truth), "--seed", str(seed), "--window-bins", str(L),
            "--infer-batch", str(B), "--fit-regions", half, "--out", str(out)]
    if arrays is not None:
        args += ["--arrays", str(arrays)]
    if ref is not None:
        args += ["--ref-pred-store", str(ref)]
    assert CV.main(args) == 0
    return out


def test_fit_regions_even_uses_only_even_regions(tmp_path, corr):
    _, truth, arrays, ref = corr[:4]
    f_odd = CV.Fit(corr[4])
    p_even = _fit(tmp_path, truth, arrays, ref, "even", "fit_even")
    f_even = CV.Fit(p_even)
    assert (f_odd.fit_regions, f_even.fit_regions) == ("odd", "even")
    # the rho parts of the even fit are those of estimate_rho on the even regions alone
    t = np.load(truth / "T0.npz")
    m = np.load(arrays / "T0.npz")
    ev = np.arange(0, R, 2)
    num, den = CC.estimate_rho(t["s"][ev].astype(float), m["mu_sub"][ev].astype(float),
                               m["n_sub"][ev].astype(float))
    i = f_even.pos["T0"]
    assert np.allclose(f_even.num[i], num) and np.allclose(f_even.den[i], den)
    assert not np.allclose(f_odd.num[f_odd.pos["T0"]], num)
    # score after an even fit scores the odd regions
    rec = tmp_path / "rec_even.jsonl"
    assert CV.main(["score", "--truth", str(truth), "--arrays", str(arrays), "--ref-pred-store",
                    str(ref), "--seed", "0", "--window-bins", str(L), "--infer-batch", str(B),
                    "--fit", str(p_even), "--out", str(rec)]) == 0
    rows = [json.loads(x) for x in rec.read_text().splitlines()]
    assert {r["regions"] for r in rows if "variant" in r} == {"odd"}


def test_fit_count_output_only(tmp_path, corr):
    _, truth, _, ref = corr[:4]
    p = _fit(tmp_path, truth, None, ref, "odd", "fit_count")
    f = CV.Fit(p)
    assert not f.has_arrays and f.has_ref
    assert set(f.targets) == {t for t, _, _ in TARGETS}
    assert all(np.isfinite(v) for v in f.c_ref_loo.values())
    assert all(np.isnan(v) for v in f.c_rs_loo.values())
    assert not f.num.any() and not f.den.any()
    # c of the count-output store equals that of the full fit (same store, same half)
    full = CV.Fit(corr[4])
    for t in f.targets:
        assert f.c_ref_loo[t] == pytest.approx(full.c_ref_loo[t], rel=1e-9)
    # score refuses a fit without the rho parts
    assert CV.main(["score", "--truth", str(truth), "--arrays", str(corr[2]), "--ref-pred-store",
                    str(ref), "--seed", "0", "--window-bins", str(L), "--infer-batch", str(B),
                    "--fit", str(p), "--out", str(tmp_path / "x.jsonl")]) == 2


def test_fit_needs_an_input(tmp_path, corr):
    _, truth, _, _ = corr[:4]
    assert CV.main(["fit", "--truth", str(truth), "--seed", "0",
                    "--out", str(tmp_path / "f.npz")]) == 2
