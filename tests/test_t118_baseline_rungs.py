"""t118 — `tools/t118/baseline_rungs.py`: the noSolution, QuantileMatching and oracle rungs.

Products are written with the t112 writer itself (`tools/t112/bin25.py::write_npz`), so the loader
is tested against the exact layout the corpus has: one npz per product and space, one array per
main chromosome keyed by name, counts uint32, p float32.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from scipy import integrate
from scipy.stats import norm, poisson

from candi.metrics import spearman

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


br = _load("t118_baseline_rungs", ROOT / "tools" / "t118" / "baseline_rungs.py")
bin25 = _load("t112_bin25_for_t118", ROOT / "tools" / "t112" / "bin25.py")

#: bins per chromosome; chr19/chr21 scored, chr22 validation, the rest training
N_BINS = {"chr1": 4000, "chr2": 3000, "chr19": 1500, "chr21": 1200, "chr22": 800, "chrX": 900}


def _write_product(root: Path, pid: str, counts: dict, pval: dict) -> Path:
    d = root / pid
    bin25.write_npz(d / "counts25.npz", {c: np.asarray(v, np.uint32) for c, v in counts.items()})
    bin25.write_npz(d / "pval25.npz", {c: np.asarray(v, np.float32) for c, v in pval.items()})
    return d


def _synthetic(seed=0):
    rng = np.random.default_rng(seed)
    counts = {c: rng.negative_binomial(1.0, 1.0 / (1.0 + rng.gamma(0.5, 4.0, n)))
              for c, n in N_BINS.items()}
    pval = {c: (rng.gamma(0.4, 1.5, n) * (rng.random(n) < 0.7)).astype(np.float32)
            for c, n in N_BINS.items()}
    return counts, pval


def _blacklist(tmp_path: Path) -> Path:
    bed = tmp_path / "blacklist.bed"
    # chr19 bins 10..19 (250..500 bp), chr21 bin 0 (by 1 bp), chr1 bins 0..99 (training: ignored)
    bed.write_text("# test\nchr19\t250\t500\nchr21\t0\t1\nchr1\t0\t2500\n")
    return bed


# ---------------------------------------------------------------------------------------------
# QuantileMatching
# ---------------------------------------------------------------------------------------------


def _brute_tie_block(s, t):
    ss, ts = np.sort(s, kind="stable"), np.sort(t)
    vals = np.unique(ss)
    return vals, np.array([ts[ss == v].mean() for v in vals])


def test_qm_recovers_known_monotone_map_tie_free():
    rng = np.random.default_rng(1)
    s = rng.permutation(np.unique(rng.gamma(2.0, 3.0, 5000)))       # distinct
    t = 2.0 * s ** 1.5 + 1.0                                          # strictly increasing map
    t = rng.permutation(t)  # pairing is irrelevant to QM; only the marginals are matched
    knots = br.qm_fit(*np.unique(s, return_counts=True), *np.unique(t, return_counts=True))
    np.testing.assert_allclose(br.qm_apply(knots, s), 2.0 * s ** 1.5 + 1.0, rtol=1e-10)


def test_qm_tie_block_mean_hand_case():
    s = np.array([0, 0, 0, 1, 2])
    t = np.array([4, 0, 2, 1, 3])
    kx, ky = br.qm_fit(*np.unique(s, return_counts=True), *np.unique(t, return_counts=True))
    np.testing.assert_array_equal(kx, [0, 1, 2])
    np.testing.assert_allclose(ky, [1.0, 3.0, 4.0])                  # mean(0,1,2), 3, 4


def test_qm_histogram_path_matches_brute_force_with_ties():
    rng = np.random.default_rng(2)
    s = rng.poisson(0.7, 20000)
    t = rng.negative_binomial(0.5, 0.2, 20000)
    knots = br.qm_fit(*np.unique(s, return_counts=True), *np.unique(t, return_counts=True))
    bx, by = _brute_tie_block(s, t)
    np.testing.assert_array_equal(knots[0], bx)
    np.testing.assert_allclose(knots[1], by, rtol=1e-12)
    # per-chromosome merge gives the same histogram as the whole
    parts = [np.unique(a, return_counts=True) for a in np.array_split(s, 7)]
    u, c = br.merge_hist(parts)
    u0, c0 = np.unique(s, return_counts=True)
    np.testing.assert_array_equal(u, u0)
    np.testing.assert_array_equal(c, c0)


def test_qm_on_identical_tracks_is_identity_on_seen_values():
    rng = np.random.default_rng(8)
    s = rng.negative_binomial(0.8, 0.3, 5000)
    knots = br.qm_fit(*np.unique(s, return_counts=True), *np.unique(s, return_counts=True))
    np.testing.assert_allclose(br.qm_apply(knots, s), s, rtol=0, atol=1e-9)


def test_qm_unseen_values_interpolate_and_clamp():
    knots = (np.array([0.0, 2.0, 4.0]), np.array([0.0, 10.0, 12.0]))
    np.testing.assert_allclose(br.qm_apply(knots, [1.0, 3.0, -1.0, 9.0]), [5.0, 11.0, 0.0, 12.0])


def test_qm_spearman_equals_identity_when_tie_free():
    rng = np.random.default_rng(3)
    s = rng.gamma(1.0, 2.0, 3000)
    t = np.exp(rng.normal(0, 1, 3000))
    y = s * 1.3 + rng.normal(0, 1, 3000)
    knots = br.qm_fit(*np.unique(s, return_counts=True), *np.unique(t, return_counts=True))
    assert spearman(br.qm_apply(knots, s), y) == pytest.approx(spearman(s, y), abs=1e-12)


# ---------------------------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------------------------


def _subsets(y):
    return br.subset_index(np.asarray(y))


def test_point_crps_is_mae_and_subsets():
    rng = np.random.default_rng(4)
    y = rng.poisson(1.0, 1000).astype(np.uint32)
    pred = rng.gamma(1.0, 1.0, 1000)
    sub = _subsets(y)
    assert sub["top1"].size == 10
    assert set(sub["nonzero"]) == set(np.flatnonzero(y > 0))
    assert y[sub["top1"]].min() >= np.sort(y)[-10]
    out = br.score_rung(pred, y, "counts", {}, sub)
    assert out["point_crps_all"] == pytest.approx(np.mean(np.abs(pred - y)))
    nz = y > 0
    assert out["point_crps_nonzero"] == pytest.approx(np.mean(np.abs(pred[nz] - y[nz])))
    out_p = br.score_rung(pred, y.astype(np.float32), "pval", {"sigma": 0.5}, sub)
    assert out_p["point_crps_all"] == pytest.approx(np.mean(np.abs(pred - y)))


def test_top1_ties_broken_by_position():
    y = np.zeros(300, np.uint32)
    y[[5, 50, 100, 200]] = 7                                           # 4 tied maxima, k = 3
    assert list(_subsets(y)["top1"]) == [5, 50, 100]


def _poisson_crps_direct(lam, y, kmax=None):
    """CRPS = sum_k (F(k) - 1{k >= y})^2 over the integers: the definition, summed directly."""
    kmax = kmax or int(max(y, lam) + 40 * math.sqrt(lam + 1) + 50)
    k = np.arange(kmax + 1)
    return float(np.sum((poisson.cdf(k, lam) - (k >= y)) ** 2))


def test_poisson_crps_matches_direct_sum():
    lams = [1e-3, 0.05, 0.7, 1.0, 3.3, 12.0, 85.0, 400.0]
    ys = [0, 1, 2, 5, 17, 90, 420]
    got = br.poisson_crps(np.repeat(lams, len(ys)), np.tile(ys, len(lams)))
    ref = np.array([_poisson_crps_direct(lam, y) for lam in lams for y in ys])
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-12)


def _lognormal_crps_numeric(median, sigma, y):
    """CRPS = int_0^inf (F(x) - 1{x >= y})^2 dx, integrated numerically (split at y)."""
    def F(x):
        return norm.cdf((math.log(x) - math.log(median)) / sigma) if x > 0 else 0.0
    kw = {"limit": 400, "epsabs": 1e-12, "epsrel": 1e-10}
    lo = integrate.quad(lambda x: F(x) ** 2, 0.0, y, **kw)[0] if y > 0 else 0.0
    hi = integrate.quad(lambda x: (1.0 - F(x)) ** 2, y, np.inf, **kw)[0]
    return lo + hi


def test_lognormal_crps_matches_numeric_integral():
    cases = [(1.0, 0.5, 0.0), (1.0, 0.5, 1.0), (0.04, 0.3, 0.05), (2.5, 1.2, 0.3),
             (2.5, 1.2, 40.0), (1e-3, 0.8, 0.0), (1e-3, 0.8, 0.02), (30.0, 0.05, 29.0)]
    got = br.lognormal_crps(*[np.array(c) for c in zip(*cases)])
    ref = np.array([_lognormal_crps_numeric(*c) for c in cases])
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-10)


def test_lognormal_crps_tiny_sigma_is_absolute_error():
    y = np.array([0.0, 0.5, 2.0, 7.0])
    np.testing.assert_allclose(br.lognormal_crps(np.full(4, 2.0), 0.0, y), np.abs(y - 2.0),
                               atol=1e-9)


def test_sigma_ml_recovers_known_sigma():
    rng = np.random.default_rng(6)
    pred = rng.gamma(0.8, 2.0, 400_000) + 0.01
    sigma = 0.37
    y = pred * np.exp(sigma * rng.standard_normal(pred.size))
    y[:50] = 0.0                                                    # zero targets leave the fit
    ss, n_pos, n_zero = br.log_residual_sums(pred, y)
    fit = br.sigma_from_sums(ss, n_pos, n_zero)
    assert fit["sigma"] == pytest.approx(sigma, rel=0.01)
    assert (fit["n_pos_target"], fit["n_zero_target"]) == (pred.size - 50, 50)
    # the floor applies to the prediction before the log
    ss2, _, _ = br.log_residual_sums(np.array([0.0]), np.array([1.0]))
    assert ss2 == pytest.approx(math.log(1.0 / br.PRED_FLOOR) ** 2)


def test_poisson_spread_crps_and_split():
    rng = np.random.default_rng(9)
    lam = rng.gamma(1.0, 4.0, 5000)
    y = rng.poisson(lam).astype(np.uint32)
    pred = lam / 4.0                                                # 4x too low: c* ~ +2
    pred[:10] = 0.0                                                 # exercises the floor
    out = br.score_rung(pred, y, "counts", {}, _subsets(y))
    ref = br.poisson_crps(np.maximum(pred, br.PRED_FLOOR), y.astype(float))
    assert out["spread_crps_all"] == pytest.approx(ref.mean(), rel=1e-12)
    assert out["c_star_all"] == pytest.approx(2.0, abs=0.1)
    assert out["scale_error_all"] == pytest.approx(out["spread_crps_all"]
                                                   - out["crps_oracle_scaled_all"])
    assert out["scale_error_all"] > 0
    assert out["frac_bins_at_floor"] == pytest.approx(10 / 5000)


def test_lognormal_spread_crps_and_split():
    rng = np.random.default_rng(5)
    med = rng.gamma(0.5, 2.0, 4000) + 0.02
    y = (med * np.exp(0.4 * rng.standard_normal(med.size))).astype(np.float32)
    pred = med * 0.5                                                # median half: c* ~ +1
    out = br.score_rung(pred, y, "pval", {"sigma": 0.4}, _subsets(y))
    ref = br.lognormal_crps(pred, 0.4, y.astype(np.float64))
    assert out["spread_crps_all"] == pytest.approx(ref.mean(), rel=1e-12)
    assert out["c_star_all"] == pytest.approx(1.0, abs=0.1)
    assert out["crps_oracle_scaled_all"] < out["spread_crps_all"]


# ---------------------------------------------------------------------------------------------
# loader, manifest, end to end
# ---------------------------------------------------------------------------------------------


def test_loader_reads_t112_layout(tmp_path):
    counts, pval = _synthetic()
    d = _write_product(tmp_path, "P", counts, pval)
    with br.Track(d, "counts") as tc, br.Track(d, "pval") as tp:
        assert tc.chroms == [c for c in bin25.MAIN_CHROMS if c in N_BINS]   # MAIN_CHROMS order
        for c in N_BINS:
            a = tc.get(c)
            assert a.dtype == np.uint32 and a.shape == (N_BINS[c],)
            np.testing.assert_array_equal(a, counts[c])
            assert tp.get(c).dtype == np.float32
        train, val, score = br.split_chroms(tc.chroms)
        assert train == ["chr1", "chr2", "chrX"] and val == ["chr22"] and score == ["chr19", "chr21"]
    # wrong dtype is refused, not cast
    bin25.write_npz(tmp_path / "Q" / "pval25.npz", {c: np.asarray(v, np.float64)
                                                   for c, v in pval.items()})
    with br.Track(tmp_path / "Q", "pval") as bad, pytest.raises(ValueError, match="dtype"):
        bad.get("chr1")


def test_load_eval_removes_blacklist_bins(tmp_path):
    counts, pval = _synthetic()
    d = _write_product(tmp_path, "P", counts, pval)
    bl = br.read_blacklist(_blacklist(tmp_path))
    with br.Track(d, "counts") as t:
        ev = br.load_eval(t, t, ["chr19", "chr21"], bl)
    assert ev["n_bins_blacklisted"] == 11
    assert ev["n_bins_total"] == 1500 + 1200
    keep19 = np.ones(1500, bool)
    keep19[10:20] = False
    np.testing.assert_array_equal(ev["y"], np.concatenate([counts["chr19"][keep19],
                                                           counts["chr21"][1:]]))


def _manifest(tmp_path, rows):
    path = tmp_path / "MANIFEST.tsv"
    cols = ["pid", "track", "cell", "assay", "arm", "level", "knob"]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(dict(zip(cols, r)))
    return path


def test_list_pairs(tmp_path):
    m = _manifest(tmp_path, [
        ("T1__depth__15M", "T1", "C19", "H3K27ac", "depth", "15M", "treatment_reads"),
        ("T1__base__base", "T1", "C19", "H3K27ac", "base", "base", "none"),
        ("T1__pe__pe", "T1", "C19", "H3K27ac", "pe", "pe", "chip.paired_end"),
        ("T0__base__base", "T0", "C12", "DNase-seq", "base", "base", "none"),
        ("T0__mapq__0", "T0", "C12", "DNase-seq", "mapq", "0", "atac.mapq_thresh"),
    ])
    pairs = br.list_pairs(br.read_manifest(m))
    assert len(pairs) == 6
    assert [p["index"] for p in pairs] == list(range(6))
    assert (pairs[0]["source_pid"], pairs[0]["target_pid"]) == ("T0__base__base", "T0__mapq__0")
    assert (pairs[1]["source_pid"], pairs[1]["target_pid"]) == ("T0__mapq__0", "T0__base__base")
    assert pairs[0]["mark_class"] == "DNase" and pairs[2]["mark_class"] == "narrow"
    m2 = _manifest(tmp_path, [("X__pe__pe", "X", "C1", "H3K9me3", "pe", "pe", "k")])
    with pytest.raises(ValueError, match="base"):
        br.list_pairs(br.read_manifest(m2))


def test_identity_pair_scores_zero_crps(tmp_path):
    counts, pval = _synthetic()
    products = tmp_path / "products"
    _write_product(products, "T1__base__base", counts, pval)
    _write_product(products, "T1__ratio__k2", counts, pval)           # X' == X
    m = _manifest(tmp_path, [
        ("T1__base__base", "T1", "C19", "H3K27ac", "base", "base", "none"),
        ("T1__ratio__k2", "T1", "C19", "H3K27ac", "ratio", "k2", "macs2_ratio")])
    pair = br.list_pairs(br.read_manifest(m))[0]
    res = json.loads(br.run_pair(pair, products, _blacklist(tmp_path), tmp_path / "out")
                     .read_text())
    assert len(res["records"]) == 2 * 2 * 2                           # space x rung x eval
    for rec in res["records"]:
        if rec["rung"] != "noSolution":
            continue      # QM interpolates values unseen in training, so it is not exactly 0 here
        for s in br.SUBSETS:
            assert rec[f"point_crps_{s}"] == 0.0, (rec["space"], rec["rung"], s)
    for space in br.SPACES:
        assert res["spaces"][space]["source_equals_target_train"] is True
        assert res["spaces"][space]["evals"]["score"]["n_bins_blacklisted"] == 11
    # X' == X, so the log residual is non-zero only where the 1e-3 floor lifts the prediction
    y = np.concatenate([pval[c] for c in ("chr1", "chr2", "chrX")]).astype(np.float64)
    y = y[y > 0]
    r = np.log(y) - np.log(np.maximum(y, br.PRED_FLOOR))
    assert res["spaces"]["pval"]["spread"]["noSolution"]["sigma"] == pytest.approx(
        math.sqrt(np.mean(r ** 2)), rel=1e-9)
    assert res["spaces"]["counts"]["spread"]["noSolution"] == {}           # Poisson: no fit


def _write_pseudoreps(root: Path, pid: str, counts: dict, pval: dict, rng) -> None:
    """pr1/pr2: binomial halves of the counts; p halved with multiplicative noise."""
    for half in ("pr1", "pr2"):
        c = {k: rng.binomial(v, 0.5) for k, v in counts.items()}
        q = {k: (v * 0.6 * np.exp(0.3 * rng.standard_normal(v.size))) for k, v in pval.items()}
        _write_product(root / pid, half, c, q)


def test_end_to_end_run_oracle_and_aggregate(tmp_path):
    rng = np.random.default_rng(7)
    counts, pval = _synthetic(0)
    # the arm: roughly half depth, and a monotone bend of p plus noise
    arm_counts = {c: rng.binomial(v, 0.5) for c, v in counts.items()}
    arm_pval = {c: (np.sqrt(v) * 1.5 + np.abs(rng.normal(0, 0.1, v.size)) * (v > 0))
                .astype(np.float32) for c, v in pval.items()}
    products, preps = tmp_path / "products", tmp_path / "pseudoreps"
    _write_product(products, "T1__base__base", counts, pval)
    _write_product(products, "T1__depth__15M", arm_counts, arm_pval)
    _write_pseudoreps(preps, "T1__base__base", counts, pval, rng)
    _write_pseudoreps(preps, "T1__depth__15M", arm_counts, arm_pval, rng)
    m = _manifest(tmp_path, [
        ("T1__base__base", "T1", "C19", "H3K27ac", "base", "base", "none"),
        ("T1__depth__15M", "T1", "C19", "H3K27ac", "depth", "15M", "treatment_reads")])
    out = tmp_path / "out"
    bl = _blacklist(tmp_path)
    for i in range(2):
        assert br.main(["run", "--manifest", str(m), "--products", str(products),
                        "--blacklist", str(bl), "--out", str(out), "--index", str(i)]) == 0
    # without the oracle, aggregate reports it missing and exits 1
    assert br.main(["aggregate", "--manifest", str(m), "--out", str(out)]) == 1
    prods = br.list_products(br.read_manifest(m))
    assert [p["pid"] for p in prods] == ["T1__base__base", "T1__depth__15M"]
    for i in range(2):
        assert br.main(["oracle", "--manifest", str(m), "--pseudoreps", str(preps),
                        "--blacklist", str(bl), "--out", str(out), "--index", str(i)]) == 0
    res = json.loads((out / "T1__depth__15M__base_to_arm.json").read_text())
    by = {(r["space"], r["rung"], r["eval"]): r for r in res["records"]}
    # QM fixes the level: it must beat identity on counts at half depth
    assert (by[("counts", "QuantileMatching", "score")]["point_crps_all"]
            < by[("counts", "noSolution", "score")]["point_crps_all"])
    assert res["spaces"]["pval"]["spread"]["QuantileMatching"]["sigma"] > 0
    orc = json.loads((out / "oracle" / "T1__depth__15M.json").read_text())
    assert orc["depth"].startswith("half")
    assert len(orc["records"]) == 2 * 2 * 2                           # space x direction x eval
    # the oracle's sigma is fitted per direction on the training chromosomes: ~0.3 * sqrt(2)
    for d in br.ORACLE_DIRECTIONS:
        assert orc["spaces"]["pval"][d]["spread"]["sigma"] == pytest.approx(0.3 * math.sqrt(2),
                                                                          rel=0.1)
    assert br.main(["aggregate", "--manifest", str(m), "--out", str(out)]) == 0
    with (out / "rungs_v2.tsv").open() as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    assert len(rows) == 2 * (8 + 2 * 2 * 3)             # pairs x (own rows + space x eval x 3)

    def rec_of(pid, space, direction):
        return next(r for r in json.loads((out / "oracle" / f"{pid}.json").read_text())["records"]
                    if r["space"] == space and r["eval"] == "score" and r["direction"] == direction)

    # the join: base_to_arm takes the ARM's oracle, arm_to_base the BASE's; `oracle` = the mean
    for pair_dir, target in (("base_to_arm", "T1__depth__15M"), ("arm_to_base", "T1__base__base")):
        for space in br.SPACES:
            sel = {r["rung"]: r for r in rows if r["direction"] == pair_dir
                   and r["space"] == space and r["eval"] == "score"}
            a, b = rec_of(target, space, "pr1_to_pr2"), rec_of(target, space, "pr2_to_pr1")
            assert float(sel["oracle_pr1_to_pr2"]["spread_crps_all"]) == pytest.approx(
                a["spread_crps_all"])
            assert float(sel["oracle_pr2_to_pr1"]["spread_crps_all"]) == pytest.approx(
                b["spread_crps_all"])
            for k in ("spread_crps_all", "spread_crps_top1", "spearman_all"):
                assert float(sel["oracle"][k]) == pytest.approx((a[k] + b[k]) / 2)
            assert sel["oracle"]["target_pid"] == target
    md = (out / "rungs_v2.md").read_text()
    assert "By mark class — counts" in md and "| narrow | QuantileMatching |" in md
    assert "| narrow | oracle |" in md and "fraction closed" in md
    p_table = md.split("## By mark class — pval")[1].split("##")[0]
    assert "spread_crps_nonzero" not in p_table
    assert "spread_crps_nonzero" in md.split("## By mark class — counts")[1].split("##")[0]


def test_gap_fraction():
    assert br.gap_fraction(1.0, 0.6, 0.2) == pytest.approx(0.5)
    assert br.gap_fraction(0.2, 0.1, 0.3) is None                     # noSolution below oracle
    assert br.gap_fraction(0.3, 0.1, 0.3) is None
    assert br.gap_fraction(0.5, 0.5, 0.9, higher_is_better=True) == pytest.approx(0.0)
    assert br.gap_fraction(0.9, 0.9, 0.5, higher_is_better=True) is None
    assert br.gap_fraction(None, 0.1, 0.0) is None
