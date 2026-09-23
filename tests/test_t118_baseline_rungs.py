"""t118 — `tools/t118/baseline_rungs.py`: the noSolution and QuantileMatching rungs.

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

from candi.bench.distributional import gauss_crps, p_from_mu
from candi.metrics import nb_crps, spearman

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
    out = br.score_rung(pred, y, "counts", {"n": 2.0}, sub)
    assert out["point_crps_all"] == pytest.approx(np.mean(np.abs(pred - y)))
    nz = y > 0
    assert out["point_crps_nonzero"] == pytest.approx(np.mean(np.abs(pred[nz] - y[nz])))
    out_p = br.score_rung(pred, y.astype(np.float32), "pval", {"sigma": 0.5}, sub)
    assert out_p["point_crps_all"] == pytest.approx(np.mean(np.abs(pred - y)))


def test_top1_ties_broken_by_position():
    y = np.zeros(300, np.uint32)
    y[[5, 50, 100, 200]] = 7                                           # 4 tied maxima, k = 3
    assert list(_subsets(y)["top1"]) == [5, 50, 100]


def test_nb_spread_crps_matches_repo_nb_crps_hand_case():
    pred = np.array([0.0, 0.5, 3.0, 10.0, 2.0])                       # 0 exercises the floor
    y = np.array([0, 1, 2, 15, 0], np.uint32)
    n = 1.7
    out = br.score_rung(pred, y, "counts", {"n": n}, _subsets(y))
    mu = np.maximum(pred, br.NB_MEAN_FLOOR)
    nn = np.full(5, n)
    ref = nb_crps(nn, p_from_mu(nn, mu), y.astype(float))
    assert out["spread_crps_all"] == pytest.approx(ref.mean(), rel=1e-12)
    nz = y > 0
    assert out["spread_crps_nonzero"] == pytest.approx(ref[nz].mean(), rel=1e-12)
    assert out["crps_all"] == out["spread_crps_all"]
    assert out["scale_error_all"] == pytest.approx(out["crps_all"] - out["crps_oracle_scaled_all"])
    assert out["frac_bins_at_nb_floor"] == pytest.approx(0.2)


def test_gauss_spread_crps_matches_repo_gauss_crps():
    rng = np.random.default_rng(5)
    y = rng.gamma(0.5, 2.0, 400).astype(np.float32)
    pred = y + rng.normal(0, 0.3, 400)
    out = br.score_rung(pred, y, "pval", {"sigma": 0.3}, _subsets(y))
    ref = gauss_crps(pred, np.full(400, 0.3), y.astype(np.float64))
    assert out["spread_crps_all"] == pytest.approx(ref.mean(), rel=1e-12)
    # closed form at z = 0: sigma * (2 phi(0) - 1/sqrt(pi))
    one = br.score_rung(np.array([1.0]), np.array([1.0], np.float32), "pval", {"sigma": 2.0},
                        {"all": np.array([0])})
    assert one["spread_crps_all"] == pytest.approx(2.0 * (2 / math.sqrt(2 * math.pi)
                                                          - 1 / math.sqrt(math.pi)))


def test_nb_dispersion_mle_recovers_known_n():
    rng = np.random.default_rng(6)
    mu = rng.gamma(1.0, 5.0, 200_000) + 0.1
    n_true = 2.0
    y = rng.negative_binomial(n_true, n_true / (n_true + mu))
    fit = br.nb_fit_dispersion(mu, y, np.ones_like(mu))
    assert fit["n"] == pytest.approx(n_true, rel=0.05)
    assert not fit["at_bound"]


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
    assert res["spaces"]["pval"]["spread"]["noSolution"]["sigma"] == 0.0


def test_end_to_end_run_and_aggregate(tmp_path):
    rng = np.random.default_rng(7)
    counts, pval = _synthetic(0)
    # the arm: roughly half depth, and a monotone bend of p plus noise
    arm_counts = {c: rng.binomial(v, 0.5) for c, v in counts.items()}
    arm_pval = {c: (np.sqrt(v) * 1.5 + np.abs(rng.normal(0, 0.1, v.size)) * (v > 0))
                .astype(np.float32) for c, v in pval.items()}
    products = tmp_path / "products"
    _write_product(products, "T1__base__base", counts, pval)
    _write_product(products, "T1__depth__15M", arm_counts, arm_pval)
    m = _manifest(tmp_path, [
        ("T1__base__base", "T1", "C19", "H3K27ac", "base", "base", "none"),
        ("T1__depth__15M", "T1", "C19", "H3K27ac", "depth", "15M", "treatment_reads")])
    out = tmp_path / "out"
    bl = _blacklist(tmp_path)
    for i in range(2):
        assert br.main(["run", "--manifest", str(m), "--products", str(products),
                        "--blacklist", str(bl), "--out", str(out), "--index", str(i)]) == 0
    res = json.loads((out / "T1__depth__15M__base_to_arm.json").read_text())
    by = {(r["space"], r["rung"], r["eval"]): r for r in res["records"]}
    # QM fixes the level: it must beat identity on counts at half depth
    assert (by[("counts", "QuantileMatching", "score")]["point_crps_all"]
            < by[("counts", "noSolution", "score")]["point_crps_all"])
    fit_n = res["spaces"]["counts"]["spread"]["QuantileMatching"]["n"]
    assert br.DISPERSION_BOUNDS[0] < fit_n <= br.DISPERSION_BOUNDS[1]
    assert br.main(["aggregate", "--manifest", str(m), "--out", str(out)]) == 0
    with (out / "baseline_rungs.tsv").open() as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    assert len(rows) == 2 * 8
    md = (out / "baseline_rungs.md").read_text()
    assert "By mark class" in md and "| narrow | counts | QuantileMatching |" in md
