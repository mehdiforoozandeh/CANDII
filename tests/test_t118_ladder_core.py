"""t118 ladder core — `tools/t118/ladder/{pairs,data,base,synth}.py`: the contracts wave 1 builds on."""
from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "t118"))

import ladder  # noqa: E402
from ladder import base, data, pairs, synth  # noqa: E402

#: the t112 manifest: the tracked fixture (t118-C0) if present, else `T118_MANIFEST`
_FIXTURE = ROOT / "tests" / "fixtures" / "t112_meta" / "MANIFEST.tsv"
MANIFEST_MD5 = "599e2ca607961fe550b477558f894edf"


def _real_manifest() -> Path:
    for p in (os.environ.get("T118_MANIFEST"), _FIXTURE):
        if p and Path(p).is_file():
            return Path(p)
    pytest.skip("t112 MANIFEST.tsv absent (tests/fixtures/t112_meta/ or $T118_MANIFEST)")


@pytest.fixture(scope="module")
def real_rows():
    path = _real_manifest()
    assert hashlib.md5(path.read_bytes()).hexdigest() == MANIFEST_MD5
    return pairs.read_manifest(path)


@pytest.fixture(scope="module")
def synth_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth")
    manifest, cov = synth.make_products(root, seed=0)
    return root, manifest, cov


# ---------------------------------------------------------------------------------------------
# constants and the package
# ---------------------------------------------------------------------------------------------


def test_constants_match_their_sources():
    spec = importlib.util.spec_from_file_location("t112_bin25_check", ROOT / "tools/t112/bin25.py")
    bin25 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bin25)
    assert pairs.MAIN_CHROMS == bin25.MAIN_CHROMS
    import baseline_rungs
    assert pairs.MARK_CLASS == baseline_rungs.MARK_CLASS
    assert pairs.TRACKS == tuple(sorted(pairs.TRACKS)) and len(pairs.TRACKS) == 7
    assert pairs.G_IDS[-1] == "all" and len(pairs.G_IDS) == 8
    assert ladder.__doc__ and importlib.import_module("ladder.fforms").__doc__


# ---------------------------------------------------------------------------------------------
# pairs on the real manifest
# ---------------------------------------------------------------------------------------------


EXPECTED_COUNT = ("usable_products 128\ntrain_pairs C07M20 38\ntrain_pairs C07M29 38\n"
                  "train_pairs C12M02 14\ntrain_pairs C19M16 38\ntrain_pairs C19M22 38\n"
                  "train_pairs C40M17 38\ntrain_pairs C40M18 38\ntrain_pairs all 242\n"
                  "law_pairs all 2094\ntasks 576\n")


def test_real_counts(real_rows):
    rows = real_rows
    assert len(pairs.usable_products(rows)) == 128
    assert [r["pid"] for r in pairs.usable_products(rows)] == \
        sorted(r["pid"] for r in rows if r["pid"] not in pairs.EXCLUDED_PIDS)
    assert pairs.track_ids(rows) == list(pairs.TRACKS)
    for t in pairs.TRACKS:
        n_tr = 14 if t == "C12M02" else 38
        assert len(pairs.train_pairs(rows, t)) == n_tr
        assert len(pairs.law_pairs(rows, t)) == (42 if t == "C12M02" else 342)
        assert len(pairs.fit_pids(rows, t)) == (8 if t == "C12M02" else 20)
    assert len(pairs.train_pairs(rows, "all")) == 242
    assert len(pairs.law_pairs(rows, "all")) == 2094
    assert len(pairs.fit_pids(rows, "all")) == 128
    assert "\n".join(pairs.count_lines(rows)) + "\n" == EXPECTED_COUNT


def test_real_count_cli(real_rows):
    out = subprocess.run([sys.executable, str(ROOT / "tools/t118/ladder/pairs.py"), "count",
                          str(_real_manifest())], capture_output=True, text=True, check=True,
                         env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    assert out.stdout == EXPECTED_COUNT


def test_real_pair_fields(real_rows):
    tr = pairs.train_pairs(real_rows, "C19M16")
    assert all(set(p) == set(pairs.PAIR_KEYS) for p in tr)
    # (pid, direction) order, base_to_arm first
    assert [p["direction"] for p in tr[:2]] == ["base_to_arm", "arm_to_base"]
    arm_pids = [p["target_pid"] if p["direction"] == "base_to_arm" else p["source_pid"] for p in tr]
    assert arm_pids == sorted(arm_pids)
    p0 = tr[0]
    assert p0["source_pid"] == "C19M16__base__base" and p0["target_pid"] == "C19M16__abproxy__f0.5"
    assert p0["knob_combo"] == "base:base->abproxy:f0.5" and p0["mark_class"] == "narrow"
    assert tr[1]["knob_combo"] == "abproxy:f0.5->base:base"
    # the p-only arms (ratio, ctlid, ctldepth, extsize): 8 products, 16 pairs
    assert sum(p["counts_identical"] for p in tr) == 16
    assert not any(p["pval_identical"] for p in tr)
    dn = pairs.train_pairs(real_rows, "C12M02")
    assert not any(pairs.EXCLUDED_PIDS[0] in (p["source_pid"], p["target_pid"]) for p in dn)
    assert all(p["mark_class"] == "DNase" for p in dn)
    law = pairs.law_pairs(real_rows, "C19M16")
    assert all(p["direction"] == "arm_to_arm" and p["source_pid"] != p["target_pid"] for p in law)
    assert not any("__base__" in p["source_pid"] + p["target_pid"] for p in law)
    allp = pairs.train_pairs(real_rows, "all")
    assert allp == [p for t in pairs.TRACKS for p in pairs.train_pairs(real_rows, t)]


def test_real_shuffle_never_identical(real_rows):
    rows = real_rows
    md5 = {r["pid"]: r for r in rows}
    for space in pairs.SPACES:
        col = pairs.MD5_COLUMN[space]
        for seed in pairs.SEEDS:
            for p in pairs.train_pairs(rows, "all"):
                s = pairs.shuffle_target(rows, p, space, seed)
                assert s not in (p["target_pid"], p["source_pid"])
                assert md5[s][col] != md5[p["target_pid"]][col]
                assert md5[s]["track"] == p["track"] and s not in pairs.EXCLUDED_PIDS
    p = pairs.train_pairs(rows, "C19M16")[5]
    assert pairs.shuffle_target(rows, p, "counts", 0) == pairs.shuffle_target(rows, p, "counts", 0)


# ---------------------------------------------------------------------------------------------
# tasks, derangement
# ---------------------------------------------------------------------------------------------


def test_task_index_round_trip():
    t = pairs.tasks([])
    assert len(t) == 576 and [r["index"] for r in t] == list(range(576))
    for r in t:
        i = (pairs.RUNGS.index(r["rung"]) * 144 + pairs.G_IDS.index(r["g"]) * 18
             + pairs.SPACES.index(r["space"]) * 9 + pairs.MODELS.index(r["model"]) * 3 + r["seed"])
        assert i == r["index"]
        assert r["run_name"] == f"{r['rung']}_{r['g']}_{r['space']}_{r['model']}_s{r['seed']}"
        assert set(r) == set(pairs.TASK_KEYS)
    assert t[0]["run_name"] == "A_C07M20_counts_real_s0"
    assert t[54]["run_name"] == "A_C19M16_counts_real_s0"
    assert t[-1]["run_name"] == "D_all_pval_ids_s2"
    assert [t[i]["run_name"] for i in (54, 57, 60, 63, 66, 69)] == [
        f"A_C19M16_{s}_{m}_s0" for s in pairs.SPACES for m in pairs.MODELS]
    assert len({r["run_name"] for r in t}) == 576


@pytest.mark.parametrize("n", [2, 3, 14, 38, 242])
def test_ids_permutation_is_a_derangement(n):
    for seed in range(5):
        perm = pairs.ids_permutation(n, seed)
        assert sorted(perm.tolist()) == list(range(n))
        assert not np.any(perm == np.arange(n))
        np.testing.assert_array_equal(perm, pairs.ids_permutation(n, seed))
    assert not np.array_equal(pairs.ids_permutation(242, 0), pairs.ids_permutation(242, 1))


def test_ids_permutation_single_pair_refused():
    with pytest.raises(ValueError):
        pairs.ids_permutation(1, 0)


# ---------------------------------------------------------------------------------------------
# knot_stats
# ---------------------------------------------------------------------------------------------


def _check_knots(k):
    assert k.shape == (12,) and k.dtype == np.float64
    assert np.all(np.diff(k) > 0)


def test_knot_stats_continuous():
    x = np.random.default_rng(0).normal(size=100_000)
    k = base.knot_stats(x)
    _check_knots(k)
    np.testing.assert_allclose(k, np.quantile(x, base.KNOT_PROBS))


def test_knot_stats_heavily_tied_zero_inflated():
    rng = np.random.default_rng(1)
    for p_zero in (0.9, 0.995, 0.9999, 1.0):
        n = 1_000_000
        c = np.where(rng.random(n) < p_zero, 0, rng.negative_binomial(1, 0.3, n))
        for x in (np.log1p(c), np.log(np.maximum(c * 0.1, 1e-3))):
            k = base.knot_stats(x)
            _check_knots(k)
            assert k[0] == x.min()
    _check_knots(base.knot_stats(np.array([0.0, 0.0, 1.0, 1.0, 2.0])))
    _check_knots(base.knot_stats(np.zeros(10)))


# ---------------------------------------------------------------------------------------------
# FForm and load_form
# ---------------------------------------------------------------------------------------------


def test_fform_interface_and_load_form(monkeypatch):
    stats = {"knots_x": np.linspace(0, 5, 12), "n0": 5.0, "sigma0": 0.5}
    # load_form imports ladder.fforms.form_<rung> by name; inject a stub module for the test
    import types
    mod = types.ModuleType("ladder.fforms.form_zz")

    class Stub(base.FForm):
        rung, n_theta, context = "ZZ", 2, 1

        def init_theta(self):
            d = np.log(self.stats["n0"] if self.space == "counts" else self.stats["sigma0"])
            return torch.tensor([0.0, float(d)])

        def forward(self, x, theta):
            xc = x[:, self.context:x.shape[1] - self.context]
            return xc + theta[:, :1], theta[:, 1:2].expand_as(xc)

        def describe(self, theta):
            return {"a": float(theta[0]), "disp": float(theta[1])}

    mod.build = lambda space, stats: Stub(space, stats)
    monkeypatch.setitem(sys.modules, "ladder.fforms.form_zz", mod)
    f = base.load_form("ZZ", "counts", stats)
    assert isinstance(f, base.FForm) and f.space == "counts" and f.stats["n0"] == 5.0
    x = torch.randn(3, 10 + 2)
    th = f.init_theta().expand(3, -1)
    loc, disp = f(x, th)
    assert loc.shape == (3, 10) and disp.shape == (3, 10)
    torch.testing.assert_close(loc, x[:, 1:-1])
    torch.testing.assert_close(disp, torch.full((3, 10), float(np.log(5.0))))
    assert f.describe(f.init_theta()) == {"a": 0.0, "disp": pytest.approx(np.log(5.0))}
    with pytest.raises(ValueError):
        Stub("bogus", stats)
    with pytest.raises(NotImplementedError):
        base.FForm("pval", stats).init_theta()


# ---------------------------------------------------------------------------------------------
# synth + Corpus + cache
# ---------------------------------------------------------------------------------------------


def test_synth_products_readable(synth_dir):
    root, manifest, cov = synth_dir
    rows = pairs.read_manifest(manifest)
    assert [r["pid"] for r in rows] == sorted(r["pid"] for r in rows)
    assert len(rows) == 10
    assert {r["pid"] for r in rows} >= {"T1__base__base", "T1__depth__15M", "T1__depth__7.5M",
                                        "T1__extsize__k2", "T1__pe__pe", "T2__base__base"}
    assert list(pairs.read_manifest(cov)[0]) == list(synth.COV_COLUMNS)
    assert list(rows[0]) == list(synth.MANIFEST_COLUMNS)
    corpus = data.Corpus(root)
    assert not corpus.is_cache
    for r in rows:
        for space, col in pairs.MD5_COLUMN.items():
            npz = root / r["pid"] / data.SPACE_FILES[space][0]
            assert data.md5_file(npz) == r[col]
            assert corpus.chroms(r["pid"], space) == list(synth.DEFAULT_N_BINS)
            for c, n in synth.DEFAULT_N_BINS.items():
                a = corpus.get(r["pid"], space, c)
                assert a.shape == (n,) and a.dtype == data.SPACE_FILES[space][1]
                assert corpus.n_bins(r["pid"], space, c) == n
                if space == "pval":
                    assert np.isfinite(a).all() and a.min() >= 0
    # the planted relations
    b = corpus.get("T1__base__base", "counts", "chr1").astype(float)
    d = corpus.get("T1__depth__15M", "counts", "chr1").astype(float)
    assert abs(d.sum() / b.sum() - 0.5) < 0.02
    np.testing.assert_array_equal(corpus.get("T1__extsize__k2", "counts", "chr1"), b)
    np.testing.assert_allclose(corpus.get("T1__depth__7.5M", "pval", "chr2"),
                               0.25 * corpus.get("T1__base__base", "pval", "chr2"), rtol=1e-6)
    tp = pairs.train_pairs(rows, "T1")
    assert len(tp) == 8 and sum(p["counts_identical"] for p in tp) == 2
    assert len(pairs.train_pairs(rows, "all")) == 16 and len(pairs.law_pairs(rows, "all")) == 24
    covs = {r["pid"]: r for r in pairs.read_manifest(cov)}
    assert covs["T2__base__base"]["has_control"] == "0"
    assert covs["T2__base__base"]["ctl_identity"] == "none"
    assert covs["T1__pe__pe"]["run_type_pe"] == "1"
    assert covs["T1__extsize__k2"]["extsize_k"] == "2.0"
    assert float(covs["T1__depth__7.5M"]["depth_reads"]) == 7_500_000


def test_synth_shuffle_never_identical(synth_dir):
    root, manifest, _ = synth_dir
    rows = pairs.read_manifest(manifest)
    by = {r["pid"]: r for r in rows}
    for space in pairs.SPACES:
        col = pairs.MD5_COLUMN[space]
        for p in pairs.train_pairs(rows, "all"):
            for seed in range(4):
                s = pairs.shuffle_target(rows, p, space, seed)
                assert by[s][col] != by[p["target_pid"]][col]
                assert s not in (p["source_pid"], p["target_pid"])


def test_synth_without_depth_arms(tmp_path):
    manifest, _ = synth.make_products(tmp_path, seed=1, tracks=("T1",), depth_ratio_arm=False,
                                      n_bins={"chr1": 300, "chr19": 200, "chr21": 200,
                                              "chr22": 100})
    assert [r["pid"] for r in pairs.read_manifest(manifest)] == [
        "T1__base__base", "T1__extsize__k2", "T1__pe__pe"]


def test_cache_round_trip(synth_dir, tmp_path):
    root, manifest, _ = synth_dir
    rows = pairs.read_manifest(manifest)
    cache = tmp_path / "cache"
    for r in rows:
        assert data.build_cache(root, cache, r["pid"], r) == {"counts": "built", "pval": "built"}
    assert data.build_cache(root, cache, rows[0]["pid"], rows[0]) == \
        {"counts": "skipped", "pval": "skipped"}
    assert not list(cache.glob("*.tmp"))
    cc, cn = data.Corpus(cache), data.Corpus(root)
    assert cc.is_cache and not cn.is_cache
    import json
    meta = json.loads((cache / "T1__base__base__counts.json").read_text())
    assert set(meta) == {"pid", "space", "dtype", "chroms", "offsets", "n_bins", "source_npz_md5"}
    assert meta["chroms"] == [c for c in pairs.MAIN_CHROMS if c in synth.DEFAULT_N_BINS]
    assert meta["n_bins"] == sum(synth.DEFAULT_N_BINS.values())
    assert meta["dtype"] == "uint32"
    for r in rows:
        for space in pairs.SPACES:
            assert cc.chroms(r["pid"], space) == cn.chroms(r["pid"], space)
            for c in cn.chroms(r["pid"], space):
                a = cc.get(r["pid"], space, c)
                assert isinstance(a, np.memmap) and not a.flags.writeable
                np.testing.assert_array_equal(a, cn.get(r["pid"], space, c))
                assert a.dtype == cn.get(r["pid"], space, c).dtype
                assert cc.n_bins(r["pid"], space, c) == cn.n_bins(r["pid"], space, c)


def test_cache_refuses_md5_mismatch(synth_dir, tmp_path):
    root, manifest, _ = synth_dir
    row = dict(pairs.read_manifest(manifest)[0])
    row["pval25_md5"] = "0" * 32
    with pytest.raises(ValueError, match="md5"):
        data.build_cache(root, tmp_path / "c", row["pid"], row)
    assert not (tmp_path / "c" / f"{row['pid']}__pval.json").exists()


def test_window_zero_padding(synth_dir, tmp_path):
    root, manifest, _ = synth_dir
    rows = pairs.read_manifest(manifest)
    r = rows[0]
    data.build_cache(root, tmp_path / "c", r["pid"], r)
    for corpus in (data.Corpus(root), data.Corpus(tmp_path / "c")):
        for space in pairs.SPACES:
            a = np.asarray(corpus.get(r["pid"], space, "chr22"))
            n = a.shape[0]
            w = corpus.window(r["pid"], space, "chr22", -5, 10)
            assert w.shape == (15,) and w.dtype == a.dtype
            assert np.all(w[:5] == 0)
            np.testing.assert_array_equal(w[5:], a[:10])
            w = corpus.window(r["pid"], space, "chr22", n - 3, n + 4)
            np.testing.assert_array_equal(w[:3], a[-3:])
            assert np.all(w[3:] == 0)
            np.testing.assert_array_equal(corpus.window(r["pid"], space, "chr22", 7, 20), a[7:20])
            assert np.all(corpus.window(r["pid"], space, "chr22", n + 10, n + 20) == 0)
            assert np.all(corpus.window(r["pid"], space, "chr22", -30, -10) == 0)
            w = corpus.window(r["pid"], space, "chr22", -2, n + 2)
            np.testing.assert_array_equal(w[2:-2], a)


def test_npz_lru_is_bounded(synth_dir):
    root, manifest, _ = synth_dir
    corpus = data.Corpus(root)
    for r in pairs.read_manifest(manifest):
        for space in pairs.SPACES:
            for c in synth.DEFAULT_N_BINS:
                corpus.get(r["pid"], space, c)
    assert len(corpus._lru) == data.LRU_SIZE


def test_read_blacklist_flags(tmp_path):
    bed = tmp_path / "bl.bed"
    bed.write_text("chr19\t250\t500\nchr21\t0\t1\nchr19\t499\t510\n")
    f = data.read_blacklist_flags(bed, "chr19", 40)
    assert f.dtype == bool and f.shape == (40,)
    assert np.flatnonzero(f).tolist() == list(range(10, 21))
    assert np.flatnonzero(data.read_blacklist_flags(bed, "chr21", 5)).tolist() == [0]
    assert not data.read_blacklist_flags(bed, "chr1", 5).any()
