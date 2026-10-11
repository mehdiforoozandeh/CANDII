"""t125-P2a: h's data side — caller constants, per-product numbers, features, control cache, task tables.

Written from `plan/ETA_OFFSET_AND_H_DESIGN.md`, `plan/T125_STATUS.md` (Phase 0 findings, PI decisions)
and the pinned interfaces of chunk t125-P2a. Expected values are computed by hand here, never by the
code under test.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "tools" / "t118", ROOT / "tools" / "t125"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ladder import data as ldata  # noqa: E402
from ladder import encoding, synth  # noqa: E402

from hmap import caller, controls, features, formula, products, tasks  # noqa: E402

MANIFEST = ROOT / "tests" / "fixtures" / "t112_meta" / "MANIFEST.tsv"
COVARIATES = ROOT / "tools" / "t118" / "covariates.tsv"
CTL_RL = ROOT / "tools" / "t125" / "hmap" / "control_read_length.tsv"
ASSAYS = ("DNase-seq", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me3", "H3K9me3")


# --------------------------------------------------------------------------------------------
# caller constants
# --------------------------------------------------------------------------------------------

def test_caller_constants():
    assert caller.GENOME_SIZE == 2.7e9
    assert caller.BIN == 25
    assert caller.SPAN_EXTRA == 24
    assert caller.WINDOWS_CTL_BP == {"frag": None, "1k": 1000, "10k": 10000}
    assert caller.WINDOW_NOCTL_BP == 10000
    assert caller.PILEUP_RULE == "count * fraglen / (read_length + 24)"
    assert caller.TAIL == "strict"
    assert caller.PSEUDOCOUNT == 0


def test_scale_control_larger_scales_control():
    # C19M16 base: 30 M treatment lines, 107 039 349 control lines -> control scaled down
    s_t, s_c = caller.scale(30e6, 107039349)
    assert s_t == 1.0
    assert s_c == pytest.approx(30e6 / 107039349, rel=1e-12)


def test_scale_treatment_larger_scales_treatment():
    # C19M16 ctldepth q0.25: 26 759 837 control lines < 30 M -> MACS2 scales the treatment down
    r = 30e6 / 26759837
    assert r > 1
    s_t, s_c = caller.scale(30e6, 26759837)
    assert s_c == 1.0
    assert s_t == pytest.approx(1 / r, rel=1e-12)


def test_scale_equal_and_ratio_override():
    assert caller.scale(5e6, 5e6) == (1.0, 1.0)
    v = 2 * 30e6 / 107039349
    s_t, s_c = caller.scale(30e6, 107039349, ratio_V=v)
    assert (s_t, s_c) == (1.0, pytest.approx(v, rel=1e-12))
    # a ratio above 1 scales the treatment, whatever the read counts say
    s_t, s_c = caller.scale(30e6, 107039349, ratio_V=2.0)
    assert (s_t, s_c) == (pytest.approx(0.5), 1.0)


def test_lambda_bg_dnase_base():
    assert caller.lambda_bg(50e6, 150) == pytest.approx(150 * 50e6 / 2.7e9, rel=1e-12)


@pytest.mark.parametrize("pid", ["C19M16__ratio__k2", "C19M16__ctldepth__q0.25",
                                 "C19M16__base__base", "C07M20__ctlid__none", "C12M02__base__base"])
def test_lambda_bg_agrees_with_formula(pid):
    info = products.load(MANIFEST, COVARIATES)[pid]
    s_t, _ = features.scales(info)
    ours = caller.lambda_bg(info.depth_reads * s_t, info.fraglen)
    assert math.log(ours) == pytest.approx(features.log_lambda_bg(info), rel=1e-12)
    assert ours == pytest.approx(formula.lambda_bg(info.depth_reads * formula.scale(info)[0],
                                                   info.fraglen), rel=1e-12)
    # formula's local rate with all-zero windows is its lambda_bg alone
    z = np.zeros(3)
    if info.has_control:
        lam = formula.local_lambda(info, {"frag": z, "1k": z, "10k": z}, None)
    else:
        lam = formula.local_lambda(info, None, z)
    np.testing.assert_allclose(lam, ours, rtol=1e-12)
    if pid == "C19M16__ratio__k2":
        assert ours == pytest.approx(180 * 30e6 / 2.7e9, rel=1e-12)


def test_lambda_window_hand_case():
    # 40 tags in a 1 kb window, fragment 200 bp, control scale 0.5 -> 40 * 200 / 1000 * 0.5 = 4
    assert caller.lambda_window(40.0, 1000, 200, 0.5) == pytest.approx(4.0)
    arr = caller.lambda_window(np.array([0.0, 10.0]), 10000, 180, 1.0)
    np.testing.assert_allclose(arr, [0.0, 10 * 180 / 10000])


# --------------------------------------------------------------------------------------------
# products
# --------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def infos():
    return products.load(MANIFEST, COVARIATES)


def test_products_load_every_manifest_pid(infos):
    with open(MANIFEST, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    assert set(infos) == {r["pid"] for r in rows}
    assert len(infos) == 130
    for r in rows:
        info = infos[r["pid"]]
        assert info.has_control == int(r["control_counts25_md5"] != "")
        assert info.fraglen == int(r["fraglen"])
        assert info.depth_reads == int(r["depth"])


def test_product_ratio_arm(infos):
    i = infos["C19M16__ratio__k2"]
    assert i.pid == "C19M16__ratio__k2"
    assert i.track == "C19M16"
    assert i.assay == "H3K27ac"
    assert i.mark_class == "narrow"
    assert (i.arm, i.level) == ("ratio", "k2")
    assert i.depth_reads == 30_000_000
    assert i.read_length == 101
    assert i.run_type_pe == 0
    assert i.fraglen == 180
    assert i.has_control == 1
    assert i.control_reads == 107039349
    assert i.ratio_V == pytest.approx(2 * 30e6 / 107039349, rel=1e-12)
    # the manifest's knob_value is MACS2's --ratio on this arm
    assert i.ratio_V == pytest.approx(0.5605415257, rel=1e-9)
    assert i.ctl_identity == "matched"
    assert i.control_read_length == pytest.approx(101.0, abs=0.01)    # measured mean


def test_product_dnase_base(infos):
    i = infos["C12M02__base__base"]
    assert i.assay == "DNase-seq"
    assert i.mark_class == "DNase"
    assert i.has_control == 0
    assert i.control_reads == 0
    assert i.ratio_V is None
    assert i.fraglen == 150
    assert i.read_length == 76
    assert i.depth_reads == 50_000_000
    assert i.ctl_identity == "none"
    assert i.control_read_length == 0.0


def test_product_dnase_extsize_fraglen(infos):
    assert infos["C12M02__extsize__k0.5"].fraglen == 75
    assert infos["C12M02__extsize__k2"].fraglen == 300


def test_product_non_ratio_arms_have_no_V(infos):
    for pid, i in infos.items():
        if i.arm != "ratio":
            assert i.ratio_V is None, pid


def _read(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def test_control_read_length_table_covers_every_control_product():
    rows = _read(CTL_RL)
    assert list(rows[0]) == ["pid", "control_tagalign", "control_read_length_mean", "n_lines_read"]
    ctl_pids = {r["pid"] for r in _read(MANIFEST) if r["control_counts25_md5"]}
    assert len(ctl_pids) == 114
    assert {r["pid"] for r in rows} == ctl_pids
    for r in rows:
        assert int(r["n_lines_read"]) == 100000
        assert r["control_tagalign"].endswith("tagAlign.gz")


def test_control_read_length_is_measured(infos):
    # the crop arms cropped the control too: its read length follows the crop, not the base
    for track in ("C07M20", "C19M16"):
        assert infos[f"{track}__crop__36"].read_length == 36
        assert infos[f"{track}__crop__36"].control_read_length == pytest.approx(36.0, abs=0.01)
        assert infos[f"{track}__crop__50"].control_read_length == pytest.approx(50.0, abs=0.01)
        assert infos[f"{track}__base__base"].control_read_length == pytest.approx(101.0, abs=0.01)
    assert infos["C07M20__ctlid__none"].control_read_length == 0.0


def test_control_read_length_missing_raises(tmp_path):
    rows = _read(CTL_RL)
    short = tmp_path / "short.tsv"
    with open(short, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows([r for r in rows if r["pid"] != "C19M16__base__base"])
    with pytest.raises(KeyError, match="C19M16__base__base"):
        products.load(MANIFEST, COVARIATES, short)


# --------------------------------------------------------------------------------------------
# feature names and windows
# --------------------------------------------------------------------------------------------

def test_feature_names():
    assert encoding.ASSAYS == ASSAYS
    assert len(features.FEATURES_R1) == 9
    assert features.FEATURES_R1[:2] == ("log1p_k", "log1p_ctl_bin")
    assert len(features.FEATURES_R5) == 21
    assert features.FEATURES_R5[:9] == features.FEATURES_R1
    assert features.FEATURES_R5[9:] == ("log1p_k_frag", "log1p_ctl_frag", "log1p_ctl_1k",
                                        "log1p_ctl_10k", "log_lambda_bg", "log_depth_t",
                                        "log_depth_c", "log_scale_ratio", "log_fraglen",
                                        "log_read_length", "log_read_length_c", "has_control")
    assert features.FEATURES_R5.index("has_control") == 20
    assert features.FEATURE_FORMULA == "log1p_P_formula"


@pytest.mark.parametrize("fraglen,w", [(180, 7), (150, 7), (75, 3), (300, 13), (215, 9),
                                       (90, 5), (10, 1), (25, 1), (50, 3)])
def test_frag_bins(fraglen, w):
    assert features.frag_bins(fraglen) == w


def test_window_bins_constants():
    assert features.window_bins(180) == {"frag": 7, "1k": 40, "10k": 400}


def test_window_mean_odd_at_ends():
    a = np.arange(10, dtype=np.float64)
    m = features.window_mean(a, 5)
    assert m[0] == pytest.approx((0 + 1 + 2) / 3)          # bins 0..2 (truncated left)
    assert m[1] == pytest.approx((0 + 1 + 2 + 3) / 4)
    assert m[5] == pytest.approx((3 + 4 + 5 + 6 + 7) / 5)
    assert m[9] == pytest.approx((7 + 8 + 9) / 3)          # truncated right


def test_window_mean_even_centring():
    # an even window w has w//2 bins on the left of the bin and w - 1 - w//2 on the right
    a = np.arange(10, dtype=np.float64)
    m = features.window_mean(a, 4)
    assert m[0] == pytest.approx((0 + 1) / 2)              # bins 0..1
    assert m[5] == pytest.approx((3 + 4 + 5 + 6) / 4)      # bins 3..6
    assert m[9] == pytest.approx((7 + 8 + 9) / 3)          # bins 7..9


def test_window_sum_and_one_bin_window():
    a = np.array([3, 0, 5, 1], dtype=np.uint32)
    np.testing.assert_allclose(features.window_sum(a, 3), [3, 8, 6, 6])
    np.testing.assert_allclose(features.window_mean(a, 1), a.astype(float))


def test_window_longer_than_chromosome():
    a = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(features.window_mean(a, 400), [2.0, 2.0, 2.0])


# --------------------------------------------------------------------------------------------
# fixed_features
# --------------------------------------------------------------------------------------------

def _info(**kw):
    base = dict(pid="X__base__base", track="X", assay="H3K27ac", mark_class="narrow", arm="base",
                level="base", depth_reads=30_000_000, read_length=101, run_type_pe=0, fraglen=180,
                has_control=1, control_reads=60_000_000, ratio_V=None, ctl_identity="matched",
                control_read_length=101.0)
    base.update(kw)
    return products.ProductInfo(**base)


def _wmean(a, w):
    """Independent reference: a loop over bins, truncated at both ends."""
    a = np.asarray(a, dtype=np.float64)
    left, right = w // 2, w - 1 - w // 2
    out = np.empty(a.size)
    for i in range(a.size):
        lo, hi = max(0, i - left), min(a.size, i + right + 1)
        out[i] = a[lo:hi].mean()
    return out


def test_fixed_features_shapes():
    rng = np.random.default_rng(0)
    n = 1000
    t = rng.poisson(2, n).astype(np.uint32)
    c = rng.poisson(1, n).astype(np.uint32)
    f1 = features.fixed_features(_info(), t, c, 1)
    f5 = features.fixed_features(_info(), t, c, 5)
    assert f1.shape == (n, 8) and f1.dtype == np.float32
    assert f5.shape == (n, 20) and f5.dtype == np.float32
    np.testing.assert_array_equal(f5[:, :8], f1)
    with pytest.raises(ValueError):
        features.fixed_features(_info(), t, c, 3)


def test_fixed_features_with_control_by_hand():
    rng = np.random.default_rng(1)
    n = 900
    t = rng.poisson(3, n).astype(np.uint32)
    c = rng.poisson(2, n).astype(np.uint32)
    info = _info(assay="H3K4me3", control_read_length=50.0)
    f = features.fixed_features(info, t, c, 5)
    cols = {name: f[:, j] for j, name in enumerate(features.FEATURES_R5[1:])}
    np.testing.assert_allclose(cols["log1p_ctl_bin"], np.log1p(c), rtol=1e-6)
    assert features.FEATURES_R5[2:9] == tuple(f"assay_{a}" for a in ASSAYS)
    onehot = f[:, 1:8]
    np.testing.assert_array_equal(onehot, np.tile(np.eye(7)[ASSAYS.index("H3K4me3")], (n, 1)))
    np.testing.assert_allclose(cols["log1p_k_frag"], np.log1p(_wmean(t, 7)), rtol=1e-5)
    np.testing.assert_allclose(cols["log1p_ctl_frag"], np.log1p(_wmean(c, 7)), rtol=1e-5)
    np.testing.assert_allclose(cols["log1p_ctl_1k"], np.log1p(_wmean(c, 40)), rtol=1e-5)
    np.testing.assert_allclose(cols["log1p_ctl_10k"], np.log1p(_wmean(c, 400)), rtol=1e-5)
    # control 60 M > treatment 30 M -> s_C = 0.5, s_T = 1; lambda_bg from the treatment
    lam_bg = 180 * 30e6 / 2.7e9
    np.testing.assert_allclose(cols["log_lambda_bg"], math.log(lam_bg), rtol=1e-6)
    np.testing.assert_allclose(cols["log_depth_t"], math.log(30.0), rtol=1e-6)
    np.testing.assert_allclose(cols["log_depth_c"], math.log(60.0), rtol=1e-6)
    np.testing.assert_allclose(cols["log_scale_ratio"], math.log(0.5), rtol=1e-6)
    np.testing.assert_allclose(cols["log_fraglen"], math.log(180), rtol=1e-6)
    np.testing.assert_allclose(cols["log_read_length"], math.log(101), rtol=1e-6)
    np.testing.assert_allclose(cols["log_read_length_c"], math.log(50), rtol=1e-6)
    np.testing.assert_array_equal(cols["has_control"], 1.0)
    np.testing.assert_array_equal(f[:, -1], 1.0)


def test_fixed_features_ratio_arm_uses_V():
    n = 50
    t = np.ones(n, np.uint32)
    c = np.ones(n, np.uint32)
    v = 2 * 30e6 / 107039349
    info = _info(arm="ratio", level="k2", control_reads=107039349, ratio_V=v)
    f = features.fixed_features(info, t, c, 5)
    j = features.FEATURES_R5.index("log_scale_ratio") - 1
    np.testing.assert_allclose(f[:, j], math.log(v), rtol=1e-6)        # s_C = V, s_T = 1
    j = features.FEATURES_R5.index("log_lambda_bg") - 1
    # --ratio does not move lambda_bg: the treatment's 30 M lines, s_T = 1
    np.testing.assert_allclose(f[:, j], math.log(180 * 30e6 / 2.7e9), rtol=1e-6)


def test_fixed_features_q025_scales_treatment():
    n = 20
    info = _info(control_reads=26759837)
    f = features.fixed_features(info, np.ones(n, np.uint32), np.ones(n, np.uint32), 5)
    r = 30e6 / 26759837
    j = features.FEATURES_R5.index("log_scale_ratio") - 1
    np.testing.assert_allclose(f[:, j], math.log(1.0 / (1.0 / r)), rtol=1e-6)   # s_C/s_T = r
    j = features.FEATURES_R5.index("log_lambda_bg") - 1
    # lambda_bg from the treatment scaled down: 180 * 30e6 * (1 / r) / 2.7e9
    np.testing.assert_allclose(f[:, j], math.log(180 * 30e6 * (1 / r) / 2.7e9), rtol=1e-6)


def test_fixed_features_no_control_branch():
    rng = np.random.default_rng(2)
    n = 600
    t = rng.poisson(4, n).astype(np.uint32)
    info = _info(assay="DNase-seq", mark_class="DNase", depth_reads=50_000_000, read_length=76,
                 fraglen=150, has_control=0, control_reads=0, ctl_identity="none",
                 control_read_length=0.0)
    f = features.fixed_features(info, t, None, 5)
    cols = {name: f[:, j] for j, name in enumerate(features.FEATURES_R5[1:])}
    for k in ("log1p_ctl_bin", "log1p_ctl_frag", "log1p_ctl_1k", "log1p_ctl_10k",
              "log_depth_c", "log_scale_ratio", "log_read_length_c", "has_control"):
        np.testing.assert_array_equal(cols[k], 0.0)
    np.testing.assert_allclose(cols["log_read_length"], math.log(76), rtol=1e-6)
    np.testing.assert_allclose(cols["log_lambda_bg"], math.log(150 * 50e6 / 2.7e9), rtol=1e-6)
    np.testing.assert_allclose(cols["log1p_k_frag"], np.log1p(_wmean(t, 7)), rtol=1e-5)
    np.testing.assert_array_equal(f[:, 1:8], np.tile(np.eye(7)[0], (n, 1)))
    f1 = features.fixed_features(info, t, None, 1)
    np.testing.assert_array_equal(f1[:, 0], 0.0)


def test_fixed_features_control_mismatch_raises():
    t = np.ones(10, np.uint32)
    with pytest.raises(ValueError):
        features.fixed_features(_info(), t, None, 5)
    with pytest.raises(ValueError):
        features.fixed_features(_info(has_control=0, control_reads=0, ctl_identity="none"),
                                t, t, 5)
    with pytest.raises(ValueError):
        features.fixed_features(_info(), t, np.ones(9, np.uint32), 5)


def test_fixed_features_treat_mean_only_moves_treatment_windows():
    rng = np.random.default_rng(3)
    n = 500
    t = rng.poisson(3, n).astype(np.uint32)
    c = rng.poisson(2, n).astype(np.uint32)
    mean_t = rng.gamma(2.0, 2.0, n)
    a = features.fixed_features(_info(), t, c, 5)
    b = features.fixed_features(_info(), t, c, 5, treat_mean=mean_t)
    j = features.FEATURES_R5.index("log1p_k_frag") - 1
    np.testing.assert_allclose(b[:, j], np.log1p(_wmean(mean_t, 7)), rtol=1e-5)
    keep = [i for i in range(a.shape[1]) if i != j]
    np.testing.assert_array_equal(a[:, keep], b[:, keep])
    with pytest.raises(ValueError):
        features.fixed_features(_info(), t, c, 5, treat_mean=mean_t[:-1])


def test_k_column():
    k = np.array([0, 1, 9], dtype=np.int64)
    np.testing.assert_allclose(features.k_column(k), np.log1p([0.0, 1.0, 9.0]))


def test_k_frag_chain_hand_case():
    mean_t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    k = np.array([[0, 10]] * 5, dtype=np.int64)
    out = features.k_frag_chain(k, mean_t, 3)
    assert out.shape == (5, 2)
    # interior bin 2: window 1..3, neighbours' means 2 + 4 = 6
    np.testing.assert_allclose(out[2], [(0 + 6) / 3, (10 + 6) / 3])
    # bin 0: window 0..1 (truncated), neighbour 2, two bins in the window
    np.testing.assert_allclose(out[0], [(0 + 2) / 2, (10 + 2) / 2])
    # bin 4: window 3..4, neighbour 4
    np.testing.assert_allclose(out[4], [(0 + 4) / 2, (10 + 4) / 2])
    np.testing.assert_allclose(features.frag_rest(mean_t, 3), [2, 4, 6, 8, 4])


def test_k_frag_chain_reduces_to_window_mean_at_the_true_counts():
    rng = np.random.default_rng(4)
    x = rng.poisson(3, 200).astype(np.float64)
    out = features.k_frag_chain(x[:, None], x, 7)
    np.testing.assert_allclose(out[:, 0], _wmean(x, 7), rtol=1e-12)


# --------------------------------------------------------------------------------------------
# control cache
# --------------------------------------------------------------------------------------------

N_BINS = {"chr1": 400, "chr19": 150, "chr21": 120, "chr22": 100}


def _md5(path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


@pytest.fixture()
def synth_with_controls(tmp_path):
    root = tmp_path / "products"
    manifest, covs = synth.make_products(root, 0, n_bins=N_BINS)
    with open(manifest, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    ctl = {}
    rng = np.random.default_rng(11)
    for r in rows:
        if r["track"] != "T1":
            continue
        arrays = {c: rng.poisson(2.0, n).astype(np.uint32) for c, n in N_BINS.items()}
        path = root / r["pid"] / "control_counts25.npz"
        np.savez_compressed(path, **arrays)
        r["control_counts25_md5"] = _md5(path)
        ctl[r["pid"]] = arrays
    with open(root / "control_read_length.tsv", "w") as fh:
        fh.write("pid\tcontrol_tagalign\tcontrol_read_length_mean\tn_lines_read\n")
        for pid in sorted(ctl):
            fh.write(f"{pid}\tsynthetic\t101.0\t100000\n")
    with open(manifest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return root, manifest, {r["pid"]: r for r in rows}, ctl


def test_control_cache_round_trip(tmp_path, synth_with_controls):
    root, manifest, rows, ctl = synth_with_controls
    cache = tmp_path / "ctl_cache"
    pid = "T1__base__base"
    st = controls.build_control_cache(root, cache, pid, rows[pid])
    assert st == {"counts": "built"}
    meta = json.loads((cache / f"{pid}__counts.json").read_text())
    assert meta["space"] == "counts" and meta["pid"] == pid
    assert meta["source_npz_md5"] == rows[pid]["control_counts25_md5"]
    corpus = ldata.Corpus(cache)
    assert corpus.is_cache
    for c, a in ctl[pid].items():
        got = corpus.get(pid, "counts", c)
        assert got.dtype == np.uint32
        np.testing.assert_array_equal(got, a)
    # the treatment counts are not what the cache returns
    treat = ldata.Corpus(root).get(pid, "counts", "chr1")
    assert not np.array_equal(treat, corpus.get(pid, "counts", "chr1"))
    assert controls.build_control_cache(root, cache, pid, rows[pid]) == {"counts": "skipped"}


def test_products_load_needs_the_control_md5(tmp_path, synth_with_controls):
    root, manifest, rows, ctl = synth_with_controls
    infos = products.load(manifest, root / "covariates.tsv", root / "control_read_length.tsv")
    assert infos["T1__base__base"].has_control == 1
    assert infos["T1__base__base"].control_read_length == 101.0
    assert infos["T2__base__base"].has_control == 0
    raw = tmp_path / "raw"
    m2, c2 = synth.make_products(raw, 0, n_bins=N_BINS)   # covariates say control, no npz
    with pytest.raises(ValueError, match="control_counts25_md5"):
        products.load(m2, c2, root / "control_read_length.tsv")


def test_control_cache_absent_and_md5(tmp_path, synth_with_controls):
    root, manifest, rows, ctl = synth_with_controls
    cache = tmp_path / "ctl_cache"
    dn = "T2__base__base"
    assert rows[dn]["control_counts25_md5"] == ""
    assert controls.build_control_cache(root, cache, dn, rows[dn]) == {"counts": "absent"}
    assert not (cache / f"{dn}__counts.json").exists()
    bad = dict(rows["T1__pe__pe"], control_counts25_md5="0" * 32)
    with pytest.raises(ValueError, match="md5"):
        controls.build_control_cache(root, cache, "T1__pe__pe", bad)
    assert not (cache / "T1__pe__pe__counts.json").exists()
    with pytest.raises(ValueError):
        controls.build_control_cache(root, cache, "T1__pe__pe", rows["T1__base__base"])


def _env():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    return env


def test_control_cache_cli(tmp_path, synth_with_controls):
    root, manifest, rows, ctl = synth_with_controls
    cache = tmp_path / "ctl_cache_cli"
    order = sorted(rows)
    i = order.index("T1__depth__15M")
    out = subprocess.run([sys.executable, str(ROOT / "tools" / "t125" / "hmap" / "controls.py"),
                          "cache", str(manifest), str(root), str(cache), str(i)],
                         capture_output=True, text=True, env=_env(), check=True)
    assert out.stdout.strip() == "T1__depth__15M: counts=built"
    np.testing.assert_array_equal(ldata.Corpus(cache).get("T1__depth__15M", "counts", "chr22"),
                                  ctl["T1__depth__15M"]["chr22"])
    j = order.index("T2__base__base")
    out = subprocess.run([sys.executable, str(ROOT / "tools" / "t125" / "hmap" / "controls.py"),
                          "cache", str(manifest), str(root), str(cache), str(j)],
                         capture_output=True, text=True, env=_env(), check=True)
    assert out.stdout.strip() == "T2__base__base: counts=absent"


# --------------------------------------------------------------------------------------------
# task tables
# --------------------------------------------------------------------------------------------

def test_h_variants():
    assert tasks.H_FORMULA == (("formula", "bin"), ("formula", "frag"))
    want = ([(v, r, n) for v in ("affine", "mlp") for r in (1, 5) for n in (0, 1)]
            + [(v, 5, n) for v in ("formula_affine", "formula_mlp") for n in (0, 1)])
    assert list(tasks.H_LEARNED) == want
    assert len(tasks.H_LEARNED) == 12


def test_h_name():
    assert tasks.h_name("mlp", 5, 1, "true", 2) == "h_mlp_r5_noise_true_s2"
    assert tasks.h_name("formula_affine", 5, 0, "B2_all_counts_real_s0", 0) == \
        "h_formula_affine_r5_push_B2_all_counts_real_s0_s0"


def test_tasks_true():
    t = tasks.tasks_true()
    assert len(t) == 36
    assert t[0]["h_name"] == "h_affine_r1_push_true_s0"
    for row in t:
        assert set(row) >= {"index", "variant", "rung", "noise", "source", "seed", "h_name"}
        li = tasks.H_LEARNED.index((row["variant"], row["rung"], row["noise"]))
        assert row["index"] == li * 3 + row["seed"]
        assert row["source"] == "true"
    assert [r["index"] for r in t] == list(range(36))
    assert len({r["h_name"] for r in t}) == 36
    assert t[35]["h_name"] == "h_formula_mlp_r5_noise_true_s2"


def test_tasks_fnb():
    t = tasks.tasks_fnb(["A_all_counts_real_s0"])
    assert len(t) == 12
    assert t[0]["h_name"] == "h_affine_r1_push_A_all_counts_real_s0_s0"
    assert t[0]["source"] == "fnb:A_all_counts_real_s0"
    f_runs = [f"R{i}" for i in range(64)]
    t = tasks.tasks_fnb(f_runs)
    assert len(t) == 768
    assert [r["index"] for r in t] == list(range(768))
    t3 = tasks.tasks_fnb(f_runs, seeds=(0, 1, 2))
    assert len(t3) == 2304
    for row in t3[:100] + t3[-100:]:
        fi = f_runs.index(row["source"].split(":", 1)[1])
        li = tasks.H_LEARNED.index((row["variant"], row["rung"], row["noise"]))
        assert row["index"] == fi * 36 + li * 3 + row["seed"]
        assert row["h_name"] == tasks.h_name(row["variant"], row["rung"], row["noise"],
                                             f_runs[fi], row["seed"])


def test_tasks_alone():
    t = tasks.tasks_alone()
    assert len(t) == 38
    assert [r["h_name"] for r in t[:2]] == ["h_formula_bin", "h_formula_frag"]
    assert [r["h_name"] for r in t[2:]] == [r["h_name"] for r in tasks.tasks_true()]
    assert [r["index"] for r in t] == list(range(38))


def test_chain_h_list():
    one = tasks.chain_h_list("A_C07M20_counts_real_s0", 1)
    assert len(one) == 2 + 36 + 12
    assert one[:2] == ["h_formula_bin", "h_formula_frag"]
    assert one[2:38] == [r["h_name"] for r in tasks.tasks_true()]
    assert one[38] == "h_affine_r1_push_A_C07M20_counts_real_s0_s0"
    three = tasks.chain_h_list("A_C07M20_counts_real_s0", 3)
    assert len(three) == 2 + 36 + 36
    assert len(set(three)) == len(three)


def _cli(*args):
    return subprocess.run([sys.executable, str(ROOT / "tools" / "t125" / "hmap" / "tasks.py"),
                           *args], capture_output=True, text=True, env=_env(), check=True).stdout


def test_tasks_cli_true_and_alone():
    out = _cli("true", str(MANIFEST)).splitlines()
    assert len(out) == 37
    header = out[0].split("\t")
    assert header[0] == "index" and header[-1] == "h_name"
    assert out[1].split("\t")[-1] == "h_affine_r1_push_true_s0"
    out = _cli("alone", str(MANIFEST)).splitlines()
    assert len(out) == 39
    assert out[1].split("\t")[-1] == "h_formula_bin"


def test_tasks_cli_chain_list():
    out = _cli("chain-list", "A_C07M20_counts_real_s0", "--fnb-seeds", "1").splitlines()
    assert out == tasks.chain_h_list("A_C07M20_counts_real_s0", 1)


def test_tasks_cli_fnb():
    pytest.importorskip("eta.pairs", reason="eta.pairs (chunk t125-P1a) not in this tree yet")
    out = _cli("fnb", str(MANIFEST), "--fnb-seeds", "1").splitlines()
    assert len(out) == 769
    assert out[1].split("\t")[-1] == "h_affine_r1_push_A_C07M20_counts_real_s0_s0"
