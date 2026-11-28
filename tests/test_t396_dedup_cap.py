"""Tests of tools/t396/dedup_cap.py: the dedup cap test at 25 bp on small synthetic stores (t396 C10a)."""
import csv
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
from t396 import caps  # noqa: E402
from t396 import dedup_cap as D  # noqa: E402
from t396 import deconv  # noqa: E402

CHROMS = {"chr1": 12000, "chr2": 8000}
# span kernel: a read covers its start bin and the next two bins downstream (3 bins, ~75 bp)
KERNEL = np.zeros((2, 17))
KERNEL[0, 8:11] = 1.0
KERNEL[1, 6:9] = 1.0
COLUMNS = ["measurement_id", "experiment", "assay", "in_eic", "unit", "layer", "arm_knob",
           "parent_measurement_id", "depth_reads", "run_type", "read_length", "pipeline", "split"]


# ------------------------------------------------------------------ synthetic stores
def write_view(root_rs, root_ct, mid, starts, n_units, kernel=KERNEL):
    """starts: {chrom: int [n, 2]} -> starts.h5 + provenance.json (rs store), counts.h5 (count
    store; overlap = starts pushed through the kernel, exact integers)."""
    import h5py
    for root in (root_rs, root_ct):
        os.makedirs(os.path.join(root, mid), exist_ok=True)
    with h5py.File(os.path.join(root_rs, mid, "starts.h5"), "w") as f:
        for c, s in starts.items():
            f[c] = s.astype(np.uint32)
        f.attrs["kernel"] = kernel
    with h5py.File(os.path.join(root_ct, mid, "counts.h5"), "w") as f:
        for c, s in starts.items():
            ov = deconv.forward(s[:, 0], kernel[0]) + deconv.forward(s[:, 1], kernel[1])
            f[c] = np.rint(ov).astype(np.uint32)
    files = [f"/x/{mid}/br{i + 1}/pr1.tagAlign.gz" for i in range(n_units)]
    prov = {"read_files": {"pr1": files, "pr2": files}, "lines_per_fragment": 1}
    for root in (root_rs, root_ct):
        json.dump(prov, open(os.path.join(root, mid, "provenance.json"), "w"))


def write_measurements(path, rows):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "measurements.tsv"), "w", newline="") as f:
        w = csv.DictWriter(f, COLUMNS, delimiter="\t", extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in COLUMNS})


def intensities(rng, scale=1.0):
    """Per-chromosome strand-bin intensities: a low background with a few deep peaks."""
    out = {}
    for c, n in CHROMS.items():
        lam = rng.gamma(0.3, 3.0, (n, 2))
        peaks = rng.choice(n, n // 40, replace=False)
        lam[peaks] += rng.gamma(2.0, 60.0, (peaks.size, 2))
        out[c] = lam * scale
    return out


def dedup_pair(rng, a, units, mode="cap", kappa=3.0, L=50):
    """(off starts, on starts). mode "cap": the on view keeps the distinct positions of each unit,
    the units sharing Dirichlet(a) weights over 25 positions. mode "count": the on starts are the
    off starts thinned per bin by the count-space saturation of the bin's overlap count."""
    lam = intensities(rng)
    off, on = {}, {}
    shares = np.full(units, 1.0 / units)
    for c, l in lam.items():
        s = rng.poisson(l)
        off[c] = s
        if mode == "cap":
            t = np.zeros_like(s)
            for j in (0, 1):
                t[:, j] = D.draw_shared(s[:, j], a, shares, rng)
        else:
            ov = deconv.forward(s[:, 0], KERNEL[0]) + deconv.forward(s[:, 1], KERNEL[1])
            keep = np.where(ov > 0, caps.count_saturation(ov, kappa, L) / np.maximum(ov, 1), 1.0)
            t = rng.binomial(s, np.clip(keep, 0, 1)[:, None])
        on[c] = t
    return off, on


def build_dedup_case(tmp, rng, a=1.0, units=2, mode="cap", n_train=4, n_val=2, L=50):
    rs, ct, tb = (str(tmp / x) for x in ("rs", "ct", "tables"))
    rows = []
    for i in range(n_train + n_val):
        exp = f"ENCSR{i:03d}"
        split = "train" if i < n_train else "val"
        off_id, on_id = f"{exp}.pooled.uniform.nodedup", f"{exp}.pooled.uniform.base"
        off, on = dedup_pair(rng, a, units, mode=mode, L=L)
        write_view(rs, ct, off_id, off, units)
        write_view(rs, ct, on_id, on, units)
        common = {"experiment": exp, "assay": "DNase-seq", "in_eic": split, "unit": "pooled",
                  "layer": "uniform", "run_type": "single-ended", "read_length": str(L),
                  "split": "blind"}               # a decoy: rows must read in_eic
        rows.append({**common, "measurement_id": on_id, "arm_knob": "",
                     "depth_reads": str(sum(int(v.sum()) for v in on.values()))})
        rows.append({**common, "measurement_id": off_id, "arm_knob": "nodedup",
                     "parent_measurement_id": on_id,
                     "depth_reads": str(sum(int(v.sum()) for v in off.values()))})
    # one blind pair that rows must skip
    rows.append({"measurement_id": "ENCSR999.pooled.testbed.nodedup", "experiment": "ENCSR999",
                 "assay": "DNase-seq", "in_eic": "blind", "layer": "testbed", "arm_knob": "nodedup",
                 "parent_measurement_id": "ENCSR999.pooled.testbed.base", "run_type": "single-ended",
                 "read_length": "36", "split": "train"})
    write_measurements(tb, rows)
    return rs, ct, tb


def run_all(tmp, rs, ct, tb, rule, floor_count=True, n_boot=3):
    json.dump({"dedup_unit": rule}, open(tmp / "UNIT_RULE.json", "w"))
    D.main(["rows", "--tables", tb, "--rs-store", rs, "--count-store", ct, "--out", str(tmp / "rows.tsv")])
    rows = D.read_tsv(tmp / "rows.tsv")
    stats = []
    for r in rows:
        out = str(tmp / "stats" / f"{r['on_id']}.npz")
        D.main(["stats", "--pair", r["off_id"], r["on_id"], "--unit-rule", str(tmp / "UNIT_RULE.json"),
                "--rows", str(tmp / "rows.tsv"), "--rs-store", rs, "--count-store", ct, "--out", out])
        stats.append(out)
    D.main(["fit", "--stats", *stats, "--rows", str(tmp / "rows.tsv"), "--n-boot", str(n_boot),
            "--out", str(tmp / "per_assay.tsv")])
    floors = []
    if floor_count:
        for r in rows:
            if r["in_eic"] != "val":
                continue
            out = str(tmp / "floor" / f"{r['on_id']}.npz")
            D.main(["floor-count", "--pair", r["off_id"], r["on_id"], "--unit-rule",
                    str(tmp / "UNIT_RULE.json"), "--fit", str(tmp / "per_assay.tsv"), "--rows",
                    str(tmp / "rows.tsv"), "--rs-store", rs, "--count-store", ct, "--out", out])
            floors.append(out)
    D.main(["score", "--stats", *stats, "--fit", str(tmp / "per_assay.tsv"), "--rows",
            str(tmp / "rows.tsv"), "--floor-count", *floors, "--n-boot", "200", "--n-rep", "10",
            "--out", str(tmp / "score")])
    return D.read_tsv(tmp / "score" / "table.tsv")


def pick(table, law, scope="top1pct", in_eic="val"):
    r = [x for x in table if x["law"] == law and x["scope"] == scope and x["in_eic"] == in_eic]
    assert len(r) == 1, (law, scope, in_eic, r)
    return r[0]


# ------------------------------------------------------------------ helpers
def test_fast_table_and_top_masks_match_caps_joint_hist():
    rng = np.random.default_rng(1)
    v = {"a": rng.poisson(3, 500), "b": rng.poisson(3, 700)}   # many ties at the threshold
    t = {c: rng.binomial(x, 0.6) for c, x in v.items()}
    ref = caps.joint_hist(np.concatenate(list(v.values())), np.concatenate(list(t.values())))
    m = D.top_masks(v)
    assert sum(x.sum() for x in m.values()) == int(np.ceil(0.01 * 1200))
    got_top = caps.merge_hists([D.fast_table(v[c][m[c]], t[c][m[c]]) for c in v])
    got_gen = caps.merge_hists([D.fast_table(v[c], t[c]) for c in v])
    assert got_top == ref["top1pct"]
    assert got_gen == ref["genome"]


def test_hist_roundtrip():
    h = {(0, 0): 5, (3, 2): 7, (40, 25): 1}
    assert D.arr_to_hist(D.hist_to_arr(h)) == h


def test_read_length_of():
    assert D.read_length_of("36") == 36
    assert D.read_length_of("50,36") == 36


# ------------------------------------------------------------------ shared positional weights
@pytest.mark.parametrize("a", [0.05, 0.2, 1.0, 5.0])
def test_shared_weight_law_matches_simulation_and_sizes_the_independent_slot_bias(a):
    """Two units that share their 25 positional weights (reads split 50/50): the closed form
    sum_u thinned_dm_distinct(c, 1/2, a, 25) is the Monte Carlo mean; caps.dm_distinct(c, a, 50)
    (50 independent slots) is low by up to ~8% at a = 0.05 and by under 1% at a >= 1."""
    rng = np.random.default_rng(7)
    n = 40000
    biases = []
    for c in (5, 10, 25, 100):
        mc = D.draw_shared(np.full(n, c), a, [0.5, 0.5], rng)
        se = mc.std() / np.sqrt(n)
        shared = float(D.dm_distinct_shared(np.array([c]), a, [0.5, 0.5])[0])
        indep = float(caps.dm_distinct(c, a, 50))
        assert abs(shared - mc.mean()) < 4 * se + 1e-9, (c, shared, mc.mean(), se)
        biases.append(indep / shared - 1.0)
    worst = min(biases)
    if a <= 0.05:
        assert worst < -0.05                 # large: the shared-weight law is needed
    elif a >= 1.0:
        assert worst > -0.01
    assert max(biases) <= 1e-9               # the independent-slot law never over-predicts here


def test_shared_weight_law_single_unit_is_dm_distinct():
    c = np.arange(0, 60)
    np.testing.assert_allclose(D.dm_distinct_shared(c, 0.4, [1.0]), caps.dm_distinct(c, 0.4, 25),
                               rtol=1e-10)


# ------------------------------------------------------------------ rows
def test_rows_read_in_eic_not_split_and_count_units(tmp_path):
    rng = np.random.default_rng(3)
    rs, ct, tb = build_dedup_case(tmp_path, rng, n_train=2, n_val=1)
    D.main(["rows", "--tables", tb, "--rs-store", rs, "--count-store", ct,
            "--out", str(tmp_path / "rows.tsv")])
    rows = D.read_tsv(tmp_path / "rows.tsv")
    assert len(rows) == 3                                     # the blind pair is skipped
    assert sorted(r["in_eic"] for r in rows) == ["train", "train", "val"]
    assert all(r["units"] == "2" and r["in_store"] == "1" for r in rows)
    assert all(r["layer"] == "uniform" and r["run_type"] == "single-ended" for r in rows)
    assert set(D.ROW_COLUMNS) <= set(rows[0])
    with open(tmp_path / "rows.tsv") as f:
        assert "split" not in f.readline().split("\t")


# ------------------------------------------------------------------ the cap test end to end
@pytest.fixture(scope="module")
def cap_case(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cap")
    rng = np.random.default_rng(11)
    rs, ct, tb = build_dedup_case(tmp, rng, a=1.0, units=2)
    return tmp, rs, ct, tb


def test_per_replicate_dedup_needs_K50(cap_case):
    """Data deduplicated per replicate with 2 units: the start law with K = 50 (replicate rule)
    closes over 0.9 of the gap to its floor in the top 1%; with K = 25 (pooled rule) it does not."""
    tmp, rs, ct, tb = cap_case
    (tmp / "k50").mkdir()
    (tmp / "k25").mkdir()
    t50 = run_all(tmp / "k50", rs, ct, tb, "replicate", floor_count=False)
    t25 = run_all(tmp / "k25", rs, ct, tb, "pooled", floor_count=False)
    g50 = float(pick(t50, "dm_distinct")["gap_closed_vs_depth_scaling"])
    g25 = float(pick(t25, "dm_distinct")["gap_closed_vs_depth_scaling"])
    assert g50 > 0.9, g50
    assert g25 < 0.9, g25
    assert g50 > g25
    fit = D.read_tsv(tmp / "k50" / "per_assay.tsv")
    a_fit = float([r for r in fit if r["assay"] == "DNase-seq" and r["run_type"] == "single-ended"][0]["a"])
    assert 0.5 < a_fit < 2.0, a_fit
    assert pick(t50, "dm_distinct")["explains"] == "yes"
    head = open(tmp / "k50" / "score" / "table.tsv").readline().rstrip("\n").split("\t")
    for col in ("law", "space", "layer", "run_type", "in_eic", "scope", "gap_closed_vs_depth_scaling",
                "gap_closed_vs_identity", "attainable_floor", "bootstrap_spread", "n_pairs",
                "explains", "beats_other_law"):
        assert col in head


def test_count_law_data_favor_count_law(tmp_path):
    """On the dedup-on track made by count-space saturation, the count law closes more of its gap
    than the start law, in the top 1% of the val pairs."""
    rng = np.random.default_rng(5)
    rs, ct, tb = build_dedup_case(tmp_path, rng, mode="count", units=1)
    t = run_all(tmp_path, rs, ct, tb, "replicate", floor_count=True)
    gc = float(pick(t, "count_saturation")["gap_closed_vs_depth_scaling"])
    gs = float(pick(t, "dm_distinct")["gap_closed_vs_depth_scaling"])
    assert gc > gs, (gc, gs)
    assert np.isfinite(float(pick(t, "count_saturation")["attainable_floor"]))
