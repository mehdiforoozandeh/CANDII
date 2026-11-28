"""Tests of tools/t396/sepe_cap.py: the SE/PE cap test at 25 bp on small synthetic stores (t396 C10a)."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
from t396 import caps  # noqa: E402
from t396 import dedup_cap as D  # noqa: E402
from t396 import sepe_cap as S  # noqa: E402
from tests.test_t396_dedup_cap import intensities, write_measurements, write_view  # noqa: E402

A_TRUE = 0.7


def se_pair(rng, a=A_TRUE, q=0.5):
    """(PE dedup-off, PE base, SE view): the base keeps 80% of the dedup-off starts; the SE view
    keeps one mate (each line with probability q) and caps it per position, Dirichlet(a) over 25."""
    lam = intensities(rng)
    nd, base, se = {}, {}, {}
    for c, l in lam.items():
        x = rng.poisson(l * 1.25)
        b = rng.binomial(x, 0.8)
        m = rng.binomial(b, q)
        t = np.stack([caps._draw_distinct(m[:, j], a, 25, rng) for j in (0, 1)], 1)
        nd[c], base[c], se[c] = x, b, t
    return nd, base, se


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sepe")
    rng = np.random.default_rng(21)
    rs, ct, tb = (str(tmp / x) for x in ("rs", "ct", "tables"))
    rows = []
    kinds = ["paired-ended"] * 4 + ["mixed"] * 2
    for i, rt in enumerate(kinds):
        exp = f"ENCSR{i:03d}"
        nd, base, se = se_pair(rng)
        ids = {k: f"{exp}.pooled.uniform.{k}" for k in ("base", "nodedup", "se")}
        for k, v in (("base", base), ("nodedup", nd), ("se", se)):
            write_view(rs, ct, ids[k], v, 1)
        common = {"experiment": exp, "assay": "H3K4me3", "in_eic": "val" if i == 0 else "train",
                  "unit": "pooled", "layer": "uniform", "read_length": "50",
                  "pipeline": "chip-seq-pipeline2", "split": "blind"}
        rows.append({**common, "measurement_id": ids["base"], "arm_knob": "", "run_type": rt,
                     "depth_reads": "1000"})
        rows.append({**common, "measurement_id": ids["nodedup"], "arm_knob": "nodedup",
                     "parent_measurement_id": ids["base"], "run_type": rt})
        rows.append({**common, "measurement_id": ids["se"], "arm_knob": "se",
                     "parent_measurement_id": ids["base"], "run_type": "single-ended",
                     "depth_reads": "400"})
    # an SE view whose parent is not in the store: rows skips it
    rows.append({"measurement_id": "ENCSR900.pooled.uniform.se", "experiment": "ENCSR900",
                 "assay": "H3K4me3", "in_eic": "train", "arm_knob": "se", "layer": "uniform",
                 "parent_measurement_id": "ENCSR900.pooled.uniform.base", "run_type": "single-ended",
                 "read_length": "36"})
    write_measurements(tb, rows)
    json.dump({"dedup_unit": "replicate"}, open(tmp / "UNIT_RULE.json", "w"))
    rule = str(tmp / "UNIT_RULE.json")
    S.main(["rows", "--tables", tb, "--rs-store", rs, "--count-store", ct, "--out", str(tmp / "rows.tsv")])
    prs = D.read_tsv(tmp / "rows.tsv")
    stats = []
    for r in prs:
        out = str(tmp / "stats" / f"{r['se_id']}.npz")
        S.main(["stats", "--pair", r["parent_id"], r["se_id"], "--unit-rule", rule,
                "--rs-store", rs, "--count-store", ct, "--out", out])
        stats.append(out)
    S.main(["fit", "--stats", *stats, "--rows", str(tmp / "rows.tsv"), "--out", str(tmp / "per_assay.tsv")])
    floors = []
    for r in prs:
        out = str(tmp / "floor" / f"{r['se_id']}.npz")
        S.main(["floor-count", "--pair", r["parent_id"], r["se_id"], "--unit-rule", rule, "--fit",
                str(tmp / "per_assay.tsv"), "--rows", str(tmp / "rows.tsv"), "--rs-store", rs,
                "--count-store", ct, "--out", out])
        floors.append(out)
    S.main(["score", "--stats", *stats, "--fit", str(tmp / "per_assay.tsv"), "--rows",
            str(tmp / "rows.tsv"), "--floor-count", *floors, "--n-boot", "200", "--n-rep", "10",
            "--out", str(tmp / "score")])
    return tmp, prs


def test_rows_keep_parents_in_store_and_read_in_eic(case):
    tmp, prs = case
    assert len(prs) == 6
    assert sorted(r["parent_run_type"] for r in prs).count("mixed") == 2
    assert sorted(r["in_eic"] for r in prs).count("val") == 1        # from in_eic, not split
    assert all(r["se_mate"] and r["se_dedup_rule"] and r["units"] == "1" for r in prs)
    assert all(r["parent_nodedup_id"].endswith(".nodedup") for r in prs)
    with open(tmp / "rows.tsv") as f:
        assert "split" not in f.readline().rstrip("\n").split("\t")


def test_recovers_half_thinning(case):
    tmp, _ = case
    fit = {r["experiment"]: r for r in D.read_tsv(tmp / "per_assay.tsv")}
    assert abs(float(fit["all"]["q_fit"]) - 0.5) <= 0.04, fit["all"]
    assert 0.35 < float(fit["all"]["a"]) < 1.4, fit["all"]
    assert len(fit) == 7                                   # 6 held-out experiments + all
    loeo = [float(fit[e]["a"]) for e in fit if e != "all"]
    assert max(loeo) - min(loeo) < 0.5


def test_thinned_cap_explains_se_views_and_reports_parents_apart(case):
    tmp, _ = case
    t = D.read_tsv(tmp / "score" / "table.tsv")
    head = open(tmp / "score" / "table.tsv").readline().rstrip("\n").split("\t")
    assert "parent_run_type" in head and "layer" not in head
    for rt in ("paired-ended", "mixed"):
        r = [x for x in t if x["law"] == S.PRIMARY and x["scope"] == "top1pct"
             and x["parent_run_type"] == rt and x["in_eic"] == "all"]
        assert len(r) == 1
        assert float(r[0]["gap_closed_vs_depth_scaling"]) > 0.9, r[0]
        assert r[0]["explains"] == "yes"
    laws = {x["law"] for x in t}
    assert {S.PRIMARY, "count_saturation"} <= laws


def test_pe_cap_size_reported(case):
    tmp, _ = case
    pc = D.read_tsv(tmp / "score" / "pe_cap.tsv")
    assert len(pc) == 6
    v = np.array([float(r["pe_base_over_nodedup_top1pct"]) for r in pc])
    assert np.all(np.abs(v - 0.8) < 0.03), v                 # the base kept 80% of the reads


# ------------------------------------------------------------------ testbed pairs (iteration 2)
from tests.test_t396_dedup_cap import TB_RATES, build_testbed, run_testbed, tb_pick  # noqa: E402


@pytest.fixture(scope="module")
def sepe_tb(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sepe_testbed")
    rs, ct, tb = build_testbed(tmp, np.random.default_rng(43))
    res = run_testbed(S, tmp, rs, ct, tb, str(tmp / "totals.tsv"), "sepe", floor_count=True)
    return tmp, rs, ct, tb, res


def test_sepe_testbed_rows(sepe_tb):
    rows = sepe_tb[4]["rows"]
    assert len(rows) == 7
    assert all(r["source_id"].endswith(".pe") and r["target_id"].endswith(".base")
               and r["covariate"] == "run_type" and r["share"] == "0.5"
               and r["lines_per_fragment_source"] == "2" and r["run_type"] == "single-ended"
               for r in rows)
    assert not any(r["source_id"].endswith(".sex") for r in rows)       # arm_value must be pe
    r0 = [r for r in rows if r["experiment"] == "ENCSR100"][0]
    assert float(r0["q_source"]) == pytest.approx(TB_RATES[0][1], rel=1e-4)


def test_sepe_testbed_mate_share_and_rates_close_the_gap(sepe_tb):
    t = sepe_tb[4]["table"]
    assert {x["covariate"] for x in t} == {"run_type"}
    for scope in ("genome", "top1pct"):
        row = tb_pick(t, D.TB_PRIMARY, scope=scope)
        assert float(row["gap_closed_vs_depth_scaling"]) > 0.9, row
        assert np.isfinite(float(row["halves_floor"]))
    assert float(tb_pick(t, D.TB_UNSUB, scope="genome")["gap_closed_vs_depth_scaling"]) < 0.9
    cnt = tb_pick(t, D.COUNT_LAW, space="count", scope="genome")
    assert np.isfinite(float(cnt["gap_closed_vs_depth_scaling"]))
    assert all(r["share"] == "0.5" for r in sepe_tb[4]["pairs"])


def test_testbed_rows_files_concatenate(sepe_tb):
    """The two tools' testbed rows share one header; each tool keeps its covariate's rows."""
    tmp, rs, ct, tb, res = sepe_tb
    D.main(["rows", "--layer", "testbed", "--tables", tb, "--rs-store", rs, "--count-store", ct,
            "--totals", str(tmp / "totals.tsv"), "--out", str(tmp / "rd.tsv")])
    a = open(tmp / "rd.tsv").read().splitlines()
    b = open(tmp / "sepe" / "rows.tsv").read().splitlines()
    assert a[0] == b[0]
    with open(tmp / "both.tsv", "w") as f:
        f.write("\n".join(a + b[1:]) + "\n")
    stats = [str(tmp / "sepe" / "stats" / f"{r['target_id']}.npz") for r in res["rows"]]
    S.main(["fit", "--stats", *stats, "--rows", str(tmp / "both.tsv"), "--out", str(tmp / "f.tsv")])
    fit = D.read_tsv(tmp / "f.tsv")
    assert len(fit) == 7 and {f["covariate"] for f in fit} == {"run_type"}
    assert [f["a_loto"] for f in fit] == [f["a_loto"] for f in res["fit"]]
