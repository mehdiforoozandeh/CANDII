"""t130 blind check-set readings (tools/t130/): the extractor on a synthetic run tree, the analyzer
and the stdlib second method on the fabricated fixture tests/fixtures/t130/, the sbatch dry run.

The fixture (one narrow track C19M16; pids B = base, D15 = depth 15M, PE = pe; rungs A, D; g the
track (per_track) and `all` (across); models real / nocov / ids; seeds 0-2; both spaces):
  model trained crps_all = SP * (P[pair] + R[rung] + S[seed] + M[model] + G[g]); crps_top1 = 2x;
    split oracle_scaled 0.8x, scale_error 0.2x; val records 99 (must be ignored);
    P: D15 b2a 1.0, D15 a2b 1.2, PE b2a 1.4, PE a2b 1.6; R: A 0.3, D 0; S: 0, 0.02, 0.05;
    M: real 0, nocov 0.2, ids 0.25; G: per_track 0, all 0.01; SP: counts 1, pval 0.5.
  QuantileMatching crps_all: D15 b2a 1.5, D15 a2b 1.7, PE b2a 1.9, PE a2b 2.1 (x SP).
  floor_pairs crps_all = F[pid] + 0.1 log2(1/ratio) + 0.02 seed + 0.04 [pr2_to_pr1];
    F: B 0.30, D15 0.40, PE 0.35; reads_per_half = n_lines x ratio (n_lines B 15M, D15 7.5M,
    PE 15M). floor_curve = floor.py's per-product fit of those points (its real columns).
  floor_p15M crps_all: B 0.60 / 0.62 by direction (reads_per_half 15M).
The floor rule (plan_gaps.md A3): the same track's BASE curve read at the target's depth X.
  nested_oracle.tsv (PI ruling 2026-10-30, the depth-pair oracle): D15 counts, r_source manifest,
    base_to_arm binomial crps_all 0.9, arm_to_base eb_binomial 1.1 (top1 2x, split 0.85x / 0.15x);
    decoys that must NOT be read: empirical rows, poisson_remainder, an arm_to_arm row.
"""
import csv
import gzip
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures" / "t130"
T130 = ROOT / "tools" / "t130"
PY = sys.executable


def _load(name):
    spec = importlib.util.spec_from_file_location(f"t130_{name}", T130 / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EX = _load("extract_records")
T = "C19M16"
B, D15, PE = f"{T}__base__base", f"{T}__depth__15M", f"{T}__pe__pe"


# ---------------------------------------------------------------------------------------------
# analyzer + second method on the fixture
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def res(tmp_path_factory):
    d = tmp_path_factory.mktemp("t130res")
    for f in FIX.iterdir():
        shutil.copy(f, d / f.name)
    subprocess.run([PY, str(T130 / "analyze.py"), str(d)], check=True, capture_output=True)
    return d


@pytest.fixture(scope="module")
def table(res):
    return pd.read_csv(res / "blind_table.tsv", sep="\t")


def cell(t, **k):
    for c, v in k.items():
        t = t[t[c] == v]
    assert len(t) == 1, k
    return t.iloc[0]


HEAD = dict(space="counts", knob="depth", mark_class="narrow", rung="D", g_version="per_track")


def test_analyze_writes_every_file(res):
    for f in ("blind_table.tsv", "checks_readings.tsv", "BLIND.md", "analyze_out.txt"):
        assert (res / f).stat().st_size > 0, f


def test_gapclosed_by_hand(table):
    r = cell(table, **HEAD, direction="both", subset="all")
    d_model = (1.0 + 1.2) / 2 + (0.0 + 0.02 + 0.05) / 3        # 1.1233333
    d_onewarp = (1.5 + 1.7) / 2                                  # 1.6
    d_floor = (base_curve("crps_all", 15e6) + base_curve("crps_all", 30e6)) / 2   # X 15M, 30M
    d_oracle = (0.9 + 1.1) / 2                                   # the nested-read oracle
    assert r.d_model == pytest.approx(d_model, abs=1e-8)
    assert r.d_onewarp == pytest.approx(d_onewarp, abs=1e-8)
    assert r.d_floor == pytest.approx(d_floor, abs=1e-8)        # the floor survives as context
    assert r.d_oracle == pytest.approx(d_oracle, abs=1e-8)
    assert r.oracle_kind == "nested_binomial,nested_eb_binomial"
    assert r.gapclosed_onewarp == pytest.approx((d_onewarp - d_model) / (d_onewarp - d_oracle),
                                                abs=1e-8)   # 0.794444
    assert r.gapclosed_shuffle == pytest.approx((d_onewarp - (d_model + 0.15))
                                                / (d_onewarp - d_oracle), abs=1e-8)
    assert r.d_nocov == pytest.approx(d_model + 0.2, abs=1e-8)
    assert r.gain_over_nocov == pytest.approx(0.2, abs=1e-8)
    assert r.d_nosolution == pytest.approx(d_onewarp + 0.5, abs=1e-8)
    assert r.d_shuffle == pytest.approx(d_model + 0.15, abs=1e-8)
    assert r.n_pairs == 2 and r.n_runs == 3


CURVE = pd.read_csv(FIX / "floor_curve.tsv", sep="\t").set_index(["pid", "metric"])


def base_curve(metric, x):
    """The base curve read at reads_per_half x, by its line (not by the value_ columns)."""
    r = CURVE.loc[(B, metric)]
    return math.exp(r.alpha + r.beta * math.log2(x))


def test_floor_is_the_base_curve_at_the_target_depth(table):
    b2a = cell(table, **HEAD, direction="base_to_arm", subset="all")      # target D15, X = 15M
    a2b = cell(table, **HEAD, direction="arm_to_base", subset="all")      # target the base, 30M
    assert b2a.d_floor == pytest.approx(CURVE.loc[(B, "crps_all"), "value_15M"], abs=1e-8)
    assert b2a.d_floor == pytest.approx(base_curve("crps_all", 15e6), abs=1e-8)
    assert (b2a.floor_depth, b2a.floor_extrap, b2a.split_depth) == ("15M", 0, "15M")
    assert a2b.d_floor == pytest.approx(CURVE.loc[(B, "crps_all"), "value_30M_extrap"], abs=1e-8)
    assert (a2b.floor_depth, a2b.floor_extrap, a2b.split_depth) == ("30M", 1, "15M")
    assert cell(table, **HEAD, direction="both", subset="all").floor_extrap == 1
    # the split is the base's ladder split at ratio 1: mean crps_all 0.32 x 0.9
    assert a2b.floor_oracle_scaled_all == pytest.approx(0.9 * 0.32, abs=1e-8)
    # seed spread at r0.5 (X / n_lines >= 0.5): |0.42 - 0.44|
    assert b2a.floor_seed_spread == pytest.approx(0.02, abs=1e-8)
    top = cell(table, **HEAD, direction="arm_to_base", subset="top1")
    assert top.d_floor == pytest.approx(base_curve("crps_top1", 30e6), abs=1e-8)
    pe = cell(table, **{**HEAD, "knob": "pe"}, direction="base_to_arm", subset="all")
    assert pe.d_floor == pytest.approx(base_curve("crps_all", 30e6), abs=1e-8)   # pe target 30M
    assert math.isnan(b2a.p_floor_matched)


def test_pval_floor_is_the_base_own_depth_oracle(table):
    hp = {**HEAD, "space": "pval"}
    b2a = cell(table, **hp, direction="base_to_arm", subset="all")       # X = 15M = base half
    a2b = cell(table, **hp, direction="arm_to_base", subset="all")       # X = 30M: not matched
    for r in (b2a, a2b):
        assert r.d_floor == pytest.approx((0.60 + 0.62) / 2, abs=1e-8)
        assert r.floor_depth == "15M" and math.isnan(r.floor_seed_spread)
    assert (b2a.p_floor_matched, a2b.p_floor_matched) == (1, 0)
    assert cell(table, **hp, direction="both", subset="all").p_floor_matched == 0
    pe = cell(table, **{**hp, "knob": "pe"}, direction="base_to_arm", subset="all")
    assert pe.p_floor_matched == 0 and pe.floor_depth == "15M"


AN = _load("analyze")
DN = "C12M02__base__base"


def _dnase_tabs(with_alpha=True):
    alpha, beta = 1.0, -0.1
    row = {"pid": DN, "metric": "crps_all", "reads_per_half_full": 25_000_000, "beta": beta,
           "seed_spread_r0.5": 0.01, "seed_spread_r0.25": 0.02, "seed_spread_r0.125": 0.03,
           "value_30M_extrap": math.exp(alpha + beta * math.log2(30e6))}
    for k, x in AN.READ_AT.items():
        row[f"value_{k}"] = math.exp(alpha + beta * math.log2(x))
    if with_alpha:
        row["alpha"] = alpha
    split = {(DN, r): pd.Series({"crps_oracle_scaled_all": r, "scale_error_all": 0.0})
             for r in AN.RATIOS}
    return {(DN, "crps_all"): row}, split, {}, (alpha, beta)


@pytest.mark.parametrize("with_alpha", [True, False])
def test_dnase_base_floor_at_50M_and_the_spread_ratio(with_alpha):
    tabs = _dnase_tabs(with_alpha)
    alpha, beta = tabs[3]
    f = AN.pair_floor("counts", "C12M02", DN, "base", "base", tabs[:3])   # the 50M DNase base
    assert f["fl_all"] == pytest.approx(math.exp(alpha + beta * math.log2(50e6)), rel=1e-9)
    assert (f["floor_depth"], f["floor_extrap"], f["fl_sp_all"]) == ("50M", 1, 0.01)
    assert f["split_depth"] == "25M" and f["fl_os"] == 1.0
    cases = {"15M": (0.01, "12.5M", 0.5), "7.5M": (0.02, "6.25M", 0.25),
             "3.75M": (0.03, "3.125M", 0.125)}   # X / 25M = 0.6, 0.3, 0.15
    for lvl, (spread, sdepth, sratio) in cases.items():
        f = AN.pair_floor("counts", "C12M02", f"C12M02__depth__{lvl}", "depth", lvl, tabs[:3])
        x = float(lvl[:-1]) * 1e6
        assert f["fl_all"] == pytest.approx(math.exp(alpha + beta * math.log2(x)), rel=1e-9)
        assert (f["floor_depth"], f["floor_extrap"], f["fl_sp_all"]) == (lvl, 0, spread)
        assert (f["split_depth"], f["fl_os"]) == (sdepth, sratio)


def test_seed_abs_delta_is_max_pairwise(table):
    r = cell(table, **HEAD, direction="both", subset="all")
    assert r.seed_abs_delta == pytest.approx(0.05, abs=1e-8)
    t = cell(table, **HEAD, direction="both", subset="top1")
    assert t.seed_abs_delta == pytest.approx(0.10, abs=1e-8)
    assert bool(r.gain_gt_2seed) == ((r.d_onewarp - r.d_model) > 2 * 0.05)


def test_swap_and_depthlaw(table):
    r = cell(table, **HEAD, direction="both", subset="all")
    assert r.swap_ratio_median == pytest.approx(0.021, abs=1e-8)        # D15 swaps, seeds 0-2
    assert r.depthlaw_median_abs_log2err == pytest.approx(0.11, abs=1e-8)
    pe = cell(table, **{**HEAD, "knob": "pe"}, direction="both", subset="all")
    assert pe.swap_ratio_median == pytest.approx(0.031, abs=1e-8)
    assert math.isnan(pe.depthlaw_median_abs_log2err)


def test_no_nan_and_split_on_every_row(table):
    cols = ["d_floor", "seed_abs_delta", "crps_oracle_scaled_all", "scale_error_all", "d_model",
            "d_onewarp", "d_oracle", "oracle_crps_oracle_scaled_all", "oracle_scale_error_all"]
    assert not table[cols].isna().any().any()
    assert len(table) == 2 * 2 * 2 * 2 * 2 * 3 * 2     # space knob class(narrow,all) rung gv dir sub


def test_checks_readings_never_tick(res):
    c = pd.read_csv(res / "checks_readings.tsv", sep="\t", keep_default_na=False)
    assert len(c) and (c["threshold"] == "TODO(PI)").all()
    assert set(c["check"]) == {"gapclosed", "shuffledrop", "swapfidelity", "crps_split",
                               "composegap"}
    assert (c.loc[c["check"] == "composegap", "reading"] == "not_extracted").all()
    md = (res / "BLIND.md").read_text()
    assert "PI's pick" in md and "TODO(PI)" in md
    assert "neighbouring bins can beat it" in md
    assert "what a fresh replicate would differ by" in md and "NOT a subsample" in md
    g = c[(c["check"] == "gapclosed") & (c["knob"] == "depth") & (c["space"] == "counts")]
    assert g["detail"].str.contains("oracle nested_binomial,nested_eb_binomial").all()
    assert g["detail"].str.contains("what a fresh replicate would differ by").all()


def test_oracle_per_direction_and_its_split(table):
    b2a = cell(table, **HEAD, direction="base_to_arm", subset="all")
    a2b = cell(table, **HEAD, direction="arm_to_base", subset="all")
    assert (b2a.d_oracle, b2a.oracle_kind) == (pytest.approx(0.9), "nested_binomial")
    assert (a2b.d_oracle, a2b.oracle_kind) == (pytest.approx(1.1), "nested_eb_binomial")
    assert b2a.oracle_crps_oracle_scaled_all == pytest.approx(0.85 * 0.9)
    assert a2b.oracle_scale_error_all == pytest.approx(0.15 * 1.1)
    assert cell(table, **HEAD, direction="arm_to_base", subset="top1").d_oracle == pytest.approx(2.2)


@pytest.mark.parametrize("cellkey", [{**HEAD, "knob": "pe"}, {**HEAD, "space": "pval"}])
def test_pe_and_pval_keep_the_floor_as_oracle(table, cellkey):
    for d in ("base_to_arm", "arm_to_base", "both"):
        for s in ("all", "top1"):
            r = cell(table, **cellkey, direction=d, subset=s)
            assert r.oracle_kind == "floor"
            assert r.d_oracle == pytest.approx(r.d_floor, abs=1e-12)
            assert r.gapclosed_onewarp == pytest.approx(
                (r.d_onewarp - r.d_model) / (r.d_onewarp - r.d_floor), abs=1e-8)


def test_missing_nested_oracle_falls_back_to_the_floor_in_both_methods(tmp_path):
    for f in FIX.iterdir():
        if f.name != "nested_oracle.tsv":
            shutil.copy(f, tmp_path / f.name)
    subprocess.run([PY, str(T130 / "analyze.py"), str(tmp_path)], check=True, capture_output=True)
    t = pd.read_csv(tmp_path / "blind_table.tsv", sep="\t")
    for d in ("base_to_arm", "arm_to_base", "both"):
        for s in ("all", "top1"):
            r = cell(t, **HEAD, direction=d, subset=s)
            assert r.oracle_kind == "floor_fallback"
            assert r.d_oracle == pytest.approx(r.d_floor, abs=1e-12)
    assert "ABSENT" in (tmp_path / "BLIND.md").read_text()
    subprocess.run([PY, str(T130 / "verify_raw.py")] + [str(tmp_path / f) for f in (
        "records.tsv.gz", "baselines.tsv.gz", "floor_pairs.tsv", "floor_curve.tsv",
        "nested_oracle.tsv")], check=True, capture_output=True)
    subprocess.run([PY, str(T130 / "verify_compare.py"), str(tmp_path)], check=True,
                   capture_output=True)
    text = (tmp_path / "verify.txt").read_text()
    assert text.count("DISAGREE") == 0 and "AGREE d_oracle: 12 cells" in text


def test_missing_floor_propagates_as_na(tmp_path):
    for f in FIX.iterdir():
        shutil.copy(f, tmp_path / f.name)
    fc = pd.read_csv(tmp_path / "floor_curve.tsv", sep="\t")
    fc[fc["pid"] != D15].to_csv(tmp_path / "floor_curve.tsv", sep="\t", index=False)
    subprocess.run([PY, str(T130 / "analyze.py"), str(tmp_path)], check=True, capture_output=True)
    t = pd.read_csv(tmp_path / "blind_table.tsv", sep="\t")
    assert not math.isnan(cell(t, **HEAD, direction="base_to_arm", subset="all").d_floor)  # base only
    fc[fc["pid"] != B].to_csv(tmp_path / "floor_curve.tsv", sep="\t", index=False)
    subprocess.run([PY, str(T130 / "analyze.py"), str(tmp_path)], check=True, capture_output=True)
    t = pd.read_csv(tmp_path / "blind_table.tsv", sep="\t")
    for d in ("base_to_arm", "both"):
        assert math.isnan(cell(t, **HEAD, direction=d, subset="all").d_floor)


def test_second_method_agrees(res):
    subprocess.run([PY, str(T130 / "verify_raw.py"), str(res / "records.tsv.gz"),
                    str(res / "baselines.tsv.gz"), str(res / "floor_pairs.tsv"),
                    str(res / "floor_curve.tsv"), str(res / "nested_oracle.tsv")], check=True,
                   capture_output=True)
    subprocess.run([PY, str(T130 / "verify_compare.py"), str(res)], check=True,
                   capture_output=True)
    text = (res / "verify.txt").read_text()
    assert text.count("DISAGREE") == 0
    assert text.count("AGREE") == 7
    # d_oracle: rungs D, A x direction b2a, a2b, both x subset all, top1
    assert "AGREE d_oracle: 12 cells" in text
    v = json.loads((res / "verify_raw.json").read_text())
    assert v["d_oracle"]["counts|depth|all|D|per_track|arm_to_base|top1"] == pytest.approx(2.2)


def test_second_method_catches_a_change(res, tmp_path):
    for f in ("records.tsv.gz", "baselines.tsv.gz", "floor_pairs.tsv", "floor_curve.tsv",
              "nested_oracle.tsv", "blind_table.tsv"):
        shutil.copy(res / f, tmp_path / f)
    subprocess.run([PY, str(T130 / "verify_raw.py")] + [str(tmp_path / f) for f in (
        "records.tsv.gz", "baselines.tsv.gz", "floor_pairs.tsv", "floor_curve.tsv",
        "nested_oracle.tsv")],
        check=True, capture_output=True)
    v = json.loads((tmp_path / "verify_raw.json").read_text())
    k = next(iter(v["d_model"]))
    v["d_model"][k] += 1e-3
    (tmp_path / "verify_raw.json").write_text(json.dumps(v))
    subprocess.run([PY, str(T130 / "verify_compare.py"), str(tmp_path)], check=True,
                   capture_output=True)
    assert "DISAGREE d_model" in (tmp_path / "verify.txt").read_text()


# ---------------------------------------------------------------------------------------------
# the extractor on a synthetic run tree
# ---------------------------------------------------------------------------------------------


def _pair(src, tgt, d):
    arm = lambda p: p.split("__")[1]
    lvl = lambda p: p.split("__")[2]
    return {"track": T, "cell": "C19", "assay": "H3K27ac", "mark_class": "narrow",
            "arm_src": arm(src), "level_src": lvl(src), "arm_tgt": arm(tgt), "level_tgt": lvl(tgt),
            "source_pid": src, "target_pid": tgt, "direction": d,
            "knob_combo": f"{arm(src)}:{lvl(src)}->{arm(tgt)}:{lvl(tgt)}",
            "counts_identical": False, "pval_identical": False}


def _metrics(v, split=False):
    out = {"n_all": 100, "n_nonzero": 50, "n_top1": 1, "crps_all": v, "crps_nonzero": v + 1,
           "crps_top1": v + 2, "spearman_all": 0.5, "spearman_nonzero": 0.4,
           "spearman_top1": 0.3, "n_crps_nonfinite": 0}
    if split:
        out.update(crps_oracle_scaled_all=0.8 * v, scale_error_all=0.2 * v, c_star_all=0.25)
    return out


def _write_run(runs, name, g):
    rung, _, space, model, s = name.split("_")
    CTL = f"{T}__ctldepth__q0.5"
    recs = [
        {"kind": "trained", "eval": "score", **_pair(B, D15, "base_to_arm"), **_metrics(1.0, True)},
        {"kind": "trained", "eval": "val", **_pair(B, D15, "base_to_arm"), **_metrics(9.0)},
        {"kind": "trained", "eval": "score", **_pair(PE, B, "arm_to_base"), **_metrics(1.5, True)},
        {"kind": "trained", "eval": "score", **_pair(B, CTL, "base_to_arm"), **_metrics(2.0, True)},
        {"kind": "shuffle", "eval": "score", **_pair(B, D15, "base_to_arm"), **_metrics(1.2)},
        {"kind": "swap", "eval": "score", **_pair(D15, D15, "swap"), **_metrics(0.5),
         "swap_median_abs_log_ratio": 0.07},
        {"kind": "swap", "eval": "score", **_pair(B, B, "swap"), **_metrics(0.5),
         "swap_median_abs_log_ratio": 0.03},
        {"kind": "swap", "eval": "score", **_pair(CTL, CTL, "swap"), **_metrics(0.5),
         "swap_median_abs_log_ratio": 0.09},
        {"kind": "depthlaw", "eval": "score", **_pair(B, D15, "base_to_arm"), **_metrics(1.0),
         "depth_source": "trained", "depth_log2_ratio_true": -1.0,
         "depth_log2_scale_pred": -0.9},
        {"kind": "law", "eval": "score", **_pair(D15, PE, "arm_to_arm"), **_metrics(3.0)},
    ]
    d = runs / name
    d.mkdir(parents=True)
    run = {"run_name": name, "rung": rung, "g": g, "g_version": "across" if g == "all" else
           "per_track", "space": space, "model": model, "seed": int(s[1:])}
    (d / "scores.json").write_text(json.dumps({"run": run, "records": recs}))


def _write_baseline(rungs, arm_pid, d):
    arm, level = arm_pid.split("__")[1:]
    src, tgt = (B, arm_pid) if d == "base_to_arm" else (arm_pid, B)
    pair = {"index": 0, "track": T, "cell": "C19", "assay": "H3K27ac", "mark_class": "narrow",
            "arm": arm, "level": level, "knob": arm, "arm_pid": arm_pid, "direction": d,
            "source_pid": src, "target_pid": tgt}
    recs = []
    for space in ("counts", "pval"):
        for rung, v in (("noSolution", 2.0), ("QuantileMatching", 1.6)):
            for ev in ("score", "val"):
                recs.append({"space": space, "rung": rung, "eval": ev,
                             "point_crps_all": 99.0, "spread_crps_all": v,
                             "spread_crps_top1": v + 1, "crps_oracle_scaled_all": 0.9 * v,
                             "scale_error_all": 0.1 * v, "spearman_all": 0.6})
    (rungs / f"{arm_pid}__{d}.json").write_text(json.dumps({"pair": pair, "records": recs}))


def _read(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def test_extract_records(tmp_path):
    runs, rungs, out = tmp_path / "runs", tmp_path / "rungs_v2", tmp_path / "x" / "t130"
    _write_run(runs, f"D_{T}_counts_real_s0", T)
    _write_run(runs, "A_all_pval_nocov_s2", "all")
    (runs / "A2_all_counts_real_s0").mkdir()                     # row 2: skipped
    (runs / f"B_{T}_counts_ids_s1").mkdir()                       # no scores.json yet
    (runs / f"C_{T}_counts_real_s1").mkdir()                      # a truncated scores.json
    (runs / f"C_{T}_counts_real_s1" / "scores.json").write_text('{"run": {"rung": "C"}, "reco')
    rungs.mkdir()
    for pid in (D15, PE, f"{T}__ctldepth__q0.5"):
        for d in ("base_to_arm", "arm_to_base"):
            _write_baseline(rungs, pid, d)
    (rungs / "oracle").mkdir()
    (rungs / f"{D15}.json").write_text("{}")                      # an oracle file: not a pair
    r = subprocess.run([PY, str(T130 / "extract_records.py"), str(runs), str(rungs), str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (out / "EXTRACT_DONE").exists()
    with gzip.open(out / "records.tsv.gz", "rt") as fh:
        assert fh.readline().rstrip("\n").split("\t") == EX.RCOLS
    rec = _read(out / "records.tsv.gz")
    assert len(rec) == 2 * 7          # per run: 2 trained score + 1 val, shuffle, 2 swaps, depthlaw
    one = [x for x in rec if x["run"] == f"D_{T}_counts_real_s0"]
    kinds = sorted((x["kind"], x["eval"], x["pid"]) for x in one)
    assert kinds == sorted([("trained", "score", D15), ("trained", "val", D15),
                            ("trained", "score", PE), ("shuffle", "score", D15),
                            ("swap", "score", D15), ("swap", "score", B),
                            ("depthlaw", "score", D15)])
    assert {x["arm"] for x in one} == {"depth", "pe", "base"}
    assert not any("ctldepth" in x["pid"] for x in rec)
    assert not any(x["kind"] == "law" for x in rec)
    tr = next(x for x in one if x["kind"] == "trained" and x["eval"] == "score"
              and x["pid"] == PE)
    assert tr["direction"] == "arm_to_base" and tr["arm"] == "pe" and tr["level"] == "pe"
    assert tr["target_pid"] == B and float(tr["crps_oracle_scaled_all"]) == pytest.approx(1.2)
    assert tr["g_version"] == "per_track" and tr["seed"] == "0" and tr["model"] == "real"
    sw = next(x for x in one if x["kind"] == "swap" and x["pid"] == D15)
    assert float(sw["swap_ratio_median"]) == pytest.approx(0.07)
    dl = next(x for x in one if x["kind"] == "depthlaw")
    assert float(dl["log2_depth_ratio"]) == -1.0 and float(dl["log2_pred_scale"]) == -0.9
    assert next(x for x in rec if x["run"].startswith("A_all"))["g_version"] == "across"
    base = _read(out / "baselines.tsv.gz")
    assert len(base) == 2 * 2 * 2 * 2        # pids (D15, PE) x dirs x spaces x rungs, score only
    q = next(x for x in base if x["rung"] == "QuantileMatching" and x["pid"] == D15)
    assert float(q["crps_all"]) == pytest.approx(1.6) and float(q["crps_top1"]) == pytest.approx(2.6)
    assert q["level"] == "15M" and q["arm"] == "depth"
    runs_tsv = _read(out / "runs.tsv")
    assert [x["run"] for x in runs_tsv] == sorted(["A_all_pval_nocov_s2", f"B_{T}_counts_ids_s1",
                                                   f"C_{T}_counts_real_s1", f"D_{T}_counts_real_s0"])
    has = {x["run"]: (x["has_scores"], x["n_records"]) for x in runs_tsv}
    assert has[f"B_{T}_counts_ids_s1"] == ("0", "0")
    assert has[f"C_{T}_counts_real_s1"] == ("0", "0")
    assert has[f"D_{T}_counts_real_s0"] == ("1", "7")


# ---------------------------------------------------------------------------------------------
# the sbatch script
# ---------------------------------------------------------------------------------------------


def test_extract_sh_dry_run_and_guard():
    env = {**os.environ, "DRY_RUN": "1"}
    sh = str(ROOT / "slurm" / "t130" / "extract.sh")
    r = subprocess.run(["bash", sh, "/k", "/r", "/q", "/x/t130"], env=env, capture_output=True,
                       text=True)
    assert r.returncode == 0
    assert r.stdout.strip() == "python3 /k/tools/t130/extract_records.py /r /q /x/t130"
    bad = subprocess.run(["bash", sh, "/k", "/r", "/q", "/x/t118"], env=env, capture_output=True,
                         text=True)
    assert bad.returncode != 0 and bad.stdout == ""
