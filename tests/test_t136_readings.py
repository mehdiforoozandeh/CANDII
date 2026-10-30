"""t136 readings (tools/t136/readings/, h28): extraction, the tables, the readings of c1-c5 and the
rule, and the stdlib second method, on a fabricated grid + reference tree built in tmp_path.

    export PYTHONPATH=$PWD/src; /Users/mforooz/miniforge3/envs/candii/bin/python -m pytest \
        tests/test_t136_readings.py -q
"""
import gzip
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas", reason="the t136 readings need pandas (candii env)")

ROOT = Path(__file__).resolve().parents[1]
RD = ROOT / "tools" / "t136" / "readings"
REAL = Path("/Users/mforooz/Desktop/research/libbrechteam@sfu/CANDII/.claude/worktrees/"
            "counterfactual-f/cruxvault/results/t136")
TRACKS = ("C07M20", "C07M29", "C12M02", "C19M16", "C19M22", "C40M17", "C40M18")
READS = {"base": 30.0, "15M": 15.0, "7.5M": 7.5, "3.75M": 3.75}
PRODUCTS = [("base", "base"), ("abproxy", "f0.5"), ("depth", "15M"), ("depth", "7.5M"),
            ("depth", "3.75M")]
# depth-law |log2 error| per arm x group (seed s adds 0.001 x s)
ERR = {"Dref": {"trained_base": 0.05, "heldout_base": 0.1, "heldout_r2": 0.20, "trained_r4": 0.30},
       "Dviabase": {"heldout_r2": 0.10, "trained_r4": 0.11},
       "Darm1": {"trained_base": 0.052, "heldout_base": 0.1, "heldout_r2": 0.12, "trained_r4": 0.118},
       "Darm2": {"trained_base": 0.05, "heldout_base": 0.1, "heldout_r2": 0.11, "trained_r4": 0.10}}
HELD_CRPS = {"Dref": 0.6, "Dviabase": 0.4, "Darm1": 0.45, "Darm2": 0.44}
TRAINED_CRPS = {"Dref": 0.445, "Darm1": 0.446, "Darm2": 0.445}


def _load(name):
    spec = importlib.util.spec_from_file_location(f"t136r_{name}", RD / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def pid(tr, a, lv):
    return f"{tr}__{a}__{lv}"


def pair(tr, s, t):
    d = "base_to_arm" if s[0] == "base" else ("arm_to_base" if t[0] == "base" else "arm_to_arm")
    return {"track": tr, "mark_class": "dnase" if tr == "C12M02" else "narrow", "arm_src": s[0],
            "level_src": s[1], "arm_tgt": t[0], "level_tgt": t[1], "direction": d,
            "source_pid": pid(tr, *s), "target_pid": pid(tr, *t),
            "knob_combo": f"{s[0]}:{s[1]}->{t[0]}:{t[1]}"}


def crps(v, seed):
    c = v + 0.001 * seed
    return {"crps_all": c, "crps_oracle_scaled_all": 0.9 * c, "scale_error_all": 0.1 * c}


def depthrec(p, src, arm, grp, seed):
    lt = math.log2(READS[p["level_tgt"]] / READS[p["level_src"]])
    return {**p, "kind": "depthlaw", "eval": "score", "depth_source": src, **crps(0.3, seed),
            "depth_log2_ratio_true": lt,
            "depth_log2_scale_pred": lt + ERR[arm][grp] + 0.001 * seed}


def is75(p):
    return "depth:7.5M" in p["knob_combo"]


def make_run(d, arm, tr, seed, marks=("TRAIN_DONE", "SCORE_DONE", "LAW_DONE"), leak=False,
             files=("scores.json", "law.json"), best=2000):
    d.mkdir(parents=True)
    trained = [pair(tr, ("base", "base"), p) for p in PRODUCTS[1:] if p != ("depth", "7.5M")]
    trained += [pair(tr, p, ("base", "base")) for p in PRODUCTS[1:] if p != ("depth", "7.5M")]
    law = [pair(tr, s, t) for s in PRODUCTS[1:] for t in PRODUCTS[1:] if s != t]
    r4 = {"depth:15M->depth:3.75M", "depth:3.75M->depth:15M"}
    if arm == "Darm1":
        extra = [p for p in law if p["knob_combo"] in r4]
    elif arm == "Darm2":
        extra = [p for p in law if not is75(p)]
        n = 30 if tr == "C12M02" else 306
        extra += [{"knob_combo": f"pad:{i}->pad:x", "source_pid": f"pad{i}", "target_pid": "padx"}
                  for i in range(n - len(extra))]
    else:
        extra = []
    tp = trained + extra
    if leak:
        tp = tp + [pair(tr, ("base", "base"), ("depth", "7.5M"))]
    cfg = {"run_name": d.name, "arm": arm if arm.startswith("Darm") else None, "g": tr,
           "seed": seed, "max_steps": 12000, "train_pairs": tp, "extra_pairs": extra,
           "holdout": [["depth", "7.5M"]], "git_sha": "abc1234"}
    if arm != "Dviabase":
        (d / "config.json").write_text(json.dumps(cfg))
    run = {"run_name": d.name, "g": tr, "g_version": "per_track", "space": "counts",
           "model": "real", "seed": seed, "steps": 3000, "best_step": best}
    ek = {p["knob_combo"] for p in extra}
    sc, lw = [], []
    if arm != "Dviabase":
        for p in trained:
            sc.append({**p, "kind": "trained", "eval": "score", **crps(TRAINED_CRPS[arm], seed)})
            sc.append({**p, "kind": "trained", "eval": "val", **crps(9.0, seed)})
            if p["arm_src"] == "depth" or p["arm_tgt"] == "depth":
                sc.append(depthrec(p, "trained", arm, "trained_base", seed))
        for p in (pair(tr, ("base", "base"), ("depth", "7.5M")),
                  pair(tr, ("depth", "7.5M"), ("base", "base"))):
            sc.append({**p, "kind": "heldout", "eval": "score", **crps(0.41, seed)})
            sc.append(depthrec(p, "heldout", arm, "heldout_base", seed))
    for p in law:
        v = HELD_CRPS[arm] if is75(p) else (0.2 if p["knob_combo"] in ek else 0.5)
        lw.append({**p, "kind": "law", "eval": "score", **crps(v, seed)})
        if p["arm_src"] == "depth" and p["arm_tgt"] == "depth":
            lw.append(depthrec(p, "law", arm, "heldout_r2" if is75(p) else "trained_r4", seed))
    for name, recs in (("scores.json", sc), ("law.json", lw)):
        if name in files:
            (d / name).write_text(json.dumps({"run": run, "records": recs}))
    (d / "timing.json").write_text(json.dumps({"steps_run": 3000, "best_step": best,
                                               "val_nll": 0.7, "train": 100.0}))
    if "TRAIN_DONE" in marks:
        (d / "TRAIN_DONE").write_text(f"{d.name} best_step {best} val_nll 0.7\n")
    for m in marks:
        if m != "TRAIN_DONE":
            (d / m).write_text("")


def fabricate(tmp, d2_score=4, d2_law=3, leak=False, best_darm1=2000):
    grid, ref = tmp / "nibi_t136", tmp / "fir_t128p2"
    rows = ["index\trun_name"]
    for ai, arm in enumerate(("Darm1", "Darm2")):
        for gi, tr in enumerate(TRACKS):
            for s in range(3):
                name = f"{arm}_{tr}_counts_real_s{s}"
                rows.append(f"{ai * 21 + gi * 3 + s}\t{name}")
                if arm == "Darm2" and gi >= d2_score:
                    continue
                marks = ("TRAIN_DONE", "SCORE_DONE") + (("LAW_DONE",) if arm == "Darm1" or gi < d2_law else ())
                make_run(grid / "runs" / name, arm, tr, s, marks=marks,
                         leak=leak and arm == "Darm1" and gi == 0,
                         best=best_darm1 if arm == "Darm1" else 2500)
    (grid / "tasks.tsv").write_text("\n".join(rows) + "\n")
    for tr in TRACKS:
        for s in range(3):
            name = f"D_{tr}_counts_real_s{s}"
            make_run(ref / "runs" / name, "Dref", tr, s)
            make_run(ref / "viabase" / name, "Dviabase", tr, s, marks=("LAW_DONE",),
                     files=("law.json",))
    (ref / "runs" / "D_all_counts_real_s0").mkdir()        # the across-track g: never read
    return grid, ref


def pipeline(tmp, **kw):
    grid, ref = fabricate(tmp, **kw)
    out = tmp / "res"
    out.mkdir()
    py = sys.executable
    for args in (["extract_records.py", "grid", str(out), str(grid)],
                 ["extract_records.py", "ref", str(out), str(ref)],
                 ["verify_raw.py", "collect", str(out / "verify_collect_nibi.json"), str(grid), "grid"],
                 ["verify_raw.py", "collect", str(out / "verify_collect_fir.json"), str(ref), "ref"]):
        subprocess.run([py, str(RD / args[0])] + args[1:], check=True, capture_output=True)
    _load("analyze").main(out)
    subprocess.run([py, str(RD / "verify_raw.py"), "reduce", str(out / "verify_raw.json"),
                    str(out / "verify_collect_nibi.json"), str(out / "verify_collect_fir.json")],
                   check=True, capture_output=True)
    _load("verify_compare").main(out)
    ck = pd.read_csv(out / "checks_readings.tsv", sep="\t")
    return out, ck


def reading(ck, check, arm, item):
    s = ck[(ck.check == check) & (ck.arm == arm) & (ck["item"] == item)]
    assert len(s) == 1, (check, arm, item)
    return s.iloc[0]


def test_incomplete_darm2(tmp_path):
    out, ck = pipeline(tmp_path)
    v = (out / "verify.txt").read_text()
    assert "DISAGREE" not in v and v.count("AGREE") >= 12
    for f in ("m5_depthlaw_table.tsv", "m1_heldout_table.tsv", "m3_trained_table.tsv",
              "n_table.tsv", "READINGS.md", "DIGEST.md"):
        assert (out / f).is_file(), f
    # extraction: is_extra marks exactly Darm1's 15M <-> 3.75M law pairs
    with gzip.open(out / "records_nibi.tsv.gz", "rt") as fh:
        r = pd.read_csv(fh, sep="\t")
    d1 = r[(r.arm == "Darm1") & (r.kind == "law")]
    assert set(d1[d1.is_extra == 1].level_src) == {"15M", "3.75M"}
    assert (d1.is_extra == 1).sum() == 2 * 21
    # c1: (0.201 - 0.121) / (0.201 - 0.101) = 0.8 on pooled medians
    c1 = reading(ck, "c1", "Darm1", "gap closed on heldout_r2")
    assert c1.reading == "PASS" and abs(c1.value - 0.8) < 1e-9
    assert reading(ck, "c3", "Darm1", "heldout_r2 vs trained_r4 + 2 x spread").reading == "PASS"
    c4 = ck[(ck.check == "c4")].set_index("item")
    assert abs(c4.loc["crps_all gap closed", "value"] - 0.75) < 1e-9
    assert c4.loc["check reading", "reading"] == "PASS"
    assert reading(ck, "c2", "Darm1", "arm reading").reading == "PASS"
    assert reading(ck, "c2", "Darm2", "arm reading").reading == "NOT-READABLE"
    f = reading(ck, "c5", "Darm2", "runs finished")
    assert f.reading == "NOT-READABLE" and f.value == 9
    assert reading(ck, "c5", "Darm2", "extra_pairs count per track").reading == "PASS"
    assert reading(ck, "c5", "Darm1", "arm reading").reading == "PASS"
    rule = reading(ck, "rule", "Darm1", "rule reading")
    assert rule.reading == "NOT-READABLE (controls not yet readable)"
    assert "-> all = PASS" in rule.detail
    m5 = pd.read_csv(out / "m5_depthlaw_table.tsv", sep="\t")
    p = m5[(m5.scope == "darm2_tracks") & (m5.arm == "Darm2") & (m5.group == "heldout_r2")
           & (m5["class"] == "all")].iloc[0]
    assert p.complete and p.n_runs == 9 and p.tracks == "C07M20,C07M29,C12M02"
    t5 = m5[(m5.scope == "all7") & (m5["class"] == "all")].set_index(["arm", "group"])
    assert t5.loc[("Dref", "trained_r4"), "status"] == "never-trained"
    assert t5.loc[("Darm1", "trained_r4"), "status"] == "trained"
    assert not t5.loc[("Darm2", "heldout_r2"), "complete"]
    assert t5.loc[("Darm2", "heldout_r2"), "label_vs_Dref"] == "incomplete"


def test_complete_grid_rule_reads(tmp_path):
    out, ck = pipeline(tmp_path, d2_score=7, d2_law=7)
    assert "DISAGREE" not in (out / "verify.txt").read_text()
    assert reading(ck, "c5", "both", "check reading").reading == "PASS"
    assert reading(ck, "c2", "both", "check reading").reading == "PASS"
    assert reading(ck, "rule", "Darm1", "rule reading").reading == "PASS"


def test_leak_and_late_best_step_fail_c5(tmp_path):
    out, ck = pipeline(tmp_path, d2_score=7, d2_law=7, leak=True, best_darm1=11000)
    assert "DISAGREE" not in (out / "verify.txt").read_text()
    assert reading(ck, "c5", "Darm1", "runs with a 7.5M product in train_pairs").reading == "FAIL"
    b = reading(ck, "c5", "Darm1", "runs with best step > 10800")
    assert b.reading == "FAIL" and b.value == 21
    assert reading(ck, "rule", "Darm1", "rule reading").reading == "INVALID (a control fails)"


@pytest.mark.skipif(not (REAL / "m5_depthlaw_table.tsv").is_file(), reason="no real readings here")
def test_real_references_match_phase2():
    """The recomputed Dref / Dviabase cells equal plan section 5's phase-2 reference values."""
    m5 = pd.read_csv(REAL / "m5_depthlaw_table.tsv", sep="\t")
    t = m5[(m5.scope == "all7") & (m5["class"] == "all")].set_index(["arm", "group"])
    assert round(t.loc[("Dref", "heldout_r2"), "median_abs_err"], 3) == 0.195
    assert round(t.loc[("Dviabase", "heldout_r2"), "median_abs_err"], 3) == 0.125
    assert round(t.loc[("Dref", "trained_base"), "median_abs_err"], 3) == 0.052
    m3 = pd.read_csv(REAL / "m3_trained_table.tsv", sep="\t")
    r = m3[(m3.scope == "all7") & (m3["class"] == "all") & (m3.arm == "Dref")
           & (m3.group == "trained_base")].iloc[0]
    assert round(r.crps_all, 3) == 0.445
    assert "DISAGREE" not in (REAL / "verify.txt").read_text()
