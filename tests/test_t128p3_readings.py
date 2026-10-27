"""t128 phase 3 readings (tools/t128p3/readings/): extraction, the metric tables, the checks and
the second method, on a fabricated tree and on the real phase-2 mirror (the reference).

Needs pandas and pytest (the candii env has both; candi-local has pandas but no pytest):
    PYTHONPATH=$PWD/src /Users/mforooz/miniforge3/envs/candii/bin/python -m pytest \
        tests/test_t128p3_readings.py -q
Regenerate the fixture (tests/fixtures/t128p3_readings/) from the fabricated tree:
    /Users/mforooz/miniforge3/envs/candii/bin/python tests/test_t128p3_readings.py
"""
import importlib.util
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas", reason="the t128p3 readings need pandas (candii or candi-local env)")

ROOT = Path(__file__).resolve().parents[1]
RD = ROOT / "tools" / "t128p3" / "readings"
P2 = ROOT / "tools" / "t128p2" / "readings"
FIX = ROOT / "tests" / "fixtures" / "t128p3_readings"
MIRROR = Path("/Users/mforooz/Desktop/research/libbrechteam@sfu/CANDII/cruxvault/results/t128p2")
PHASE3_IN = ("records_fir.tsv.gz", "viabase_fir.tsv.gz", "runs_fir.tsv")
REF_IN = ("records_fir.tsv.gz", "viabase_fir.tsv.gz", "runs_fir.tsv", "m1_never_table.tsv",
          "m9_steps_table.tsv")
TABLES = ("n_table.tsv", "m1_never_table.tsv", "m2_gap_table.tsv", "m3_trained_table.tsv",
          "m4_heldout_table.tsv", "m5_depthlaw_table.tsv", "m9_steps_table.tsv",
          "step_confounded_runs.tsv", "rank_table.tsv", "checks.txt", "pair_table.tsv.gz")


def _analyze():
    spec = importlib.util.spec_from_file_location("t128p3_analyze", RD / "analyze.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------------------- fabricated raw tree

TRACKS = {"C07M20": "narrow", "C19M16": "broad"}
GS = ("C07M20", "C19M16", "all")
TRAINED = [("base_to_arm", "base", "base", "abproxy", "f0.5"),
           ("arm_to_base", "abproxy", "f0.5", "base", "base"),
           ("base_to_arm", "base", "base", "depth", "15M"),
           ("arm_to_base", "depth", "15M", "base", "base")]
HELDOUT = [("base_to_arm", "base", "base", "depth", "7.5M"),
           ("arm_to_base", "depth", "7.5M", "base", "base")]
LAW = [("abproxy", "f0.5", "abproxy", "f0.9"), ("abproxy", "f0.9", "abproxy", "f0.5"),
       ("abproxy", "f0.5", "depth", "15M"), ("depth", "15M", "abproxy", "f0.9"),
       ("abproxy", "f0.9", "depth", "7.5M")]
DEPTHLAW = [("trained", "base", "base", "depth", "15M"), ("heldout", "base", "base", "depth", "7.5M"),
            ("law", "depth", "15M", "depth", "7.5M"), ("law", "abproxy", "f0.5", "depth", "15M")]
OFFSET = {"Dbigg": -0.03, "Dbigf": 0.01, "D": 0.0, "DSfree": 0.05}


def _crps(rng, arm, via):
    c = 0.55 + OFFSET[arm] - (0.1 if via else 0.0) + 0.05 * rng.random()
    os_ = c * (0.9 + 0.05 * rng.random())
    return {"crps_all": c, "crps_oracle_scaled_all": os_, "scale_error_all": c - os_,
            "crps_top1": 3 * c, "spearman_all": 0.3, "c_star_all": 0.02, "n_crps_nonfinite": 0}


def _records(arm, g, seed, files, via=False):
    tracks = list(TRACKS) if g == "all" else [g]
    out = {"scores.json": [], "law.json": []}
    for tr in tracks:
        def rng(*k):
            return random.Random("|".join(map(str, (arm, g, seed, tr, via) + k)))
        base = {"eval": "score", "track": tr, "mark_class": TRACKS[tr], "counts_identical": 0}
        if not via:
            for kind, rows in (("trained", TRAINED), ("heldout", HELDOUT)):
                for dr, a1, l1, a2, l2 in rows:
                    out["scores.json"].append({**base, "kind": kind, "direction": dr, "arm_src": a1,
                                               "level_src": l1, "arm_tgt": a2, "level_tgt": l2,
                                               **_crps(rng(kind, a1, l1, a2, l2), arm, via)})
            out["scores.json"].append({**base, "kind": "swap", "direction": "arm_to_arm",
                                       "arm_src": "abproxy", "level_src": "f0.5",
                                       "arm_tgt": "abproxy", "level_tgt": "f0.9",
                                       "swap_median_abs_log_ratio": 0.1 * rng("swap").random()})
            out["scores.json"].append({**base, "kind": "trained", "eval": "val", "direction": "x",
                                       "arm_src": "x", "level_src": "x", "arm_tgt": "x",
                                       "level_tgt": "x", "crps_all": 99.0})   # not eval == score
        for a1, l1, a2, l2 in LAW:
            out["law.json"].append({**base, "kind": "law", "direction": "arm_to_arm",
                                    "arm_src": a1, "level_src": l1, "arm_tgt": a2, "level_tgt": l2,
                                    **_crps(rng("law", a1, l1, a2, l2), arm, via)})
        for i, (src, a1, l1, a2, l2) in enumerate(DEPTHLAW):
            r = rng("depth", i)
            true = r.uniform(-1, 1)
            out["law.json"].append({**base, "kind": "depthlaw", "direction": "arm_to_arm",
                                    "depth_source": src, "arm_src": a1, "level_src": l1,
                                    "arm_tgt": a2, "level_tgt": l2,
                                    "depth_log2_ratio_true": true,
                                    "depth_log2_scale_pred": None if (i == 3 and seed == 2)
                                    else true + r.uniform(-0.3, 0.3)})
    return {f: out[f] for f in files}


def _steps(arm, g, seed, max_steps):
    if arm == "Dbigf" and g == "C07M20" and seed == 1:
        return 12500, 13500                 # step-confounded, not at cap
    if arm == "Dbigg" and g == "all" and seed == 2:
        return 13750, 15000                 # best in the last 10 %, at cap
    best = 2500 + 250 * seed + (500 if g == "all" else 0)
    return best, min(best + 1000, max_steps)


def _write(p, obj):
    p.write_text(json.dumps(obj))


def make_tree(root, arms=("Dbigg", "Dbigf"), max_steps=15000, phase2=False):
    """A fabricated Fir tree: runs/<name>/ (config, timing, TRAIN_DONE, train_log, marks, scores,
    law), viabase/<name>/ (viabase.json, law.json, LAW_DONE), logs/, tasks.tsv."""
    root = Path(root)
    sizes = {"Dbigg": (256, 32, 256), "Dbigf": (64, 64, 512), "D": (64, 32, 256),
             "DSfree": (64, 32, 256)}
    tasks = ["index\trun_name"]
    ix = 0
    for arm in arms:
        for g in GS:
            for seed in (0, 1, 2):
                name = f"{arm}_{g}_counts_real_s{seed}"
                tasks.append(f"{ix}\t{name}")
                d = root / "runs" / name
                d.mkdir(parents=True)
                gh, fc, nt = sizes[arm]
                cfg = {"run_name": name, "rung": "DS" if arm == "DSfree" else "D", "g": g,
                       "space": "counts", "model": "real", "seed": seed, "max_steps": max_steps,
                       "device": "cuda", "val_nll_rule": "t118 cross NLL", "n_theta": nt,
                       "train_pairs": [], "loss": "nb"}
                if not phase2:
                    cfg.update(arm=arm, g_hidden=gh, f_channels=fc)
                _write(d / "config.json", cfg)
                best, steps = _steps(arm, g, seed, max_steps)
                train_s = 0.9 * steps * (1.5 if g == "all" else 1.0)
                _write(d / "timing.json", {"steps_run": steps, "best_step": best, "val_nll": 1.0,
                                           "train": train_s})
                (d / "train_log.tsv").write_text(
                    "step\tloss\tval_nll\tseconds\n"
                    + "".join(f"{s}\t1.0\t1.0\t{s}\n" for s in range(250, steps + 1, 250)))
                (d / "TRAIN_DONE").write_text(f"{name} best_step {best} val_nll 1.0\n")
                run = {"run_name": name, "rung": cfg["rung"], "g": g,
                       "g_version": "across" if g == "all" else "per_track", "space": "counts",
                       "model": "real", "seed": seed, "steps": steps, "best_step": best,
                       "val_nll": 1.0}
                recs = _records(arm, g, seed, ("scores.json", "law.json"))
                for f, rs in recs.items():
                    _write(d / f, {"run": run, "records": rs})
                for m in ("SCORE_DONE", "LAW_DONE"):
                    (d / m).write_text("ok\n")
                logs = root / "logs"
                logs.mkdir(exist_ok=True)
                tag = "t128p2" if phase2 else "t128p3"
                (logs / f"{tag}_train_100_{ix}.out").write_text(
                    f"=== {tag} train idx={ix} job 100_{ix} host=fc{1000 + ix} now\n")
                if arm != "DSfree":
                    v = root / "viabase" / name
                    v.mkdir(parents=True)
                    _write(v / "viabase.json", {"old_run_dir": str(d), "intermediate": "nb_mean"})
                    _write(v / "law.json", {"run": run, "records": _records(arm, g, seed,
                                                                         ("law.json",), via=True)
                                            ["law.json"]})
                    (v / "LAW_DONE").write_text("ok\n")
                ix += 1
    (root / "tasks.tsv").write_text("\n".join(tasks) + "\n")
    return root


def _extract(script, out, root):
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(script), str(out), str(root)], check=True,
                   capture_output=True, text=True)


def build_fixture(dest, work):
    """Phase-3 inputs from this chunk's extract; ref_* in phase 2's layout from phase 2's extract
    (arm D + one DSfree run to be dropped); ref_m1 / ref_m9 = the re-pooled reference rows."""
    dest, work = Path(dest), Path(work)
    dest.mkdir(parents=True, exist_ok=True)
    t3 = make_tree(work / "p3")
    _extract(RD / "extract_records.py", work / "o3", t3)
    t2 = make_tree(work / "p2", arms=("D", "DSfree"), max_steps=12000, phase2=True)
    _extract(P2 / "extract_records.py", work / "o2", t2)
    for f in PHASE3_IN:
        shutil.copy(work / "o3" / f, dest / f)
        shutil.copy(work / "o2" / f, dest / f"ref_{f}")
    _analyze().main(dest)
    m1 = pd.read_csv(dest / "m1_never_table.tsv", sep="\t")
    m1[m1.arm.isin(["D", "Dviabase"])].iloc[:, :17].to_csv(dest / "ref_m1_never_table.tsv",
                                                           sep="\t", index=False)
    m9 = pd.read_csv(dest / "m9_steps_table.tsv", sep="\t")
    m9[m9.arm == "D"].to_csv(dest / "ref_m9_steps_table.tsv", sep="\t", index=False)
    for f in dest.iterdir():
        if not f.name.startswith(PHASE3_IN) and not f.name.startswith("ref_"):
            f.unlink()


# ------------------------------------------------------------------------------------ helpers


@pytest.fixture
def fx(tmp_path):
    d = tmp_path / "res"
    shutil.copytree(FIX, d)
    return d


def _checks(d):
    return (d / "checks.txt").read_text().splitlines()


def _line(lines, text):
    hit = [ln for ln in lines if text in ln]
    assert len(hit) == 1, (text, lines)
    return hit[0]


def _pad_viabase(path, n_law, n_depth):
    """Add fabricated law / depthlaw rows so every (arm, gv, seed) has n_law distinct law pairs
    (the same pairs on every seed) and n_depth depthlaw rows."""
    v = pd.read_csv(path, sep="\t", low_memory=False)
    out = [v]
    gv_arm = v.groupby(["arm", "g_version"])
    for (arm, gv), s in gv_arm:
        for seed, ss in s.groupby("seed"):
            law, dep = ss[ss.kind == "law"], ss[ss.kind == "depthlaw"]
            t = law.iloc[[0] * (n_law - len(law))].copy()
            t["level_src"] = [f"pad{i}" for i in range(len(t))]
            u = dep.iloc[[0] * (n_depth - len(dep))].copy()
            out += [t, u]
    pd.concat(out, ignore_index=True).to_csv(path, sep="\t", index=False)


# ------------------------------------------------------------------------------------- tests


def test_fixture_present():
    for f in PHASE3_IN:
        assert (FIX / f).is_file() and (FIX / f"ref_{f}").is_file()
    assert (FIX / "ref_m1_never_table.tsv").is_file() and (FIX / "ref_m9_steps_table.tsv").is_file()


def test_analyze_writes_every_table(fx):
    _analyze().main(fx)
    for t in TABLES:
        assert (fx / t).is_file() and (fx / t).stat().st_size > 0, t
    arms = set(pd.read_csv(fx / "m1_never_table.tsv", sep="\t").arm)
    assert arms == {"Dbigg", "Dbigf", "D", "Dbiggviabase", "Dbigfviabase", "Dviabase"}
    assert "DSfree" not in set(pd.read_csv(fx / "pair_table.tsv.gz", sep="\t").arm)
    n = pd.read_csv(fx / "n_table.tsv", sep="\t").set_index("arm")
    assert n.loc["Dbigg", "TRAIN_DONE"] == 9 and n.loc["Dbigg", "viabase_LAW_DONE"] == 9
    assert n.loc["Dbiggviabase", "viabase_LAW_DONE"] == 9 and n.loc["D", "LAW_DONE"] == 9
    m2 = pd.read_csv(fx / "m2_gap_table.tsv", sep="\t")
    assert set(m2.arm) == {"Dbigg", "Dbigf", "D"}
    m4 = pd.read_csv(fx / "m4_heldout_table.tsv", sep="\t")
    assert set(m4.direction) == {"base->7.5M", "7.5M->base", "both"}
    assert set(pd.read_csv(fx / "m5_depthlaw_table.tsv", sep="\t").arm) == arms
    rk = pd.read_csv(fx / "rank_table.tsv", sep="\t")
    assert set(rk.arm) == {"Dbiggviabase", "Dbigfviabase", "Dviabase"}
    for gv, s in rk.groupby("g_version"):
        assert list(s["rank"]) == [1, 2, 3] and s.m1_crps_never.is_monotonic_increasing
    assert rk.direct_m1_crps_never.notna().all() and rk.direct_m2_gen_gap.notna().all()


def test_m1_split_and_prereg(fx):
    a = _analyze()
    a.main(fx)
    m1 = pd.read_csv(fx / "m1_never_table.tsv", sep="\t")
    for c in ("crps_all", "crps_oracle_scaled_all", "scale_error_all", "crps_s0", "crps_s1",
              "crps_s2", "seed_spread", "n_pairs", "n_runs", "n_step_confounded", "gap_vs_D",
              "label_vs_D", "gap_vs_Dviabase", "label_vs_Dviabase", "prereg_threshold",
              "prereg_label"):
        assert c in m1, c
    assert m1.crps_oracle_scaled_all.notna().all() and m1.scale_error_all.notna().all()
    assert ((m1.crps_oracle_scaled_all + m1.scale_error_all - m1.crps_all).abs() < 1e-9).all()
    via = m1[m1.arm.isin(["Dbiggviabase", "Dbigfviabase"])]
    ok = {"gain", "loss", a.WITHIN}
    assert set(via.prereg_label) <= ok and len(via)
    other = m1[~m1.arm.isin(["Dbiggviabase", "Dbigfviabase"])].prereg_label
    assert other.isna().all()                     # "n/a" (pandas reads it as NaN)
    pt = via[via.g_version == "per_track"]
    assert (pt.prereg_threshold >= 0.015 - 1e-12).all()
    assert (abs(pt.prereg_threshold - (2 * pt.seed_spread.clip(lower=0.0077)).clip(lower=0.015))
            < 1e-12).all()
    ac = via[via.g_version == "across"]
    assert (abs(ac.prereg_threshold - 2 * ac.seed_spread.clip(lower=0.064)) < 1e-12).all()
    # the rule itself
    assert a.prereg("per_track", 0.50, 0.545, 0.001, 3, 3) == (0.0154, "gain")
    assert a.prereg("per_track", 0.54, 0.545, 0.001, 3, 3)[1] == a.WITHIN
    assert a.prereg("per_track", 0.60, 0.545, 0.001, 3, 3)[1] == "loss"
    assert a.prereg("per_track", 0.50, 0.545, 0.03, 3, 3) == (0.06, a.WITHIN)
    assert a.prereg("across", 0.50, 0.636, 0.01, 3, 3) == (0.128, "gain")
    assert a.prereg("across", 0.55, 0.636, 0.01, 3, 3)[1] == a.WITHIN
    assert a.prereg("across", 0.55, 0.636, 0.01, 2, 3)[1] == "incomplete (n_seeds=2)"


def test_n_step_confounded_and_m9(fx):
    _analyze().main(fx)
    m1 = pd.read_csv(fx / "m1_never_table.tsv", sep="\t")
    cell = m1[(m1.g_version == "per_track") & (m1.knob == "all") & (m1.subgroup == "never")] \
        .set_index("arm").n_step_confounded
    assert cell["Dbigf"] == 1 and cell["Dbigfviabase"] == 1 and cell["Dbigg"] == 0
    assert cell["D"] == 0 and cell["Dviabase"] == 0
    conf = pd.read_csv(fx / "step_confounded_runs.tsv", sep="\t")
    assert set(conf.run) == {"Dbigf_C07M20_counts_real_s1", "Dbigg_all_counts_real_s2"}
    m9 = pd.read_csv(fx / "m9_steps_table.tsv", sep="\t").set_index(["arm", "g_version"])
    assert m9.loc[("Dbigf", "per_track"), "n_best_gt12000"] == 1
    assert m9.loc[("Dbigg", "across"), "n_best_last10"] == 1      # 13,750 >= 13,500
    assert m9.loc[("Dbigg", "across"), "n_at_cap"] == 1
    assert m9.loc[("Dbigf", "per_track"), "n_best_last10"] == 0   # 12,500 < 13,500
    assert m9.loc[("D", "per_track"), "max_steps"] == 12000
    lines = _checks(fx)
    assert "Dbigf_C07M20_counts_real_s1 (12500)" in _line(lines, "INFO step-confounded")


def test_2094_check_fails_small_then_passes_padded(fx):
    a = _analyze()
    a.main(fx)
    lines = _checks(fx)
    assert _line(lines, "via-base never pairs n_pairs == 2094").startswith("FAIL")
    assert _line(lines, "raw via-base records").startswith("FAIL")
    for f in ("viabase_fir.tsv.gz", "ref_viabase_fir.tsv.gz"):
        _pad_viabase(fx / f, 2094, 42)
    a.main(fx)
    m1 = pd.read_csv(fx / "m1_never_table.tsv", sep="\t")
    nv = m1[(m1.knob == "all") & (m1.subgroup == "never") & m1.arm.str.endswith("viabase")]
    assert set(nv.n_pairs) == {2094} and len(nv) == 6
    lines = _checks(fx)
    assert _line(lines, "via-base never pairs n_pairs == 2094").startswith("PASS")
    assert _line(lines, "raw via-base records").startswith("PASS")
    assert _line(lines, "re-pooled reference").startswith("FAIL")    # Dviabase moved


def test_checks_on_fixture(fx, monkeypatch):
    _analyze().main(fx)
    lines = _checks(fx)
    assert _line(lines, "re-pooled reference").startswith("PASS")
    assert "(4 cells)" in _line(lines, "re-pooled reference")
    run_line = _line(lines, "every phase-3 run has rung D")
    assert run_line.startswith("PASS") and "(18 runs)" in run_line
    opt = [ln for ln in lines if ln.startswith("INFO optimisation-failure reading")]
    assert len(opt) == 4
    assert all(ln.startswith(("PASS", "FAIL", "INFO")) for ln in lines)
    # a wrong size reads FAIL
    r = pd.read_csv(fx / "runs_fir.tsv", sep="\t")
    r.loc[r.run == "Dbigg_C07M20_counts_real_s0", "g_hidden"] = 64
    r.to_csv(fx / "runs_fir.tsv", sep="\t", index=False)
    _analyze().main(fx)
    bad = _line(_checks(fx), "every phase-3 run has rung D")
    assert bad.startswith("FAIL") and "Dbigg_C07M20_counts_real_s0" in bad


def test_full_tree_requires_15000(fx, monkeypatch):
    monkeypatch.setenv("T128P3_FULL_TREE", "1")
    _analyze().main(fx)                    # every fabricated run has max_steps 15,000
    assert "and max_steps 15000" in _line(_checks(fx), "every phase-3 run has rung D")
    r = pd.read_csv(fx / "runs_fir.tsv", sep="\t")
    r.loc[0, "max_steps"] = 300
    r.to_csv(fx / "runs_fir.tsv", sep="\t", index=False)
    with pytest.raises(SystemExit, match="max_steps != 15000"):
        _analyze().main(fx)


def test_extract_and_second_method_agree(tmp_path):
    root = make_tree(tmp_path / "fir")
    out = tmp_path / "readings"
    _extract(RD / "extract_records.py", out, root)
    rec = pd.read_csv(out / "records_fir.tsv.gz", sep="\t")
    assert set(rec.arm) == {"Dbigg", "Dbigf"} and set(rec.kind) == {"trained", "heldout", "law",
                                                                    "depthlaw", "swap"}
    assert not (rec.crps_all == 99.0).any()                           # eval != score dropped
    for c in ("g_hidden", "f_channels", "n_theta", "arm"):
        assert c in rec
    for c in ("cclean_form", "cclean_frozen", "lam", "vae", "setting"):
        assert c not in rec
    assert set(zip(rec.arm, rec.g_hidden, rec.f_channels, rec.n_theta)) == \
        {("Dbigg", 256, 32, 256), ("Dbigf", 64, 64, 512)}
    via = pd.read_csv(out / "viabase_fir.tsv.gz", sep="\t")
    assert set(via.arm) == {"Dbiggviabase", "Dbigfviabase"} and "Dviabase" not in set(via.arm)
    assert set(via.kind) == {"law", "depthlaw"}
    assert set(via.run) == set(rec.run)
    runs = pd.read_csv(out / "runs_fir.tsv", sep="\t")
    for c in ("TRAIN_DONE", "SCORE_DONE", "LAW_DONE", "viabase_LAW_DONE", "max_steps",
              "steps_run", "best_step", "val_nll", "s_per_step", "g_hidden", "f_channels", "node"):
        assert c in runs, c
    assert len(runs) == 18 and runs.viabase_LAW_DONE.eq(1).all() and runs["index"].tolist() == \
        list(range(18))
    assert runs.set_index("run").loc["Dbigf_all_counts_real_s0", "node"] == "fc1015"
    assert not (out / "latent_fir.tsv.gz").exists() and not (out / "cclean_fir.tsv").exists()
    # the cache: a second pass reads nothing
    r2 = subprocess.run([sys.executable, str(RD / "extract_records.py"), str(out), str(root)],
                        check=True, capture_output=True, text=True)
    assert "cache hits 18, read 0" in r2.stdout
    # the second method, from the raw files
    subprocess.run([sys.executable, str(RD / "verify_raw.py"), "collect",
                    str(out / "verify_collect_fir.json"), str(root)], check=True,
                   capture_output=True)
    res = tmp_path / "res"
    res.mkdir()
    for f in PHASE3_IN + ("verify_collect_fir.json",):
        shutil.copy(out / f, res / f)
    for f in REF_IN:
        shutil.copy(FIX / f"ref_{f}", res / f"ref_{f}")
    _analyze().main(res)
    subprocess.run([sys.executable, str(RD / "verify_raw.py"), "reduce",
                    str(res / "verify_raw.json"), str(res / "verify_collect_fir.json")],
                   check=True, capture_output=True)
    subprocess.run([sys.executable, str(RD / "verify_compare.py"), str(res)], check=True,
                   capture_output=True)
    v = (res / "verify.txt").read_text().splitlines()
    assert v and all(ln.startswith("AGREE") for ln in v), v
    assert len(v) == 12
    # a perturbed table is caught
    m3 = pd.read_csv(res / "m3_trained_table.tsv", sep="\t")
    m3.loc[(m3.arm == "Dbigg") & (m3.knob == "all"), "crps_all"] += 1e-3
    m3.to_csv(res / "m3_trained_table.tsv", sep="\t", index=False)
    subprocess.run([sys.executable, str(RD / "verify_compare.py"), str(res)], check=True,
                   capture_output=True)
    assert "DISAGREE metric 3" in (res / "verify.txt").read_text()


def test_fixture_is_current(tmp_path):
    """The committed phase-3 fixture equals what this chunk's extract writes today."""
    build_fixture(tmp_path / "fix", tmp_path / "work")
    for f in PHASE3_IN:
        a = pd.read_csv(tmp_path / "fix" / f, sep="\t")
        b = pd.read_csv(FIX / f, sep="\t")
        drop = ["node", "train_log"] if f == "runs_fir.tsv" else []
        pd.testing.assert_frame_equal(a.drop(columns=drop), b.drop(columns=drop))


@pytest.mark.skipif(not (MIRROR / "records_fir.tsv.gz").is_file(),
                    reason=f"the real phase-2 mirror is absent ({MIRROR})")
def test_reference_repooled_from_real_phase2_mirror(tmp_path):
    res = tmp_path / "res"
    res.mkdir()
    for f in PHASE3_IN:
        shutil.copy(FIX / f, res / f)
    for f in REF_IN:
        shutil.copy(MIRROR / f, res / f"ref_{f}")
    _analyze().main(res)
    m1 = pd.read_csv(res / "m1_never_table.tsv", sep="\t")
    nv = m1[(m1.knob == "all") & (m1.subgroup == "never")].set_index(["arm", "g_version"])
    assert abs(nv.loc[("D", "per_track"), "crps_all"] - 0.6624113425283543) <= 1e-9
    assert abs(nv.loc[("Dviabase", "per_track"), "crps_all"] - 0.5446540994208223) <= 1e-9
    assert abs(nv.loc[("Dviabase", "across"), "crps_all"] - 0.6363833387704192) <= 1e-9
    assert nv.loc[("Dviabase", "per_track"), "n_pairs"] == 2094
    assert nv.loc[("Dviabase", "across"), "n_pairs"] == 2094
    assert abs(nv.loc[("Dviabase", "per_track"), "seed_spread"] - 0.00770305551997752) <= 1e-9
    lines = _checks(res)
    assert _line(lines, "re-pooled reference").startswith("PASS")
    n = pd.read_csv(res / "n_table.tsv", sep="\t").set_index("arm")
    assert n.loc["D", "assigned"] == 24 and n.loc["Dviabase", "viabase_LAW_DONE"] == 24
    m9 = pd.read_csv(res / "m9_steps_table.tsv", sep="\t").set_index(["arm", "g_version"])
    ref9 = pd.read_csv(MIRROR / "m9_steps_table.tsv", sep="\t")
    ref9 = ref9[ref9.arm == "D"].set_index(["arm", "g_version"])
    for c in ("n_runs", "n_best_last10", "n_at_cap", "best_step_median", "s_per_step_median"):
        for k in ref9.index:
            assert abs(m9.loc[k, c] - ref9.loc[k, c]) <= 1e-9, (k, c)


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as w:
        if FIX.exists():
            shutil.rmtree(FIX)
        build_fixture(FIX, w)
    print("wrote", sorted(p.name for p in FIX.iterdir()))
