"""t128 phase 3b readings (plan/T126_ENCDEC_D2.md § "Phase 3b" and its "Reading" block, crux-design
2026-10-29, PI-approved): the bigger f against the small reference on six seeds, per-track g only.

Inputs (beside this file, or in the dir given as the one argument; small tables only):
  records_fir.tsv.gz, viabase_fir.tsv.gz, runs_fir.tsv   new seeds 3-5 (Dbigf, D; extract_records.py
                                                         of phase 3 on the seeds3b tree)
  p3_records_fir.tsv.gz, p3_viabase_fir.tsv.gz, p3_runs_fir.tsv, p3_m1_never_table.tsv
                                                         Dbigf seeds 0-2 (the phase-3 mirror)
  ref_records_fir.tsv.gz, ref_viabase_fir.tsv.gz, ref_runs_fir.tsv
                                                         D seeds 0-2 (the phase-2 mirror)
  rescore/viabase.json + rescore/law.json                control (b), optional ("not run")
  analyze_p3.py (or ../readings/analyze.py)              phase-3 labelling (label, with_all)
Arms: Dbigf, D (direct); Dbigfviabase, Dviabase (the same runs read through the base). Only
  g_version per_track (one g per track, 7 g per seed); the across g is not read.

Per-seed value (t118 pooling, as phase 3): per (arm, cell, seed), mean over a track's pairs, then
  mean over the 7 tracks (= one per-track g each). oracle_scaled / scale_error split beside it.
  Metric 5: per seed, the median |log2 error| over the seed's rows (phase 3's convention).
Bars, per (metric, cell); lower is better; gap = mean_ref - mean_f (> 0 = the bigger f is better):
  SE = sqrt(sd_f^2/n_f + sd_ref^2/n_ref) (sd ddof 1); real bar = 4 SE;
  size bar = max(2 max(0.0077, c R_f, c R_ref), 0.015), R = seed range, c = 0.669 at 6 seeds (the
  6-seed range to its 3-seed equivalent), 1 at 3 seeds. Metric 5 uses floor 0 and minimum 0 (the
  CRPS constants do not apply to a log2 error).
  label: gain (|gap| > both bars) / gain real, below the size bar (real only) / within seed noise
  (not real) / loss, loss real, below the size bar (mirror). 95 % CI: gap -+ t(0.975, Welch df) SE.
Replication (seeds 3-5 alone): gap3 > 2 SE3, one-sided -> PASS / FAIL.
Control (a) kit drift: the reference's seeds 3-5 mean against its seeds 0-2 mean, |delta| <=
  4 SE3 (SE3 from the two 3-seed groups) -> PASS; FAIL -> the cell is ALSO read on seeds 3-5 only
  (both arms; new_* columns) and flagged.
Control (b) readout: a rescore of one phase-2 D run through the base (rescore/law.json) against
  its recorded via-base records (ref_viabase_fir.tsv.gz), law pairs matched on the pair key,
  max |delta crps_all| <= 1e-6 -> PASS; no rescore -> "not run".

Outputs (beside the inputs): per_seed_table.tsv, headline_table.tsv (metric 1 via the base,
  Dbigfviabase vs Dviabase, knob all first, then per knob), m1_direct_table.tsv,
  m3_trained_table.tsv, m4_heldout_table.tsv, m5_depthlaw_table.tsv, m9_steps_table.tsv,
  n_table.tsv, readout_control.json, checks.txt.

    /Users/mforooz/miniforge3/envs/candi-local/bin/python analyze.py [<dir>]
"""
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
OLD, NEW = (0, 1, 2), (3, 4, 5)
SEEDS = OLD + NEW
FLOOR, MINIMUM = 0.0077, 0.015
RANGE_TO_3 = {6: 0.669, 3: 1.0}
REAL_K, REP_K, DRIFT_K = 4.0, 2.0, 4.0
READOUT_TOL = 1e-6
P3_TOL = 1e-9
METRICS = ("crps_all", "crps_oracle_scaled_all", "scale_error_all")
PAIR = ["track", "arm_src", "level_src", "arm_tgt", "level_tgt"]
#: (g_hidden, f_channels, n_theta, max_steps) of each new run's arm (plan § Phase 3b a, b)
SPECS = {"Dbigf": (64, 64, 512, 15000), "D": (64, 32, 256, 12000)}
N_G = 7
DIRECT, VIA = ["Dbigf", "D"], ["Dbigfviabase", "Dviabase"]
MARKS = ["TRAIN_DONE", "SCORE_DONE", "LAW_DONE", "viabase_LAW_DONE"]


def _phase3():
    for f in (HERE / "analyze_p3.py", HERE.parent / "readings" / "analyze.py"):
        if f.is_file():
            spec = importlib.util.spec_from_file_location("t128p3_analyze", f)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            return m
    raise SystemExit("phase-3 analyze.py not found (analyze_p3.py or ../readings/analyze.py)")


# ------------------------------------------------------------------------------------- stats


def _vals(d, seeds=None):
    v = [float(d[s]) for s in sorted(d) if (seeds is None or s in seeds) and pd.notna(d[s])]
    return np.array(v)


def summary(d, seeds=None):
    """n, mean, sd (ddof 1), range of {seed: value} (restricted to `seeds`)."""
    v = _vals(d, seeds)
    n = len(v)
    return {"n": n, "mean": float(v.mean()) if n else np.nan,
            "sd": float(v.std(ddof=1)) if n >= 2 else np.nan,
            "range": float(v.max() - v.min()) if n >= 2 else np.nan}


def compare(f, r, seeds=SEEDS, floor=FLOOR, minimum=MINIMUM):
    """The two bars, label and 95 % CI of the bigger f ({seed: value}) against the reference."""
    a, b = summary(f, seeds), summary(r, seeds)
    want = len(seeds)
    out = {"n_f": a["n"], "mean_f": a["mean"], "sd_f": a["sd"], "range_f": a["range"],
           "n_ref": b["n"], "mean_ref": b["mean"], "sd_ref": b["sd"], "range_ref": b["range"],
           "gap": b["mean"] - a["mean"]}
    if min(a["n"], b["n"]) < 2:
        return {**out, "se": np.nan, "real_bar": np.nan, "size_bar": np.nan, "ci_lo": np.nan,
                "ci_hi": np.nan, "df": np.nan, "label": f"incomplete (n_seeds={min(a['n'], b['n'])})"}
    va, vb = a["sd"] ** 2 / a["n"], b["sd"] ** 2 / b["n"]
    se = math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (a["n"] - 1) + vb ** 2 / (b["n"] - 1)) if va + vb > 0 else np.inf
    half = float(stats.t.ppf(0.975, df)) * se
    ca, cb = RANGE_TO_3.get(a["n"], np.nan), RANGE_TO_3.get(b["n"], np.nan)
    size = max(2 * max(floor, ca * a["range"], cb * b["range"]), minimum)
    gap = out["gap"]
    out.update(se=se, real_bar=REAL_K * se, size_bar=size, df=df, ci_lo=gap - half, ci_hi=gap + half)
    if a["n"] < want or b["n"] < want:
        out["label"] = f"incomplete (n_seeds={min(a['n'], b['n'])})"
    elif not abs(gap) > REAL_K * se:
        out["label"] = "within seed noise"
    else:
        side = "gain" if gap > 0 else "loss"
        out["label"] = side if abs(gap) > size else f"{side} real, below the size bar"
    return out


def replication(f, r):
    """Seeds 3-5 alone: gap3 > 2 SE3, one-sided."""
    a, b = summary(f, NEW), summary(r, NEW)
    gap = b["mean"] - a["mean"]
    if min(a["n"], b["n"]) < len(NEW):
        return {"gap3": gap, "se3": np.nan, "rep_bar": np.nan,
                "replication": f"n/a (n_seeds={min(a['n'], b['n'])})"}
    se = math.sqrt(a["sd"] ** 2 / a["n"] + b["sd"] ** 2 / b["n"])
    return {"gap3": gap, "se3": se, "rep_bar": REP_K * se,
            "replication": "PASS" if gap > REP_K * se else "FAIL"}


def drift(r):
    """Control (a): the reference's seeds 3-5 mean against its seeds 0-2 mean, |delta| <= 4 SE3."""
    o, n = summary(r, OLD), summary(r, NEW)
    delta = n["mean"] - o["mean"]
    if min(o["n"], n["n"]) < 3:
        return {"drift_delta": delta, "drift_se3": np.nan, "drift_bar": np.nan,
                "control_a": f"n/a (n_seeds={min(o['n'], n['n'])})"}
    se = math.sqrt(o["sd"] ** 2 / o["n"] + n["sd"] ** 2 / n["n"])
    return {"drift_delta": delta, "drift_se3": se, "drift_bar": DRIFT_K * se,
            "control_a": "PASS" if abs(delta) <= DRIFT_K * se else "FAIL"}


def reading(f, r, floor=FLOOR, minimum=MINIMUM):
    """One cell: six-seed bars, replication, control (a), and on a control (a) FAIL the same
    bars on seeds 3-5 alone (new_*) with the flag."""
    out = compare(f, r, SEEDS, floor, minimum)
    out.update(replication(f, r))
    out.update(drift(r))
    if out["control_a"] == "FAIL":
        nw = compare(f, r, NEW, floor, minimum)
        out.update({f"new_{k}": v for k, v in nw.items()})
        out["flag"] = "kit drift: also read on seeds 3-5 only (new_*)"
    else:
        out["flag"] = ""
    return out


def _key(v):
    """A pair-key cell as one spelling on both sides (a level read as 36, 36.0 or "36")."""
    try:
        return repr(float(v))
    except (TypeError, ValueError):
        return str(v)


def readout(rescore_dir, recorded):
    """Control (b): rescore_dir/law.json (+ viabase.json naming the old run) against the
    recorded via-base law records of that run (a frame of extracted rows)."""
    rd = Path(rescore_dir)
    vj, lj = rd / "viabase.json", rd / "law.json"
    if not (vj.is_file() and lj.is_file()):
        return {"control_b": "not run", "why": f"no {vj} / {lj}"}
    old = json.loads(vj.read_text()).get("old_run_dir")
    run = Path(str(old).rstrip("/")).name
    s = json.loads(lj.read_text())
    new = pd.DataFrame([r for r in s["records"] if r.get("kind") == "law" and r.get("eval") == "score"])
    rec = recorded[(recorded.run == run) & (recorded.kind == "law")] if len(recorded) else recorded
    res = {"run": run, "old_run_dir": old, "n_rescored": len(new), "n_recorded": len(rec)}
    if not len(new) or not len(rec):
        return {**res, "control_b": "FAIL", "why": "no law records on one side"}
    a = pd.to_numeric(new.crps_all, errors="coerce").groupby([new[c].map(_key) for c in PAIR]).mean()
    b = pd.to_numeric(rec.crps_all, errors="coerce").groupby([rec[c].map(_key) for c in PAIR]).mean()
    j = pd.concat([a.rename("rescored"), b.rename("recorded")], axis=1)
    miss = int(j.isna().any(axis=1).sum())
    d = (j.rescored - j.recorded).abs()
    res.update(n_pairs=len(j), n_unmatched=miss, max_abs_delta=float(d.max()),
               mean_rescored=float(j.rescored.mean()), mean_recorded=float(j.recorded.mean()))
    ok = miss == 0 and bool(d.max() <= READOUT_TOL)
    return {**res, "control_b": "PASS" if ok else "FAIL"}


# ------------------------------------------------------------------------------------- data


def read(here, name):
    f = Path(here) / name
    return pd.read_csv(f, sep="\t", low_memory=False) if f.is_file() else pd.DataFrame()


def _num(d, cols):
    for c in cols:
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def load(here):
    """Per-track rows of the four arms on seeds 0-5, each seed set from its own tree."""
    n_rec, n_via = read(here, "records_fir.tsv.gz"), read(here, "viabase_fir.tsv.gz")
    p_rec, p_via = read(here, "p3_records_fir.tsv.gz"), read(here, "p3_viabase_fir.tsv.gz")
    r_rec, r_via = read(here, "ref_records_fir.tsv.gz"), read(here, "ref_viabase_fir.tsv.gz")
    parts = []
    for x, arms, seeds in [(n_rec, DIRECT, NEW), (n_via, VIA, NEW), (p_rec, ["Dbigf"], OLD),
                           (p_via, ["Dbigfviabase"], OLD), (r_rec, ["D"], OLD),
                           (r_via.assign(arm="Dviabase") if len(r_via) else r_via, ["Dviabase"], OLD)]:
        if not len(x):
            continue
        x = _num(x.copy(), list(METRICS) + ["seed", "log2_true", "log2_pred"])
        x = x[x.arm.isin(arms) & (x.g_version == "per_track")]
        bad = sorted(set(x.seed.astype(int)) - set(seeds))
        if bad:
            raise SystemExit(f"arms {arms}: seeds {bad} outside {seeds}")
        parts.append(x)
    d = pd.concat(parts, ignore_index=True)
    d["seed"] = d.seed.astype(int)
    return d


def per_seed(x, keys, value="crps_all", how="pooled"):
    """One value per (keys, seed). pooled: mean over a track's pairs, then over tracks, for every
    CRPS of METRICS; median: the median of `value` over the seed's rows."""
    if how == "pooled":
        tr = x.groupby(keys + ["seed", "track"])[list(METRICS)].mean()
        out = tr.groupby(keys + ["seed"]).agg(**{m: (m, "mean") for m in METRICS},
                                              n_tracks=("crps_all", "size"))
    else:
        out = x.groupby(keys + ["seed"]).agg(**{value: (value, "median")},
                                             n_tracks=("track", "nunique"))
    return out.reset_index()


def tables(d, P3):
    """Per-seed long table: metric, arm, cell, seed, value (+ split)."""
    p = P3.label(d)
    longs = []

    def put(metric, t, cell, value="crps_all"):
        t = t.rename(columns={cell: "cell", value: "value"})
        t.insert(0, "metric", metric)
        longs.append(t)

    nv = P3.with_all(p[p.group.isin(["same", "cross"])])
    nv = nv[nv.group == "never"]
    put("m1_never", per_seed(nv, ["arm", "knob"]), "knob")
    tr = P3.with_all(p[(p.group == "trained") & p.arm.isin(DIRECT)], subgroups=False)
    put("m3_trained", per_seed(tr, ["arm", "knob"]), "knob")
    h = p[(p.group == "heldout") & p.arm.isin(DIRECT)].copy()
    h["direction"] = h.direction.map({"base_to_arm": "base->7.5M", "arm_to_base": "7.5M->base"})
    h = pd.concat([h, h.assign(direction="both")], ignore_index=True)
    put("m4_heldout", per_seed(h, ["arm", "direction"]), "direction")
    x = d[d.kind == "depthlaw"].copy()
    touches = (x.level_src == "7.5M") | (x.level_tgt == "7.5M")
    x = x[(x.depth_source == "law") & ~touches].assign(cell="never")
    x["abs_err"] = (x.log2_pred - x.log2_true).abs()
    put("m5_depthlaw", per_seed(x, ["arm", "cell"], "abs_err", how="median"), "cell", "abs_err")
    t = pd.concat(longs, ignore_index=True)
    cols = ["metric", "arm", "cell", "seed", "value", "crps_oracle_scaled_all", "scale_error_all",
            "n_tracks"]
    return t[[c for c in cols if c in t]].sort_values(["metric", "arm", "cell", "seed"])


def cell_table(ps, metric, f_arm, r_arm, floor=FLOOR, minimum=MINIMUM, comparison=""):
    """One row per cell: both arms' per-seed values, mean, sd, R6, split, and reading()."""
    s = ps[ps.metric == metric]
    cells = sorted(set(s[s.arm == f_arm].cell) | set(s[s.arm == r_arm].cell),
                   key=lambda c: (c != "all", c))
    rows = []
    for c in cells:
        rec = {"metric": metric, "comparison": comparison, "cell": c, "arm_f": f_arm, "arm_ref": r_arm}
        vals = {}
        for tag, arm in (("f", f_arm), ("ref", r_arm)):
            a = s[(s.arm == arm) & (s.cell == c)].set_index("seed")
            vals[tag] = a.value.to_dict()
            for sd in SEEDS:
                rec[f"{tag}_s{sd}"] = vals[tag].get(sd, np.nan)
            for m in METRICS[1:]:
                if m in a:
                    rec[f"{tag}_{m}"] = float(a[m].mean()) if a[m].notna().any() else np.nan
            rec[f"{tag}_n_tracks"] = ",".join(str(int(v)) for v in a.n_tracks)
        rec.update(reading(vals["f"], vals["ref"], floor, minimum))
        rows.append(rec)
    return pd.DataFrame(rows)


def m9_steps(here):
    rs = []
    for name, arms, seeds in [("runs_fir.tsv", DIRECT, NEW), ("p3_runs_fir.tsv", ["Dbigf"], OLD),
                              ("ref_runs_fir.tsv", ["D"], OLD)]:
        r = read(here, name)
        if not len(r):
            continue
        r = _num(r, ["seed", "TRAIN_DONE", "best_step", "steps_run", "max_steps"])
        rs.append(r[r.arm.isin(arms) & (r.g_version == "per_track") & r.seed.isin(seeds)
                    & (r.TRAIN_DONE == 1)])
    r = pd.concat(rs, ignore_index=True)
    r["seeds"] = np.where(r.seed.isin(OLD), "0-2", "3-5")
    r = pd.concat([r, r.assign(seeds="0-5")], ignore_index=True)
    r["last10"] = r.best_step >= 0.9 * r.max_steps
    r["at_cap"] = r.steps_run >= r.max_steps
    return r.groupby(["arm", "seeds"]).agg(
        n_runs=("run", "size"), max_steps=("max_steps", "max"), n_best_last10=("last10", "sum"),
        n_at_cap=("at_cap", "sum"), best_step_median=("best_step", "median"),
        best_step_min=("best_step", "min"), best_step_max=("best_step", "max"),
        steps_run_median=("steps_run", "median")).reset_index()


def n_table(here, d):
    r = read(here, "runs_fir.tsv")
    rows = []
    for arm in DIRECT:
        x = r[r.arm == arm] if len(r) else r
        rec = {"arm": arm, "tree": "seeds3b", "assigned": N_G * len(NEW), "run_dirs": 0}
        if len(x):
            rec["run_dirs"] = int(pd.to_numeric(x.run_dir, errors="coerce").fillna(0).sum())
            for m in MARKS:
                rec[m] = int(pd.to_numeric(x[m], errors="coerce").fillna(0).sum()) if m in x else 0
        rows.append(rec)
    for arm in DIRECT + VIA:
        for sd in (OLD, NEW):
            x = d[(d.arm == arm) & d.seed.isin(sd)]
            rows.append({"arm": arm, "tree": f"seeds {sd[0]}-{sd[-1]} rows",
                         "runs_read": x.run.nunique(), "seeds_read": x.seed.nunique()})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------- checks


def checks(here, d, ps, head, rc):
    out = []

    def put(ok, text):
        out.append(f"{'PASS' if ok else 'FAIL'} {text}")

    r = read(here, "runs_fir.tsv")
    bad = []
    for arm, (gh, fc, nt, ms) in SPECS.items():
        x = r[r.arm == arm] if len(r) else r
        if len(x) != N_G * len(NEW):
            bad.append(f"{arm}: {len(x)} runs")
        for _, row in x.iterrows():
            got = tuple(pd.to_numeric(pd.Series([row.get(k) for k in
                        ("g_hidden", "f_channels", "n_theta", "max_steps")]), errors="coerce"))
            if got != (gh, fc, nt, ms) or int(row.seed) not in NEW:
                bad.append(f"{row.run}: sizes/max_steps {got}, seed {row.seed}")
            miss = [m for m in MARKS if int(pd.to_numeric(row.get(m), errors="coerce") or 0) != 1]
            if miss:
                bad.append(f"{row.run}: no {','.join(miss)}")
    put(not bad, f"new tree: {N_G * len(NEW)} runs per arm, seeds 3-5, every mark, (g_hidden, "
                 f"f_channels, n_theta, max_steps) = {SPECS}" + (": " + "; ".join(bad) if bad else ""))
    hs = ps[(ps.metric == "m1_never") & (ps.cell == "all") & ps.arm.isin(DIRECT + VIA)]
    bad = [f"{a}: seeds {sorted(s.seed)}" for a, s in hs.groupby("arm") if sorted(s.seed) != list(SEEDS)]
    bad += [f"{a}|s{sd}: {n} tracks" for a, sd, n in zip(hs.arm, hs.seed, hs.n_tracks) if n != N_G]
    bad += [f"{a}: missing" for a in DIRECT + VIA if a not in set(hs.arm)]
    put(not bad, f"metric 1 knob all: every arm has seeds 0-5, each with {N_G} tracks"
        + (": " + "; ".join(bad) if bad else ""))
    p3 = read(here, "p3_m1_never_table.tsv")
    bad, n = [], 0
    for arm in DIRECT + VIA:
        row = p3[(p3.arm == arm) & (p3.g_version == "per_track") & (p3.knob == "all")
                 & (p3.subgroup == "never")] if len(p3) else p3
        for sd in OLD:
            mine = hs[(hs.arm == arm) & (hs.seed == sd)].value
            if len(row) != 1 or len(mine) != 1:
                bad.append(f"{arm}|s{sd}=missing")
                continue
            n += 1
            x, y = float(mine.iloc[0]), float(row[f"crps_s{sd}"].iloc[0])
            if not abs(x - y) <= P3_TOL:
                bad.append(f"{arm}|s{sd}: {x!r} vs phase-3 {y!r}")
    put(not bad, f"seeds 0-2 per-seed metric 1 (knob all, never) equal the phase-3 m1 table to "
                 f"{P3_TOL:g} ({n} values)" + (": " + "; ".join(bad) if bad else ""))
    h = head[head.cell == "all"].iloc[0] if len(head) and (head.cell == "all").any() else None
    if h is not None:
        out.append(f"INFO headline (metric 1 via the base, knob all): Dbigfviabase "
                   f"{h.mean_f:.6f} vs Dviabase {h.mean_ref:.6f}; gap6 {h.gap:+.6f}, SE {h.se:.6f}, "
                   f"real bar {h.real_bar:.6f}, size bar {h.size_bar:.6f}, 95% CI "
                   f"[{h.ci_lo:+.6f}, {h.ci_hi:+.6f}] -> {h.label}")
        put(h.replication == "PASS", f"replication seeds 3-5: gap3 {h.gap3:+.6f} > 2 SE3 "
                                     f"{h.rep_bar:.6f} ({h.replication})")
        put(h.control_a == "PASS", f"control (a) kit drift: Dviabase seeds 3-5 - seeds 0-2 "
                                   f"{h.drift_delta:+.6f}, |delta| <= 4 SE3 {h.drift_bar:.6f} "
                                   f"({h.control_a})" + (f"; seeds 3-5 only: gap3 "
                                   f"{h.new_gap:+.6f} -> {h.new_label}" if h.flag else ""))
    else:
        out.append("FAIL headline cell missing")
    if rc["control_b"] == "not run":
        out.append("INFO control (b) readout: not run (no rescore/viabase.json + law.json)")
    else:
        put(rc["control_b"] == "PASS",
            f"control (b) readout: rescore of {rc.get('run')} vs its recorded via-base law records, "
            f"{rc.get('n_pairs')} pairs, {rc.get('n_unmatched')} unmatched, max |delta crps_all| "
            f"{rc.get('max_abs_delta')!r} <= {READOUT_TOL:g}")
    return out


# ------------------------------------------------------------------------------------- main


def show(title, t, cols):
    print(f"\n{title}")
    print(t[[c for c in cols if c in t]].round(6).to_string(index=False) if len(t) else "(empty)")


COLS = ["comparison", "cell", "arm_f", "arm_ref", "mean_f", "sd_f", "range_f", "mean_ref", "sd_ref",
        "range_ref", "gap", "se", "real_bar", "size_bar", "label", "ci_lo", "ci_hi", "gap3",
        "rep_bar", "replication", "drift_delta", "drift_bar", "control_a", "flag"]


def main(here=HERE):
    here = Path(here)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    P3 = _phase3()
    d = load(here)
    ps = tables(d, P3)
    ps.to_csv(here / "per_seed_table.tsv", sep="\t", index=False)
    nt = n_table(here, d)
    nt.to_csv(here / "n_table.tsv", sep="\t", index=False)
    show("n table", nt, list(nt.columns))
    head = cell_table(ps, "m1_never", "Dbigfviabase", "Dviabase", comparison="via the base")
    head.to_csv(here / "headline_table.tsv", sep="\t", index=False)
    show("HEADLINE metric 1 never-trained CRPS through the base, per-track g (knob all, then per "
         "knob)", head, COLS + ["f_crps_oracle_scaled_all", "f_scale_error_all",
                                "ref_crps_oracle_scaled_all", "ref_scale_error_all"])
    for name, metric, cmp_, kw in [
            ("m1_direct_table.tsv", "m1_never", [("direct", "Dbigf", "D")], {}),
            ("m3_trained_table.tsv", "m3_trained", [("direct", "Dbigf", "D")], {}),
            ("m4_heldout_table.tsv", "m4_heldout", [("direct", "Dbigf", "D")], {}),
            ("m5_depthlaw_table.tsv", "m5_depthlaw", [("via the base", "Dbigfviabase", "Dviabase"),
                                                       ("direct", "Dbigf", "D")],
             {"floor": 0.0, "minimum": 0.0})]:
        t = pd.concat([cell_table(ps, metric, f, r, comparison=c, **kw) for c, f, r in cmp_],
                      ignore_index=True)
        t.to_csv(here / name, sep="\t", index=False)
        show(f"{metric} ({name})", t, COLS)
    m9 = m9_steps(here)
    m9.to_csv(here / "m9_steps_table.tsv", sep="\t", index=False)
    show("metric 9: best steps", m9, list(m9.columns))
    rc = readout(here / "rescore", read(here, "ref_viabase_fir.tsv.gz"))
    (here / "readout_control.json").write_text(json.dumps(rc, indent=1, default=str) + "\n")
    ck = checks(here, d, ps, head, rc)
    (here / "checks.txt").write_text("\n".join(ck) + "\n")
    print("\nchecks\n" + "\n".join(ck))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else HERE)
