"""t128 phase 3 readings (plan/T126_ENCDEC_D2.md § "Phase 3: PI decisions, 2026-10-27" and its
"Design check" block; metrics 1, 2, 3, 4, 5, 9).

Inputs (beside this file, or in the dir given as the one argument; small tables only):
  records_fir.tsv.gz, viabase_fir.tsv.gz, runs_fir.tsv   (extract_records.py on Fir, phase 3)
  ref_records_fir.tsv.gz, ref_viabase_fir.tsv.gz, ref_runs_fir.tsv, ref_m1_never_table.tsv,
  ref_m9_steps_table.tsv   (copies of the phase-2 mirror; never recomputed on Fir)
Arms: Dbigg, Dbigf (direct), Dbiggviabase, Dbigfviabase (the same runs read through the base),
  and the reference D, Dviabase: the ref_* rows with arm D / Dviabase only (the DS rows dropped),
  re-pooled here from the raw phase-2 records with the same code path as phase 3.
Outputs (beside the inputs):
  n_table.tsv          runs per arm: assigned, run dirs, TRAIN/SCORE/LAW_DONE, via-base LAW_DONE
  m1_never_table.tsv   metric 1: never-trained arm->arm CRPS per arm x gv x knob x subgroup, gaps
                       vs D and Dviabase, n_step_confounded, prereg_threshold / prereg_label
  m2_gap_table.tsv     metric 2: generalisation gap = never-trained (same knob) - trained CRPS
  m3_trained_table.tsv metric 3: trained-pair CRPS per arm x gv x knob
  m4_heldout_table.tsv metric 4: held-out depth 7.5M CRPS per arm x gv x direction
  m5_depthlaw_table.tsv metric 5: median |log2 error| of the depth law (sanity check only)
  m9_steps_table.tsv   metric 9: best step in the last 10 % of the cap; > 12,000 counts
  step_confounded_runs.tsv  every run with best_step > 12,000
  rank_table.tsv       per gv, the via-base arms ranked by metric 1 (a reading, not a gate)
  checks.txt           the design checks (PASS / FAIL lines; INFO lines are readings)
  pair_table.tsv.gz, (stdout -> analyze_out.txt via run_all.sh)

Conventions (phase 2's analyze.py, verbatim):
  Pair groups: trained = kind trained (knob = arm_tgt for base_to_arm, else arm_src); heldout =
  kind heldout; same = law pair with arm_src == arm_tgt; cross = law pair with arm_src != arm_tgt;
  never = same + cross. A cross pair belongs to BOTH knobs it touches; knob "all" pools every pair
  of the group once. Pooled CRPS (t118): per seed, mean over a track's pairs, then mean over
  tracks; the reading is the mean over seeds. Every CRPS has crps_oracle_scaled_all and
  scale_error_all beside it.
Noise rule: seed_spread = max - min over the seeds of a per-seed value. A gap counts only if
  |gap| > 2 x max(spread_a, spread_ref); else "within noise"; "incomplete (n_seeds=k)" below 3.
Pre-registered reading (design check), via-base arms against Dviabase on metric 1:
  per_track threshold = max(2 x max(0.0077, spread_Dviabase_cell, spread_arm), 0.015); across =
  2 x max(0.064, spread_Dviabase_cell, spread_arm). "gain" when Dviabase - arm > threshold,
  "loss" when arm - Dviabase > threshold, else "within noise (no gain above threshold)".
Optimisation-failure reading (checks.txt, INFO): via-base never not a "gain" AND the direct
  arm's trained CRPS worse than D (gap > 0 with noise label "counts").
Metric 9 (run_all.sh sets T128P3_FULL_TREE=1 on the full tree, where every phase-3 run must have
  max_steps 15,000 or the script stops): best step "in the last 10 %" when best_step >= 0.9 x the
  run's max_steps (13,500 at 15,000; 10,800 for the reference at 12,000); "at cap" when steps_run
  >= max_steps; step-confounded when best_step > 12,000.

    /Users/mforooz/miniforge3/envs/candi-local/bin/python analyze.py [<dir>]
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DIRECT = ["Dbigg", "Dbigf"]
VIA = ["Dbiggviabase", "Dbigfviabase"]
REF = ["D", "Dviabase"]
ARMS = ["Dbigg", "Dbigf", "D", "Dbiggviabase", "Dbigfviabase", "Dviabase"]
#: the run arm's expected (g_hidden, f_channels) (plan_t128p3 C1/C2)
SIZES = {"Dbigg": (256, 32), "Dbigf": (64, 64)}
MARKS = ["TRAIN_DONE", "SCORE_DONE", "LAW_DONE"]
PAIR = ["track", "arm_src", "level_src", "arm_tgt", "level_tgt"]
METRICS = ("crps_all", "crps_oracle_scaled_all", "scale_error_all")
NUM = list(METRICS) + ["log2_true", "log2_pred", "val_nll", "swap_mal", "steps_run", "seed",
                       "best_step", "max_steps"]
N_SEEDS = 3
FULL_MAX_STEPS = 15000
STEP_CONFOUND = 12000            # a best step above this is step-confounded (design check)
N_ASSIGNED = 24                  # 8 g x 3 seeds per arm
N_VIA_NEVER = 2094               # via-base never-trained pairs per seed (phase 2's count)
N_VIA_RECORDS = 2136             # raw via-base records (law + depthlaw) per seed per gv
PREREG = {"per_track": (0.0077, 0.015), "across": (0.064, 0.0)}   # (spread floor, minimum)
REF_TOL = 1e-9
WITHIN = "within noise (no gain above threshold)"


def read(here, name):
    f = Path(here) / name
    return pd.read_csv(f, sep="\t", low_memory=False) if f.is_file() else pd.DataFrame()


def numeric(d, cols):
    for c in cols:
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def order(t, by):
    t = t.copy()
    t["_a"] = t.arm.map({a: i for i, a in enumerate(ARMS)})
    return t.sort_values(["_a"] + by).drop(columns="_a").reset_index(drop=True)


def load(here):
    d = read(here, "records_fir.tsv.gz")
    v = read(here, "viabase_fir.tsv.gz")
    rr = read(here, "ref_records_fir.tsv.gz")
    rv = read(here, "ref_viabase_fir.tsv.gz")
    parts = [d[d.arm.isin(DIRECT)] if len(d) else d, v[v.arm.isin(VIA)] if len(v) else v,
             rr[rr.arm == "D"] if len(rr) else rr, rv.assign(arm="Dviabase") if len(rv) else rv]
    d = pd.concat([p for p in parts if len(p)], ignore_index=True)
    d = numeric(d, NUM)
    d["seed"] = d.seed.astype(int)
    assert d.space.eq("counts").all() and d.model.eq("real").all()
    return d


def load_runs(here):
    r = read(here, "runs_fir.tsv")
    r = r[r.arm.isin(DIRECT)].copy() if len(r) else r
    ref = read(here, "ref_runs_fir.tsv")
    ref = ref[ref.arm == "D"].copy() if len(ref) else ref
    for x in (r, ref):
        for m in MARKS + ["viabase_LAW_DONE"]:
            if m in x:
                x[m] = pd.to_numeric(x[m], errors="coerce").fillna(0).astype(int)
        numeric(x, ["steps_run", "best_step", "max_steps", "s_per_step", "train_s", "g_hidden",
                    "f_channels", "n_theta", "seed"])
    return r, ref


# ------------------------------------------------------------------------------------- noise


def spread(s):
    s = pd.Series(s).dropna()
    return float(s.max() - s.min()) if len(s) >= 2 else np.nan


def noise_label(gap, sa, sb, na, nb):
    """Spec noise rule. Fewer than N_SEEDS seeds on either side: "incomplete (n_seeds=k)"."""
    if pd.isna(gap):
        return np.nan, "n/a"
    k = min(int(na) if pd.notna(na) else 0, int(nb) if pd.notna(nb) else 0)
    if k < N_SEEDS or pd.isna(sa) or pd.isna(sb):
        return np.nan, f"incomplete (n_seeds={k})"
    c = bool(abs(gap) > 2 * max(sa, sb))
    return c, ("counts" if c else "within noise")


def add_gaps(t, keys, value, refs=("D",), sp="seed_spread", suffix="", ns="n_seeds"):
    """Per row, the gap of `value` to the same cell (keys) of each reference arm, with the noise
    rule. Columns gap_vs_<ref>, spread_a, spread_<ref>, counts_vs_<ref>, label_vs_<ref>."""
    t = t.copy()
    t["spread_a" + suffix] = t[sp]
    for ref in refs:
        r = t[t.arm == ref].drop_duplicates(keys).set_index(keys)
        g, sb, cn, lb = [], [], [], []
        for _, row in t.iterrows():
            k = tuple(row[c] for c in keys)
            k = k[0] if len(k) == 1 else k
            if row.arm == ref or k not in r.index:
                g.append(np.nan); sb.append(np.nan); cn.append(np.nan); lb.append("n/a")
                continue
            ref_row = r.loc[k]
            gap = row[value] - ref_row[value]
            c, l = noise_label(gap, row[sp], ref_row[sp], row[ns], ref_row[ns])
            g.append(gap); sb.append(ref_row[sp]); cn.append(c); lb.append(l)
        t[f"gap_vs_{ref}{suffix}"] = g
        t[f"spread_{ref}{suffix}"] = sb
        t[f"counts_vs_{ref}{suffix}"] = cn
        t[f"label_vs_{ref}{suffix}"] = lb
    return t


def prereg(gv, value, ref_value, spread_arm, na, nr, spread_ref=np.nan):
    """Pre-registered reading of a via-base arm against Dviabase: (threshold, label). The floor
    constants are the Dviabase spreads of the (knob all, never) cell; every cell also takes its
    own reference spread, so a per-knob gain never passes below the noise rule."""
    floor, minimum = PREREG[gv]
    sa = 0.0 if pd.isna(spread_arm) else float(spread_arm)
    sr = 0.0 if pd.isna(spread_ref) else float(spread_ref)
    thr = max(2 * max(floor, sr, sa), minimum)
    if pd.isna(value) or pd.isna(ref_value):
        return thr, "n/a"
    k = min(int(na) if pd.notna(na) else 0, int(nr) if pd.notna(nr) else 0)
    if k < N_SEEDS:
        return thr, f"incomplete (n_seeds={k})"
    if ref_value - value > thr:
        return thr, "gain"
    if value - ref_value > thr:
        return thr, "loss"
    return thr, WITHIN


def add_prereg(t, keys):
    t = t.copy()
    r = t[t.arm == "Dviabase"].drop_duplicates(keys).set_index(keys)
    th, lb = [], []
    for _, row in t.iterrows():
        k = tuple(row[c] for c in keys)
        if row.arm not in VIA or k not in r.index:
            th.append(np.nan); lb.append("n/a")
            continue
        ref = r.loc[k]
        a, b = prereg(row.g_version, row.crps_all, ref.crps_all, row.seed_spread, row.n_seeds,
                      ref.n_seeds, ref.seed_spread)
        th.append(a); lb.append(b)
    t["prereg_threshold"] = th
    t["prereg_label"] = lb
    return t


def per_seed_summary(x, keys, value, how="median"):
    """median (or mean) over all rows, the per-seed values, their spread, n_seeds."""
    f = (lambda s: float(np.nanmedian(s)) if s.notna().any() else np.nan) if how == "median" \
        else (lambda s: float(np.nanmean(s)) if s.notna().any() else np.nan)
    out = []
    for k, s in x.groupby(keys, dropna=False):
        ps = s.groupby("seed")[value].apply(f)
        rec = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        rec.update({value: f(s[value]), "seed_spread": spread(ps), "n_seeds": int(ps.notna().sum()),
                    "n_runs": s.run.nunique(), "n_rows": int(s[value].notna().sum())})
        for sd, val in ps.items():
            rec[f"{value}_s{int(sd)}"] = val
        out.append(rec)
    return pd.DataFrame(out)


def pooled(p, keys):
    """t118 pooling: per (keys, seed): mean over a track's pairs, then mean over tracks; the
    reading = mean over seeds. Per-seed crps_s<seed>, seed_spread on crps_all."""
    tr = p.groupby(keys + ["seed", "track"])[list(METRICS)].mean()
    tr["n"] = p.groupby(keys + ["seed", "track"]).size()
    ps = tr.groupby(keys + ["seed"]).agg(**{m: (m, "mean") for m in METRICS},
                                         n_tracks=("n", "size"), n_rec=("n", "sum")).reset_index()
    npairs = p.drop_duplicates(keys + PAIR).groupby(keys).size()
    nruns = p.groupby(keys).run.nunique()
    out = []
    for k, s in ps.groupby(keys):
        k = k if isinstance(k, tuple) else (k,)
        rec = dict(zip(keys, k))
        for m in METRICS:
            rec[m] = float(s[m].mean()) if s[m].notna().any() else np.nan
        rec["seed_spread"] = spread(s.crps_all)
        rec["n_seeds"] = len(s)
        for _, r in s.iterrows():
            rec[f"crps_s{int(r.seed)}"] = r.crps_all
        kk = k[0] if len(k) == 1 else k
        rec["n_pairs"] = int(npairs.loc[kk])
        rec["n_pair_records"] = int(s.n_rec.sum())
        rec["n_tracks_per_seed"] = ",".join(str(int(v)) for v in s.n_tracks)
        rec["n_runs"] = int(nruns.loc[kk])
        out.append(rec)
    t = pd.DataFrame(out)
    for sd in (0, 1, 2):
        if f"crps_s{sd}" not in t:
            t[f"crps_s{sd}"] = np.nan
    ss = [f"crps_s{sd}" for sd in (0, 1, 2)]
    rest = [c for c in t.columns if c not in ss]
    i = rest.index("n_seeds") + 1
    return t[rest[:i] + ss + rest[i:]]


# ------------------------------------------------------------------------------------- labels


def knob_of(x):
    return np.where(x.direction == "base_to_arm", x.arm_tgt, x.arm_src)


def label(d):
    """trained / heldout / same / cross pair rows with their knob (cross: both knobs)."""
    t = d[d.kind.isin(["trained", "heldout"])].copy()
    t["group"] = t.kind
    t["knob"] = knob_of(t)
    law = d[d.kind == "law"].copy()
    law["group"] = np.where(law.arm_src == law.arm_tgt, "same", "cross")
    a = law.assign(knob=law.arm_src)
    b = law[law.group == "cross"].assign(knob=lambda x: x.arm_tgt)
    return pd.concat([t, a, b], ignore_index=True)


def with_all(p, subgroups=True):
    """Add knob 'all' per group (each pair once) and the never subgroup (same + cross)."""
    once = p.drop_duplicates(["arm", "run", "kind", "group"] + PAIR)
    parts = [p, once.assign(knob="all")]
    if subgroups:
        nev = p[p.group.isin(["same", "cross"])]
        parts += [nev.assign(group="never"),
                  once[once.group.isin(["same", "cross"])].assign(group="never", knob="all")]
    return pd.concat(parts, ignore_index=True)


# ------------------------------------------------------------------------------------- tables


def n_table(d, runs, ref_runs):
    rows = []
    for arm, r in [(a, runs[runs.arm == a]) for a in DIRECT] + [("D", ref_runs)]:
        has = pd.to_numeric(r.run_dir, errors="coerce").fillna(0).astype(int) if len(r) else r
        rec = {"arm": arm, "assigned": N_ASSIGNED if arm in DIRECT else len(r),
               "run_dirs": int(has.sum()) if len(r) else 0}
        for m in MARKS:
            rec[m] = int(r[m].sum()) if len(r) and m in r else 0
        rec["viabase_LAW_DONE"] = (int(r["viabase_LAW_DONE"].sum())
                                   if len(r) and "viabase_LAW_DONE" in r else np.nan)
        rows.append(rec)
    for arm, src in zip(VIA + ["Dviabase"], DIRECT + ["D"]):
        rows.append({"arm": arm, "assigned": next(x["assigned"] for x in rows if x["arm"] == src),
                     "viabase_LAW_DONE": d[d.arm == arm].run.nunique()})
    t = pd.DataFrame(rows)
    for c in ["run_dirs"] + MARKS + ["viabase_LAW_DONE"]:
        t[c] = t[c].astype("Int64")
    return order(t, [])


def m1_never(p, best):
    x = with_all(p[p.group.isin(["same", "cross"])])
    k = ["arm", "g_version", "knob", "group"]
    t = pooled(x, k).rename(columns={"group": "subgroup"})
    conf = x[x.run.map(best).gt(STEP_CONFOUND)].groupby(k).run.nunique()
    conf.index = conf.index.set_names(["arm", "g_version", "knob", "subgroup"])
    t = t.merge(conf.rename("n_step_confounded").reset_index(),
                on=["arm", "g_version", "knob", "subgroup"], how="left")
    t["n_step_confounded"] = t.n_step_confounded.fillna(0).astype(int)
    t = add_gaps(t, ["g_version", "knob", "subgroup"], "crps_all", refs=("D", "Dviabase"))
    t = add_prereg(t, ["g_version", "knob", "subgroup"])
    return order(t, ["g_version", "subgroup", "knob"])


def m3_trained(p):
    x = with_all(p[(p.group == "trained") & ~p.arm.isin(VIA + ["Dviabase"])], subgroups=False)
    t = pooled(x, ["arm", "g_version", "knob"])
    t = add_gaps(t, ["g_version", "knob"], "crps_all")
    return order(t, ["g_version", "knob"])


def m2_gap(m1, m3):
    """gen_gap = never - trained, per seed and pooled; crps_never = same-knob law pairs only
    (knob k: the same pairs of k; knob all: every same pair, pooled once)."""
    nv = m1[m1.subgroup == "same"]
    nv = nv[~nv.arm.isin(VIA + ["Dviabase"])]
    k = ["arm", "g_version", "knob"]
    cols = list(METRICS) + [f"crps_s{s}" for s in (0, 1, 2)] + ["n_pairs"]
    t = nv[k + cols].merge(m3[k + cols], on=k, suffixes=("_never", "_trained"))
    t = t.rename(columns={"crps_all_never": "crps_never", "crps_all_trained": "crps_trained"})
    t["gen_gap"] = t.crps_never - t.crps_trained
    for m in METRICS[1:]:
        t[f"gen_gap_{m}"] = t[f"{m}_never"] - t[f"{m}_trained"]
    for s in (0, 1, 2):
        t[f"gen_gap_s{s}"] = t[f"crps_s{s}_never"] - t[f"crps_s{s}_trained"]
    gs = t[[f"gen_gap_s{s}" for s in (0, 1, 2)]]
    t["seed_spread"] = gs.apply(spread, axis=1)
    t["n_seeds"] = gs.notna().sum(axis=1)
    t = add_gaps(t, ["g_version", "knob"], "gen_gap")
    return order(t, ["g_version", "knob"])


def m4_heldout(p):
    x = p[(p.group == "heldout") & ~p.arm.isin(VIA + ["Dviabase"])].copy()
    x["direction"] = x.direction.map({"base_to_arm": "base->7.5M", "arm_to_base": "7.5M->base"})
    x = pd.concat([x, x.assign(direction="both")], ignore_index=True)
    t = pooled(x, ["arm", "g_version", "direction"])
    t = add_gaps(t, ["g_version", "direction"], "crps_all")
    return order(t, ["g_version", "direction"])


def m5_depthlaw(d):
    x = d[d.kind == "depthlaw"].copy()
    touches = (x.level_src == "7.5M") | (x.level_tgt == "7.5M")
    law = x.depth_source == "law"
    x["group"] = np.select([law & ~touches, law & touches, touches],
                           ["never", "never75", "heldout"], "trained")
    x["err"] = x.log2_pred - x.log2_true
    x["abs_err"] = x.err.abs()
    k = ["arm", "g_version", "group"]
    t = per_seed_summary(x, k, "abs_err").rename(columns={"abs_err": "median_abs_err"})
    t.columns = [c.replace("abs_err_s", "median_abs_err_s") for c in t.columns]
    t = t.merge(x.groupby(k).agg(median_err=("err", "median"),
                                 n_none=("err", lambda s: int(s.isna().sum()))).reset_index(), on=k)
    t = add_gaps(t, ["g_version", "group"], "median_abs_err", refs=("D", "Dviabase"))
    return order(t, ["g_version", "group"])


def m9_steps(runs, ref_runs, full):
    r = runs[runs.TRAIN_DONE == 1].copy() if len(runs) else runs
    if full:
        bad = r[r.max_steps != FULL_MAX_STEPS]
        if len(bad):
            raise SystemExit(f"full tree: {len(bad)} runs with max_steps != {FULL_MAX_STEPS}: "
                             + ", ".join(f"{a}={b}" for a, b in zip(bad.run, bad.max_steps)))
    ref = ref_runs[ref_runs.TRAIN_DONE == 1] if len(ref_runs) else ref_runs
    r = pd.concat([x for x in (r, ref) if len(x)], ignore_index=True)
    r["last10"] = r.best_step >= 0.9 * r.max_steps
    r["at_cap"] = r.steps_run >= r.max_steps
    r["gt12000"] = r.best_step > STEP_CONFOUND
    t = r.groupby(["arm", "g_version"]).agg(
        n_runs=("run", "size"), max_steps=("max_steps", "max"), n_best_last10=("last10", "sum"),
        n_at_cap=("at_cap", "sum"), n_best_gt12000=("gt12000", "sum"),
        best_step_median=("best_step", "median"), best_step_min=("best_step", "min"),
        best_step_max=("best_step", "max"), steps_run_median=("steps_run", "median"),
        s_per_step_median=("s_per_step", "median"),
        train_hours_median=("train_s", lambda v: float(v.median() / 3600))).reset_index()
    t["share_best_last10"] = t.n_best_last10 / t.n_runs
    cols = ["arm", "g_version", "n_runs", "max_steps", "n_best_last10", "share_best_last10",
            "n_at_cap", "n_best_gt12000", "best_step_median", "best_step_min", "best_step_max",
            "steps_run_median", "s_per_step_median", "train_hours_median"]
    conf = r[r.gt12000][["run", "arm", "g", "seed", "best_step", "steps_run"]]
    return order(t[cols], ["g_version"]), conf.sort_values("run").reset_index(drop=True)


def rank_table(m1, m2):
    a = m1[(m1.subgroup == "never") & (m1.knob == "all")]
    via = a[a.arm.isin(VIA + ["Dviabase"])][
        ["arm", "g_version", "crps_all", "crps_oracle_scaled_all", "scale_error_all",
         "seed_spread", "n_seeds", "n_runs", "n_step_confounded", "gap_vs_Dviabase",
         "label_vs_Dviabase", "prereg_threshold", "prereg_label"]].rename(
        columns={"crps_all": "m1_crps_never", "crps_oracle_scaled_all": "m1_crps_oracle_scaled_all",
                 "scale_error_all": "m1_scale_error_all", "seed_spread": "m1_seed_spread",
                 "n_seeds": "m1_n_seeds", "gap_vs_Dviabase": "m1_gap_vs_Dviabase",
                 "label_vs_Dviabase": "m1_label_vs_Dviabase"})
    via["direct_arm"] = via.arm.str.replace("viabase", "", regex=False)
    dr = a[a.arm.isin(DIRECT + ["D"])][
        ["arm", "g_version", "crps_all", "crps_oracle_scaled_all", "scale_error_all",
         "seed_spread", "gap_vs_D", "label_vs_D"]].rename(
        columns={"arm": "direct_arm", "crps_all": "direct_m1_crps_never",
                 "crps_oracle_scaled_all": "direct_m1_crps_oracle_scaled_all",
                 "scale_error_all": "direct_m1_scale_error_all",
                 "seed_spread": "direct_m1_seed_spread", "gap_vs_D": "direct_m1_gap_vs_D",
                 "label_vs_D": "direct_m1_label_vs_D"})
    b = m2[m2.knob == "all"][["arm", "g_version", "gen_gap", "gen_gap_crps_oracle_scaled_all",
                              "gen_gap_scale_error_all", "seed_spread", "gap_vs_D", "label_vs_D"]]
    b = b.rename(columns={"arm": "direct_arm", "gen_gap": "direct_m2_gen_gap",
                          "gen_gap_crps_oracle_scaled_all": "direct_m2_gen_gap_oracle_scaled_all",
                          "gen_gap_scale_error_all": "direct_m2_gen_gap_scale_error_all",
                          "seed_spread": "direct_m2_seed_spread", "gap_vs_D": "direct_m2_gap_vs_D",
                          "label_vs_D": "direct_m2_label_vs_D"})
    t = via.merge(dr, on=["direct_arm", "g_version"], how="left") \
           .merge(b, on=["direct_arm", "g_version"], how="left")
    out = []
    for gv, s in t.groupby("g_version"):
        s = s.sort_values("m1_crps_never").reset_index(drop=True)
        s.insert(1, "rank", range(1, len(s) + 1))
        out.append(s)
    return pd.concat(out, ignore_index=True) if out else t


def pair_table(p):
    k = ["arm", "g_version", "group", "knob"] + PAIR
    return p.groupby(k)[list(METRICS)].mean().reset_index()


# ------------------------------------------------------------------------------------- checks


def checks(here, d, m1, m3, runs, conf, full):
    out = []

    def put(ok, text):
        out.append(f"{'PASS' if ok else 'FAIL'} {text}")

    nv = m1[(m1.subgroup == "never") & (m1.knob == "all") & m1.arm.isin(VIA + ["Dviabase"])]
    cells = {(a, g): n for a, g, n in zip(nv.arm, nv.g_version, nv.n_pairs)}
    want = [(a, g) for a in VIA + ["Dviabase"] for g in ("per_track", "across")]
    bad = [f"{a}|{g}={cells.get((a, g), 'missing')}" for a, g in want
           if cells.get((a, g)) != N_VIA_NEVER]
    put(not bad, f"via-base never pairs n_pairs == {N_VIA_NEVER} at knob all / subgroup never in "
                 f"every (arm, gv) cell ({len(want)} cells)" + (": " + ", ".join(bad) if bad else ""))
    v = d[d.arm.isin(VIA + ["Dviabase"]) & d.kind.isin(["law", "depthlaw"])]
    n = v.groupby(["arm", "g_version", "seed"]).size()
    bad = [f"{a}|{g}|s{s}={c}" for (a, g, s), c in n.items() if c != N_VIA_RECORDS]
    miss = [f"{a}|{g}" for a, g in want if (a, g) not in {(x, y) for x, y, _ in n.index}]
    put(not bad and not miss, f"raw via-base records (law + depthlaw) == {N_VIA_RECORDS} per (arm, "
                              f"gv, seed) ({len(n)} cells)"
        + (": " + ", ".join(bad + [m + "=missing" for m in miss]) if bad or miss else ""))
    ref = read(here, "ref_m1_never_table.tsv")
    bad, nchk = [], 0
    for a in REF:
        for g in ("per_track", "across"):
            mine = m1[(m1.arm == a) & (m1.g_version == g) & (m1.knob == "all")
                      & (m1.subgroup == "never")].crps_all
            theirs = (ref[(ref.arm == a) & (ref.g_version == g) & (ref.knob == "all")
                          & (ref.subgroup == "never")].crps_all if len(ref) else pd.Series([]))
            if len(mine) != 1 or len(theirs) != 1:
                bad.append(f"{a}|{g}=missing")
                continue
            nchk += 1
            x, y = float(mine.iloc[0]), float(theirs.iloc[0])
            if not abs(x - y) <= REF_TOL:
                bad.append(f"{a}|{g}: re-pooled {x!r} phase-2 table {y!r}")
    put(not bad, f"re-pooled reference D / Dviabase all-never crps_all equals "
                 f"ref_m1_never_table.tsv to {REF_TOL:g} ({nchk} cells)"
        + (": " + "; ".join(bad) if bad else ""))
    bad = []
    rr = runs[pd.to_numeric(runs.run_dir, errors="coerce").fillna(0).astype(int) == 1] \
        if len(runs) else runs
    for _, r in rr.iterrows():
        why = []
        if r.get("rung") != "D":
            why.append(f"rung {r.get('rung')}")
        if full and r.max_steps != FULL_MAX_STEPS:
            why.append(f"max_steps {r.max_steps}")
        if (r.g_hidden, r.f_channels) != SIZES.get(r.arm):
            why.append(f"(g_hidden, f_channels) ({r.g_hidden}, {r.f_channels})")
        if why:
            bad.append(f"{r.run}: " + ", ".join(why))
    put(not bad and len(rr) > 0,
        f"every phase-3 run has rung D, (g_hidden, f_channels) of its arm "
        f"{{Dbigg: (256, 32), Dbigf: (64, 64)}}"
        + (f" and max_steps {FULL_MAX_STEPS}" if full else " (max_steps not checked: not the full tree)")
        + f" ({len(rr)} runs)" + (": " + "; ".join(bad) if bad else ""))
    a = m1[(m1.subgroup == "never") & (m1.knob == "all")].set_index(["arm", "g_version"])
    tr = m3[m3.knob == "all"].set_index(["arm", "g_version"])
    for arm in DIRECT:
        for g in ("per_track", "across"):
            k, kv = (arm, g), (arm + "viabase", g)
            if kv not in a.index or k not in tr.index:
                out.append(f"INFO optimisation-failure reading {arm} {g}: n/a (cell missing)")
                continue
            nv_lab = a.loc[kv, "prereg_label"]
            gap = tr.loc[k, "gap_vs_D"]
            not_improved = nv_lab != "gain"
            worse = pd.notna(gap) and gap > 0 and tr.loc[k, "label_vs_D"] == "counts"
            read_ = ("read as optimisation failure (lr)" if not_improved and worse
                     else "not an optimisation failure")
            out.append(f"INFO optimisation-failure reading {arm} {g}: via-base never prereg "
                       f"'{nv_lab}', trained crps gap vs D {gap:+.6f} "
                       f"({tr.loc[k, 'label_vs_D']}) -> {read_}")
    out.append(f"INFO step-confounded runs (best_step > {STEP_CONFOUND}): {len(conf)}"
               + (": " + ", ".join(f"{r.run} ({int(r.best_step)})" for r in conf.itertuples())
                  if len(conf) else ""))
    return out


# ------------------------------------------------------------------------------------- main


def show(title, t, cols):
    print(f"\n{title}")
    print(t[[c for c in cols if c in t]].round(4).to_string(index=False) if len(t) else "(empty)")


def main(here=HERE):
    here = Path(here)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    full = os.environ.get("T128P3_FULL_TREE") == "1"
    d = load(here)
    runs, ref_runs = load_runs(here)
    best = pd.concat([runs.set_index("run").best_step if len(runs) else pd.Series(dtype=float),
                      ref_runs.set_index("run").best_step if len(ref_runs) else
                      pd.Series(dtype=float)])
    nt = n_table(d, runs, ref_runs)
    nt.to_csv(here / "n_table.tsv", sep="\t", index=False)
    show("n table", nt, list(nt.columns))
    p = label(d)
    G = ["arm", "g_version"]
    m1 = m1_never(p, best)
    m1.to_csv(here / "m1_never_table.tsv", sep="\t", index=False)
    show("metric 1: never-trained pooled CRPS (knob all)", m1[m1.knob == "all"],
         G + ["subgroup", "crps_all", "crps_oracle_scaled_all", "scale_error_all", "seed_spread",
              "n_pairs", "n_runs", "n_step_confounded", "gap_vs_D", "label_vs_D",
              "gap_vs_Dviabase", "label_vs_Dviabase", "prereg_threshold", "prereg_label"])
    m3 = m3_trained(p)
    m2 = m2_gap(m1, m3)
    m2.to_csv(here / "m2_gap_table.tsv", sep="\t", index=False)
    show("metric 2: generalisation gap (knob all)", m2[m2.knob == "all"],
         G + ["crps_never", "crps_trained", "gen_gap", "gen_gap_crps_oracle_scaled_all",
              "gen_gap_scale_error_all", "seed_spread", "gap_vs_D", "label_vs_D"])
    m3.to_csv(here / "m3_trained_table.tsv", sep="\t", index=False)
    show("metric 3: trained-pair pooled CRPS (knob all)", m3[m3.knob == "all"],
         G + ["crps_all", "crps_oracle_scaled_all", "scale_error_all", "seed_spread", "n_pairs",
              "gap_vs_D", "label_vs_D"])
    m4 = m4_heldout(p)
    m4.to_csv(here / "m4_heldout_table.tsv", sep="\t", index=False)
    show("metric 4: held-out 7.5M CRPS", m4,
         G + ["direction", "crps_all", "crps_oracle_scaled_all", "scale_error_all", "seed_spread",
              "gap_vs_D", "label_vs_D"])
    m5 = m5_depthlaw(d)
    m5.to_csv(here / "m5_depthlaw_table.tsv", sep="\t", index=False)
    show("metric 5: depth-law median |log2 error| (sanity check)", m5,
         G + ["group", "median_abs_err", "seed_spread", "n_rows", "gap_vs_D", "label_vs_D",
              "gap_vs_Dviabase", "label_vs_Dviabase"])
    m9, conf = m9_steps(runs, ref_runs, full)
    m9.to_csv(here / "m9_steps_table.tsv", sep="\t", index=False)
    conf.to_csv(here / "step_confounded_runs.tsv", sep="\t", index=False)
    show("metric 9: best step in the last 10 % of the cap; > 12,000", m9, list(m9.columns))
    rk = rank_table(m1, m2)
    rk.to_csv(here / "rank_table.tsv", sep="\t", index=False)
    show("rank by metric 1 through the base (never, knob all), the direct arm beside", rk,
         ["g_version", "rank", "arm", "m1_crps_never", "m1_crps_oracle_scaled_all",
          "m1_scale_error_all", "m1_seed_spread", "prereg_threshold", "prereg_label",
          "direct_m1_crps_never", "direct_m2_gen_gap"])
    pair_table(p).to_csv(here / "pair_table.tsv.gz", sep="\t", index=False)
    ck = checks(here, d, m1, m3, runs, conf, full)
    (here / "checks.txt").write_text("\n".join(ck) + "\n")
    print("\nchecks\n" + "\n".join(ck))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else HERE)
