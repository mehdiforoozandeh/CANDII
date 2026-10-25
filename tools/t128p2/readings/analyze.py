"""t128 phase 2 readings (plan/T126_ENCDEC_D2.md § "Phase 2: PI decisions, 2026-10-25", metrics 1-9).

Inputs (beside this file, brought down by run_all.sh; small tables only):
  records_fir.tsv.gz, latent_fir.tsv.gz, cclean_fir.tsv, runs_fir.tsv, viabase_fir.tsv.gz
  (extract_records.py on Fir), and, when present, phase1_{knob,depth,fit}_table.tsv (copies of
  the phase-1 tables, for the side column "phase 1 (4000 steps)"; never recomputed).
Arms: DSfree, DSfixed, DSanchor, D (runs) and Dviabase (old-D maps through the base; never-
trained law pairs and the depth law only).
Outputs (beside this file):
  n_table.tsv        runs per arm: assigned, run dirs, TRAIN/SCORE/LAW/LATENT/HOP1_DONE, via-base
  m1_never_table.tsv metric 1: never-trained arm->arm CRPS per arm x gv x knob x subgroup
  m2_gap_table.tsv   metric 2: generalisation gap = never-trained - trained CRPS, same arm
  m3_trained_table.tsv metric 3: trained-pair CRPS per arm x gv x knob
  m4_heldout_table.tsv metric 4: held-out depth 7.5M CRPS per arm x gv x direction
  m5_depthlaw_table.tsv metric 5: median |log2 error| of the depth law per arm x gv x group
  m6_hop1_table.tsv  metric 6: hop-1 CRPS (prediction of X_base against the real base)
  m7_agree_table.tsv, m7_agree_points.tsv.gz  metric 7: |Z - Z'| against |x - x'|
  m8_swap_table.tsv, m8_swap_points.tsv.gz    metric 8: swap check against identity distance
  m9_steps_table.tsv metric 9: share of runs whose best step is in the last 10 % of the cap
  rank_table.tsv     per g version, the arms ranked by metric 1 with metric 2 beside
  cclean_table.tsv, pair_table.tsv.gz, (stdout -> analyze_out.txt via run_all.sh)

Conventions (phase-1 analyze.py, restated):
  arm = run-name prefix (Dviabase for via-base rows). Pair groups: trained = kind trained (knob
  = arm_tgt for base_to_arm, else arm_src); heldout = kind heldout (base <-> depth 7.5M, never
  trained); same = law pair with arm_src == arm_tgt; cross = law pair with arm_src != arm_tgt;
  never = same + cross. A cross pair belongs to BOTH knobs it touches (so "never, knob k" = every
  never-trained pair touching k); knob "all" pools every pair of the group once.
  Pooled CRPS (t118): per seed, mean over a track's pairs, then mean over tracks; the reading is
  the mean over seeds. Every CRPS column has crps_oracle_scaled_all and scale_error_all beside it.
Noise rule (spec): seed_spread = max - min over the seeds of a per-seed value. A gap between two
  cells counts only if |gap| > 2 x max(spread_a, spread_ref); otherwise "within noise". Cells
  with fewer than 2 seeds have no spread: "no spread (n_seeds < 2)".
Metric 9: a run's best step is "in the last 10 %" when best_step >= 0.9 x its max_steps (10,800
  at the 12,000 cap); "at cap" when steps_run >= max_steps (12,000).

    /Users/mforooz/miniforge3/envs/candi-local/bin/python analyze.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ARMS = ["DSfree", "DSfixed", "DSanchor", "D", "Dviabase"]
DS_ARMS = ["DSfree", "DSfixed", "DSanchor"]
HOP1_ARMS = ["DSfixed", "DSanchor", "D"]
MARKS = ["TRAIN_DONE", "SCORE_DONE", "LAW_DONE", "LATENT_DONE", "HOP1_DONE"]
PAIR = ["track", "arm_src", "level_src", "arm_tgt", "level_tgt"]
METRICS = ("crps_all", "crps_oracle_scaled_all", "scale_error_all")
NUM = list(METRICS) + ["log2_true", "log2_pred", "val_nll", "swap_mal", "steps_run", "seed",
                       "best_step", "max_steps", "c_star_all"]
PHASE1_ARM = {"DSfree": "DS", "D": "D", "Dviabase": "Dviabase"}   # Decision (vii)
P1 = "phase1_4000steps_"


def read(name):
    f = HERE / name
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


def load():
    d = read("records_fir.tsv.gz")
    v = read("viabase_fir.tsv.gz")
    d = pd.concat([d, v.assign(arm="Dviabase")], ignore_index=True)
    d = numeric(d, NUM)
    d["seed"] = d.seed.astype(int)
    assert d.space.eq("counts").all() and d.model.eq("real").all()
    return d


# ------------------------------------------------------------------------------------- noise


def spread(s):
    s = pd.Series(s).dropna()
    return float(s.max() - s.min()) if len(s) >= 2 else np.nan


def noise_label(gap, sa, sb):
    if pd.isna(gap):
        return np.nan, "n/a"
    if pd.isna(sa) or pd.isna(sb):
        return np.nan, "no spread (n_seeds < 2)"
    c = bool(abs(gap) > 2 * max(sa, sb))
    return c, ("counts" if c else "within noise")


def add_gaps(t, keys, value, refs=("D",), sp="seed_spread", suffix=""):
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
            c, l = noise_label(gap, row[sp], ref_row[sp])
            g.append(gap); sb.append(ref_row[sp]); cn.append(c); lb.append(l)
        t[f"gap_vs_{ref}{suffix}"] = g
        t[f"spread_{ref}{suffix}"] = sb
        t[f"counts_vs_{ref}{suffix}"] = cn
        t[f"label_vs_{ref}{suffix}"] = lb
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


def phase1(t, table, keys, value, col):
    """Side column "phase 1 (4000 steps)" (Decision (vii)): the phase-1 value of the mapped arm."""
    p = read(f"phase1_{table}.tsv")
    t = t.copy()
    if p.empty:
        t[P1 + col] = np.nan
        return t
    if "model" in p:
        p = p[p.model == "real"]
    inv = {v: k for k, v in PHASE1_ARM.items()}
    p = p[p.arm.isin(inv)].assign(arm=lambda x: x.arm.map(inv))
    p = p[["arm"] + keys + [value]].rename(columns={value: P1 + col}).drop_duplicates(["arm"] + keys)
    return t.merge(p, on=["arm"] + keys, how="left")


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


def n_table(d):
    r = read("runs_fir.tsv")
    for m in MARKS:
        r[m] = pd.to_numeric(r[m], errors="coerce").fillna(0).astype(int)
    r["has_dir"] = r.TRAIN_DONE.notna() & r.g.notna()
    t = r.groupby("arm").agg(assigned=("run", "size"), run_dirs=("has_dir", "sum"),
                             **{m: (m, "sum") for m in MARKS}).reset_index()
    t["viabase_LAW_DONE"] = np.nan
    nv = d[d.arm == "Dviabase"].run.nunique()
    for c in ["run_dirs"] + MARKS:
        t[c] = t[c].astype(int)
    t.loc[len(t)] = {"arm": "Dviabase", "assigned": int(t.loc[t.arm == "D", "assigned"].sum()),
                     "viabase_LAW_DONE": nv}
    for c in ["run_dirs"] + MARKS + ["viabase_LAW_DONE"]:
        t[c] = t[c].astype("Int64")
    return order(t, []), r


def m1_never(p):
    x = with_all(p[p.group.isin(["same", "cross"])])
    t = pooled(x, ["arm", "g_version", "knob", "group"]).rename(columns={"group": "subgroup"})
    t = phase1(t.rename(columns={"subgroup": "group"}), "knob_table", ["g_version", "group", "knob"],
               "crps_all", "crps_all").rename(columns={"group": "subgroup"})
    t = add_gaps(t, ["g_version", "knob", "subgroup"], "crps_all", refs=("D", "Dviabase"))
    return order(t, ["g_version", "subgroup", "knob"])


def m3_trained(p):
    x = with_all(p[(p.group == "trained") & (p.arm != "Dviabase")], subgroups=False)
    t = pooled(x, ["arm", "g_version", "knob"])
    t = phase1(t, "fit_table", ["g_version"], "crps_all", "crps_all")
    t.loc[t.knob != "all", P1 + "crps_all"] = np.nan
    t = add_gaps(t, ["g_version", "knob"], "crps_all")
    return order(t, ["g_version", "knob"])


def m2_gap(m1, m3):
    """gen_gap = never - trained, per seed and pooled; knob k: same-knob law pairs; knob all:
    every never-trained pair (same + cross)."""
    nv = pd.concat([m1[(m1.subgroup == "same") & (m1.knob != "all")],
                    m1[(m1.subgroup == "never") & (m1.knob == "all")]])
    nv = nv[nv.arm != "Dviabase"]
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
    x = p[(p.group == "heldout") & (p.arm != "Dviabase")].copy()
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
    t = phase1(t, "depth_table", ["g_version", "group"], "median_abs_err", "median_abs_err")
    t = add_gaps(t, ["g_version", "group"], "median_abs_err", refs=("D", "Dviabase"))
    return order(t, ["g_version", "group"])


def m6_hop1(d):
    dec = d[d.kind.isin(["trained", "heldout"]) & (d.direction == "arm_to_base")
            & d.arm.isin(HOP1_ARMS)].copy()
    dec["group"], dec["source"] = dec.kind, "decoder arm->base (trained record)"
    enc = d[(d.kind == "hop1") & d.arm.isin(HOP1_ARMS)].copy()
    enc["source"] = "encoder NB head (hop1.json)"
    x = pd.concat([dec, enc], ignore_index=True)
    x["knob"] = x.arm_src
    x = pd.concat([x, x.assign(knob="all")], ignore_index=True)
    t = pooled(x, ["arm", "g_version", "knob", "source", "group"])
    t = add_gaps(t, ["g_version", "knob", "group"], "crps_all")   # vs D's decoder record
    return order(t, ["g_version", "group", "knob", "source"])


def load_latent():
    lt = read("latent_fir.tsv.gz")
    lt = numeric(lt, ["seed", "median_abs_z_diff", "median_abs_x_diff", "corr_z_a_b", "ident"])
    lt["seed"] = lt.seed.astype(int)
    return lt


def m7_agree(lt):
    x = lt[(lt.kind == "agree") & lt.arm.isin(DS_ARMS)].copy()
    one = (x.arm_a == "base") | (x.arm_b == "base") | (x.arm_a == x.arm_b)
    x["n_knobs_differ"] = np.where(one, 1, 2)
    x["ratio"] = x.median_abs_z_diff / x.median_abs_x_diff.where(x.median_abs_x_diff > 0)
    k = ["arm", "g_version", "n_knobs_differ"]
    a = per_seed_summary(x, k, "median_abs_z_diff")
    b = per_seed_summary(x, k, "median_abs_x_diff")
    c = per_seed_summary(x, k, "ratio")
    t = a.merge(b.drop(columns=["n_runs", "n_rows", "n_seeds"])
                .rename(columns={"seed_spread": "x_diff_seed_spread"}), on=k)
    t = t.merge(c.drop(columns=["n_runs", "n_rows", "n_seeds"])
                .rename(columns={"ratio": "median_ratio_z_over_x",
                                 "seed_spread": "ratio_seed_spread"}), on=k)
    t.columns = [c.replace("ratio_s", "median_ratio_z_over_x_s") for c in t.columns]
    t["ratio_of_medians"] = t.median_abs_z_diff / t.median_abs_x_diff
    pts = x.groupby(["arm", "g_version", "g", "track", "pid_a", "pid_b", "n_knobs_differ"])[
        ["median_abs_z_diff", "median_abs_x_diff", "corr_z_a_b"]].mean().reset_index()
    return order(t, ["g_version", "n_knobs_differ"]), pts


def m8_swap(d, lt):
    s = d[(d.kind == "swap") & (d.arm != "Dviabase")].copy()
    s["knob"] = s.arm_src
    s = pd.concat([s, s.assign(knob="all")], ignore_index=True)
    k = ["arm", "g_version", "knob"]
    a = per_seed_summary(s, k, "swap_mal")
    a.columns = [c.replace("swap_mal", "swap_median_abs_log_ratio") for c in a.columns]
    i = lt[lt.kind == "ident"].copy()
    tr = i[i.group != "law"].assign(knob=lambda x: knob_of(x))
    lw = i[i.group == "law"]
    i = pd.concat([tr, lw.assign(knob=lw.arm_src),
                   lw[lw.arm_src != lw.arm_tgt].assign(knob=lambda x: x.arm_tgt),
                   i.assign(knob="all")], ignore_index=True)
    b = per_seed_summary(i, k, "ident")
    b.columns = [c.replace("ident", "identity_median_abs_log_ratio") for c in b.columns]
    b = b.rename(columns={"seed_spread": "ident_seed_spread", "n_runs": "ident_n_runs",
                          "n_rows": "ident_n_pairs", "n_seeds": "ident_n_seeds"})
    t = a.merge(b, on=k, how="left")
    t = add_gaps(t, ["g_version", "knob"], "swap_median_abs_log_ratio")
    pa = s[s.knob != "all"].groupby(["arm", "g_version", "seed", "track", "knob"]) \
        .swap_mal.median().rename("swap_median_abs_log_ratio")
    pb = i[i.knob != "all"].groupby(["arm", "g_version", "seed", "track", "knob"]) \
        .ident.median().rename("identity_median_abs_log_ratio")
    pts = pd.concat([pa, pb], axis=1).reset_index()
    return order(t, ["g_version", "knob"]), pts


def m9_steps(runs):
    r = numeric(runs[runs.TRAIN_DONE == 1].copy(),
                ["steps_run", "best_step", "max_steps", "s_per_step", "train_s"])
    r["last10"] = r.best_step >= 0.9 * r.max_steps
    r["at_cap"] = r.steps_run >= r.max_steps
    t = r.groupby(["arm", "g_version"]).agg(
        n_runs=("run", "size"), max_steps=("max_steps", "max"), n_best_last10=("last10", "sum"),
        n_at_cap=("at_cap", "sum"), best_step_median=("best_step", "median"),
        best_step_min=("best_step", "min"), best_step_max=("best_step", "max"),
        steps_run_median=("steps_run", "median"), s_per_step_median=("s_per_step", "median"),
        train_hours_median=("train_s", lambda v: float(v.median() / 3600))).reset_index()
    t["share_best_last10"] = t.n_best_last10 / t.n_runs
    return order(t, ["g_version"])


def rank_table(m1, m2):
    a = m1[(m1.subgroup == "never") & (m1.knob == "all")]
    a = a[["arm", "g_version", "crps_all", "crps_oracle_scaled_all", "scale_error_all",
           "seed_spread", "n_seeds", "n_runs", "gap_vs_D", "label_vs_D", "gap_vs_Dviabase",
           "label_vs_Dviabase"]].rename(columns={"crps_all": "m1_crps_never",
                                                 "crps_oracle_scaled_all": "m1_crps_oracle_scaled_all",
                                                 "scale_error_all": "m1_scale_error_all",
                                                 "seed_spread": "m1_seed_spread",
                                                 "gap_vs_D": "m1_gap_vs_D", "label_vs_D": "m1_label_vs_D",
                                                 "gap_vs_Dviabase": "m1_gap_vs_Dviabase",
                                                 "label_vs_Dviabase": "m1_label_vs_Dviabase"})
    b = m2[m2.knob == "all"][["arm", "g_version", "gen_gap", "gen_gap_crps_oracle_scaled_all",
                              "gen_gap_scale_error_all", "seed_spread", "gap_vs_D", "label_vs_D"]]
    b = b.rename(columns={"gen_gap": "m2_gen_gap",
                          "gen_gap_crps_oracle_scaled_all": "m2_gen_gap_oracle_scaled_all",
                          "gen_gap_scale_error_all": "m2_gen_gap_scale_error_all",
                          "seed_spread": "m2_seed_spread", "gap_vs_D": "m2_gap_vs_D",
                          "label_vs_D": "m2_label_vs_D"})
    t = a.merge(b, on=["arm", "g_version"], how="left")
    out = []
    for gv, s in t.groupby("g_version"):
        s = s.sort_values("m1_crps_never").reset_index(drop=True)
        s.insert(1, "rank", range(1, len(s) + 1))
        nxt = s.shift(-1)
        s["next_arm"] = nxt.arm
        for m in ("m1", "m2"):
            val = "m1_crps_never" if m == "m1" else "m2_gen_gap"
            s[f"{m}_gap_vs_next"] = s[val] - nxt[val]
            s[f"{m}_label_vs_next"] = [
                noise_label(g, sa, sb)[1] if isinstance(na, str) else "n/a"
                for g, sa, sb, na in zip(s[f"{m}_gap_vs_next"], s[f"{m}_seed_spread"],
                                         nxt[f"{m}_seed_spread"], s.next_arm)]
        out.append(s)
    return pd.concat(out, ignore_index=True)


def cclean_table():
    c = read("cclean_fir.tsv")
    c = numeric(c, ["init", "learned", "seed", "knob_i"])
    c["in_g"] = pd.to_numeric(c.in_g).astype(bool)
    c = c[c.in_g].copy()
    c["drift_from_zero"] = c.learned.abs()
    c["drift_from_init"] = (c.learned - c.init).abs()
    k = ["arm", "g_version", "assay", "knob_i", "knob"]
    t = per_seed_summary(c, k, "learned")
    a = c.groupby(k).agg(init=("init", "median"), frozen=("frozen", "max"),
                         drift_from_zero=("drift_from_zero", "median"),
                         drift_from_init=("drift_from_init", "median"),
                         max_drift_from_init=("drift_from_init", "max")).reset_index()
    return order(t.merge(a, on=k), ["g_version", "assay", "knob_i"])


def pair_table(p):
    k = ["arm", "g_version", "group", "knob"] + PAIR
    return p.groupby(k)[list(METRICS)].mean().reset_index()


# ------------------------------------------------------------------------------------- main


def show(title, t, cols):
    print(f"\n{title}")
    print(t[[c for c in cols if c in t]].round(4).to_string(index=False) if len(t) else "(empty)")


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    d = load()
    nt, runs = n_table(d)
    nt.to_csv(HERE / "n_table.tsv", sep="\t", index=False)
    show("n table", nt, list(nt.columns))
    p = label(d)
    G = ["arm", "g_version"]
    m1 = m1_never(p)
    m1.to_csv(HERE / "m1_never_table.tsv", sep="\t", index=False)
    show("metric 1: never-trained pooled CRPS (knob all)", m1[m1.knob == "all"],
         G + ["subgroup", "crps_all", "crps_oracle_scaled_all", "scale_error_all", "seed_spread",
              "n_pairs", "n_runs", "gap_vs_D", "label_vs_D", "gap_vs_Dviabase", "label_vs_Dviabase",
              P1 + "crps_all"])
    m3 = m3_trained(p)
    m2 = m2_gap(m1, m3)
    m2.to_csv(HERE / "m2_gap_table.tsv", sep="\t", index=False)
    show("metric 2: generalisation gap (knob all)", m2[m2.knob == "all"],
         G + ["crps_never", "crps_trained", "gen_gap", "gen_gap_crps_oracle_scaled_all",
              "gen_gap_scale_error_all", "seed_spread", "gap_vs_D", "label_vs_D"])
    m3.to_csv(HERE / "m3_trained_table.tsv", sep="\t", index=False)
    show("metric 3: trained-pair pooled CRPS (knob all)", m3[m3.knob == "all"],
         G + ["crps_all", "crps_oracle_scaled_all", "scale_error_all", "seed_spread", "n_pairs",
              "gap_vs_D", "label_vs_D", P1 + "crps_all"])
    m4 = m4_heldout(p)
    m4.to_csv(HERE / "m4_heldout_table.tsv", sep="\t", index=False)
    show("metric 4: held-out 7.5M CRPS", m4,
         G + ["direction", "crps_all", "crps_oracle_scaled_all", "scale_error_all", "seed_spread",
              "gap_vs_D", "label_vs_D"])
    m5 = m5_depthlaw(d)
    m5.to_csv(HERE / "m5_depthlaw_table.tsv", sep="\t", index=False)
    show("metric 5: depth-law median |log2 error| (sanity check)", m5,
         G + ["group", "median_abs_err", "seed_spread", "n_rows", "gap_vs_D", "label_vs_D",
              P1 + "median_abs_err"])
    m6 = m6_hop1(d)
    m6.to_csv(HERE / "m6_hop1_table.tsv", sep="\t", index=False)
    show("metric 6: hop-1 CRPS (knob all)", m6[m6.knob == "all"],
         G + ["source", "group", "crps_all", "crps_oracle_scaled_all", "scale_error_all",
              "seed_spread", "n_pairs", "gap_vs_D", "label_vs_D"])
    lt = load_latent()
    m7, m7p = m7_agree(lt)
    m7.to_csv(HERE / "m7_agree_table.tsv", sep="\t", index=False)
    m7p.to_csv(HERE / "m7_agree_points.tsv.gz", sep="\t", index=False)
    show("metric 7: middle consistency", m7,
         G + ["n_knobs_differ", "median_abs_z_diff", "median_abs_x_diff", "median_ratio_z_over_x",
              "seed_spread", "n_rows"])
    m8, m8p = m8_swap(d, lt)
    m8.to_csv(HERE / "m8_swap_table.tsv", sep="\t", index=False)
    m8p.to_csv(HERE / "m8_swap_points.tsv.gz", sep="\t", index=False)
    show("metric 8: swap vs identity (knob all)", m8[m8.knob == "all"],
         G + ["swap_median_abs_log_ratio", "seed_spread", "identity_median_abs_log_ratio",
              "ident_seed_spread", "gap_vs_D", "label_vs_D"])
    m9 = m9_steps(runs)
    m9.to_csv(HERE / "m9_steps_table.tsv", sep="\t", index=False)
    show("metric 9: best step in the last 10 %", m9, list(m9.columns))
    rk = rank_table(m1, m2)
    rk.to_csv(HERE / "rank_table.tsv", sep="\t", index=False)
    show("rank by metric 1 (never, knob all), metric 2 beside", rk,
         ["g_version", "rank", "arm", "m1_crps_never", "m1_crps_oracle_scaled_all",
          "m1_scale_error_all", "m1_seed_spread", "m1_label_vs_next", "m2_gen_gap",
          "m2_seed_spread", "m1_label_vs_D", "m1_label_vs_Dviabase"])
    cc = cclean_table()
    cc.to_csv(HERE / "cclean_table.tsv", sep="\t", index=False)
    show("C_clean drift (median over knobs)", cc.groupby(G)[
        ["drift_from_zero", "drift_from_init", "max_drift_from_init"]].median().reset_index(),
         G + ["drift_from_zero", "drift_from_init", "max_drift_from_init"])
    pair_table(p).to_csv(HERE / "pair_table.tsv.gz", sep="\t", index=False)


if __name__ == "__main__":
    main()
