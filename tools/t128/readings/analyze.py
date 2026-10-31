"""t128 readings (plan/T126_ENCDEC_D2.md "Readings"): tables from the flattened per-run records.

Inputs (beside this file, brought down by run_all.sh; small tables only):
  records_{fir,nibi}.tsv.gz, latent_{fir,nibi}.tsv.gz, cclean_{fir,nibi}.tsv, runs_{fir,nibi}.tsv
  (extract_records.py on each cluster, only the runs that cluster's index list assigns to it),
  viabase_nibi.tsv.gz (old D through the base, arm Dviabase), and records_D_t126.tsv.gz (a copy
  of t126's records_D.tsv.gz; arm D = old rung D, real and its no-covariates twin; no recompute).
  oldd_split_nibi.tsv.gz (old D's law + depthlaw records re-scored with the CRPS split), when
  present, replaces the matching (run, kind, depth_source) rows of records_D_t126.tsv.gz.
Outputs (beside this file):
  n_table.tsv       runs per arm x model: assigned, TRAIN_DONE, LAW_DONE, LATENT_DONE, by cluster
  depth_table.tsv   reading 1: median |log2 error| per arm x g_version x group
  knob_table.tsv    reading 2: pooled CRPS + oracle_scaled + scale_error per arm x gv x group x knob
  twin_table.tsv    reading 3: gain over the no-covariates twin, settings 1 and 3 and old D
  collapse_table.tsv reading 4: median |Z - x| and corr(Z, x) per arm x gv x product knob
  agree_table.tsv   reading 5: median |Z - Z'| against median |x - x'| (one knob / two knobs)
  swap_table.tsv    reading 6: swap statistic against the identity distance per arm x gv x knob
  lambda_table.tsv, lambda_per_g.tsv, lambda_chosen.json: the lambda rule over settings 5-7
  fit_table.tsv, steps_table.tsv, cclean_table.tsv, pair_table.tsv.gz, agree_points.tsv.gz,
  swap_points.tsv.gz, depth_points.tsv.gz

Conventions (t126 analyze.py, restated):
  arm = run-name prefix; counts only. groups: trained = kind trained (knob = arm_tgt for
  base_to_arm, else arm_src); heldout = kind heldout (base <-> depth 7.5M, never trained);
  same = law pair with arm_src == arm_tgt; cross = law pair with arm_src != arm_tgt; never = same
  + cross. knob "all" pools every knob of a group.
  pooled CRPS (t118): per seed, mean over a track's pairs, then mean over tracks; the reading is
  the mean over seeds. Depth law: err = log2_pred - log2_true of depthlaw records, real model;
  median |err| over records (all seeds); groups trained / heldout / never (law, no 7.5M) /
  never75 (law, touches 7.5M). Gain over the twin, per pair: (CRPS_twin - CRPS_real) / CRPS_twin
  on seed-mean CRPS; the reading is the median over pairs.
Noise rule (spec): seed_spread = the largest pairwise |difference| over the seeds of a per-seed
  value (= max - min). A gap between two cells counts only if |gap| > 2 x max(spread_a,
  spread_b); otherwise its label is "within noise". Cells with fewer than 2 seeds have no spread
  and the label "no spread (n_seeds < 2)".
Selection (lambda): chr22 only: val_nll = the run's best validation NLL under its own rule
  (timing.json; verify_raw.py reads TRAIN_DONE instead). Per g version, lowest mean over runs.

    /Users/mforooz/miniforge3/envs/candi-local/bin/python analyze.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SETTINGS = ["DS", "DSbox", "DSrt", "DSrtvae", "DSl0.1", "DSl1", "DSl10", "DSvae"]
LAMS = {"DSl0.1": 0.1, "DSl1": 1.0, "DSl10": 10.0}
REFS = ["D", "DS"]
TWIN_ARMS = ["DS", "DSrt", "D"]
CLUSTERS = ["fir", "nibi"]
N_RUNS_FULL = {"per_track": 21, "across": 3}
PAIR = ["track", "arm_src", "level_src", "arm_tgt", "level_tgt"]
NUM = ["crps_all", "crps_oracle_scaled_all", "scale_error_all", "log2_true", "log2_pred",
       "val_nll", "swap_mal", "steps_run", "seed", "best_step"]
EDGE = 1.0 / (1.0 + np.exp(4.0))          # sigmoid(-4): the form-D box edge (PI ruling vi)


def read_many(stem, ext=".tsv.gz"):
    fs = [HERE / f"{stem}_{c}{ext}" for c in CLUSTERS if (HERE / f"{stem}_{c}{ext}").is_file()]
    if not fs:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f, sep="\t", low_memory=False) for f in fs], ignore_index=True)


def numeric(d, cols):
    for c in cols:
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def load():
    d = read_many("records")
    v = read_many("viabase")
    old = pd.read_csv(HERE / "records_D_t126.tsv.gz", sep="\t", low_memory=False)
    old["arm"] = old.run.str.split("_").str[0]
    old = old[old.arm == "D"].copy()
    old["cluster"] = "nibi (t126)"
    o = read_many("oldd_split")
    if len(o):                    # old D's law rows re-scored with the split replace t126's
        k = ["run", "kind", "depth_source"]
        o["cluster"] = "nibi (t128 oldD_split)"
        hit = old[k].fillna("").apply(tuple, axis=1).isin(set(o[k].fillna("").apply(tuple, axis=1)))
        print(f"old D: {o.run.nunique()} runs re-scored with the split replace {int(hit.sum())} "
              f"t126 rows with {len(o)}")
        old = pd.concat([old[~hit], o.assign(arm="D")], ignore_index=True)
    d = pd.concat([d, v.assign(arm="Dviabase"), old], ignore_index=True)
    d = numeric(d, NUM)
    d["seed"] = d.seed.astype(int)
    assert d.space.eq("counts").all()
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


def add_gaps(t, keys, value, sp="seed_spread", refs=REFS, on="arm"):
    """For every row, the gap to the same cell of each reference arm, with the noise rule."""
    idx = t.set_index([on] + keys)
    for ref in refs:
        g, sa, sb, cn, lb = [], [], [], [], []
        for _, r in t.iterrows():
            k = (ref,) + tuple(r[c] for c in keys)
            if r[on] == ref or k not in idx.index:
                g.append(np.nan); sa.append(np.nan); sb.append(np.nan); cn.append(np.nan)
                lb.append("n/a")
                continue
            ref_row = idx.loc[k]
            if isinstance(ref_row, pd.DataFrame):
                ref_row = ref_row.iloc[0]
            gap = r[value] - ref_row[value]
            c, l = noise_label(gap, r[sp], ref_row[sp])
            g.append(gap); sa.append(r[sp]); sb.append(ref_row[sp]); cn.append(c); lb.append(l)
        t[f"gap_vs_{ref}"] = g
        t[f"spread_a_vs_{ref}"] = sa
        t[f"spread_b_vs_{ref}"] = sb
        t[f"counts_vs_{ref}"] = cn
        t[f"label_vs_{ref}"] = lb
    return t


def per_seed_summary(x, keys, value, how="median"):
    """median (or mean) over all rows, the per-seed values, their spread, n_seeds."""
    f = (lambda s: float(np.nanmedian(s))) if how == "median" else (lambda s: float(np.nanmean(s)))
    out = []
    for k, s in x.groupby(keys, dropna=False):
        ps = s.groupby("seed")[value].apply(f)
        rec = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        rec.update({value: f(s[value]), "seed_spread": spread(ps), "n_seeds": int(ps.notna().sum()),
                    "n_runs": s.run.nunique(), "n_rows": int(s[value].notna().sum())})
        for sd, v in ps.items():
            rec[f"seed{int(sd)}"] = v
        out.append(rec)
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------- n table


def n_table(d):
    r = read_many("runs", ".tsv")
    r["model"] = r.run.str.split("_").str[3]
    for m in ("TRAIN_DONE", "SCORE_DONE", "LAW_DONE", "LATENT_DONE"):
        r[m] = pd.to_numeric(r[m], errors="coerce").fillna(0).astype(int)
    t = r.groupby(["arm", "model", "cluster"]).agg(
        assigned=("run", "size"), TRAIN_DONE=("TRAIN_DONE", "sum"), SCORE_DONE=("SCORE_DONE", "sum"),
        LAW_DONE=("LAW_DONE", "sum"), LATENT_DONE=("LATENT_DONE", "sum")).reset_index()
    tot = t.groupby(["arm", "model"]).sum(numeric_only=True).reset_index().assign(cluster="both")
    t = pd.concat([t, tot], ignore_index=True)
    for arm, lab in (("D", "t126 records_D"), ("Dviabase", "viabase LAW_DONE")):
        s = d[d.arm == arm].drop_duplicates("run")
        for m, n in s.groupby("model").size().items():
            t.loc[len(t)] = {"arm": arm, "model": m, "cluster": lab, "assigned": n, "LAW_DONE": n}
    return t, r


# ------------------------------------------------------------------------------------- reading 1


def depth_group(x):
    touches = (x.level_src == "7.5M") | (x.level_tgt == "7.5M")
    law = x.depth_source == "law"
    return np.select([law & ~touches, law & touches, touches], ["never", "never75", "heldout"],
                     "trained")


def depth_table(d):
    x = d[(d.kind == "depthlaw") & (d.model == "real")].copy()
    x["group"] = depth_group(x)
    x["err"] = x.log2_pred - x.log2_true
    x["abs_err"] = x.err.abs()
    t = per_seed_summary(x, ["arm", "g_version", "group"], "abs_err")
    t = t.rename(columns={"abs_err": "median_abs_err"})
    q = x.groupby(["arm", "g_version", "group"]).agg(
        q25_abs_err=("abs_err", lambda s: s.quantile(0.25)),
        q75_abs_err=("abs_err", lambda s: s.quantile(0.75)),
        median_err=("err", "median"), n_none=("err", lambda s: int(s.isna().sum()))).reset_index()
    t = t.merge(q, on=["arm", "g_version", "group"])
    sl = []
    for k, s in x.groupby(["arm", "g_version", "group"]):
        ok = s.log2_pred.notna()
        sl.append(dict(zip(["arm", "g_version", "group"], k),
                       slope=float(np.polyfit(s.log2_true[ok], s.log2_pred[ok], 1)[0])
                       if ok.sum() > 2 and s.log2_true[ok].nunique() > 1 else np.nan))
    t = t.merge(pd.DataFrame(sl), on=["arm", "g_version", "group"])
    t = add_gaps(t, ["g_version", "group"], "median_abs_err")
    return t, x


# ------------------------------------------------------------------------------------- reading 2


def label(d):
    t = d[d.kind.isin(["trained", "heldout"])].copy()
    t["group"] = t.kind
    t["knob"] = np.where(t.direction == "base_to_arm", t.arm_tgt, t.arm_src)
    law = d[d.kind == "law"].copy()
    law["group"] = np.where(law.arm_src == law.arm_tgt, "same", "cross")
    law["knob"] = np.where(law.group == "same", law.arm_src, "cross")
    p = pd.concat([t, law], ignore_index=True)
    allk = p.copy()
    allk["knob"] = "all"
    nev = p[p.group.isin(["same", "cross"])].copy()
    nev["group"], nev["knob"] = "never", "all"
    return pd.concat([p, allk, nev], ignore_index=True)


METRICS = ("crps_all", "crps_oracle_scaled_all", "scale_error_all")


def pooled(p, keys):
    """t118 pooling: per (keys, seed): mean over a track's pairs, then mean over tracks."""
    tr = p.groupby(keys + ["seed", "track"])[list(METRICS)].mean()
    tr["n_pairs"] = p.groupby(keys + ["seed", "track"]).size()
    ps = tr.groupby(keys + ["seed"]).agg(
        **{m: (m, "mean") for m in METRICS}, n_tracks=("n_pairs", "size"),
        n_pairs=("n_pairs", "sum")).reset_index()
    out = []
    for k, s in ps.groupby(keys):
        rec = dict(zip(keys, k))
        for m in METRICS:
            rec[m] = float(s[m].mean()) if s[m].notna().any() else np.nan
        rec["seed_spread"] = spread(s.crps_all)
        rec["n_seeds"] = len(s)
        rec["n_tracks_per_seed"] = ",".join(str(int(v)) for v in s.n_tracks)
        rec["n_pair_records"] = int(s.n_pairs.sum())
        for _, r in s.iterrows():
            rec[f"seed{int(r.seed)}"] = r.crps_all
        out.append(rec)
    t = pd.DataFrame(out)
    nr = p.groupby(keys).run.nunique().rename("n_runs").reset_index()
    return t.merge(nr, on=keys)


def knob_table(p):
    x = p[p.model == "real"]
    t = pooled(x, ["arm", "g_version", "group", "knob"])
    t["split_available"] = t.crps_oracle_scaled_all.notna()
    return add_gaps(t, ["g_version", "group", "knob"], "crps_all")


def fit_table(p):
    x = p[(p.group == "trained") & (p.knob == "all")]
    t = pooled(x, ["arm", "g_version", "model"])
    pm = x.groupby(["arm", "g_version", "model"] + PAIR).crps_all.mean()
    t = t.merge(pm.groupby(["arm", "g_version", "model"]).median().rename("crps_median_pair")
                .reset_index(), how="left")
    r = t[t.model == "real"].copy()
    r = add_gaps(r, ["g_version"], "crps_all")
    return pd.concat([r, t[t.model != "real"]], ignore_index=True)


# ------------------------------------------------------------------------------------- reading 3


def pair_level(p):
    k = ["arm", "g_version", "group", "knob"] + PAIR
    w = p.groupby(k + ["model"], dropna=False).crps_all.mean().unstack("model").reset_index()
    if "nocov" in w:
        w["rel_nocov"] = (w["nocov"] - w["real"]) / w["nocov"].abs()
    return w


def twin_table(p):
    x = p[p.arm.isin(TWIN_ARMS)]
    k = ["arm", "g_version", "group", "knob"]
    w = pair_level(x)
    w = w[w.real.notna() & w.nocov.notna()]
    ps = x.groupby(k + PAIR + ["seed", "model"]).crps_all.mean().unstack("model").reset_index()
    ps = ps[ps.real.notna() & ps.nocov.notna()]
    ps["rel"] = (ps.nocov - ps.real) / ps.nocov.abs()
    seed_med = ps.groupby(k + ["seed"]).rel.median().reset_index()
    by_seed = {kk: s.set_index("seed").rel for kk, s in seed_med.groupby(k)}
    out = []
    for kk, s in w.groupby(k):
        sv = by_seed.get(kk, pd.Series(dtype=float))
        rec = dict(zip(k, kk))
        rec.update(gain_median=float(s.rel_nocov.median()), win_frac=float((s.real < s.nocov).mean()),
                   n_pairs=len(s), seed_spread=spread(sv), n_seeds=int(sv.notna().sum()))
        for sd, v in sv.items():
            rec[f"seed{int(sd)}"] = v
        out.append(rec)
    t = pd.DataFrame(out)
    runs = x.groupby(["arm", "g_version", "model"]).run.nunique().unstack("model")
    t = t.merge(runs.rename(columns={"real": "n_runs_real", "nocov": "n_runs_nocov"}).reset_index(),
                on=["arm", "g_version"], how="left")
    # trained versus never-trained: each row's gap to the trained row of the same knob (or "all")
    tr = t[t.group == "trained"].set_index(["arm", "g_version", "knob"])
    g, cn, lb, sb = [], [], [], []
    for _, r in t.iterrows():
        kn = r.knob if (r.arm, r.g_version, r.knob) in tr.index else "all"
        key = (r.arm, r.g_version, kn)
        if r.group == "trained" or key not in tr.index:
            g.append(np.nan); cn.append(np.nan); lb.append("n/a"); sb.append(np.nan)
            continue
        ref = tr.loc[key]
        gap = r.gain_median - ref.gain_median
        c, l = noise_label(gap, r.seed_spread, ref.seed_spread)
        g.append(gap); cn.append(c); lb.append(l); sb.append(ref.seed_spread)
    t["gap_vs_trained"], t["spread_trained"], t["counts_vs_trained"], t["label_vs_trained"] = g, sb, cn, lb
    t = add_gaps(t, ["g_version", "group", "knob"], "gain_median", refs=["D", "DS"])
    return t, w


# ------------------------------------------------------------------------------------- readings 4-6


def load_latent():
    lt = read_many("latent")
    lt = numeric(lt, ["seed", "median_abs_z_minus_x", "corr_z_x", "median_sigma_z",
                      "median_abs_z_diff", "median_abs_x_diff", "corr_z_a_b", "ident"])
    lt["seed"] = lt.seed.astype(int)
    return lt


def collapse_table(lt):
    x = lt[lt.kind == "latent"].copy()
    x["knob"] = x.arm_p
    x = pd.concat([x, x.assign(knob="all")], ignore_index=True)
    k = ["arm", "g_version", "model", "knob"]
    a = per_seed_summary(x, k, "median_abs_z_minus_x")
    b = per_seed_summary(x, k, "corr_z_x")
    c = x.groupby(k).median_sigma_z.median().rename("median_sigma_z").reset_index()
    t = a.merge(b[k + ["corr_z_x", "seed_spread"]].rename(columns={"seed_spread": "corr_seed_spread"}),
                on=k).merge(c, on=k)
    r = add_gaps(t[t.model == "real"].copy(), ["g_version", "knob"], "median_abs_z_minus_x",
                 refs=["DS"])
    return pd.concat([r, t[t.model != "real"]], ignore_index=True)


def agree_table(lt):
    x = lt[lt.kind == "agree"].copy()
    one = (x.arm_a == "base") | (x.arm_b == "base") | (x.arm_a == x.arm_b)
    x["n_knobs"] = np.where(one, "one", "two")
    x["ratio"] = x.median_abs_z_diff / x.median_abs_x_diff.where(x.median_abs_x_diff > 0)
    x = pd.concat([x, x.assign(n_knobs="all")], ignore_index=True)
    k = ["arm", "g_version", "model", "n_knobs"]
    a = per_seed_summary(x, k, "median_abs_z_diff")
    b = per_seed_summary(x, k, "median_abs_x_diff")
    c = per_seed_summary(x, k, "ratio")
    t = a.merge(b[k + ["median_abs_x_diff"]], on=k).merge(
        c[k + ["ratio", "seed_spread"]].rename(columns={"ratio": "median_ratio_z_over_x",
                                                        "seed_spread": "ratio_seed_spread"}), on=k)
    r = add_gaps(t[t.model == "real"].copy(), ["g_version", "n_knobs"], "median_abs_z_diff",
                 refs=["DS"])
    pts = x[x.n_knobs != "all"].groupby(["arm", "g_version", "model", "track", "pid_a", "pid_b",
                                         "n_knobs"])[["median_abs_z_diff", "median_abs_x_diff",
                                                      "corr_z_a_b"]].mean().reset_index()
    return pd.concat([r, t[t.model != "real"]], ignore_index=True), pts


def swap_table(d, lt):
    s = d[d.kind == "swap"].copy()
    s["knob"] = s.arm_src
    s = pd.concat([s, s.assign(knob="all")], ignore_index=True)
    k = ["arm", "g_version", "model", "knob"]
    a = per_seed_summary(s, k, "swap_mal").rename(columns={"swap_mal": "swap_median_abs_log_ratio"})
    i = lt[lt.kind == "ident"].copy()
    i["knob"] = np.where(i.group == "law",
                         np.where(i.arm_src == i.arm_tgt, i.arm_src, "cross"),
                         np.where(i.direction == "base_to_arm", i.arm_tgt, i.arm_src))
    i = pd.concat([i, i.assign(knob="all")], ignore_index=True)
    b = per_seed_summary(i, k, "ident").rename(columns={"ident": "identity_median_abs_log_ratio",
                                                        "seed_spread": "ident_seed_spread",
                                                        "n_runs": "ident_n_runs",
                                                        "n_rows": "ident_n_pairs"})
    keep = k + ["identity_median_abs_log_ratio", "ident_seed_spread", "ident_n_runs", "ident_n_pairs"]
    t = a.merge(b[keep], on=k, how="outer")
    for c in ("n_runs", "n_seeds", "n_rows", "ident_n_runs", "ident_n_pairs"):
        t[c] = t[c].astype("Int64")
    r = add_gaps(t[t.model == "real"].copy(), ["g_version", "knob"], "swap_median_abs_log_ratio",
                 refs=["DS"])
    pa = s[s.knob != "all"].groupby(["arm", "g_version", "model", "seed", "track", "knob"]) \
        .swap_mal.median().rename("swap_median_abs_log_ratio")
    pb = i[i.knob != "all"].groupby(["arm", "g_version", "model", "seed", "track", "knob"]) \
        .ident.median().rename("identity_median_abs_log_ratio")
    pts = pd.concat([pa, pb], axis=1).reset_index()
    return pd.concat([r, t[t.model != "real"]], ignore_index=True), pts


# ------------------------------------------------------------------------------------- lambda


def lambda_table(runs):
    r = runs[runs.arm.isin(LAMS) & (runs.model == "real") & (runs.TRAIN_DONE == 1)].copy()
    r = numeric(r, ["val_nll", "best_step", "seed"])
    r["g_version"] = np.where(r.g == "all", "across", "per_track")
    per_g = r.groupby(["arm", "g_version", "g"]).agg(val_nll_mean=("val_nll", "mean"),
                                                     n_seeds=("val_nll", "size")).reset_index()
    out = []
    for (arm, gv), s in r.groupby(["arm", "g_version"]):
        ps = s.groupby("seed").val_nll.mean()
        out.append(dict(arm=arm, g_version=gv, lam=LAMS[arm], val_nll_mean=float(s.val_nll.mean()),
                        n_runs=len(s), n_runs_full=N_RUNS_FULL[gv], seed_spread=spread(ps),
                        best_step_median=float(s.best_step.median()),
                        **{f"seed{int(k)}": v for k, v in ps.items()}))
    t = pd.DataFrame(out, columns=["arm", "g_version", "lam", "val_nll_mean", "n_runs",
                                   "n_runs_full", "seed_spread", "best_step_median"]
                     if not out else None)
    t["chosen"] = False
    chosen = {}
    for gv, s in t.groupby("g_version"):
        s = s.sort_values("val_nll_mean")
        best = s.iloc[0]
        t.loc[best.name, "chosen"] = True
        rec = {"arm": best.arm, "lam": best.lam, "val_nll_mean": best.val_nll_mean,
               "n_runs": {a: int(n) for a, n in zip(s.arm, s.n_runs)},
               "complete": bool(len(s) == 3 and (s.n_runs == N_RUNS_FULL[gv]).all())}
        if len(s) > 1:
            ru = s.iloc[1]
            gap = ru.val_nll_mean - best.val_nll_mean
            c, l = noise_label(gap, best.seed_spread, ru.seed_spread)
            rec.update(runner_up=ru.arm, gap_to_runner_up=gap, spread_a=best.seed_spread,
                       spread_b=ru.seed_spread, counts=c, label=l)
        chosen[gv] = rec
    if t.empty:
        return t, per_g, chosen
    nw = per_g.loc[per_g.groupby(["g_version", "g"]).val_nll_mean.idxmin()]
    t = t.merge(nw.groupby(["g_version", "arm"]).size().rename("n_g_lowest").reset_index(),
                how="left").fillna({"n_g_lowest": 0})
    return t, per_g, chosen


def steps_table(runs):
    r = runs[runs.TRAIN_DONE == 1].copy()
    r = numeric(r, ["steps_run", "best_step", "train_s"])
    return r.groupby(["arm", "model"]).agg(
        n_runs=("run", "size"), n_at_cap_4000=("steps_run", lambda v: int((v >= 4000).sum())),
        steps_median=("steps_run", "median"), best_step_median=("best_step", "median"),
        train_hours_median=("train_s", lambda v: float(v.median() / 3600))).reset_index()


# ------------------------------------------------------------------------------------- C_clean


def cclean_table():
    c = read_many("cclean", ".tsv")
    c = numeric(c, ["init", "learned", "lo", "hi", "seed", "knob_i"])
    c["in_g"] = pd.to_numeric(c.in_g).astype(bool)
    c = c[c.in_g].copy()
    box = c[c.arm == "DSbox"].groupby(["g", "assay", "knob_i"]).agg(
        base=("init", "first"), box_lo=("lo", "first"), box_hi=("hi", "first")).reset_index()
    c = c.merge(box, on=["g", "assay", "knob_i"], how="left")
    c["drift_from_init"] = (c.learned - c.init).abs()
    c["drift_from_base"] = (c.learned - c.base).abs()
    w = c.box_hi - c.box_lo
    c["outside_box"] = ((c.learned < c.box_lo - 1e-6) | (c.learned > c.box_hi + 1e-6)).where(c.base.notna())
    c["at_box_edge"] = ((c.learned < c.box_lo + EDGE * w) | (c.learned > c.box_hi - EDGE * w)) \
        .where(w > 1e-9)
    k = ["arm", "g_version", "model", "assay", "knob_i", "knob"]
    t = per_seed_summary(c, k, "learned")
    a = c.groupby(k).agg(init=("init", "median"), base=("base", "first"), box_lo=("box_lo", "first"),
                         box_hi=("box_hi", "first"), drift_from_init=("drift_from_init", "median"),
                         drift_from_base=("drift_from_base", "median"),
                         frac_outside_box=("outside_box", "mean"),
                         frac_at_box_edge=("at_box_edge", "mean")).reset_index()
    return t.merge(a, on=k)


# ------------------------------------------------------------------------------------- main


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    d = load()
    nt, runs = n_table(d)
    nt.to_csv(HERE / "n_table.tsv", sep="\t", index=False)
    print(nt[nt.cluster.isin(["both", "t126 records_D", "viabase LAW_DONE"])].to_string(index=False))

    dt, pts = depth_table(d)
    dt.to_csv(HERE / "depth_table.tsv", sep="\t", index=False)
    pts[["arm", "g_version", "g", "seed", "group"] + PAIR + ["log2_true", "log2_pred", "err"]] \
        .to_csv(HERE / "depth_points.tsv.gz", sep="\t", index=False)
    print("\nreading 1 depth median |log2 error|")
    print(dt.pivot_table(index="arm", columns=["g_version", "group"], values="median_abs_err")
          .round(3).to_string())

    p = label(d)
    kt = knob_table(p)
    kt.to_csv(HERE / "knob_table.tsv", sep="\t", index=False)
    print("\nreading 2 never-trained pooled CRPS (knob all)")
    print(kt[(kt.knob == "all") & kt.group.isin(["same", "cross", "never"])]
          [["arm", "g_version", "group", "crps_all", "crps_oracle_scaled_all", "scale_error_all",
            "seed_spread", "n_runs", "gap_vs_D", "label_vs_D", "gap_vs_DS", "label_vs_DS"]]
          .round(4).to_string(index=False))
    ft = fit_table(p)
    ft.to_csv(HERE / "fit_table.tsv", sep="\t", index=False)

    tw, pl = twin_table(p)
    tw.to_csv(HERE / "twin_table.tsv", sep="\t", index=False)
    pair_level(p).to_csv(HERE / "pair_table.tsv.gz", sep="\t", index=False)
    print("\nreading 3 gain over the twin (knob all)")
    print(tw[tw.knob == "all"][["arm", "g_version", "group", "gain_median", "seed_spread", "n_pairs",
                                "n_runs_real", "n_runs_nocov", "gap_vs_trained", "label_vs_trained"]]
          .round(4).to_string(index=False))

    lt = load_latent()
    ct = collapse_table(lt)
    ct.to_csv(HERE / "collapse_table.tsv", sep="\t", index=False)
    print("\nreading 4 collapse (knob all)")
    print(ct[ct.knob == "all"][["arm", "g_version", "model", "median_abs_z_minus_x", "seed_spread",
                                "corr_z_x", "median_sigma_z", "n_runs", "label_vs_DS"]]
          .round(4).to_string(index=False))
    at, apts = agree_table(lt)
    at.to_csv(HERE / "agree_table.tsv", sep="\t", index=False)
    apts.to_csv(HERE / "agree_points.tsv.gz", sep="\t", index=False)
    print("\nreading 5 agreement")
    print(at[["arm", "g_version", "model", "n_knobs", "median_abs_z_diff", "median_abs_x_diff",
              "median_ratio_z_over_x", "seed_spread", "n_runs"]].round(4).to_string(index=False))
    st, spts = swap_table(d, lt)
    st.to_csv(HERE / "swap_table.tsv", sep="\t", index=False)
    spts.to_csv(HERE / "swap_points.tsv.gz", sep="\t", index=False)
    print("\nreading 6 swap vs identity (knob all)")
    print(st[st.knob == "all"][["arm", "g_version", "model", "swap_median_abs_log_ratio", "seed_spread",
                                "identity_median_abs_log_ratio", "ident_seed_spread", "n_runs"]]
          .round(4).to_string(index=False))

    lam, per_g, chosen = lambda_table(runs)
    lam.to_csv(HERE / "lambda_table.tsv", sep="\t", index=False)
    per_g.to_csv(HERE / "lambda_per_g.tsv", sep="\t", index=False)
    (HERE / "lambda_chosen.json").write_text(json.dumps(chosen, indent=1, default=str) + "\n")
    print("\nlambda"); print(lam.round(4).to_string(index=False)); print(json.dumps(chosen, default=str))
    stp = steps_table(runs)
    stp.to_csv(HERE / "steps_table.tsv", sep="\t", index=False)
    print("\nsteps"); print(stp.to_string(index=False))
    cc = cclean_table()
    cc.to_csv(HERE / "cclean_table.tsv", sep="\t", index=False)
    print("\nC_clean drift (median over knobs of the per-cell medians)")
    print(cc[cc.model == "real"].groupby(["arm", "g_version"])[
        ["drift_from_init", "drift_from_base", "frac_outside_box", "frac_at_box_edge"]].median()
          .round(4).to_string())


if __name__ == "__main__":
    main()
