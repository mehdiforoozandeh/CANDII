"""t130 blind check-set readings, step 2 (laptop, pandas): the blind table of the h1 (depth) and
h2 (run type, arm `pe`) check set, from the extract and the t137 floor.

READINGS ONLY. Nothing here ticks a box or gives a verdict. Every threshold is the PI's
(`TODO(PI)` in checks_readings.tsv); the model rung for any verdict is the PI's pick (h1 item vi).

Inputs in <results_dir> (run_all.sh copies them in):
  records.tsv.gz, baselines.tsv.gz   extract_records.py
  floor_pairs.tsv, floor_curve.tsv, floor_p15M.tsv   t137 (tools/t137/floor.py aggregate)
  nested_oracle.tsv                  tools/t130/nested_oracle.py (absent -> every pair falls back
                                     to the floor as its oracle, oracle_kind `floor_fallback`, with
                                     a line in BLIND.md; verify_raw.py does the same)

THE ORACLE (PI ruling 2026-10-30, plan_gaps.md A9). gap-closed runs one-warp -> d_oracle:
  depth pairs, counts   the NESTED-READ oracle: a depth arm is a read subsample of its base, so the
                        base and the arm share reads and an independent-replicate floor is no upper
                        bound there. base_to_arm: `binomial` (Binomial(base, rho)); arm_to_base:
                        `eb_binomial` (the exact posterior of the base given the arm under binomial
                        thinning, prior = the base's training-chromosome histogram); rho from the
                        MANIFEST depth (ORACLE_R). oracle_kind `nested_binomial` / `nested_eb_binomial`.
  every other pair      the floor, as before (oracle_kind `floor`): h2's pe arm is a separate FASTQ
                        reprocessing, not a subsample, and -log10 p cannot be thinned.
  d_floor stays in every row as CONTEXT: what a fresh replicate would differ by.

Per pair (space, arm product pid, direction):
  d_onewarp     the pair's QuantileMatching record (rungs_v2), Poisson / log-normal spread CRPS.
  d_nosolution  the pair's noSolution record.
  d_oracle      the gap top (THE ORACLE above), with its split oracle_crps_oracle_scaled_all /
                oracle_scale_error_all and oracle_kind.
  d_floor       context: what a fresh replicate would differ by. The floor at the TARGET's real depth X (pinned rule, plan_gaps.md A3). X = the
                target product's total depth (products MANIFEST `depth`): a depth arm -> its level
                (15M / 7.5M / 3.75M); the DNase base C12M02__base__base -> 50M; every other base
                and every pe target -> 30M. counts: the SAME TRACK's BASE product's t137 curve
                (floor_curve.tsv, one row per product x metric) read at reads_per_half = X:
                value_<X> for X in 15M / 7.5M / 3.75M / 1.875M, value_30M_extrap for X = 30M, else
                exp(alpha + beta * log2 X). floor_extrap = 1 when X > the base's reads_per_half_full
                (its n_lines: beyond its measured ladder), else 0. The split (oracle_scaled /
                scale_error) is not fitted: it is the base product's floor_pairs split (mean over
                seeds and directions) at the ladder ratio nearest X / n_lines in log2 (clamped to
                1 .. 0.125), labelled split_depth = that ratio's reads per half.
                pval: no thinning; the only p floor is the base product's floor_p15M row (the
                rungs_v2 oracle between its own halves), at reads_per_half = the base's n_lines.
                floor_depth carries THAT depth, and p_floor_matched = 1 only when X equals it
                (a 30M target is NOT matched: p_floor_matched 0).
  floor_seed_spread  counts: the base product's seed_spread_r<k> (floor_curve) at k = X / n_lines
                rounded to the nearest of 0.5 / 0.25 / 0.125 in log2; k >= 0.5 -> r0.5. NA in pval.
  Cell level: floor_depth / split_depth = the distinct labels of the cell's pairs, comma-joined;
  floor_extrap = 1 if any pair's floor is extrapolated; p_floor_matched = 1 only if every pair is
  matched (NA in counts).
Pooling (the t118 rule), the SAME for every column, so the baselines and the floor are weighted
exactly as the model is: per run the mean over its check-set pairs in the cell -> per seed the mean
over runs -> the mean over seeds. Baseline and floor values ride on the real model's trained
records. seed_abs_delta = the largest pairwise |difference| over the per-seed d_model values.
A missing value propagates as NA (never skipped).

blind_table.tsv: one row per space x knob(depth|pe) x mark_class(+all) x rung x g_version x
direction(base_to_arm|arm_to_base|both) x subset(all|top1):
  d_model d_onewarp d_nocov d_ids d_nosolution d_floor  pooled CRPS (subset `subset`)
  gapclosed_onewarp = (d_onewarp - d_model) / (d_onewarp - d_oracle)
  gain_over_nocov   = d_nocov - d_model
  gain_gt_2seed     = (d_onewarp - d_model) > 2 x seed_abs_delta   (the h1 PI ruling's gain)
  crps_oracle_scaled_all, scale_error_all   the model's split (subset all, on every row)
  d_shuffle, gapclosed_shuffle = (d_onewarp - d_shuffle) / (d_onewarp - d_oracle)   t118's
                    shuffle (C' from another arm of the same track, every knob changes)
  swap_ratio_median  median over the real model's swap records of the cell's arm products of
                    median |log(pred / X)| (direction- and subset-free)
  depthlaw_median_abs_log2err  median |log2 pred scale - log2 true depth ratio| over the real
                    model's depthlaw records (counts, depth only)
  then, appended: gain_over_onewarp, gain_over_nocov_gt_2seed, onewarp_oracle_scaled_all,
  onewarp_scale_error_all, floor_oracle_scaled_all, floor_scale_error_all, split_depth,
  floor_extrap, p_floor_matched, d_oracle, oracle_kind, oracle_crps_oracle_scaled_all,
  oracle_scale_error_all.

    python analyze.py <results_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

#: plan_gaps.md "Mark classes" (knob_effects.MARK_CLASS by assay)
TRACK_CLASS = {"C12M02": "DNase", "C19M16": "narrow", "C07M20": "narrow", "C19M22": "narrow",
               "C40M17": "broad", "C40M18": "broad", "C07M29": "broad"}
KNOBS = ("depth", "pe")
RATIOS = (1.0, 0.5, 0.25, 0.125)           # the t137 ladder (floor.RATIOS)
SPREAD_RATIOS = (0.5, 0.25, 0.125)        # seeded ladder points: floor_curve seed_spread_r<k>
READ_AT = {"15M": 15e6, "7.5M": 7.5e6, "3.75M": 3.75e6, "1.875M": 1.875e6}   # floor.READ_AT
DNASE_BASE = "C12M02__base__base"         # the one 50M base (pseudoreps MANIFEST: 25M per half)
NESTED_KNOB = "depth"                     # the arm whose products are read subsamples of the base
ORACLE_R = "manifest"                     # nested_oracle.tsv r_source analyze reads
ORACLE_PREDICTIVE = {"base_to_arm": "binomial", "arm_to_base": "eb_binomial"}
FLOOR_LABEL = "what a fresh replicate would differ by"
ONEWARP = ("QuantileMatching", "QM")
CELL = ["space", "knob", "mark_class", "rung", "g_version", "dirc"]
SUBSETS = ("all", "top1")
SPLIT = ["crps_oracle_scaled_all", "scale_error_all"]
COLS = (CELL[:5] + ["direction", "subset", "d_model", "d_onewarp", "d_nocov", "d_ids",
                    "d_nosolution", "d_floor", "floor_depth", "floor_seed_spread",
                    "gapclosed_onewarp", "gain_over_nocov", "seed_abs_delta", "gain_gt_2seed",
                    "crps_oracle_scaled_all", "scale_error_all", "d_shuffle",
                    "gapclosed_shuffle", "swap_ratio_median", "depthlaw_median_abs_log2err",
                    "n_pairs", "n_runs", "gain_over_onewarp", "gain_over_nocov_gt_2seed",
                    "onewarp_oracle_scaled_all", "onewarp_scale_error_all",
                    "floor_oracle_scaled_all", "floor_scale_error_all", "split_depth",
                    "floor_extrap", "p_floor_matched", "d_oracle", "oracle_kind",
                    "oracle_crps_oracle_scaled_all", "oracle_scale_error_all"])
CHECK_COLS = ["check", "hypothesis", "knob", "space", "rung", "g_version", "mark_class",
              "direction", "subset", "reading", "detail", "threshold"]


def _mean_na(s: pd.Series) -> float:
    return s.mean(skipna=False)


# ---------------------------------------------------------------------------------------------
# the floor per target
# ---------------------------------------------------------------------------------------------


def label(reads) -> str:
    """15000000 -> '15M', 7500000 -> '7.5M', 50000000 -> '50M'."""
    return f"{float(reads) / 1e6:.4g}M"


def target_depth(target_pid: str, arm: str, level: str) -> float:
    """X: the target product's total depth (products MANIFEST `depth`)."""
    if arm == "depth":
        return float(level.rstrip("M")) * 1e6
    return 50e6 if target_pid == DNASE_BASE else 30e6


def base_pid(track: str) -> str:
    return f"{track}__base__base"


def nearest_log2(x: float, grid) -> float:
    return min(grid, key=lambda g: abs(np.log2(x) - np.log2(g)))


def spread_ratio(x: float, n_lines: float) -> float:
    k = x / n_lines
    return 0.5 if k >= 0.5 else nearest_log2(k, SPREAD_RATIOS)


def curve_at(row: dict, x: float) -> float:
    """The product's curve read at reads_per_half x: value_<label>, value_30M_extrap, or the line."""
    for k, v in READ_AT.items():
        if abs(x - v) < 0.5:
            return float(row[f"value_{k}"])
    if abs(x - 30e6) < 0.5:
        return float(row["value_30M_extrap"])
    beta = float(row["beta"])
    alpha = (float(row["alpha"]) if "alpha" in row and pd.notna(row["alpha"])
             else np.log(float(row["value_15M"])) - beta * np.log2(15e6))
    return float(np.exp(alpha + beta * np.log2(x)))


def floor_tables(fp: pd.DataFrame, fc: pd.DataFrame, f15: pd.DataFrame):
    """(curve, split, p15): the floor sources keyed by pid (curve: (pid, metric); split: (pid,
    ratio)). floor_pairs serves the split only; the value and the spread come from the curve."""
    fp = fp.copy()
    fp["ratio"] = fp["thin_ratio"].astype(float).round(6)
    split = {k: r for k, r in fp.groupby(["pid", "ratio"])[SPLIT].mean().iterrows()}
    curve = {(r["pid"], r["metric"]): r.to_dict()
             for _, r in fc[fc["metric"].isin(["crps_all", "crps_top1"])].iterrows()}
    vals = ["crps_all", "crps_top1"] + SPLIT + ["reads_per_half"]
    p15 = {k: r for k, r in f15.groupby("pid")[vals].mean().iterrows()}
    return curve, split, p15


def pair_floor(space, track, target_pid, target_arm, target_level, tabs) -> dict:
    curve, split, p15 = tabs
    base = base_pid(track)
    x = target_depth(target_pid, target_arm, target_level)
    if space == "pval":
        r = p15.get(base)
        if r is None:
            return {"floor_depth": None, "split_depth": None}
        half = float(r["reads_per_half"])
        return {"fl_all": r["crps_all"], "fl_top1": r["crps_top1"],
                "fl_os": r["crps_oracle_scaled_all"], "fl_se": r["scale_error_all"],
                "floor_depth": label(half), "split_depth": label(half),
                "p_floor_matched": float(abs(x - half) <= 0.01 * half)}
    ca, ct = curve.get((base, "crps_all")), curve.get((base, "crps_top1"))
    if ca is None:
        return {"floor_depth": label(x), "split_depth": None}
    n = float(ca["reads_per_half_full"])
    out = {"fl_all": curve_at(ca, x), "floor_depth": label(x), "floor_extrap": float(x > n)}
    if ct is not None:
        out["fl_top1"] = curve_at(ct, x)
    k = f"seed_spread_r{spread_ratio(x, n):g}"
    out["fl_sp_all"] = float(ca[k])
    if ct is not None:
        out["fl_sp_top1"] = float(ct[k])
    ratio = nearest_log2(min(max(x / n, RATIOS[-1]), 1.0), RATIOS)
    sp = split.get((base, round(ratio, 6)))
    out["split_depth"] = label(n * ratio)
    if sp is not None:
        out.update(fl_os=sp["crps_oracle_scaled_all"], fl_se=sp["scale_error_all"])
    return out


def nested_table(no: pd.DataFrame | None) -> dict | None:
    """(pid, direction) -> the nested_oracle.tsv row analyze reads (ORACLE_R, ORACLE_PREDICTIVE)."""
    if no is None:
        return None
    sel = no[(no["space"] == "counts") & (no["eval"] == "score") & (no["r_source"] == ORACLE_R)
             & (no["predictive"] == no["direction"].map(ORACLE_PREDICTIVE))]
    if sel.duplicated(["pid", "direction"]).any():
        raise SystemExit("nested_oracle.tsv: duplicate (pid, direction) rows")
    return {(r["pid"], r["direction"]): r for _, r in sel.iterrows()}


def pair_oracle(space, arm, pid, direction, floor: dict, nested) -> dict:
    """The gap top of one pair: the nested-read oracle on depth pairs in counts, else the floor.
    No nested_oracle.tsv at all (nested None): the floor, oracle_kind `floor_fallback`."""
    if space == "counts" and arm == NESTED_KNOB and nested is not None:
        kind = f"nested_{ORACLE_PREDICTIVE[direction]}"
        r = nested.get((pid, direction))
        if r is None:
            return {"oracle_kind": kind}
        return {"or_all": float(r["crps_all"]), "or_top1": float(r["crps_top1"]),
                "or_os": float(r["crps_oracle_scaled_all"]), "or_se": float(r["scale_error_all"]),
                "oracle_kind": kind}
    kind = "floor_fallback" if space == "counts" and arm == NESTED_KNOB else "floor"
    return {"or_all": floor.get("fl_all"), "or_top1": floor.get("fl_top1"),
            "or_os": floor.get("fl_os"), "or_se": floor.get("fl_se"), "oracle_kind": kind}


def pair_values(rec: pd.DataFrame, base: pd.DataFrame, tabs, nested=None) -> pd.DataFrame:
    """One row per (space, pid, direction) of the trained check set: baselines, the floor (context)
    and the oracle (the gap top)."""
    keys = ["space", "pid", "direction"]
    pairs = rec.loc[rec["kind"] == "trained",
                    keys + ["track", "arm", "level", "target_pid"]].drop_duplicates(keys)
    rows = []
    for p in pairs.itertuples(index=False):
        b2a = p.direction == "base_to_arm"
        t_arm, t_level = (p.arm, p.level) if b2a else ("base", "base")
        fl = pair_floor(p.space, p.track, p.target_pid, t_arm, t_level, tabs)
        rows.append({"space": p.space, "pid": p.pid, "direction": p.direction, **fl,
                     **pair_oracle(p.space, p.arm, p.pid, p.direction, fl, nested)})
    pv = pd.DataFrame(rows)
    for c in ("fl_all", "fl_top1", "fl_os", "fl_se", "fl_sp_all", "fl_sp_top1", "floor_extrap",
              "p_floor_matched", "or_all", "or_top1", "or_os", "or_se"):
        if c not in pv:
            pv[c] = np.nan
    for tag, rungs in (("ow", ONEWARP), ("ns", ("noSolution",))):
        b = base[base["rung"].isin(rungs)].set_index(keys)
        b = b[~b.index.duplicated()]
        sub = b[["crps_all", "crps_top1"] + SPLIT].rename(columns={
            "crps_all": f"{tag}_all", "crps_top1": f"{tag}_top1",
            "crps_oracle_scaled_all": f"{tag}_os", "scale_error_all": f"{tag}_se"})
        pv = pv.merge(sub, left_on=keys, right_index=True, how="left")
    return pv


# ---------------------------------------------------------------------------------------------
# pooling
# ---------------------------------------------------------------------------------------------


def expand(df: pd.DataFrame, by_direction: bool = True) -> pd.DataFrame:
    """Each row also counted in mark_class `all` and (by_direction) direction `both`."""
    df = pd.concat([df, df.assign(mark_class="all")], ignore_index=True)
    df["dirc"] = df["direction"] if by_direction else "both"
    if by_direction:
        df = pd.concat([df, df.assign(dirc="both")], ignore_index=True)
    return df


def pool(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """per run mean over pairs -> per seed mean over runs; indexed by CELL + seed."""
    run = df.groupby(CELL + ["seed", "run"])[cols].agg(_mean_na)
    return run.groupby(level=CELL + ["seed"]).agg(_mean_na)


def seed_stats(per_seed: pd.DataFrame, col: str) -> tuple[pd.Series, pd.Series]:
    g = per_seed[col].groupby(level=CELL)
    mean = g.agg(_mean_na)
    delta = g.agg(lambda s: (s.max() - s.min()) if s.notna().all() and len(s) > 1 else np.nan)
    return mean, delta


def _ratio(num, den):
    num, den = np.asarray(num, float), np.asarray(den, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den != 0, num / den, np.nan)


def blind_table(rec: pd.DataFrame, base: pd.DataFrame, tabs, nested=None) -> pd.DataFrame:
    rec = rec[(rec["eval"] == "score") & rec["arm"].isin(KNOBS + ("base",))].copy()
    rec["knob"] = rec["arm"]
    rec["mark_class"] = rec["track"].map(TRACK_CLASS)
    if rec["mark_class"].isna().any():
        bad = sorted(rec.loc[rec["mark_class"].isna(), "track"].unique())
        raise SystemExit(f"tracks with no mark class: {bad}")
    pv = pair_values(rec, base, tabs, nested)
    tr = rec[rec["kind"] == "trained"]
    real = expand(tr[tr["model"] == "real"].merge(pv, on=["space", "pid", "direction"], how="left"))
    vcols = (["crps_all", "crps_top1"] + SPLIT
             + [f"{t}_{s}" for t in ("ow", "ns", "fl", "or") for s in ("all", "top1")]
             + ["ow_os", "ow_se", "fl_os", "fl_se", "fl_sp_all", "fl_sp_top1", "or_os", "or_se"])
    ps_real = pool(real, vcols)
    twins = {m: pool(expand(tr[tr["model"] == m]), ["crps_all", "crps_top1"])
             for m in ("nocov", "ids")}
    sh = rec[(rec["kind"] == "shuffle") & (rec["model"] == "real")]
    ps_shuf = pool(expand(sh), ["crps_all", "crps_top1"])
    g = real.groupby(CELL)
    n_pairs = real.drop_duplicates(CELL + ["pid", "direction"]).groupby(CELL).size()
    n_runs = g["run"].nunique()
    fdepth = g["floor_depth"].agg(lambda s: ",".join(sorted(s.dropna().unique())))
    sdepth = g["split_depth"].agg(lambda s: ",".join(sorted(s.dropna().unique())))
    fextrap = g["floor_extrap"].max()
    pmatch = g["p_floor_matched"].min()
    okind = g["oracle_kind"].agg(lambda s: ",".join(sorted(s.dropna().unique())))

    sw = rec[(rec["kind"] == "swap") & (rec["model"] == "real")]
    sw = pd.concat([sw, sw.assign(mark_class="all")], ignore_index=True)
    swap = sw.groupby(["space", "knob", "mark_class", "rung", "g_version"])["swap_ratio_median"].median()
    dl = rec[(rec["kind"] == "depthlaw") & (rec["model"] == "real")].copy()
    dl["abs_err"] = (dl["log2_pred_scale"] - dl["log2_depth_ratio"]).abs()
    dl = expand(dl)
    dlaw = dl.groupby(CELL)["abs_err"].median()

    idx = n_runs.index
    out = []
    for s in SUBSETS:
        col = f"crps_{s}"
        t = pd.DataFrame(index=idx)
        t["d_model"], t["seed_abs_delta"] = seed_stats(ps_real, col)
        for name, src in (("d_onewarp", f"ow_{s}"), ("d_nosolution", f"ns_{s}"),
                          ("d_floor", f"fl_{s}"), ("floor_seed_spread", f"fl_sp_{s}"),
                          ("crps_oracle_scaled_all", "crps_oracle_scaled_all"),
                          ("scale_error_all", "scale_error_all"),
                          ("onewarp_oracle_scaled_all", "ow_os"),
                          ("onewarp_scale_error_all", "ow_se"),
                          ("floor_oracle_scaled_all", "fl_os"),
                          ("floor_scale_error_all", "fl_se"), ("d_oracle", f"or_{s}"),
                          ("oracle_crps_oracle_scaled_all", "or_os"),
                          ("oracle_scale_error_all", "or_se")):
            t[name] = seed_stats(ps_real, src)[0]
        t["d_nocov"] = seed_stats(twins["nocov"], col)[0].reindex(idx)
        t["d_ids"] = seed_stats(twins["ids"], col)[0].reindex(idx)
        t["d_shuffle"] = seed_stats(ps_shuf, col)[0].reindex(idx)
        t["subset"] = s
        t["floor_depth"] = fdepth
        t["split_depth"] = sdepth
        t["floor_extrap"] = fextrap
        t["p_floor_matched"] = pmatch
        t["oracle_kind"] = okind
        t["n_pairs"] = n_pairs
        t["n_runs"] = n_runs
        out.append(t)
    t = pd.concat(out).reset_index()
    t["direction"] = t["dirc"]
    t["gain_over_onewarp"] = t["d_onewarp"] - t["d_model"]
    t["gapclosed_onewarp"] = _ratio(t["gain_over_onewarp"], t["d_onewarp"] - t["d_oracle"])
    t["gain_over_nocov"] = t["d_nocov"] - t["d_model"]
    two = 2 * t["seed_abs_delta"]
    t["gain_gt_2seed"] = (t["gain_over_onewarp"] > two).where(two.notna())
    t["gain_over_nocov_gt_2seed"] = (t["gain_over_nocov"] > two).where(two.notna())
    t["gapclosed_shuffle"] = _ratio(t["d_onewarp"] - t["d_shuffle"], t["d_onewarp"] - t["d_oracle"])
    sk = ["space", "knob", "mark_class", "rung", "g_version"]
    t["swap_ratio_median"] = [swap.get(k, np.nan) for k in t[sk].itertuples(index=False, name=None)]
    t["depthlaw_median_abs_log2err"] = [dlaw.get(k, np.nan)
                                        for k in t[CELL].itertuples(index=False, name=None)]
    dir_order = {"base_to_arm": 0, "arm_to_base": 1, "both": 2}
    t = t.assign(_d=t["direction"].map(dir_order)).sort_values(
        ["space", "knob", "mark_class", "rung", "g_version", "_d", "subset"])
    return t[COLS].reset_index(drop=True)


# ---------------------------------------------------------------------------------------------
# the check readings (no ticks)
# ---------------------------------------------------------------------------------------------


def _f(x) -> str:
    return "NA" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.6g}"


def checks_readings(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    head = t[(t["direction"] == "both") & (t["subset"] == "all")]
    for r in head.itertuples(index=False):
        h = "h1" if r.knob == "depth" else "h2"
        common = {"hypothesis": h, "knob": r.knob, "space": r.space, "rung": r.rung,
                  "g_version": r.g_version, "mark_class": r.mark_class, "direction": "both",
                  "subset": "all", "threshold": "TODO(PI)"}
        rows.append({**common, "check": "gapclosed", "reading": _f(r.gapclosed_onewarp),
                     "detail": f"gain over onewarp {_f(r.gain_over_onewarp)}; 2 x seed |delta| "
                               f"{_f(2 * r.seed_abs_delta)}; oracle {r.oracle_kind} "
                               f"{_f(r.d_oracle)}; floor ({FLOOR_LABEL}) {_f(r.d_floor)} "
                               f"floor_depth {r.floor_depth} "
                               f"floor_extrap {_f(r.floor_extrap)} p_floor_matched "
                               f"{_f(r.p_floor_matched)}"})
        rows.append({**common, "check": "shuffledrop", "reading": _f(r.gapclosed_shuffle),
                     "detail": "t118's shuffle check (C' of another arm of the same track: every "
                               "knob changes), NOT the marginal resample of the depth / run-type "
                               f"entry the hypothesis names; gapclosed unshuffled "
                               f"{_f(r.gapclosed_onewarp)}"})
        rows.append({**common, "check": "swapfidelity", "reading": _f(r.swap_ratio_median),
                     "detail": "median |log(pred / X)| with C' = C (t118's swap), real model"})
        rows.append({**common, "check": "crps_split",
                     "reading": _f(r.crps_oracle_scaled_all - r.onewarp_oracle_scaled_all),
                     "detail": f"model oracle_scaled {_f(r.crps_oracle_scaled_all)} scale_error "
                               f"{_f(r.scale_error_all)}; onewarp oracle_scaled "
                               f"{_f(r.onewarp_oracle_scaled_all)} scale_error "
                               f"{_f(r.onewarp_scale_error_all)}; reading = model minus onewarp "
                               "oracle_scaled"})
    for k, _ in head.groupby(["space", "rung", "g_version", "mark_class"]):
        rows.append({"check": "composegap", "hypothesis": "h1,h2", "knob": "depth x pe",
                     "space": k[0], "rung": k[1], "g_version": k[2], "mark_class": k[3],
                     "direction": "arm_to_arm", "subset": "all", "reading": "not_extracted",
                     "detail": "the never-trained depth x pe law pairs (law.json) are not in "
                               "this extract", "threshold": "TODO(PI)"})
    return pd.DataFrame(rows, columns=CHECK_COLS)


BLIND_MD = """# t130 — blind check-set readings (h1 depth, h2 run type)

Readings only. No box is ticked and no verdict is given here; every threshold is `TODO(PI)`.

## Procedure
1. `extract_records.py` (Nibi, stdlib) flattened the h1/h2 check-set records of the 576 row-1
   ladder runs and the rungs_v2 noSolution / QuantileMatching records of the same pairs.
2. This table was produced in ONE run of `analyze.py` from that extract and the t137 floor,
   before anyone read a per-rung number. The model rung for any verdict is the PI's pick
   (h1 item vi); nothing here picks one.
3. `verify_raw.py` (stdlib, no shared code) recomputed the headline cells from the raw files;
   `verify_compare.py` wrote `verify.txt` (AGREE / DISAGREE at 1e-6).

## Definitions
- The ORACLE (the top of the gap-closed interval; PI ruling 2026-10-30). Depth pairs in counts
  use the NESTED-READ oracle (`nested_oracle.tsv`, `tools/t130/nested_oracle.py`): each t112
  depth arm is a read subsample of its base, so the base and the arm share reads, and two
  independent replicates are no upper bound. base_to_arm: Binomial(base count, rho), exact;
  arm_to_base: the exact posterior of the base count given the arm count under binomial
  thinning, with the base's training-chromosome histogram as prior. rho = the arm's MANIFEST
  depth / the base's. `oracle_kind` names it (`nested_binomial`, `nested_eb_binomial`).
- What the oracle bounds: `nested_binomial` (base_to_arm) is a true ceiling up to the
  hypergeometric factor; `nested_eb_binomial` (arm_to_base) is the Bayes optimum only for a
  predictor that sees the arm count at that one bin with a genome-wide prior, so a model that
  reads neighbouring bins can beat it, and `gapclosed_onewarp` > 1 there is possible and not a
  bug. On subset top1 (ranked by the target) neither is a strict bound.
- If `nested_oracle.tsv` is absent, depth pairs in counts fall back to the floor as their oracle
  (`oracle_kind floor_fallback`), and the summary below says so.
- Every other pair keeps the floor as its oracle (`oracle_kind floor`). h2's run-type arm `pe`
  is NOT a subsample of the base (a separate FASTQ reprocessing), so it is not nested-by-
  subsampling and keeps the floor; depth pairs in -log10 p keep it too (p cannot be thinned).
- `d_floor` stays in every row as context: what a fresh replicate would differ by.
- The floor at a target of total depth X (15M / 7.5M / 3.75M depth arm, 50M DNase base, else
  30M) is the same track's BASE product's t137 curve read at reads_per_half = X
  (`floor_depth` = X; `floor_extrap 1` when X is beyond the base's own reads per half). The
  split is the base's ladder split at the nearest ratio (`split_depth`). In -log10 p the only
  floor is the base's own-depth oracle (`floor_depth` = its reads per half); `p_floor_matched 0`
  marks every target whose X is not that depth.
- Pooling: per run mean over its check-set pairs -> per seed mean over runs -> mean over seeds.
  `seed_abs_delta` = the largest pairwise |difference| of the three per-seed d_model values.
- `gapclosed_onewarp` = (d_onewarp - d_model) / (d_onewarp - d_oracle);
  `gain_gt_2seed` = (d_onewarp - d_model) > 2 x seed_abs_delta.
- Every CRPS carries its split: the model's `crps_oracle_scaled_all` / `scale_error_all`, and
  the one-warp, floor and oracle splits in the appended columns.
- `shuffledrop` reads t118's shuffle (C' from another arm of the same track), which changes
  every knob, not only the one the hypothesis names.
- `composegap`: the never-trained depth x pe law pairs are not extracted.

## This run
{summary}
"""


def main(argv) -> int:
    if len(argv) != 1:
        print("usage: analyze.py <results_dir>", file=sys.stderr)
        return 2
    d = Path(argv[0])
    rec = pd.read_csv(d / "records.tsv.gz", sep="\t", dtype={"level": str, "g": str})
    base = pd.read_csv(d / "baselines.tsv.gz", sep="\t", dtype={"level": str})
    fp = pd.read_csv(d / "floor_pairs.tsv", sep="\t")
    fc = pd.read_csv(d / "floor_curve.tsv", sep="\t")
    f15 = pd.read_csv(d / "floor_p15M.tsv", sep="\t")
    nop = d / "nested_oracle.tsv"
    no = pd.read_csv(nop, sep="\t") if nop.is_file() else None
    t = blind_table(rec, base, floor_tables(fp, fc, f15), nested_table(no))
    t.to_csv(d / "blind_table.tsv", sep="\t", index=False, na_rep="NA", float_format="%.10g")
    c = checks_readings(t)
    c.to_csv(d / "checks_readings.tsv", sep="\t", index=False, na_rep="NA")
    key = ["d_model", "d_onewarp", "d_oracle", "d_floor", "seed_abs_delta",
           "crps_oracle_scaled_all", "scale_error_all"]
    lines = [f"records {len(rec)} (runs {rec['run'].nunique()}), baselines {len(base)}, "
             f"floor_pairs {len(fp)}, floor_curve {len(fc)}, floor_p15M {len(f15)}, "
             + (f"nested_oracle {len(no)}" if no is not None
                else "nested_oracle.tsv ABSENT: depth pairs in counts fall back to the floor as their "
                     "oracle (oracle_kind floor_fallback)"),
             f"oracle: depth pairs in counts -> nested-read ({ORACLE_R} rho; base_to_arm "
             f"{ORACLE_PREDICTIVE['base_to_arm']}, arm_to_base {ORACLE_PREDICTIVE['arm_to_base']}); "
             f"every other pair -> the floor. d_floor = context ({FLOOR_LABEL})",
             f"blind_table rows {len(t)}; checks_readings rows {len(c)}"]
    lines += [f"NA in {k}: {int(t[k].isna().sum())}" for k in key]
    head = t[(t["direction"] == "both") & (t["subset"] == "all") & (t["mark_class"] == "all")]
    lines.append(f"headline cells (direction both, subset all, mark_class all): {len(head)}")
    text = "\n".join(lines)
    (d / "BLIND.md").write_text(BLIND_MD.format(summary="\n".join("- " + x for x in lines)))
    (d / "analyze_out.txt").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
