"""t130 blind check-set readings, step 2 (laptop, pandas): the blind table of the h1 (depth) and
h2 (run type, arm `pe`) check set, from the extract and the t137 floor.

READINGS ONLY. Nothing here ticks a box or gives a verdict. Every threshold is the PI's
(`TODO(PI)` in checks_readings.tsv); the model rung for any verdict is the PI's pick (h1 item vi).

Inputs in <results_dir> (run_all.sh copies them in):
  records.tsv.gz, baselines.tsv.gz   extract_records.py
  floor_pairs.tsv, floor_curve.tsv, floor_p15M.tsv   t137 (tools/t137/floor.py aggregate)

Per pair (space, arm product pid, direction):
  d_onewarp     the pair's QuantileMatching record (rungs_v2), Poisson / log-normal spread CRPS.
  d_nosolution  the pair's noSolution record.
  d_floor       the floor at the TARGET's real depth X (pinned rule, plan_gaps.md A3). X = the
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
  gapclosed_onewarp = (d_onewarp - d_model) / (d_onewarp - d_floor)
  gain_over_nocov   = d_nocov - d_model
  gain_gt_2seed     = (d_onewarp - d_model) > 2 x seed_abs_delta   (the h1 PI ruling's gain)
  crps_oracle_scaled_all, scale_error_all   the model's split (subset all, on every row)
  d_shuffle, gapclosed_shuffle = (d_onewarp - d_shuffle) / (d_onewarp - d_floor)   t118's
                    shuffle (C' from another arm of the same track, every knob changes)
  swap_ratio_median  median over the real model's swap records of the cell's arm products of
                    median |log(pred / X)| (direction- and subset-free)
  depthlaw_median_abs_log2err  median |log2 pred scale - log2 true depth ratio| over the real
                    model's depthlaw records (counts, depth only)
  then, appended: gain_over_onewarp, gain_over_nocov_gt_2seed, onewarp_oracle_scaled_all,
  onewarp_scale_error_all, floor_oracle_scaled_all, floor_scale_error_all, split_depth,
  floor_extrap, p_floor_matched.

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
                    "floor_extrap", "p_floor_matched"])
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


def pair_values(rec: pd.DataFrame, base: pd.DataFrame, tabs) -> pd.DataFrame:
    """One row per (space, pid, direction) of the trained check set: baselines and the floor."""
    keys = ["space", "pid", "direction"]
    pairs = rec.loc[rec["kind"] == "trained",
                    keys + ["track", "arm", "level", "target_pid"]].drop_duplicates(keys)
    rows = []
    for p in pairs.itertuples(index=False):
        b2a = p.direction == "base_to_arm"
        t_arm, t_level = (p.arm, p.level) if b2a else ("base", "base")
        rows.append({"space": p.space, "pid": p.pid, "direction": p.direction,
                     **pair_floor(p.space, p.track, p.target_pid, t_arm, t_level, tabs)})
    pv = pd.DataFrame(rows)
    for c in ("fl_all", "fl_top1", "fl_os", "fl_se", "fl_sp_all", "fl_sp_top1", "floor_extrap",
              "p_floor_matched"):
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


def blind_table(rec: pd.DataFrame, base: pd.DataFrame, tabs) -> pd.DataFrame:
    rec = rec[(rec["eval"] == "score") & rec["arm"].isin(KNOBS + ("base",))].copy()
    rec["knob"] = rec["arm"]
    rec["mark_class"] = rec["track"].map(TRACK_CLASS)
    if rec["mark_class"].isna().any():
        bad = sorted(rec.loc[rec["mark_class"].isna(), "track"].unique())
        raise SystemExit(f"tracks with no mark class: {bad}")
    pv = pair_values(rec, base, tabs)
    tr = rec[rec["kind"] == "trained"]
    real = expand(tr[tr["model"] == "real"].merge(pv, on=["space", "pid", "direction"], how="left"))
    vcols = (["crps_all", "crps_top1"] + SPLIT
             + [f"{t}_{s}" for t in ("ow", "ns", "fl") for s in ("all", "top1")]
             + ["ow_os", "ow_se", "fl_os", "fl_se", "fl_sp_all", "fl_sp_top1"])
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
                          ("floor_scale_error_all", "fl_se")):
            t[name] = seed_stats(ps_real, src)[0]
        t["d_nocov"] = seed_stats(twins["nocov"], col)[0].reindex(idx)
        t["d_ids"] = seed_stats(twins["ids"], col)[0].reindex(idx)
        t["d_shuffle"] = seed_stats(ps_shuf, col)[0].reindex(idx)
        t["subset"] = s
        t["floor_depth"] = fdepth
        t["split_depth"] = sdepth
        t["floor_extrap"] = fextrap
        t["p_floor_matched"] = pmatch
        t["n_pairs"] = n_pairs
        t["n_runs"] = n_runs
        out.append(t)
    t = pd.concat(out).reset_index()
    t["direction"] = t["dirc"]
    t["gain_over_onewarp"] = t["d_onewarp"] - t["d_model"]
    t["gapclosed_onewarp"] = _ratio(t["gain_over_onewarp"], t["d_onewarp"] - t["d_floor"])
    t["gain_over_nocov"] = t["d_nocov"] - t["d_model"]
    two = 2 * t["seed_abs_delta"]
    t["gain_gt_2seed"] = (t["gain_over_onewarp"] > two).where(two.notna())
    t["gain_over_nocov_gt_2seed"] = (t["gain_over_nocov"] > two).where(two.notna())
    t["gapclosed_shuffle"] = _ratio(t["d_onewarp"] - t["d_shuffle"], t["d_onewarp"] - t["d_floor"])
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
                               f"{_f(2 * r.seed_abs_delta)}; floor_depth {r.floor_depth} "
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
- The floor at a target of total depth X (15M / 7.5M / 3.75M depth arm, 50M DNase base, else
  30M) is the same track's BASE product's t137 curve read at reads_per_half = X
  (`floor_depth` = X; `floor_extrap 1` when X is beyond the base's own reads per half). The
  split is the base's ladder split at the nearest ratio (`split_depth`). In -log10 p the only
  floor is the base's own-depth oracle (`floor_depth` = its reads per half); `p_floor_matched 0`
  marks every target whose X is not that depth.
- Pooling: per run mean over its check-set pairs -> per seed mean over runs -> mean over seeds.
  `seed_abs_delta` = the largest pairwise |difference| of the three per-seed d_model values.
- `gapclosed_onewarp` = (d_onewarp - d_model) / (d_onewarp - d_floor);
  `gain_gt_2seed` = (d_onewarp - d_model) > 2 x seed_abs_delta.
- Every CRPS carries its split: the model's `crps_oracle_scaled_all` / `scale_error_all`, and
  the one-warp and floor splits in the appended columns.
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
    t = blind_table(rec, base, floor_tables(fp, fc, f15))
    t.to_csv(d / "blind_table.tsv", sep="\t", index=False, na_rep="NA", float_format="%.10g")
    c = checks_readings(t)
    c.to_csv(d / "checks_readings.tsv", sep="\t", index=False, na_rep="NA")
    key = ["d_model", "d_onewarp", "d_floor", "seed_abs_delta", "crps_oracle_scaled_all",
           "scale_error_all"]
    lines = [f"records {len(rec)} (runs {rec['run'].nunique()}), baselines {len(base)}, "
             f"floor_pairs {len(fp)}, floor_curve {len(fc)}, floor_p15M {len(f15)}",
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
