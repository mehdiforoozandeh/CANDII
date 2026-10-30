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
  d_floor       the floor of the TARGET product at the target's depth. counts: a depth-15M /
                7.5M / 3.75M target -> that product's floor_pairs rows at thin_ratio 1.0 / 0.5 /
                0.25 (mean over seeds and both directions); a base or pe target (30M) -> its
                floor_curve value_30M_extrap. The split (oracle_scaled / scale_error) at 30M is
                not extrapolated: the ratio-1.0 split stands beside it, flagged split_depth 15M.
                pval: floor_p15M (the 15M rungs_v2 oracle) for every target, floor_depth 15M.
  floor_seed_spread  |seed 0 - seed 1| of the direction-mean floor at the target's ratio; at a
                ratio-1.0 or 30M target (no draw) the spread at ratio 0.5. NA in pval (no seeds).
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
  onewarp_scale_error_all, floor_oracle_scaled_all, floor_scale_error_all, split_depth.

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
DEPTH_RATIO = {"15M": 1.0, "7.5M": 0.5, "3.75M": 0.25}
SPREAD_RATIO = 0.5                        # the nearest seeded ladder point for a no-draw target
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
                    "floor_oracle_scaled_all", "floor_scale_error_all", "split_depth"])
CHECK_COLS = ["check", "hypothesis", "knob", "space", "rung", "g_version", "mark_class",
              "direction", "subset", "reading", "detail", "threshold"]


def _mean_na(s: pd.Series) -> float:
    return s.mean(skipna=False)


# ---------------------------------------------------------------------------------------------
# the floor per target
# ---------------------------------------------------------------------------------------------


def floor_tables(fp: pd.DataFrame, fc: pd.DataFrame, f15: pd.DataFrame):
    """(means, spreads, curve30, p15): dicts keyed by pid (and ratio) of the floor values."""
    fp = fp.copy()
    fp["ratio"] = fp["thin_ratio"].astype(float).round(6)
    vals = ["crps_all", "crps_top1"] + SPLIT
    means = {k: r for k, r in fp.groupby(["pid", "ratio"])[vals].mean().iterrows()}
    per_seed = fp.groupby(["pid", "ratio", "seed"])[["crps_all", "crps_top1"]].mean()
    spreads = {}
    for (pid, ratio), g in per_seed.groupby(level=[0, 1]):
        spreads[(pid, ratio)] = (g.max() - g.min()) if len(g) > 1 else g.iloc[0] * np.nan
    curve = fc[fc["metric"].isin(["crps_all", "crps_top1"])]
    curve30 = {(r.pid, r.metric): float(r.value_30M_extrap) for r in curve.itertuples()}
    p15 = {k: r for k, r in f15.groupby("pid")[vals].mean().iterrows()}
    return means, spreads, curve30, p15


def pair_floor(space, target_pid, target_arm, target_level, tabs) -> dict:
    means, spreads, curve30, p15 = tabs
    nan = float("nan")
    if space == "pval":
        r = p15.get(target_pid)
        if r is None:
            return {"floor_depth": "15M", "split_depth": "15M"}
        return {"fl_all": r["crps_all"], "fl_top1": r["crps_top1"],
                "fl_os": r["crps_oracle_scaled_all"], "fl_se": r["scale_error_all"],
                "fl_sp_all": nan, "fl_sp_top1": nan, "floor_depth": "15M", "split_depth": "15M"}
    out = {}
    if target_arm == "depth" and target_level in DEPTH_RATIO:
        ratio = DEPTH_RATIO[target_level]
        m = means.get((target_pid, ratio))
        if m is not None:
            out.update(fl_all=m["crps_all"], fl_top1=m["crps_top1"])
        out.update(floor_depth=target_level, split_depth=target_level)
    else:
        ratio = 1.0
        out.update(fl_all=curve30.get((target_pid, "crps_all"), nan),
                   fl_top1=curve30.get((target_pid, "crps_top1"), nan),
                   floor_depth="30M_extrap", split_depth="15M")
        m = means.get((target_pid, ratio))
    if m is not None:
        out.update(fl_os=m["crps_oracle_scaled_all"], fl_se=m["scale_error_all"])
    sp = spreads.get((target_pid, ratio if ratio < 1.0 else SPREAD_RATIO))
    if sp is not None:
        out.update(fl_sp_all=sp["crps_all"], fl_sp_top1=sp["crps_top1"])
    return out


def pair_values(rec: pd.DataFrame, base: pd.DataFrame, tabs) -> pd.DataFrame:
    """One row per (space, pid, direction) of the trained check set: baselines and the floor."""
    keys = ["space", "pid", "direction"]
    pairs = rec.loc[rec["kind"] == "trained", keys + ["arm", "level", "target_pid"]].drop_duplicates(keys)
    rows = []
    for p in pairs.itertuples(index=False):
        b2a = p.direction == "base_to_arm"
        t_arm, t_level = (p.arm, p.level) if b2a else ("base", "base")
        rows.append({"space": p.space, "pid": p.pid, "direction": p.direction,
                     **pair_floor(p.space, p.target_pid, t_arm, t_level, tabs)})
    pv = pd.DataFrame(rows)
    for c in ("fl_all", "fl_top1", "fl_os", "fl_se", "fl_sp_all", "fl_sp_top1"):
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
                               f"{_f(2 * r.seed_abs_delta)}; floor {r.floor_depth}"})
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
- The floor is the TARGET product's resampling floor at the target's depth (t137): ladder
  ratio 1.0 / 0.5 / 0.25 for a 15M / 7.5M / 3.75M target; the curve read at 30M for a base or
  pe target (its split is the 15M one, `split_depth 15M`); in -log10 p the 15M oracle only.
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
