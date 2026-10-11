"""t125 part 1 — aggregate every scored eta run to the readings of handoff Phase 7 part 1: each
design against the old grid, each model against its no-covariates twin, A against Aold, where a gain
sits (mean or spread), the depth law, and the count of exploding and non-finite CRPS records.

**Aggregation** (as t118, `ladder.aggregate`; eval `score`, space `counts`). Mean over bins per pair
(done by `score.py`); mean over pairs per track; macro mean over the tracks of a mark class
(`baseline_rungs.MARK_CLASS`), per seed; then `mean` over the seeds and `seed_wobble` = the largest
pairwise |delta| of the per-seed class values (None with fewer than two seeds). Variants `all` and
`nonidentical` as t118. Kinds trained, shuffle, swap, law. Metrics: crps_* and spearman_* (t118's;
spearman is reported only, never a reading), mse_log1p_* (MSE on log(1 + counts) of the predicted
mean), and the CRPS split crps_oracle_scaled_all + scale_error_all (trained/score only).
A None metric (not finite, or no bins) is left out of a mean and counted below, never hidden.

**Readings** (`results.json["readings"]`; each `{reading, rung, g_version, mark_class, metric, model,
value, bar, met, seed_wobble, components}`). Every metric is lower-is-better; a gain is
`other - this`, positive when `this` is better. `bar` = 2 x the larger seed wobble of the two arms;
`met` = gain > bar ("the gain exceeds 2 x the larger seed wobble"), False with a `met_reason` when
the bar cannot be formed (one seed). `met` is a drafted reading, never a tick or a verdict.
  beatstwin       nocov - real (trained), metrics crps_all, crps_top1, mse_log1p_all, mse_log1p_top1.
                  The twin KEEPS the depth offset (PI), so this gap is not directly comparable with
                  the old grid's real-vs-twin gap; the old gap is beside it as `oldgrid_gap`.
  vs_oldgrid      old-grid real (`--old-grid-tsv`: space counts, kind trained, variant all) minus the
                  t125 real mean of the same rung, g_version, class, metric; Aold against the old
                  grid's A (the depth offset alone). No row when the old grid has no value (it has
                  no MSE).
  vs_oldspread    A only: Aold - A, real (the new spread rule alone, both with the offset).
  mean_or_spread  one row per comparison above and (rung, g_version, class); metric = the
                  comparison's name; value, bar, met None; `components.parts` holds, for crps_all,
                  its two split parts, crps_top1, mse_log1p_all and mse_log1p_top1: this, other,
                  both wobbles, gain, bar, gain_exceeds_bar. The old grid's split comes from
                  `--old-grid-json` (its results.json per_pair records, aggregated by the same rule).
  depthlaw        per (rung, g_version), real, mark_class "all": max over the depth->depth law
                  pairs of |2^(mean over seeds of (log2_pred - log2_true)) - 1|; bar 0.10;
                  met = value <= bar (as t118). Beside it: the same over the trained base<->depth
                  pairs, and the twin's value.
  n_pairs_crps_gt_20 / n_pairs_crps_nonfinite
                  per (rung, g_version, class, model): the (pair, seed) trained/score records whose
                  crps_all is > 20 / not finite (None). Reported only (value; bar and met None).
`results.json["explosions"]` has the same two counts for every record kind and eval; a depthlaw
record is a copy of a scored record and is never counted.

**Outputs** in `<agg_dir>`: results.json, results_summary.tsv (t118's columns), pairs_summary.tsv,
depth_law.tsv, grid.md, summary_data.json. Missing runs (of the 432 task names) are listed in
`results.json["runs_missing"]`, not fatal. An empty or absent old-grid file gives no vs_oldgrid rows.

    python tools/t125/eta/aggregate.py <manifest> <covariates.tsv> <runs_dir> <agg_dir>
                                       [--old-grid-tsv PATH] [--old-grid-json PATH]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[2]
for _p in (_TOOLS / "t125", _TOOLS / "t118"):   # import convention: tools/t118 and tools/t125
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ladder import aggregate as la  # noqa: E402
from ladder.score import METRICS, SWAP_KEY, write_json  # noqa: E402
from eta import pairs as ep  # noqa: E402

SPACE = "counts"
MSE_KEYS = ("mse_log1p_all", "mse_log1p_nonzero", "mse_log1p_top1")
SPLIT_KEYS = ("crps_oracle_scaled_all", "scale_error_all")
AGG_METRICS = METRICS + MSE_KEYS + SPLIT_KEYS
READING_METRICS = ("crps_all", "crps_top1", "mse_log1p_all", "mse_log1p_top1")
PARTS = ("crps_all", "crps_oracle_scaled_all", "scale_error_all", "crps_top1", "mse_log1p_all",
         "mse_log1p_top1")
COMPARISONS = ("beatstwin", "vs_oldgrid", "vs_oldspread")
EXPLODE_AT = 20.0
DEPTH_BAR = la.DEPTH_BAR
COLUMNS = ("A", "B", "C", "D")
COLUMN_LABEL = {"A": "A: affine in log space", "B": "B: monotone curve, 12 knots",
                "C": "C: 33-bin kernel, then the curve", "D": "D: FiLM-modulated dilated CNN"}
CLASS_ORDER = ("DNase", "narrow", "broad")
GV_LABEL = {"per_track": "one g per track", "across": "one g across tracks"}
METRIC_LABEL = {"crps_all": "CRPS, all bins", "crps_top1": "CRPS, top 1 % bins",
                "mse_log1p_all": "MSE on log(1 + counts) of the predicted mean, all bins",
                "mse_log1p_top1": "MSE on log(1 + counts) of the predicted mean, top 1 % bins"}
TWIN_NOTE = ("the no-covariates twin keeps the depth offset (PI), so the real-vs-twin gap shows "
             "what the covariates add beyond the fixed d'/d scaling; it is not directly "
             "comparable with the old grid's real-vs-twin gap")
SUMMARY_COLS = la.SUMMARY_COLS
PAIRS_COLS = ("run_name", "rung", "g_version", "model", "seed", "track", "mark_class",
              "source_pid", "target_pid", "knob_combo", "crps_all", "crps_top1",
              "crps_oracle_scaled_all", "scale_error_all", "mse_log1p_all", "mse_log1p_top1")
DEPTH_COLS = ("rung", "g_version", "model", "seed", "depth_source", "source_pid", "target_pid",
              "depth_log2_ratio_true", "depth_log2_scale_pred", "track", "mark_class")

_mean, wobble, _warn = la._mean, la.wobble, la._warn


def _finite(v) -> bool:
    return v is not None and not isinstance(v, bool) and math.isfinite(float(v))


def _twice(w):
    return None if w is None else 2.0 * w


def _wmax(*ws):
    return None if any(w is None for w in ws) else max(ws)


# ---------------------------------------------------------------------------------------------
# pair -> track -> class
# ---------------------------------------------------------------------------------------------


def per_track_rows(per_pair: list[dict], metrics=AGG_METRICS, ev: str = "score") -> list[dict]:
    """t118's per-track rows over `metrics` (+ the swap key on swap records), space counts."""
    groups: dict = defaultdict(list)
    for r in per_pair:
        if r.get("eval") != ev or r.get("kind") not in la.KINDS or r.get("space") != SPACE:
            continue
        key = tuple(r[k] for k in ("rung", "g_version", "space", "model", "seed", "kind",
                                   "track", "mark_class"))
        groups[key].append(r)
    out = []
    for key, rs in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        rung, gv, space, model, seed, kind, track, mc = key
        ms = tuple(metrics) + ((SWAP_KEY,) if kind == "swap" else ())
        for variant in la.VARIANTS:
            sel = rs if variant == "all" else [r for r in rs if not r[f"{SPACE}_identical"]]
            for m in ms:
                vals = [r.get(m) for r in sel if _finite(r.get(m))]
                if not vals:
                    continue
                out.append({"rung": rung, "g_version": gv, "space": space, "model": model,
                            "seed": seed, "kind": kind, "track": track, "mark_class": mc,
                            "metric": m, "value": _mean(vals), "n_pairs": len(vals),
                            "variant": variant})
    return out


def _class_index(per_class: list[dict]) -> dict:
    """(rung, g_version, model, kind, mark_class, metric) -> per_class row (variant all)."""
    return {(r["rung"], r["g_version"], r["model"], r["kind"], r["mark_class"], r["metric"]): r
            for r in per_class if r["variant"] == "all" and r["space"] == SPACE}


# ---------------------------------------------------------------------------------------------
# the old grid
# ---------------------------------------------------------------------------------------------


def load_old_tsv(path) -> dict:
    """{(rung, g_version, mark_class, metric, model): {mean, seed_wobble, per_seed, n_tracks}}
    from t118's results_summary.tsv (space counts, kind trained, variant all); {} when no path,
    a missing or empty file, or missing columns (a warning, not an error)."""
    if path is None:
        return {}
    path = Path(path)
    if not path.is_file():
        _warn(f"old-grid TSV {path} not found; no vs_oldgrid readings")
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    if not rows or not set(SUMMARY_COLS) <= set(rows[0]):
        _warn(f"old-grid TSV {path} is empty or lacks t118's columns; no vs_oldgrid readings")
        return {}

    def num(v):
        return float(v) if v not in ("", None) else None

    out = {}
    for r in rows:
        if r["space"] != SPACE or r["kind"] != "trained" or r["variant"] != "all":
            continue
        out[(r["rung"], r["g_version"], r["mark_class"], r["metric"], r["model"])] = {
            "mean": num(r["mean"]), "seed_wobble": num(r["seed_wobble"]),
            "per_seed": [num(r[f"per_seed_{k}"]) for k in range(3)],
            "n_tracks": int(float(r["n_tracks"])) if r["n_tracks"] else None}
    return out


def load_old_json(path) -> tuple[dict, list[dict]]:
    """(split index {(rung, g_version, mark_class, metric, model): per_class row}, explosion rows
    of the real model) from the old grid's results.json per_pair records (trained/score, counts),
    aggregated by the same rule; ({}, []) when no path or no file."""
    if path is None:
        return {}, []
    path = Path(path)
    if not path.is_file():
        _warn(f"old-grid JSON {path} not found; old-grid split and explosions left out")
        return {}, []
    per_pair = [r for r in json.loads(path.read_text("utf-8")).get("per_pair", [])
                if r.get("kind") == "trained" and r.get("eval") == "score"
                and r.get("space") == SPACE]
    per_class = la.per_class_rows(per_track_rows(per_pair, SPLIT_KEYS))
    idx = {(r["rung"], r["g_version"], r["mark_class"], r["metric"], r["model"]): r
           for r in per_class if r["variant"] == "all"}
    return idx, explosion_rows([r for r in per_pair if r.get("model") == "real"])


# ---------------------------------------------------------------------------------------------
# explosions, depth law
# ---------------------------------------------------------------------------------------------


def explosion_rows(per_pair: list[dict]) -> list[dict]:
    """Per (rung, g_version, mark_class, model, kind, eval): n_records, n_crps_gt_20 (finite
    crps_all > 20) and n_crps_nonfinite (crps_all None or not finite), each record once; depthlaw
    records (copies of scored records) are skipped."""
    acc: dict = defaultdict(lambda: [0, 0, 0])
    for r in per_pair:
        if r.get("kind") == "depthlaw" or "crps_all" not in r:
            continue
        key = (r["rung"], r["g_version"], r["mark_class"], r["model"], r["kind"], r["eval"])
        v = r["crps_all"]
        a = acc[key]
        a[0] += 1
        if not _finite(v):
            a[2] += 1
        elif v > EXPLODE_AT:
            a[1] += 1
    return [{"rung": k[0], "g_version": k[1], "mark_class": k[2], "model": k[3], "kind": k[4],
             "eval": k[5], "n_records": a[0], "n_crps_gt_20": a[1], "n_crps_nonfinite": a[2]}
            for k, a in sorted(acc.items())]


def _depth_stat(rows: list[dict]) -> dict:
    """max over pairs of |2^(mean over seeds of log2_pred - log2_true) - 1|, the per-seed max,
    its wobble, the worst pair; records without a prediction are counted, not used."""
    by_pair: dict = defaultdict(dict)
    n_nopred = 0
    for r in rows:
        if r["log2_pred"] is None:
            n_nopred += 1
            continue
        by_pair[(r["source_pid"], r["target_pid"], r["log2_true"])][r["seed"]] = r["log2_pred"]
    if not by_pair:
        return {"value": None, "per_seed": [], "seeds": [], "seed_wobble": None, "n_pairs": 0,
                "worst_pair": None, "n_records_no_pred": n_nopred}
    errs = {pk: abs(2.0 ** (_mean(s.values()) - pk[2]) - 1.0) for pk, s in by_pair.items()}
    seeds = sorted({sd for s in by_pair.values() for sd in s})
    per_seed = [max(abs(2.0 ** (s[sd] - pk[2]) - 1.0) for pk, s in by_pair.items() if sd in s)
                for sd in seeds]
    worst = max(errs, key=errs.get)
    return {"value": errs[worst], "per_seed": per_seed, "seeds": seeds,
            "seed_wobble": wobble(per_seed), "n_pairs": len(errs), "worst_pair": list(worst),
            "n_records_no_pred": n_nopred}


def depth_readings(depth_law: list[dict]) -> list[dict]:
    groups: dict = defaultdict(list)
    for r in depth_law:
        if r["space"] == SPACE:
            groups[(r["rung"], r["g_version"], r["model"], r["kind"])].append(r)
    out = []
    for rung, gv in sorted({k[:2] for k in groups}):
        law = _depth_stat(groups.get((rung, gv, "real", "law"), []))
        tr = _depth_stat(groups.get((rung, gv, "real", "trained"), []))
        noc = _depth_stat(groups.get((rung, gv, "nocov", "law"), []))
        v = law["value"]
        comp = {"n_pairs": law["n_pairs"], "worst_pair": law["worst_pair"],
                "per_seed": law["per_seed"], "seeds": law["seeds"],
                "n_records_no_pred": law["n_records_no_pred"],
                "trained_value": tr["value"], "trained_seed_wobble": tr["seed_wobble"],
                "trained_n_pairs": tr["n_pairs"],
                "nocov_value": noc["value"], "nocov_seed_wobble": noc["seed_wobble"],
                "rule": "met = value <= bar (the depth law holds within 10 %)"}
        if v is None:
            comp["met_reason"] = "no depth->depth law records of the real model"
        out.append({"reading": "depthlaw", "rung": rung, "g_version": gv, "mark_class": "all",
                    "metric": "depth_scale_rel_error", "model": "real", "value": v,
                    "bar": DEPTH_BAR, "met": bool(v is not None and v <= DEPTH_BAR),
                    "seed_wobble": law["seed_wobble"], "components": comp})
    return out


# ---------------------------------------------------------------------------------------------
# readings
# ---------------------------------------------------------------------------------------------


def _row(reading, rung, gv, mc, metric, value, bar, wob, components, model="real") -> dict:
    if value is None or bar is None:
        met = False
        components = {**components, "met_reason": "no value" if value is None else
                      "no bar: the seed wobble needs at least two seeds"}
    else:
        met = bool(value > bar)
    return {"reading": reading, "rung": rung, "g_version": gv, "mark_class": mc,
            "metric": metric, "model": model, "value": value, "bar": bar, "met": met,
            "seed_wobble": wob, "components": components}


def _part(this, other) -> dict:
    """this / other: per_class-like dicts (mean, seed_wobble) or None; gain = other - this."""
    t = None if this is None else this["mean"]
    o = None if other is None else other["mean"]
    tw = None if this is None else this["seed_wobble"]
    ow = None if other is None else other["seed_wobble"]
    gain = None if t is None or o is None else o - t
    bar = _twice(_wmax(tw, ow)) if gain is not None else None
    return {"this": t, "this_wobble": tw, "other": o, "other_wobble": ow, "gain": gain,
            "bar": bar, "gain_exceeds_bar": bool(gain is not None and bar is not None
                                                 and gain > bar)}


def compute_readings(per_class: list[dict], per_pair: list[dict], old: dict,
                     old_split: dict) -> list[dict]:
    idx = _class_index(per_class)
    combos = sorted({(r["rung"], r["g_version"], r["mark_class"]) for r in per_class
                     if r["kind"] == "trained" and r["space"] == SPACE})
    out = []

    def get(rung, gv, model, mc, metric):
        return idx.get((rung, gv, model, "trained", mc, metric))

    def old_get(rung, gv, mc, metric, model="real"):
        if metric in SPLIT_KEYS:
            return old_split.get((rung, gv, mc, metric, model))
        return old.get((rung, gv, mc, metric, model))

    for rung, gv, mc in combos:
        old_rung = "A" if rung == ep.RUNG_OLDSPREAD else rung
        sides = {"beatstwin": lambda m: (get(rung, gv, "real", mc, m),
                                         get(rung, gv, "nocov", mc, m)),
                 "vs_oldgrid": lambda m: (get(rung, gv, "real", mc, m),
                                          old_get(old_rung, gv, mc, m)),
                 "vs_oldspread": lambda m: (get("A", gv, "real", mc, m),
                                            get(ep.RUNG_OLDSPREAD, gv, "real", mc, m))}
        for name in COMPARISONS:
            if name == "vs_oldspread" and rung != "A":
                continue
            any_row = False
            for metric in READING_METRICS:
                this, other = sides[name](metric)
                if this is None or other is None:
                    continue
                any_row = True
                p = _part(this, other)
                comp = {"this": p["this"], "other": p["other"], "wobble_this": p["this_wobble"],
                        "wobble_other": p["other_wobble"],
                        "per_seed_this": this.get("per_seed"),
                        "per_seed_other": other.get("per_seed")}
                if name == "beatstwin":
                    comp.update({"d_real": p["this"], "d_nocov": p["other"], "note": TWIN_NOTE})
                    o_r, o_n = old_get(rung, gv, mc, metric), old_get(rung, gv, mc, metric,
                                                                      "nocov")
                    comp["oldgrid_gap"] = (None if o_r is None or o_n is None
                                           or o_r["mean"] is None or o_n["mean"] is None
                                           else o_n["mean"] - o_r["mean"])
                elif name == "vs_oldgrid":
                    comp.update({"oldgrid_rung": old_rung, "d_t125": p["this"],
                                 "d_oldgrid": p["other"]})
                else:
                    comp.update({"d_A": p["this"], "d_Aold": p["other"]})
                out.append(_row(name, rung, gv, mc, metric, p["gain"], p["bar"],
                                _wmax(p["this_wobble"], p["other_wobble"]), comp))
            if any_row:
                parts = {m: _part(*sides[name](m)) for m in PARTS}
                comp = {"comparison": name, "parts": parts,
                        "gain": "other - this, positive = this better (every part lower-is-better)"}
                if name == "vs_oldgrid":
                    comp["oldgrid_rung"] = old_rung
                    comp["note"] = ("old-grid split from --old-grid-json; the old grid has no "
                                    "MSE")
                if name == "beatstwin":
                    comp["note"] = TWIN_NOTE
                out.append({"reading": "mean_or_spread", "rung": rung, "g_version": gv,
                            "mark_class": mc, "metric": name, "model": "real", "value": None,
                            "bar": None, "met": None, "seed_wobble": None, "components": comp})
    # explosion and non-finite counts, trained/score, per (rung, g_version, class, model)
    for e in explosion_rows(per_pair):
        if e["kind"] != "trained" or e["eval"] != "score":
            continue
        for reading, key in (("n_pairs_crps_gt_20", "n_crps_gt_20"),
                             ("n_pairs_crps_nonfinite", "n_crps_nonfinite")):
            out.append({"reading": reading, "rung": e["rung"], "g_version": e["g_version"],
                        "mark_class": e["mark_class"], "metric": "crps_all", "model": e["model"],
                        "value": e[key], "bar": None, "met": None, "seed_wobble": None,
                        "components": {"n_records": e["n_records"], "kind": "trained",
                                       "eval": "score", "reported_only": True}})
    return out


# ---------------------------------------------------------------------------------------------
# grid cells and grid.md
# ---------------------------------------------------------------------------------------------


def grid_cells(per_class: list[dict], old: dict, old_split: dict) -> list[dict]:
    idx = _class_index(per_class)
    out = []
    for (rung, gv, model, kind, mc, metric), r in sorted(idx.items()):
        if kind != "trained" or metric not in READING_METRICS:
            continue
        cell = {"source": f"t125 {model}", "model": model, "rung": rung, "column": rung[0],
                "row": ep.row_of(rung), "oldspread": rung == ep.RUNG_OLDSPREAD,
                "g_version": gv, "mark_class": mc, "metric": metric, "mean": r["mean"],
                "seed_wobble": r["seed_wobble"], "per_seed": r["per_seed"],
                "n_tracks": r["n_tracks"]}
        if metric == "crps_all":
            cell["split"] = {k: _ms(idx.get((rung, gv, model, kind, mc, k))) for k in SPLIT_KEYS}
        out.append(cell)
    for (rung, gv, mc, metric, model), r in sorted(old.items()):
        if model != "real" or metric not in READING_METRICS or rung not in ep.RUNGS:
            continue
        cell = {"source": "old grid", "model": model, "rung": rung, "column": rung[0],
                "row": ep.row_of(rung), "oldspread": False, "g_version": gv, "mark_class": mc,
                "metric": metric, "mean": r["mean"], "seed_wobble": r["seed_wobble"],
                "per_seed": r["per_seed"], "n_tracks": r["n_tracks"]}
        if metric == "crps_all":
            cell["split"] = {k: _ms(old_split.get((rung, gv, mc, k, model))) for k in SPLIT_KEYS}
        out.append(cell)
    return out


def _ms(r):
    return None if r is None else {"mean": r["mean"], "seed_wobble": r["seed_wobble"]}


def _f(v) -> str:
    return "—" if v is None else f"{v:.4g}"


def _pm(m, w) -> str:
    if m is None:
        return "—"
    return f"{_f(m)} ± {_f(w)}" if w is not None else f"{_f(m)} (one seed)"


def _cell_text(c) -> str:
    if c is None:
        return "—"
    s = _pm(c["mean"], c["seed_wobble"])
    if c["metric"] == "crps_all":
        sp = c.get("split") or {}
        o, e = sp.get("crps_oracle_scaled_all"), sp.get("scale_error_all")
        if o is None or e is None:
            s += "; split: not available (the old grid's needs --old-grid-json)"
        else:
            s += (f"; oracle-scaled {_pm(o['mean'], o['seed_wobble'])} + scale error "
                  f"{_pm(e['mean'], e['seed_wobble'])}")
    return s


def _yn(b) -> str:
    return "yes" if b else "no"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    return (["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
            + ["| " + " | ".join(r) + " |" for r in rows] + [""])


GRID_ROWS = (("row 1: g reads (C, C')", "t125 real", 1),
             ("row 2: g reads (eta_in, C, C')", "t125 real", 2),
             ("old grid row 1: no depth offset", "old grid", 1),
             ("old grid row 2: no depth offset", "old grid", 2),
             ("twin row 1: no covariates, keeps the offset", "t125 nocov", 1),
             ("twin row 2: no covariates, keeps the offset", "t125 nocov", 2))
AOLD_ROWS = (("Aold: row-1 A with the old spread (one per pair) and the offset", "t125 real"),
             ("Aold twin: no covariates, keeps the offset", "t125 nocov"))


def write_grid_md(path: Path, cells: list[dict], readings: list[dict], explosions: list[dict],
                  old_explosions: list[dict], depth: list[dict], n_present: int,
                  n_missing: int) -> None:
    ci = {(c["source"], c["rung"], c["g_version"], c["mark_class"], c["metric"]): c
          for c in cells}
    ex = {(e["rung"], e["g_version"], e["mark_class"], e["model"]): e for e in explosions
          if e["kind"] == "trained" and e["eval"] == "score"}
    oex = {(e["rung"], e["g_version"], e["mark_class"]): e for e in old_explosions}
    L = ["# t125 part 1: the count grid with a depth offset", "",
         "Columns are the four forms of f (A-D). Row 1: g reads the covariates of the source and "
         "the wanted track (C, C'). Row 2: g also reads the bin's eta_in = log(1 + X) - log d. "
         "Every t125 run has the depth offset: f reads eta_in and the target counts are NB with "
         "log mu' = log d' + eta_out. The old grid (t118 row 2) had no offset. Aold is row-1 A "
         "with the old spread (one value per pair) and the offset: Aold against the old grid's A "
         "isolates the offset, A against Aold isolates the new spread rule.", "",
         "Each cell: trained pairs, held-out chromosomes chr19 + chr21, all pairs kept; the mean "
         "over the 3 seeds of the macro mean over the class's tracks ± the seed wobble (the "
         "largest pairwise difference over the seeds). CRPS is the NB CRPS on counts; lower is "
         "better. CRPS on all bins carries its oracle-scaled + scale-error split; the split is "
         "not computed on the top 1 % bins. The MSE is on log(1 + counts) of the predicted mean; "
         "the old grid has no MSE.", "",
         f"Twin rows: {TWIN_NOTE}.", "",
         f"Runs present: {n_present}; missing: {n_missing} (listed in results.json).", "",
         "Nothing in this file is a verdict. A reading marked \"gain > 2 × wobble: yes\" only "
         "says the gain is larger than 2 × the larger seed wobble of the two arms.", ""]
    gvs = [g for g in ("per_track", "across") if any(c["g_version"] == g for c in cells)]
    for gv in gvs:
        L += [f"## {GV_LABEL[gv]}, counts", ""]
        classes = [m for m in CLASS_ORDER if any(c["g_version"] == gv and c["mark_class"] == m
                                                 for c in cells)]
        for mc in classes:
            L += [f"### {mc}", ""]
            for metric in READING_METRICS:
                rows = []
                for label, src, row in GRID_ROWS:
                    rows.append([label] + [_cell_text(ci.get((src, col if row == 1 else f"{col}2",
                                                              gv, mc, metric)))
                                           for col in COLUMNS])
                for label, src in AOLD_ROWS:
                    rows.append([label, _cell_text(ci.get((src, ep.RUNG_OLDSPREAD, gv, mc,
                                                           metric))), "", "", ""])
                L += [f"**{METRIC_LABEL[metric]}**", ""]
                L += _table([""] + [COLUMN_LABEL[c] for c in COLUMNS], rows)
            L += ["(pair, seed) trained records: CRPS > 20 / non-finite CRPS / all records "
                  "(reported only; non-finite records are left out of every mean above):", ""]
            erows = []
            for label, src, row in GRID_ROWS:
                model = src.split()[-1]
                cells_e = []
                for col in COLUMNS:
                    rung = col if row == 1 else f"{col}2"
                    e = (oex.get((rung, gv, mc)) if src == "old grid"
                         else ex.get((rung, gv, mc, model)))
                    cells_e.append("—" if e is None else
                                   f"{e['n_crps_gt_20']} / {e['n_crps_nonfinite']} / "
                                   f"{e['n_records']}")
                erows.append([label] + cells_e)
            for label, src in AOLD_ROWS:
                e = ex.get((ep.RUNG_OLDSPREAD, gv, mc, src.split()[-1]))
                erows.append([label, "—" if e is None else
                              f"{e['n_crps_gt_20']} / {e['n_crps_nonfinite']} / "
                              f"{e['n_records']}", "", "", ""])
            L += _table([""] + [COLUMN_LABEL[c] for c in COLUMNS], erows)
    # depth law
    L += ["## Depth law", "",
          "Depth-only pairs, real g: max over the never-trained depth->depth pairs of "
          "|2^(mean over seeds of (log2 predicted total / source total - log2 d'/d)) - 1|, with "
          "the per-seed maximum's wobble; beside it the same over the trained base<->depth pairs "
          "and the twin's value. Bar 0.10 (reported, as t118).", ""]
    L += _table(["rung", "g", "depth->depth law pairs", "seed wobble", "n pairs",
                 "within 0.10", "trained base<->depth", "twin (law pairs)"],
                [[d["rung"], GV_LABEL[d["g_version"]], _f(d["value"]), _f(d["seed_wobble"]),
                  str(d["components"]["n_pairs"]), _yn(d["met"]),
                  _pm(d["components"]["trained_value"], d["components"]["trained_seed_wobble"]),
                  _pm(d["components"]["nocov_value"], d["components"]["nocov_seed_wobble"])]
                 for d in depth])
    # readings
    titles = {"vs_oldgrid": "Each design against the same design in the old grid "
                            "(old - t125; Aold against the old grid's A)",
              "beatstwin": "Each model against its no-covariates twin (twin - real)",
              "vs_oldspread": "A against Aold: the new spread rule (Aold - A)"}
    for name in ("vs_oldgrid", "beatstwin", "vs_oldspread"):
        rs = [r for r in readings if r["reading"] == name]
        L += [f"## {titles[name]}", ""]
        if name == "beatstwin":
            L += [f"Note: {TWIN_NOTE}.", ""]
        L += ["Positive = the first-named arm is better (lower). Bar = 2 × the larger seed "
              "wobble.", ""]
        L += _table(["rung", "g", "class", "metric", "gain", "larger seed wobble", "bar",
                     "gain > 2 × wobble"],
                    [[r["rung"], GV_LABEL[r["g_version"]], r["mark_class"], r["metric"],
                      _f(r["value"]), _f(r["seed_wobble"]), _f(r["bar"]), _yn(r["met"])]
                     for r in rs])
    L += ["## Mean or spread", "",
          "Per comparison: the gain on CRPS (all bins), on its oracle-scaled part and its "
          "scale-error part, and on the MSE of the predicted mean (log(1 + counts)), each with "
          "its own bar (2 × the larger seed wobble; * = gain exceeds the bar). A gain in the MSE "
          "is a gain in the mean; a CRPS gain with no MSE gain sits in the spread.", ""]
    ms_parts = ("crps_all", "crps_oracle_scaled_all", "scale_error_all", "mse_log1p_all",
                "mse_log1p_top1")
    rows = []
    for r in readings:
        if r["reading"] != "mean_or_spread":
            continue
        ps = r["components"]["parts"]
        rows.append([r["metric"], r["rung"], GV_LABEL[r["g_version"]], r["mark_class"]]
                    + [_f(ps[k]["gain"]) + ("*" if ps[k]["gain_exceeds_bar"] else "")
                       for k in ms_parts])
    L += _table(["comparison", "rung", "g", "class", "CRPS all", "oracle-scaled",
                 "scale error", "MSE all", "MSE top 1 %"], rows)
    path.write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------------------------------------
# the aggregate command
# ---------------------------------------------------------------------------------------------


def _tsv(path: Path, cols, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            w.writerow(["" if r.get(c) is None else r.get(c) for c in cols])


def aggregate(manifest, covariates, runs_dir, agg_dir, old_grid_tsv=None,
              old_grid_json=None) -> dict:
    ep.read_manifest(manifest)               # fail early on a bad manifest
    agg_dir = Path(agg_dir)
    agg_dir.mkdir(parents=True, exist_ok=True)
    per_pair, present, no_law, figdata = la.load_runs(runs_dir)
    expected = [t["run_name"] for t in ep.tasks()]
    missing = [n for n in expected if n not in set(present)]
    if missing:
        _warn(f"{len(missing)} of {len(expected)} expected runs missing (listed in results.json)")
    old = load_old_tsv(old_grid_tsv)
    old_split, old_explosions = load_old_json(old_grid_json)
    per_track = per_track_rows(per_pair)
    per_class = la.per_class_rows(per_track)
    depth_law = la.depth_law_rows(per_pair)
    depth = depth_readings(depth_law)
    readings = depth + compute_readings(per_class, per_pair, old, old_split)
    explosions = explosion_rows(per_pair)
    cells = grid_cells(per_class, old, old_split)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    results = {"created_utc": now, "runs_dir": str(runs_dir), "manifest": str(manifest),
               "covariates": str(covariates),
               "old_grid_tsv": None if old_grid_tsv is None else str(old_grid_tsv),
               "old_grid_json": None if old_grid_json is None else str(old_grid_json),
               "n_expected_runs": len(expected), "runs_present": present,
               "runs_missing": missing, "runs_without_law": no_law,
               "explode_at": EXPLODE_AT, "twin_note": TWIN_NOTE,
               "per_pair": [la._slim(r) for r in per_pair if la._keep_in_results(r)],
               "per_track": per_track, "per_class": per_class, "readings": readings,
               "depth_law": depth_law, "explosions": explosions,
               "oldgrid_explosions": old_explosions, "grid": cells, "figdata": figdata}
    write_json(results, agg_dir / "results.json", indent=None)
    _tsv(agg_dir / "results_summary.tsv", SUMMARY_COLS,
         [{**r, **{f"per_seed_{k}": r["per_seed"][k] for k in range(3)}} for r in per_class])
    _tsv(agg_dir / "pairs_summary.tsv", PAIRS_COLS,
         [r for r in per_pair if r.get("kind") == "trained" and r.get("eval") == "score"])
    _tsv(agg_dir / "depth_law.tsv", DEPTH_COLS,
         [{**r, "depth_source": r["kind"], "depth_log2_ratio_true": r["log2_true"],
           "depth_log2_scale_pred": r["log2_pred"]} for r in depth_law])
    write_grid_md(agg_dir / "grid.md", cells, readings, explosions, old_explosions, depth,
                  len(present), len(missing))
    write_json({"created_utc": now, "runs_present": len(present), "runs_missing": len(missing),
                "n_expected_runs": len(expected), "explode_at": EXPLODE_AT,
                "twin_note": TWIN_NOTE, "cells": cells, "readings": readings,
                "depth_law": depth, "explosions": explosions,
                "oldgrid_explosions": old_explosions},
               agg_dir / "summary_data.json")
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for pos in ("manifest", "covariates", "runs_dir", "agg_dir"):
        ap.add_argument(pos, type=Path)
    ap.add_argument("--old-grid-tsv", type=Path, default=None,
                    help="t118 row-2 results_summary.tsv (the old grid's class means)")
    ap.add_argument("--old-grid-json", type=Path, default=None,
                    help="t118 row-2 agg/results.json (the old grid's per-pair split)")
    a = ap.parse_args(argv)
    res = aggregate(a.manifest, a.covariates, a.runs_dir, a.agg_dir, a.old_grid_tsv,
                    a.old_grid_json)
    print(f"{len(res['runs_present'])} runs present, {len(res['runs_missing'])} missing; "
          f"{len(res['readings'])} readings -> {a.agg_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
