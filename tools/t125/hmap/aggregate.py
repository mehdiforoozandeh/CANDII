"""t125 part 2 — aggregate every scored h (alone and in the chain) to the part-2 readings of handoff
Phase 7: the complexity x inputs grid, push only against push + own noise, the repair reading (h
versions compared on f's NB), and the chain against the old grid's p results.

**Inputs** (`hmap/score.py`, chunk t125-P2f): `<h_root>/alone/<h_name>/scores.json` (one record per
usable product, h on the true X') and `<h_root>/chain/<f_run>/<h_name>/scores.json` (one record per
training pair of the f, h on f's predicted NB, f = seed 0 of each real f). Each file carries the h
(`h`: variant, rung, noise, source, seed) and, in the chain, the f (`f_run`: rung, g_version).

**h versions.** A version is an h without its seed: `h_formula_bin`, `h_formula_frag` (no seed),
and `h_<variant>_r<rung>_<push|noise>_<true|fnb>` (true: h seeds 0..2; fnb: one h seed per f, PI
decision (ii) b). In a per-track f cell the fnb version collects the 7 fnb h of the 7 per-track f.

**Aggregation.** Alone (plan): per seed, the mean over the products of a mark class; chain (as
t118, like-with-like with the old grid): per seed, the mean over the pairs of a track, then the
macro mean over the tracks of the class. Then `mean` over the seeds and `seed_wobble` = the largest
pairwise |delta| of the per-seed values (`ladder.aggregate.wobble`; None with one seed). A metric
that is None in a record (not finite, or no bins) is left out of that metric's mean and counted.

**Metrics** (design note "Scoring"; PI decisions 2026-10-11). Push only: `crps_all`, `crps_top1`
(proper CRPS of the pushed set on P) with `crps_oracle_scaled_all` / `scale_error_all`. Own noise:
`crps_mean_point_all`, `crps_mean_point_top1` (the point CRPS of the P-scale mean; never put
beside push's proper CRPS as the same number, never compared with a proper CRPS) with
`crps_mean_point_oracle_scaled_all` / `crps_mean_point_scale_error_all`. Both: `crps_log_all`,
`crps_log_top1` (proper, on log(1 + P)) with `crps_log_oracle_scaled_all` /
`crps_log_scale_error_all`, and `mse_log1p_all`, `mse_log1p_top1` (log(1 + P) of the mean).

**Readings** (every metric lower-is-better; `gain` positive = the named winner side is lower).
A comparison of two arms uses only the products / pairs present in every seed of both arms and
finite in every one of those records (no `n_*_nonfinite` > 0, every compared metric finite; PI
2026-10-11); `components.pairs` counts the used, dropped-as-not-finite and one-arm-only keys.
`bar` = 2 x the larger seed wobble of the two arms; an arm with one seed (fnb h) takes the seed
wobble of the true-X' h of the same version, f cell, class and metric (PI decision (ii) b: one fnb
seed, the spread quoted from the true-X' h); a formula h has no h seed and adds no wobble.
`gain_exceeds_bar` only says the gain is larger than the bar; no row is a conclusion.
  push_vs_noise  alone and per chain f cell, per (variant, rung, source): gain = push - noise on
                 crps_log_all / crps_log_top1 (with the log split's gains) and mse_log1p_all /
                 mse_log1p_top1 (PI: the two variants are compared on the log scale + the MSE).
  repair         per chain f cell x class x (variant, rung, noise): the true-X'-trained h against
                 the fnb-trained h, both in the chain on the same f; delta = true - fnb (positive
                 = the fnb-trained h is lower); `abs_delta_exceeds_bar` = |delta| > bar (plan).
  vs_oldgrid     per chain f cell (rung, g_version) x class x h version: the old grid's p-space
                 real-f values (`--old-grid-tsv`: space pval, kind trained, variant all, model
                 real) beside the chain h; delta = old - h (positive = the chain h is lower). With
                 `--old-grid-json` (t118 row-2 agg/results.json) the two are pair-matched and the
                 old split and explosion counts are read from its per_pair pval records; with the
                 TSV alone the class means are compared as they are, and a row whose track counts
                 differ is kept with gain_exceeds_bar no and the reason. An own-noise h is shown
                 (its point CRPS) with no delta: a point CRPS is not compared with a proper CRPS.
**Explosions** per (scope, f cell, h version, class): the (pair or product, seed) records whose
flag of the h's kind is set (`crps_all_gt_20` push, `crps_mean_point_all_gt_20` own noise), and,
apart, the records with a non-finite native bin and with a non-finite log-scale bin. Reported
only; a non-finite record never enters a mean. **Capped NB rows** per f run (`n_capped`, shared by
every h of that f).

**Outputs** in `<agg_dir>`: results_h.json, h_alone_summary.tsv, h_chain_summary.tsv,
h_repair.tsv, h_vs_oldgrid.tsv, grid_h.md, summary_data_h.json. Missing h runs (of
`tasks.tasks_alone()` and, per f in `eta.pairs.f_runs_seed0()`, `tasks.chain_h_list`) are listed in
results_h.json, not fatal; comparisons with no reading go to `readings_skipped` with the reason.

    python tools/t125/hmap/aggregate.py <manifest> <covariates.tsv> <h_root> <agg_dir>
                                        --old-grid-tsv PATH [--old-grid-json PATH]
                                        [--eta-runs DIR] [--fnb-seeds 1|3]
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
from ladder import pairs as lpairs  # noqa: E402
from ladder.score import write_json  # noqa: E402
from eta import pairs as ep  # noqa: E402
from hmap import tasks  # noqa: E402

EXPLODE_AT = 20.0
CLASS_ORDER = ("DNase", "narrow", "broad")
NAT = {False: "crps", True: "crps_mean_point"}
LOG_KEYS = ("crps_log_all", "crps_log_top1")
LOG_SPLIT = ("crps_log_oracle_scaled_all", "crps_log_scale_error_all")
MSE_KEYS = ("mse_log1p_all", "mse_log1p_top1")
PVN_READ = LOG_KEYS + MSE_KEYS
OLD_METRICS = ("crps_all", "crps_top1", "crps_oracle_scaled_all", "scale_error_all")
OLD_SPACE = "pval"
GV_LABEL = {"per_track": "one g per track", "across": "one g across tracks"}
WOBBLE_FNB = ("one fnb h seed: the seed wobble is the true-X' h's of the same version, f cell, "
              "class and metric (PI)")
WOBBLE_FORMULA = "formula h: no h seed, no wobble of its own"
NOISE_VS_OLD = ("own noise: its native number is the point CRPS of the mean; it is shown, not "
                "compared with the old grid's proper CRPS")

ALONE_COLS = ("h_name", "variant", "rung", "noise", "source", "mark_class", "metric", "mean",
              "seed_wobble", "per_seed_0", "per_seed_1", "per_seed_2", "n_products",
              "explosion_kind", "n_products_crps_gt_20", "n_products_crps_nonfinite",
              "n_products_crps_log_nonfinite")
CHAIN_COLS = ("f_rung", "f_g_version", "h_name", "variant", "rung", "noise", "source",
              "mark_class", "metric", "mean", "seed_wobble", "per_seed_0", "per_seed_1",
              "per_seed_2", "n_pairs", "n_tracks", "explosion_kind", "n_pairs_crps_gt_20",
              "n_pairs_crps_nonfinite", "n_pairs_crps_log_nonfinite")
REPAIR_COLS = ("f_rung", "f_g_version", "mark_class", "variant", "rung", "noise", "metric",
               "true_h", "fnb_h", "true_mean", "true_wobble", "fnb_mean", "delta", "bar",
               "abs_delta_exceeds_bar", "gain_exceeds_bar", "n_pairs_used",
               "n_pairs_dropped_not_finite", "n_pairs_only_one_arm", "note")
VS_COLS = ("f_rung", "f_g_version", "mark_class", "h_name", "variant", "rung", "noise", "source",
           "metric", "h_metric", "old_mean", "old_wobble", "h_mean", "h_wobble", "delta", "bar",
           "gain_exceeds_bar", "matched", "n_pairs_used", "n_pairs_dropped_not_finite",
           "n_pairs_only_one_arm", "old_tsv_mean", "old_tsv_wobble", "old_n_tracks",
           "h_n_tracks", "old_n_crps_gt_20", "old_n_crps_nonfinite", "h_n_crps_gt_20",
           "h_n_crps_nonfinite", "note")

_mean, wobble, _warn = la._mean, la.wobble, la._warn


def _finite(v) -> bool:
    return v is not None and not isinstance(v, bool) and math.isfinite(float(v))


def _skey(s):
    return -1 if s is None else int(s)


def _bar(*ws):
    av = [w for w in ws if w is not None]
    return 2.0 * max(av) if av else None


def _exceeds(gain, bar) -> bool:
    return bool(gain is not None and bar is not None and gain > bar)


def _diff(a, b):
    return None if a is None or b is None else a - b


def split_keys(noise: bool) -> tuple:
    return (("crps_mean_point_oracle_scaled_all", "crps_mean_point_scale_error_all") if noise
            else ("crps_oracle_scaled_all", "scale_error_all"))


def native_keys(noise: bool) -> tuple:
    k = NAT[bool(noise)]
    return (f"{k}_all", f"{k}_top1") + split_keys(noise)


def metrics_of(noise: bool) -> tuple:
    return native_keys(noise) + LOG_KEYS + LOG_SPLIT + MSE_KEYS


def _classes(*arms) -> list[str]:
    got = {r.get("mark_class") for a in arms if a for recs in a["seeds"].values()
           for r in recs.values()}
    got.discard(None)
    return [c for c in CLASS_ORDER if c in got] + sorted(got - set(CLASS_ORDER))


# ---------------------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------------------


def version_of(h: dict) -> dict:
    """The seed-free version of a scores.json `h` block (see the module docstring)."""
    if h.get("variant") == "formula" or h.get("kind") == "formula":
        window = h.get("window") or str(h.get("h_name", "")).rsplit("_", 1)[-1]
        return {"h_name": f"h_formula_{window}", "variant": "formula", "window": window,
                "rung": None, "noise": 0, "source": "formula", "seed": None, "fnb_f": None}
    src = str(h.get("source") or "")
    if src == "true":
        source, fnb_f = "true", None
    elif src.startswith("fnb"):
        source = "fnb"
        fnb_f = (src.split(":", 1)[1] if ":" in src else
                 Path(h["fnb_run"]).name if h.get("fnb_run") else None)
    else:
        raise ValueError(f"h {h.get('h_name')!r}: unknown source {src!r}")
    rung, noise = int(h["rung"]), int(h["noise"])
    return {"h_name": f"h_{h['variant']}_r{rung}_{'noise' if noise else 'push'}_{source}",
            "variant": h["variant"], "window": None, "rung": rung, "noise": noise,
            "source": source, "seed": None if h.get("seed") is None else int(h["seed"]),
            "fnb_f": fnb_f}


def _new_arm(v: dict) -> dict:
    return {"info": {k: v[k] for k in ("h_name", "variant", "window", "rung", "noise", "source")},
            "seeds": {}, "h_names": [], "f_runs": set()}


def _read(path: Path):
    try:
        return json.loads(path.read_text("utf-8"))
    except (OSError, ValueError) as e:
        _warn(f"{path} unreadable ({e}); left out")
        return None


def _add(arm: dict, seed, key, r: dict, warnings: list, where: str) -> None:
    recs = arm["seeds"].setdefault(seed, {})
    if key in recs:
        warnings.append(f"{where}: duplicate record {key} for {arm['info']['h_name']} seed "
                        f"{seed}; the first kept")
        return
    recs[key] = r


def load_alone(h_root: Path, warnings: list) -> tuple[dict, list[str]]:
    """({version: arm}, the h names present); arm seeds -> {pid: record}."""
    arms: dict = {}
    present = []
    for f in sorted((Path(h_root) / "alone").glob("*/scores.json")):
        obj = _read(f)
        if not obj:
            continue
        v = version_of(obj["h"])
        arm = arms.setdefault(v["h_name"], _new_arm(v))
        name = obj["h"].get("h_name") or f.parent.name
        arm["h_names"].append(name)
        present.append(name)
        arm["seeds"].setdefault(v["seed"], {})
        for r in obj.get("records", []):
            _add(arm, v["seed"], r["pid"], r, warnings, str(f))
    return arms, present


def _f_cell(obj: dict, f_run: str) -> tuple[str, str]:
    fi = obj.get("f_run") or {}
    rung, gv = fi.get("rung"), fi.get("g_version")
    if rung is None or gv is None:
        parts = f_run.split("_")
        rung = rung or parts[0]
        gv = gv or ("across" if parts[1] == "all" else "per_track")
    return rung, gv


def pair_key(r: dict) -> tuple:
    return (r["track"], r["source_pid"], r["target_pid"])


def load_chain(h_root: Path, warnings: list, skipped: list):
    """({(f_rung, f_g_version): {version: arm}}, {f_run: [h names]}, capped rows,
    {f_run: {h_name: n_records}}); arm seeds -> {(track, source, target): record}."""
    cells: dict = defaultdict(dict)
    present: dict = defaultdict(list)
    n_recs: dict = defaultdict(dict)
    capped: dict = {}
    for f in sorted((Path(h_root) / "chain").glob("*/*/scores.json")):
        f_run = f.parent.parent.name
        obj = _read(f)
        if not obj:
            continue
        rung, gv = _f_cell(obj, f_run)
        v = version_of(obj["h"])
        name = obj["h"].get("h_name") or f.parent.name
        present[f_run].append(name)
        recs = obj.get("records", [])
        n_recs[f_run][name] = len(recs)
        if v["source"] == "fnb" and v["fnb_f"] != f_run:
            skipped.append({"reading": "chain_load", "scope": "chain", "f_run": f_run,
                            "h_name": name, "reason": f"fnb h trained on f {v['fnb_f']!r}, "
                                                      f"scored in the chain of {f_run!r}"})
            continue
        arm = cells[(rung, gv)].setdefault(v["h_name"], _new_arm(v))
        arm["h_names"].append(name)
        arm["f_runs"].add(f_run)
        arm["seeds"].setdefault(v["seed"], {})
        for r in recs:
            _add(arm, v["seed"], pair_key(r), r, warnings, str(f))
        cap = {"f_run": f_run, "f_rung": rung, "f_g_version": gv, "n_pairs": len(recs),
               "n_pairs_capped": sum((r.get("n_capped") or 0) > 0 for r in recs),
               "n_capped_rows": sum(int(r.get("n_capped") or 0) for r in recs)}
        if f_run not in capped:
            capped[f_run] = {**cap, "consistent_across_h": True, "from_h": name}
        elif any(capped[f_run][k] != cap[k] for k in ("n_pairs", "n_pairs_capped",
                                                      "n_capped_rows")):
            capped[f_run]["consistent_across_h"] = False
    return dict(cells), dict(present), [capped[k] for k in sorted(capped)], dict(n_recs)


def load_old_tsv(path) -> dict:
    """{(rung, g_version, mark_class, metric): {mean, seed_wobble, per_seed, n_tracks}} from t118's
    results_summary.tsv (space pval, kind trained, variant all, model real); {} when no path, a
    missing or empty file, or missing columns (a warning, not an error)."""
    if path is None:
        return {}
    path = Path(path)
    if not path.is_file():
        _warn(f"old-grid TSV {path} not found; no vs_oldgrid readings")
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    if not rows or not set(la.SUMMARY_COLS) <= set(rows[0]):
        _warn(f"old-grid TSV {path} is empty or lacks t118's columns; no vs_oldgrid readings")
        return {}

    def num(v):
        return float(v) if v not in ("", None) else None

    out = {}
    for r in rows:
        if (r["space"], r["kind"], r["variant"], r["model"]) != (OLD_SPACE, "trained", "all",
                                                                 "real"):
            continue
        out[(r["rung"], r["g_version"], r["mark_class"], r["metric"])] = {
            "mean": num(r["mean"]), "seed_wobble": num(r["seed_wobble"]),
            "per_seed": [num(r[f"per_seed_{k}"]) for k in range(3)],
            "n_tracks": int(float(r["n_tracks"])) if r["n_tracks"] else None}
    return out


OLD_KEEP = ("rung", "g_version", "seed", "track", "mark_class", "source_pid", "target_pid",
            "n_crps_nonfinite") + OLD_METRICS


def load_old_json(path) -> tuple[dict, dict]:
    """({(rung, g_version): arm of the old grid's real f, seeds -> {pair key: record}},
    {(rung, g_version, mark_class): explosion counts}) from the old grid's results.json per_pair
    records (kind trained, eval score, space pval, model real); ({}, {}) when absent."""
    if path is None:
        return {}, {}
    path = Path(path)
    if not path.is_file():
        _warn(f"old-grid JSON {path} not found; the old grid is not pair-matched")
        return {}, {}
    arms: dict = {}
    ex: dict = defaultdict(lambda: {"n_records": 0, "n_crps_gt_20": 0, "n_crps_nonfinite": 0})
    warnings: list = []
    for r in json.loads(path.read_text("utf-8")).get("per_pair", []):
        if (r.get("kind"), r.get("eval"), r.get("space"), r.get("model")) != (
                "trained", "score", OLD_SPACE, "real"):
            continue
        r = {k: r[k] for k in OLD_KEEP if k in r}
        cell = (r["rung"], r["g_version"])
        arm = arms.setdefault(cell, {"info": {"h_name": "old grid"}, "seeds": {}})
        _add(arm, int(r["seed"]), pair_key(r), r, warnings, str(path))
        e = ex[(r["rung"], r["g_version"], r["mark_class"])]
        e["n_records"] += 1
        if (r.get("n_crps_nonfinite") or 0) > 0:
            e["n_crps_nonfinite"] += 1
        elif _finite(r.get("crps_all")) and r["crps_all"] > EXPLODE_AT:
            e["n_crps_gt_20"] += 1
    for w in warnings:
        _warn(w)
    return arms, dict(ex)


# ---------------------------------------------------------------------------------------------
# class values, summaries, explosions
# ---------------------------------------------------------------------------------------------


def class_value(recs: dict, metric: str, mc: str, mode: str, keys=None) -> tuple:
    """(value, n records used, n tracks) of one seed: alone = mean over the class's products;
    chain = mean over each track's pairs, then over the class's tracks."""
    by_t: dict = defaultdict(list)
    for k, r in recs.items():
        if r.get("mark_class") != mc or (keys is not None and k not in keys):
            continue
        if _finite(r.get(metric)):
            by_t[r.get("track")].append(float(r[metric]))
    n = sum(len(v) for v in by_t.values())
    if not n:
        return None, 0, 0
    if mode == "alone":
        return _mean([x for v in by_t.values() for x in v]), n, len(by_t)
    return _mean([_mean(v) for v in by_t.values()]), n, len(by_t)


def explosion_row(arm: dict, mc: str) -> dict:
    noise = bool(arm["info"]["noise"])
    k = NAT[noise]
    out = {"kind": f"{k}_all_gt_20", "n_records": 0, "n_gt_20": 0, "n_native_nonfinite": 0,
           "n_log_nonfinite": 0, "native_nonfinite_key": f"n_{k}_nonfinite"}
    for recs in arm["seeds"].values():
        for r in recs.values():
            if r.get("mark_class") != mc:
                continue
            out["n_records"] += 1
            out["n_gt_20"] += bool(r.get(f"{k}_all_gt_20"))
            out["n_native_nonfinite"] += (r.get(f"n_{k}_nonfinite") or 0) > 0
            out["n_log_nonfinite"] += (r.get("n_crps_log_nonfinite") or 0) > 0
    return out


def summary_rows(arm: dict, mode: str, extra: dict) -> list[dict]:
    info = arm["info"]
    seeds = sorted(arm["seeds"], key=_skey)
    out = []
    for mc in _classes(arm):
        ex = explosion_row(arm, mc)
        for m in metrics_of(bool(info["noise"])):
            per = {s: class_value(arm["seeds"][s], m, mc, mode) for s in seeds}
            vals = [per[s][0] for s in seeds]
            if all(v is None for v in vals):
                continue
            out.append({**extra, **info, "mark_class": mc, "metric": m, "mean": _mean(vals),
                        "seed_wobble": wobble(vals), "seeds": seeds, "per_seed_list": vals,
                        "per_seed": {f"per_seed_{s}": per[s][0] for s in seeds
                                     if s is not None},
                        "n": max(per[s][1] for s in seeds),
                        "n_tracks": max(per[s][2] for s in seeds), "explosions": ex})
    return out


def _summary_index(rows: list[dict]) -> dict:
    return {(r.get("f_rung"), r.get("f_g_version"), r["h_name"], r["mark_class"], r["metric"]): r
            for r in rows}


# ---------------------------------------------------------------------------------------------
# two arms on common ground
# ---------------------------------------------------------------------------------------------


def _rec_ok(r: dict, metrics) -> bool:
    if any((v or 0) > 0 for k, v in r.items()
           if k.startswith("n_") and k.endswith("_nonfinite") and not isinstance(v, bool)):
        return False
    return all(_finite(r[m]) for m in metrics if m in r)


def compare(x: dict, y: dict, metrics, mc: str, mode: str) -> tuple[dict | None, dict]:
    """Arms x and y (`seeds -> {key: record}`) on class `mc` over the keys present in every seed
    of both and finite in all those records. ({"x"|"y": {metric: {mean, seed_wobble, per_seed,
    seeds}}}, diagnostics) or (None, diagnostics with a reason)."""

    def keys(arm):
        sets = [{k for k, r in recs.items() if r.get("mark_class") == mc}
                for recs in arm["seeds"].values()]
        return (set.intersection(*sets), set.union(*sets)) if sets else (set(), set())

    ix, ux = keys(x)
    iy, uy = keys(y)
    common = ix & iy
    ok = {k for k in common
          if all(_rec_ok(recs[k], metrics) for recs in x["seeds"].values())
          and all(_rec_ok(recs[k], metrics) for recs in y["seeds"].values())}
    diag = {"n_common": len(common), "n_used": len(ok),
            "n_dropped_not_finite": len(common) - len(ok),
            "n_only_one_arm": len(ux | uy) - len(common),
            "rule": "the keys present in every seed of both arms, finite in all those records"}
    if not ok:
        diag["reason"] = (f"no {'product' if mode == 'alone' else 'pair'} of class {mc} is in "
                          "both arms and finite in both")
        return None, diag
    out = {}
    for side, arm in (("x", x), ("y", y)):
        seeds = sorted(arm["seeds"], key=_skey)
        out[side] = {}
        for m in metrics:
            per = [class_value(arm["seeds"][s], m, mc, mode, ok)[0] for s in seeds]
            out[side][m] = {"mean": _mean(per), "seed_wobble": wobble(per), "per_seed": per,
                            "seeds": seeds}
    return out, diag


def _part(winner, other, w_sub=None, o_sub=None) -> dict:
    """gain = other - winner (positive = `winner` lower); a missing wobble takes its substitute."""
    w = winner["mean"] if winner else None
    o = other["mean"] if other else None
    ww = winner["seed_wobble"] if winner else None
    ow = other["seed_wobble"] if other else None
    ww = ww if ww is not None else w_sub
    ow = ow if ow is not None else o_sub
    gain = _diff(o, w)
    bar = _bar(ww, ow) if gain is not None else None
    return {"this": w, "this_wobble": ww, "other": o, "other_wobble": ow, "gain": gain,
            "bar": bar, "gain_exceeds_bar": _exceeds(gain, bar)}


# ---------------------------------------------------------------------------------------------
# readings
# ---------------------------------------------------------------------------------------------


def _learned_versions():
    return [(v, r, n) for v, r, n in tasks.H_LEARNED]


def _vname(variant, rung, noise, source) -> str:
    return f"h_{variant}_r{rung}_{'noise' if noise else 'push'}_{source}"


def push_vs_noise(scope: str, cell, arms: dict, sidx: dict, mode: str) -> tuple[list, list]:
    out, skipped = [], []
    f_rung, f_gv = cell if cell else (None, None)
    combos = sorted({(v, r) for v, r, _ in _learned_versions()})
    for source in ("true", "fnb"):
        for variant, rung in combos:
            push = arms.get(_vname(variant, rung, 0, source))
            noise = arms.get(_vname(variant, rung, 1, source))
            if not push and not noise:
                continue
            base = {"reading": "push_vs_noise", "scope": scope, "f_rung": f_rung,
                    "f_g_version": f_gv, "variant": variant, "rung": rung, "noise": None,
                    "source": source}
            if not push or not noise:
                skipped.append({**base, "reason": "only the "
                                + ("push" if push else "own-noise") + " arm was scored"})
                continue
            for mc in _classes(push, noise):
                res, diag = compare(push, noise, PVN_READ + LOG_SPLIT, mc, mode)
                if res is None:
                    skipped.append({**base, "mark_class": mc, "reason": diag["reason"],
                                    "pairs": diag})
                    continue

                def sub(noise_flag, metric):
                    if source != "fnb":
                        return None
                    r = sidx.get((f_rung, f_gv, _vname(variant, rung, noise_flag, "true"), mc,
                                  metric))
                    return None if r is None else r["seed_wobble"]

                parts = {m: _part(res["y"][m], res["x"][m], sub(1, m), sub(0, m))
                         for m in PVN_READ + LOG_SPLIT}
                for m in PVN_READ:
                    p = parts[m]
                    comp = {"pairs": diag, "per_seed_push": res["x"][m]["per_seed"],
                            "per_seed_noise": res["y"][m]["per_seed"],
                            "gain": "push - noise; positive = push + own noise is lower"}
                    if m == "crps_log_all":
                        comp["split_gain"] = {k: parts[k] for k in LOG_SPLIT}
                    out.append({**base, "mark_class": mc, "metric": m,
                                "push_mean": p["other"], "push_wobble": p["other_wobble"],
                                "noise_mean": p["this"], "noise_wobble": p["this_wobble"],
                                "gain": p["gain"], "bar": p["bar"],
                                "gain_exceeds_bar": p["gain_exceeds_bar"],
                                "wobble_note": WOBBLE_FNB if source == "fnb" else None,
                                "components": comp})
    return out, skipped


def repair(cell, arms: dict) -> tuple[list, list]:
    out, skipped = [], []
    f_rung, f_gv = cell
    for variant, rung, noise in _learned_versions():
        tru, fnb = arms.get(_vname(variant, rung, noise, "true")), arms.get(
            _vname(variant, rung, noise, "fnb"))
        if not tru and not fnb:
            continue
        base = {"reading": "repair", "scope": "chain", "f_rung": f_rung, "f_g_version": f_gv,
                "variant": variant, "rung": rung, "noise": noise, "source": None}
        if not tru or not fnb:
            skipped.append({**base, "reason": "no " + ("fnb-trained" if tru else "true-X'-trained")
                            + " h of this version was scored in this f cell"})
            continue
        ms = metrics_of(bool(noise))
        for mc in _classes(tru, fnb):
            res, diag = compare(tru, fnb, ms, mc, "chain")
            if res is None:
                skipped.append({**base, "mark_class": mc, "reason": diag["reason"],
                                "pairs": diag})
                continue
            for m in ms:
                t, f = res["x"][m], res["y"][m]
                delta = _diff(t["mean"], f["mean"])
                bar = _bar(t["seed_wobble"], f["seed_wobble"]) if delta is not None else None
                out.append({**base, "mark_class": mc, "metric": m, "true_h": tru["info"]["h_name"],
                            "fnb_h": fnb["info"]["h_name"], "true_mean": t["mean"],
                            "true_wobble": t["seed_wobble"], "fnb_mean": f["mean"],
                            "delta": delta, "gain": delta, "bar": bar,
                            "abs_delta_exceeds_bar": bool(delta is not None and bar is not None
                                                          and abs(delta) > bar),
                            "gain_exceeds_bar": _exceeds(delta, bar),
                            "wobble_note": WOBBLE_FNB,
                            "components": {"pairs": diag, "per_seed_true": t["per_seed"],
                                           "per_seed_fnb": f["per_seed"],
                                           "gain": "true - fnb; positive = the fnb-trained h is "
                                                   "lower"}})
    return out, skipped


def vs_oldgrid(cell, arms: dict, sidx: dict, old: dict, old_arms: dict, old_ex: dict,
               exp_idx: dict) -> tuple[list, list]:
    out, skipped = [], []
    f_rung, f_gv = cell
    old_arm = old_arms.get(cell)
    for vname in sorted(arms):
        arm = arms[vname]
        info = arm["info"]
        noise = bool(info["noise"])
        for mc in _classes(arm):
            if not any((f_rung, f_gv, mc, m) in old for m in OLD_METRICS):
                skipped.append({"reading": "vs_oldgrid", "scope": "chain", "f_rung": f_rung,
                                "f_g_version": f_gv, "mark_class": mc, "h_name": vname,
                                "reason": "the old grid has no p-space value for this rung, g "
                                          "and class"})
                continue
            ex = exp_idx.get((f_rung, f_gv, vname, mc), {})
            oe = old_ex.get((f_rung, f_gv, mc)) if old_arm else None
            res, diag, matched = None, None, False
            if old_arm and not noise:
                res, diag = compare(arm, old_arm, OLD_METRICS, mc, "chain")
                matched = res is not None
                if res is None:
                    skipped.append({"reading": "vs_oldgrid", "scope": "chain", "f_rung": f_rung,
                                    "f_g_version": f_gv, "mark_class": mc, "h_name": vname,
                                    "reason": diag["reason"], "pairs": diag})
                    continue
            for m in OLD_METRICS:
                hm = (NAT[True] + m[len("crps"):] if noise and m.startswith("crps_")
                      else "crps_mean_point_" + m if noise else m)
                srow = sidx.get((f_rung, f_gv, vname, mc, hm))
                trow = old.get((f_rung, f_gv, mc, m))
                note = []
                if matched:
                    h_mean, h_w = res["x"][m]["mean"], res["x"][m]["seed_wobble"]
                    o_mean, o_w = res["y"][m]["mean"], res["y"][m]["seed_wobble"]
                else:
                    h_mean = srow["mean"] if srow else None
                    h_w = srow["seed_wobble"] if srow else None
                    o_mean = trow["mean"] if trow else None
                    o_w = trow["seed_wobble"] if trow else None
                    if trow is None and m in OLD_METRICS[2:]:
                        note.append("the old grid's split needs --old-grid-json")
                if info["source"] == "fnb":
                    t = sidx.get((f_rung, f_gv, _vname(info["variant"], info["rung"],
                                                       info["noise"], "true"), mc, hm))
                    h_w = None if t is None else t["seed_wobble"]
                    note.append(WOBBLE_FNB)
                elif info["source"] == "formula":
                    note.append(WOBBLE_FORMULA)
                delta = bar = None
                block = None
                if noise:
                    note.append(NOISE_VS_OLD)
                else:
                    delta = _diff(o_mean, h_mean)
                    bar = _bar(o_w, h_w) if delta is not None else None
                    if not matched and trow is not None and srow is not None and (
                            trow["n_tracks"] != srow["n_tracks"]):
                        block = (f"track counts differ (chain {srow['n_tracks']}, old grid "
                                 f"{trow['n_tracks']}) and the old-grid TSV cannot be "
                                 "pair-matched")
                        note.append(block)
                    if not matched:
                        note.append("old grid from --old-grid-tsv only: class means as they are")
                out.append({"reading": "vs_oldgrid", "scope": "chain", "f_rung": f_rung,
                            "f_g_version": f_gv, "mark_class": mc, "h_name": vname,
                            "variant": info["variant"], "rung": info["rung"],
                            "noise": info["noise"], "source": info["source"], "metric": m,
                            "h_metric": hm, "old_mean": o_mean, "old_wobble": o_w,
                            "h_mean": h_mean, "h_wobble": h_w, "delta": delta, "gain": delta,
                            "bar": bar,
                            "gain_exceeds_bar": False if block else _exceeds(delta, bar),
                            "matched": matched,
                            "n_pairs_used": diag["n_used"] if matched else None,
                            "n_pairs_dropped_not_finite": (diag["n_dropped_not_finite"]
                                                           if matched else None),
                            "n_pairs_only_one_arm": diag["n_only_one_arm"] if matched else None,
                            "old_tsv_mean": trow["mean"] if trow else None,
                            "old_tsv_wobble": trow["seed_wobble"] if trow else None,
                            "old_n_tracks": trow["n_tracks"] if trow else None,
                            "h_n_tracks": srow["n_tracks"] if srow else None,
                            "old_n_crps_gt_20": oe["n_crps_gt_20"] if oe else None,
                            "old_n_crps_nonfinite": oe["n_crps_nonfinite"] if oe else None,
                            "h_n_crps_gt_20": ex.get("n_gt_20"),
                            "h_n_crps_nonfinite": ex.get("n_native_nonfinite"),
                            "note": "; ".join(note) or None,
                            "components": {"pairs": diag,
                                           "gain": "old - chain h; positive = the chain h is "
                                                   "lower"}})
    return out, skipped


# ---------------------------------------------------------------------------------------------
# grid_h.md
# ---------------------------------------------------------------------------------------------

GRID_ROWS = (("formula, bin only", "formula", "bin"),
             ("formula, fragment window", "formula", "frag"),
             ("affine, rung 1", "affine", 1), ("affine, rung 5", "affine", 5),
             ("MLP, rung 1", "mlp", 1), ("MLP, rung 5", "mlp", 5),
             ("formula + affine, rung 5", "formula_affine", 5),
             ("formula + MLP, rung 5", "formula_mlp", 5))
GRID_TABLES = (("CRPS on P, all bins", "all", "native"),
               ("CRPS on P, top 1 % bins", "top1", "native"),
               ("CRPS on log(1 + P), all bins", "crps_log_all", "log"),
               ("CRPS on log(1 + P), top 1 % bins", "crps_log_top1", "log"),
               ("MSE on log(1 + P) of the predicted mean, all bins", "mse_log1p_all", "plain"),
               ("MSE on log(1 + P) of the predicted mean, top 1 % bins", "mse_log1p_top1",
                "plain"))


def _f(v) -> str:
    return "—" if v is None else f"{v:.4g}"


def _pm(m, w, one_seed="one seed") -> str:
    if m is None:
        return "—"
    return f"{_f(m)} ± {_f(w)}" if w is not None else f"{_f(m)} ({one_seed})"


def _yn(b) -> str:
    return "yes" if b else "no"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    return (["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
            + ["| " + " | ".join(r) + " |" for r in rows] + [""])


def _grid_cell(sidx, cell, vname, mc, which, kind, noise, formula) -> str:
    if vname is None:
        return "—"
    fr, fg = cell

    def g(metric):
        return sidx.get((fr, fg, vname, mc, metric))

    tag = "no h seed" if formula else "one seed"
    if kind == "native":
        k = NAT[bool(noise)]
        r = g(f"{k}_{which}")
        if r is None:
            return "—"
        s = _pm(r["mean"], r["seed_wobble"], tag)
        if which == "all":
            o, e = (g(x) for x in split_keys(bool(noise)))
            if o is not None and e is not None:
                s += (f"; oracle-scaled {_pm(o['mean'], o['seed_wobble'], tag)} + scale error "
                      f"{_pm(e['mean'], e['seed_wobble'], tag)}")
        return ("point CRPS of the mean: " + s) if noise else s
    r = g(which)
    if r is None:
        return "—"
    s = _pm(r["mean"], r["seed_wobble"], tag)
    if which == "crps_log_all":
        o, e = (g(x) for x in LOG_SPLIT)
        if o is not None and e is not None:
            s += (f"; oracle-scaled {_pm(o['mean'], o['seed_wobble'], tag)} + scale error "
                  f"{_pm(e['mean'], e['seed_wobble'], tag)}")
    return s


def write_grid_md(path: Path, alone_rows: list[dict], readings: list[dict], explosions: list,
                  capped: list, skipped: list, counts: dict) -> None:
    sidx = _summary_index(alone_rows)
    cell = (None, None)
    L = ["# t125 part 2: the map h from the predicted NB and the control to P = -log10 p", "",
         "Rows: the h versions (complexity x inputs). Formula = the caller's Poisson tail, "
         "nothing learned (bin only, or the window one fragment wide). Rung 1 = the bin's count "
         "and the control count in the bin (+ the assay); rung 5 = every input. Columns: push "
         "only (h adds no noise) and push + own noise (a Gaussian on log(1 + P) with a learned "
         "sigma).", "",
         "Each cell: h alone, on the true X' of every usable product, held-out chromosomes "
         "chr19 + chr21; per seed the mean over the products of the class, then the mean over "
         "the 3 h seeds ± the seed wobble (the largest pairwise difference over the seeds). "
         "Lower is better. CRPS on P of push only is the proper CRPS of the pushed set; for "
         "push + own noise the number on P is the point CRPS of the mean (marked), never the "
         "same quantity. The proper CRPS of both variants is on log(1 + P). CRPS on all bins "
         "carries its oracle-scaled + scale-error split. The chain values per f cell are in "
         "h_chain_summary.tsv.", "",
         f"h runs present alone: {counts['alone_present']} of {counts['alone_expected']}; "
         f"chain f runs present: {counts['chain_f_present']} of {counts['chain_f_expected']} "
         "(missing ones listed in results_h.json).", "",
         "A row marked \"gain > 2 × wobble: yes\" only says the gain is larger than 2 × the "
         "larger seed wobble of the two arms. No row here is ticked or concluded.", ""]
    for mc in [c for c in CLASS_ORDER if any(r["mark_class"] == c for r in alone_rows)]:
        L += [f"## {mc}, h alone", ""]
        for title, which, kind in GRID_TABLES:
            rows = []
            for label, variant, x in GRID_ROWS:
                formula = variant == "formula"
                push = f"h_formula_{x}" if formula else _vname(variant, x, 0, "true")
                noise = None if formula else _vname(variant, x, 1, "true")
                rows.append([label,
                             _grid_cell(sidx, cell, push, mc, which, kind, 0, formula),
                             _grid_cell(sidx, cell, noise, mc, which, kind, 1, formula)])
            L += [f"**{title}**", ""]
            L += _table(["h", "push only", "push + own noise"], rows)

    def where(r):
        if r["scope"] == "alone":
            return "alone"
        return f"chain {r['f_rung']}, {GV_LABEL.get(r['f_g_version'], r['f_g_version'])}"

    def ver(r):
        return f"{r['variant']} r{r['rung']} {r['source']}"

    def dropped(r):
        p = r["components"].get("pairs") or {}
        if "n_used" not in p:
            return "not matched (TSV)"
        return f"{p['n_used']} / {p['n_dropped_not_finite']} / {p['n_only_one_arm']}"

    def split(r, k):
        sg = r["components"].get("split_gain")
        if not sg:
            return "—"
        return _f(sg[k]["gain"]) + ("*" if sg[k]["gain_exceeds_bar"] else "")

    L += ["## Push only against push + own noise", "",
          "PI: the two variants are compared on the CRPS on log(1 + P) and on the MSE on "
          "log(1 + P) of the mean. Gain = push - noise (positive = push + own noise is lower). "
          "Bar = 2 × the larger seed wobble; an fnb arm (one h seed) takes the true-X' h's "
          "wobble. Used / dropped not finite / in one arm only: products (alone) or pairs "
          "(chain), over every seed of both arms. * = larger than that part's own bar.", ""]
    L += _table(["where", "h", "class", "metric", "push", "own noise", "gain",
                 "oracle-scaled gain", "scale-error gain", "bar", "gain > 2 × wobble",
                 "used / dropped / one arm"],
                [[where(r), ver(r), r["mark_class"], r["metric"],
                  _pm(r["push_mean"], r["push_wobble"]), _pm(r["noise_mean"], r["noise_wobble"]),
                  _f(r["gain"]), split(r, LOG_SPLIT[0]), split(r, LOG_SPLIT[1]), _f(r["bar"]),
                  _yn(r["gain_exceeds_bar"]), dropped(r)]
                 for r in readings if r["reading"] == "push_vs_noise"])
    L += ["## Repair: h versions compared on f's NB", "",
          "In the chain on the same f (seed 0 of each real f): the h trained on the true X' "
          "against the h trained on that f's NB. Delta = true - fnb (positive = the fnb-trained "
          "h is lower). One fnb h seed (PI): the bar is 2 × the true-X' h's seed wobble. "
          "Own-noise rows use the point CRPS of the mean on P (crps_mean_point_*).", ""]
    L += _table(["f", "class", "h", "noise", "metric", "true-X' h", "fnb h", "delta", "bar",
                 "|delta| > 2 × wobble", "used / dropped / one arm"],
                [[where(r), r["mark_class"], f"{r['variant']} r{r['rung']}",
                  "own noise" if r["noise"] else "push", r["metric"],
                  _pm(r["true_mean"], r["true_wobble"]), _pm(r["fnb_mean"], None),
                  _f(r["delta"]), _f(r["bar"]), _yn(r["abs_delta_exceeds_bar"]), dropped(r)]
                 for r in readings if r["reading"] == "repair"])
    L += ["## The chain against the old grid's p results", "",
          "The old grid (t118 row 2): real f, p space, trained pairs, mean over its 3 f seeds ± "
          "wobble. The chain: t125 f (seed 0) then h; ± the h seeds' wobble (fnb: the true-X' "
          "h's; formula: none). Delta = old - chain h (positive = the chain h is lower). "
          "Own-noise h are shown with their point CRPS of the mean and no delta.", ""]
    L += _table(["f", "class", "h", "metric", "old grid", "chain h", "delta", "bar",
                 "gain > 2 × wobble", "pair-matched", "CRPS > 20: old / h", "note"],
                [[where(r), r["mark_class"], r["h_name"], r["metric"],
                  _pm(r["old_mean"], r["old_wobble"]), _pm(r["h_mean"], r["h_wobble"]),
                  _f(r["delta"]), _f(r["bar"]), _yn(r["gain_exceeds_bar"]), _yn(r["matched"]),
                  f"{'—' if r['old_n_crps_gt_20'] is None else r['old_n_crps_gt_20']} / "
                  f"{'—' if r['h_n_crps_gt_20'] is None else r['h_n_crps_gt_20']}",
                  r["note"] or ""]
                 for r in readings if r["reading"] == "vs_oldgrid"])
    L += ["## Exploding records (CRPS > 20) and non-finite CRPS", "",
          "Per (product or pair, seed) record: the flag of the h's kind (crps_all_gt_20 for "
          "push only, crps_mean_point_all_gt_20 for own noise), and apart from it the records "
          "with a non-finite bin on P and on log(1 + P). Reported only; a non-finite record is "
          "in no mean.", ""]
    L += _table(["where", "h", "class", "flag", "CRPS > 20", "non-finite on P",
                 "non-finite on log(1 + P)", "records"],
                [["alone" if e["scope"] == "alone" else
                  f"chain {e['f_rung']}, {GV_LABEL.get(e['f_g_version'], e['f_g_version'])}",
                  e["h_name"], e["mark_class"], e["kind"], str(e["n_gt_20"]),
                  str(e["n_native_nonfinite"]), str(e["n_log_nonfinite"]), str(e["n_records"])]
                 for e in explosions])
    L += ["## Capped NB rows", "",
          "Per f run: bins whose NB support was cut at K_MAX (n_capped), shared by every h. A "
          "nonzero count goes to the PI.", ""]
    L += _table(["f run", "pairs", "pairs with a capped row", "capped rows",
                 "same in every h file"],
                [[c["f_run"], str(c["n_pairs"]), str(c["n_pairs_capped"]),
                  str(c["n_capped_rows"]), _yn(c["consistent_across_h"])] for c in capped])
    if skipped:
        L += ["## Comparisons with no reading", ""]
        L += _table(["reading", "where", "h", "class", "why"],
                    [[k["reading"], "alone" if k.get("scope") == "alone" else
                      f"chain {k.get('f_rung')}, {k.get('f_g_version')}",
                      str(k.get("h_name") or f"{k.get('variant')} r{k.get('rung')}"),
                      str(k.get("mark_class", "")), k["reason"]] for k in skipped])
    path.write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------------------------------------
# the aggregate command
# ---------------------------------------------------------------------------------------------


def _tsv(path: Path, cols, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            w.writerow(["" if r.get(c) is None else _yn(r[c]) if isinstance(r[c], bool)
                        else r[c] for c in cols])


def _flat_summary(r: dict, scope: str) -> dict:
    ex = r["explosions"]
    unit = "products" if scope == "alone" else "pairs"
    return {**{k: v for k, v in r.items() if k not in ("per_seed", "explosions", "seeds",
                                                        "per_seed_list")},
            **r["per_seed"], f"n_{unit}": r["n"], "explosion_kind": ex["kind"],
            f"n_{unit}_crps_gt_20": ex["n_gt_20"],
            f"n_{unit}_crps_nonfinite": ex["n_native_nonfinite"],
            f"n_{unit}_crps_log_nonfinite": ex["n_log_nonfinite"]}


def _json_row(r: dict) -> dict:
    return {k: v for k, v in r.items() if k not in ("per_seed",)} | {
        "per_seed": dict(zip([str(s) for s in r["seeds"]], r["per_seed_list"]))}


def aggregate(manifest, covariates, h_root, agg_dir, old_grid_tsv=None, old_grid_json=None,
              eta_runs=None, fnb_seeds: int = 1) -> dict:
    lpairs.read_manifest(manifest)               # fail early on a bad manifest
    if not Path(covariates).is_file():
        raise FileNotFoundError(covariates)
    h_root, agg_dir = Path(h_root), Path(agg_dir)
    agg_dir.mkdir(parents=True, exist_ok=True)
    warnings: list = []
    skipped: list = []
    alone_arms, alone_present = load_alone(h_root, warnings)
    cells, chain_present, capped, n_recs = load_chain(h_root, warnings, skipped)
    for w in warnings:
        _warn(w)
    old = load_old_tsv(old_grid_tsv)
    old_arms, old_ex = load_old_json(old_grid_json)

    # expected runs
    alone_expected = [r["h_name"] for r in tasks.tasks_alone()]
    alone_missing = [n for n in alone_expected if n not in set(alone_present)]
    f_expected = ep.f_runs_seed0()
    f_missing = [f for f in f_expected if f not in chain_present]
    chain_missing = {f: [n for n in tasks.chain_h_list(f, fnb_seeds)
                         if n not in set(chain_present[f])] for f in sorted(chain_present)}
    chain_done = {f: (h_root / "chain" / f / "CHAIN_DONE").is_file() for f in chain_present}
    incomplete = []
    for f, by_h in sorted(n_recs.items()):
        want = None
        if eta_runs is not None and (Path(eta_runs) / f / "config.json").is_file():
            want = len(json.loads((Path(eta_runs) / f / "config.json").read_text())
                       .get("train_pairs", []))
        ref = want if want is not None else max(by_h.values())
        for h, n in sorted(by_h.items()):
            if n != ref:
                incomplete.append({"f_run": f, "h_name": h, "n_records": n, "n_expected": ref,
                                   "expected_from": "f config train_pairs" if want is not None
                                   else "the most records of any h of this f"})
    if alone_missing:
        _warn(f"{len(alone_missing)} of {len(alone_expected)} alone h runs missing")
    if f_missing:
        _warn(f"{len(f_missing)} of {len(f_expected)} chain f runs missing")

    # summaries
    alone_rows = [r for v in sorted(alone_arms) for r in summary_rows(alone_arms[v], "alone",
                                                                      {"f_rung": None,
                                                                       "f_g_version": None})]
    chain_rows = [r for cell in sorted(cells) for v in sorted(cells[cell])
                  for r in summary_rows(cells[cell][v], "chain",
                                        {"f_rung": cell[0], "f_g_version": cell[1]})]
    sidx_alone, sidx_chain = _summary_index(alone_rows), _summary_index(chain_rows)
    explosions = []
    for scope, rows in (("alone", alone_rows), ("chain", chain_rows)):
        seen = set()
        for r in rows:
            key = (r["f_rung"], r["f_g_version"], r["h_name"], r["mark_class"])
            if key in seen:
                continue
            seen.add(key)
            explosions.append({"scope": scope, "f_rung": r["f_rung"],
                               "f_g_version": r["f_g_version"], "h_name": r["h_name"],
                               "variant": r["variant"], "rung": r["rung"], "noise": r["noise"],
                               "source": r["source"], "mark_class": r["mark_class"],
                               **r["explosions"]})
    exp_idx = {(e["f_rung"], e["f_g_version"], e["h_name"], e["mark_class"]): e
               for e in explosions if e["scope"] == "chain"}

    # readings
    readings = []
    rs, sk = push_vs_noise("alone", None, alone_arms, sidx_alone, "alone")
    readings += rs
    skipped += sk
    for cell in sorted(cells):
        for fn in (lambda c: push_vs_noise("chain", c, cells[c], sidx_chain, "chain"),
                   lambda c: repair(c, cells[c]),
                   lambda c: vs_oldgrid(c, cells[c], sidx_chain, old, old_arms, old_ex,
                                        exp_idx) if old else ([], [])):
            rs, sk = fn(cell)
            readings += rs
            skipped += sk

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    j_alone = [_json_row(r) for r in alone_rows]
    j_chain = [_json_row(r) for r in chain_rows]
    old_ex_rows = [{"rung": k[0], "g_version": k[1], "mark_class": k[2], **v}
                   for k, v in sorted(old_ex.items())]
    results = {"created_utc": now, "h_root": str(h_root), "manifest": str(manifest),
               "covariates": str(covariates),
               "old_grid_tsv": None if old_grid_tsv is None else str(old_grid_tsv),
               "old_grid_json": None if old_grid_json is None else str(old_grid_json),
               "eta_runs": None if eta_runs is None else str(eta_runs), "fnb_seeds": fnb_seeds,
               "explode_at": EXPLODE_AT,
               "alone_present": sorted(alone_present), "alone_missing": alone_missing,
               "chain_present": {f: sorted(v) for f, v in sorted(chain_present.items())},
               "chain_missing": chain_missing, "chain_f_runs_missing": f_missing,
               "chain_done": chain_done, "chain_incomplete": incomplete,
               "warnings": warnings, "alone_rows": j_alone, "chain_rows": j_chain,
               "readings": readings, "readings_skipped": skipped, "explosions": explosions,
               "capped": capped, "oldgrid_explosions": old_ex_rows,
               "notes": {"wobble_fnb": WOBBLE_FNB, "wobble_formula": WOBBLE_FORMULA,
                         "noise_vs_old": NOISE_VS_OLD}}
    write_json(results, agg_dir / "results_h.json", indent=None)
    _tsv(agg_dir / "h_alone_summary.tsv", ALONE_COLS,
         [_flat_summary(r, "alone") for r in alone_rows])
    _tsv(agg_dir / "h_chain_summary.tsv", CHAIN_COLS,
         [_flat_summary(r, "chain") for r in chain_rows])
    _tsv(agg_dir / "h_repair.tsv", REPAIR_COLS,
         [{**r, "n_pairs_used": r["components"]["pairs"]["n_used"],
           "n_pairs_dropped_not_finite": r["components"]["pairs"]["n_dropped_not_finite"],
           "n_pairs_only_one_arm": r["components"]["pairs"]["n_only_one_arm"],
           "note": r["wobble_note"]} for r in readings if r["reading"] == "repair"])
    _tsv(agg_dir / "h_vs_oldgrid.tsv", VS_COLS,
         [r for r in readings if r["reading"] == "vs_oldgrid"])
    counts = {"alone_present": len(alone_present), "alone_expected": len(alone_expected),
              "chain_f_present": len(chain_present), "chain_f_expected": len(f_expected)}
    write_grid_md(agg_dir / "grid_h.md", alone_rows, readings, explosions, capped, skipped,
                  counts)
    write_json({"created_utc": now, **counts, "explode_at": EXPLODE_AT,
                "alone_cells": j_alone, "chain_cells": j_chain, "readings": readings,
                "readings_skipped": skipped, "explosions": explosions, "capped": capped,
                "oldgrid_explosions": old_ex_rows,
                "notes": results["notes"]}, agg_dir / "summary_data_h.json")
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for pos in ("manifest", "covariates", "h_root", "agg_dir"):
        ap.add_argument(pos, type=Path)
    ap.add_argument("--old-grid-tsv", type=Path, required=True,
                    help="t118 row-2 results_summary.tsv (its p-space class means); an empty "
                         "file gives no vs_oldgrid rows")
    ap.add_argument("--old-grid-json", type=Path, default=None,
                    help="t118 row-2 agg/results.json (per_pair pval records: pair matching, "
                         "the old split and explosion counts)")
    ap.add_argument("--eta-runs", type=Path, default=None,
                    help="the eta runs dir (each f's config.json train_pairs: expected pairs)")
    ap.add_argument("--fnb-seeds", type=int, choices=(1, 3), default=1)
    a = ap.parse_args(argv)
    res = aggregate(a.manifest, a.covariates, a.h_root, a.agg_dir, a.old_grid_tsv,
                    a.old_grid_json, a.eta_runs, a.fnb_seeds)
    print(f"alone h runs {len(res['alone_present'])} present, {len(res['alone_missing'])} "
          f"missing; chain f runs {len(res['chain_present'])} present, "
          f"{len(res['chain_f_runs_missing'])} missing; {len(res['readings'])} readings -> "
          f"{a.agg_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
