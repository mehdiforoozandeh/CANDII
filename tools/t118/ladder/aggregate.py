"""t118 ladder — aggregate every scored run to pair -> track -> mark class with seed wobble, compute
every check's value against its bar, and choose the main-claim rung on chr22.

**Aggregation** (eval `score` unless stated). Mean over bins per pair (done by `score.py`); mean
over pairs per track; macro mean over the tracks of a mark class, per seed; then `mean` over the
seeds and `seed_wobble` = the largest pairwise |delta| of the per-seed class values (None with
fewer than two seeds). `variant = all` keeps every pair; `nonidentical` drops pairs whose target is
identical to the source in that space (`counts_identical` / `pval_identical`). Kinds aggregated:
trained, shuffle, swap, law (depthlaw records go to `depth_law`). `g_version` = across for g "all",
else per_track.

**Checks** — each `{rung g_version space mark_class metric check value bar met seed_wobble
components}`. `met` is a computed comparison of value against bar (None when the bar cannot be
formed, e.g. one seed). Nothing here ticks anything or writes to the notebook.
  beatstwin      D_nocov - D_real, kind trained;       bar 2 x wobble_real;  met value > bar
  lawtest_nocov  D_nocov - D_real, kind law;           bar 2 x wobble_real;  met value > bar
  lawtest_ids    D_ids - D_real, kind law;             bar 2 x wobble_real;  met value > bar
  depthlaw       counts: max over depth pairs (trained base<->depth and law depth->depth) of
                 |2^(mean over seeds of log2_pred - log2_true) - 1|, real;  bar 0.10; met value <= bar
  beatsbelow     (B, C, D) D_below - D_this, real, trained; bar 2 x max(wobble_below, wobble_this);
                 met value > bar; left out when the rung below has no runs
  shuffle        D_nocov(trained) - D_real(shuffle); bar 2 x wobble_real (trained); met value < bar
  swap           macro over the class's tracks of the per-track mean swap_median_abs_log_ratio,
                 real, mean over seeds; bar 0.1; met value < bar
(D metrics crps_all and crps_top1, variant all.) `checks_<rung>.json` holds the rung's checks
(shuffle and swap excluded — they belong to the main claim); `checks_main.json` holds every check
of the rung chosen per (g_version, space): the lowest rung whose chr22 `crps_all` (real, mean over
seeds, macro over all tracks, variant all) is within the best rung's seed wobble of the best.

    python tools/t118/ladder/aggregate.py <manifest> <covariates.tsv> <runs_dir> <refs.tsv> <agg_dir>
    python tools/t118/ladder/aggregate.py qm-curves <manifest> <data_dir> <out.json>
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_T118 = Path(__file__).resolve().parents[1]
if str(_T118) not in sys.path:
    sys.path.insert(0, str(_T118))

import baseline_rungs as br  # noqa: E402
from ladder import data, pairs  # noqa: E402
from ladder.score import METRICS, SWAP_KEY, g_version, write_json  # noqa: E402

KINDS = ("trained", "shuffle", "swap", "law")
VARIANTS = ("all", "nonidentical")
CHECK_METRICS = ("crps_all", "crps_top1")
RUN_KEYS = ("run_name", "rung", "g", "g_version", "space", "model", "seed")
REF_RUNGS = ("noSolution", "QuantileMatching")
#: our metric name <- the baseline rungs_v2.tsv column
REF_COLUMNS = {"crps_all": "spread_crps_all", "crps_nonzero": "spread_crps_nonzero",
               "crps_top1": "spread_crps_top1", "spearman_all": "spearman_all",
               "spearman_nonzero": "spearman_nonzero", "spearman_top1": "spearman_top1"}
DEPTH_BAR = 0.10
SWAP_BAR = 0.10
MAIN_ONLY = ("shuffle", "swap")
QM_MAX_KNOTS = 256


def _mean(vals):
    v = [float(x) for x in vals if x is not None and not isinstance(x, bool) and np.isfinite(x)]
    return float(np.mean(v)) if v else None


def wobble(vals):
    """Largest pairwise |delta| of the non-None values; None with fewer than two."""
    v = [x for x in vals if x is not None]
    if len(v) < 2:
        return None
    return float(max(abs(a - b) for a, b in itertools.combinations(v, 2)))


def _warn(msg: str) -> None:
    print(f"WARNING: {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------------------


def expected_runs(rows: list[dict]) -> list[str]:
    """Every run name the manifest implies (its tracks + "all"), in task order."""
    g_ids = tuple(pairs.track_ids(rows)) + ("all",)
    return [f"{r}_{g}_{s}_{m}_s{sd}" for r in pairs.RUNGS for g in g_ids for s in pairs.SPACES
            for m in pairs.MODELS for sd in pairs.SEEDS]


def load_runs(runs_dir) -> tuple[list[dict], list[str], list[str], dict]:
    """(per_pair records with run fields flattened in, runs present, runs lacking law.json,
    figdata paths). A run is present when its scores.json exists."""
    runs_dir = Path(runs_dir)
    per_pair, present, no_law, figdata = [], [], [], {}
    for d in sorted(p for p in runs_dir.iterdir() if p.is_dir()) if runs_dir.is_dir() else []:
        sj = d / "scores.json"
        if not sj.is_file():
            continue
        s = json.loads(sj.read_text("utf-8"))
        run = dict(s["run"])
        run.setdefault("run_name", d.name)
        run["g_version"] = g_version(run["g"])
        flat = {k: run[k] for k in RUN_KEYS}
        present.append(flat["run_name"])
        per_pair.extend({**r, **flat} for r in s["records"])
        lj = d / "law.json"
        if lj.is_file():
            per_pair.extend({**r, **flat} for r in json.loads(lj.read_text("utf-8"))["records"])
        else:
            no_law.append(flat["run_name"])
        if (d / "figdata.npz").is_file():
            figdata[flat["run_name"]] = str(d / "figdata.npz")
    return per_pair, present, no_law, figdata


def load_refs(refs_tsv) -> dict:
    """{track: {space: {rung: {metric: mean over the track's pairs}}}} from the baseline TSV
    (eval score, noSolution and QuantileMatching, the excluded products left out); {} + a warning
    when the file is missing, empty or lacks the columns."""
    path = Path(refs_tsv) if refs_tsv else None
    if path is None or not path.is_file():
        _warn(f"refs file {refs_tsv} not found; refs = {{}}")
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    need = {"track", "space", "rung", "eval"}
    if not rows or not need <= set(rows[0]):
        _warn(f"refs file {path} is empty or lacks {sorted(need)}; refs = {{}}")
        return {}
    acc: dict = defaultdict(list)
    for r in rows:
        if r["eval"] != "score" or r["rung"] not in REF_RUNGS:
            continue
        if r.get("arm_pid") in pairs.EXCLUDED_PIDS:
            continue
        acc[(r["track"], r["space"], r["rung"])].append(r)
    refs: dict = {}
    for (track, space, rung), rs in sorted(acc.items()):
        vals = {}
        for ours, col in REF_COLUMNS.items():
            if col in rs[0]:
                vals[ours] = _mean(float(r[col]) if r[col] not in ("", None) else None for r in rs)
        refs.setdefault(track, {}).setdefault(space, {})[rung] = vals
    return refs


# ---------------------------------------------------------------------------------------------
# pair -> track -> class
# ---------------------------------------------------------------------------------------------


def _identical(r: dict) -> bool:
    return bool(r[f"{r['space']}_identical"])


def per_track_rows(per_pair: list[dict], ev: str = "score") -> list[dict]:
    groups: dict = defaultdict(list)
    for r in per_pair:
        if r.get("eval") != ev or r.get("kind") not in KINDS:
            continue
        key = tuple(r[k] for k in ("rung", "g_version", "space", "model", "seed", "kind", "track",
                                   "mark_class"))
        groups[key].append(r)
    out = []
    for key, rs in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        rung, gv, space, model, seed, kind, track, mc = key
        metrics = METRICS + ((SWAP_KEY,) if kind == "swap" else ())
        for variant in VARIANTS:
            sel = rs if variant == "all" else [r for r in rs if not _identical(r)]
            for m in metrics:
                vals = [r.get(m) for r in sel if r.get(m) is not None]
                if not vals:
                    continue
                out.append({"rung": rung, "g_version": gv, "space": space, "model": model,
                            "seed": seed, "kind": kind, "track": track, "mark_class": mc,
                            "metric": m, "value": _mean(vals), "n_pairs": len(vals),
                            "variant": variant})
    return out


def per_class_rows(per_track: list[dict]) -> list[dict]:
    groups: dict = defaultdict(lambda: defaultdict(list))
    for r in per_track:
        key = tuple(r[k] for k in ("rung", "g_version", "space", "model", "kind", "mark_class",
                                   "metric", "variant"))
        groups[key][r["seed"]].append(r["value"])
    out = []
    for key, by_seed in sorted(groups.items()):
        per_seed = [_mean(by_seed[s]) if s in by_seed else None for s in pairs.SEEDS]
        out.append({**dict(zip(("rung", "g_version", "space", "model", "kind", "mark_class",
                                "metric", "variant"), key)),
                    "per_seed": per_seed, "mean": _mean(per_seed), "seed_wobble": wobble(per_seed),
                    "n_tracks": max(len(v) for v in by_seed.values())})
    return out


def _class_index(per_class: list[dict]) -> dict:
    return {tuple(r[k] for k in ("rung", "g_version", "space", "model", "kind", "mark_class",
                                 "metric", "variant")): r for r in per_class}


# ---------------------------------------------------------------------------------------------
# derived tables
# ---------------------------------------------------------------------------------------------


def depth_law_rows(per_pair: list[dict]) -> list[dict]:
    return [{"rung": r["rung"], "g_version": r["g_version"], "space": r["space"],
             "track": r["track"], "mark_class": r["mark_class"], "source_pid": r["source_pid"],
             "target_pid": r["target_pid"], "model": r["model"], "seed": r["seed"],
             "kind": r.get("depth_source", "trained"), "log2_true": r["depth_log2_ratio_true"],
             "log2_pred": r["depth_log2_scale_pred"]}
            for r in per_pair if r.get("kind") == "depthlaw"]


def _gain(metric: str, d_twin, d_real):
    """Absolute gain of the real model over a twin; positive = real better."""
    if d_twin is None or d_real is None:
        return None
    return d_twin - d_real if metric.startswith("crps") else d_real - d_twin


def knob_gain_rows(per_pair: list[dict]) -> list[dict]:
    acc: dict = defaultdict(list)
    for r in per_pair:
        if r.get("kind") != "trained" or r.get("eval") != "score":
            continue
        arm = r["arm_tgt"] if r["direction"] == "base_to_arm" else r["arm_src"]
        for m in METRICS:
            acc[(r["rung"], r["g_version"], r["space"], arm, m, r["model"])].append(r.get(m))
    out = []
    for rung, gv, space, arm, m in sorted({k[:5] for k in acc}):
        d_real = _mean(acc.get((rung, gv, space, arm, m, "real"), []))
        d_nocov = _mean(acc.get((rung, gv, space, arm, m, "nocov"), []))
        g = _gain(m, d_nocov, d_real)
        rel = g / abs(d_nocov) if g is not None and d_nocov not in (None, 0.0) else None
        out.append({"rung": rung, "g_version": gv, "space": space, "arm": arm, "metric": m,
                    "gain_rel": rel, "d_real": d_real, "d_nocov": d_nocov})
    return out


def law_grid_rows(per_pair: list[dict]) -> list[dict]:
    acc: dict = defaultdict(list)
    ident: dict = {}
    for r in per_pair:
        if r.get("kind") != "law":
            continue
        cell = (r["rung"], r["g_version"], r["space"], r["track"], r["arm_src"], r["level_src"],
                r["arm_tgt"], r["level_tgt"])
        ident[cell] = _identical(r)
        for m in CHECK_METRICS:
            acc[cell + (m, r["model"])].append(r.get(m))
    out = []
    for cell in sorted(ident):
        for m in CHECK_METRICS:
            d = {mod: _mean(acc.get(cell + (m, mod), [])) for mod in pairs.MODELS}
            out.append({**dict(zip(("rung", "g_version", "space", "track", "arm_src", "level_src",
                                    "arm_tgt", "level_tgt"), cell)),
                        "metric": m, "d_real": d["real"], "d_nocov": d["nocov"], "d_ids": d["ids"],
                        "gain_nocov": _gain(m, d["nocov"], d["real"]),
                        "gain_ids": _gain(m, d["ids"], d["real"]), "identical_target": ident[cell]})
    return out


# ---------------------------------------------------------------------------------------------
# checks and the rung choice
# ---------------------------------------------------------------------------------------------


def _check(rung, gv, space, mc, metric, name, value, bar, met_fn, wob, components) -> dict:
    met = None if value is None or bar is None else bool(met_fn(value, bar))
    return {"rung": rung, "g_version": gv, "space": space, "mark_class": mc, "metric": metric,
            "check": name, "value": value, "bar": bar, "met": met, "seed_wobble": wob,
            "components": components}


def _twice(w):
    return None if w is None else 2.0 * w


def _diff(a, b):
    return None if a is None or b is None else a - b


def compute_checks(per_class: list[dict], depth_law: list[dict]) -> list[dict]:
    idx = _class_index(per_class)
    combos = sorted({(r["rung"], r["g_version"], r["space"], r["mark_class"]) for r in per_class})
    rungs_present = {r["rung"] for r in per_class}
    out = []

    def get(rung, gv, space, model, kind, mc, metric):
        return idx.get((rung, gv, space, model, kind, mc, metric, "all"))

    def val(row, key="mean"):
        return None if row is None else row[key]

    gt, lt, le = (lambda v, b: v > b), (lambda v, b: v < b), (lambda v, b: v <= b)
    for rung, gv, space, mc in combos:
        for metric in CHECK_METRICS:
            real = get(rung, gv, space, "real", "trained", mc, metric)
            nocov = get(rung, gv, space, "nocov", "trained", mc, metric)
            if real is not None or nocov is not None:
                w = val(real, "seed_wobble")
                out.append(_check(rung, gv, space, mc, metric, "beatstwin",
                                  _diff(val(nocov), val(real)), _twice(w), gt, w,
                                  {"d_real": val(real), "d_nocov": val(nocov),
                                   "wobble_real": w, "wobble_nocov": val(nocov, "seed_wobble")}))
            lreal = get(rung, gv, space, "real", "law", mc, metric)
            for twin in ("nocov", "ids"):
                lt_row = get(rung, gv, space, twin, "law", mc, metric)
                if lreal is None and lt_row is None:
                    continue
                w = val(lreal, "seed_wobble")
                out.append(_check(rung, gv, space, mc, metric, f"lawtest_{twin}",
                                  _diff(val(lt_row), val(lreal)), _twice(w), gt, w,
                                  {"d_real": val(lreal), f"d_{twin}": val(lt_row),
                                   "wobble_real": w, f"wobble_{twin}": val(lt_row, "seed_wobble")}))
            ri = pairs.RUNGS.index(rung)
            if ri > 0 and pairs.RUNGS[ri - 1] in rungs_present and real is not None:
                below = get(pairs.RUNGS[ri - 1], gv, space, "real", "trained", mc, metric)
                if below is not None:
                    wb, wt = val(below, "seed_wobble"), val(real, "seed_wobble")
                    wmax = None if wb is None or wt is None else max(wb, wt)
                    out.append(_check(rung, gv, space, mc, metric, "beatsbelow",
                                      _diff(val(below), val(real)), _twice(wmax), gt, wmax,
                                      {"rung_below": pairs.RUNGS[ri - 1], "d_below": val(below),
                                       "d_this": val(real), "wobble_below": wb,
                                       "wobble_this": wt}))
            shuf = get(rung, gv, space, "real", "shuffle", mc, metric)
            if shuf is not None:
                w = val(real, "seed_wobble")
                out.append(_check(rung, gv, space, mc, metric, "shuffle",
                                  _diff(val(nocov), val(shuf)), _twice(w), lt, w,
                                  {"d_nocov": val(nocov), "d_real_shuffled": val(shuf),
                                   "wobble_real": w,
                                   "wobble_real_shuffled": val(shuf, "seed_wobble")}))
        sw = get(rung, gv, space, "real", "swap", mc, SWAP_KEY)
        if sw is not None:
            out.append(_check(rung, gv, space, mc, SWAP_KEY, "swap", val(sw), SWAP_BAR, lt,
                              val(sw, "seed_wobble"), {"per_seed": sw["per_seed"],
                                                       "n_tracks": sw["n_tracks"]}))
    # depth law: counts, real, per (rung, g_version, mark_class)
    groups: dict = defaultdict(lambda: defaultdict(dict))
    for r in depth_law:
        if r["model"] != "real" or r["space"] != "counts" or r["log2_pred"] is None:
            continue
        key = (r["rung"], r["g_version"], r["mark_class"])
        groups[key][(r["kind"], r["source_pid"], r["target_pid"], r["log2_true"])][r["seed"]] = \
            r["log2_pred"]
    for (rung, gv, mc), by_pair in sorted(groups.items()):
        errs = {pk: abs(2.0 ** (_mean(s.values()) - pk[3]) - 1.0) for pk, s in by_pair.items()}
        seeds = sorted({sd for s in by_pair.values() for sd in s})
        per_seed = [max(abs(2.0 ** (s[sd] - pk[3]) - 1.0) for pk, s in by_pair.items() if sd in s)
                    for sd in seeds]
        worst = max(errs, key=errs.get)
        out.append(_check(rung, gv, "counts", mc, "depth_scale", "depthlaw", errs[worst],
                          DEPTH_BAR, le, wobble(per_seed),
                          {"n_pairs": len(errs), "worst_pair": list(worst[:3]),
                           "per_seed": per_seed, "seeds": seeds}))
    return out


def rung_choice(per_pair: list[dict]) -> dict:
    """Per "<g_version>|<space>": val crps_all, real, variant all, macro over all tracks per seed."""
    vt = [r for r in per_track_rows(per_pair, ev="val")
          if r["kind"] == "trained" and r["model"] == "real" and r["metric"] == "crps_all"
          and r["variant"] == "all"]
    acc: dict = defaultdict(lambda: defaultdict(list))
    for r in vt:
        acc[(r["g_version"], r["space"], r["rung"])][r["seed"]].append(r["value"])
    out: dict = {}
    for gv, space in sorted({k[:2] for k in acc}):
        val_by, wob_by = {}, {}
        for rung in pairs.RUNGS:
            by_seed = acc.get((gv, space, rung))
            if not by_seed:
                continue
            per_seed = [_mean(by_seed[s]) if s in by_seed else None for s in pairs.SEEDS]
            val_by[rung], wob_by[rung] = _mean(per_seed), wobble(per_seed)
        best = min(val_by, key=val_by.get)
        tol = wob_by[best] if wob_by[best] is not None else 0.0
        chosen = next(r for r in pairs.RUNGS if r in val_by and val_by[r] <= val_by[best] + tol)
        out[f"{gv}|{space}"] = {"chosen": chosen, "val_by_rung": val_by, "wobble_by_rung": wob_by,
                                "best": best, "tolerance": tol,
                                "rule": "lowest rung with val crps_all <= best + best's seed "
                                        "wobble (real, mean over seeds, macro over tracks)"}
    return out


# ---------------------------------------------------------------------------------------------
# the aggregate command
# ---------------------------------------------------------------------------------------------

NOTE = ("values against bars; `met` is a computed comparison, not a tick and not a verdict")
SUMMARY_COLS = ("rung", "g_version", "space", "model", "kind", "mark_class", "metric", "variant",
                "mean", "seed_wobble", "per_seed_0", "per_seed_1", "per_seed_2", "n_tracks")


def aggregate(manifest, covariates, runs_dir, refs_tsv, agg_dir) -> dict:
    rows = pairs.read_manifest(manifest)
    agg_dir = Path(agg_dir)
    agg_dir.mkdir(parents=True, exist_ok=True)
    per_pair, present, no_law, figdata = load_runs(runs_dir)
    missing = [n for n in expected_runs(rows) if n not in set(present)]
    if missing:
        _warn(f"{len(missing)} expected runs missing (listed in results.json)")
    per_track = per_track_rows(per_pair)
    per_class = per_class_rows(per_track)
    depth_law = depth_law_rows(per_pair)
    checks = compute_checks(per_class, depth_law)
    choice = rung_choice(per_pair)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    results = {"created_utc": now, "runs_dir": str(runs_dir), "runs_present": present,
               "runs_missing": missing, "runs_without_law": no_law,
               "refs": load_refs(refs_tsv), "per_pair": per_pair, "per_track": per_track,
               "per_class": per_class, "checks": checks, "knob_gain": knob_gain_rows(per_pair),
               "law_grid": law_grid_rows(per_pair), "depth_law": depth_law, "rung_choice": choice,
               "figdata": figdata}
    write_json(results, agg_dir / "results.json")
    with (agg_dir / "results_summary.tsv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(SUMMARY_COLS)
        for r in per_class:
            ps = r["per_seed"]
            w.writerow(["" if v is None else v for v in
                        (r["rung"], r["g_version"], r["space"], r["model"], r["kind"],
                         r["mark_class"], r["metric"], r["variant"], r["mean"], r["seed_wobble"],
                         ps[0], ps[1], ps[2], r["n_tracks"])])
    for rung in sorted({r["rung"] for r in per_class}):
        write_json({"rung": rung, "created_utc": now, "note": NOTE,
                    "checks": [c for c in checks if c["rung"] == rung
                               and c["check"] not in MAIN_ONLY]},
                   agg_dir / f"checks_{rung}.json")
    main_checks = []
    for key, ch in choice.items():
        gv, space = key.split("|")
        main_checks += [c for c in checks if c["rung"] == ch["chosen"] and c["g_version"] == gv
                        and c["space"] == space]
    write_json({"created_utc": now, "note": NOTE, "rung_choice": choice, "checks": main_checks},
               agg_dir / "checks_main.json")
    return results


# ---------------------------------------------------------------------------------------------
# QuantileMatching reference curves (figure 5B)
# ---------------------------------------------------------------------------------------------


def _hist(a: np.ndarray, space: str):
    if space == "counts":
        h = np.bincount(a)
        nz = np.flatnonzero(h)
        return nz, h[nz]
    return np.unique(a, return_counts=True)


def qm_curves(manifest, data_dir, out_json, chrom: str = "chr1") -> dict:
    """{space: {track: {arm_pid: {"knots_x", "knots_y"}}}}: `baseline_rungs.qm_fit` of the base->arm
    pair of each arm's first level (sorted) per track, on `chrom` only; at most QM_MAX_KNOTS knots
    kept (evenly spaced in knot index, both ends kept)."""
    rows = pairs.read_manifest(manifest)
    corpus = data.Corpus(data_dir)
    out: dict = {s: {} for s in pairs.SPACES}
    for track in pairs.track_ids(rows):
        prods = [r for r in pairs.usable_products(rows) if r["track"] == track]
        base = next(r for r in prods if r["arm"] == "base")
        firsts = {}
        for r in sorted((r for r in prods if r["arm"] != "base"), key=lambda r: r["pid"]):
            firsts.setdefault(r["arm"], r)
        for space in pairs.SPACES:
            x = corpus.get(base["pid"], space, chrom)
            hx = _hist(x, space)
            for arm, r in sorted(firsts.items()):
                kx, ky = br.qm_fit(*hx, *_hist(corpus.get(r["pid"], space, chrom), space))
                if kx.size > QM_MAX_KNOTS:
                    keep = np.unique(np.linspace(0, kx.size - 1, QM_MAX_KNOTS).round().astype(int))
                    kx, ky = kx[keep], ky[keep]
                out[space].setdefault(track, {})[r["pid"]] = {
                    "knots_x": [float(v) for v in kx], "knots_y": [float(v) for v in ky]}
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    write_json(out, Path(out_json))
    return out


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "qm-curves":
        ap = argparse.ArgumentParser(prog="aggregate.py qm-curves")
        for pos in ("manifest", "data_dir", "out_json"):
            ap.add_argument(pos, type=Path)
        a = ap.parse_args(argv[1:])
        qm_curves(a.manifest, a.data_dir, a.out_json)
        print(f"qm curves -> {a.out_json}")
        return 0
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for pos in ("manifest", "covariates", "runs_dir", "refs_tsv", "agg_dir"):
        ap.add_argument(pos, type=Path)
    a = ap.parse_args(argv)
    res = aggregate(a.manifest, a.covariates, a.runs_dir, a.refs_tsv, a.agg_dir)
    print(f"{len(res['runs_present'])} runs present, {len(res['runs_missing'])} missing; "
          f"{len(res['checks'])} checks -> {a.agg_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
