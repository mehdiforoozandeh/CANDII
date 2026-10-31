"""t130 blind check-set readings, step 1 (on Nibi, stdlib only): flatten the h1/h2 check-set
records of the t118 row-1 ladder runs and of the rungs_v2 baselines into small gzipped tables.

Reads, never writes, the run and baseline directories.

THE CHECK SET (pinned, plan_gaps.md G3): every trained pair of the row-1 ladder whose arm product
is `depth` (h1: levels 15M, 7.5M, 3.75M) or `pe` (h2), both directions, 7 tracks; spaces counts
and pval; rungs A, B, C, D; run names `<rung>_<g>_<space>_<model>_s<seed>`, g = a track id or
`all`, model in real / nocov / ids, seeds 0-2 (the 576 dirs under $LADDER/runs). A run dir whose
name does not parse as a row-1 run is skipped and counted.

SOURCE FIELD MAP. NOT yet re-read off a real scores.json on Nibi (the builder had no cluster
access); read off `tools/t118/ladder/score.py` (score_job, metric_fields, run_info, SWAP_KEY) and
`tools/t118/baseline_rungs.py` (run_pair, score_rung) at 7004fa4. Check it against one real file
before the first full extraction: `python3 extract_records.py --fields <run_dir>/scores.json`.
  scores.json  top level `run` {run_name, rung, g, g_version, space, model, seed}, `records`:
    kind (trained | shuffle | swap | depthlaw | heldout), eval (score | val), the pair keys
    `pairs.PAIR_KEYS` (track, cell, assay, mark_class, arm_src, level_src, arm_tgt, level_tgt,
    source_pid, target_pid, direction, knob_combo, counts_identical, pval_identical),
    crps_<all|nonzero|top1>, spearman_<all|nonzero|top1>, n_<all|nonzero|top1>; trained/score
    only: crps_oracle_scaled_all, scale_error_all, c_star_all; swap: swap_median_abs_log_ratio
    (-> column swap_ratio_median); depthlaw: depth_log2_scale_pred (-> log2_pred_scale),
    depth_log2_ratio_true (-> log2_depth_ratio).
  rungs_v2/<arm_pid>__<base_to_arm|arm_to_base>.json  `pair` {track, arm, level, arm_pid,
    direction, source_pid, target_pid, ...}, `records`: space, rung (noSolution |
    QuantileMatching), eval, spread_crps_<subset> (the Poisson / log-normal CRPS: -> crps_<subset>),
    crps_oracle_scaled_all, scale_error_all, spearman_all. The rung is written as in the source;
    analyze.py reads QuantileMatching as one-warp. Only eval == score records are kept.

Row identity. `pid`, `arm`, `level` name the ARM product of the pair (the non-base side); for a
swap record (source == target) they name that product. Swap records are kept for products whose
arm is depth, pe or base.

Outputs in <out_dir>:
  records.tsv.gz    one row per kept model record (RCOLS)
  baselines.tsv.gz  one row per kept baseline record (BCOLS)
  runs.tsv          one row per row-1 run dir (UCOLS)
  EXTRACT_DONE      written last

    python3 extract_records.py <ladder_runs_dir> <rungs_v2_dir> <out_dir>
    python3 extract_records.py --fields <scores.json>     # print the field names of one file
"""
import gzip
import json
import os
import re
import sys

KNOBS = ("depth", "pe")
KINDS = ("trained", "shuffle", "swap", "depthlaw")
SWAP_ARMS = KNOBS + ("base",)
BASE_RUNGS = ("noSolution", "QuantileMatching")
DIRECTIONS = ("base_to_arm", "arm_to_base")
RUN_RE = re.compile(r"^([ABCD])_(.+)_(counts|pval)_(real|nocov|ids)_s([0-2])$")

RCOLS = ["run", "rung", "g", "g_version", "model", "seed", "space", "kind", "eval", "track", "pid",
         "arm", "level", "direction", "source_pid", "target_pid", "crps_all", "crps_top1",
         "crps_nonzero", "spearman_all", "spearman_top1", "crps_oracle_scaled_all",
         "scale_error_all", "c_star_all", "swap_ratio_median", "log2_pred_scale",
         "log2_depth_ratio", "n_all", "n_top1"]
BCOLS = ["space", "rung", "track", "pid", "arm", "level", "direction", "source_pid", "target_pid",
         "crps_all", "crps_top1", "crps_oracle_scaled_all", "scale_error_all", "spearman_all"]
#: has_scores 0 = scores.json missing, truncated or not valid JSON (the run is listed, not read)
UCOLS = ["run", "rung", "g", "model", "seed", "space", "has_scores", "n_records"]
#: output column <- source field, where the names differ
RENAME = {"swap_ratio_median": "swap_median_abs_log_ratio",
          "log2_pred_scale": "depth_log2_scale_pred",
          "log2_depth_ratio": "depth_log2_ratio_true"}
BRENAME = {"crps_all": "spread_crps_all", "crps_top1": "spread_crps_top1"}


def cell(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(int(v))
    return str(v)


def line(rec, cols):
    return "\t".join(cell(rec.get(c)) for c in cols) + "\n"


def gv_of(g):
    return "across" if g == "all" else "per_track"


def arm_side(r):
    """(pid, arm, level) of the pair's arm product; None when neither side is the base."""
    if r["arm_src"] == "base" and r["arm_tgt"] != "base":
        return r["target_pid"], r["arm_tgt"], r["level_tgt"]
    if r["arm_tgt"] == "base" and r["arm_src"] != "base":
        return r["source_pid"], r["arm_src"], r["level_src"]
    if r["source_pid"] == r["target_pid"]:                  # swap: X = target = one product
        return r["source_pid"], r["arm_src"], r["level_src"]
    return None


def keep(kind, arm):
    return arm in (SWAP_ARMS if kind == "swap" else KNOBS)


def score_rows(path, meta):
    """The kept rows of one scores.json; None when the file is truncated or not valid JSON."""
    try:
        with open(path) as fh:
            s = json.load(fh)
        recs = s["records"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    run = s.get("run") or {}
    base = {"run": meta["run"], "rung": run.get("rung", meta["rung"]), "g": run.get("g", meta["g"]),
            "model": run.get("model", meta["model"]), "seed": run.get("seed", meta["seed"]),
            "space": run.get("space", meta["space"])}
    base["g_version"] = run.get("g_version") or gv_of(base["g"])
    out = []
    for r in recs:
        k = r.get("kind")
        if k not in KINDS:
            continue
        side = arm_side(r)
        if side is None or not keep(k, side[1]):
            continue
        rec = dict(base)
        rec.update({c: r.get(RENAME.get(c, c)) for c in RCOLS if c not in base})
        rec.update({"kind": k, "pid": side[0], "arm": side[1], "level": side[2]})
        out.append(line(rec, RCOLS))
    return out


def baseline_rows(path):
    with open(path) as fh:
        b = json.load(fh)
    p = b["pair"]
    if p.get("arm") not in KNOBS or p.get("direction") not in DIRECTIONS:
        return []
    out = []
    for r in b["records"]:
        if r.get("eval") != "score" or r.get("rung") not in BASE_RUNGS:
            continue
        rec = {"space": r["space"], "rung": r["rung"], "track": p["track"], "pid": p["arm_pid"],
               "arm": p["arm"], "level": p["level"], "direction": p["direction"],
               "source_pid": p["source_pid"], "target_pid": p["target_pid"]}
        for c in BCOLS:
            if c not in rec:
                rec[c] = r.get(BRENAME.get(c, c))
        out.append(line(rec, BCOLS))
    return out


def fields(path):
    with open(path) as fh:
        s = json.load(fh)
    print("run:", sorted((s.get("run") or {}).keys()))
    by_kind = {}
    for r in s["records"]:
        by_kind.setdefault((r.get("kind"), r.get("eval")), set()).update(r.keys())
    for k in sorted(by_kind, key=str):
        print(k, sorted(by_kind[k]))


def main(argv):
    if len(argv) == 2 and argv[0] == "--fields":
        fields(argv[1])
        return 0
    if len(argv) != 3:
        print(__doc__.strip().splitlines()[-2], file=sys.stderr)
        return 2
    runs, rungs, out = argv
    os.makedirs(out, exist_ok=True)
    n_runs = n_scores = n_rec = n_skip = 0
    with gzip.open(os.path.join(out, "records.tsv.gz.tmp"), "wt") as rh, \
            open(os.path.join(out, "runs.tsv.tmp"), "w") as uh:
        rh.write("\t".join(RCOLS) + "\n")
        uh.write("\t".join(UCOLS) + "\n")
        for name in sorted(os.listdir(runs)):
            d = os.path.join(runs, name)
            m = RUN_RE.match(name)
            if not os.path.isdir(d) or not m:
                n_skip += os.path.isdir(d)
                continue
            n_runs += 1
            meta = dict(zip(("rung", "g", "space", "model", "seed"), m.groups()), run=name)
            meta["seed"] = int(meta["seed"])
            sp = os.path.join(d, "scores.json")
            rows = score_rows(sp, meta) if os.path.isfile(sp) else None
            has = rows is not None                  # missing, truncated or invalid -> 0
            rows = rows or []
            n_scores += has
            n_rec += len(rows)
            rh.writelines(rows)
            uh.write(line({**meta, "has_scores": has, "n_records": len(rows)}, UCOLS))
    n_base = 0
    with gzip.open(os.path.join(out, "baselines.tsv.gz.tmp"), "wt") as bh:
        bh.write("\t".join(BCOLS) + "\n")
        for name in sorted(os.listdir(rungs)):
            stem = name[:-5] if name.endswith(".json") else None
            if not stem or not any(stem.endswith("__" + d) for d in DIRECTIONS):
                continue
            if not any(f"__{k}__" in stem for k in KNOBS):     # pid = <track>__<arm>__<level>
                continue
            rows = baseline_rows(os.path.join(rungs, name))
            n_base += len(rows)
            bh.writelines(rows)
    for f in ("records.tsv.gz", "baselines.tsv.gz", "runs.tsv"):
        os.replace(os.path.join(out, f + ".tmp"), os.path.join(out, f))
    with open(os.path.join(out, "EXTRACT_DONE"), "w") as fh:
        fh.write(f"runs {n_runs} with_scores {n_scores} records {n_rec} baselines {n_base} "
                 f"skipped_dirs {n_skip}\n")
    print(f"row-1 runs {n_runs} (scores.json in {n_scores}); records {n_rec}; baseline rows "
          f"{n_base}; non-row-1 dirs skipped {n_skip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
