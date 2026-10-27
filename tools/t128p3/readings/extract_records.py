"""t128 phase 3 readings, step 1 (on Fir, stdlib only): flatten the raw run files into small tables.

Reads, never writes, the run directories. Every directory under <root>/runs/ is read (one
cluster, no index lists); <root>/tasks.tsv, when present, adds the run index and the assigned
runs that have no directory yet. A run's scored records are read when it has LAW_DONE
(scores.json + law.json). Per-run fragments are cached under <out>/cache/<run>/ and rebuilt only
when the run's marks, its via-base mark, its train log, or this script (sha256) change.

Phase 2's extraction (tools/t128p2/readings/extract_records.py) without the DS-only columns and
files (no latent, cclean, hop1, train terms), with the sizes of each run:

Outputs in <out_dir>:
  records_fir.tsv.gz  one row per scored record (eval == score; kinds trained, heldout, law,
                      depthlaw, swap): run columns + arm, g_hidden, f_channels, n_theta,
                      max_steps, steps_run, best_step
  runs_fir.tsv        one row per run: run_dir (directory exists), marks TRAIN/SCORE/LAW_DONE,
                      viabase_LAW_DONE (<root>/viabase/<run>/LAW_DONE), rung, sizes, max_steps,
                      steps_run, best_step, val_nll, seconds per step, device, node (from the
                      newest train log of its index)
  viabase_fir.tsv.gz  law + depthlaw records of <root>/viabase/*/ ; arm = <the old run's arm> +
                      "viabase" (Dbiggviabase, Dbigfviabase), the old run read from
                      viabase.json["old_run_dir"]; run = the old run's name

arm = config["arm"], falling back to the run-name prefix.

    python3 extract_records.py <out_dir> <root>
"""
import glob
import gzip
import hashlib
import json
import os
import re
import sys

TAG = "fir"
COLS = ["run", "cluster", "arm", "loss", "rung", "g", "g_version", "space", "model", "seed",
        "g_hidden", "f_channels", "n_theta", "val_nll", "max_steps", "best_step", "steps",
        "steps_run", "kind", "group", "depth_source", "track", "mark_class", "arm_src",
        "level_src", "arm_tgt", "level_tgt", "direction", "identical", "crps_all", "crps_top1",
        "spearman_all", "crps_oracle_scaled_all", "scale_error_all", "c_star_all",
        "n_crps_nonfinite", "log2_true", "log2_pred", "swap_mal"]
RCOLS = ["index", "run", "cluster", "arm", "g", "g_version", "model", "seed", "rung", "g_hidden",
         "f_channels", "n_theta", "TRAIN_DONE", "SCORE_DONE", "LAW_DONE", "viabase_LAW_DONE",
         "max_steps", "steps_run", "best_step", "val_nll", "val_cross_nll", "val_nll_rule",
         "train_s", "s_per_step", "device", "node", "train_log", "run_dir"]
KINDS = ("trained", "heldout", "law", "depthlaw", "swap")
MARKS = ("TRAIN_DONE", "SCORE_DONE", "LAW_DONE")
SIZES = ("g_hidden", "f_channels", "n_theta")
with open(os.path.abspath(__file__), "rb") as _fh:     # a code change invalidates the cache
    CODE_SHA = hashlib.sha256(_fh.read()).hexdigest()[:16]


def cell(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(int(v))
    return str(v)


def line(rec, cols):
    return "\t".join(cell(rec.get(c)) for c in cols) + "\n"


def load(path):
    with open(path) as fh:
        return json.load(fh)


def load_cfg(d):
    p = os.path.join(d, "config.json")
    return load(p) if os.path.isfile(p) else None


def arm_of(name, cfg=None):
    if cfg and cfg.get("arm"):
        return cfg["arm"]
    return name.split("_")[0]


def gv_of(g):
    return "across" if g == "all" else "per_track"


def score_rows(path, extra, run_name=None):
    s = load(path)
    run = s["run"]
    out = []
    for r in s["records"]:
        k = r["kind"]
        if k not in KINDS or r.get("eval") != "score":
            continue
        rec = dict(extra)
        rec.update({"run": run_name or run["run_name"], "cluster": TAG, "rung": run["rung"],
                    "g": run["g"], "g_version": run["g_version"], "space": run["space"],
                    "model": run["model"], "seed": run["seed"], "val_nll": run.get("val_nll"),
                    "best_step": run.get("best_step"), "steps": run.get("steps"), "kind": k,
                    "identical": r.get(run["space"] + "_identical"),
                    "log2_true": r.get("depth_log2_ratio_true"),
                    "log2_pred": r.get("depth_log2_scale_pred"),
                    "swap_mal": r.get("swap_median_abs_log_ratio")})
        for c in COLS:
            if c not in rec:
                rec[c] = r.get(c)
        out.append(line(rec, COLS))
    return out


def node_of(logs, index):
    """(node, log name) from the newest train log of this index: '=== ... host=<node> ...'."""
    if index is None or not os.path.isdir(logs):
        return None, None
    fs = glob.glob(os.path.join(logs, f"t128p3_train_*_{index}.out"))
    if not fs:
        return None, None
    f = max(fs, key=os.path.getmtime)
    with open(f, errors="replace") as fh:
        for ln in fh:
            m = re.search(r"host=(\S+)", ln)
            if m:
                return m.group(1), os.path.basename(f)
    return None, os.path.basename(f)


def stamp_of(d, vdir, log):
    parts = []
    for m in MARKS:
        p = os.path.join(d, m)
        parts.append(f"{m}:{os.path.getmtime(p):.3f}" if os.path.isfile(p) else f"{m}:-")
    p = os.path.join(vdir, "LAW_DONE")
    parts.append(f"viabase:{os.path.getmtime(p):.3f}" if os.path.isfile(p) else "viabase:-")
    parts.append(f"log:{log}")
    parts.append(f"code:{CODE_SHA}")
    return "|".join(parts) + "\n"


def run_fragments(d, vdir, name, index, logs, cache):
    """(rec, row) lines of one run, from the cache when its marks are unchanged."""
    node, log = node_of(logs, index)
    cdir = os.path.join(cache, name)
    stamp = stamp_of(d, vdir, log)
    sfile = os.path.join(cdir, "stamp")
    files = [os.path.join(cdir, f) for f in ("rec.tsv", "row.tsv")]
    if os.path.isfile(sfile) and open(sfile).read() == stamp and all(map(os.path.isfile, files)):
        return [open(f).readlines() for f in files], True
    has = {m: os.path.isfile(os.path.join(d, m)) for m in MARKS}
    cfg = load_cfg(d)
    arm = arm_of(name, cfg)
    row = {"run": name, "cluster": TAG, "arm": arm, "node": node, "train_log": log, "run_dir": True,
           "viabase_LAW_DONE": os.path.isfile(os.path.join(vdir, "LAW_DONE")),
           **{m: has[m] for m in MARKS}}
    rec = []
    if cfg:
        row.update(g=cfg["g"], g_version=gv_of(cfg["g"]), model=cfg["model"], seed=cfg["seed"],
                   rung=cfg.get("rung"), max_steps=cfg.get("max_steps"), device=cfg.get("device"),
                   val_nll_rule=cfg.get("val_nll_rule"), **{k: cfg.get(k) for k in SIZES})
    if has["TRAIN_DONE"]:
        t = load(os.path.join(d, "timing.json"))
        sr = t.get("steps_run")
        row.update(steps_run=sr, best_step=t.get("best_step"), val_nll=t.get("val_nll"),
                   val_cross_nll=t.get("val_cross_nll"), train_s=t.get("train"),
                   s_per_step=(t["train"] / sr) if sr and t.get("train") is not None else None)
    if has["LAW_DONE"] and cfg:
        extra = {"arm": arm, "loss": cfg.get("loss"), "max_steps": cfg.get("max_steps"),
                 "steps_run": row.get("steps_run"), **{k: cfg.get(k) for k in SIZES}}
        for f in ("scores.json", "law.json"):
            rec += score_rows(os.path.join(d, f), extra)
    os.makedirs(cdir, exist_ok=True)
    frags = [rec, [line(row, RCOLS)]]
    for f, lines in zip(files, frags):
        with open(f, "w") as fh:
            fh.writelines(lines)
    with open(sfile, "w") as fh:
        fh.write(stamp)
    return frags, False


def viabase_rows(d, runs):
    """law + depthlaw rows of one via-base dir; the arm and sizes from the old run it read."""
    vj = os.path.join(d, "viabase.json")
    old = load(vj).get("old_run_dir") if os.path.isfile(vj) else None
    name = os.path.basename(os.path.normpath(old)) if old else os.path.basename(d)
    cfg = load_cfg(old) if old and os.path.isdir(old) else load_cfg(os.path.join(runs, name))
    extra = {"arm": arm_of(name, cfg) + "viabase", "rung": "D",
             **{k: (cfg or {}).get(k) for k in SIZES},
             "max_steps": (cfg or {}).get("max_steps")}
    return score_rows(os.path.join(d, "law.json"), extra, run_name=name)


def main():
    out, root = sys.argv[1:3]
    runs, logs = os.path.join(root, "runs"), os.path.join(root, "logs")
    via = os.path.join(root, "viabase")
    cache = os.path.join(out, "cache")
    os.makedirs(cache, exist_ok=True)
    index = {}
    tp = os.path.join(root, "tasks.tsv")
    if os.path.isfile(tp):
        with open(tp) as fh:
            head = fh.readline().rstrip("\n").split("\t")
            for ln in fh:
                r = dict(zip(head, ln.rstrip("\n").split("\t")))
                index[r["run_name"]] = int(r["index"])
    dirs = sorted(n for n in os.listdir(runs) if os.path.isdir(os.path.join(runs, n)))
    names = sorted(set(dirs) | set(index), key=lambda n: (index.get(n, 10 ** 9), n))
    n_hit = n_read = 0
    rech = gzip.open(os.path.join(out, f"records_{TAG}.tsv.gz.tmp"), "wt")
    rech.write("\t".join(COLS) + "\n")
    rh = open(os.path.join(out, f"runs_{TAG}.tsv.tmp"), "w")
    rh.write("\t".join(RCOLS) + "\n")
    counts = {m: 0 for m in MARKS}
    for name in names:
        d = os.path.join(runs, name)
        ix = index.get(name)
        if not os.path.isdir(d):
            rh.write(line({"index": ix, "run": name, "cluster": TAG, "arm": arm_of(name),
                           "run_dir": False}, RCOLS))
            continue
        (rec, row), hit = run_fragments(d, os.path.join(via, name), name, ix, logs, cache)
        n_hit += hit
        n_read += not hit
        rech.writelines(rec)
        rh.write(cell(ix) + row[0])          # the cached row line starts with an empty index
        for m in MARKS:
            counts[m] += os.path.isfile(os.path.join(d, m))
    rech.close()
    rh.close()
    os.replace(os.path.join(out, f"records_{TAG}.tsv.gz.tmp"),
               os.path.join(out, f"records_{TAG}.tsv.gz"))
    os.replace(os.path.join(out, f"runs_{TAG}.tsv.tmp"), os.path.join(out, f"runs_{TAG}.tsv"))
    print(f"{TAG}: {len(index)} assigned runs in tasks.tsv, {len(dirs)} run dirs; marks {counts}; "
          f"cache hits {n_hit}, read {n_read}")
    n = 0
    with gzip.open(os.path.join(out, f"viabase_{TAG}.tsv.gz.tmp"), "wt") as fh:
        fh.write("\t".join(COLS) + "\n")
        for name in sorted(os.listdir(via)) if os.path.isdir(via) else []:
            d = os.path.join(via, name)
            if not os.path.isfile(os.path.join(d, "LAW_DONE")):
                continue
            fh.writelines(viabase_rows(d, runs))
            n += 1
    os.replace(os.path.join(out, f"viabase_{TAG}.tsv.gz.tmp"),
               os.path.join(out, f"viabase_{TAG}.tsv.gz"))
    print(f"{TAG}: via-base runs with LAW_DONE: {n}")


if __name__ == "__main__":
    main()
