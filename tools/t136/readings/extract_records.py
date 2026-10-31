"""t136 readings, step 1 (on the cluster, stdlib only): flatten the raw run files into small tables.

Reads, never writes, the run directories. Two modes:

  grid (Nibi): every directory under <root>/runs/ (the Darm1 / Darm2 grid) plus the assigned runs
    of <root>/tasks.tsv that have no directory yet. arm = config["arm"] (fallback: the run-name
    prefix). scores.json is read when the run has SCORE_DONE, law.json when it has LAW_DONE.
    Law records carry `is_extra` = 1 when the (source_pid, target_pid) pair is in
    config["extra_pairs"] (the arm->arm pairs the arm trained on), else 0.
      python3 extract_records.py grid <out_dir> <root>       -> records_nibi.tsv.gz, runs_nibi.tsv
  ref (Fir): the phase-2 old-D references, read-only. <root>/runs/D_<g>_counts_real_s<seed> with
    g != "all" -> arm Dref (scores.json + law.json); <root>/viabase/D_<g>_counts_real_s<seed> ->
    arm Dviabase (law.json only). is_extra = 0 on every reference record.
      python3 extract_records.py ref <out_dir> <root>   -> ref_records_fir.tsv.gz, ref_runs_fir.tsv

Records kept: eval == "score" and kind in trained, heldout, law, depthlaw.
Runs rows: marks, steps (timing.json), best step, the train-pair and extra-pair counts, and the
number of train / extra pairs that hold a held-out depth-7.5M product (pid "<track>__depth__7.5M").
"""
import gzip
import json
import os
import sys

COLS = ["run", "cluster", "arm", "g", "g_version", "space", "model", "seed", "max_steps",
        "best_step", "steps_run", "kind", "depth_source", "track", "mark_class", "arm_src",
        "level_src", "arm_tgt", "level_tgt", "direction", "source_pid", "target_pid", "is_extra",
        "crps_all", "crps_oracle_scaled_all", "scale_error_all", "n_crps_nonfinite", "log2_true",
        "log2_pred"]
RCOLS = ["index", "run", "cluster", "arm", "g", "seed", "run_dir", "TRAIN_DONE", "SCORE_DONE",
         "LAW_DONE", "max_steps", "steps_run", "best_step", "val_nll", "n_train_pairs",
         "n_train_pairs_75", "n_extra_pairs", "n_extra_pairs_75", "holdout", "git_sha"]
KINDS = ("trained", "heldout", "law", "depthlaw")
MARKS = ("TRAIN_DONE", "SCORE_DONE", "LAW_DONE")
HELD_TAG = "__depth__7.5M"


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


def n75(pairs):
    return sum(1 for p in pairs if HELD_TAG in (p.get("source_pid") or "")
               or HELD_TAG in (p.get("target_pid") or ""))


def score_rows(path, base, extra_keys, cluster):
    s = load(path)
    run = s["run"]
    out = []
    for r in s["records"]:
        if r.get("kind") not in KINDS or r.get("eval") != "score":
            continue
        rec = dict(base)
        rec.update({"run": run["run_name"], "cluster": cluster, "g": run["g"],
                    "g_version": run["g_version"], "space": run["space"], "model": run["model"],
                    "seed": run["seed"], "kind": r["kind"],
                    "log2_true": r.get("depth_log2_ratio_true"),
                    "log2_pred": r.get("depth_log2_scale_pred")})
        rec["is_extra"] = int(r["kind"] == "law"
                              and (r.get("source_pid"), r.get("target_pid")) in extra_keys)
        for c in COLS:
            if c not in rec:
                rec[c] = r.get(c)
        out.append(line(rec, COLS))
    return out


def run_dir(d, name, arm, cluster, index=None, files=("scores.json", "law.json")):
    """(record lines, run row) of one run directory."""
    has = {m: os.path.isfile(os.path.join(d, m)) for m in MARKS}
    row = {"index": index, "run": name, "cluster": cluster, "arm": arm, "run_dir": True, **has}
    cfg = None
    cp = os.path.join(d, "config.json")
    if os.path.isfile(cp):
        cfg = load(cp)
        tp, ep = cfg.get("train_pairs") or [], cfg.get("extra_pairs") or []
        row.update(g=cfg.get("g"), seed=cfg.get("seed"), max_steps=cfg.get("max_steps"),
                   n_train_pairs=len(tp), n_train_pairs_75=n75(tp), n_extra_pairs=len(ep),
                   n_extra_pairs_75=n75(ep), git_sha=(cfg.get("git_sha") or "")[:7],
                   holdout=";".join(":".join(h) for h in cfg.get("holdout") or []))
    tj = os.path.join(d, "timing.json")
    if has["TRAIN_DONE"] and os.path.isfile(tj):
        t = load(tj)
        row.update(steps_run=t.get("steps_run"), best_step=t.get("best_step"),
                   val_nll=t.get("val_nll"))
    extra_keys = {(p.get("source_pid"), p.get("target_pid"))
                  for p in ((cfg or {}).get("extra_pairs") or [])}
    base = {"arm": arm, "max_steps": row.get("max_steps"), "best_step": row.get("best_step"),
            "steps_run": row.get("steps_run")}
    rec = []
    for f in files:
        mark = "SCORE_DONE" if f == "scores.json" else "LAW_DONE"
        if has[mark] and os.path.isfile(os.path.join(d, f)):
            rec += score_rows(os.path.join(d, f), base, extra_keys, cluster)
    return rec, row


def grid(out, root, tag="nibi"):
    runs = os.path.join(root, "runs")
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
    recs, rows = [], []
    for name in names:
        d = os.path.join(runs, name)
        if not os.path.isdir(d):
            rows.append(line({"index": index.get(name), "run": name, "cluster": tag,
                              "arm": name.split("_")[0], "g": name.split("_")[1],
                              "seed": int(name.rsplit("_s", 1)[1]), "run_dir": False}, RCOLS))
            continue
        cp = os.path.join(d, "config.json")
        arm = (load(cp).get("arm") if os.path.isfile(cp) else None) or name.split("_")[0]
        r, row = run_dir(d, name, arm, tag, index.get(name))
        recs += r
        rows.append(line(row, RCOLS))
    write(out, f"records_{tag}.tsv.gz", f"runs_{tag}.tsv", recs, rows)
    marks = {m: sum(ln.split("\t")[RCOLS.index(m)] == "1" for ln in rows) for m in MARKS}
    print(f"{tag} grid: {len(index)} assigned, {len(dirs)} run dirs, marks {marks}, "
          f"{len(recs)} records")


def ref(out, root, tag="fir"):
    recs, rows = [], []
    for sub, arm, files in (("runs", "Dref", ("scores.json", "law.json")),
                            ("viabase", "Dviabase", ("law.json",))):
        src = os.path.join(root, sub)
        for name in sorted(os.listdir(src)):
            parts = name.split("_")
            if parts[0] != "D" or parts[1] == "all" or "_counts_real_s" not in name:
                continue
            r, row = run_dir(os.path.join(src, name), name, arm, tag, files=files)
            row.setdefault("g", parts[1])
            row.setdefault("seed", int(name.rsplit("_s", 1)[1]))
            recs += r
            rows.append(line(row, RCOLS))
    write(out, f"ref_records_{tag}.tsv.gz", f"ref_runs_{tag}.tsv", recs, rows)
    print(f"{tag} ref: {len(rows)} runs (Dref + Dviabase), {len(recs)} records")


def write(out, rname, runs_name, recs, rows):
    os.makedirs(out, exist_ok=True)
    with gzip.open(os.path.join(out, rname + ".tmp"), "wt") as fh:
        fh.write("\t".join(COLS) + "\n")
        fh.writelines(recs)
    os.replace(os.path.join(out, rname + ".tmp"), os.path.join(out, rname))
    with open(os.path.join(out, runs_name + ".tmp"), "w") as fh:
        fh.write("\t".join(RCOLS) + "\n")
        fh.writelines(rows)
    os.replace(os.path.join(out, runs_name + ".tmp"), os.path.join(out, runs_name))


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) == 3 and a[0] == "grid":
        grid(a[1], a[2])
    elif len(a) == 3 and a[0] == "ref":
        ref(a[1], a[2])
    else:
        raise SystemExit(__doc__)
