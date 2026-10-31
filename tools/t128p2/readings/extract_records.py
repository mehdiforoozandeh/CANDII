"""t128 phase 2 readings, step 1 (on Fir, stdlib only): flatten the raw run files into small tables.

Reads, never writes, the run directories. Every directory under <root>/runs/ is read (one
cluster, no index lists); <root>/tasks.tsv, when present, adds the run index and the assigned
runs that have no directory yet. A run's scored records are read when it has LAW_DONE
(scores.json + law.json); latent.json when it has LATENT_DONE; hop1.json when it has HOP1_DONE.
Per-run fragments are cached under <out>/cache/<run>/ and rebuilt only when the run's marks,
its train log, or this script (sha256) change (incremental re-runs).

Outputs in <out_dir>:
  records_fir.tsv.gz  one row per scored record (eval == score; kinds trained, heldout, law,
                      depthlaw, swap, hop1): phase-1 columns + arm, setting, form_rung,
                      cclean_form, cclean_frozen, max_steps, steps_run, best_step, c_star_all,
                      group (hop1 records carry their own group: trained / heldout)
  latent_fir.tsv.gz   one row per latent.json record (kinds latent, agree, ident)
  cclean_fir.tsv      one row per run x assay x knob: init, learned, frozen, in_g
  runs_fir.tsv        one row per run: run_dir (directory exists), marks, steps_run,
                      best_step, val_nll, val_cross_nll, seconds per step, device, node (from the newest train log of its index)
  viabase_fir.tsv.gz  law + depthlaw records of <root>/viabase/*/ (arm Dviabase)

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
COLS = ["run", "cluster", "arm", "setting", "form_rung", "cclean_form", "cclean_frozen", "loss",
        "lam", "vae", "rung", "g", "g_version", "space", "model", "seed", "val_nll", "max_steps",
        "best_step", "steps", "steps_run", "kind", "group", "depth_source", "track", "mark_class",
        "arm_src", "level_src", "arm_tgt", "level_tgt", "direction", "identical", "crps_all",
        "crps_top1", "spearman_all", "crps_oracle_scaled_all", "scale_error_all", "c_star_all",
        "n_crps_nonfinite", "log2_true", "log2_pred", "swap_mal"]
LCOLS = ["run", "cluster", "arm", "g", "g_version", "model", "seed", "kind", "track", "pid",
         "arm_p", "level", "n_bins", "median_abs_z_minus_x", "corr_z_x", "mean_z", "mean_x",
         "median_sigma_z", "pid_a", "pid_b", "arm_a", "level_a", "arm_b", "level_b",
         "median_abs_z_diff", "median_abs_x_diff", "corr_z_a_b", "group", "arm_src", "level_src",
         "arm_tgt", "level_tgt", "direction", "ident"]
CCOLS = ["run", "cluster", "arm", "g", "g_version", "model", "seed", "assay", "knob_i", "knob",
         "init", "learned", "frozen", "in_g"]
RCOLS = ["index", "run", "cluster", "arm", "g", "g_version", "model", "seed", "TRAIN_DONE",
         "SCORE_DONE", "LAW_DONE", "LATENT_DONE", "HOP1_DONE", "max_steps", "steps_run",
         "best_step", "val_nll", "val_cross_nll", "val_nll_rule", "train_s", "s_per_step",
         "device", "node", "train_log", "run_dir"]
KINDS = ("trained", "heldout", "law", "depthlaw", "swap", "hop1")
MARKS = ("TRAIN_DONE", "SCORE_DONE", "LAW_DONE", "LATENT_DONE", "HOP1_DONE")
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


def arm_of(name):
    return name.split("_")[0]


def gv_of(g):
    return "across" if g == "all" else "per_track"


def score_rows(path, extra):
    s = load(path)
    run = s["run"]
    out = []
    for r in s["records"]:
        k = r["kind"]
        if k not in KINDS or r.get("eval") != "score":
            continue
        rec = dict(extra)
        rec.update({"run": run["run_name"], "cluster": TAG, "rung": run["rung"], "g": run["g"],
                    "g_version": run["g_version"], "space": run["space"], "model": run["model"],
                    "seed": run["seed"], "val_nll": run.get("val_nll"),
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


def latent_rows(path, arm):
    s = load(path)
    run = s["run"]
    base = {"run": run["run_name"], "cluster": TAG, "arm": arm, "g": run["g"],
            "g_version": run["g_version"], "model": run["model"], "seed": run["seed"]}
    out = []
    for r in s["records"]:
        rec = dict(base)
        rec.update(r)
        if r["kind"] == "latent":
            rec["arm_p"] = r["arm"]
            rec["arm"] = arm
        rec["ident"] = r.get("identity_median_abs_log_ratio")
        out.append(line(rec, LCOLS))
    return out


def cclean_rows(d, arm, cfg):
    cc = load(os.path.join(d, "cclean.json"))
    knobs = cfg["cov_fields"][:13]
    in_g = {p["assay"] for p in cfg["train_pairs"]}
    out = []
    for assay, vals in cc["assays"].items():
        init = cc["init"].get(assay) or [None] * len(vals)
        for i, v in enumerate(vals):
            rec = {"run": cfg["run_name"], "cluster": TAG, "arm": arm, "g": cfg["g"],
                   "g_version": gv_of(cfg["g"]), "model": cfg["model"], "seed": cfg["seed"],
                   "assay": assay, "knob_i": i, "knob": knobs[i], "init": init[i], "learned": v,
                   "frozen": bool(cc.get("frozen")), "in_g": assay in in_g}
            out.append(line(rec, CCOLS))
    return out


def node_of(logs, index):
    """(node, log name) from the newest train log of this index: '=== ... host=<node> ...'."""
    if index is None or not os.path.isdir(logs):
        return None, None
    fs = glob.glob(os.path.join(logs, f"t128p2_train_*_{index}.out"))
    if not fs:
        return None, None
    f = max(fs, key=os.path.getmtime)
    with open(f, errors="replace") as fh:
        for ln in fh:
            m = re.search(r"host=(\S+)", ln)
            if m:
                return m.group(1), os.path.basename(f)
    return None, os.path.basename(f)


def stamp_of(d, log):
    parts = []
    for m in MARKS:
        p = os.path.join(d, m)
        parts.append(f"{m}:{os.path.getmtime(p):.3f}" if os.path.isfile(p) else f"{m}:-")
    parts.append(f"log:{log}")
    parts.append(f"code:{CODE_SHA}")
    return "|".join(parts) + "\n"


def run_fragments(d, name, index, logs, cache):
    """(rec, lat, cc, row) lines of one run, from the cache when its marks are unchanged."""
    node, log = node_of(logs, index)
    cdir = os.path.join(cache, name)
    stamp = stamp_of(d, log)
    sfile = os.path.join(cdir, "stamp")
    files = [os.path.join(cdir, f) for f in ("rec.tsv", "lat.tsv", "cc.tsv", "row.tsv")]
    if os.path.isfile(sfile) and open(sfile).read() == stamp and all(map(os.path.isfile, files)):
        return [open(f).readlines() for f in files], True
    has = {m: os.path.isfile(os.path.join(d, m)) for m in MARKS}
    arm = arm_of(name)
    row = {"run": name, "cluster": TAG, "arm": arm, "node": node, "train_log": log, "run_dir": True,
           **{m: has[m] for m in MARKS}}
    rec, lat, cc = [], [], []
    cp = os.path.join(d, "config.json")
    cfg = load(cp) if os.path.isfile(cp) else None
    if cfg:
        row.update(g=cfg["g"], g_version=gv_of(cfg["g"]), model=cfg["model"], seed=cfg["seed"],
                   max_steps=cfg.get("max_steps"), device=cfg.get("device"),
                   val_nll_rule=cfg.get("val_nll_rule"))
    if has["TRAIN_DONE"]:
        t = load(os.path.join(d, "timing.json"))
        sr = t.get("steps_run")
        row.update(steps_run=sr, best_step=t.get("best_step"), val_nll=t.get("val_nll"),
                   val_cross_nll=t.get("val_cross_nll"), train_s=t.get("train"),
                   s_per_step=(t["train"] / sr) if sr and t.get("train") is not None else None)
        if os.path.isfile(os.path.join(d, "cclean.json")):
            cc = cclean_rows(d, arm, cfg)
    if has["LAW_DONE"]:
        st = cfg.get("setting") or {}
        extra = {"arm": arm, "setting": st.get("number"), "form_rung": cfg["rung"],
                 "cclean_form": cfg.get("cclean_form"), "cclean_frozen": cfg.get("cclean_frozen"),
                 "loss": st.get("loss"), "lam": cfg.get("lam"), "vae": cfg.get("vae"),
                 "max_steps": cfg.get("max_steps"), "steps_run": row.get("steps_run")}
        for f in ("scores.json", "law.json"):
            rec += score_rows(os.path.join(d, f), extra)
        if has["HOP1_DONE"]:
            rec += score_rows(os.path.join(d, "hop1.json"), extra)
        if has["LATENT_DONE"]:
            lat = latent_rows(os.path.join(d, "latent.json"), arm)
    os.makedirs(cdir, exist_ok=True)
    frags = [rec, lat, cc, [line(row, RCOLS)]]
    for f, lines in zip(files, frags):
        with open(f, "w") as fh:
            fh.writelines(lines)
    with open(sfile, "w") as fh:
        fh.write(stamp)
    return frags, False


def main():
    out, root = sys.argv[1:3]
    runs, logs = os.path.join(root, "runs"), os.path.join(root, "logs")
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
    handles = {k: gzip.open(os.path.join(out, f"{k}_{TAG}.tsv.gz.tmp"), "wt")
               for k in ("records", "latent")}
    handles["records"].write("\t".join(COLS) + "\n")
    handles["latent"].write("\t".join(LCOLS) + "\n")
    cch = open(os.path.join(out, f"cclean_{TAG}.tsv.tmp"), "w")
    cch.write("\t".join(CCOLS) + "\n")
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
        (rec, lat, cc, row), hit = run_fragments(d, name, ix, logs, cache)
        n_hit += hit
        n_read += not hit
        handles["records"].writelines(rec)
        handles["latent"].writelines(lat)
        cch.writelines(cc)
        rh.write(cell(ix) + row[0])          # the cached row line starts with an empty index
        for m in MARKS:
            counts[m] += os.path.isfile(os.path.join(d, m))
    for h in list(handles.values()) + [cch, rh]:
        h.close()
    for k in ("records", "latent"):
        os.replace(os.path.join(out, f"{k}_{TAG}.tsv.gz.tmp"), os.path.join(out, f"{k}_{TAG}.tsv.gz"))
    for k in ("cclean", "runs"):
        os.replace(os.path.join(out, f"{k}_{TAG}.tsv.tmp"), os.path.join(out, f"{k}_{TAG}.tsv"))
    print(f"{TAG}: {len(index)} assigned runs in tasks.tsv, {len(dirs)} run dirs; marks {counts}; "
          f"cache hits {n_hit}, read {n_read}")
    src, n = os.path.join(root, "viabase"), 0
    with gzip.open(os.path.join(out, f"viabase_{TAG}.tsv.gz.tmp"), "wt") as fh:
        fh.write("\t".join(COLS) + "\n")
        for name in sorted(os.listdir(src)) if os.path.isdir(src) else []:
            d = os.path.join(src, name)
            if not os.path.isfile(os.path.join(d, "LAW_DONE")):
                continue
            extra = {"arm": "Dviabase", "form_rung": "D"}
            fh.writelines(score_rows(os.path.join(d, "law.json"), extra))
            n += 1
    os.replace(os.path.join(out, f"viabase_{TAG}.tsv.gz.tmp"),
               os.path.join(out, f"viabase_{TAG}.tsv.gz"))
    print(f"{TAG}: via-base runs with LAW_DONE: {n}")


if __name__ == "__main__":
    main()
