"""t128 readings, step 1 (on a cluster, stdlib only): flatten the raw run files into small tables.

Reads, never writes, the run directories. Only the runs that the index file assigns to THIS
cluster are read (Fir: fir_idx.txt, Nibi: nibi_idx.txt); a copy of a run on the other cluster
is ignored. A run is read when it has LAW_DONE (scores.json + law.json); its latent records
are read when it also has LATENT_DONE. Per-run fragments are cached under <out>/cache/<run>/
and rebuilt only when the run's DONE marks change (incremental re-runs).

Outputs in <out_dir> (tag = cluster name):
  records_<tag>.tsv.gz  one row per scored record (kinds trained, heldout, law, depthlaw, swap;
                        eval == score): t126's columns + arm, setting, cclean_form, loss, lam, vae,
                        steps_run, c_star_all, swap_mal, cluster
  latent_<tag>.tsv.gz   one row per latent.json record (kinds latent, agree, ident)
  cclean_<tag>.tsv      one row per run x assay x knob: init, learned, lo, hi, in_g
  runs_<tag>.tsv        one row per assigned run: its marks and training summary
  viabase_<tag>.tsv.gz  (with --viabase) law + depthlaw records of old D via the base, arm Dviabase
  oldd_split_<tag>.tsv.gz (with --oldd-split) old D's law + depthlaw records re-scored with the
                        CRPS split (t128/oldD_split/), arm D

    python3 extract_records.py <out_dir> <tag> <runs_dir> <idx_file> [--viabase <viabase_dir>]
        [--oldd-split <oldD_split_dir>]
"""
import gzip
import json
import os
import sys

COLS = ["run", "cluster", "arm", "setting", "cclean_form", "loss", "lam", "vae", "rung", "g",
        "g_version", "space", "model", "seed", "val_nll", "best_step", "steps", "steps_run",
        "kind", "depth_source", "track", "mark_class", "arm_src", "level_src", "arm_tgt",
        "level_tgt", "direction", "identical", "crps_all", "crps_top1", "spearman_all",
        "crps_oracle_scaled_all", "scale_error_all", "c_star_all", "n_crps_nonfinite",
        "log2_true", "log2_pred", "swap_mal"]
LCOLS = ["run", "cluster", "arm", "g", "g_version", "model", "seed", "kind", "track", "pid",
         "arm_p", "level", "n_bins", "median_abs_z_minus_x", "corr_z_x", "mean_z", "mean_x",
         "median_sigma_z", "pid_a", "pid_b", "arm_a", "level_a", "arm_b", "level_b",
         "median_abs_z_diff", "median_abs_x_diff", "corr_z_a_b", "group", "arm_src", "level_src",
         "arm_tgt", "level_tgt", "direction", "ident"]
CCOLS = ["run", "cluster", "arm", "g", "g_version", "model", "seed", "assay", "knob_i", "knob",
         "init", "learned", "lo", "hi", "in_g", "box"]
RCOLS = ["index", "run", "cluster", "arm", "g", "model", "seed", "TRAIN_DONE", "SCORE_DONE",
         "LAW_DONE", "LATENT_DONE", "steps_run", "best_step", "val_nll", "val_cross_nll",
         "val_nll_rule", "train_s"]
KINDS = ("trained", "heldout", "law", "depthlaw", "swap")
MARKS = ("TRAIN_DONE", "SCORE_DONE", "LAW_DONE", "LATENT_DONE")


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


def score_rows(path, tag, extra):
    s = load(path)
    run = s["run"]
    out = []
    for r in s["records"]:
        k = r["kind"]
        if k not in KINDS or r.get("eval") != "score":
            continue
        rec = dict(extra)
        rec.update({"run": run["run_name"], "cluster": tag, "rung": run["rung"], "g": run["g"],
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


def latent_rows(path, tag, arm):
    s = load(path)
    run = s["run"]
    base = {"run": run["run_name"], "cluster": tag, "arm": arm, "g": run["g"],
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


def cclean_rows(d, tag, arm, cfg):
    cc = load(os.path.join(d, "cclean.json"))
    knobs = cfg["cov_fields"][:13]
    in_g = {p["assay"] for p in cfg["train_pairs"]}
    lo, hi = cfg.get("box_lo") or {}, cfg.get("box_hi") or {}
    out = []
    for assay, vals in cc["assays"].items():
        init = cc["init"].get(assay) or [None] * len(vals)
        for i, v in enumerate(vals):
            rec = {"run": cfg["run_name"], "cluster": tag, "arm": arm, "g": cfg["g"],
                   "g_version": "across" if cfg["g"] == "all" else "per_track",
                   "model": cfg["model"], "seed": cfg["seed"], "assay": assay, "knob_i": i,
                   "knob": knobs[i], "init": init[i], "learned": v,
                   "lo": lo[assay][i] if assay in lo else None,
                   "hi": hi[assay][i] if assay in hi else None,
                   "in_g": assay in in_g, "box": bool(cc.get("box"))}
            out.append(line(rec, CCOLS))
    return out


def stamp_of(d):
    parts = []
    for m in MARKS:
        p = os.path.join(d, m)
        parts.append(f"{m}:{os.path.getmtime(p):.3f}" if os.path.isfile(p) else f"{m}:-")
    return "|".join(parts) + "\n"


def run_fragments(d, name, tag, cache):
    """(rec, lat, cc, row) lines of one run, from the cache when its marks are unchanged."""
    cdir = os.path.join(cache, name)
    stamp = stamp_of(d)
    sfile = os.path.join(cdir, "stamp")
    files = [os.path.join(cdir, f) for f in ("rec.tsv", "lat.tsv", "cc.tsv", "row.tsv")]
    if os.path.isfile(sfile) and open(sfile).read() == stamp and all(map(os.path.isfile, files)):
        return [open(f).readlines() for f in files], True
    has = {m: os.path.isfile(os.path.join(d, m)) for m in MARKS}
    arm = arm_of(name)
    row = {"run": name, "cluster": tag, "arm": arm, **{m: has[m] for m in MARKS}}
    rec, lat, cc = [], [], []
    cfg = load(os.path.join(d, "config.json")) if os.path.isfile(os.path.join(d, "config.json")) else None
    if has["TRAIN_DONE"]:
        t = load(os.path.join(d, "timing.json"))
        row.update(steps_run=t.get("steps_run"), best_step=t.get("best_step"),
                   val_nll=t.get("val_nll"), val_cross_nll=t.get("val_cross_nll"),
                   train_s=t.get("train"), val_nll_rule=cfg and cfg.get("val_nll_rule"))
        cc = cclean_rows(d, tag, arm, cfg)
    if cfg:
        row.update(g=cfg["g"], model=cfg["model"], seed=cfg["seed"])
    if has["LAW_DONE"]:
        st = cfg["setting"]
        extra = {"arm": arm, "setting": st["number"], "cclean_form": cfg["cclean_form"],
                 "loss": st["loss"], "lam": cfg["lam"], "vae": cfg["vae"],
                 "steps_run": row.get("steps_run")}
        for f in ("scores.json", "law.json"):
            rec += score_rows(os.path.join(d, f), tag, extra)
        if has["LATENT_DONE"]:
            lat = latent_rows(os.path.join(d, "latent.json"), tag, arm)
    os.makedirs(cdir, exist_ok=True)
    frags = [rec, lat, cc, [line(row, RCOLS)]]
    for f, lines in zip(files, frags):
        with open(f, "w") as fh:
            fh.writelines(lines)
    with open(sfile, "w") as fh:
        fh.write(stamp)
    return frags, False


def main():
    args = sys.argv[1:]
    opt = {}
    for flag in ("--viabase", "--oldd-split"):
        if flag in args:
            i = args.index(flag)
            opt[flag] = args[i + 1]
            args = args[:i] + args[i + 2:]
    out, tag, runs, idx_file = args
    cache = os.path.join(out, "cache")
    os.makedirs(cache, exist_ok=True)
    assigned = []
    for ln in open(idx_file):
        p = ln.split()
        if len(p) == 2:
            assigned.append((int(p[0]), p[1]))
    n_hit = n_read = 0
    handles = {k: gzip.open(os.path.join(out, f"{k}_{tag}.tsv.gz.tmp"), "wt")
               for k in ("records", "latent")}
    handles["records"].write("\t".join(COLS) + "\n")
    handles["latent"].write("\t".join(LCOLS) + "\n")
    cch = open(os.path.join(out, f"cclean_{tag}.tsv.tmp"), "w")
    cch.write("\t".join(CCOLS) + "\n")
    rh = open(os.path.join(out, f"runs_{tag}.tsv.tmp"), "w")
    rh.write("\t".join(RCOLS) + "\n")
    counts = {m: 0 for m in MARKS}
    for index, name in assigned:
        d = os.path.join(runs, name)
        if not os.path.isdir(d):
            rh.write(line({"index": index, "run": name, "cluster": tag, "arm": arm_of(name)}, RCOLS))
            continue
        (rec, lat, cc, row), hit = run_fragments(d, name, tag, cache)
        n_hit += hit
        n_read += not hit
        handles["records"].writelines(rec)
        handles["latent"].writelines(lat)
        cch.writelines(cc)
        rh.write(str(index) + row[0])          # the cached row line starts with an empty index
        for m in MARKS:
            counts[m] += os.path.isfile(os.path.join(d, m))
    for h in list(handles.values()) + [cch, rh]:
        h.close()
    for k in ("records", "latent"):
        os.replace(os.path.join(out, f"{k}_{tag}.tsv.gz.tmp"), os.path.join(out, f"{k}_{tag}.tsv.gz"))
    for k in ("cclean", "runs"):
        os.replace(os.path.join(out, f"{k}_{tag}.tsv.tmp"), os.path.join(out, f"{k}_{tag}.tsv"))
    print(f"{tag}: {len(assigned)} assigned runs; marks {counts}; cache hits {n_hit}, read {n_read}")
    for flag, stem, arm, label in (("--viabase", "viabase", "Dviabase", "via-base"),
                                   ("--oldd-split", "oldd_split", "D", "old D split")):
        if flag not in opt:
            continue
        src, n = opt[flag], 0
        with gzip.open(os.path.join(out, f"{stem}_{tag}.tsv.gz.tmp"), "wt") as fh:
            fh.write("\t".join(COLS) + "\n")
            for name in sorted(os.listdir(src)) if os.path.isdir(src) else []:
                d = os.path.join(src, name)
                if not os.path.isfile(os.path.join(d, "LAW_DONE")):
                    continue
                extra = {"arm": arm, "setting": "", "cclean_form": "", "loss": "",
                         "lam": "", "vae": ""}
                fh.writelines(score_rows(os.path.join(d, "law.json"), tag, extra))
                n += 1
        os.replace(os.path.join(out, f"{stem}_{tag}.tsv.gz.tmp"),
                   os.path.join(out, f"{stem}_{tag}.tsv.gz"))
        print(f"{tag}: {label} runs with LAW_DONE: {n}")


if __name__ == "__main__":
    main()
