"""Flatten the per-biosample ingestion-census JSONs into the vault mirror (t78).

Writes `per_track.tsv` (one row per track) and `per_track.json` (the same rows plus the
top ingested windows with coordinates).

Four window sets per track — `{eic.19, eic.pilot} x {train, eval}`.  Which one applies is
decided by the BIOSAMPLE, not by the track: a `T_` biosample is in the regime's train pool
and so sees train-split windows; `V_`/`B_` biosamples are the eval pool and see eval-split
windows.  The `applies_*` columns carry that choice, and the per-regime columns carry all
four so the choice is checkable.

Read-only on the census output.

    python ingestion_emit.py <out_dir> <mirror_dir>
"""
import argparse
import glob
import json
import os

REGIMES = ("eic.19", "eic.pilot")
SPLITS = ("train", "eval")
PREFIX = {("eic.19", "train"): "e19_train", ("eic.19", "eval"): "e19_eval",
          ("eic.pilot", "train"): "pilot_train", ("eic.pilot", "eval"): "pilot_eval"}
FIELDS = [("n_eligible_windows", "n_elig"),
          ("n_windows_over_ceiling", "n_over"),
          ("frac_windows_over_ceiling", "frac_over"),
          ("n_windows_over_10x", "n_over10x"),
          ("n_windows_over_100x", "n_over100x"),
          ("max_ingested_count", "max"),
          ("max_ingested_locus", "max_locus"),
          ("max_over_ceiling_ratio", "max_x_ceil"),
          ("max_over_median_nonzero", "max_x_mednz"),
          ("n_windows_with_track_top1", "has_top1")]

BASE = ["biosample", "assay", "assay_class", "split", "run_type", "read_length", "depth",
        "total_counts", "keepdup1_ceiling", "median_nonzero_bin", "global_max_count",
        "global_max_locus", "file_accession", "exp_accession"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("indir")
    ap.add_argument("mirrordir")
    a = ap.parse_args()

    rows = []
    for p in sorted(glob.glob(f"{a.indir}/*.json")):
        d = json.load(open(p))
        for name, r in d["tracks"].items():
            row = {k: r.get(k) for k in BASE}
            row["biosample"] = d["biosample"]
            row["split"] = d["split"]
            row["assay"] = name
            applies = "train" if d["split"] == "T" else "eval"
            row["applies_split"] = applies
            for reg in REGIMES:
                for sp in SPLITS:
                    s = r["regimes"][reg][sp]
                    for src, short in FIELDS:
                        row[f"{PREFIX[(reg, sp)]}_{short}"] = s[src]
            for reg, tag in (("eic.19", "e19"), ("eic.pilot", "pilot")):
                s = r["regimes"][reg][applies]
                for src, short in FIELDS:
                    row[f"applies_{tag}_{short}"] = s[src]
            row["_top_windows"] = {f"{reg}/{sp}": r["regimes"][reg][sp]["top_windows"]
                                   for reg in REGIMES for sp in SPLITS}
            rows.append(row)

    cols = [c for c in rows[0] if c != "_top_windows"]
    os.makedirs(a.mirrordir, exist_ok=True)
    with open(f"{a.mirrordir}/per_track.tsv", "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r[c]) for c in cols) + "\n")
    with open(f"{a.mirrordir}/per_track.json", "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"{len(rows)} tracks -> {a.mirrordir}/per_track.{{tsv,json}}")


if __name__ == "__main__":
    main()
