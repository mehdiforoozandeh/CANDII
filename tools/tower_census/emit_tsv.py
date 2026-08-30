"""Flatten the per-biosample tower-census JSONs into the vault mirror.

Writes `per_track.tsv` (one row per track, the 29 columns of
`cruxvault/results/t78/tower_census/per_track.tsv`) and `per_track.json` (the same rows
plus each track's top 10 bins with coordinates, blacklist and mask).

`nb_*` is the largest bin NOT inside hg38 blacklist v2, taken over the top 50 bins the
census emits, and its ratio to that track's own keep-dup-1 ceiling.

Read-only on the census output.

    python emit_tsv.py <out_dir> <mirror_dir> [--what "one line describing the scope"]
"""
import argparse
import glob
import json
import os

COLS = ["biosample", "assay", "assay_class", "split", "run_type", "read_length", "L_eff_bp",
        "depth", "total_counts", "keepdup1_ceiling", "global_max_count",
        "max_over_ceiling_ratio", "n_bins_over_ceiling", "frac_bins_over_ceiling",
        "excess_mass_over_ceiling", "frac_mass_over_ceiling", "frac_mass_in_bins_over_ceiling",
        "frac_mass_top1", "frac_mass_top2", "frac_mass_top10", "nb_max", "nb_ratio", "nb_locus",
        "file_accession", "exp_accession", "top1_locus", "top1_count", "top1_in_blacklist",
        "top1_mask"]

CEILING_NOTE = ("keepdup1_ceiling = int(round(2*(L_eff+24))), L_eff = 25*total_counts/depth - 24; "
                "identical to results/t78/diag_keepdup/.")
CAVEAT = ("the ceiling is a SINGLE-END bound (one tag per start position per strand). For "
          "paired-ended tracks it is not a proof of duplication; the 7 ATAC tracks, whose BAMs "
          "ENCODE ships duplicate-free, still read 13.1x-54.2x.")
NB_NOTE = "nb_* = the largest bin NOT inside hg38 blacklist v2, and its ratio to the ceiling."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("indir")
    ap.add_argument("mirrordir")
    ap.add_argument("--what", default="tower census, counts layer, genome-wide.")
    a = ap.parse_args()

    rows = []
    for p in sorted(glob.glob(f"{a.indir}/*.json")):
        d = json.load(open(p))
        for name, r in d["tracks"].items():
            r = dict(r)
            r["biosample"] = d["biosample"]
            r["split"] = d["split"]
            r["n_bins_genome"] = d["n_bins_genome"]
            top = r["top_bins"]
            nb = next((b for b in top if not b["in_blacklist"]), None)
            r["nb_max"] = nb["count"] if nb else 0
            r["nb_ratio"] = nb["over_ceiling_ratio"] if nb else 0.0
            r["nb_locus"] = f"{nb['chrom']}:{nb['start_bp']}" if nb else "-"
            t1 = top[0] if top else None
            r["top1_locus"] = f"{t1['chrom']}:{t1['start_bp']}" if t1 else "-"
            r["top1_count"] = t1["count"] if t1 else 0
            r["top1_in_blacklist"] = t1["in_blacklist"] if t1 else False
            r["top1_mask"] = t1["mask"] if t1 else -1
            r["top_bins"] = top[:10]
            rows.append(r)

    rows.sort(key=lambda r: (r["biosample"], r["assay"]))
    os.makedirs(a.mirrordir, exist_ok=True)
    with open(f"{a.mirrordir}/per_track.tsv", "w") as fh:
        fh.write("\t".join(COLS) + "\n")
        for r in rows:
            fh.write("\t".join(str(r.get(c, "")) for c in COLS) + "\n")
    json.dump({"_what": a.what, "_ceiling": CEILING_NOTE, "_caveat": CAVEAT, "_nb": NB_NOTE,
               "tracks": rows},
              open(f"{a.mirrordir}/per_track.json", "w"), indent=1)
    print(f"{len(rows)} tracks -> {a.mirrordir}/per_track.{{tsv,json}}")


if __name__ == "__main__":
    main()
