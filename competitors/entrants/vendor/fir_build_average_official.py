#!/usr/bin/env python3
"""Build the average-activity baseline from the *challenge's own* training
tracks (downloaded from Synapse).  Runs on Fir.

Same definition as fir_build_average.py -- per assay, the mean across training
cell types at every 25 bp bin, plus the cross-cell-type variance that msevar
needs -- but reading the bigWigs the challenge actually distributed, binned the
way the challenge's own bw_to_npy.py bins them.

One job per chromosome.  Writes <out>/<variant>/<assay_id>/<chrom>.npz.
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eic_metrics as EM
import fir_tracks as FT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chrom", required=True)
    ap.add_argument("--bridge", required=True)
    ap.add_argument("--track-dir", required=True,
                    help="directory of official training bigWigs, C##M##.bigwig")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", default="train",
                    choices=["train", "trainval", "all"],
                    help="which splits to average over. 'all' includes the blind "
                         "test experiments themselves -- only meaningful for "
                         "diagnosing how the organizers built msevar's variance "
                         "vector, never as a prediction")
    args = ap.parse_args()

    splits = {"train": ["T"], "trainval": ["T", "V"],
              "all": ["T", "V", "B"]}[args.variant]
    rows = list(csv.DictReader(open(args.bridge)))
    needed = {r["assay_id"] for r in rows if r["split"] in ("B", "V")}
    by_assay = {}
    for r in rows:
        if r["split"] in splits and r["assay_id"] in needed:
            by_assay.setdefault(r["assay_id"], []).append(r)

    print(f"[official {args.chrom}/{args.variant}] {len(by_assay)} assays", flush=True)

    for assay_id in sorted(by_assay):
        odir = os.path.join(args.out, args.variant, assay_id)
        os.makedirs(odir, exist_ok=True)
        opath = os.path.join(odir, f"{args.chrom}.npz")
        if os.path.exists(opath):
            print(f"  {assay_id}: exists, skip", flush=True)
            continue

        t0 = time.time()
        s = ss = None
        used, missing = [], []
        for r in by_assay[assay_id]:
            p = os.path.join(args.track_dir, f"{r['filename']}.bigwig")
            if not os.path.exists(p):
                missing.append(r["filename"])
                continue
            y = FT.load_official_bigwig(p, [args.chrom])[args.chrom].astype(np.float64)
            if s is None:
                s, ss = np.zeros_like(y), np.zeros_like(y)
            elif len(y) != len(s):
                raise SystemExit(f"bin mismatch on {args.chrom} for {r['filename']}")
            s += y
            ss += y * y
            used.append(r["filename"])

        if not used:
            print(f"  {assay_id}: no source tracks present, skipping", flush=True)
            continue
        if missing:
            print(f"  {assay_id}: WARNING {len(missing)} source tracks missing "
                  f"-- the average will not match the published baseline "
                  f"({missing[:5]}...)", flush=True)

        n = len(used)
        mean = s / n
        var = np.maximum(ss / n - mean * mean, 0.0)
        np.savez_compressed(opath, mean=mean.astype(np.float32),
                            var=var.astype(np.float32), n=np.int32(n))
        with open(os.path.join(odir, "sources.json"), "w") as fh:
            json.dump({"n": n, "used": sorted(used), "missing": sorted(missing)},
                      fh, indent=1)
        print(f"  {assay_id}: n={n} bins={len(mean)} ({time.time()-t0:.0f}s)", flush=True)

    print(f"[official {args.chrom}/{args.variant}] done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
