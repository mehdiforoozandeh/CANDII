#!/usr/bin/env python3
"""Bin one chromosome of the 312 train+validation EIC experiments into a single
(n_bins, n_tracks) float32 matrix, for one of the two processings.  Runs on Fir.

Two processings, same 312 experiments, same order of rows:

  d3   the challenge's own -log10 p-value bigWigs (Synapse syn17083203), binned
       by 001's `fir_tracks.load_official_bigwig`, which follows the challenge's
       own bw_to_npy.py exactly.  This is the grid the published numbers live on:
       n_bins = ceil(chrom_len / 25).
  d2   ENCODE's present-day signal for the same experiment accessions, as binned
       by Mehdi's pipeline into DATA_CANDI_EIC/.../signal_BW_res25/<chrom>.npz.
       That grid is floor(chrom_len / 25), i.e. one bin shorter; bin i means the
       same interval in both, so we simply zero-pad the tail to the d3 length and
       every downstream stage can ignore the difference.

Stored transposed -- (position, track) rather than (track, position) -- because
training samples batches of *positions* and wants all 312 tracks at those
positions to be contiguous.

Values are stored raw (not arcsinh-transformed) so that later stages can choose
their own transform, and as float32 rather than float16 because **d3** carries
extreme values that float16 (max 65504) could not hold: the challenge's DNase
-log10 p-value tracks reach 199,666 on chr1, against 5,270 for the same
experiments in d2.  That direction is worth stating explicitly because it is the
opposite of the intuition that "Mehdi's tree is the odd one" -- for DNase-seq d2
holds `read-depth normalized signal`, a different and much smaller quantity than
the p-value d3 holds (001 Result 3).  Measured after binning:

    pooled mean over all 312 train+validation tracks, d2/d3 = 0.941
    (per chromosome, range 0.852 - 0.995)

which is consistent with 001's finding that the two processings are close to
unbiased with respect to each other on the training split.
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

from hpc_vendor import FT     # 001's loaders, Dataset 2 root redirected


def train_val_rows(bridge_path):
    """The 312 train+validation experiments, in a fixed, sorted order."""
    rows = [r for r in csv.DictReader(open(bridge_path)) if r["split"] in ("T", "V")]
    return sorted(rows, key=lambda r: r["filename"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["d2", "d3"])
    ap.add_argument("--chrom", required=True)
    ap.add_argument("--bridge", required=True)
    ap.add_argument("--d3-track-dir", required=True,
                    help="directory of challenge bigWigs, C##M##.bigwig")
    ap.add_argument("--chrom-sizes", required=True,
                    help="json of chrom -> official n_bins")
    ap.add_argument("--out", required=True, help="output root; writes <out>/<dataset>/<chrom>.npy")
    args = ap.parse_args()

    rows = train_val_rows(args.bridge)
    n_bins = json.load(open(args.chrom_sizes))[args.chrom]

    # Check every input is present before allocating a 12 GB array and reading a
    # whole chromosome.  Without this, a wrong --d3-track-dir (which is exactly
    # what happened when this experiment moved from Fir to Nibi: the job file
    # still pointed at 001's Fir path) fails one track at a time, deep inside
    # pyBigWig, after the job has already been queued and started.
    missing = []
    for r in rows:
        if args.dataset == "d3":
            p = os.path.join(args.d3_track_dir, f"{r['filename']}.bigwig")
        else:
            p = os.path.join(FT.EIC, r["biosample_dir"], r["assay_name"],
                             "signal_BW_res25", f"{args.chrom}.npz")
        if not os.path.exists(p):
            missing.append(p)
    if missing:
        sys.exit(f"[bin {args.dataset}/{args.chrom}] {len(missing)} of {len(rows)} "
                 f"inputs are missing, e.g.:\n  " + "\n  ".join(missing[:3]))

    odir = os.path.join(args.out, args.dataset)
    os.makedirs(odir, exist_ok=True)
    opath = os.path.join(odir, f"{args.chrom}.npy")
    if os.path.exists(opath):
        print(f"[bin {args.dataset}/{args.chrom}] exists, skipping", flush=True)
        return 0

    print(f"[bin {args.dataset}/{args.chrom}] {len(rows)} tracks x {n_bins} bins",
          flush=True)
    t0 = time.time()
    Y = np.zeros((n_bins, len(rows)), dtype=np.float32)

    for j, r in enumerate(rows):
        if args.dataset == "d3":
            p = os.path.join(args.d3_track_dir, f"{r['filename']}.bigwig")
            y = FT.load_official_bigwig(p, [args.chrom])[args.chrom]
        else:
            y = FT.load_mehdi(r["biosample_dir"], r["assay_name"],
                              [args.chrom])[args.chrom]
        n = min(len(y), n_bins)
        Y[:n, j] = y[:n]        # d2 is one bin short; the tail stays zero
        if (j + 1) % 50 == 0:
            print(f"    {j+1}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)

    tmp = opath + ".tmp.npy"
    np.save(tmp, Y)
    os.replace(tmp, opath)

    with open(os.path.join(odir, "tracks.txt"), "w") as fh:
        fh.write("\n".join(r["filename"] for r in rows) + "\n")

    print(f"[bin {args.dataset}/{args.chrom}] wrote {opath} "
          f"({Y.nbytes/2**30:.1f} GiB, {time.time()-t0:.0f}s); "
          f"mean={Y.mean():.4f} max={Y.max():.1f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
