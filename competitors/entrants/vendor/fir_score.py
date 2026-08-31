#!/usr/bin/env python3
"""Score one prediction track against one truth track with the nine challenge
measures, on each of the ten bootstrap chromosome groups.  Runs on Fir.

Truth and prediction can each come from either data source (see fir_tracks.py),
so the same code scores:

  * the average-activity baseline we built from Mehdi's tree, against Mehdi's
    blind-test tracks                     (--truth mehdi   --pred average)
  * the average-activity baseline we built from the challenge's own training
    tracks, against the challenge's blind-test tracks
                                          (--truth official --pred average)
  * the organizers' own baseline predictions and the competitors' submitted
    bigWigs, against the challenge's blind-test tracks
                                          (--truth official --pred bigwig)

Emits one CSV row per bootstrap.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eic_metrics as EM
import fir_tracks as FT


def apply_calibration(pred, calib_dir, assay_id, mode):
    """Map a prediction from one processing's scale onto the other's.

    The map was fitted on training experiments only (fir_build_calibration.py),
    so applying it to a blind-test prediction leaks nothing.
    """
    path = os.path.join(calib_dir, f"{assay_id}.npz")
    z = np.load(path)
    if mode == "scalar":
        k = float(z["scalar"])
        print(f"  calibration: scalar x{k:.4f}", flush=True)
        return {c: (v.astype(np.float32) * k) for c, v in pred.items()}

    q_src, q_dst = z["q_src"], z["q_dst"]
    print(f"  calibration: quantile map, {len(q_src)} knots, "
          f"src[{q_src.min():.3g},{q_src.max():.3g}] -> "
          f"dst[{q_dst.min():.3g},{q_dst.max():.3g}]", flush=True)
    out = {}
    # np.interp clamps outside the fitted range; above the top knot we keep the
    # slope of the last segment so that very high predicted bins are not
    # flattened onto a ceiling, which would distort mse1imp badly.
    hi_slope = ((q_dst[-1] - q_dst[-2]) / (q_src[-1] - q_src[-2])
                if q_src[-1] > q_src[-2] else 1.0)
    for c, v in pred.items():
        y = np.interp(v, q_src, q_dst).astype(np.float32)
        above = v > q_src[-1]
        if above.any():
            y[above] = q_dst[-1] + (v[above] - q_src[-1]) * hi_slope
        out[c] = y
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, help="C##M##")
    ap.add_argument("--bridge", required=True)
    ap.add_argument("--repo", required=True, help="imputation_challenge checkout")
    ap.add_argument("--truth", default="mehdi", choices=["mehdi", "official"])
    ap.add_argument("--truth-dir", help="directory of official truth bigWigs")
    ap.add_argument("--pred", default="average", choices=["average", "bigwig"])
    ap.add_argument("--pred-dir", help="directory of prediction bigWigs (C##M##.bigwig)")
    ap.add_argument("--avg-dir", help="root of a built average baseline; also "
                                      "supplies msevar's variance vector")
    ap.add_argument("--pred-avg-dir",
                    help="take the *prediction* from a different built average "
                         "than the one supplying msevar's variance. Used for the "
                         "cross-processing cells: predict from DataDataset 2 (ENCODE's processing, DATA_CANDI_EIC), score "
                         "against Dataset 3 (the challenge's own processing), with the variance matching the truth")
    ap.add_argument("--variant", default="train")
    ap.add_argument("--calibrate", help="directory of per-assay calibration .npz")
    ap.add_argument("--calibrate-mode", default="quantile",
                    choices=["quantile", "scalar"])
    ap.add_argument("--var-source", default="auto",
                    choices=["auto", "none"],
                    help="'auto' takes the cross-cell-type variance from --avg-dir")
    ap.add_argument("--label", required=True)
    ap.add_argument("--no-blacklist", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = {r["filename"]: r for r in csv.DictReader(open(args.bridge))}
    row = rows[args.experiment]
    assay_id = row["assay_id"]

    t0 = time.time()
    prom, body, enh = EM.load_annotations(args.repo)
    blacklist = EM.load_blacklist_bins(args.repo)

    # ---- truth ------------------------------------------------------------
    if args.truth == "mehdi":
        truth = FT.load_mehdi(row["biosample_dir"], row["assay_name"], EM.CHROMS)
    else:
        truth = FT.load_official_bigwig(
            os.path.join(args.truth_dir, f"{args.experiment}.bigwig"), EM.CHROMS)

    # ---- prediction -------------------------------------------------------
    var = None
    if args.pred == "average":
        pred_root = os.path.join(args.pred_avg_dir or args.avg_dir,
                                 args.variant, assay_id)
        pred = FT.load_npz_dir(pred_root, EM.CHROMS, "mean")
        if args.var_source == "auto":
            # the variance must match the *truth*'s processing, so it always
            # comes from --avg-dir even when the prediction does not
            var = FT.load_npz_dir(os.path.join(args.avg_dir, args.variant, assay_id),
                                  EM.CHROMS, "var")
    else:
        pred = FT.load_official_bigwig(
            os.path.join(args.pred_dir, f"{args.experiment}.bigwig"), EM.CHROMS)
        if args.var_source == "auto" and args.avg_dir:
            var = FT.load_npz_dir(
                os.path.join(args.avg_dir, args.variant, assay_id), EM.CHROMS, "var")

    if args.calibrate:
        pred = apply_calibration(pred, args.calibrate, assay_id, args.calibrate_mode)

    # grids from the two sources differ by one bin per chromosome
    parts = [truth, pred] + ([var] if var is not None else [])
    parts = FT.align(parts)
    truth, pred = parts[0], parts[1]
    var = parts[2] if var is not None else None

    if not args.no_blacklist:
        truth = EM.apply_blacklist(truth, blacklist)
        pred = EM.apply_blacklist(pred, blacklist)
        if var is not None:
            var = EM.apply_blacklist(var, blacklist)

    nbins = sum(len(v) for v in truth.values())
    print(f"[{args.experiment}] {args.label}: loaded in {time.time()-t0:.0f}s; "
          f"{nbins} bins; truth={args.truth} pred={args.pred} "
          f"var={'yes' if var is not None else 'no'}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "experiment", "cell", "assay", "assay_name",
                    "bootstrap_id", "truth_source", "blacklist_filtered"]
                   + EM.MEASURES)
        for b, chroms in enumerate(EM.BOOTSTRAP_CHROM):
            t1 = time.time()
            s = EM.score(truth, pred, chroms, prom, body, enh, var)
            w.writerow([args.label, args.experiment, row["cell_id"], assay_id,
                        row["assay_name"], b, args.truth, int(not args.no_blacklist)]
                       + [s[m] for m in EM.MEASURES])
            fh.flush()
            print(f"  bootstrap {b}: mse={s['mse']:.4f} gwcorr={s['gwcorr']:.4f} "
                  f"({time.time()-t1:.0f}s)", flush=True)

    print(f"[{args.experiment}] done in {time.time()-t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
