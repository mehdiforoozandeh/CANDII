#!/usr/bin/env python3
"""Fit a per-assay affine calibration from Dataset 2's scale onto Dataset 3's,
on arcsinh-transformed signal.  Runs on Fir.

  arcsinh(y_D3)  ~=  a * arcsinh(y_D2) + b

fitted by ordinary least squares on *paired* bins -- the same experiment, the
same 25 bp bin, in both processings -- over training experiments only.  Nothing
about the blind test enters, on either side.

Why affine on arcsinh rather than 001's per-assay quantile map: the quantile map
was fitted on the marginal distribution of individual experiments and then
applied to an *average* over experiments, whose distribution is much narrower.
For seven of the eight assays that mismatch was harmless, but for DNase-seq --
where Dataset 2 holds a different quantity entirely -- the map is violently
non-linear and it turned a 2.7x error into an 18x one (001 Result 8).  A
two-parameter map fitted by least squares on paired values cannot misbehave that
way, which matters here because an Avocado prediction is sharper than the
average-activity baseline the quantile map was validated on.

The fitted map is written in the same `q_src`/`q_dst` knot format that 001's
`fir_score.py --calibrate` already consumes, so the scoring code runs unchanged:
q_src is a fine arcsinh-spaced grid of Dataset 2 values and q_dst is
sinh(a*arcsinh(q_src) + b), clipped at zero.
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

from hpc_vendor import FT     # 001's loaders, Dataset 2 root redirected

# Same fitting subset as 001's calibration, so the two are comparable.
FIT_CHROMS = ["chr1", "chr21"]
THIN = 7
KNOT_MAX = 1e6
N_KNOTS = 20000


def knots(a, b):
    """Represent y -> sinh(a*arcsinh(y) + b) as a piecewise-linear knot table."""
    q_src = np.sinh(np.linspace(0.0, np.arcsinh(KNOT_MAX), N_KNOTS))
    q_dst = np.sinh(a * np.arcsinh(q_src) + b)
    q_dst = np.maximum(q_dst, 0.0)
    return np.maximum.accumulate(q_src), np.maximum.accumulate(q_dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge", required=True)
    ap.add_argument("--d3-track-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="*", default=["T"],
                    help="which splits to fit on -- keep this training-only")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.bridge)))
    blind_assays = {r["assay_id"] for r in rows if r["split"] == "B"}
    by_assay = {}
    for r in rows:
        if r["split"] in args.splits and r["assay_id"] in blind_assays:
            by_assay.setdefault(r["assay_id"], []).append(r)

    os.makedirs(args.out, exist_ok=True)
    summary = []
    print(f"[affine] {len(by_assay)} assays, fitting on splits {args.splits}",
          flush=True)

    for assay in sorted(by_assay):
        opath = os.path.join(args.out, f"{assay}.npz")
        if os.path.exists(opath):
            print(f"  {assay}: exists, skip", flush=True)
            continue
        t0 = time.time()
        src_l, dst_l, used = [], [], []
        for r in by_assay[assay]:
            p = os.path.join(args.d3_track_dir, f"{r['filename']}.bigwig")
            if not os.path.exists(p):
                continue
            try:
                dst = FT.load_official_bigwig(p, FIT_CHROMS)
                src = FT.load_mehdi(r["biosample_dir"], r["assay_name"], FIT_CHROMS)
            except Exception as e:                      # noqa: BLE001
                print(f"    {r['filename']}: skipped ({type(e).__name__}: {e})",
                      flush=True)
                continue
            src, dst = FT.align([src, dst])
            for c in FIT_CHROMS:
                src_l.append(src[c][::THIN])
                dst_l.append(dst[c][::THIN])
            used.append(r["filename"])

        if not used:
            print(f"  {assay}: no usable training experiments, skipping", flush=True)
            continue

        x = np.arcsinh(np.concatenate(src_l).astype(np.float64))
        y = np.arcsinh(np.concatenate(dst_l).astype(np.float64))
        A = np.stack([x, np.ones_like(x)], axis=1)
        (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - (a * x + b)
        r2 = 1.0 - float(resid.var() / y.var()) if y.var() > 0 else float("nan")
        pearson = float(np.corrcoef(x, y)[0, 1])

        q_src, q_dst = knots(float(a), float(b))
        np.savez_compressed(opath, q_src=q_src, q_dst=q_dst,
                            scalar=np.float64(np.exp(b) if a == 1 else 1.0),
                            affine_a=np.float64(a), affine_b=np.float64(b),
                            r2=np.float64(r2), pearson_arcsinh=np.float64(pearson),
                            n_experiments=np.int32(len(used)),
                            n_bins=np.int64(len(x)))
        with open(os.path.join(args.out, f"{assay}_sources.json"), "w") as fh:
            json.dump({"assay": assay, "splits": args.splits, "n": len(used),
                       "used": sorted(used), "fit_chroms": FIT_CHROMS,
                       "thin": THIN, "a": float(a), "b": float(b),
                       "r2": r2, "pearson_arcsinh": pearson}, fh, indent=1)
        summary.append({"assay": assay, "n_experiments": len(used),
                        "a": float(a), "b": float(b), "r2": r2,
                        "pearson_arcsinh": pearson,
                        "src_mean": float(np.sinh(x).mean()),
                        "dst_mean": float(np.sinh(y).mean())})
        print(f"  {assay}: n={len(used)} bins={len(x)} a={a:.4f} b={b:+.4f} "
              f"R2={r2:.4f} r={pearson:.4f} ({time.time()-t0:.0f}s)", flush=True)

    if summary:
        with open(os.path.join(args.out, "affine_summary.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summary[0]))
            w.writeheader()
            w.writerows(summary)
    print("[affine] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
