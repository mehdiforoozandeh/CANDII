#!/usr/bin/env python3
"""Max's sprint 06 error decomposition, applied to the ENCODE Imputation Challenge blind test (t148).

One task = one entry x one assay. The entry's prediction for each blind experiment of the assay is read
once (Max's sprint 04 reader: `tracks_io.load`, with CANDI's `tcfloor` map applied exactly as in his
`score_multi.py`) and decomposed against each truth layer:

  pe  original 2019 truth                                   (the MOESM2 board)
  pq  original truth, quantile-normalized (Max's rebuild)   (the MOESM3 board)
  qn  single-end reprocessed + quantile-normalized (Zenodo) (the MOESM4 board)

Bins are the challenge's scoring grid: 25 bp, chr1-22 + chrX, blacklisted bins deleted. For the Haar
transform the deleted bins are put back as invalid positions of the full chromosome (`errdecomp.Layout`),
so wavelet scales stay genomic distances. Unit: asinh of the track value (Max's convention; for `pe` this
is asinh(-log10 p)). Per track: `errdecomp.calibrate` (offset, scale, nonlinear, residual), then the
standardized truth x~ and the calibrated prediction y~ are Haar-transformed, and their Gram matrices
across the assay's experiments (G_x, G_y, G_xy, divided by the number of valid bins) are saved so that
`aggregate.py` can split them into shared / specific parts (groups = cell type) and into explained,
unexplained and added variance (Max's sprint 06 report2 definitions; no sampling-noise term here).

Usage: decompose.py --joblist JOBLIST --slug SLUG --assay ASSAY --code S04_CODE --cache S04_CACHE
                    --avg-bw DIR --out OUT [--truths pe,pq,qn]
Writes OUT/<truth>/<assay>/<slug>.{tracks.tsv,npz,DONE}.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import errdecomp as E

K = E.J + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joblist", required=True, help="Max's score joblist: slug, model, experiment, tcfloor, label")
    ap.add_argument("--slug", required=True)
    ap.add_argument("--assay", required=True, help="assay name, e.g. H3K27ac")
    ap.add_argument("--code", required=True, help="Max's sprint 04 code dir on the cluster (vendor031/, inputs/)")
    ap.add_argument("--cache", required=True, help="sprint 04 cache: <cache>/<truth>/truth/<exp>/<chrom>.npy")
    ap.add_argument("--avg-bw", required=True, help="the Average baseline's bigWigs (for tcfloor)")
    ap.add_argument("--truths", default="pe,pq,qn")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sys.path.insert(0, os.path.join(a.code, "vendor031"))
    import tracks_io
    inputs = os.path.join(a.code, "inputs")
    sys.path.insert(0, os.path.join(inputs, "vendor_001"))
    import eic_metrics as EM

    truths = [t for t in a.truths.split(",") if t]
    todo = [t for t in truths if not os.path.exists(os.path.join(a.out, t, a.assay, f"{a.slug}.DONE"))]
    if not todo:
        print(f"[decomp] {a.slug}/{a.assay}: done", flush=True)
        return 0

    bridge = {r["filename"]: r for r in csv.DictReader(open(os.path.join(inputs, "bridge", "eic_bridge.csv")))}
    jobs = [r for r in csv.reader(open(a.joblist), delimiter="\t") if r and r[0] == a.slug]
    jobs = [r for r in jobs if bridge[r[2]]["assay_name"] == a.assay]
    jobs.sort(key=lambda r: r[2])
    if not jobs:
        sys.exit(f"[decomp] no job lines for {a.slug} / {a.assay}")
    label = jobs[0][4]
    chroms = list(EM.CHROMS)
    blacklist = EM.load_blacklist_bins(os.path.join(inputs, "imputation_challenge"))

    # ---- predictions, read once (the slow step), on the challenge grid with blacklisted bins deleted
    T0 = time.time()
    exps, preds, raw_len = [], [], {c: [] for c in chroms}
    for _, model, exp, tcf, _ in jobs:
        if model.startswith("bigwig:") and not os.path.exists(model[len("bigwig:"):]):
            print(f"[skip] {exp}: no submitted track", flush=True)
            continue
        P = tracks_io.load(model, exp, chroms)
        if tcf == "1":                                   # score_multi.py's tcfloor, verbatim
            V = tracks_io.load("bigwig:" + os.path.join(a.avg_bw, exp + ".bigwig"), exp, chroms)
            for c in chroms:
                raw_len[c].append(min(P[c].shape[0], V[c].shape[0]))
            P, V = tracks_io.align_and_filter([P, V], chroms, blacklist)
            W = {r["assay_name"]: r for r in csv.DictReader(
                open(os.path.join(inputs, "tcfloor_coefficients.csv")))}[a.assay]
            b0, b1, b2 = float(W["b0"]), float(W["b1_candi"]), float(W["b2_average"])
            P = {c: np.maximum(b0 + b1 * P[c].astype(np.float64) + b2 * V[c].astype(np.float64), 0.0)
                 .astype(np.float32) for c in chroms}
        else:
            for c in chroms:
                raw_len[c].append(P[c].shape[0])
            (P,) = tracks_io.align_and_filter([P], chroms, blacklist)
        exps.append(exp)
        preds.append(P)
        print(f"[load] {a.slug} {exp} {time.time() - T0:.0f}s", flush=True)
    if not exps:
        for t in todo:
            od = os.path.join(a.out, t, a.assay)
            os.makedirs(od, exist_ok=True)
            open(os.path.join(od, f"{a.slug}.DONE"), "w").write("no predictions\n")
        return 0
    cells = [bridge[e]["cell_id"] for e in exps]

    for t in todo:
        t1 = time.time()
        Z = []
        for e in exps:
            Z.append({c: np.load(os.path.join(a.cache, t, "truth", e, f"{c}.npy")) for c in chroms})
        # one layout for every track of the assay: the full grid of each chromosome, blacklisted bins invalid,
        # trimmed at the end to the shortest valid length any track has (score_multi trims the same way)
        masks, nkeep = {}, {}
        for c in chroms:
            L = min(raw_len[c])
            bl = blacklist.get(c)
            bl = bl[bl < L] if bl is not None else np.zeros(0, np.int64)
            m = np.ones(L, bool)
            m[bl] = False
            n = min([p[c].shape[0] for p in preds] + [z[c].shape[0] for z in Z] + [int(m.sum())])
            m[np.flatnonzero(m)[n:]] = False
            masks[c], nkeep[c] = m, n
        lay = E.Layout(masks)
        nv = lay.nvalid
        B = len(exps)
        X = np.zeros((B, lay.L), np.float32)
        Y = np.zeros((B, lay.L), np.float32)
        rows = []
        for j, e in enumerate(exps):
            x = np.concatenate([np.arcsinh(Z[j][c][:nkeep[c]].astype(np.float64)) for c in chroms])
            y = np.concatenate([np.arcsinh(preds[j][c][:nkeep[c]].astype(np.float64)) for c in chroms])
            tc, yc = E.calibrate(x, y)
            sx = np.sqrt(tc["var_x"])
            X[j, lay.idx] = ((x - tc["mean_x"]) / sx).astype(np.float32)
            Y[j, lay.idx] = ((yc - tc["mean_x"]) / sx).astype(np.float32)
            rows.append({"entry": label, "slug": a.slug, "truth": t, "assay": a.assay, "experiment": e,
                         "cell": cells[j], "nvalid": nv,
                         **{k: tc[k] for k in ("mse", "offset", "scale", "nonlinear", "residual", "r",
                                               "slope_ratio", "mono_used", "var_x", "mean_x", "ident_err")}})
            del x, y, yc
        del Z
        cx = E.haar(X)
        del X
        cy = E.haar(Y)
        del Y
        Gx = np.zeros((K, B, B)); Gy = Gx.copy(); Gxy = Gx.copy()
        for k in range(K):
            for s0 in range(0, cx[k].shape[1], 1 << 21):              # float64 accumulation in chunks
                A_ = cx[k][:, s0:s0 + (1 << 21)].astype(np.float64)
                B_ = cy[k][:, s0:s0 + (1 << 21)].astype(np.float64)
                Gx[k] += A_ @ A_.T; Gy[k] += B_ @ B_.T; Gxy[k] += A_ @ B_.T
        del cx, cy
        Gx /= nv; Gy /= nv; Gxy /= nv
        for j, r in enumerate(rows):
            for k in range(K):
                xx, yy, xy = Gx[k, j, j], Gy[k, j, j], Gxy[k, j, j]
                r[f"e_s{k}"] = xx - 2 * xy + yy
            # the Haar energies of the calibrated error add up to the residual term (identity check)
            r["e_sum_check"] = sum(r[f"e_s{k}"] for k in range(K)) - r["residual"]

        od = os.path.join(a.out, t, a.assay)
        os.makedirs(od, exist_ok=True)
        cols = list(rows[0].keys())
        tmp = os.path.join(od, f"{a.slug}.tracks.tsv.tmp")
        with open(tmp, "w") as f:
            f.write("\t".join(cols) + "\n")
            for q in rows:
                f.write("\t".join(f"{q[k]:.8g}" if isinstance(q[k], (float, np.floating)) else str(q[k])
                                  for k in cols) + "\n")
        os.replace(tmp, os.path.join(od, f"{a.slug}.tracks.tsv"))
        np.savez_compressed(os.path.join(od, f"{a.slug}.tmp.npz"), experiments=np.array(exps),
                            cells=np.array(cells), Gx=Gx, Gy=Gy, Gxy=Gxy, nvalid=nv)
        os.replace(os.path.join(od, f"{a.slug}.tmp.npz"), os.path.join(od, f"{a.slug}.npz"))
        open(os.path.join(od, f"{a.slug}.DONE"), "w").write("ok\n")
        print(f"[decomp] {a.slug} {a.assay} vs {t}: {B} tracks, {nv} bins, {time.time() - t1:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
