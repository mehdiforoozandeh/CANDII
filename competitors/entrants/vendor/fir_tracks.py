#!/usr/bin/env python3
"""Loading 25 bp-binned signal from the two sources we have.  Runs on Fir.

`official`
    A bigWig from the challenge's own Synapse project (syn17083203): the truth
    tracks the leaderboard was scored against, the organizers' baseline
    predictions, and the competitors' submissions.  Binned exactly the way
    bw_to_npy.py does it, because these are the arrays the published numbers
    came from:
      * bins = ceil(chrom_len / 25), the tail padded with zeros
      * NaN -> 0 (pyBigWig returns NaN off the end of the data)
      * the last bin overwritten with bw.stats(..., exact=True), which is
        bw_to_npy.py's fix for the partial final window

`mehdi`
    /project/def-maxwl/mforooz/DATA_CANDI_EIC/.../signal_BW_res25/<chrom>.npz --
    ENCODE's *current* "signal p-value" bigWig for the same experiment
    accession, binned with a nanmean by his pipeline.  Note bins =
    floor(chrom_len / 25), one short of the official grid, and NaN is skipped
    rather than treated as zero.

These are NOT the same numbers.  See the notebook: for BE2C H3K27me3 they
correlate at r = 0.83 at 25 bp and differ by an order of magnitude in the tail.
"""
import os

import numpy as np

import eic_metrics as EM

EIC = "/project/def-maxwl/mforooz/DATA_CANDI_EIC"


def official_n_bins(bw, chroms):
    return {c: (bw.chroms()[c] - 1) // EM.WINDOW + 1 for c in chroms}


def load_official_bigwig(path, chroms):
    """Bin a challenge bigWig, following bw_to_npy.py."""
    import pyBigWig
    bw = pyBigWig.open(path)
    out = {}
    for c in chroms:
        clen = bw.chroms()[c]
        n = (clen - 1) // EM.WINDOW + 1
        raw = bw.values(c, 0, clen, numpy=True)
        buf = np.zeros(n * EM.WINDOW, dtype=np.float64)
        buf[:raw.shape[0]] = np.nan_to_num(raw)
        y = buf.reshape(-1, EM.WINDOW).mean(axis=1)

        # bw_to_npy.py's special treatment of the window containing the last
        # interval: recompute it with an exact stats() call.
        ivs = bw.intervals(c)
        if ivs:
            last = ivs[-1][1] // EM.WINDOW
            if 0 <= last < n:
                start = last * EM.WINDOW
                end = min((last + 1) * EM.WINDOW, clen)
                stat = bw.stats(c, start, end, exact=True)
                y[last] = 0.0 if stat[0] is None else stat[0]
        out[c] = y.astype(np.float32)
    bw.close()
    return out


def load_mehdi(biosample_dir, assay_name, chroms):
    return {c: np.load(os.path.join(
                EIC, biosample_dir, assay_name, "signal_BW_res25", f"{c}.npz")
            )["arr_0"].astype(np.float32)
            for c in chroms}


def load_npz_dir(root, chroms, key="mean"):
    return {c: np.load(os.path.join(root, f"{c}.npz"))[key] for c in chroms}


def align(dicts):
    """Truncate a set of per-chromosome dicts to their common bin count.

    Only needed when mixing the two sources, whose grids differ by one bin.
    """
    chroms = set(dicts[0])
    out = [dict() for _ in dicts]
    for c in sorted(chroms):
        n = min(len(d[c]) for d in dicts)
        for o, d in zip(out, dicts):
            o[c] = d[c][:n]
    return out
