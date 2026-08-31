#!/usr/bin/env python3
"""The nine ENCODE Imputation Challenge performance measures.

A deliberately faithful re-implementation of `score_metrics.py` and the relevant
parts of `score.py` from ENCODE-DCC/imputation_challenge @181b8023.  Faithful
includes reproducing behaviour that is arguably wrong, because our goal is to
land on the *published* numbers, not on better ones:

  * `normalize_dict` in the official code is a no-op (its body is commented
    out), so every MSE is on the raw -log10 p-value scale.  We do the same.
  * `mseprom` uses gene *bin* indices, so a gene starting within the first 80
    bins of a chromosome produces a negative slice start.  numpy then returns an
    empty slice, silently contributing nothing.  We skip those genes, which is
    what the official code effectively does.
  * The gene annotation is GENCODE v29 (what the code ships), not v38 (what the
    paper's text says).
  * Bins overlapping the ENCODE blacklist are *deleted* from the arrays rather
    than masked, which shifts every downstream bin index.  Annotation intervals
    are then applied to the shifted array.  We reproduce this, and expose
    `blacklist=False` so the size of the effect can be measured.

The ten "bootstraps" are not resamples of genomic positions (as the paper's text
says) but ten fixed subsets of chromosomes; the exact groups are in the repo's
README and are hard-coded below in the order the organizers used.
"""
import gzip
import os

import numpy as np
from scipy.stats import rankdata

CHROMS = sorted(["chr" + str(i) for i in range(1, 23)] + ["chrX"])

# The ten bootstrap chromosome groups, verbatim from the --bootstrap-chrom
# argument of the round-2 scoring command in the challenge repo's README.
BOOTSTRAP_CHROM = [
    "chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr2,chr20,chr21,chr3,chr4,chr5,chr6,chr7,chr8,chr9,chrX",
    "chr1,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr20,chr21,chr22,chr3,chr4,chr5,chr6,chr7,chr8,chr9,chrX",
    "chr1,chr10,chr11,chr12,chr13,chr15,chr16,chr17,chr18,chr19,chr2,chr20,chr21,chr22,chr4,chr5,chr6,chr7,chr8,chr9,chrX",
    "chr1,chr10,chr11,chr12,chr14,chr15,chr16,chr17,chr18,chr19,chr2,chr20,chr21,chr22,chr3,chr5,chr6,chr7,chr8,chr9,chrX",
    "chr1,chr10,chr11,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr2,chr20,chr21,chr22,chr3,chr4,chr6,chr7,chr8,chr9,chrX",
    "chr1,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr2,chr20,chr21,chr22,chr3,chr4,chr5,chr7,chr8,chr9,chrX",
    "chr1,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr2,chr20,chr21,chr22,chr3,chr4,chr5,chr6,chr9,chrX",
    "chr1,chr10,chr11,chr12,chr13,chr14,chr16,chr17,chr18,chr19,chr2,chr21,chr22,chr3,chr4,chr5,chr6,chr7,chr8,chrX",
    "chr1,chr10,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr2,chr20,chr21,chr22,chr3,chr4,chr5,chr6,chr7,chr8,chr9",
    "chr1,chr10,chr11,chr12,chr13,chr14,chr15,chr19,chr2,chr20,chr22,chr3,chr4,chr5,chr6,chr7,chr8,chr9,chrX",
]
BOOTSTRAP_CHROM = [c.split(",") for c in BOOTSTRAP_CHROM]

MEASURES = ["mse", "gwcorr", "gwspear", "mseprom", "msegene",
            "mseenh", "msevar", "mse1obs", "mse1imp"]

WINDOW = 25
PROM_LOC = 80


# ---------------------------------------------------------------------------
# annotations
# ---------------------------------------------------------------------------
def _open_bed(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh:
        for line in fh:
            if line.strip():
                yield line.split()


def load_annotations(repo_dir):
    """Return per-chromosome bin intervals for promoters, gene bodies, enhancers.

    Bin arithmetic copied from score_metrics.py: start//25, end//25 + 1.
    """
    genes = os.path.join(repo_dir, "annot/hg38/gencode.v29.genes.gtf.bed.gz")
    enhs = os.path.join(repo_dir, "annot/hg38/F5.hg38.enhancers.bed.gz")

    prom, body, enh = {}, {}, {}
    for f in _open_bed(genes):
        chrom, start, end, _, _, strand = f[:6]
        s = int(start) // WINDOW
        e = int(end) // WINDOW + 1
        body.setdefault(chrom, []).append((s, e))
        if strand == "+":
            # y[s-80 : s]; a negative start yields an empty slice in numpy, so
            # genes closer than 80 bins to the chromosome start contribute nothing
            if s - PROM_LOC >= 0:
                prom.setdefault(chrom, []).append((s - PROM_LOC, s))
        else:
            prom.setdefault(chrom, []).append((e, e + PROM_LOC))
    for f in _open_bed(enhs):
        chrom, start, end = f[0], f[1], f[2]
        enh.setdefault(chrom, []).append((int(start) // WINDOW,
                                          int(end) // WINDOW + 1))

    to_arr = lambda d: {c: np.array(v, dtype=np.int64) for c, v in d.items()}
    return to_arr(prom), to_arr(body), to_arr(enh)


def load_blacklist_bins(repo_dir):
    """chrom -> sorted array of bin ids overlapping the ENCODE blacklist."""
    path = os.path.join(repo_dir, "annot/hg38/hg38.blacklist.bed.gz")
    bins = {}
    for f in _open_bed(path):
        chrom, start, end = f[0], int(f[1]), int(f[2])
        s = start // WINDOW
        e = end // WINDOW + 1
        bins.setdefault(chrom, set()).update(range(s, e))
    return {c: np.array(sorted(v), dtype=np.int64) for c, v in bins.items()}


def apply_blacklist(y_dict, blacklist):
    """Delete blacklisted bins, exactly as bw_to_npy.py does (indices shift)."""
    out = {}
    for c, y in y_dict.items():
        ids = blacklist.get(c)
        if ids is None or len(ids) == 0:
            out[c] = y
        else:
            out[c] = np.delete(y, ids[ids < len(y)])
    return out


# ---------------------------------------------------------------------------
# the nine measures
# ---------------------------------------------------------------------------
def _cat(d, chroms):
    return np.concatenate([d[c] for c in chroms])


def _interval_sse(y_true_d, y_pred_d, chroms, intervals):
    sse, n = 0.0, 0.0
    for c in chroms:
        iv = intervals.get(c)
        if iv is None:
            continue
        yt, yp = y_true_d[c], y_pred_d[c]
        L = len(yt)
        s = np.clip(iv[:, 0], 0, L)
        e = np.clip(iv[:, 1], 0, L)
        for a, b in zip(s, e):
            if b <= a:
                continue
            d = yt[a:b] - yp[a:b]
            sse += float(np.dot(d, d))
            n += b - a
    return sse / n if n else float("nan")


def _spearman(a, b):
    """Spearman rho without scipy.stats.spearmanr's extra copies."""
    ra = rankdata(a, method="average")
    rb = rankdata(b, method="average")
    return float(np.corrcoef(ra, rb)[0, 1])


def score(y_true_d, y_pred_d, chroms, prom, body, enh, var_d=None):
    """All nine measures for one (prediction, truth) pair on one chromosome set."""
    yt = _cat(y_true_d, chroms).astype(np.float64)
    yp = _cat(y_pred_d, chroms).astype(np.float64)

    out = {}
    diff = yt - yp
    out["mse"] = float(np.mean(diff * diff))
    out["gwcorr"] = float(np.corrcoef(yt, yp)[0, 1])
    out["gwspear"] = _spearman(yt, yp)
    out["mseprom"] = _interval_sse(y_true_d, y_pred_d, chroms, prom)
    out["msegene"] = _interval_sse(y_true_d, y_pred_d, chroms, body)
    out["mseenh"] = _interval_sse(y_true_d, y_pred_d, chroms, enh)

    if var_d is None:
        out["msevar"] = 0.0
    else:
        v = _cat(var_d, chroms).astype(np.float64)
        out["msevar"] = float(np.dot(diff * diff, v) / v.sum())

    # top 1% of positions by observed, then by predicted signal
    n1 = int(yt.shape[0] * 0.01)
    thr = np.sort(yt)[-n1]
    idx = yt >= thr
    d = yt[idx] - yp[idx]
    out["mse1obs"] = float(np.mean(d * d))

    thr = np.sort(yp)[-n1]
    idx = yp >= thr
    d = yt[idx] - yp[idx]
    out["mse1imp"] = float(np.mean(d * d))

    return out
