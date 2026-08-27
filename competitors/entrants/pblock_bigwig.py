#!/usr/bin/env python3
"""The bench P-block (partition metrics), ported onto the bigwig path.

This is a **port, not a wrapper**, and the reason is the environment rather than the science.
`candi.bench.partitions` imports through `candi`, which pulls in torch; the Dataset-3 scoring jobs
run in the light scorer env (numpy + scipy + pyBigWig) that experiment 001 and 005 both used, where
`import candi` is not available. So the definitions are transcribed here against numpy alone, and
`tests/test_pblock_port.py` holds this file and `candi.bench.partitions` to bit-identical agreement
on identical input. If the two ever disagree, the store-path module is right and this file is the
bug -- it is the copy.

Three things differ from the store path, all of them forced by what Dataset 3 actually distributes,
and all of them caveats that must travel with any number this file produces:

1. **No peak calls exist.** The store path binarises the truth with MACS2 peak membership and the
   prediction with `Y >= 2`, on purpose -- "the truth has real peak calls and the prediction does
   not". The challenge distributed signal tracks only; there is no peak call for a blind-test
   experiment. Both sides are therefore binarised at `>= 2` here, which makes
   `acc_by_*_strength` a self-consistency measure on a threshold rather than a comparison against
   called peaks. Every emitted dict carries `truth_binarisation: "signal>=2"` so no reader can
   mistake one for the other.

2. **The blacklist is not applied.** The official scorer *deletes* blacklisted bins, shifting every
   downstream index; the nine measures are computed that way because that is what produced the
   published numbers. The P-block is positional -- promoter windows are bin coordinates -- so on a
   deleted grid its windows would read displaced loci. `src/candi/bench/annotations.py` refuses the
   deletion on the store path for exactly this reason. The P-block is ours and has no published
   counterpart to match, so it takes the register over the convention.

3. **`peak_shape_corr_dnase` never fires.** DNase-seq is excluded from every Dataset-3 row (plan
   §2/P3, decision B3), and its window construction needs the peak calls point 1 says do not exist.
   The branch is kept so the port stays a transcription of the original, and is unreachable here.

`precision_recall_by_specificity` is panel-level, not per-track: it needs the binarised matrix over
every cell type for one assay. It is ported here and driven by `placement_table.py` once per assay,
the same way the harness computes it once per panel rather than once per track.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "BINARISE_THRESHOLD", "PROM_LOC", "WINDOW_SIZE", "pearson",
    "strength_bin_edges", "accuracy_by_strength", "specificity_scores",
    "precision_recall_by_specificity", "region_correlation", "promoter_windows",
    "peak_regions", "partition_suite", "partition_suite_multichrom",
]

# -log10 p >= 2, i.e. p <= 0.01. Mirrors candi.bench.partitions.BINARISE_THRESHOLD.
BINARISE_THRESHOLD = 2.0
PROM_LOC = 80        # bins each side of a gene start = 2 kb at 25 bp
WINDOW_SIZE = 25


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Transcribed from `candi.metrics.pearson`, including its guards.

    The guards are load-bearing for `region_correlation`: a constant window must come back nan so it
    lands in `n_undefined` rather than being scored, and `np.corrcoef` on a constant input returns
    nan with a RuntimeWarning instead of raising.
    """
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def strength_bin_edges(lo_exp: float = -1.0, hi_exp: float = 2.5,
                       width: float = 0.1) -> np.ndarray:
    """35 logarithmic edges from 10**lo_exp to 10**hi_exp, the paper's grid."""
    n = int(round((hi_exp - lo_exp) / width))
    return 10.0 ** (lo_exp + width * np.arange(n + 1))


def accuracy_by_strength(y_signal: np.ndarray, y_binary_true: np.ndarray,
                         y_pred_signal: np.ndarray, *, bin_by: str = "obs",
                         threshold: float = BINARISE_THRESHOLD,
                         edges: Optional[np.ndarray] = None) -> Dict[str, object]:
    """Accuracy of the binarised prediction against binarised truth, per strength bin.

    `macro_accuracy` is the mean over **occupied** bins -- an empty partition contributes nothing
    rather than a 0 or a nan, since an unoccupied partition is not a failure to be accurate in it.
    Index 0 is the underflow bin and index -1 the overflow bin; both are reported rather than
    clamped, because most of the genome sits below the first edge and folding it in would restore
    exactly the background domination the block exists to defeat.
    """
    edges = strength_bin_edges() if edges is None else np.asarray(edges)
    basis = np.asarray(y_signal if bin_by == "obs" else y_pred_signal, dtype=np.float64)
    truth = np.asarray(y_binary_true).astype(bool)
    call = np.asarray(y_pred_signal, dtype=np.float64) >= threshold
    correct = call == truth

    idx = np.digitize(basis, edges)
    n_bins = len(edges) + 1
    acc: List[float] = []
    counts: List[int] = []
    for b in range(n_bins):
        sel = idx == b
        c = int(sel.sum())
        counts.append(c)
        acc.append(float(correct[sel].mean()) if c else float("nan"))

    occupied = [a for a in acc if np.isfinite(a)]
    return {
        "edges": [float(e) for e in edges],
        "accuracy": acc,
        "n": counts,
        "macro_accuracy": float(np.mean(occupied)) if occupied else float("nan"),
        "n_occupied_bins": len(occupied),
        "pooled_accuracy": float(correct.mean()) if correct.size else float("nan"),
    }


def specificity_scores(binary_matrix: np.ndarray) -> np.ndarray:
    """Per-locus count of cell types whose binarised signal is 1, for one assay.

    `binary_matrix` is `(n_cell_types, n_positions)`.
    """
    return np.asarray(binary_matrix).astype(bool).sum(axis=0).astype(np.int64)


def precision_recall_by_specificity(spec: np.ndarray, y_binary_true: np.ndarray,
                                    y_pred_binary: np.ndarray) -> Dict[str, object]:
    """Precision and recall per group of loci sharing a specificity score.

    This is the measure that separates a model from the average-activity baseline directly: a method
    that has learned only average activity scores well at high specificity and collapses at low
    specificity, and no pooled number shows that.
    """
    spec = np.asarray(spec)
    truth = np.asarray(y_binary_true).astype(bool)
    call = np.asarray(y_pred_binary).astype(bool)
    groups = np.unique(spec)

    out: Dict[str, object] = {"specificity": [], "precision": [], "recall": [], "n": [],
                              "n_positive": []}
    for g in groups:
        sel = spec == g
        t, c = truth[sel], call[sel]
        tp = int((t & c).sum())
        fp = int((~t & c).sum())
        fn = int((t & ~c).sum())
        out["specificity"].append(int(g))
        out["precision"].append(tp / (tp + fp) if (tp + fp) else float("nan"))
        out["recall"].append(tp / (tp + fn) if (tp + fn) else float("nan"))
        out["n"].append(int(sel.sum()))
        out["n_positive"].append(int(t.sum()))

    for key in ("precision", "recall"):
        vals = [v for v in out[key] if np.isfinite(v)]
        out[f"macro_{key}"] = float(np.mean(vals)) if vals else float("nan")
    return out


def promoter_windows(gene_annotations: Sequence[str], chrom: str, n: int, *,
                     window_size: int = WINDOW_SIZE, half_bins: int = PROM_LOC
                     ) -> List[Tuple[int, int]]:
    """+/-2 kb windows centred on GENCODE gene starts, as bin ranges, clipped to the array.

    **Not the same region as `eic_metrics.mseprom`**: that takes `PROM_LOC` bins upstream only,
    strand-aware. This takes `half_bins` on each side. Same annotation, same 2 kb, different
    windows; neither number is a check on the other.
    """
    out: List[Tuple[int, int]] = []
    for line in gene_annotations:
        chrom_, start, end, _, _, strand = line.split()
        if chrom_ != chrom:
            continue
        anchor = (int(start) if strand == "+" else int(end)) // window_size
        lo, hi = max(anchor - half_bins, 0), min(anchor + half_bins, n)
        if hi - lo > 1:
            out.append((lo, hi))
    return out


def peak_regions(peak_binary: np.ndarray, min_bins: int = 2) -> List[Tuple[int, int]]:
    """Contiguous runs of called peak, as bin ranges. Runs shorter than `min_bins` are dropped.

    A one-bin peak has no shape and Pearson over a single point is undefined, so including such runs
    would make the shape measure a mean over a pile of nans.
    """
    b = np.asarray(peak_binary).astype(bool)
    if not b.any():
        return []
    d = np.diff(np.r_[0, b.view(np.int8), 0])
    starts = np.flatnonzero(d == 1)
    stops = np.flatnonzero(d == -1)
    return [(int(s), int(e)) for s, e in zip(starts, stops) if e - s >= min_bins]


def region_correlation(y_true: np.ndarray, y_pred: np.ndarray,
                       regions: Sequence[Tuple[int, int]]) -> Dict[str, float]:
    """Mean Pearson **within** each region, averaged over regions.

    Averaging per region rather than pooling is what makes this a shape measure: pooled, a method
    that gets every promoter's height right and every promoter's profile wrong still scores highly,
    because between-region variance dominates. Regions where either side is constant are counted in
    `n_undefined` and excluded, not scored 0 -- a 0 would claim the shape was got wrong when there
    was no shape.
    """
    vals: List[float] = []
    undefined = 0
    for lo, hi in regions:
        v = pearson(np.asarray(y_true[lo:hi], dtype=np.float64),
                    np.asarray(y_pred[lo:hi], dtype=np.float64))
        if np.isfinite(v):
            vals.append(v)
        else:
            undefined += 1
    return {
        "mean_corr": float(np.mean(vals)) if vals else float("nan"),
        "median_corr": float(np.median(vals)) if vals else float("nan"),
        "n_regions": len(regions),
        "n_scored": len(vals),
        "n_undefined": undefined,
    }


def partition_suite(y_signal: np.ndarray, y_pred_signal: np.ndarray, y_peaks: np.ndarray, *,
                    assay: str, chrom: str, gene_annotations: Optional[Sequence[str]] = None,
                    threshold: float = BINARISE_THRESHOLD) -> Dict[str, object]:
    """Single-chromosome per-track P-block. Transcribed from `partitions.partition_suite`.

    The two region correlations are assay-gated exactly as the paper defines them: promoter
    correlation is an H3K4me3 measure, peak-shape correlation a DNase measure. For an assay they do
    not apply to they are simply absent rather than present and nan.
    """
    out: Dict[str, object] = {
        "acc_by_obs_strength": accuracy_by_strength(y_signal, y_peaks, y_pred_signal,
                                                    bin_by="obs", threshold=threshold),
        "acc_by_imp_strength": accuracy_by_strength(y_signal, y_peaks, y_pred_signal,
                                                    bin_by="imp", threshold=threshold),
    }
    if assay == "H3K4me3" and gene_annotations is not None:
        windows = promoter_windows(gene_annotations, chrom, len(y_signal))
        out["prom_corr_h3k4me3"] = region_correlation(y_signal, y_pred_signal, windows)
    if assay in ("DNase-seq", "DNase"):
        out["peak_shape_corr_dnase"] = region_correlation(
            y_signal, y_pred_signal, peak_regions(y_peaks))
    return out


# ---------------------------------------------------------------------------
# the bigwig driver half
# ---------------------------------------------------------------------------

def partition_suite_multichrom(truth: Mapping[str, np.ndarray],
                               pred: Mapping[str, np.ndarray],
                               chroms: Sequence[str], *, assay: str,
                               gene_annotations: Optional[Sequence[str]] = None,
                               peaks: Optional[Mapping[str, np.ndarray]] = None,
                               threshold: float = BINARISE_THRESHOLD) -> Dict[str, object]:
    """The P-block over a concatenation of chromosomes -- the shape `harness._p_block` computes.

    `accuracy_by_strength` is position-free, so it sees the whole concatenation at once and its
    strength bins are the panel's rather than one chromosome's. `promoter_windows` IS positional, so
    windows are built per chromosome and shifted into the concatenation by that chromosome's offset.
    Scoring chr21 alone and scoring chr21 with chr22 are different numbers and the chromosome set is
    part of the number, exactly as it is for `mse1obs`.

    `peaks` defaults to `truth >= threshold`, which is the Dataset-3 substitute for MACS2 calls that
    the module docstring's point 1 describes. Pass real calls if a source of them ever exists.
    """
    sig = np.concatenate([np.asarray(truth[c], dtype=np.float64) for c in chroms])
    prd = np.concatenate([np.asarray(pred[c], dtype=np.float64) for c in chroms])
    if peaks is None:
        pk = sig >= threshold
        binarisation = f"signal>={threshold:g}"
    else:
        pk = np.concatenate([np.asarray(peaks[c]) for c in chroms]).astype(bool)
        binarisation = "peak_calls"

    out: Dict[str, object] = {
        "acc_by_obs_strength": accuracy_by_strength(sig, pk, prd, bin_by="obs",
                                                    threshold=threshold),
        "acc_by_imp_strength": accuracy_by_strength(sig, pk, prd, bin_by="imp",
                                                    threshold=threshold),
        "truth_binarisation": binarisation,
        "blacklist_deleted": False,
        "chroms": list(chroms),
    }

    offset, acc = {}, 0
    for c in chroms:
        offset[c] = acc
        acc += len(truth[c])

    if assay == "H3K4me3" and gene_annotations is not None:
        wins = [(lo + offset[c], hi + offset[c]) for c in chroms
                for lo, hi in promoter_windows(gene_annotations, c, len(truth[c]))]
        out["prom_corr_h3k4me3"] = region_correlation(sig, prd, wins)
    if assay in ("DNase-seq", "DNase") and peaks is not None:
        wins = [(lo + offset[c], hi + offset[c]) for c in chroms
                for lo, hi in peak_regions(np.asarray(peaks[c]).astype(bool))]
        out["peak_shape_corr_dnase"] = region_correlation(sig, prd, wins)
    return out


def load_gene_annotations(repo_dir: str) -> List[str]:
    """GENCODE v29 gene bed lines from the challenge repo's `annot/hg38/`.

    Byte-identical to `src/candi/bench/assets/gencode.v29.genes.gtf.bed.gz`
    (md5 3c2897b51371ecc2eeba4f4cb4db295e, checked 2026-08-25 against the 001 checkout on Fir), so
    the port and the store path read the same annotation and not merely the same annotation version.
    """
    import gzip
    import os
    path = os.path.join(repo_dir, "annot/hg38/gencode.v29.genes.gtf.bed.gz")
    with gzip.open(path, "rt") as fh:
        return [line for line in (l.strip() for l in fh) if line]
