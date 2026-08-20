"""P-block — the four measures the challenge's own retrospective recommends *instead* of the nine.

The structural insight behind all of them, from
`cruxvault/wiki/imputation-evaluation-measures.md`: a measure computed over the whole genome at once
is dominated by background, and background is both the easiest part to predict and the part where
prediction matters least. The paper's evidence is blunt — most models showed **four orders of
magnitude higher MSE on H3K4me3 than on H3K9me3**, purely because the marks differ in dynamic range
and punctateness, so aggregating unstratified is close to meaningless.

Every measure here therefore **partitions the genome, scores inside each partition, and averages over
partitions** — so each partition weighs equally rather than each locus. That reweighting is the whole
mechanism; drop it and these collapse back into the measures they were meant to replace.

Binarisation follows the paper: `Yᵇ = Yᶜ ≥ 2` (a signal p-value of 0.01) for imputations, and MACS2
peak membership for experimental data. The two sides are binarised by *different rules on purpose* —
the truth has real peak calls and the prediction does not.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from candi.bench.annotations import PROM_LOC, WINDOW_SIZE
from candi.metrics import pearson

__all__ = [
    "BINARISE_THRESHOLD", "strength_bin_edges", "accuracy_by_strength",
    "specificity_scores", "precision_recall_by_specificity",
    "region_correlation", "promoter_windows", "peak_regions", "partition_suite",
]

# -log10 p >= 2, i.e. p <= 0.01. The paper's threshold for turning an imputed signal into a call.
BINARISE_THRESHOLD = 2.0


def strength_bin_edges(lo_exp: float = -1.0, hi_exp: float = 2.5,
                       width: float = 0.1) -> np.ndarray:
    """Logarithmic bin edges of size `width` in log10, from `10**lo_exp` to `10**hi_exp`.

    The paper's grid: "logarithmic bins of size 0.1 from 10⁻¹ to 10^2.5", which is 35 bins. Positions
    outside the range fall into an underflow and an overflow bin, reported separately rather than
    silently clamped — most of the genome is below 10⁻¹, and folding it into the first real bin would
    reintroduce exactly the background domination the block exists to defeat.
    """
    n = int(round((hi_exp - lo_exp) / width))
    return 10.0 ** (lo_exp + width * np.arange(n + 1))


def accuracy_by_strength(y_signal: np.ndarray, y_binary_true: np.ndarray,
                         y_pred_signal: np.ndarray, *, bin_by: str = "obs",
                         threshold: float = BINARISE_THRESHOLD,
                         edges: Optional[np.ndarray] = None) -> Dict[str, object]:
    """Accuracy of the binarised prediction against binarised truth, per signal-strength bin.

    `bin_by="obs"` partitions on the experimental signal; `bin_by="imp"` partitions on the imputed
    signal. Running both is the point — a model accurate only where signal is high and a model
    accurate only where signal is absent look identical under either one alone.

    Returns per-bin accuracy, per-bin n, and `macro_accuracy`: the mean over **occupied bins**, which
    is the partition-weighted number the paper argues for. An empty bin contributes nothing rather
    than a 0 or a nan, since an unoccupied partition is not a failure to be accurate in it.
    """
    edges = strength_bin_edges() if edges is None else np.asarray(edges)
    basis = np.asarray(y_signal if bin_by == "obs" else y_pred_signal, dtype=np.float64)
    truth = np.asarray(y_binary_true).astype(bool)
    call = np.asarray(y_pred_signal, dtype=np.float64) >= threshold
    correct = call == truth

    # np.digitize: 0 = below the first edge (underflow), len(edges) = at or above the last (overflow).
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
        "accuracy": acc,                       # index 0 is underflow, index -1 is overflow
        "n": counts,
        "macro_accuracy": float(np.mean(occupied)) if occupied else float("nan"),
        "n_occupied_bins": len(occupied),
        "pooled_accuracy": float(correct.mean()) if correct.size else float("nan"),
    }


# ---------------------------------------------------------------------------
# cell-type specificity
# ---------------------------------------------------------------------------

def specificity_scores(binary_matrix: np.ndarray) -> np.ndarray:
    """Per-locus specificity: the number of cell types whose binarised signal is 1, for one assay.

    `binary_matrix` is `(n_cell_types, n_positions)`. The score is the column sum, so 0 means "peak in
    no cell type" and `n_cell_types` means "peak everywhere" — a constitutive site that the
    average-activity baseline gets for free. The interesting groups are the small non-zero ones.
    """
    return np.asarray(binary_matrix).astype(bool).sum(axis=0).astype(np.int64)


def precision_recall_by_specificity(spec: np.ndarray, y_binary_true: np.ndarray,
                                    y_pred_binary: np.ndarray) -> Dict[str, object]:
    """Precision and recall per group of loci sharing the same specificity score.

    This is the measure that separates a model from the [[average-activity-baseline]] directly: a
    model that has learned only average activity scores well at high specificity (sites peaked in
    every cell type) and collapses at low specificity (sites peaked in one or two), and no pooled
    number shows that.
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


# ---------------------------------------------------------------------------
# region-restricted correlation
# ---------------------------------------------------------------------------

def promoter_windows(gene_annotations: Sequence[str], chrom: str, n: int, *,
                     window_size: int = WINDOW_SIZE, half_bins: int = PROM_LOC
                     ) -> List[Tuple[int, int]]:
    """±2 kb windows centred on GENCODE gene starts, as bin ranges, clipped to the array.

    **Not the same region as `eic.mseprom`**, and the difference is easy to miss. `mseprom` takes
    `prom_loc` bins *upstream only*, strand-aware. The paper's promoter correlation takes ±2 kb
    *around* the start, i.e. `half_bins` on each side. Same annotation, same 2 kb, different windows;
    the numbers are not interchangeable and neither is a check on the other.
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

    A one-bin peak has no shape, and Pearson over a single point is undefined — including such runs
    would make the peak-shape measure a mean over a pile of nans.
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

    Averaging per region rather than pooling is what makes this a shape measure. Pooled across
    regions, a model that gets every promoter's *height* right and every promoter's *profile* wrong
    still scores highly, because between-region variance dominates. Within a region that variance is
    gone and only the profile is left.

    Regions where either side is constant give an undefined correlation; they are counted in
    `n_undefined` and excluded from the mean rather than being scored 0, which would be a claim that
    the model got the shape wrong when in fact there was no shape.
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


# ---------------------------------------------------------------------------
# the whole block
# ---------------------------------------------------------------------------

def partition_suite(y_signal: np.ndarray, y_pred_signal: np.ndarray, y_peaks: np.ndarray, *,
                    assay: str, chrom: str, gene_annotations: Optional[Sequence[str]] = None,
                    threshold: float = BINARISE_THRESHOLD) -> Dict[str, object]:
    """The per-track half of the P-block.

    `precision_recall_by_specificity` is **not** here: it needs the binarised matrix across every
    cell type for the assay, so it is a panel-level measure computed once in the harness, not per
    track. Putting it here would force each track to re-read the whole panel.

    The two region-correlation measures are assay-gated exactly as the paper defines them — promoter
    correlation is an H3K4me3 measure and peak-shape correlation is a DNase measure. Computing them
    for other assays is possible and meaningless, so they are simply absent from the result for an
    assay they do not apply to, rather than present and nan.
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
