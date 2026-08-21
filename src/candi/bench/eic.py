"""The nine ENCODE Imputation Challenge measures — E-block.

The reference is `ENCODE-DCC/imputation_challenge/score_metrics.py`, vendored byte-identical at
`tests/fixtures/encode_score_metrics_vendored.py` (git blob `5dcb343139a0…`, unmodified). This
module is written independently of it and then **proven equal to it** in
`tests/test_bench_reference.py`. A transcription would make that test tautological; the point of
layer 1 is that two implementations agree, not that one file equals itself.

Where the reference does something odd, this module does the same odd thing on purpose, names it,
and a test pins it (`EVAL_PLAN.md` D16, §5.1). The catalogue:

1. **Overlapping annotations double-count.** `sse` and `n` accumulate per annotation line, so a
   position covered by two genes enters twice — with weight 2 in the numerator and 2 in the
   denominator, so it is a re-weighting, not a bias.
2. **A promoter running off the front of the array is silently dropped.** `y[start-80:start]` with
   `start < 80` has a negative left index, which Python resolves from the *end* of the array; for
   any realistic length that yields an empty slice contributing 0 to both `sse` and `n`. We
   reproduce it exactly by resolving bounds through `slice(...).indices(n)` rather than by
   special-casing `start < 80`, because on a short array the two differ.
3. **`end` is `int(end) // 25 + 1`** — one bin past the true end.
4. **`mse1obs` with fewer than 100 positions takes the whole array.** `n = int(N * 0.01)` is 0, and
   `numpy.sort(y)[-0]` is `y_sorted[0]`, the *minimum*, so `y >= min` selects everything. This bites
   only on small inputs, which is exactly what a test fixture is.
5. **`msevar` returns a bare `0.0`** with no variance vector. We do not: `annotations.py` raises,
   because a 0.0 is indistinguishable from a perfect score in any table it reaches.
6. **No transform.** `normalize_dict` is a no-op stub in the reference, so the published numbers are
   on untransformed −log10 p-values. No arcsinh anywhere in this module.

Two things this module does *better* than the reference without changing any arithmetic. It parses
each annotation bed once instead of re-parsing 58,721 lines per chromosome per track, and it caches
the resolved intervals. Both are outside the float arithmetic, so layer 1 still holds to 0 ULP.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr

from candi.bench.annotations import PROM_LOC, WINDOW_SIZE

# Ascending (bigger is better) or descending, per measure. Lifted from the reference's
# RANK_METHOD_FOR_EACH_METRIC — `ranking.py` needs it to turn scores into ranks the way the
# challenge did, and it belongs beside the measures rather than beside the ranker.
RANK_DIRECTION: Dict[str, str] = {
    "mse": "DESCENDING", "gwcorr": "ASCENDING", "gwspear": "ASCENDING",
    "mseprom": "DESCENDING", "msegene": "DESCENDING", "mseenh": "DESCENDING",
    "msevar": "DESCENDING", "mse1obs": "DESCENDING", "mse1imp": "DESCENDING",
}

MEASURES: Tuple[str, ...] = tuple(RANK_DIRECTION)


# ---------------------------------------------------------------------------
# genome-wide
# ---------------------------------------------------------------------------

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(((y_true - y_pred) ** 2.0).mean())


def gwcorr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Genome-wide Pearson. `np.corrcoef` returns nan on a constant input; the reference does too."""
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def gwspear(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(spearmanr(y_true, y_pred)[0])


def msevar(y_true: np.ndarray, y_pred: np.ndarray, var: np.ndarray) -> float:
    """MSE weighted by the cross-cell-type variance, normalised to sum 1 across bins.

    `var` is required here, unlike the reference, which returns `0.0` when it is absent (quirk 5).
    `annotations.load_variance_pool` is where that refusal is raised and explained.
    """
    var = np.asarray(var, dtype=y_true.dtype if y_true.dtype.kind == "f" else np.float64)
    return float(((y_true - y_pred) ** 2).dot(var) / var.sum())


def mse1obs(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MSE over the top 1% of positions ranked by the **observed** signal.

    `>=` against the `int(N*0.01)`-th largest value, so ties admit more than 1%. With `N < 100` the
    threshold index is `-0 == 0` and the whole array is selected (quirk 4).
    """
    idx = _top1_mask(y_true)
    return mse(y_true[idx], y_pred[idx])


def mse1imp(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MSE over the top 1% of positions ranked by the **predicted** signal."""
    idx = _top1_mask(y_pred)
    return mse(y_true[idx], y_pred[idx])


def _top1_mask(rank_by: np.ndarray) -> np.ndarray:
    n = int(rank_by.shape[0] * 0.01)
    threshold = np.sort(rank_by)[-n]          # -0 is 0: the minimum, hence quirk 4
    return rank_by >= threshold


# ---------------------------------------------------------------------------
# region-restricted
# ---------------------------------------------------------------------------

def _resolve(lo: int, hi: int, n: int) -> Tuple[int, int]:
    """Resolve `array[lo:hi]` to concrete bounds with Python's own slice semantics.

    This is the whole of quirk 2. `slice(lo, hi).indices(n)` applies negative-index wrapping and
    clamping exactly as the interpreter does, so a promoter running off the front of a long array
    resolves to an empty range and one running off the front of a *short* array does whatever
    Python would have done — which is not the same thing, and is why this is not a `max(lo, 0)`.
    """
    start, stop, _ = slice(lo, hi).indices(n)
    return start, stop


def _gene_intervals(gene_annotations: Sequence[str], chrom: str, n: int, *,
                    window_size: int, prom_loc: int, kind: str) -> List[Tuple[int, int, int]]:
    """`(sse_start, sse_stop, denominator)` per annotation line. `kind` is 'prom' or 'gene'.

    The denominator is a separate number from the slice, and that is quirk 7 — see `_sse_over`.
    """
    out: List[Tuple[int, int, int]] = []
    for line in gene_annotations:
        chrom_, start, end, _, _, strand = line.split()
        if chrom_ != chrom:
            continue
        s = int(start) // window_size
        e = int(end) // window_size + 1          # quirk 3
        if kind == "gene":
            lo, hi = _resolve(s, e, n)
            out.append((lo, hi, e - s))          # denominator is UNCLIPPED (quirk 7)
        elif strand == "+":
            lo, hi = _resolve(s - prom_loc, s, n)
            out.append((lo, hi, max(hi - lo, 0)))       # denominator is the clipped slice
        else:
            lo, hi = _resolve(e, e + prom_loc, n)
            out.append((lo, hi, max(hi - lo, 0)))
    return out


def _enh_intervals(enh_annotations: Sequence[str], chrom: str, n: int, *,
                   window_size: int) -> List[Tuple[int, int, int]]:
    out: List[Tuple[int, int, int]] = []
    for line in enh_annotations:
        chrom_, start, end, _, _, _, _, _, _, _, _, _ = line.split()
        if chrom_ != chrom:
            continue
        s = int(start) // window_size
        e = int(end) // window_size + 1
        lo, hi = _resolve(s, e, n)
        out.append((lo, hi, e - s))              # denominator is UNCLIPPED (quirk 7)
    return out


def _sse_over(y_true: np.ndarray, y_pred: np.ndarray,
              intervals: Sequence[Tuple[int, int, int]]) -> Tuple[float, float]:
    """Accumulate `sse` and `n` interval by interval, in annotation order.

    **Quirk 7 — the numerator and the denominator do not use the same interval, and they disagree
    differently in each metric.** All three reference metrics take their squared error from the
    Python slice, which clips at the end of the array. But:

    - `mseprom` counts `y_true[a:b].shape[0]` — the **clipped** length.
    - `msegene` and `mseenh` count `end - start` — the **unclipped** bin span.

    So a gene or enhancer whose annotation runs past the end of the scored array contributes zero
    error over the missing bins while still paying for them in the denominator, and `msegene` comes
    out deflated. On a whole chromosome this touches only the handful of features at the telomeric
    end; on a truncated array it dominates, which is why it surfaced on a 7-element fixture and
    would never have surfaced on random data alone.

    **Quirk 8 — what happens when nothing is selected depends on WHY nothing was selected.** The
    reference's `sse` starts as the Python float `0.0` and becomes a `numpy.float64` the first time a
    slice is summed into it, even an empty one. So:

    - annotations exist for the chromosome but every one selects zero bins → `sse` is `np.float64`,
      and `sse / n` is `np.float64(0.0) / 0.0` → **nan**, with a RuntimeWarning;
    - no annotation matches the chromosome at all → the loop body never runs, `sse` is still a Python
      float, and `sse / n` → **ZeroDivisionError**.

    Two different failures for the same empty result. We reproduce both, which is why the empty slice
    is summed rather than skipped: skipping it is faster, correct-looking, and turns the first case
    into a crash. Summing an empty slice adds `np.float64(0.0)` and changes no value and no addition
    order, so layer 1 still holds at 0 ULP.

    Deliberately NOT a cumulative-sum trick. Differencing a cumsum reorders the float additions, and
    over ~2M positions that moves the last few ULP — enough to fail a 0 ULP comparison for no gain.
    """
    sse, n = 0.0, 0.0
    for lo, hi, denom in intervals:
        sse = sse + ((y_true[lo:hi] - y_pred[lo:hi]) ** 2).sum()
        n += denom
    return sse, n


def mseprom(y_true_dict: Mapping[str, np.ndarray], y_pred_dict: Mapping[str, np.ndarray],
            chroms: Sequence[str], gene_annotations: Sequence[str], *,
            window_size: int = WINDOW_SIZE, prom_loc: int = PROM_LOC) -> float:
    """MSE over ±2 kb promoters: `prom_loc` bins upstream of the gene start, strand-aware."""
    sse, n = 0.0, 0.0
    for chrom in chroms:
        iv = _gene_intervals(gene_annotations, chrom, len(y_true_dict[chrom]),
                             window_size=window_size, prom_loc=prom_loc, kind="prom")
        s, c = _sse_over(y_true_dict[chrom], y_pred_dict[chrom], iv)
        sse += s
        n += c
    return sse / n


def msegene(y_true_dict: Mapping[str, np.ndarray], y_pred_dict: Mapping[str, np.ndarray],
            chroms: Sequence[str], gene_annotations: Sequence[str], *,
            window_size: int = WINDOW_SIZE) -> float:
    """MSE over GENCODE gene bodies. Overlapping genes re-weight shared positions (quirk 1)."""
    sse, n = 0.0, 0.0
    for chrom in chroms:
        iv = _gene_intervals(gene_annotations, chrom, len(y_true_dict[chrom]),
                             window_size=window_size, prom_loc=0, kind="gene")
        s, c = _sse_over(y_true_dict[chrom], y_pred_dict[chrom], iv)
        sse += s
        n += c
    return sse / n


def mseenh(y_true_dict: Mapping[str, np.ndarray], y_pred_dict: Mapping[str, np.ndarray],
           chroms: Sequence[str], enh_annotations: Sequence[str], *,
           window_size: int = WINDOW_SIZE) -> float:
    """MSE over FANTOM5 permissive enhancers."""
    sse, n = 0.0, 0.0
    for chrom in chroms:
        iv = _enh_intervals(enh_annotations, chrom, len(y_true_dict[chrom]),
                            window_size=window_size)
        s, c = _sse_over(y_true_dict[chrom], y_pred_dict[chrom], iv)
        sse += s
        n += c
    return sse / n


# ---------------------------------------------------------------------------
# the whole block
# ---------------------------------------------------------------------------

def dict_to_arr(d: Mapping[str, np.ndarray], chroms: Sequence[str]) -> np.ndarray:
    """Concatenate per-chromosome vectors in `chroms` order.

    Load-bearing for `mse1obs`/`mse1imp`: the top 1% is taken over the **concatenation of every
    scored chromosome**, not per chromosome. Scoring chr21 alone and scoring chr21 with chr22 give
    different thresholds, so the chromosome set is part of the number.
    """
    return np.concatenate([np.asarray(d[c]) for c in chroms])


def score_track(y_true_dict: Mapping[str, np.ndarray], y_pred_dict: Mapping[str, np.ndarray],
                chroms: Sequence[str], *, gene_annotations: Sequence[str],
                enh_annotations: Sequence[str], var: Optional[np.ndarray] = None,
                window_size: int = WINDOW_SIZE, prom_loc: int = PROM_LOC) -> Dict[str, float]:
    """All nine measures for one (cell, assay) track. The per-track score is the primitive (D4).

    `var` is the concatenated variance vector, aligned with `dict_to_arr(..., chroms)`. Omitting it
    omits `msevar` from the result rather than scoring it 0.0.
    """
    y_true = dict_to_arr(y_true_dict, chroms)
    y_pred = dict_to_arr(y_pred_dict, chroms)
    out = {
        "mse": mse(y_true, y_pred),
        "gwcorr": gwcorr(y_true, y_pred),
        "gwspear": gwspear(y_true, y_pred),
        "mseprom": mseprom(y_true_dict, y_pred_dict, chroms, gene_annotations,
                           window_size=window_size, prom_loc=prom_loc),
        "msegene": msegene(y_true_dict, y_pred_dict, chroms, gene_annotations,
                           window_size=window_size),
        "mseenh": mseenh(y_true_dict, y_pred_dict, chroms, enh_annotations,
                         window_size=window_size),
        "mse1obs": mse1obs(y_true, y_pred),
        "mse1imp": mse1imp(y_true, y_pred),
    }
    if var is not None:
        out["msevar"] = msevar(y_true, y_pred, var)
    return out
