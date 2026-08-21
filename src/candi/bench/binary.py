"""B-block — how well the model recovers *which* positions are peaks, not how tall they are.

Three measures, and one deliberate absence.

**AUROC is not here, and that is a decision rather than an oversight** (`EVAL_PLAN.md` D14). Peak
positives are a small minority of loci, and ROC's false-positive rate carries the (large) negative
count in its denominator: a model can move its FPR from 0.02 to 0.01, shift AUROC visibly, and still
return mostly false positives among the positions it actually called. Average precision conditions on
the predicted positives instead, so it moves only when the calls get better.
`cruxvault/wiki/imputation-evaluation-measures.md` records the source and the argument.

`average_precision` is implemented here rather than imported from scikit-learn, which this
environment does not have and which would be a heavy dependency for one function. The implementation
is the standard step-wise sum `Σ (R_k − R_{k−1}) · P_k`, verified against hand-computable cases in
the analytic tests.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

__all__ = ["average_precision", "peak_overlap", "correspondence_curve", "binary_suite"]


def average_precision(y_binary: np.ndarray, score: np.ndarray) -> float:
    """Area under the precision–recall curve, the step-wise (non-interpolated) definition.

    `AP = Σ_k (R_k − R_{k−1}) · P_k` over thresholds descending through the sorted scores. Ties are
    handled by taking each *group* of equal scores as one threshold, which matters here: predicted
    peak probabilities saturate, so ties are common and treating them one-by-one would let the metric
    reward an arbitrary within-tie ordering the model never expressed.

    A perfect ranker gives 1.0; a random ranker gives the base rate. Both are asserted in the tests.
    """
    y = np.asarray(y_binary).astype(bool)
    s = np.asarray(score, dtype=np.float64)
    if y.size == 0 or not y.any():
        return float("nan")                      # no positives: precision is undefined everywhere

    order = np.argsort(-s, kind="mergesort")     # stable, so equal scores keep input order
    y, s = y[order], s[order]

    tp = np.cumsum(y)
    fp = np.cumsum(~y)
    # Collapse tied scores to their last index: every member of a tie shares one threshold.
    last_of_tie = np.r_[np.flatnonzero(np.diff(s)), s.size - 1]
    tp, fp = tp[last_of_tie], fp[last_of_tie]

    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / tp[-1]
    d_recall = np.diff(np.r_[0.0, recall])
    return float(np.sum(d_recall * precision))


def peak_overlap(y_true: np.ndarray, y_pred: np.ndarray, p: float = 0.01) -> float:
    """Fraction of the top-`p` observed positions that are also in the top-`p` predicted positions.

    Ported from `CANDI/_utils.py::METRICS.peak_overlap`, including its endpoint conventions (`p=0`
    returns 0, `p=1` returns 1). Note it is a **rank** overlap, not a peak-call overlap: it needs no
    peak file and asks only whether the model puts the same positions at the top.
    """
    if p == 0:
        return 0.0
    if p == 1:
        return 1.0
    top = int(p * len(y_true))
    if top <= 0:
        return float("nan")
    obs_i = np.argsort(np.asarray(y_true))[-top:]
    pred_i = np.argsort(np.asarray(y_pred))[-top:]
    return float(np.intersect1d(obs_i, pred_i).size / top)


def correspondence_curve(y_true: np.ndarray, y_pred: np.ndarray,
                         steps: Sequence[float] = tuple(p / 100 for p in range(0, 101))
                         ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Overlap of the top-`p` sets as `p` sweeps 0 to 1, plus its finite-difference derivative.

    Ported from `CANDI/_utils.py::METRICS.correspondence_curve`, **including the normalisation the
    original used**: interior points divide the overlap by `len(y_true)`, not by the size of the
    top-`p` set, so the curve runs from 0 to 1 across the sweep rather than being a per-`p` fraction.
    Under a perfect ranker the curve is the identity `y = p` and the derivative is 1 everywhere, which
    is what makes the derivative the readable half — a model that ranks the extreme tail well and the
    middle poorly shows it as a dip, where a single overlap number would not.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    obs_rank = np.argsort(y_true)
    pred_rank = np.argsort(y_pred)

    curve: List[Tuple[float, float]] = []
    for p in steps:
        if p == 0 or p == 1:
            overlap_frac = float(p)
        else:
            top = int(p * n)
            overlap = np.intersect1d(obs_rank[-top:], pred_rank[-top:]).size if top > 0 else 0
            overlap_frac = overlap / n
        curve.append((float(p), float(overlap_frac)))

    derivatives = [
        (curve[i][0], (curve[i][1] - curve[i - 1][1]) / (curve[i][0] - curve[i - 1][0]))
        for i in range(1, len(curve))
    ]
    return curve, derivatives


def binary_suite(y_peaks: np.ndarray, score: np.ndarray, y_signal: np.ndarray, *,
                 p: float = 0.01, with_curve: bool = False) -> Dict[str, object]:
    """Every B-block key for one track.

    `y_peaks` is the MACS2 peak label, `score` the model's ranking signal for it (the peak head's
    probability where the head is on, otherwise the predicted level), and `y_signal` the experimental
    signal that `peak_overlap` ranks by. They are three different things and the call site must not
    conflate them, which is why none of them defaults.
    """
    out: Dict[str, object] = {
        "auprc": average_precision(y_peaks, score),
        "peak_base_rate": float(np.mean(np.asarray(y_peaks).astype(bool))),
        f"peak_overlap_{p}": peak_overlap(y_signal, score, p=p),
        "n_points": int(len(y_signal)),
    }
    if with_curve:
        curve, deriv = correspondence_curve(y_signal, score)
        out["correspondence_curve"] = curve
        out["correspondence_derivative"] = deriv
    return out
