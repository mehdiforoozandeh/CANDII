"""The two numbers the eDICE gate is scored on, computed the way the reference computes them.

`models/metrics.py` accumulates PER-TRACK sums and only averages over tracks in `result()`. So
"MSE" in their tables is the MEAN OVER TRACKS of each track's mean squared error, not the pooled
error over all (track, bin) cells -- the two differ whenever tracks have unequal variance, which
they always do. `PearsonCorrelation` is likewise per track, then averaged.

`models/base.py::get_raw_metrics` inverse-transforms BOTH truth and prediction with `sinh` before
the metric, so the reported numbers live in raw `-log10 p` space, not arcsinh space. Supplementary
Table 2 quotes `mean +- s.e.m.` over the 203 test tracks, so `summarise` returns both.

This module is deliberately its own implementation and not an import from `candi.bench`: the gate
asks whether we reproduce eDICE's published numbers under eDICE's own definitions. Our instrument
enters later, through the §4 npz contract.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

__all__ = ["per_track_mse", "per_track_pearson", "summarise", "gate_report"]


def per_track_mse(truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """(n_bins, n_tracks) x2 -> (n_tracks,)."""
    d = np.asarray(pred, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    return np.mean(d * d, axis=0)


def per_track_pearson(truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """(n_bins, n_tracks) x2 -> (n_tracks,). A constant column gives NaN, as it must."""
    t = np.asarray(truth, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    t = t - t.mean(axis=0, keepdims=True)
    p = p - p.mean(axis=0, keepdims=True)
    denom = np.sqrt((t * t).sum(axis=0) * (p * p).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, (t * p).sum(axis=0) / denom, np.nan)


def summarise(values: np.ndarray) -> Dict[str, float]:
    """mean +- s.e.m. over tracks, the form Supplementary Table 2 quotes."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    n = int(v.size)
    return {
        "mean": float(v.mean()) if n else float("nan"),
        "sem": float(v.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
        "n_tracks": n,
    }


def gate_report(truth_raw: np.ndarray, pred_raw: np.ndarray,
                truth_arcsinh: np.ndarray, pred_arcsinh: np.ndarray) -> Dict[str, Dict]:
    """Both spaces, because which one Supplementary Table 2 is quoted in is not written down.

    The reference's code inverse-transforms before scoring, which makes `raw` the reading we act
    on; `arcsinh` is carried alongside so the README's comparison table cannot be accused of having
    picked the flattering space after the fact.
    """
    return {
        "raw": {
            "mse": summarise(per_track_mse(truth_raw, pred_raw)),
            "pearson": summarise(per_track_pearson(truth_raw, pred_raw)),
        },
        "arcsinh": {
            "mse": summarise(per_track_mse(truth_arcsinh, pred_arcsinh)),
            "pearson": summarise(per_track_pearson(truth_arcsinh, pred_arcsinh)),
        },
    }
