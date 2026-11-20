"""E4 -- the three training-free baselines of D14, with the miniCANDI output format.

Each baseline has `fit(training batches)` and `predict(batch) -> {"n", "mu"}`, float32 torch
tensors of shape [B, L, A], as `MiniCandi.forward` (pinned interface C). A baseline writes NaN
where it is not defined (an assay it never saw, a track kind it does not serve), so a scorer that
reads it there fails loudly. No baseline is ever blended with a model (D10).

  MarginalNB        one constant NB per assay, fitted on the training targets
                    (`candi.bench.distributional.marginal_nb`, the CRPS-optimal constant NB).
  PositionAverage   the leave-one-cell-out per-position mean rate (count / depth) over the
                    training cells that measured the assay, scaled to the target depth from C',
                    with one NB size per assay fitted on the training targets.
  DepthOnlyScaling  correction and replicate pairs only (y_kind 2): the source counts times the
                    depth ratio C'/C, with one NB size per pair (`tools/t146/depth_baseline.py::
                    fit_size`). On a pure binomial-thinning pair this mean is exact.

Depth is column 0 of `c_in` and `c_out` (`log2_depth`, the first K_META column). Positions are
`meta[b]["chrom"]`, `meta[b]["start_bin"] + arange(L)`; the cell is `meta[b]["cell"]`. A pair is
named by `meta[b]["measurements_in"][a]` and `meta[b]["measurements_out"][a]` (the plan pins
"measurement ids in and out" but not their key names; these two names are this module's choice).
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

PRED_FLOOR = 1e-3            # the floor on a predicted mean, as tools/t118/baseline_rungs.py
FIT_MAX_BINS = 1_000_000     # systematic subsample for one NB size fit (t150 fitted ~1.1 M bins)
DEPTH_COL = 0                # log2_depth in c_in and c_out
KIND_CORRECTION = 2
PAIR_IN, PAIR_OUT = "measurements_in", "measurements_out"


def _np(x) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _out(n: np.ndarray, mu: np.ndarray) -> Dict[str, torch.Tensor]:
    return {"n": torch.as_tensor(n, dtype=torch.float32),
            "mu": torch.as_tensor(mu, dtype=torch.float32)}


def _fit_size(m: np.ndarray, y: np.ndarray) -> float:
    """ML NB size with the mean held fixed, on a systematic subsample of at most FIT_MAX_BINS."""
    from t146.depth_baseline import fit_size  # lazy: pulls in the t118 ladder modules
    stride = max(1, -(-m.size // FIT_MAX_BINS))
    return fit_size(np.asarray(m, np.float64)[::stride], np.asarray(y, np.float64)[::stride])


def _shape(batch) -> tuple:
    return tuple(_np(batch["y_counts"]).shape)


class MarginalNB:
    """One constant NB(mu_a, n_a) per assay a, fitted on every available training target."""

    def __init__(self):
        self.params: Dict[int, tuple] = {}

    def fit(self, batches: Iterable[dict]) -> "MarginalNB":
        from candi.bench.distributional import marginal_nb
        ys: Dict[int, List[np.ndarray]] = defaultdict(list)
        for batch in batches:
            y, avail = _np(batch["y_counts"]), _np(batch["y_avail"]).astype(bool)
            for b, a in zip(*np.nonzero(avail)):
                ys[int(a)].append(y[b, :, a])
        for a, parts in ys.items():
            r = marginal_nb(np.concatenate(parts))
            self.params[a] = (float(r["marg_n"]), float(r["marg_mu"]))
        return self

    def predict(self, batch) -> Dict[str, torch.Tensor]:
        B, L, A = _shape(batch)
        n, mu = np.full((B, L, A), np.nan), np.full((B, L, A), np.nan)
        for a, (na, ma) in self.params.items():
            n[:, :, a], mu[:, :, a] = na, ma
        return _out(n, mu)


class PositionAverage:
    """Leave-one-cell-out per-position mean rate over the training cells, scaled by C' depth.

    A cell's rate at a position is its count / 2**log2_depth, averaged over its windows that cover
    the position. The average for a target in cell c is the mean of the other cells' rates at that
    position, so it never includes c. Where no other cell covers the position, it falls back to
    the mean over the other cells of their whole-track mean rate; with no other cell it is NaN.
    """

    def __init__(self):
        self._acc = defaultdict(lambda: defaultdict(list))   # (chrom, a) -> cell -> [(bins, rate)]
        self.table: Dict[tuple, dict] = {}                   # (chrom, a) -> finalized arrays
        self.cell_rate: Dict[int, Dict[str, list]] = defaultdict(dict)  # a -> cell -> [sum, n]
        self.size: Dict[int, float] = {}

    @staticmethod
    def _rows(batch):
        y, avail = _np(batch["y_counts"]), _np(batch["y_avail"]).astype(bool)
        depth = 2.0 ** _np(batch["c_out"])[:, :, DEPTH_COL].astype(np.float64)
        L = y.shape[1]
        for b, a in zip(*np.nonzero(avail)):
            m = batch["meta"][b]
            bins = int(m["start_bin"]) + np.arange(L, dtype=np.int64)
            yield b, int(a), m, bins, y[b, :, a].astype(np.float64), float(depth[b, a])

    def fit(self, batches: Iterable[dict]) -> "PositionAverage":
        batches = list(batches)
        for batch in batches:
            for _, a, m, bins, y, d in self._rows(batch):
                self._acc[(m["chrom"], a)][m["cell"]].append((bins, y / d))
                s = self.cell_rate[a].setdefault(m["cell"], [0.0, 0])
                s[0] += float(np.sum(y / d))
                s[1] += y.size
        for key, cells in self._acc.items():
            U = np.unique(np.concatenate([bn for parts in cells.values() for bn, _ in parts]))
            per_cell, S, N = {}, np.zeros(U.size), np.zeros(U.size)
            for cell, parts in cells.items():
                idx = np.searchsorted(U, np.concatenate([bn for bn, _ in parts]))
                r = np.concatenate([rt for _, rt in parts])
                cnt = np.bincount(idx, minlength=U.size)
                mean = np.bincount(idx, r, minlength=U.size) / np.maximum(cnt, 1)
                per_cell[cell] = (mean, cnt > 0)
                S += mean
                N += cnt > 0
            self.table[key] = {"U": U, "S": S, "N": N, "cells": per_cell}
        self._acc.clear()
        xs: Dict[int, List[np.ndarray]] = defaultdict(list)
        ys: Dict[int, List[np.ndarray]] = defaultdict(list)
        for batch in batches:
            for _, a, m, bins, y, d in self._rows(batch):
                xs[a].append(self.loo_rate(m["chrom"], a, m["cell"], bins) * d)
                ys[a].append(y)
        for a in xs:
            mu = np.concatenate(xs[a])
            ok = np.isfinite(mu)
            if ok.any():
                self.size[a] = _fit_size(np.maximum(mu[ok], PRED_FLOOR), np.concatenate(ys[a])[ok])
        return self

    def loo_rate(self, chrom, a: int, cell, bins: np.ndarray) -> np.ndarray:
        """Mean rate of the training cells other than `cell` at `bins`; NaN where undefined."""
        others = [s / n for c, (s, n) in self.cell_rate.get(a, {}).items() if c != cell and n]
        fallback = float(np.mean(others)) if others else np.nan
        out = np.full(bins.size, fallback)
        t = self.table.get((chrom, a))
        if t is None:
            return out
        U = t["U"]
        idx = np.minimum(np.searchsorted(U, bins), U.size - 1)
        hit = U[idx] == bins
        S, N = t["S"][idx].copy(), t["N"][idx].copy()
        if cell in t["cells"]:
            mean, present = t["cells"][cell]
            S -= mean[idx]
            N -= present[idx]
        ok = hit & (N > 0)
        out[ok] = S[ok] / N[ok]
        return out

    def predict(self, batch) -> Dict[str, torch.Tensor]:
        B, L, A = _shape(batch)
        n, mu = np.full((B, L, A), np.nan), np.full((B, L, A), np.nan)
        depth = 2.0 ** _np(batch["c_out"])[:, :, DEPTH_COL].astype(np.float64)
        for b in range(B):
            m = batch["meta"][b]
            bins = int(m["start_bin"]) + np.arange(L, dtype=np.int64)
            for a, size in self.size.items():
                r = self.loo_rate(m["chrom"], a, m["cell"], bins)
                mu[b, :, a] = np.maximum(r * depth[b, a], PRED_FLOOR)   # NaN stays NaN
                n[b, :, a] = np.where(np.isfinite(r), size, np.nan)
        return _out(n, mu)


class DepthOnlyScaling:
    """mu = max(x * 2**(log2_depth C' - log2_depth C), PRED_FLOOR), one NB size per pair.

    Defined only on y_kind == 2 entries (correction and replicate pairs); NaN elsewhere. `fit`
    takes batches of the same pairs on fitting bins (t150 used the training chromosomes); a pair
    seen by `predict` but never fitted raises KeyError.
    """

    def __init__(self):
        self.size: Dict[tuple, float] = {}

    @staticmethod
    def _rows(batch):
        x = _np(batch["x_counts"]).astype(np.float64)
        kind = _np(batch["y_kind"])
        ok = (kind == KIND_CORRECTION) & _np(batch["x_avail"]).astype(bool)
        ratio = 2.0 ** (_np(batch["c_out"])[:, :, DEPTH_COL].astype(np.float64)
                        - _np(batch["c_in"])[:, :, DEPTH_COL].astype(np.float64))
        for b, a in zip(*np.nonzero(ok)):
            m = batch["meta"][b]
            key = (m[PAIR_IN][a], m[PAIR_OUT][a])
            yield b, int(a), key, np.maximum(x[b, :, a] * ratio[b, a], PRED_FLOOR)

    def fit(self, batches: Iterable[dict]) -> "DepthOnlyScaling":
        ms: Dict[tuple, List[np.ndarray]] = defaultdict(list)
        ys: Dict[tuple, List[np.ndarray]] = defaultdict(list)
        for batch in batches:
            y, yav = _np(batch["y_counts"]), _np(batch["y_avail"]).astype(bool)
            for b, a, key, m in self._rows(batch):
                if yav[b, a]:
                    ms[key].append(m)
                    ys[key].append(y[b, :, a])
        for key in ms:
            self.size[key] = _fit_size(np.concatenate(ms[key]), np.concatenate(ys[key]))
        return self

    def predict(self, batch) -> Dict[str, torch.Tensor]:
        B, L, A = _shape(batch)
        n, mu = np.full((B, L, A), np.nan), np.full((B, L, A), np.nan)
        for b, a, key, m in self._rows(batch):
            if key not in self.size:
                raise KeyError(f"depth-only scaling: pair {key} has no fitted size")
            mu[b, :, a], n[b, :, a] = m, self.size[key]
        return _out(n, mu)


BASELINES = {"marginal_nb": MarginalNB, "position_average": PositionAverage,
             "depth_only_scaling": DepthOnlyScaling}
