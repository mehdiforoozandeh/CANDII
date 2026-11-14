"""Development eval: Max's error budget, ported to count predictions with a pseudoreplicate noise term (D12).

Ported from mlibbrecht/01_2026-10-19_research @ 3e7478a1 (projects/03_2026-10-20_epi_imputation/sprints/):
  06_2026-10-31_decomposing_error/scripts/lib/errdecomp.py    (calibrate, haar, shared_specific; imported here,
                                                              byte-identical to CANDII tools/eic_decomp/errdecomp.py)
  06_2026-10-31_decomposing_error/scripts/report2/aggregate2.py  (Gram build, eue(): explained / unexplained /
                                                              unexplainable / added, noise in specific parts only)
  07_2026-11-04/scripts/floors/eue_rep.py                     (per-level Spearman-Brown floor from a replicate pair,
                                                              weighted by the target's own spectrum; summary sums)

All inputs are already in one space (e.g. log1p counts at the target's own depth). One call covers the held-out
tracks of ONE assay. Per track (units of the truth's variance Var(x)):

  MSE / Var(x) = level + shared + specific,   level = offset + scale + nonlinear      (errdecomp.calibrate)

The standardized truth (x - mean x) / SD x and calibrated prediction (f(y) - mean x) / SD x are Haar-transformed
(15 scales, 25 bp .. 409.6 kb plus the coarse remainder); their Gram matrices (V truth, P prediction, C cross,
E = V - 2C + P error) are split into shared and specific parts with groups = cell (errdecomp.shared_specific). Per
component (shared or specific part of one band), on track means:

  explained   = C^2 / P   (0 if C <= 0; at most V)
  added       = E - (V - explained)
  noise       = N, the sampling noise (specific parts only; 0 in shared parts and when no floor is given)
  unexplained = V - explained - N   (clipped at 0; then N is reduced to V - explained and floor_flag is set)

so V = explained + unexplained + noise and E = unexplained + noise + added, exactly, in every component and in the
sums. N for track t at Haar scale k is noise_by_level[t, k] * V_tot[t, k]: the noise share of that scale (1 minus
the Spearman-Brown reliability, see noise_share_from_halves) times the truth's energy at that scale.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)
from eic_decomp import errdecomp as E  # noqa: E402

J = E.J
K = J + 1                      # 14 Haar detail levels + the coarse remainder
BLOCK = 2 ** J                 # 16384 bins: one eval region; the Haar transform never mixes two blocks
BANDS = {"fine": [0, 1, 2], "peak": [3, 4, 5, 6], "domain": [7, 8, 9, 10], "broad": [11, 12, 13, 14]}
PARTS = ("shared", "specific")


def _pad(v):
    """Zero-pad the last axis to a multiple of BLOCK."""
    n = v.shape[-1]
    npad = -(-n // BLOCK) * BLOCK
    if npad == n:
        return v
    out = np.zeros(v.shape[:-1] + (npad,), v.dtype)
    out[..., :n] = v
    return out


def _gram(A, B):
    """Per-scale Gram matrices <A_b, B_c> of the Haar coefficients of rows of A and B (T, Mpad): (K, T, T)."""
    ca, cb = E.haar(A), E.haar(B)
    return np.stack([a @ b.T for a, b in zip(ca, cb)])


def _standardize(v):
    v = np.asarray(v, np.float64)
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else np.zeros_like(v)


def noise_share_from_halves(half1, half2, depth_ratio=2.0, valid=None, shares=None):
    """Noise share per Haar scale of each target, from its two pseudoreplicate halves (eue_rep.py convention).

    half1, half2: (T, M) in the same space as the truth. R_k = per-scale correlation of the two standardized halves
    (Gram entry over sqrt of the two energies), clipped to [0, 1]. The full track holds depth_ratio halves' worth of
    reads (2 for pr1/pr2 of the pooled target; depth in fragments), so its reliability is the Spearman-Brown
    k R / (1 + (k - 1) R) and the noise share is 1 minus that. Returns (T, K) in [0, 1].

    shares: (T, 2) or (2,), each half's real share (f1, f2) of the target's reads (halves rescaled by 1 / f_i
    before log1p, as score.py does). It replaces depth_ratio with the unequal-length form of
    `metrics.noise_floor`: R^2 (s + 1/f1)(s + 1/f2) = s^2 for the signal-to-noise ratio s, noise share
    1 / (1 + s). A target with f1 == f2 takes the form above with k = 1 / f1 (bit-identical for even halves)."""
    h1, h2 = np.atleast_2d(np.asarray(half1, np.float64)), np.atleast_2d(np.asarray(half2, np.float64))
    if h1.shape != h2.shape:
        raise ValueError(f"halves differ in shape: {h1.shape} vs {h2.shape}")
    keep = np.ones(h1.shape[1], bool) if valid is None else np.asarray(valid, bool)
    A = np.zeros(h1.shape); B = np.zeros(h2.shape)
    for t in range(h1.shape[0]):
        A[t, keep] = _standardize(h1[t, keep]); B[t, keep] = _standardize(h2[t, keep])
    ca, cb = E.haar(_pad(A)), E.haar(_pad(B))
    R = np.zeros((h1.shape[0], K))
    for k in range(K):
        xz = np.einsum("tm,tm->t", ca[k], cb[k])
        xx = np.einsum("tm,tm->t", ca[k], ca[k]); zz = np.einsum("tm,tm->t", cb[k], cb[k])
        with np.errstate(invalid="ignore", divide="ignore"):
            R[:, k] = np.where(xx * zz > 0, xz / np.sqrt(xx * zz), 0.0)
    R = R.clip(0, 1)
    if shares is None:
        k_ = float(depth_ratio)
        return 1 - k_ * R / (1 + (k_ - 1) * R)
    f = np.broadcast_to(np.asarray(shares, np.float64), (h1.shape[0], 2))
    out = np.empty_like(R)
    for t in range(R.shape[0]):
        f1, f2 = float(f[t, 0]), float(f[t, 1])
        r = R[t]
        if f1 == f2:
            k_ = 1.0 / f1
            out[t] = 1 - k_ * r / (1 + (k_ - 1) * r)
            continue
        a, b, r2 = 1.0 / f1, 1.0 / f2, r * r
        with np.errstate(invalid="ignore", divide="ignore"):
            s = (r2 * (a + b) + np.sqrt(r2 * r2 * (a + b) ** 2 + 4.0 * (1.0 - r2) * r2 * a * b)) / (2.0 * (1.0 - r2))
        out[t] = np.where(r2 >= 1.0, 0.0, 1.0 / (1.0 + s))
    return out


def _eue(V, C, P, Ev, N):
    """Max's eue() (aggregate2.py / eue_rep.py), on track means of one component."""
    expl = (C * C / P) if (C > 0 and P > 0) else 0.0
    capped = expl > V
    expl = min(expl, max(V, 0.0))
    added = Ev - (V - expl)
    unex = V - expl - N
    flag = unex < 0
    if flag:
        N, unex = V - expl, 0.0
    return dict(V=V, explained=expl, unexplained=unex, noise=N, added=added, E=Ev,
                cap_flag=bool(capped), floor_flag=bool(flag))


def error_budget(truth, pred, groups, noise_by_level=None, valid=None):
    """Error budget of one assay's held-out tracks.

    truth, pred: (T, M) in one space; M bins laid out as consecutive 16384-bin regions (a shorter tail is
    zero-padded). groups: (T,) cell labels; a track never estimates its own shared part from its own cell.
    noise_by_level: (T, 15) noise share per Haar scale (noise_share_from_halves) or None (no noise term: noise = 0,
    as CANDII's EIC aggregate.py). valid: optional (M,) bool; invalid bins are dropped from the calibration and
    hold 0 in the Haar transform (errdecomp.Layout convention).

    Returns track means: level, shared, specific (error energies), explained, unexplained, noise, added (summed over
    the shared and specific components of the four bands), V, E, mse, n_tracks, and per_band[part][band] with the
    component rows of Max's eue table. With tracks from one cell only there is no shared estimate: shared, specific
    and every budget term are NaN (level and mse stay finite)."""
    x = np.atleast_2d(np.asarray(truth, np.float64)); y = np.atleast_2d(np.asarray(pred, np.float64))
    groups = np.asarray(groups)
    T, M = x.shape
    if y.shape != (T, M) or groups.shape != (T,):
        raise ValueError(f"shapes: truth {x.shape}, pred {y.shape}, groups {groups.shape}")
    keep = np.ones(M, bool) if valid is None else np.asarray(valid, bool)
    if noise_by_level is not None:
        noise_by_level = np.asarray(noise_by_level, np.float64)
        if noise_by_level.shape != (T, K) or not np.isfinite(noise_by_level).all():
            raise ValueError(f"noise_by_level must be finite, shape {(T, K)}; got {noise_by_level.shape}")

    X = np.zeros((T, M)); Y = np.zeros((T, M))
    calib = []
    for t in range(T):
        tc, yc = E.calibrate(x[t, keep], y[t, keep])
        sx = np.sqrt(tc["var_x"])
        X[t, keep] = (x[t, keep] - tc["mean_x"]) / sx
        Y[t, keep] = (yc - tc["mean_x"]) / sx
        calib.append(tc)
    nv = int(keep.sum())
    X, Y = _pad(X), _pad(Y)
    Gx = _gram(X, X) / nv
    Gy = _gram(Y, Y) / nv
    Gxy = _gram(X, Y) / nv
    Gc = 0.5 * (Gxy + np.transpose(Gxy, (0, 2, 1)))
    Ge = Gx - 2 * Gc + Gy
    parts = {nm: E.shared_specific(G, groups) for nm, G in (("V", Gx), ("P", Gy), ("C", Gc), ("E", Ge))}

    level = float(np.mean([c["offset"] + c["scale"] + c["nonlinear"] for c in calib]))
    mse = float(np.mean([c["mse"] for c in calib]))
    out = {"level": level, "mse": mse, "n_tracks": T}
    if np.isnan(parts["V"][0]).any():                                  # one cell only: no shared estimate
        nan = float("nan")
        out.update(shared=nan, specific=nan, explained=nan, unexplained=nan, noise=nan, added=nan, V=nan, E=nan,
                   per_band={p: {b: None for b in BANDS} for p in PARTS})
        return out

    vtot = np.einsum("kbb->bk", Gx)                                    # (T, K) truth energy per scale
    Nlev = np.zeros((T, K)) if noise_by_level is None else noise_by_level * vtot
    per_band = {p: {} for p in PARTS}
    for i, p in enumerate(PARTS):
        for b, idx in BANDS.items():
            V, P_, C, Ev = (float(parts[nm][i][idx].sum(0).mean()) for nm in ("V", "P", "C", "E"))
            N = float(Nlev[:, idx].sum(1).mean()) if p == "specific" else 0.0
            per_band[p][b] = _eue(V, C, P_, Ev, N)
    rows = [per_band[p][b] for p in PARTS for b in BANDS]
    out.update(shared=float(parts["E"][0].sum(0).mean()), specific=float(parts["E"][1].sum(0).mean()),
               **{k: float(sum(r[k] for r in rows)) for k in ("explained", "unexplained", "noise", "added", "V", "E")},
               per_band=per_band)
    return out
