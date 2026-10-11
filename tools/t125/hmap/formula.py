"""t125 formula h: the caller's Poisson upper tail at the local λ, with nothing learned.

The recipe that ran (plan/T125_STATUS.md, Phase 0 findings and PI decisions, 2026-10-11):

- ChIP: chip-seq-pipeline2 v2.2.2, MACS2 `--nomodel --extsize FRAGLEN --keep-dup all`, `bdgcmp -m ppois`.
  DNase: atac-seq-pipeline v2.2.3 in DNase mode (`--shift -75 --extsize 150`, no control).
- pileup ≈ count × fraglen / (read_length + 24) × s_T. Variant "bin" uses the bin's count; variant
  "frag" uses the mean count over the fragment window (zero outside the chromosome, as the caller's
  pileup is).
- λ = max(λ_bg, λ_frag, λ_1k, λ_10k) from the binned control, scaled to the smaller set (MACS2
  `--scale-to small`), ratio V in ratio arms. No control: max(λ_bg(N_T), λ_10k of the treatment); in
  the chain that treatment window reads f's predicted mean.
- One tail per bin: P = −log10 P(X > floor(pileup)) (strict, no pseudocount). Phase 0 check: obs 0 at
  λ = 2 gives 0.06315 = −log10(1 − e^−2).

The caller constants and `scale` / `lambda_bg` / `lambda_window` restate P2a's pinned `hmap.caller`
definitions, so this module stands alone (P2a and P2b were built in parallel).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import poisson

GENOME_SIZE = 2.7e9
BIN = 25
SPAN_EXTRA = 24
WINDOWS_CTL_BP = {"frag": None, "1k": 1000, "10k": 10000}   # frag = the product's fraglen
WINDOW_NOCTL_BP = 10000
LAMBDA_FLOOR = 1e-12
VARIANTS = ("bin", "frag")

_LN10 = math.log(10.0)
_LOGSF_SAFE = -690.0          # below this scipy's sf is near the double range edge; use the series
_SERIES_MAX_TERMS = 1_000_000


# ---------------------------------------------------------------------------------------------
# the tail
def _logsf_series(k: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """log P(X > k) = log pmf(k + 1) + log Σ_i Π_{m<i} λ / (k + 2 + m), summed until negligible."""
    out = poisson.logpmf(k + 1.0, lam)
    total = np.ones_like(k)
    term = np.ones_like(k)
    live = np.ones(k.shape, dtype=bool)
    i = 0
    while live.any() and i < _SERIES_MAX_TERMS:
        term[live] *= lam[live] / (k[live] + 2.0 + i)
        total[live] += term[live]
        ratio = lam / (k + 3.0 + i)
        live &= ~((ratio < 1.0) & (term < 1e-17 * total * (1.0 - ratio)))
        i += 1
    return out + np.log(total)


def neglog10_sf_strict(k_floor, lam) -> np.ndarray:
    """−log10 P(Poisson(λ) > floor(k)), float64, broadcast over k and λ.

    λ is clipped at 1e-12 from below, so the value is finite for every λ >= 0; it is 0 where λ is 0
    and k is 0. Where scipy's logsf underflows, the tail is summed in log space instead.
    """
    k, lam = np.broadcast_arrays(np.floor(np.asarray(k_floor, dtype=np.float64)),
                                 np.asarray(lam, dtype=np.float64))
    lam_c = np.maximum(lam, LAMBDA_FLOOR)
    with np.errstate(divide="ignore"):
        lsf = np.asarray(poisson.logsf(k, lam_c), dtype=np.float64)
    bad = ~np.isfinite(lsf) | (lsf < _LOGSF_SAFE)
    if bad.any():
        lsf = lsf.copy()
        lsf[bad] = _logsf_series(k[bad], lam_c[bad])
    out = np.maximum(-lsf / _LN10, 0.0)
    out[(lam <= 0.0) & (k <= 0.0)] = 0.0
    return out


# ---------------------------------------------------------------------------------------------
# geometry and scaling
def frag_bins(fraglen) -> int:
    """Fragment window in bins: max(1, round(fraglen / 25)), made odd."""
    w = max(1, int(round(float(fraglen) / BIN)))
    return w + 1 if w % 2 == 0 else w


def window_sum(x, w_bins: int) -> np.ndarray:
    """Centred window sum over [i − w//2, i − w//2 + w), zero outside the chromosome; float64."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    c = np.concatenate([[0.0], np.cumsum(x)])
    i = np.arange(n)
    lo = np.clip(i - w_bins // 2, 0, n)
    hi = np.clip(i - w_bins // 2 + w_bins, 0, n)
    return c[hi] - c[lo]


def _has_control(info) -> bool:
    return bool(int(info.has_control)) and float(info.control_reads) > 0


def scale(info) -> tuple[float, float]:
    """(s_T, s_C), MACS2 `--scale-to small`: r = V (ratio arms) or N_T / N_C; r <= 1 → (1, r), else (1/r, 1).

    No control: (1, 1).
    """
    if not _has_control(info):
        return 1.0, 1.0
    r = float(info.ratio_V) if info.ratio_V is not None else float(info.depth_reads) / float(info.control_reads)
    return (1.0, r) if r <= 1.0 else (1.0 / r, 1.0)


def lambda_bg(n_scaled, fraglen) -> float:
    return float(fraglen) * float(n_scaled) / GENOME_SIZE


def lambda_window(tags, window_bp, fraglen, s):
    return np.asarray(tags, dtype=np.float64) * float(fraglen) / float(window_bp) * float(s)


def _tags(count_sum, read_length) -> np.ndarray:
    """Read-span overlap counts → tags: a read touches (read_length + 24) / 25 bins on average."""
    return np.asarray(count_sum, dtype=np.float64) * BIN / (float(read_length) + SPAN_EXTRA)


# ---------------------------------------------------------------------------------------------
# λ and pileup
def local_lambda(info, counts_c_windows: dict | None, treat_10k) -> np.ndarray:
    """The local λ per bin, float64.

    With a control: `counts_c_windows` = {"frag", "1k", "10k"} → the control count window SUMS;
    λ = max(λ_bg(N_C · s_C), λ_frag, λ_1k, λ_10k), the window λ's at s_C. Without a control:
    `treat_10k` = the 10 kb window SUM of the treatment track (f's mean in the chain);
    λ = max(λ_bg(N_T), λ_10k).
    """
    fl = float(info.fraglen)
    if _has_control(info):
        _, s_c = scale(info)
        rl_c = info.control_read_length
        lam = None
        for key, w_bp in WINDOWS_CTL_BP.items():
            lw = lambda_window(_tags(counts_c_windows[key], rl_c), fl if w_bp is None else w_bp, fl, s_c)
            lam = lw if lam is None else np.maximum(lam, lw)
        return np.maximum(lam, lambda_bg(float(info.control_reads) * s_c, fl))
    lw = lambda_window(_tags(treat_10k, info.read_length), WINDOW_NOCTL_BP, fl, 1.0)
    return np.maximum(lw, lambda_bg(float(info.depth_reads), fl))


def lambda_track(info, counts_c, treat) -> np.ndarray:
    """`local_lambda` from whole tracks: the control (None without one) and the treatment track
    (the true counts, or f's predicted mean in the chain; read only without a control)."""
    if _has_control(info):
        c = np.asarray(counts_c, dtype=np.float64)
        windows = {key: window_sum(c, frag_bins(info.fraglen) if w_bp is None else w_bp // BIN)
                   for key, w_bp in WINDOWS_CTL_BP.items()}
        return local_lambda(info, windows, None)
    return local_lambda(info, None, window_sum(treat, WINDOW_NOCTL_BP // BIN))


def frag_rest(treat, info) -> np.ndarray:
    """The neighbours' part of the fragment window: window sum of `treat` minus the bin itself."""
    t = np.asarray(treat, dtype=np.float64)
    return window_sum(t, frag_bins(info.fraglen)) - t


def pileup(count, info):
    """count × fraglen / (read_length + 24) × s_T."""
    s_t, _ = scale(info)
    return np.asarray(count, dtype=np.float64) * float(info.fraglen) / (float(info.read_length) + SPAN_EXTRA) * s_t


# ---------------------------------------------------------------------------------------------
# P
def P_enumerated(variant: str, info, k, lam, frag_rest_: np.ndarray | None = None) -> np.ndarray:
    """P over an enumerated count axis: k [n, K], λ [n] → float64 [n, K].

    "bin": pileup of k. "frag": pileup of (k + frag_rest) / w_frag, `frag_rest` [n] the neighbours'
    (predicted-mean) sum over the fragment window.
    """
    k = np.asarray(k, dtype=np.float64)
    if variant == "bin":
        c = k
    elif variant == "frag":
        if frag_rest_ is None:
            raise ValueError("variant 'frag' needs frag_rest")
        rest = np.asarray(frag_rest_, dtype=np.float64)
        c = (k + (rest[:, None] if k.ndim == 2 else rest)) / frag_bins(info.fraglen)
    else:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    lam = np.asarray(lam, dtype=np.float64)
    return neglog10_sf_strict(np.floor(pileup(c, info)), lam[:, None] if k.ndim == 2 else lam)


def P(variant: str, info, counts_t, counts_c, treat_mean=None) -> np.ndarray:
    """The formula's P per bin, float64 [n]. `treat_mean` (the chain) replaces the treatment in the
    window terms (the no-control λ window, the fragment window's neighbours); the bin's own count is
    always `counts_t`."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    counts_t = np.asarray(counts_t, dtype=np.float64)
    treat = counts_t if treat_mean is None else np.asarray(treat_mean, dtype=np.float64)
    lam = lambda_track(info, counts_c, treat)
    rest = frag_rest(treat, info) if variant == "frag" else None
    return P_enumerated(variant, info, counts_t, lam, rest)
