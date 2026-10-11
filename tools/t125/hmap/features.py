"""h's input features at rung 1 and rung 5, raw (P2d's standardiser z-scores them at training).

Rung 1 (`FEATURES_R1`, 9 columns): the bin's count, the control count in the same bin, and the
assay one-hot (`ladder.encoding.ASSAYS` order). Rung 5 (`FEATURES_R5`, 19 columns): rung 1 plus
the fragment-window treatment mean, the control window means (fragment, 1 kb, 10 kb), the genome
background rate, both depths, the control scaling ratio, the fragment length and the control flag.
`FEATURE_FORMULA` names the one extra column the `formula_*` variants take (last).

Windows: fragment = `frag_bins(fraglen)` bins (round(fraglen / 25), at least 1, made odd), 1 kb =
40 bins, 10 kb = 400 bins. A window of w bins around bin i covers bins i - w//2 .. i + (w - 1 - w//2)
(odd w: centred), truncated at the chromosome ends; a window feature is the MEAN count over the bins
it keeps (`window_mean`).

Column rules (all per bin, float32):
  log1p_k          log1p(bin count) — `k_column`; NOT in `fixed_features` (the chain enumerates k)
  log1p_ctl_bin    log1p(control count in the bin); 0 without a control
  assay_<A>        one-hot of the product's assay
  log1p_k_frag     log1p(fragment-window mean of the treatment); with `treat_mean` (the chain:
                   f's predicted mean) the window is taken over `treat_mean`; the chain then
                   replaces this column per enumerated k by log1p(`k_frag_chain`)
  log1p_ctl_frag / _1k / _10k   log1p(control window mean); 0 without a control
  log_lambda_bg    log caller.lambda_bg(N_C * s_C, fraglen) with a control, (N_T, fraglen) without
  log_depth_t      log(N_T / 1e6)
  log_depth_c      log(N_C / 1e6) with a control; 0 without (has_control carries the difference)
  log_scale_ratio  log(s_C / s_T) of `caller.scale` (ratio V on the ratio arms); 0 without a control
  log_fraglen      log(fraglen)
  has_control      0 / 1
The no-control local-rate window (10 kb of the treatment, f's predicted mean in the chain) is not a
feature column; P2b's `formula.local_lambda` takes it, built with `window_sum` / `window_mean`.
"""
from __future__ import annotations

import math

import numpy as np

from ladder import encoding

from hmap import caller
from hmap.products import ProductInfo

D_UNIT = 1e6
ASSAY_COLUMNS = tuple(f"assay_{a}" for a in encoding.ASSAYS)
FEATURES_R1 = ("log1p_k", "log1p_ctl_bin") + ASSAY_COLUMNS
FEATURES_R5 = FEATURES_R1 + ("log1p_k_frag", "log1p_ctl_frag", "log1p_ctl_1k", "log1p_ctl_10k",
                             "log_lambda_bg", "log_depth_t", "log_depth_c", "log_scale_ratio",
                             "log_fraglen", "has_control")
FEATURE_FORMULA = "log1p_P_formula"
FEATURES = {1: FEATURES_R1, 5: FEATURES_R5}
#: columns P2d's standardiser must never scale (the one-hot and the flag)
UNSCALED = ASSAY_COLUMNS + ("has_control",)
W_1K = caller.WINDOWS_CTL_BP["1k"] // caller.BIN          # 40
W_10K = caller.WINDOWS_CTL_BP["10k"] // caller.BIN        # 400


def frag_bins(fraglen: float) -> int:
    """The fragment window in bins: round(fraglen / 25), at least 1, +1 when even."""
    w = max(1, int(math.floor(float(fraglen) / caller.BIN + 0.5)))
    return w + 1 if w % 2 == 0 else w


def window_bins(fraglen: float) -> dict[str, int]:
    """{"frag": frag_bins(fraglen), "1k": 40, "10k": 400}."""
    return {"frag": frag_bins(fraglen), "1k": W_1K, "10k": W_10K}


def _bounds(n: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    if w < 1:
        raise ValueError(f"window of {w} bins")
    i = np.arange(n)
    lo = np.maximum(i - w // 2, 0)
    hi = np.minimum(i + (w - 1 - w // 2) + 1, n)
    return lo, hi


def window_sum(a: np.ndarray, w: int) -> np.ndarray:
    """float64[n]: the sum of `a` over the w-bin window around each bin, truncated at the ends."""
    a = np.asarray(a)
    if a.ndim != 1:
        raise ValueError(f"window_sum needs a 1-D array, got shape {a.shape}")
    cs = np.concatenate(([0.0], np.cumsum(a, dtype=np.float64)))
    lo, hi = _bounds(a.shape[0], int(w))
    return cs[hi] - cs[lo]


def window_count(n: int, w: int) -> np.ndarray:
    """int64[n]: how many bins the w-bin window keeps around each bin of an n-bin chromosome."""
    lo, hi = _bounds(int(n), int(w))
    return hi - lo


def window_mean(a: np.ndarray, w: int) -> np.ndarray:
    """float64[n]: the mean of `a` over the bins the w-bin window keeps around each bin."""
    a = np.asarray(a)
    return window_sum(a, w) / window_count(a.shape[0], w)


def frag_rest(mean_t: np.ndarray, w_frag: int) -> np.ndarray:
    """float64[n]: the neighbours' sum in the fragment window = window sum of mean_t - mean_t."""
    mean_t = np.asarray(mean_t, dtype=np.float64)
    return window_sum(mean_t, w_frag) - mean_t


def k_frag_chain(k: np.ndarray, mean_t: np.ndarray, w_frag: int) -> np.ndarray:
    """float64[n, K]: the chain's fragment-window mean for each enumerated count k of a bin.

    (k + the neighbours' predicted means) / the bins the window keeps (= w_frag inside the
    chromosome, fewer at its ends, as `window_mean`). At k = mean_t = the true counts it equals
    `window_mean(counts, w_frag)`.
    """
    k = np.asarray(k, dtype=np.float64)
    mean_t = np.asarray(mean_t, dtype=np.float64)
    if k.ndim != 2 or k.shape[0] != mean_t.shape[0]:
        raise ValueError(f"k {k.shape} must be [n, K] with n = len(mean_t) = {mean_t.shape[0]}")
    rest = frag_rest(mean_t, w_frag)
    cnt = window_count(mean_t.shape[0], w_frag).astype(np.float64)
    return (k + rest[:, None]) / cnt[:, None]


def k_column(k) -> np.ndarray:
    """log1p(k) as float32 (the `log1p_k` column; any shape)."""
    return np.log1p(np.asarray(k, dtype=np.float64)).astype(np.float32)


def scales(info: ProductInfo) -> tuple[float, float]:
    """(s_T, s_C) of the product: `caller.scale` with a control, (1, 1) without."""
    if not info.has_control:
        return 1.0, 1.0
    return caller.scale(info.depth_reads, info.control_reads, info.ratio_V)


def log_lambda_bg(info: ProductInfo) -> float:
    """log of the background rate: the scaled control's N with a control, N_T without."""
    if info.has_control:
        _, s_c = scales(info)
        return math.log(caller.lambda_bg(info.control_reads * s_c, info.fraglen))
    return math.log(caller.lambda_bg(info.depth_reads, info.fraglen))


def fixed_features(info: ProductInfo, counts_t: np.ndarray, counts_c: np.ndarray | None,
                   rung: int, treat_mean: np.ndarray | None = None) -> np.ndarray:
    """float32[n, F - 1]: every column of `FEATURES[rung]` except `log1p_k`, in that order.

    `counts_t` / `counts_c`: one chromosome of the treatment / control counts (`counts_c` None
    exactly when the product has no control). `treat_mean` (f's predicted NB mean, the chain)
    replaces `counts_t` in the fragment-window treatment feature; the bin's own k is never
    taken from it (and is not a column here).
    """
    if rung not in FEATURES:
        raise ValueError(f"rung {rung!r} not in {tuple(FEATURES)}")
    counts_t = np.asarray(counts_t)
    if counts_t.ndim != 1:
        raise ValueError(f"counts_t must be 1-D, got shape {counts_t.shape}")
    n = counts_t.shape[0]
    if bool(info.has_control) != (counts_c is not None):
        raise ValueError(f"{info.pid}: has_control = {info.has_control} but counts_c is "
                         f"{'None' if counts_c is None else 'given'}")
    if counts_c is not None:
        counts_c = np.asarray(counts_c)
        if counts_c.shape != (n,):
            raise ValueError(f"{info.pid}: counts_c shape {counts_c.shape} != ({n},)")
    if treat_mean is not None:
        treat_mean = np.asarray(treat_mean, dtype=np.float64)
        if treat_mean.shape != (n,):
            raise ValueError(f"{info.pid}: treat_mean shape {treat_mean.shape} != ({n},)")
    if info.assay not in encoding.ASSAYS:
        raise ValueError(f"{info.pid}: assay {info.assay!r} not in {encoding.ASSAYS}")

    names = FEATURES[rung][1:]
    out = np.zeros((n, len(names)), dtype=np.float32)
    col = {name: j for j, name in enumerate(names)}
    if counts_c is not None:
        out[:, col["log1p_ctl_bin"]] = np.log1p(counts_c.astype(np.float64))
    out[:, col[f"assay_{info.assay}"]] = 1.0
    if rung == 1:
        return out

    w = window_bins(info.fraglen)
    treat = counts_t if treat_mean is None else treat_mean
    out[:, col["log1p_k_frag"]] = np.log1p(window_mean(treat, w["frag"]))
    if counts_c is not None:
        for key in ("frag", "1k", "10k"):
            out[:, col[f"log1p_ctl_{key}"]] = np.log1p(window_mean(counts_c, w[key]))
    s_t, s_c = scales(info)
    out[:, col["log_lambda_bg"]] = log_lambda_bg(info)
    out[:, col["log_depth_t"]] = math.log(info.depth_reads / D_UNIT)
    out[:, col["log_depth_c"]] = math.log(info.control_reads / D_UNIT) if info.has_control else 0.0
    out[:, col["log_scale_ratio"]] = math.log(s_c / s_t)
    out[:, col["log_fraglen"]] = math.log(info.fraglen)
    out[:, col["has_control"]] = float(info.has_control)
    return out
