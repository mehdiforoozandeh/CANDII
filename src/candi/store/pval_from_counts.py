"""MACS2-equivalent Poisson p-value at bin resolution, from binned read counts.

Two rules live here, and MACS2 keeps them in two separate methods: `pval_from_counts` is the
no-control rule, `pval_from_counts_with_control` is the rule a run with an input/control track
uses. They differ ONLY in how the local lambda is built; the tail is the same function.

The rule implemented here is read off the MACS2 source, not off the paper. Anchors, at tag
`2.1.0.20140616` of `macs3-project/MACS` (the no-control lambda block is byte-identical at
`v2.1.1.20160309`, which is the version ENCODE's ATAC signal step actually ran):

* `MACS2/cPeakDetect.pyx::PeakDetect.__call_peaks_wo_control`
    - `lambda_bg = float(d) * treat_total / self.gsize`
    - `ctrl_d_s = [ self.lregion, ]` and `ctrl_scale_s = [ float(self.d) / self.lregion, ]`
      **llocal only.** The source comment on the line above says it outright: "slocal and
      d-size local bias are not calculated!". `--slocal` is documented as "Invalid if there is
      no control data", and MACS2 honours that.
* `MACS2/IO/cFixWidthTrack.pyx::FWTrackIII.pileup_a_chromosome`
    - `baseline_value` is "a value to be filled for missing values, and will be the minimum
      pileup", so the local lambda is `max(lambda_bg, local)` and the max is taken inside the
      pileup, not afterwards.
* `MACS2/IO/cScoreTrack.pyx::get_pscore` / `MACS2/Poisson.pyx::P_Score_Upper_Tail.get_pscore`
    - `score = -1 * poisson_cdf(observed, expectation, False, True)`, and
      `MACS2/cProb.pyx::log10_poisson_cdf_Q_large_lambda` sums `m` from `k+1` upward. So the
      score is `-log10 P(X > observed)`, a STRICTLY-greater tail, not `P(X >= observed)`.

So the whole no-control rule is three lines:

    lambda_i   = max(lambda_bg, mean of the counts over a centred llocal-wide window)
    observed_i = the bin's own pileup, floored to an int
    score_i    = -log10 P(Poisson(lambda_i) > observed_i)

## With a control

`MACS2/cPeakDetect.pyx::PeakDetect.__call_peaks_w_control` builds a LIST of windows instead of one,
and every one of them is piled up from the CONTROL, not from the treatment:

    ctrl_d_s     = [ self.d ]                                    # line 183
    ctrl_scale_s = [ self.ratio_treat2control ]                  # lines 186-193, `not tocontrol`
    if self.sregion:                                             # line 196
        ctrl_d_s.append( self.sregion )
        ctrl_scale_s.append( float(self.d)/self.sregion*self.ratio_treat2control )
    if self.lregion and self.lregion > self.sregion:             # line 206
        ctrl_d_s.append( self.lregion )
        ctrl_scale_s.append( float(self.d)/self.lregion*self.ratio_treat2control )

with `self.ratio_treat2control = float(treat_sum)/control_sum` (line 159, `treat_sum` and
`control_sum` both being `total * d`, so in single-end mode the ratio is just the read-count
ratio) and `lambda_bg = float(treat_sum)/self.gsize` (line 172). `pileup_a_chromosome` then takes
the elementwise max over all of them with `lambda_bg` as the floor, exactly as in the no-control
case. Three things follow, and all three are the opposite of the no-control path:

* **`slocal` IS live with a control.** The no-control path disables it in so many words; the
  with-control path adds it whenever `--slocal` is non-zero, which it is by default (1000).
* **The d-size window is live too** — `ctrl_d_s` always starts at `[ self.d ]`. It is not a
  "window" in the smoothing sense: it is the control's own pileup at fragment resolution.
* **Every window is read off the control and scaled by the depth ratio.** The treatment's own
  neighbourhood does not enter the lambda at all once a control exists.

So the with-control rule is:

    lambda_i = max( lambda_bg,
                    r * control pileup at bin i,
                    r * mean of the control counts over a centred slocal-wide window,
                    r * mean of the control counts over a centred llocal-wide window )

with `r = treat_reads / control_reads`. The `d`-size term needs no window because the store's
control counts ARE a fragment-extended pileup already: bin `i` holds the number of control
fragments overlapping it, which is what `pileup_a_chromosome(..., [d], ...)` computes. The
`slocal`/`llocal` terms reuse `local_lambda` for the same reason the no-control path does —
`sum(pileup over a W-wide window)/n_bins` equals MACS2's `reads_in_W * d / W`.

Not modelled, because ENCODE's signal step does not use them: `--to-small` (which flips the
scaling onto the treatment and takes `lambda_bg` from the control instead), `--nolambda`, and
paired-end mode.

This module is deliberately dependency-light (numpy + scipy only) and imports nothing from the
rest of `candi`, so it can be dropped onto a cluster and run beside a store without the package.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln
from scipy.stats import poisson

__all__ = [
    "MACS2_LLOCAL",
    "MACS2_SLOCAL",
    "MACS2_GSIZE_HS",
    "local_lambda",
    "log10_upper_tail",
    "pval_from_counts",
    "pval_from_counts_with_control",
]

#: `--llocal` default, `bin/macs2` argparse. The only local window a no-control run uses.
MACS2_LLOCAL = 10000
#: `--slocal` default. MACS2 uses it only WITH a control — `--slocal` is documented as "Invalid if
#: there is no control data", and `__call_peaks_wo_control` honours that.
MACS2_SLOCAL = 1000
#: `MACS2/OptValidator.py::efgsize["hs"]` — the effective human genome size behind `-g hs`.
MACS2_GSIZE_HS = 2.7e9


def local_lambda(counts: np.ndarray, window_bins: int) -> np.ndarray:
    """Centred window mean of `counts`, with the FULL window as the denominator everywhere.

    MACS2's local lambda is `(reads in the llocal window) * d / llocal` — the denominator is
    `llocal` whatever the window overlaps, so a position near a chromosome end gets a genuinely
    smaller local lambda rather than a renormalised one. This keeps that behaviour: the divisor
    is `window_bins` even where the window runs off the end.

    The window is centred the way `pileup_a_chromosome(..., directional=False)` centres it:
    `d/2` to the left and `d - d/2` to the right.
    """
    if window_bins < 1:
        raise ValueError(f"window_bins must be >= 1, got {window_bins}")
    n = counts.shape[0]
    left = window_bins // 2
    right = window_bins - left
    cs = np.empty(n + 1, dtype=np.float64)
    cs[0] = 0.0
    np.cumsum(counts, dtype=np.float64, out=cs[1:])
    idx = np.arange(n)
    hi = np.minimum(idx + right, n)
    lo = np.maximum(idx - left, 0)
    return (cs[hi] - cs[lo]) / float(window_bins)


def log10_upper_tail(k: np.ndarray, lam: np.ndarray, *, series_terms: int = 128) -> np.ndarray:
    """`-log10 P(X > k)` for `X ~ Poisson(lam)`, elementwise, exact well past float64 underflow.

    `scipy.stats.poisson.sf(k, lam)` IS `P(X > k)` and carries the double path. It flushes to
    zero below ~1e-308, and the corpus reaches `-log10 p` of 17,731, so those bins are redone in
    log space with the same series MACS2 uses in
    `cProb.pyx::log10_poisson_cdf_Q_large_lambda`:

        log P(X > k) = -lam + log( sum_{m=k+1}^inf exp(m*log(lam) - log(m!)) )

    Underflow only happens when `k >> lam`, where the term ratio `lam/(m+1)` is far below 1, so
    a fixed number of terms is plenty; `series_terms` is a ceiling, not a tuned length.
    """
    k = np.asarray(k)
    lam = np.asarray(lam, dtype=np.float64)
    sf = poisson.sf(k, lam)
    with np.errstate(divide="ignore"):
        out = -np.log10(sf)
    bad = ~np.isfinite(out)
    if not bad.any():
        return out
    kb = k[bad].astype(np.float64)
    lb = lam[bad]
    log_lam = np.log(lb)
    m0 = kb + 1.0
    # first term, in nats
    t0 = m0 * log_lam - gammaln(m0 + 1.0) - lb
    # sum_{j>=1} prod_{i=1..j} lam/(m0+i), accumulated relative to the first term
    ratio_prod = np.ones_like(t0)
    tail = np.zeros_like(t0)
    for j in range(1, series_terms + 1):
        ratio_prod = ratio_prod * (lb / (m0 + j))
        tail += ratio_prod
        if np.all(ratio_prod < 1e-18):
            break
    out[bad] = -(t0 + np.log1p(tail)) / np.log(10.0)
    return out


def pval_from_counts(
    counts: np.ndarray,
    *,
    lambda_bg: float,
    llocal: int = MACS2_LLOCAL,
    resolution: int = 25,
) -> np.ndarray:
    """`-log10 p` per bin for one chromosome of binned counts, MACS2's no-control rule.

    `counts` is one chromosome's DSF-1 bin counts. `lambda_bg` is the genome-wide background in
    the SAME units as one bin of `counts` — see `genome_lambda_bg`. `llocal` is in base pairs
    and is converted to bins with `resolution`; MACS2 refuses a window smaller than the fragment,
    and here the fragment is one bin, so `llocal` must cover at least one bin.
    """
    if lambda_bg <= 0:
        raise ValueError(f"lambda_bg must be > 0 (MACS2 asserts the same), got {lambda_bg}")
    window_bins = int(round(llocal / resolution))
    if window_bins < 1:
        raise ValueError(f"llocal={llocal} is under one {resolution} bp bin")
    lam = local_lambda(counts, window_bins)
    np.maximum(lam, lambda_bg, out=lam)
    obs = np.asarray(counts)
    if not np.issubdtype(obs.dtype, np.integer):
        obs = np.floor(obs).astype(np.int64)      # MACS2 does `int(array1[i])`
    return log10_upper_tail(obs, lam).astype(np.float32)


def pval_from_counts_with_control(
    counts: np.ndarray,
    control_counts: np.ndarray,
    *,
    lambda_bg: float,
    ratio_treat2control: float,
    slocal: int = MACS2_SLOCAL,
    llocal: int = MACS2_LLOCAL,
    resolution: int = 25,
) -> np.ndarray:
    """`-log10 p` per bin for one chromosome, MACS2's rule for a run that HAS a control.

    `counts` is the treatment's DSF-1 bin counts and `control_counts` the control's, on the same
    bins and the same chromosome. `lambda_bg` is the genome-wide background in the units of one
    bin, computed from the TREATMENT — `genome_lambda_bg(counts.sum())` — because MACS2 takes
    `lambda_bg = treat_sum/gsize` whenever it scales the control up to the treatment, which is
    every run that does not pass `--to-small`.

    `ratio_treat2control` is MACS2's `treat_sum/control_sum`, which in single-end mode is the
    read-count ratio `treat_reads/control_reads`. It has no default on purpose: `counts.sum() /
    control_counts.sum()` is only the same number when the two libraries were extended to the same
    fragment length, and a silently wrong depth ratio rescales every lambda in the chromosome.

    `slocal` and `llocal` are in base pairs. Either may be `0` to switch that window off, which is
    how MACS2 reads a `--slocal 0` / `--llocal 0`; `llocal` is used only when it is strictly larger
    than `slocal`, again following the source. MACS2 asserts `d <= slocal <= llocal`, and here the
    fragment is one bin, so both windows must cover at least one bin.
    """
    if lambda_bg <= 0:
        raise ValueError(f"lambda_bg must be > 0 (MACS2 asserts the same), got {lambda_bg}")
    if ratio_treat2control <= 0:
        raise ValueError(f"ratio_treat2control must be > 0, got {ratio_treat2control}")
    counts = np.asarray(counts)
    control_counts = np.asarray(control_counts)
    if counts.shape != control_counts.shape:
        raise ValueError(
            f"treatment and control must be binned the same: {counts.shape} vs {control_counts.shape}"
        )
    if slocal and llocal and slocal > llocal:
        raise ValueError(f"llocal can't be smaller than slocal (MACS2 asserts it): {llocal} < {slocal}")

    ctrl = control_counts.astype(np.float64)
    # ctrl_d_s[0] = d: the control's own pileup, which the store's control counts already are.
    lam = ctrl * ratio_treat2control
    np.maximum(lam, lambda_bg, out=lam)
    for window in (slocal, llocal if llocal > slocal else 0):
        if not window:
            continue
        window_bins = int(round(window / resolution))
        if window_bins < 1:
            raise ValueError(f"a {window} bp local window is under one {resolution} bp bin")
        np.maximum(lam, local_lambda(ctrl, window_bins) * ratio_treat2control, out=lam)

    obs = counts
    if not np.issubdtype(obs.dtype, np.integer):
        obs = np.floor(obs).astype(np.int64)      # MACS2 does `int(array1[i])`
    return log10_upper_tail(obs, lam).astype(np.float32)


def genome_lambda_bg(total_counts: float, *, gsize: float = MACS2_GSIZE_HS, resolution: int = 25) -> float:
    """MACS2's `lambda_bg`, in the units of one bin of the store's counts.

    MACS2 writes `lambda_bg = d * treat_total / gsize`: total pileup mass over the effective
    genome, i.e. the pileup height a position would have if every read were spread uniformly.
    The store's counts are a coverage pileup already — bin `i` holds the number of reads
    overlapping it — so the same quantity is `(sum of counts) * resolution / gsize`, and the
    store's own effective fragment length takes the place of `d`. `gsize` stays MACS2's
    EFFECTIVE genome size rather than the assembly length, which is what `-g hs` means.
    """
    return float(total_counts) * float(resolution) / float(gsize)
