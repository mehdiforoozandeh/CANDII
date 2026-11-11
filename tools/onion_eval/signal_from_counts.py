"""Signal track from a predicted NB, per assay class (decision D9b, plan chunk X2).

    signal_track(mu, n, assay_class, control_counts | None, covariates) -> (signal float32 [L], lambda_source)

`mu`, `n` are the NB mean and size per 25 bp bin of ONE contiguous stretch of one chromosome (a whole
chromosome, or a window with enough flank for the local windows: 10 kb for MACS2, 75 bp for hotspot2).
Counts follow the onion counting rule (`tools/t112/bin25.py::counts_from_tagalign`): a read adds 1 to
every bin it overlaps. Each route is the assay's OWN ENCODE pipeline step, read from the tagged source
(`cruxvault/results/t179/ROUTES.md` has the file, tag and line of every rule below).

Routes
------
* ChIP-seq (`broad_histone`, `narrow_histone`): chip-seq-pipeline2 v2.2.2
  `encode_task_macs2_signal_track_chip.py`: `macs2 callpeak -t TA -c CTL --nomodel --shift 0
  --extsize FRAGLEN --keep-dup all -B --SPMR`, then `bdgcmp -m ppois -S <treatment lines / 1e6>`.
  With `--SPMR` and that `-S`, the score is on the raw treatment pileup against the control lambda
  scaled to the treatment depth, whichever way `--scale-to small` scaled the two inside callpeak
  (MACS2 v2.2.4 `CallPeakUnit.pyx` SPMR denominator, `bdgcmp_cmd.py` `-S`). That is exactly
  `candi.store.pval_from_counts.pval_from_counts_with_control`, which this route reuses:
  lambda = max(lambda_bg, r*ctl_d, r*ctl_1k, r*ctl_10k), r = N_treat / N_control,
  lambda_bg = d * N_treat / gsize. lambda_source = `observed_control`, or `predicted_control` when
  `covariates["control_is_predicted"]` is true (Caco-2, SJSA1: the caller passes the predicted control
  mean as `control_counts`).
* ATAC-seq (`accessibility_atac`): atac-seq-pipeline v2.2.3 `encode_task_macs2_signal_track_atac.py`:
  `macs2 callpeak -t TA -f BED -g hs --shift -75 --extsize 150 --nomodel -B --SPMR --keep-dup all`,
  no control, then `bdgcmp -m ppois -S <lines / 1e6>`. MACS2 v2.2.4 `PeakDetect.pyx::
  __call_peaks_wo_control`: lambda = max(lambda_bg, llocal window of the TREATMENT), with
  lambda_bg = 150 * N / gsize; slocal and the d-size window are not used. That is
  `candi.store.pval_from_counts.pval_from_counts` (the no-control rule) with d = 150, not the ChIP
  route. The 10 kb window reads the predicted mean (PI decision 2026-10-11, t125 formula h).
  lambda_source = `local_background`.
* DNase-seq (`accessibility_dnase`): dnase-seq-pipeline v3.0.0, hotspot2 v2.1 `density-peaks.bash`
  (cleavage counts summed over a 150 bp window sliding every 20 bp) and the pipeline's
  `normalize_density_starch` (density / reads in the nuclear BAM * 1e6). No p-value and no lambda;
  lambda_source = `hotspot2_density`. Not modelled from binned counts: hotspot2's "bad spot" filter and
  its mappable-region restriction of cut sites.

Counts to the caller's units (PI decision 2026-10-11, `tools/t125/hmap/caller.py`): a read of length
rl touches (rl + 24) / 25 bins on average, so the tags behind a binned count are count * 25 / (rl + 24)
and the MACS2 pileup (tags extended to d) is count * d / (rl + 24). The bin's own pileup is the
"bin" variant of that decision.

From the NB to one signal value: the MACS2 score is a function of the bin's own count k, so the track
is its NB expectation, E_NB(n, mu)[-log10 P(Poisson(lambda) > floor(pileup(k)))], summed over k between
the NB's 1e-10 and 1 - 1e-10 quantiles. lambda is deterministic: it comes from the control, or (no
control) from the predicted mean. The hotspot2 density is linear in the counts, so its expectation is
the density of `mu`.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import nbinom

from candi.store.pval_from_counts import (
    MACS2_GSIZE_HS,
    MACS2_LLOCAL,
    MACS2_SLOCAL,
    local_lambda,
    log10_upper_tail,
)

__all__ = [
    "LAMBDA_SOURCES",
    "ATAC_SMOOTH_WIN",
    "HOTSPOT2_WINDOW_BP",
    "HOTSPOT2_SCALE",
    "route_of",
    "pileup_from_counts",
    "macs2_lambda",
    "nb_expected_neglog10p",
    "hotspot2_normalized_density",
    "signal_track",
]

LAMBDA_SOURCES = ("observed_control", "predicted_control", "local_background", "hotspot2_density")

BIN = 25
SPAN_EXTRA = BIN - 1           # a read of length rl touches (rl + 24) / 25 bins on average
#: atac.wdl v2.2.3 `Int smooth_win = 150`; the signal task sets `--extsize smooth_win`, `--shift -75`.
ATAC_SMOOTH_WIN = 150
#: hotspot2 v2.1 `density-peaks.bash`: `bins=150`, `step=20`.
HOTSPOT2_WINDOW_BP = 150
#: dnase-seq-pipeline v3.0.0 `normalize_density_starch.wdl`: `Int scale = 1000000`.
HOTSPOT2_SCALE = 1_000_000
#: NB mass left out at each end of the enumeration.
NB_TAIL = 1e-10
_MAX_CELLS = 4_000_000

_CHIP_CLASSES = ("broad_histone", "narrow_histone")
_ATAC = ("accessibility_atac", "ATAC-seq")
_DNASE = ("accessibility_dnase", "DNase-seq")


def route_of(assay_class: str) -> str:
    """`chip`, `atac` or `dnase` for an `assay_class` value (or an assay name of the panel)."""
    if assay_class in _ATAC:
        return "atac"
    if assay_class in _DNASE:
        return "dnase"
    if assay_class in _CHIP_CLASSES or (assay_class[:2] in ("H2", "H3", "H4")):
        return "chip"
    if assay_class in ("control", "chipseq-control"):
        raise ValueError("a ChIP control has no signal track of its own")
    raise ValueError(f"no signal route for assay class {assay_class!r}")


def pileup_from_counts(counts, d: float, read_length: float) -> np.ndarray:
    """MACS2 pileup (tags extended to `d` bp) from read-span bin counts: count * d / (rl + 24)."""
    if not read_length > 0 or not d > 0:
        raise ValueError(f"read_length and d must be > 0, got {read_length}, {d}")
    return np.asarray(counts, dtype=np.float64) * float(d) / (float(read_length) + SPAN_EXTRA)


def macs2_lambda(*, lambda_bg: float, treat_pileup=None, control_pileup=None,
                 ratio_treat2control: float | None = None, slocal: int = MACS2_SLOCAL,
                 llocal: int = MACS2_LLOCAL, resolution: int = BIN) -> np.ndarray:
    """MACS2's local lambda per bin, in treatment pileup units.

    With `control_pileup`: the rule of `pval_from_counts_with_control` (max of lambda_bg, the
    control's own pileup and its slocal / llocal window means, all times r). Without: the rule of
    `pval_from_counts` (max of lambda_bg and the llocal window mean of `treat_pileup`). Both are
    restated here because those functions return only the score; a test holds each restatement
    equal to the function it restates.
    """
    if not lambda_bg > 0:
        raise ValueError(f"lambda_bg must be > 0, got {lambda_bg}")
    if control_pileup is None:
        if treat_pileup is None:
            raise ValueError("need the treatment pileup when there is no control")
        lam = local_lambda(np.asarray(treat_pileup, dtype=np.float64), int(round(llocal / resolution)))
        return np.maximum(lam, lambda_bg)
    if not (ratio_treat2control and ratio_treat2control > 0):
        raise ValueError(f"ratio_treat2control must be > 0, got {ratio_treat2control}")
    ctrl = np.asarray(control_pileup, dtype=np.float64)
    lam = np.maximum(ctrl * ratio_treat2control, lambda_bg)
    for window in (slocal, llocal if llocal > slocal else 0):
        if window:
            lam = np.maximum(lam, local_lambda(ctrl, int(round(window / resolution))) * ratio_treat2control)
    return lam


def nb_expected_neglog10p(mu, n, scale: float, lam, *, tail: float = NB_TAIL) -> np.ndarray:
    """E over k ~ NB(n, mu) of -log10 P(Poisson(lam) > floor(k * scale)), per bin, float64.

    `scale` turns a count into pileup units (d / (rl + 24)). k runs over the NB's [tail, 1 - tail]
    quantile range; the sum is divided by the mass kept.
    """
    mu = np.asarray(mu, dtype=np.float64)
    n = np.broadcast_to(np.asarray(n, dtype=np.float64), mu.shape)
    lam = np.broadcast_to(np.asarray(lam, dtype=np.float64), mu.shape)
    if mu.ndim != 1:
        raise ValueError(f"mu must be 1-D [L], got shape {mu.shape}")
    if not (np.all(mu > 0) and np.all(n > 0)):
        raise ValueError("mu and n must be > 0 everywhere (contract: NB outputs are positive)")
    p = n / (n + mu)
    lo = nbinom.ppf(tail, n, p).astype(np.int64)
    hi = nbinom.ppf(1.0 - tail, n, p).astype(np.int64)
    width = hi - lo + 1
    out = np.empty(mu.shape, dtype=np.float64)
    order = np.argsort(width, kind="stable")
    i = 0
    while i < order.size:
        # widest bin of the chunk is its last one (sorted), so grow the chunk while it fits
        j = i + 1
        while j < order.size and (j + 1 - i) * width[order[j]] <= _MAX_CELLS:
            j += 1
        idx = order[i:j]
        K = int(width[idx].max())
        k = lo[idx, None] + np.arange(K)[None, :]
        valid = k <= hi[idx, None]
        pmf = np.where(valid, nbinom.pmf(k, n[idx, None], p[idx, None]), 0.0)
        obs = np.floor(k * float(scale)).astype(np.int64)
        score = log10_upper_tail(obs, np.broadcast_to(lam[idx, None], k.shape).copy())
        out[idx] = (pmf * np.where(valid, score, 0.0)).sum(axis=1) / pmf.sum(axis=1)
        i = j
    return out


def hotspot2_normalized_density(mu, *, read_length: float, depth_reads: float) -> np.ndarray:
    """hotspot2 density normalized per million reads, per 25 bp bin.

    Cleavages in the 150 bp window centred on the bin: the window covers bins i-2..i+2 wholly and half
    of bins i-3 and i+3; one cleavage per read, so tags = counts * 25 / (rl + 24). Then
    * 1e6 / depth_reads (`normalize_bed_values`). Zero outside the stretch.
    """
    if not depth_reads > 0:
        raise ValueError(f"depth_reads must be > 0, got {depth_reads}")
    x = np.asarray(mu, dtype=np.float64) * BIN / (float(read_length) + SPAN_EXTRA)
    half = HOTSPOT2_WINDOW_BP / BIN / 2.0               # 3 bins each side, the outer one half-weighted
    reach = int(np.ceil(half - 0.5))
    w = np.ones(2 * reach + 1)
    w[0] = w[-1] = half - (reach - 0.5)
    tags = np.convolve(x, w, mode="same")
    return tags * HOTSPOT2_SCALE / float(depth_reads)


def _need(cov: dict, *keys):
    missing = [k for k in keys if cov.get(k) in (None, "")]
    if missing:
        raise KeyError(f"covariates lack {missing}")
    return [float(cov[k]) for k in keys]


def signal_track(mu, n, assay_class: str, control_counts, covariates: dict,
                 *, gsize: float = MACS2_GSIZE_HS) -> tuple[np.ndarray, str]:
    """The assay's pipeline signal derived from the predicted NB, and its lambda source.

    `covariates` keys (the `measurements.tsv` names): all routes `depth_reads` (treatment tagAlign
    lines, or BAM reads for DNase) and `read_length`; ChIP also `frag_len`, `control_depth_reads`,
    `control_read_length`, and optionally `control_is_predicted` (default False).
    """
    route = route_of(assay_class)
    mu = np.asarray(mu, dtype=np.float64)
    if route == "dnase":
        depth, rl = _need(covariates, "depth_reads", "read_length")
        sig = hotspot2_normalized_density(mu, read_length=rl, depth_reads=depth)
        return sig.astype(np.float32), "hotspot2_density"

    if route == "atac":
        if control_counts is not None:
            raise ValueError("ATAC-seq has no control; its pipeline runs MACS2 without one")
        depth, rl = _need(covariates, "depth_reads", "read_length")
        d = float(ATAC_SMOOTH_WIN)
        lam = macs2_lambda(lambda_bg=d * depth / gsize, treat_pileup=pileup_from_counts(mu, d, rl))
        sig = nb_expected_neglog10p(mu, n, d / (rl + SPAN_EXTRA), lam)
        return sig.astype(np.float32), "local_background"

    # ChIP-seq
    if control_counts is None:
        raise ValueError("ChIP-seq needs a control: the observed one, or a predicted one "
                         "(set covariates['control_is_predicted'])")
    depth, rl, d, cdepth, crl = _need(covariates, "depth_reads", "read_length", "frag_len",
                                      "control_depth_reads", "control_read_length")
    control_counts = np.asarray(control_counts, dtype=np.float64)
    if control_counts.shape != mu.shape:
        raise ValueError(f"control and treatment must be binned the same: {control_counts.shape} vs {mu.shape}")
    lam = macs2_lambda(lambda_bg=d * depth / gsize,
                       control_pileup=pileup_from_counts(control_counts, d, crl),
                       ratio_treat2control=depth / cdepth)
    sig = nb_expected_neglog10p(mu, n, d / (rl + SPAN_EXTRA), lam)
    source = "predicted_control" if bool(covariates.get("control_is_predicted", False)) else "observed_control"
    return sig.astype(np.float32), source
