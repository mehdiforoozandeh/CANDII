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
* DNase-seq (`accessibility_dnase`), selectable by `dnase_route` (DNASE_ROUTES; default
  DNASE_ROUTE_DEFAULT = `macs2_no_control`, PI ruling 19 of 2026-11-11: for EIC signal scoring the
  DNase-seq route is MACS2 without control, because the EIC DNase-seq truth is a MACS2 -log10 p
  track; `hotspot2_density` stays available as the native DNase-seq signal view):
  - `macs2_no_control` (default): the ATAC-seq rule above applied to DNase-seq counts (MACS2
    without control, `--shift -75 --extsize 150`, lambda = max(lambda_bg, llocal window of the
    treatment)); lambda_source = `local_background`.
  - `hotspot2_density`: dnase-seq-pipeline v3.0.0, hotspot2 v2.1 `density-peaks.bash` (cleavage
    counts summed over a 150 bp window sliding every 20 bp) and the pipeline's
    `normalize_density_starch` (density / reads in the nuclear BAM * 1e6). No p-value and no lambda;
    lambda_source = `hotspot2_density`. Not modelled from binned counts: hotspot2's "bad spot"
    filter and its mappable-region restriction of cut sites.

Counts to the caller's units (PI decision 2026-10-11, `tools/t125/hmap/caller.py`): a read of length
rl touches (rl + 24) / 25 bins on average, so the tags behind a binned count are count * 25 / (rl + 24)
and the MACS2 pileup (tags extended to d) is count * d / (rl + 24). The bin's own pileup is the
"bin" variant of that decision.

From the NB to one signal value (`mode`, PI ruling 20 of 2026-11-11; SIGNAL_MODES):
  - `expectation` (default, the headline): the MACS2 score is a function of the bin's own count k,
    so the track is its NB expectation, E_NB(n, mu)[-log10 P(Poisson(lambda) > floor(pileup(k)))],
    summed over k between the NB's 1e-10 and 1 - 1e-10 quantiles.
  - `plug_in` (reported beside it): the score of the NB mean, -log10 P(Poisson(lambda) >
    floor(pileup(mu))). It ignores the NB's spread, so it differs from the expectation by the
    Jensen gap of the score over the NB.
lambda is deterministic in both modes: it comes from the control, or (no control) from the
predicted mean. The hotspot2 density is linear in the counts, so its expectation is the density of
`mu` and the two modes give the same track.

Pileup form (`pileup`, PILEUP_FORMS; PI ruling 2026-11-17 "Adopt the box spread"; default `box_d`).
A 25 bp read-span count is not a MACS2 pileup: the pipeline extends each read to its fragment (d bp,
on the side set by the strand) and scores every base pair. `box_d` spreads each bin's count over a
centred box of d bp (`box_spread`; ATAC-seq / DNase-seq d = 150) before the score: the treatment
and, for ChIP-seq, the control (all lambda windows read the spread control; the no-control llocal
window reads the unspread mean, a window mean that the box does not change away from the edges).
At a region end the box is cut and renormalized. On observed counts this raised the median Pearson
with the pipeline's -log10 p track from 0.831 to 0.921 (41 ChIP-seq validation targets) and the
500 bp peak-tile F1 at -log10 p >= 2 from 0.573 to 0.782; it was chosen among 4 variants on 42
validation targets (`cruxvault/results/closeout_E2/FORMULA_EXTENSION.md`). The strand and the bp
positions are lost at binning, so no 25 bp formula reproduces the pipeline exactly. `inbin` keeps
the bin's own count (the rule before the ruling) and stays reachable by that value.
Model side of `box_d`: the predicted NB is per bin and its joint law is not modelled. The spread
pileup's NB has the box mean of the means and the box mean of the sizes, i.e. the bins under one
box share one Gamma rate (fully correlated over-dispersion, and the Poisson part taken as one bin's).
This is the upper bound of the spread pileup's variance; independent bins (variance
sum w_k^2 (mu_k + mu_k^2 / n_k)) are the lower bound. The plug-in mode needs no such assumption
(the score of the box mean). The predicted-control route spreads the predicted control mean by the
same box.

Scorer adapter: `signal_from_nb(n, mu, row, *, ds, regions, rb, predicted_control=None,
dnase_route=..., mode="expectation") -> (signal, lambda_source)`. The scorer's vectors join the eval regions end to end
(region i = bins [i*rb, (i+1)*rb)), so the adapter calls `signal_track` once per region. The observed
control is read from the row's `control_measurement_id` through the dataset; without one, the
caller's `predicted_control` (the model's control mean over the same bins) is used and labelled
`predicted_control`, with the treatment's own depth and read length as the control covariates
(the scorer asks for the predicted control at the treatment's C'). Region-edge truncation: each
region is one stretch with no flank, so the MACS2 local windows (llocal = 10 kb = 400 bins, slocal
= 1 kb) and the hotspot2 150 bp window are truncated at both ends of every 2^14-bin region; about
2 x 200 of 16384 bins per region (2.4%) see a shortened llocal window. Logged, not corrected.

Fragment length of a single-end track (`load_fraglen_table`, `with_fraglen`): the frozen tables leave
`frag_len` empty for most single-end ChIP-seq tracks, so the ChIP route has no `d`. A caller may give
a side table of estimated fragment lengths (`qc_fill.py collect --fraglen-out`), first found of:
the fragment length the pipeline used for the measurement's own `signal.h5` (its provenance.json),
the rounded mean of its replicates' frozen `frag_len` (a pooled track: chip.wdl's rule), the strand
cross-correlation estimate of the pipeline's xcor task on the measurement's own kept reads.
`with_fraglen(row, table)` returns the row unchanged when `frag_len` is filled (source `table`; a
table value is never replaced), the row with the estimate when it is empty and the side table has
a plausible estimate (source `estimated`), else the row unchanged (source `none`; the ChIP route
then stays undefined with "covariates lack ['frag_len']"). Plausible: FRAGLEN_PLAUSIBLE_BP[0] <=
estimate <= FRAGLEN_PLAUSIBLE_BP[1], and for an xcor estimate its QualityTag is not -2. Only the treatment's
`frag_len` is read: the control pileup is extended by the treatment's `d` (MACS2 `--extsize`
applies to both), so a control row's `frag_len` never matters.
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
    "signal_from_nb",
    "DNASE_ROUTES",
    "DNASE_ROUTE_DEFAULT",
    "SIGNAL_MODES",
    "PILEUP_FORMS",
    "PILEUP_DEFAULT",
    "box_weights",
    "box_spread",
    "plug_in_neglog10p",
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
#: DNase-seq routes; the default is MACS2 without control (PI ruling 19, 2026-11-11).
DNASE_ROUTES = ("hotspot2_density", "macs2_no_control")
DNASE_ROUTE_DEFAULT = "macs2_no_control"
#: NB -> signal: expectation of the score (headline) or score of the NB mean (PI ruling 20).
SIGNAL_MODES = ("expectation", "plug_in")
#: pileup form of the signal routes (PI ruling 2026-11-17, "Adopt the box spread"): `box_d` spreads
#: each bin's count over a centred box of the fragment length d (ATAC-seq / DNase-seq: d = 150);
#: `inbin` is the bin's own count only (the rule before the ruling). Recorded as `signal_pileup`.
PILEUP_FORMS = ("box_d", "inbin")
PILEUP_DEFAULT = "box_d"

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


def plug_in_neglog10p(mu, scale: float, lam) -> np.ndarray:
    """-log10 P(Poisson(lam) > floor(mu * scale)) per bin: the MACS2 score of the NB mean (the
    `plug_in` mode of PI ruling 20), float64."""
    mu = np.asarray(mu, dtype=np.float64)
    if mu.ndim != 1:
        raise ValueError(f"mu must be 1-D [L], got shape {mu.shape}")
    if not np.all(mu > 0):
        raise ValueError("mu must be > 0 everywhere (contract: NB outputs are positive)")
    lam = np.broadcast_to(np.asarray(lam, dtype=np.float64), mu.shape).copy()
    obs = np.floor(mu * float(scale)).astype(np.int64)
    return log10_upper_tail(obs, lam)


def _score(mu, n, scale, lam, mode):
    if mode == "plug_in":
        return plug_in_neglog10p(mu, scale, lam)
    return nb_expected_neglog10p(mu, n, scale, lam)


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


#: plausible estimated fragment length, bp (inclusive); outside it the estimate is not used
FRAGLEN_PLAUSIBLE_BP = (80, 400)
#: spp QualityTag of "very low" cross-correlation quality; such an estimate is not used
FRAGLEN_BAD_QUALITY_TAG = "-2"
FRAGLEN_SOURCES = ("table", "estimated", "none")


def fraglen_drop_reason(est: dict) -> str:
    """'' when a side-table row's estimate may be used, else why not."""
    v = (est.get("frag_len_estimated") or "").strip()
    if not v:
        return "no estimate"
    try:
        d = float(v)
    except ValueError:
        return f"estimate {v!r} is not a number"
    lo, hi = FRAGLEN_PLAUSIBLE_BP
    if not lo <= d <= hi:
        return f"implausible estimate {v} bp (outside {lo}-{hi} bp)"
    if ((est.get("estimate_kind") or "xcor") == "xcor"
            and (est.get("quality_tag") or "").strip() == FRAGLEN_BAD_QUALITY_TAG):
        return f"xcor quality tag {FRAGLEN_BAD_QUALITY_TAG} (very low)"
    return ""


def load_fraglen_table(path) -> dict:
    """measurement_id -> side-table row (`qc_fill.py collect --fraglen-out`), with `drop_reason`."""
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if rows and ("measurement_id" not in rows[0] or "frag_len_estimated" not in rows[0]):
        raise ValueError(f"{path}: needs columns measurement_id and frag_len_estimated")
    return {r["measurement_id"]: dict(r, drop_reason=fraglen_drop_reason(r)) for r in rows}


def with_fraglen(row: dict, table) -> tuple[dict, str, str]:
    """(row to use, fraglen source, reason). The row is returned unchanged (the same object) unless
    its `frag_len` is empty and `table` holds a usable estimate; then a copy carries the estimate."""
    if (row.get("frag_len") or "").strip():
        return row, "table", ""
    if table is None:
        return row, "none", "frag_len empty and no fragment-length table given"
    est = table.get(row.get("measurement_id"))
    if est is None:
        return row, "none", "frag_len empty and no estimate in the fragment-length table"
    if est["drop_reason"]:
        return row, "none", f"frag_len empty; {est['drop_reason']}"
    return (dict(row, frag_len=est["frag_len_estimated"].strip()), "estimated",
            f"estimate_kind {est.get('estimate_kind') or 'unknown'}")


def _need(cov: dict, *keys):
    missing = [k for k in keys if cov.get(k) in (None, "")]
    if missing:
        raise KeyError(f"covariates lack {missing}")
    return [float(cov[k]) for k in keys]


def box_weights(width_bp: float, bin_bp: int = BIN) -> np.ndarray:
    """Weights over bins of a box of `width_bp` centred on a bin (bin i covers [i - 0.5, i + 0.5]
    in bin units); they sum to 1. A width of one bin or less gives the identity [1.0]; a width that
    is not a multiple of 25 bp gives the two end bins a fractional weight."""
    half = float(width_bp) / bin_bp / 2.0
    if half <= 0.5:
        return np.ones(1)
    reach = int(np.ceil(half - 0.5))
    k = np.arange(-reach, reach + 1, dtype=np.float64)
    w = np.clip(np.minimum(k + 0.5, half) - np.maximum(k - 0.5, -half), 0.0, None)
    return w / w.sum()


def box_spread(x, d: float) -> np.ndarray:
    """`x` (one contiguous stretch) averaged over the centred box of `d` bp (`box_weights`). At the
    two ends of the stretch the box is cut and renormalized to the weights inside it, so a flat
    track stays flat. d <= 25 bp returns `x` unchanged (as float64)."""
    x = np.asarray(x, dtype=np.float64)
    w = box_weights(d)
    if w.size == 1:
        return x.copy()
    num = np.convolve(x, w, mode="same")
    den = np.convolve(np.ones_like(x), w, mode="same")
    return num / den


def _spread(mu, n, ctl, d, pileup):
    """The box spread of the predicted mean, the NB size and the control, or nothing (`inbin`).
    Model side (PI ruling 2026-11-17): the pileup of a bin is the box mean of the NB means; its NB
    has that mean and the box mean of the sizes, i.e. the bins under one box share one Gamma rate
    (fully correlated over-dispersion). This keeps the NB machinery of one bin and is the upper
    bound of the pileup's variance; independent bins (variance sum w^2 (mu + mu^2 / n)) are the
    lower bound. On observed counts (n = None, plug-in) only the counts are spread."""
    if pileup not in PILEUP_FORMS:
        raise ValueError(f"pileup {pileup!r} not in {PILEUP_FORMS}")
    if pileup == "inbin":
        return mu, n, ctl
    mu = np.asarray(mu, dtype=np.float64)
    if n is not None:
        n = np.broadcast_to(np.asarray(n, dtype=np.float64), mu.shape)
    return (box_spread(mu, d), None if n is None else box_spread(n, d),
            None if ctl is None else box_spread(ctl, d))


def signal_track(mu, n, assay_class: str, control_counts, covariates: dict,
                 *, gsize: float = MACS2_GSIZE_HS,
                 dnase_route: str = DNASE_ROUTE_DEFAULT,
                 mode: str = "expectation",
                 pileup: str = PILEUP_DEFAULT) -> tuple[np.ndarray, str]:
    """The assay's pipeline signal derived from the predicted NB, and its lambda source.

    `covariates` keys (the `measurements.tsv` names): all routes `depth_reads` (treatment tagAlign
    lines, or BAM reads for DNase) and `read_length`; ChIP also `frag_len`, `control_depth_reads`,
    `control_read_length`, and optionally `control_is_predicted` (default False). `dnase_route`
    picks the DNase-seq route (DNASE_ROUTES). `mode` (SIGNAL_MODES): `expectation` of the MACS2
    score under the NB (headline) or `plug_in`, the score of the NB mean; `n` is unused in
    `plug_in` mode. `pileup` (PILEUP_FORMS): `box_d` spreads the treatment (mean and size) and the
    control over the centred box of d before the score (`_spread`); `inbin` does not. The
    hotspot2 route has no pileup and ignores it.
    """
    if dnase_route not in DNASE_ROUTES:
        raise ValueError(f"dnase_route {dnase_route!r} not in {DNASE_ROUTES}")
    if mode not in SIGNAL_MODES:
        raise ValueError(f"mode {mode!r} not in {SIGNAL_MODES}")
    if pileup not in PILEUP_FORMS:
        raise ValueError(f"pileup {pileup!r} not in {PILEUP_FORMS}")
    route = route_of(assay_class)
    mu = np.asarray(mu, dtype=np.float64)
    if route == "dnase" and dnase_route == "hotspot2_density":
        depth, rl = _need(covariates, "depth_reads", "read_length")
        sig = hotspot2_normalized_density(mu, read_length=rl, depth_reads=depth)
        return sig.astype(np.float32), "hotspot2_density"

    if route in ("atac", "dnase"):          # dnase here only with dnase_route = macs2_no_control
        if control_counts is not None:
            raise ValueError(f"{assay_class} has no control; its MACS2 route runs without one")
        depth, rl = _need(covariates, "depth_reads", "read_length")
        d = float(ATAC_SMOOTH_WIN)
        lam = macs2_lambda(lambda_bg=d * depth / gsize, treat_pileup=pileup_from_counts(mu, d, rl))
        mu_s, n_s, _ = _spread(mu, n, None, d, pileup)     # llocal reads the unspread mean
        sig = _score(mu_s, n_s, d / (rl + SPAN_EXTRA), lam, mode)
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
    mu_s, n_s, ctl_s = _spread(mu, n, control_counts, d, pileup)
    lam = macs2_lambda(lambda_bg=d * depth / gsize,
                       control_pileup=pileup_from_counts(ctl_s, d, crl),
                       ratio_treat2control=depth / cdepth)
    sig = _score(mu_s, n_s, d / (rl + SPAN_EXTRA), lam, mode)
    source = "predicted_control" if bool(covariates.get("control_is_predicted", False)) else "observed_control"
    return sig.astype(np.float32), source


def signal_from_nb(n, mu, row: dict, *, ds, regions, rb: int, predicted_control=None,
                   dnase_route: str = DNASE_ROUTE_DEFAULT,
                   mode: str = "expectation",
                   pileup: str = PILEUP_DEFAULT) -> tuple[np.ndarray, str]:
    """Signal of one target over the scorer's concatenated eval regions, and its lambda source.
    `pileup`: `box_d` (default) or `inbin`, as in `signal_track`; each region is spread alone.
    `mode`: `expectation` (headline) or `plug_in` (PI ruling 20), as in `signal_track`.

    `n`, `mu` [len(regions) * rb]: the predicted NB, region i at bins [i*rb, (i+1)*rb). `row`: the
    target's `measurements.tsv` row. `ds`: the scorer's `OnionDataset` (reads the observed control).
    `predicted_control` [len(regions) * rb] or None: the model's control mean on the same bins, used
    only when the row has no `control_measurement_id` (ChIP-seq). See the module docstring for the
    region-edge truncation of the local windows.
    """
    n = np.asarray(n, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    if mu.shape != (len(regions) * int(rb),) or n.shape != mu.shape:
        raise ValueError(f"n, mu must be [{len(regions)} x {rb}], got {n.shape}, {mu.shape}")
    route = route_of(row["assay_class"] or row["assay"])
    ctl_id = (row.get("control_measurement_id") or "").strip()
    crow = ds.rows.get(ctl_id) if ctl_id else None
    if ctl_id and crow is None:
        raise KeyError(f"{row['measurement_id']}: control {ctl_id} not in measurements.tsv")
    if route == "chip" and crow is None and predicted_control is None:
        raise ValueError(f"{row['measurement_id']}: no observed control and no predicted control")
    cov = {"depth_reads": row["depth_reads"], "read_length": row["read_length"],
           "frag_len": row.get("frag_len")}
    if route == "chip":
        if crow is not None:
            cov.update(control_depth_reads=crow["depth_reads"],
                       control_read_length=crow["read_length"], control_is_predicted=False)
        else:   # the predicted control is asked for at the treatment's own C'
            cov.update(control_depth_reads=row["depth_reads"],
                       control_read_length=row["read_length"], control_is_predicted=True)
    out, source = [], None
    for i, (c, s) in enumerate(regions):
        sl = slice(i * int(rb), (i + 1) * int(rb))
        ctl = None
        if route == "chip":
            ctl = (ds.read(ctl_id, c, int(s), int(rb)).astype(np.float64) if crow is not None
                   else np.asarray(predicted_control, dtype=np.float64)[sl])
        sig, source = signal_track(mu[sl], n[sl], row["assay_class"] or row["assay"], ctl, cov,
                                   dnase_route=dnase_route, mode=mode, pileup=pileup)
        out.append(sig)
    return np.concatenate(out), source
