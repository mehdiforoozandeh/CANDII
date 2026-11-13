"""Maps between our count model's space and AlphaGenome 2's space (plan Amendment 5, section B.2;
pinned interface H, chunk AGs).

    resample_mean(x, src_start_bp, src_bin_bp, dst_start_bp, dst_bin_bp, n_dst) -> float64[n_dst]
    resample_sum(x, src_start_bp, src_bin_bp, dst_start_bp, dst_bin_bp, n_dst) -> float64[n_dst]
    fold_change_track(mu, control_counts, covariates, *, gsize) -> (float32, lambda_source)
    normalized_insertions(mu, covariates) -> float32
    native_from_nb(n, mu, row, *, ds, regions, rb, predicted_control=None) -> (float32, lambda_source)
    peak_probability_from_nb(n, mu, row, *, ds, regions, rb, predicted_control=None,
                             threshold=2.0, dnase_route=DNASE_ROUTE_DEFAULT) -> float32
    isotonic_recalibrate(pred, truth, n_bins=2000, valid=None) -> float64

Sources.
* `resample_mean` generalizes `to25` of Max's sprint 12 `scripts/api/score_f1.py` (lines 210-214,
  repository mlibbrecht/01_2026-10-19_research at bde905a, folder projects/03_2026-10-20_epi_imputation/
  sprints/12_2026-11-12_ag2_fair_test): the cumulative integral of the source is interpolated at the
  destination bin edges. One change: a destination bin that runs past the source is divided by its
  covered length (Max divides by the full 25 bp), and a bin with no overlap is NaN. Inside the source
  span the two are identical.
* `isotonic_recalibrate` ports `pava` and `monotone_cal` of the same file (lines 238-261, with
  `NCAL = 2000` from line 43): isotonic regression of the truth on the prediction through `n_bins`
  quantile bins of the prediction, then linear interpolation between the bin means. The only addition
  is `valid`, which restricts the fit (not the output) to a subset of bins.
* `fold_change_track` and `peak_probability_from_nb` use the lambda of the D9b ChIP-seq and MACS2
  no-control routes of `signal_from_counts.signal_track` (lines 280-302) through its own
  `pileup_from_counts` and `macs2_lambda`; nothing of those two functions is restated.

Fold change (section B.2, secondary space A): FE = pileup(mu) / lambda, no pseudocount. FE is linear
in the count, so its NB expectation is the value at `mu`; `n` is unused there.
Normalized insertions (secondary space B): mu * 25 / (rl + 24) * 1e8 / depth_reads; it has no lambda,
so its lambda source is `none`.
Peak probability (section B.3): P_{K ~ NB(n, mu)}[-log10 P(Poisson(lambda) > floor(pileup(K))) >=
threshold]. The score is non-decreasing in K, so the event is K >= k*, and the probability is the NB
survival function at k* - 1.

Numpy and scipy only. AlphaGenome 2 is never imported here.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import nbinom, poisson

from candi.store.pval_from_counts import MACS2_GSIZE_HS, log10_upper_tail

from onion_eval.signal_from_counts import (ATAC_SMOOTH_WIN, DNASE_ROUTE_DEFAULT, DNASE_ROUTES,  # `tools` on sys.path, as score.py
                                           SPAN_EXTRA, macs2_lambda, pileup_from_counts, route_of)

__all__ = [
    "AG2_INSERTION_TOTAL",
    "resample_mean",
    "resample_sum",
    "fold_change_track",
    "normalized_insertions",
    "native_from_nb",
    "peak_probability_from_nb",
    "isotonic_recalibrate",
]

BIN = 25
#: AG2's accessibility unit: insertions normalized to a genome-wide sum of 1e8 (AG2_FACTS.md).
AG2_INSERTION_TOTAL = 1e8
NO_LAMBDA = "none"


# ---------------------------------------------------------------- resolution maps

def _overlap_integrals(x, src_start_bp, src_bin_bp, dst_start_bp, dst_bin_bp, n_dst):
    """(integral of the source over each destination bin, covered bp of each bin); the source value
    of a bin is taken as constant over the bin."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"x must be 1-D, got shape {x.shape}")
    if not (src_bin_bp > 0 and dst_bin_bp > 0) or int(n_dst) < 0:
        raise ValueError("bin sizes must be > 0 and n_dst >= 0")
    F = np.concatenate([[0.0], np.cumsum(x * float(src_bin_bp))])       # integral up to each source edge
    edges = float(dst_start_bp) + np.arange(int(n_dst) + 1, dtype=np.float64) * float(dst_bin_bp)
    pos = np.clip((edges - float(src_start_bp)) / float(src_bin_bp), 0.0, float(x.size))
    integ = np.diff(np.interp(pos, np.arange(x.size + 1, dtype=np.float64), F))
    covered = np.diff(pos) * float(src_bin_bp)
    return integ, covered


def resample_mean(x, src_start_bp, src_bin_bp, dst_start_bp, dst_bin_bp, n_dst) -> np.ndarray:
    """Overlap-weighted mean of `x` (bin j = [src_start + j*src_bin, +src_bin)) over each destination
    bin [dst_start + i*dst_bin, +dst_bin), i < n_dst. 128 to 25 bp and 25 to 128 bp. A bin partly
    outside the source is the mean over its covered part; a bin wholly outside is NaN."""
    integ, covered = _overlap_integrals(x, src_start_bp, src_bin_bp, dst_start_bp, dst_bin_bp, n_dst)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(covered > 0, integ / np.where(covered > 0, covered, 1.0), np.nan)


def resample_sum(x, src_start_bp, src_bin_bp, dst_start_bp, dst_bin_bp, n_dst) -> np.ndarray:
    """Overlap-weighted sum of `x` over each destination bin: a source bin adds its value times the
    fraction of it that the destination bin covers (1 bp to 25 bp: the plain sum). Outside the
    source adds 0."""
    integ, _ = _overlap_integrals(x, src_start_bp, src_bin_bp, dst_start_bp, dst_bin_bp, n_dst)
    return integ / float(src_bin_bp)


# ---------------------------------------------------------------- lambda of the D9b routes

def _need(cov: dict, *keys):
    missing = [k for k in keys if cov.get(k) in (None, "")]
    if missing:
        raise KeyError(f"covariates lack {missing}")
    return [float(cov[k]) for k in keys]


def _chip_lambda(mu, control_counts, covariates, gsize):
    """lambda of the ChIP-seq route of `signal_track` (lines 293-300), its pileup scale d/(rl+24),
    and its lambda source."""
    if control_counts is None:
        raise ValueError("ChIP-seq needs a control: the observed one, or a predicted one "
                         "(set covariates['control_is_predicted'])")
    depth, rl, d, cdepth, crl = _need(covariates, "depth_reads", "read_length", "frag_len",
                                      "control_depth_reads", "control_read_length")
    control_counts = np.asarray(control_counts, dtype=np.float64)
    if control_counts.shape != np.shape(mu):
        raise ValueError(f"control and treatment must be binned the same: {control_counts.shape} vs {np.shape(mu)}")
    lam = macs2_lambda(lambda_bg=d * depth / gsize,
                       control_pileup=pileup_from_counts(control_counts, d, crl),
                       ratio_treat2control=depth / cdepth)
    source = "predicted_control" if bool(covariates.get("control_is_predicted", False)) else "observed_control"
    return lam, d, rl, source


def _no_control_lambda(mu, covariates, gsize):
    """lambda of the MACS2 no-control route of `signal_track` (lines 283-285; ATAC-seq, and DNase-seq
    under `macs2_no_control`): the 10 kb window reads the predicted mean."""
    depth, rl = _need(covariates, "depth_reads", "read_length")
    d = float(ATAC_SMOOTH_WIN)
    lam = macs2_lambda(lambda_bg=d * depth / gsize, treat_pileup=pileup_from_counts(mu, d, rl))
    return lam, d, rl, "local_background"


# ---------------------------------------------------------------- native spaces

def fold_change_track(mu, control_counts, covariates: dict, *, gsize: float = MACS2_GSIZE_HS):
    """ChIP-seq fold change over the D9b lambda, per 25 bp bin: pileup(mu, frag_len, read_length) /
    lambda, no pseudocount. Covariate keys as the ChIP-seq route of `signal_track`. Returns
    (float32 [L], lambda_source)."""
    mu = np.asarray(mu, dtype=np.float64)
    if mu.ndim != 1:
        raise ValueError(f"mu must be 1-D [L], got shape {mu.shape}")
    lam, d, rl, source = _chip_lambda(mu, control_counts, covariates, gsize)
    return (pileup_from_counts(mu, d, rl) / lam).astype(np.float32), source


def normalized_insertions(mu, covariates: dict) -> np.ndarray:
    """AG2's accessibility unit per 25 bp bin: tags behind the count (mu * 25 / (rl + 24)) over the
    track's total tags (`depth_reads`), times 1e8."""
    depth, rl = _need(covariates, "depth_reads", "read_length")
    if not depth > 0:
        raise ValueError(f"depth_reads must be > 0, got {depth}")
    mu = np.asarray(mu, dtype=np.float64)
    return (mu * BIN / (rl + SPAN_EXTRA) * AG2_INSERTION_TOTAL / depth).astype(np.float32)


def _row_setup(n, mu, row, ds, regions, rb, predicted_control):
    """The control and covariate logic of `signal_from_nb` (signal_from_counts.py lines 318-337)."""
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
        else:
            cov.update(control_depth_reads=row["depth_reads"],
                       control_read_length=row["read_length"], control_is_predicted=True)

    def control(i, c, s):
        if route != "chip":
            return None
        if crow is not None:
            return ds.read(ctl_id, c, int(s), int(rb)).astype(np.float64)
        return np.asarray(predicted_control, dtype=np.float64)[i * int(rb):(i + 1) * int(rb)]

    return n, mu, route, cov, control


def native_from_nb(n, mu, row: dict, *, ds, regions, rb: int, predicted_control=None):
    """The target's prediction in AG2's own unit over the scorer's concatenated eval regions:
    ChIP-seq fold change (per region, as `signal_from_nb`) or, for DNase-seq and ATAC-seq, normalized
    insertions. Arguments as `signal_from_nb`; `n` is unused (both maps are linear in the count).
    Returns (float32, lambda_source); `none` for accessibility."""
    n, mu, route, cov, control = _row_setup(n, mu, row, ds, regions, rb, predicted_control)
    if route != "chip":
        return normalized_insertions(mu, cov), NO_LAMBDA
    out, source = [], None
    for i, (c, s) in enumerate(regions):
        fe, source = fold_change_track(mu[i * int(rb):(i + 1) * int(rb)], control(i, c, s), cov)
        out.append(fe)
    return np.concatenate(out), source


# ---------------------------------------------------------------- peak probability

def _first_count_at_threshold(lam, scale, threshold):
    """Per bin, the smallest integer count k with -log10 P(Poisson(lam) > floor(k * scale)) >= threshold."""
    lam = np.asarray(lam, dtype=np.float64)
    ok = lambda m: log10_upper_tail(np.maximum(m, 0), lam) >= threshold   # noqa: E731
    m = np.maximum(poisson.isf(10.0 ** (-float(threshold)), lam), 0).astype(np.int64)
    for _ in range(64):  # isf is exact up to rounding at the boundary; settle it with the scorer's own score
        lower = (m > 0) & ok(m - 1)
        higher = ~ok(m)
        if not (lower.any() or higher.any()):
            break
        m = m - lower + higher
    else:  # pragma: no cover
        raise RuntimeError("count threshold did not settle")
    k = np.ceil(m / float(scale)).astype(np.int64)
    k = k + (np.floor(k * float(scale)) < m)
    k = k - ((k > 0) & (np.floor((k - 1) * float(scale)) >= m))
    return k


def peak_probability_from_nb(n, mu, row: dict, *, ds, regions, rb: int, predicted_control=None,
                             threshold: float = 2.0, dnase_route: str = DNASE_ROUTE_DEFAULT) -> np.ndarray:
    """Per bin, the NB probability that the D9b MACS2 score of the count reaches `threshold`
    (-log10 p >= 2, i.e. p <= 0.01, by default), over the scorer's concatenated eval regions.
    Arguments as `signal_from_nb`. The DNase-seq route `hotspot2_density` has no p-value and is refused."""
    if dnase_route not in DNASE_ROUTES:
        raise ValueError(f"dnase_route {dnase_route!r} not in {DNASE_ROUTES}")
    n, mu, route, cov, control = _row_setup(n, mu, row, ds, regions, rb, predicted_control)
    if route == "dnase" and dnase_route == "hotspot2_density":
        raise ValueError("the hotspot2_density route has no p-value, so no peak probability")
    if not (np.all(mu > 0) and np.all(n > 0)):
        raise ValueError("mu and n must be > 0 everywhere (contract: NB outputs are positive)")
    out = []
    for i, (c, s) in enumerate(regions):
        sl = slice(i * int(rb), (i + 1) * int(rb))
        if route == "chip":
            lam, d, rl, _ = _chip_lambda(mu[sl], control(i, c, s), cov, MACS2_GSIZE_HS)
        else:
            lam, d, rl, _ = _no_control_lambda(mu[sl], cov, MACS2_GSIZE_HS)
        k = _first_count_at_threshold(lam, d / (rl + SPAN_EXTRA), threshold)
        out.append(nbinom.sf(k - 1, n[sl], n[sl] / (n[sl] + mu[sl])))
    return np.concatenate(out).astype(np.float32)


# ---------------------------------------------------------------- oracle recalibration

def _pava(v, w):
    """Pool-adjacent-violators: the non-decreasing fit of `v` with weights `w` (score_f1.py l.238-247)."""
    vals, wts, cnts = [], [], []
    for vi, wi in zip(v, w):
        vals.append(float(vi)); wts.append(float(wi)); cnts.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            ws = wts[-2] + wts[-1]
            vals[-2] = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / ws
            wts[-2] = ws
            cnts[-2] += cnts[-1]
            vals.pop(); wts.pop(); cnts.pop()
    return np.repeat(vals, cnts)


def isotonic_recalibrate(pred, truth, n_bins: int = 2000, valid=None) -> np.ndarray:
    """Oracle monotone recalibration of `pred` to `truth` on the track itself (score_f1.py
    `monotone_cal`, l.250-261): isotonic regression of the truth on `n_bins` quantile bins of the
    prediction, interpolated linearly between bin means and held flat beyond them. `valid` (bool,
    same shape) restricts the fit; the map is applied to every bin. Returns float64."""
    p_all = np.asarray(pred, dtype=np.float64).ravel()
    y_all = np.asarray(truth, dtype=np.float64).ravel()
    if p_all.shape != y_all.shape:
        raise ValueError(f"pred and truth differ in shape: {p_all.shape} vs {y_all.shape}")
    m = np.ones(p_all.shape, bool) if valid is None else np.asarray(valid, bool).ravel()
    p, y = p_all[m], y_all[m]
    if p.size == 0:
        raise ValueError("no valid bin to fit the recalibration on")
    o = np.argsort(p, kind="stable")
    edges = np.unique(np.linspace(0, p.size, int(n_bins) + 1).astype(np.int64))
    cp = np.concatenate([[0.0], np.cumsum(p[o])])
    cy = np.concatenate([[0.0], np.cumsum(y[o])])
    cnt = np.diff(edges).astype(np.float64)
    mp, my = np.diff(cp[edges]) / cnt, np.diff(cy[edges]) / cnt
    u, inv = np.unique(mp, return_inverse=True)   # bins with the same mean prediction share a value
    wu = np.bincount(inv, weights=cnt)
    yu = np.bincount(inv, weights=my * cnt) / wu
    return np.interp(p_all, u, _pava(yu, wu)).reshape(np.shape(pred))
