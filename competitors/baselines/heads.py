"""The arithmetic of the naive baselines (`RIVALS_PLAN.md` §5.1–§5.4), and nothing else.

Pure numpy over `[k, N]` contributor blocks: no store, no regime, no file paths. That is what lets
`tests/test_baselines.py` hand three contributors to these functions and compare against numbers
worked out by hand — the §5.5 fixture gate — instead of against a second implementation.

DEPTH, AND WHY IT IS COPIED RATHER THAN IMPORTED
------------------------------------------------
`src/candi/reference.py` already implements the depth-free leave-one-out mean for the h5 path, and
§5.1 says to reuse **its arithmetic, not necessarily its storage**. Its storage is an h5 of
`sum`/`count` slabs indexed by training window; a baseline generator walks whole chromosomes on the
absolute bin grid and keeps nothing, so the file layout is of no use here. The two lines that matter
are copied verbatim in meaning:

    scale_j      = 2 ** (depth_center - d_j)      # reference.py::build_reference, `scale[bi, ok]`
    x_j          = c_j * scale_j                  # every contribution rescaled before summing

`reference.py` stops there because it only ever needs a shape. A baseline has to hand back counts at
the TARGET track's exposure, so §5.1 adds the second half, `s = 2 ** (d_t - depth_center)`.

Worth knowing while reading: `depth_center` **cancels**. `mu = s * mean(c_j * 2**(depth_center-d_j))`
is `2**d_t * mean(c_j * 2**-d_j)` whatever the centre is, and the variance carries `s**2` the same
way. The centre is kept in the arithmetic anyway — so this file and `reference.py` say the same
thing in the same units, and so the number written into the manifest means something — but no
baseline number moves if it is re-derived on a different pool. `test_L3_generator_is_depth_free` is
the property that follows.

THE PRE-REGISTERED VARIANCE (§5.1, verbatim: amend the plan first, this file second)
-----------------------------------------------------------------------------------
The predicted spread is the CROSS-CELL BIOLOGICAL spread, depth-scaled, floored at Poisson. It
deliberately omits an extra target-side counting-noise term. `n = 1e6` is that floor: an NB cannot be
under-dispersed, so cross-cell agreement tighter than Poisson is reported as Poisson.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = [
    "POISSON_N", "MU_FLOOR", "N_FLOOR", "SIGMA_FLOOR",
    "depth_scale", "normalize_to_center", "nb_from_moments", "moment_matched_nb",
    "plain_mean", "arcsinh_mean", "cross_cell_sigma", "peak_fraction",
]

#: §5.1 — the NB's Poisson limit, used whenever the cross-cell variance is not above the mean.
#:
#: **AND `candi.bench` CANNOT SCORE IT TODAY.** `candi.metrics.nb_crps` evaluates the Gini mean
#: difference as `hyp2f1(0.5, 1 - n, 2, w)`, and that term returns NaN for every `n` above roughly
#: 2e4 at any µ. One NaN bin makes the track's whole `crps` NaN, `macro_mean` then drops the key, and
#: the count arm loses `crps`, `crps_oracle_scaled`, `scale_error` and `beats_marginal` — the entire
#: distributional tier. CANDI's own decoder exponentiates a bounded head and never reaches that
#: region, so nothing had exercised it before this suite.
#:
#: The floor stays at the pre-registered value: §5.1 says to amend the plan first and the code
#: second, and that is the PI's call, not this file's. `poisson_n` is a parameter so an amended plan
#: (or a raised ceiling in `nb_crps`) needs no edit here. Measured for the record, `y = round(µ)`
#: against the exact Poisson CRPS: at `n = 1e4` the error is -0.01 % at µ = 0.1, +0.05 % at µ = 10,
#: +0.50 % at µ = 100 and +4.9 % at µ = 1000. Where the floor actually BINDS — bins whose
#: contributors all agree, overwhelmingly the near-zero ones — 1e4 and 1e6 are the same number.
POISSON_N = 1e6

#: Numerical guards, not modelling choices. An NB needs `mu > 0` and `n > 0`: at `mu = 0` exactly,
#: `p = n / (n + mu)` is 1.0 and `log(1 - p)` is `-inf` in `bench.distributional.loss_block`. CANDI's
#: own decoder exponentiates, so it can never emit a hard zero and never exercises that path; a
#: baseline that averages a field of zeros can. Both floors are ~11 orders below a single read, so no
#: metric moves: at `mu = 1e-6` the squared error against a zero bin is 1e-12.
MU_FLOOR = 1e-6
N_FLOOR = 1e-6

#: Same kind of guard on the pval arm. `gauss_suite` already clamps at 1e-12; `loss_block`'s
#: `gaussian_nll` clamps the VARIANCE at 1e-6, so a hard-zero sigma is a `log(1e-6)` cliff rather
#: than an error. Flooring here keeps the written array and the scored array the same object.
SIGMA_FLOOR = 1e-6


def depth_scale(log2_depth: float, depth_center: float) -> float:
    """`2 ** (depth_center - d)` — `reference.py::build_reference`'s `scale`, for one track."""
    return float(2.0 ** (float(depth_center) - float(log2_depth)))


def normalize_to_center(counts: np.ndarray, log2_depths: np.ndarray,
                        depth_center: float) -> np.ndarray:
    """`[k, N]` raw counts -> `[k, N]` counts at the common `depth_center` exposure. float64.

    One row per contributor, one column per bin. `log2_depths` is `[k]`.
    """
    c = np.asarray(counts, dtype=np.float64)
    d = np.asarray(log2_depths, dtype=np.float64).reshape(-1, 1)
    return c * np.power(2.0, float(depth_center) - d)


def nb_from_moments(m: np.ndarray, v: Optional[np.ndarray], target_log2_depth: float,
                    depth_center: float, *,
                    poisson_n: float = POISSON_N) -> Tuple[np.ndarray, np.ndarray]:
    """§5.1's second half — `(mu, n)` from a mean and an unbiased variance held AT `depth_center`.

        s = 2 ** (d_t - depth_center)
        mu = s * m                       V = s**2 * v
        n  = mu**2 / (V - mu)  where V > mu,  else POISSON_N
        p  = n / (n + mu)                                       (derived by the scorer, not here)

    `v is None` means "no variance was measurable" — one contributor — and every bin takes the
    Poisson floor, the same branch a bin whose contributors agree tightly takes.

    Two callers, and they are the reason this is not inlined into `moment_matched_nb`: the
    cross-cell average measures `(m, v)` per bin, the per-assay marginal (§5.4) fits one `(m, v)`
    for the whole assay, and both must reach the NB through the same floor and the same depth
    scaling or the weakest tier stops being comparable to the tier above it.
    """
    m = np.asarray(m, dtype=np.float64)
    s = 2.0 ** (float(target_log2_depth) - float(depth_center))
    mu = np.maximum(s * m, MU_FLOOR)
    if v is None:
        return mu, np.full(mu.shape, float(poisson_n))
    V = (s * s) * np.asarray(v, dtype=np.float64)
    over = np.broadcast_to(V > mu, mu.shape)
    n = np.full(mu.shape, float(poisson_n))
    # `V - mu` is strictly positive under the mask, so no divide-by-zero and no clipping of a
    # negative dispersion into a plausible-looking one.
    Vb = np.broadcast_to(V, mu.shape)
    n[over] = np.maximum(mu[over] ** 2 / (Vb[over] - mu[over]), N_FLOOR)
    return mu, n


def moment_matched_nb(x: np.ndarray, target_log2_depth: float, depth_center: float, *,
                      poisson_n: float = POISSON_N) -> Tuple[np.ndarray, np.ndarray]:
    """§5.1 — `(mu, n)` of the NB whose mean and variance match the depth-scaled contributor spread.

    `x` is `[k, N]` **already at `depth_center`** (`normalize_to_center`); `m = mean_k(x)` and
    `v = var_k(x, ddof=1)`, undefined at `k = 1`. Returns float64 `[N]` each; the caller casts to
    float32 at write time.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 1:
        raise ValueError(f"contributor block must be [k>=1, N], got {x.shape}")
    v = None if x.shape[0] == 1 else x.var(axis=0, ddof=1)
    return nb_from_moments(x.mean(axis=0), v, target_log2_depth, depth_center,
                           poisson_n=poisson_n)


def plain_mean(x: np.ndarray) -> np.ndarray:
    """§5.2 — the EIC `Average`: arithmetic mean of `-log10 p` across contributors. `[k, N] -> [N]`."""
    return np.asarray(x, dtype=np.float64).mean(axis=0)


def arcsinh_mean(x: np.ndarray) -> np.ndarray:
    """§5.2 — the `avg-arcsinh` variant: mean taken in arcsinh space, `sinh` back.

    NEVER the EIC baseline row. It is a different estimator of a different central tendency (it
    down-weights the tall bins the challenge's own metrics care most about), so it ships as its own
    method directory and is labelled as a variant everywhere it appears.
    """
    return np.sinh(np.arcsinh(np.asarray(x, dtype=np.float64)).mean(axis=0))


def cross_cell_sigma(x: np.ndarray) -> Optional[np.ndarray]:
    """§5.2 — per-bin `std(contributors, ddof=1)` in `-log10 p`, or None when `k = 1`.

    None, not zeros. A single contributor gives no measurement of cross-cell spread, and writing a
    zero sigma would claim certainty the baseline never had — `bench.external` then scores that track
    with the E and P blocks and no `gauss_suite`, which is the honest outcome (§4.2).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] < 2:
        return None
    return np.maximum(x.std(axis=0, ddof=1), SIGMA_FLOOR)


def peak_fraction(peaks: np.ndarray) -> np.ndarray:
    """§5.3 — `(# contributors with a peak at the bin) / k`, in [0, 1]. THE ranking score itself."""
    p = np.asarray(peaks, dtype=np.float64)
    return p.mean(axis=0)
