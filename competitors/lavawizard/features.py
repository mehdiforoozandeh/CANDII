"""The two scalar inputs: the cross-cell **average** and **variance** of an assay, per bin.

These are the whole reason the model works. `Guacamole` adds the average straight to its output
(`model.py`), so everything the network does is a correction on top of this, and everything about
how the average is built is therefore a fairness question rather than an implementation detail.

**Signal space.** `arcsinh(-log10 p)`. Upstream applies `arcsinh` to the base-pair values and
*then* means them into 25 bp bins (`00_data_generation.py:56-63`). Our store already holds
`-log10 p` on the 25 bp grid (`DATA.md`:57), so we apply `arcsinh` to the binned value. `arcsinh`
is concave, so `mean(arcsinh(x)) <= arcsinh(mean(x))` and the two differ slightly within a bin.
This is a real divergence from their pipeline and belongs in any Lavawizard row's caveat.

**Who contributes (the switch the PI asked for, 2026-08-25).**

| mode | contributors | use |
|---|---|---|
| `"loo"` (default) | training biosamples carrying the assay, minus every biosample sharing the target's cell-type suffix | every number we report |
| `"upstream"` | every biosample carrying the assay, **including the target itself** | reproducing their weights only |

`"upstream"` reproduces a defect. `00_data_generation.py:147` averages every track carrying the
assay without skipping the target, while the docstring at `03_guacamole6_train.py:128` says it
should "exclude the current cell type" — so the leak is a bug, not a design choice. Its effect is
that a training target sits inside its own input feature with weight `1/k`, while a blind target
does not, so their model was trained and applied on differently-built features. We exclude by
default, matching their intent and `RIVALS_PLAN.md` §6.2. `"upstream"` exists so the parity check
against their released weights can feed them the features they actually saw.

**Which pool.** `contributors()` takes the pool explicitly and never guesses. §5 says our numbers
use training-split biosamples only; the challenge's own `Average` used training + validation, and
that variant is reproduced only on the Dataset-3 side where comparability is the point.

**ddof.** The variance here is the **population** variance, `E[x^2] - E[x]^2`, matching upstream
(`00_data_generation.py:182`). That is deliberate and is not the same quantity as §5.2's
`signal_sigma`, which is `ddof=1`. This one is a model input; that one is a predicted spread.
"""
from __future__ import annotations

from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np

__all__ = ["CONTRIBUTOR_MODES", "cell_type", "contributors", "cross_cell_moments", "blocks"]

#: `carries(biosample, assay) -> bool`, and `read_pval(biosample, assay, start, end) -> array`.
#: Both are seams: they keep this module testable without a store, and let the caller own the
#: open handles.
CarriesFn = Callable[[str, str], bool]
ReadPvalFn = Callable[[str, str, int, int], np.ndarray]

CONTRIBUTOR_MODES = ("loo", "upstream")

#: Biosample ids carry a split prefix (`DATA.md`); the cell type is what follows it.
_SPLIT_PREFIXES = ("T_", "V_", "B_")


def cell_type(biosample: str) -> str:
    """`'T_DND-41' -> 'DND-41'`. An id with no known prefix is its own cell type."""
    for p in _SPLIT_PREFIXES:
        if biosample.startswith(p):
            return biosample[len(p):]
    return biosample


def contributors(pool: Sequence[str], assay: str, target: str, *,
                 carries: CarriesFn, mode: str = "loo") -> List[str]:
    """Which biosamples feed the average for `target`'s `assay`.

    `pool` is the candidate biosamples — the regime's training split for our runs. `carries(bs,
    assay)` says whether a biosample has that assay; pass `corpus.biosamples_with(assay).__contains__`
    style predicate, or a set membership test. Order follows `pool`, so the result is deterministic.

    Under `"loo"` the exclusion is by **cell type, not by biosample id** (`RIVALS_PLAN.md` §5): T_X
    is excluded when predicting V_X, because they are the same cell measured twice and the bench's
    own impute dial masks it the same way.
    """
    if mode not in CONTRIBUTOR_MODES:
        raise ValueError(f"mode must be one of {CONTRIBUTOR_MODES}, got {mode!r}")
    have = [bs for bs in pool if carries(bs, assay)]
    if mode == "upstream":
        return have
    blocked = cell_type(target)
    return [bs for bs in have if cell_type(bs) != blocked]


def cross_cell_moments(read_pval: ReadPvalFn,
                       contributor_ids: Sequence[str], chrom: str, start: int, end: int,
                       *, assay: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """Mean and population variance of `arcsinh(-log10 p)` over `contributor_ids`, per bin.

    `read_pval(biosample, assay, start, end)` returns a `(end - start,)` float array of `-log10 p`
    for one contributor — a thin seam so this is testable without a store and so the caller decides
    the chromosome handle. Accumulates in float64 (upstream uses float32); with `arcsinh` bounding
    the values to order 10 and a few dozen contributors, the difference is far below the store's own
    0.025 % pval codec bound (`STORE.md`).

    Returns `(average, variance, k)`, both arrays `float32` of length `end - start`. With `k == 0`
    both arrays are zeros — the caller decides whether that track is skipped (§5 says it is, and is
    listed).
    """
    n = int(end) - int(start)
    if n <= 0:
        raise ValueError(f"empty bin range [{start}, {end})")
    total = np.zeros(n, dtype=np.float64)
    total_sq = np.zeros(n, dtype=np.float64)
    k = 0
    for bs in contributor_ids:
        x = np.arcsinh(np.asarray(read_pval(bs, assay, start, end), dtype=np.float64).reshape(-1))
        if x.shape[0] != n:
            raise ValueError(f"{bs}/{assay} {chrom}[{start}:{end}) gave {x.shape[0]} bins, want {n}")
        total += x
        total_sq += x * x
        k += 1
    if k == 0:
        return np.zeros(n, np.float32), np.zeros(n, np.float32), 0
    mean = total / k
    # E[x^2] - E[x]^2, as upstream. Clipped at 0: the identity is exact in real arithmetic but can
    # go a few ulp negative in floating point when every contributor agrees.
    var = np.maximum(total_sq / k - mean * mean, 0.0)
    return mean.astype(np.float32), var.astype(np.float32), k


def blocks(n_bins: int, block_bins: int) -> Iterable[Tuple[int, int]]:
    """`[start, end)` pairs tiling `n_bins`. Whole-chromosome reads for forty contributors run to
    gigabytes at 25 bp, so the callers stream; this is the tiling they share."""
    if block_bins <= 0:
        raise ValueError(f"block_bins must be positive, got {block_bins}")
    for s in range(0, int(n_bins), int(block_bins)):
        yield s, min(s + int(block_bins), int(n_bins))
