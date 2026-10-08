"""t118 ladder — the shuffled-bin-value twin's draw (row 2, model kind "xshuf").

The twin is a row-2 run with real covariates in which g reads x from another bin of the same
source track instead of the bin's own x; f still reads the true x. Every shuffled x in the code
base goes through `draw_positions`.

**The rule** (`RULE = "same_chrom_uniform"`, option (a), PI ruling 2026-09-28): the x g reads at
bin i is the source track's x at a uniformly random bin of the SAME chromosome.
  training    fresh at every step: independent uniform draws with replacement, the bin itself not
              excluded (P = 1/n), from a fixed per-(pid, chromosome) `Pool` of `POOL_BINS` bins — a
              uniform sample of the chromosome, read once per run (`draw_positions(fixed=False)`).
              A uniform draw from a uniform sample of the bins is a uniform draw over the bins.
  prediction  one fixed seeded permutation per chromosome (`pred_rng`: run seed, `SALT_PRED`, the
              chromosome's index in `pairs.MAIN_CHROMS`), shared by every source pid and every
              covariate pair: g at bin j reads x at perm[j] (`draw_positions(fixed=True)`). Used
              for validation on chr22 and for the trained/shuffle/swap/law scoring.
  pads        off-chromosome halo positions and chunk padding are not bins of the chromosome: g
              reads their own padded value there (`transform_x(0)`, as f does), in training and
              at prediction alike. The pool draws and the permutation cover bins only.

**The alternatives** (not implemented; each is a one-place change):
  (b) "window_perm"    a permutation within the window: `draw_positions(fixed=True)` with the
                       window itself as the source (`train.Sampler.draw` permutes `xs[b]`'s bins
                       instead of reading the pool; `model.predict_chrom_bins` permutes within each
                       chunk).
  (c) "split_uniform"  a uniform draw over all training chromosomes of the split: this function
                       is kept; `Pool` keeps one pool per pid over all its chromosomes (the `chrom`
                       key ignored) and the prediction source is their concatenation.

numpy and `ladder.pairs` only; no torch.
"""
from __future__ import annotations

import numpy as np

from ladder import pairs

RULE = "same_chrom_uniform"
POOL_BINS = 1 << 17
#: taken: 7118 ids permutation, 7119 shuffle target, 7120 score.py bin pick, 7121 nocov redraw
SALT_POOL, SALT_TRAIN, SALT_PRED = 7122, 7123, 7124


def draw_positions(rng: np.random.Generator, n: int, m: int, fixed: bool = False) -> np.ndarray:
    """THE draw: int64[m] positions into a source of `n` bins.

    `fixed=False` (training): m independent uniform draws over range(n), with replacement, the
    position itself not excluded. `fixed=True` (prediction and scoring): `m` must equal `n`; one
    permutation of range(n).
    """
    n, m = int(n), int(m)
    if n < 1:
        raise ValueError(f"draw_positions: n = {n}, need at least one bin")
    if fixed:
        if m != n:
            raise ValueError(f"draw_positions(fixed=True): m = {m} must equal n = {n}")
        return rng.permutation(n).astype(np.int64)
    return rng.integers(n, size=m).astype(np.int64)


class Pool:
    """Per-(pid, chromosome) raw-value pools for the training draw.

    For every (pid, chrom) in `sorted(pids)` x `chroms` (in that order) the chromosome is read once;
    `values[(pid, chrom)]` is the whole chromosome when it has at most `n_bins` bins, else
    `X[np.sort(draw_positions(rng, n, n_bins))]`, with `rng = default_rng([seed, SALT_POOL])`
    consumed in that fixed order. Values keep the corpus's raw dtype (untransformed).
    """

    def __init__(self, corpus, space: str, pids, chroms, seed: int, n_bins: int = POOL_BINS):
        self.space, self.n_bins, self.seed = space, int(n_bins), int(seed)
        self.pids, self.chroms = sorted(pids), list(chroms)
        rng = np.random.default_rng([self.seed, SALT_POOL])
        self.values: dict = {}
        for pid in self.pids:
            for chrom in self.chroms:
                X = np.asarray(corpus.get(pid, space, chrom))
                n = int(X.shape[0])
                if n <= self.n_bins:
                    self.values[(pid, chrom)] = np.array(X)
                else:
                    self.values[(pid, chrom)] = X[np.sort(draw_positions(rng, n, self.n_bins))]

    def n_source(self, pid: str, chrom: str) -> int:
        return int(len(self.values[(pid, chrom)]))

    def describe(self) -> dict:
        return {"rule": RULE, "pool_bins": self.n_bins, "n_pairs_pid_chrom": len(self.values),
                "salt_pool": SALT_POOL, "salt_train": SALT_TRAIN, "salt_pred": SALT_PRED}


def train_rng(seed: int) -> np.random.Generator:
    """The training draw's rng: `default_rng([seed, SALT_TRAIN])`."""
    return np.random.default_rng([int(seed), SALT_TRAIN])


def pred_rng(seed: int, chrom: str) -> np.random.Generator:
    """The fixed permutation's rng of one chromosome: `default_rng([seed, SALT_PRED, index])`."""
    if chrom not in pairs.MAIN_CHROMS:
        raise ValueError(f"chromosome {chrom!r} not in MAIN_CHROMS")
    return np.random.default_rng([int(seed), SALT_PRED, pairs.MAIN_CHROMS.index(chrom)])
