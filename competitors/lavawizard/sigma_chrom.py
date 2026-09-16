"""Which TRAINING chromosome Lavawizard's σ stage may take its residuals on.

    python -m lavawizard.sigma_chrom --regime configs/regime.eic_pilot.json
    python -m lavawizard.sigma_chrom --regime configs/regime.eic_pilot.json --chroms chr19

WHY THIS FILE EXISTS
--------------------
`BENCHMARK_DESIGN.md` §7 fits σ on TRAINING residuals, so `competitors/lavawizard/slurm/sigma.sh`
picks a TRAINING chromosome, fits that chromosome's position tables on top of the frozen shared
stem, and predicts the self-pairs from the result. That fit is a transfer, and this method is the
only one on the board whose transferable widths are keyed per chromosome:
`dataset3.UPSTREAM_HYPERPARAMS` gives each chromosome its own `(n_25bp_factors, n_250bp_factors,
n_5kbp_factors)`, and `model.py` sums the three into `block1.dense`'s input width. A stem fit under
one row therefore cannot load into a model built under another — `store_eic.shared_hparams_chrom`
says exactly that in its own docstring, which is why the shared stem borrows the EVAL chromosomes'
row rather than inventing one.

The σ chromosome is the one place a chromosome off that row could be reached, and on 2026-09-02
it was: Fir job 57833682 (`eic_pilot`) picked `chr1` — the first entry of the regime's
`train_chroms` — and died in `model.load_transferable` with `size mismatch for
block1.dense.weight: [2048, 225] vs [2048, 175]` after spending 7 minutes on a cache build.
`chr1`'s row is `(10, 10, 45)`; the stem's borrowed row is `chr20`'s `(25, 30, 60)`. Under `eic_19`
the first training chromosome IS `chr19`, which sits on the eval row, so the launcher had never
had to choose.

THE RULE, and it is the whole module
------------------------------------
The σ chromosome must be a TRAINING chromosome of the regime whose `UPSTREAM_HYPERPARAMS` row is
the row the shared stem borrows (`store_eic.shared_hparams_chrom`). The default is the FIRST such
chromosome in the regime's own `train_chroms` order, and an operator's `SIGMA_CHROMS` is held to
the same rule rather than trusted — the refusal costs a second on a login node, the mismatch costs
a GPU allocation and a cache build.

Under both live regimes the answer is `chr19` and there is no second candidate: `eic_19` trains on
chr19 alone, and of `eic_pilot`'s eighteen training chromosomes only chr19 carries the eval row
(the rest are `(10, 10, 45)` or `(25, 30, 45)`). So widening the σ scope with `SIGMA_CHROMS` is an
affordance for some later regime, not one either live regime can use.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from . import dataset3
from .store_eic import ScopeError, load_regime, shared_hparams_chrom

__all__ = ["EXIT_OFF_ROW", "row", "row_line", "on_row_chroms", "choose", "main"]

#: Exit code for "this chromosome is off the shared stem's row". The same 3 `sigma.sh` and
#: `score.sh` already use for "a σ table this stage refuses", so a caller branches on one number.
EXIT_OFF_ROW = 3


def row(chrom: str) -> Dict[str, int]:
    """`chrom`'s whole `UPSTREAM_HYPERPARAMS` row, through the two accessors that key on it.

    The schedule as well as the factor widths, because that is what `shared_hparams_chrom` compares
    when it decides the eval chromosomes agree — a σ chromosome held to a weaker test than the stem
    would be on a row the stem does not consider its own.
    """
    return {**dataset3.schedule(chrom), **dataset3.factor_sizes(chrom)}


def row_line(chrom: str) -> str:
    """One printable line for `chrom`'s row, for a banner or a refusal."""
    if chrom not in dataset3.UPSTREAM_HYPERPARAMS:
        return (f"{chrom}: not one of the 23 chromosomes `dataset3.UPSTREAM_HYPERPARAMS` is keyed "
                f"on, so it has no row and no factor widths at all")
    r = row(chrom)
    width = sum(dataset3.factor_sizes(chrom).values())
    return (f"{chrom}: " + " ".join(f"{k}={v}" for k, v in r.items())
            + f" (position width {width})")


def on_row_chroms(regime: dict) -> Tuple[str, List[str]]:
    """`(the chromosome the stem borrows its row from, the training chromosomes on that row)`.

    The second list keeps the regime's own `train_chroms` order, so "the first candidate" is a
    property of the config file and not of a set iteration.
    """
    stem = shared_hparams_chrom(regime)
    want = row(stem)
    train = list(regime.get("train_chroms") or [])
    if not train:
        raise ScopeError("this regime names no train_chroms, so there is no training chromosome to "
                         "take a σ residual on at all")
    return stem, [c for c in train
                  if c in dataset3.UPSTREAM_HYPERPARAMS and row(c) == want]


def choose(regime: dict, chroms: Optional[Sequence[str]] = None) -> Tuple[List[str], str]:
    """The σ chromosomes and the stem's row, or `ScopeError` naming both rows.

    `chroms=None` takes the default (the first candidate); the CLI maps an empty `--chroms` onto it,
    so an unset `SIGMA_CHROMS` needs no second command line. A named list is validated the same way,
    one chromosome at a time, and the refusal prints the row of each offender beside the stem's.
    """
    stem, candidates = on_row_chroms(regime)
    train = list(regime.get("train_chroms") or [])
    if chroms is None:
        if not candidates:
            raise ScopeError(
                f"no training chromosome of this regime sits on the row the shared stem borrows, "
                f"so its σ stage cannot transfer the stem into any of them. The stem's row is "
                f"{row_line(stem)}; the training chromosomes are {train} with rows "
                f"{[row_line(c) for c in train]}. A σ chromosome off the row fails in "
                f"`model.load_transferable` on `block1.dense.weight`, after the cache build.")
        return [candidates[0]], row_line(stem)

    named = [c for c in chroms if c]
    if not named:
        raise ScopeError("an empty `chroms` names no chromosome; pass None to take the default")
    # Two different mistakes, and they deserve two different sentences: a chromosome the regime
    # does not train on is a Rule 1 problem before it is a shape problem.
    not_train = [c for c in named if c not in train]
    if not_train:
        raise ScopeError(
            f"σ chromosome(s) {not_train} are not in this regime's train_chroms {train}, and §7 "
            f"fits σ on TRAINING residuals only. A residual taken on an eval chromosome is a "
            f"residual against a track the leaderboard then scores. On the shared stem's row and "
            f"available: {candidates}.")
    off = [c for c in named if c not in candidates]
    if off:
        raise ScopeError(
            f"σ chromosome(s) {off} are training chromosomes of this regime but are NOT on the row "
            f"the shared stem borrows, so `model.load_transferable` would refuse the stem's "
            f"`block1.dense.weight` after the cache build and the GPU allocation. "
            f"Named:  " + "  |  ".join(row_line(c) for c in off) + ". "
            f"Stem:   {row_line(stem)}. "
            f"On the row: {candidates} (of train_chroms {train}).")
    return named, row_line(stem)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Two lines on stdout — the chromosomes, comma-joined, then the stem's row — or exit 3."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--regime", required=True,
                   help="the SOURCE regime, e.g. configs/regime.eic_pilot.json -- not the derived "
                        "σ regime, whose eval_chroms are the training slice and whose row would "
                        "therefore be the wrong one to borrow")
    p.add_argument("--chroms", default=None,
                   help="a comma-separated override (SIGMA_CHROMS); held to the same rule. An "
                        "EMPTY string means the same as leaving it out, so a launcher can pass an "
                        "unset shell variable straight through")
    a = p.parse_args(argv)

    regime = load_regime(Path(a.regime))
    named = [c.strip() for c in (a.chroms or "").split(",") if c.strip()] or None
    try:
        chosen, stem_row = choose(regime, named)
    except (ScopeError, KeyError) as exc:
        print(f"REFUSING {a.regime}: {exc}", file=sys.stderr)
        return EXIT_OFF_ROW
    print(",".join(chosen))
    print(stem_row)
    return 0


if __name__ == "__main__":                                            # pragma: no cover
    raise SystemExit(main())
