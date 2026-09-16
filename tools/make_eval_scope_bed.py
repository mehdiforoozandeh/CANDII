#!/usr/bin/env python3
"""Draw a SEEDED random window set for the mid-training selection scope, and re-check it.

    python tools/make_eval_scope_bed.py make --store configs/regime.eic_19.json \
        --windows 450 --seed 890217 --out configs/regions/eval_random450_seed890217.bed

    python tools/make_eval_scope_bed.py check \
        --bed configs/regions/eval_random450_seed890217.bed \
        --windows 450 --seed 890217 \
        --n-bins chr20=2577766,chr21=1868399,chr22=2032738 --context-bins 768

**WHY A RANDOM SET AND NOT A PUBLISHED ONE.** The first candidate for `--eval-regions` was
`configs/regions/encode_pilot_hg38.bed` — the 44 ENCODE Pilot Regions — on the argument that a
named, fixed, published region set chosen in 2007 is one nobody here could be accused of picking to
flatter a method. Measured against full coverage on an 8-checkpoint ladder, it FLIPPED 4 of 15
checkpoint comparisons, at gaps of up to 0.069, because its bias DRIFTS with the model instead of
cancelling in a difference. Pedigree is not faithfulness: who chose the loci governs whether the
choice is corrupt, and says nothing about whether the loci track the genome-wide number, which is
the only property selection needs. `EVAL.md` carries the table.

Subsetting itself was never the problem — 200 random matched subsets centre on the full-coverage
value to a tenth of a standard deviation, and 194 of 200 reproduce the full checkpoint ordering
exactly. So the scope this writes is drawn by a SEED rather than by anyone's judgement, and the
seed, the rule and the hash are committed before any number is produced with it. That is a
different kind of auditability from a citation, and unlike a citation it was measured.

**TWO RULES IN THE DRAW, AND BOTH ARE LOAD-BEARING.**

*Draw from the eligible-start set the FULL-COVERAGE pass walks.* The scope is an estimator of the
full number, so it has to be a subset of exactly what it estimates — `harness.full_tiling` of the
regime's eval chromosomes, the same chromosome-0-anchored grid `EvalSource.windows` returns with no
scope. Re-tiling per region, or drawing from anywhere else, would make it an estimator of something
that is never computed.

*Do not stratify.* Not by chromosome, not by signal, not by anything. A plain uniform draw is what
the evidence covers, and every stratification is a choice we made — which is the thing the seed
exists to remove. The Pilot Regions are what a well-motivated non-uniform choice looks like when it
goes wrong.

**REPRODUCIBILITY IS THE POINT, SO IT IS CHECKABLE.** `check` re-derives the whole BED from the
recorded seed and grid and compares byte for byte. It takes `--n-bins` explicitly so it needs no
store: the bin counts are recorded in `configs/regions/PROVENANCE.md`, and a reader with the seed
and that file can regenerate the identical BED anywhere.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from candi.bench.harness import full_tiling                        # noqa: E402

DEFAULT_RESOLUTION = 25


def parse_n_bins(text: str) -> Dict[str, int]:
    """`chr20=2577766,chr21=1868399` -> dict, order preserved."""
    out: Dict[str, int] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        c, _, n = part.partition("=")
        if not n:
            raise SystemExit(f"--n-bins wants chrom=count pairs; got {part!r}")
        out[c.strip()] = int(n)
    if not out:
        raise SystemExit("--n-bins is empty")
    return out


def eligible_starts(n_bins: Dict[str, int], context_bins: int) -> List[Tuple[str, int]]:
    """`[(chrom, start_bin), …]` — the FULL-COVERAGE window plan, in chromosome order.

    This is the pool, and it is the whole reason the draw is an estimator of the full number rather
    than of something adjacent to it.
    """
    return [(c, int(s)) for c in n_bins for s in full_tiling(n_bins[c], context_bins)]


def draw(n_bins: Dict[str, int], *, windows: int, seed: int, context_bins: int
         ) -> List[Tuple[str, int]]:
    """`windows` starts drawn uniformly WITHOUT replacement from the full-coverage plan.

    Sorted by (chromosome order, start) so the BED is stable — a set has no order, and a file does.
    """
    pool = eligible_starts(n_bins, context_bins)
    if windows > len(pool):
        raise SystemExit(f"asked for {windows} windows; the plan only has {len(pool)}")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=windows, replace=False)
    order = {c: i for i, c in enumerate(n_bins)}
    return sorted((pool[int(i)] for i in idx), key=lambda cs: (order[cs[0]], cs[1]))


def render(picked: Sequence[Tuple[str, int]], *, context_bins: int, resolution: int) -> str:
    """BED4, one interval per WINDOW.

    One window per interval is what makes the round trip exact: on the bin grid the interval spans
    `[s, s + context_bins)` exactly, so `RegionSet.contained_starts` admits that one start and no
    other. A BED of merged or wider intervals would admit a different set than was drawn.
    """
    lines = [f"{c}\t{s * resolution}\t{(s + context_bins) * resolution}\tW{i:04d}"
             for i, (c, s) in enumerate(picked)]
    return "\n".join(lines) + "\n"


def build(n_bins: Dict[str, int], *, windows: int, seed: int, context_bins: int,
          resolution: int) -> str:
    return render(draw(n_bins, windows=windows, seed=seed, context_bins=context_bins),
                  context_bins=context_bins, resolution=resolution)


def n_bins_from_store(regime_path: str) -> Tuple[Dict[str, int], int]:
    from candi.store.regime import Regime
    from candi.store.reader import CorpusStore

    r = Regime.from_file(regime_path)
    corpus = CorpusStore(r.store)
    try:
        return {c: int(corpus.n_bins(c)) for c in r.eval_chroms}, int(r.context_bins)
    finally:
        corpus.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name in ("make", "check"):
        p = sub.add_parser(name)
        p.add_argument("--windows", type=int, required=True)
        p.add_argument("--seed", type=int, required=True)
        p.add_argument("--context-bins", type=int, default=None)
        p.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
        p.add_argument("--store", default=None, help="regime json — read eval chroms and n_bins")
        p.add_argument("--n-bins", default=None, help="chr20=…,chr21=… instead of --store")
        p.add_argument("--out" if name == "make" else "--bed", required=True)

    a = ap.parse_args(argv)
    if (a.store is None) == (a.n_bins is None):
        raise SystemExit("exactly one of --store / --n-bins")
    if a.store:
        n_bins, ctx = n_bins_from_store(a.store)
        context_bins = a.context_bins or ctx
    else:
        n_bins = parse_n_bins(a.n_bins)
        if a.context_bins is None:
            raise SystemExit("--context-bins is required with --n-bins")
        context_bins = a.context_bins

    text = build(n_bins, windows=a.windows, seed=a.seed, context_bins=context_bins,
                 resolution=a.resolution)
    sha = hashlib.sha256(text.encode()).hexdigest()
    pool_n = len(eligible_starts(n_bins, context_bins))
    scored = a.windows * context_bins
    total = sum(n_bins.values())

    if a.cmd == "make":
        Path(a.out).write_text(text, encoding="utf-8")
        print(f"wrote {a.out}")
    else:
        got = Path(a.bed).read_text(encoding="utf-8")
        if got != text:
            print(f"MISMATCH: {a.bed} is not what seed={a.seed} windows={a.windows} produces "
                  f"on this grid", file=sys.stderr)
            return 1
        print(f"OK {a.bed} re-derives exactly from seed={a.seed}")

    print(f"  seed          {a.seed}")
    print(f"  windows       {a.windows} of {pool_n} in the full-coverage plan "
          f"({a.windows / pool_n:.3%})")
    print(f"  chroms        {', '.join(f'{c}={n}' for c, n in n_bins.items())}")
    print(f"  context_bins  {context_bins} ({context_bins * a.resolution} bp)")
    print(f"  scored bins   {scored:,} of {total:,} ({scored / total:.3%})")
    print(f"  sha256        {sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
