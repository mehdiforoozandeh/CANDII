#!/usr/bin/env python3
"""Derive the TRAINING-RESIDUAL regime a point-only rival's σ is fitted on (D2).

    python tools/sigma_training_regime.py --regime configs/regime.eic_19.json \
        --n-cells 12 --seed 890217 --out <workspace>/regime.eic_19.sigma.json

WHY THIS FILE EXISTS
--------------------
`BENCHMARK_DESIGN.md` §6.1 turns a point-only rival into a homoscedastic Gaussian with one σ per
assay, and the old fitters took that σ from the residuals of the **eval pairs** — the very tracks
the leaderboard then scores. Rule 1 (§12.2) voids every such table: a σ fitted on `V_` truth and
reused on `B_` is still a number the eval panel paid for. This tool writes the regime the honest
fit runs on instead: a seeded sample of the TRAINING cells, each scored against ITSELF, on the
TRAINING chromosomes. Nothing an eval cell holds is opened, so the σ costs the panel nothing.

HOW A SELF-PAIRING IS SPELLED, AND WHY IT IS NOT `eval_pairs`
-------------------------------------------------------------
The obvious spelling — `eval_pairs = [["T_x", "T_x"], …]` — does not load, and neither does the
obvious `eval_chroms`. Three checks in `candi.store.regime` stand in the way, all of them right:

* `_parse_eval_pairs` refuses a pair of a cell with itself ("the target supplies the ground truth
  for assays the input does not have; the same biosample supplies neither").
* `Regime.validate_against` refuses `train_chroms & eval_chroms` ("an eval window that also
  trained is not an evaluation").
* `Regime.validate_against` refuses an `eval_pairs` target that is also in `biosamples.train`.

So the self-pairing is written the one way the code allows: **no `eval_pairs` at all**, with
`biosamples.eval` set to the sampled training cells. That is not a workaround — it is the documented
pre-t80 path, and `bench.harness.StoreSource` states it out loud: with no declared pairing "every
store pair is SELF-PAIRED", `pairs("impute")` returns `Pair(T_x, T_x)` per pool cell and `targets`
returns every assay that cell holds. `train_chroms` is emptied and the source's training slice moves
to `eval_chroms`, which is what makes the residual a training residual.

A useful side effect: `bench.external.score_external` REFUSES a regime with no cross-cell pair, so a
σ regime can never be handed to the scorer by accident. It fits σ and does nothing else.

The output is a derived file that lives beside the run, never in `configs/`.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from candi.store.reader import CorpusStore                            # noqa: E402
from candi.store.regime import Regime                                 # noqa: E402

#: The panels this regime may never name. §6.2 keeps every rival off them, and a σ fitted on one is
#: void under Rule 1 however carefully the rest of the run is done.
EVAL_PANEL_PREFIXES = ("V_", "B_")

#: Exit code for "this regime names an eval-panel cell". The same code `competitors/sigma_pass.py`
#: uses, so a caller can branch on one number wherever the refusal is raised.
EXIT_EVAL_PANEL = 3

COMMENT_PREFIX = "DERIVED training-residual sigma regime"


def sample_cells(pool: Sequence[str], n_cells: int, seed: int) -> List[str]:
    """`n_cells` of `pool`, drawn once with `seed`, in the pool's own order.

    A permutation rather than `choice(replace=False)`: the draw has to be re-derivable months later
    from the seed alone, and `permutation` is the more stable of the two across numpy versions. The
    result is re-sorted into the declared order so two runs write byte-identical files.
    """
    if n_cells > len(pool):
        raise SystemExit(
            f"--n-cells {n_cells} but the regime declares {len(pool)} training biosample(s). The "
            f"sample is drawn without replacement — a cell cannot contribute two residuals.")
    if n_cells <= 0:
        raise SystemExit(f"--n-cells must be positive, got {n_cells}")
    idx = np.random.default_rng(seed).permutation(len(pool))[:n_cells]
    return [pool[i] for i in sorted(int(i) for i in idx)]


def derive(obj: Dict[str, Any], regime: Regime, *, cells: Sequence[str], seed: int,
           n_cells: int, source_path: Path) -> Dict[str, Any]:
    """The derived regime object — the source verbatim, with six keys overridden.

    Everything not named here rides through unchanged (`store`, `assays`, `context_bins`,
    `window_plan`, `dsf`, `kinds`, `seed`), because a σ fitted under a different assay order or a
    different context is not a σ for this run.
    """
    out = dict(obj)
    regions_note = "" if regime.regions is None else (
        " The `regions` block rides along from the source and is INERT to candi.store here "
        "(Regime.windows restricts to it on the train split only, and this file trains on "
        "nothing); competitors.sigma_pass adopts it as the eval scope instead, so the sigma is "
        "fitted over the same BED the transferable parameters were."
    )
    out["_comment"] = (
        f"{COMMENT_PREFIX} written by tools/sigma_training_regime.py from {source_path.name} "
        f"(sha256 {regime.sha256}): {len(cells)} training cell(s) drawn with seed {seed}, each "
        f"SELF-PAIRED (T_x -> T_x) on the source's TRAINING chromosomes, so the residual a sigma is "
        f"fitted on is a training residual and the eval panel pays nothing for it (Rule 1, "
        f"BENCHMARK_DESIGN.md 12.2). The self-pairing carries no `eval_pairs`: "
        f"candi.store.regime refuses a pair of a cell with itself, and refuses train_chroms that "
        f"overlap eval_chroms, so `biosamples.eval` holds the sampled cells and "
        f"bench.harness.StoreSource self-pairs them (its documented no-pairing path). Fit with "
        f"`python -m competitors.sigma_pass`. NOT a scoring regime -- "
        f"bench.external.score_external refuses it, because it declares no cross-cell pair."
        f"{regions_note}"
    )
    out["_source_regime"] = {
        "path": str(source_path),
        "sha256": regime.sha256,
        "train_chroms": list(regime.train_chroms),
        "eval_chroms": list(regime.eval_chroms),
        "n_train_biosamples": len(regime.train_biosamples),
        "sample": {"n_cells": n_cells, "seed": seed, "rule": "default_rng(seed).permutation"},
    }
    out["biosamples"] = {"train": list(regime.train_biosamples), "eval": list(cells)}
    out["eval_pairs"] = []
    # The source's training slice becomes what is SCORED, and nothing trains here. Both halves are
    # required: `validate_against` refuses the two lists sharing a chromosome.
    out["eval_chroms"] = list(regime.train_chroms)
    out["train_chroms"] = []
    if regime.regions is not None:
        # An absolute path, because the derived file lives beside the run and `RegionSet.from_obj`
        # resolves a relative `bed` against the regime file's OWN directory. The sha256 is the same
        # bytes, so a moved BED is still caught.
        #
        # THIS BLOCK IS INERT TO THE LOADER AND IS CARRIED ANYWAY. `Regime.windows` restricts to
        # `regions` on the TRAIN split only, and this file's train split is empty — so nothing in
        # `candi.store` will ever read it here. It is written because `competitors.sigma_pass`
        # ADOPTS it as its `--eval-regions` scope: without the block the fit would silently go
        # genome-wide over `eic_pilot`'s 18 training chromosomes, where the rivals hold no factors.
        out["regions"] = {**dict(obj.get("regions") or {}),
                          "bed": str(Path(regime.regions.resolved).resolve())}
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--regime", required=True, help="the source regime, e.g. configs/regime.eic_19.json")
    p.add_argument("--n-cells", type=int, default=12, help="training cells to draw (default 12)")
    p.add_argument("--seed", type=int, default=890217, help="the draw's seed (default 890217)")
    p.add_argument("--out", required=True, help="the derived regime, beside the run")
    a = p.parse_args(argv)

    source_path = Path(a.regime)
    regime = Regime.from_file(source_path)
    obj = json.loads(source_path.read_text(encoding="utf-8"))

    pool = list(regime.train_biosamples)
    if not pool:
        raise SystemExit(f"{source_path} declares no `biosamples.train`; there is nothing to fit on")
    if not regime.train_chroms:
        raise SystemExit(f"{source_path} declares no `train_chroms`; the residual has no positions")
    banned = [b for b in pool if b.startswith(EVAL_PANEL_PREFIXES)]
    if banned:
        print(f"REFUSING {source_path}: `biosamples.train` names eval-panel cell(s) {banned[:5]}. "
              f"A sigma fitted on a {'/'.join(EVAL_PANEL_PREFIXES)} track is void under Rule 1 "
              f"however it is drawn.", file=sys.stderr)
        return EXIT_EVAL_PANEL

    cells = sample_cells(pool, a.n_cells, a.seed)
    out_obj = derive(obj, regime, cells=cells, seed=a.seed, n_cells=a.n_cells,
                     source_path=source_path)
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_obj, indent=2) + "\n", encoding="utf-8")

    # It must still load, and it must still validate against the corpus it names — the same check
    # `tools/declare_eval_pairs.py` runs on the regime it writes. A derived file that only looks
    # right is the failure this catches.
    derived = Regime.from_file(out_path)
    corpus = CorpusStore(Path(derived.store))
    try:
        derived.validate_against(corpus)
    finally:
        corpus.close()

    print(f"{COMMENT_PREFIX}: {len(cells)} cell(s), seed {a.seed}")
    for c in cells:
        print(f"  {c}")
    print(f"chroms   {list(derived.eval_chroms)}")
    if derived.regions is None:
        print("regions  none (full coverage)")
    else:
        print(f"regions  {derived.regions.resolved}")
        print("         competitors.sigma_pass ADOPTS this as its eval scope — `Regime.windows` "
              "applies a `regions` block to the train split only, and this file trains on nothing.")
    print(f"wrote    {out_path}  sha256 {hashlib.sha256(out_path.read_bytes()).hexdigest()[:12]}  "
          f"{_dt.datetime.now(_dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    return 0


if __name__ == "__main__":                                            # pragma: no cover
    raise SystemExit(main())
