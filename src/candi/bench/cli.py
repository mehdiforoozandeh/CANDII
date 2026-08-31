"""`python -m candi.bench` — the one CLI (`EVAL_PLAN.md` §1).

    python -m candi.bench --store regime.json --ckpt model.pt --arch-from run.json --out bench.json

One command, five blocks, both arms. What is deliberately ABSENT is as much of the contract as what
is present:

- **no `--eval-budget`, no `--max-batches`** (D2). The suite scores every 25 bp bin of every eval
  chromosome. Those two flags are how the old `eval.py` shrank a run, and both of them shrink it
  *unevenly* — `--max-batches` drops whole targets without a word, and on a 38-pair panel a habit
  like `--eval-max-batches 12` scored 12 targets and silently dropped 26. Passing either here is an
  error with that explanation attached, not an unrecognised argument.
- **no `--fix` for the organizers' quirks** (D16). `mseprom` counting a clipped slice while
  `msegene` counts an unclipped one is faithfully reproduced, and there is no flag to turn it off,
  because a "fixed" number is not comparable to the published table.

The architecture flags mirror `eval.py::main` so a checkpoint that could be scored there can be
scored here, and `--arch-from` is still the right way: every geometry, norm, FiLM and head flag
changes the state_dict, so a retyped flag shows up as a strict-load failure at best and a quietly
different model at worst.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch

from candi.bench import annotations as ann
from candi.bench.harness import ARMS, KINDS, open_source, run_bench
from candi.precision import no_autocast

__all__ = ["build_parser", "jsonable", "main"]

#: Flags that existed on `eval.py` and are refused here rather than silently accepted (D2).
RETIRED = {
    "--eval-budget": "D2 — the suite scores every 25 bp bin of the eval chromosomes. There is no "
                     "budget to set. The ONE sampled quantity is the C-index (D3), whose pair "
                     "count is --c-index-pairs and which always ships with its Monte-Carlo SE.",
    "--max-batches": "D2 — truncating the batch stream drops whole TARGETS unevenly: the old eval "
                     "advanced one (T_, V_/B_) pair per window batch, so --max-batches 12 on a "
                     "38-pair panel scored 12 targets and left 26 unscored. Use --biosamples to "
                     "choose a panel deliberately instead.",
    "--eval-max-batches": "see --max-batches.",
    "--batches-per-pair": "D2 — window thinning. Score the whole chromosome or name a smaller "
                          "chromosome set with --chroms.",
}


def jsonable(obj: Any) -> Any:
    """numpy/torch -> plain JSON. `bench` does not import `candi.train`, so it carries its own.

    `train.py::_jsonable` is the same function, and copying eleven lines is the correct price for
    not making every evaluation run import the training loop (`EVAL_PLAN.md` §11.2).
    """
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, torch.Tensor):
        return jsonable(obj.detach().cpu().tolist())
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        # JSON has no NaN or Infinity. `json.dump` will happily write the bare tokens `NaN` and
        # `Infinity`, which are not JSON and which several readers reject outright; `null` is the
        # honest spelling, and the E-block produces real nans on purpose (quirk 8 — an annotation
        # set that selects zero bins).
        return float(obj) if np.isfinite(obj) else None
    return obj


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m candi.bench",
        description="Score a CANDI checkpoint: the nine EIC measures, the four post-hoc measures, "
                    "distributional, binary/peak, and covariate sensitivity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_argument_group("data (exactly one)")
    src.add_argument("--h5", default=None, help="a baked candi.prep.bake h5")
    src.add_argument("--store", default=None,
                     help="a CANDI_STORE REGIME FILE (json). The regime's own `store` key names "
                          "the corpus root, so no second flag is needed.")
    src.add_argument("--regime", default="type1", choices=["type1", "type2_loci"],
                     help="h5 window regime. Nothing to do with --store.")
    src.add_argument("--chroms", default=None,
                     help="comma-separated subset of the eval chromosomes (default: all of them)")
    src.add_argument("--held-out-chroms", default=None,
                     help="comma-separated: the RANKED scope (plan/BENCHMARK_DESIGN.md \u00a74), e.g. "
                          "chr20,chr21,chr22. Must be a subset of what is scored. Given a proper "
                          "subset, one pass yields two aggregations -- `macro`/`panels` are the "
                          "held-out numbers and a `genome_wide` block carries the same over every "
                          "scored chromosome. Omitted, or naming everything scored, there is one "
                          "scope and no genome-wide block: that is \u00a74's blanking rule, and a "
                          "method fit at every position is run that way on purpose.")
    src.add_argument("--biosamples", default=None,
                     help="store only: comma-separated eval biosamples (default: the regime's)")

    mdl = p.add_argument_group("checkpoint")
    mdl.add_argument("--ckpt", required=True)
    mdl.add_argument("--arch-from", default=None,
                     help="a run's own JSON. Reads config.arch and rebuilds the EXACT model that "
                          "wrote the checkpoint. Prefer this over retyping the flags below.")
    mdl.add_argument("--offset", "--arm", dest="offset", default="on",
                     choices=["on", "off", "offset_on", "offset_off"])
    mdl.add_argument("--depth-center", type=float, default=None)
    mdl.add_argument("--d-model", type=int, default=0)
    mdl.add_argument("--nhead", type=int, default=4)
    mdl.add_argument("--embed-dim", type=int, default=32)
    mdl.add_argument("--dropout", type=float, default=0.1)
    mdl.add_argument("--n-transformer-layers", type=int, default=2)
    mdl.add_argument("--decoder-lane", type=int, default=8)
    mdl.add_argument("--deconv-norm", default="lane", choices=["lane", "group"])
    mdl.add_argument("--heads", default="count",
                     help="comma-separated. `count,signal` is REQUIRED for the pval arm (D1); the "
                          "count arm works on every checkpoint. Must match the checkpoint.")
    mdl.add_argument("--meta-embed-layernorm", default="on", choices=["on", "off"],
                     help="MUST match how the checkpoint was trained, or the strict load fails.")
    mdl.add_argument("--device", default=None, help="default: cuda when available, else cpu")

    run = p.add_argument_group("what to run")
    run.add_argument("--out", required=True, help="results JSON to write")
    run.add_argument("--kinds", default="impute",
                     help=f"comma-separated from {KINDS}. `impute` is the headline and the only "
                          f"one EVAL_PLAN.md §4 names; `denoise` keys carry a fourth field.")
    run.add_argument("--blocks", default="E,P,D,B,C",
                     help="E=the nine EIC measures, P=post-hoc, D=distributional, B=binary/peak, "
                          "C=covariate sensitivity")
    run.add_argument("--batch-windows", type=int, default=4,
                     help="windows per forward pass. Throughput only — it cannot change a number, "
                          "because every window is scored either way.")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--c-index-pairs", type=int, default=200_000,
                     help="D3 — the one documented exception to whole-track scoring. The C-index "
                          "is pairwise (1.7e12 pairs on chr21), so it is estimated over this many "
                          "seeded pairs and always reported beside its Monte-Carlo SE.")
    run.add_argument("--c-windows", type=int, default=8,
                     help="C-block: counterfactual contexts, spread across the chromosome")
    run.add_argument("--c-resamples", type=int, default=50,
                     help="C1: randomization-test resamples per covariate per null")
    run.add_argument("--varpool", default=None,
                     help="root of the D7 msevar variance pools. Without it msevar is ABSENT "
                          "rather than the organizers' bare 0.0.")
    run.add_argument("--varpool-corpus", default="eic")
    run.add_argument("--competitors", default=None,
                     help="a competitor score table (csv/csv.gz with bootstrap_id, cell, assay, "
                          "team_id and the nine measures). Without it the rank aggregation is "
                          "inert (D4).")
    run.add_argument("--with-curve", action="store_true",
                     help="also emit the full correspondence curve per track (large)")
    run.add_argument("--quiet", action="store_true")
    return p


def _reject_retired(argv: Sequence[str]) -> None:
    for a in argv:
        flag = a.split("=", 1)[0]
        if flag in RETIRED:
            raise SystemExit(f"{flag} does not exist in candi.bench. {RETIRED[flag]}")


def _build_model(a, source, device):
    from candi.model import build_model, build_model_from_arch, parse_heads

    depth_center = a.depth_center if a.depth_center is not None else source.depth_center()
    if a.arch_from:
        arch = (json.loads(Path(a.arch_from).read_text()).get("config") or {}).get("arch")
        if not arch:
            raise SystemExit(
                f"{a.arch_from} has no config.arch block — it predates --arch-from. Pass the "
                f"architecture flags by hand, matching the run's config.")
        if a.depth_center is not None:
            arch["depth_center"] = float(a.depth_center)
        model = build_model_from_arch(arch)
    else:
        model = build_model(
            embed_dim=a.embed_dim, dropout=a.dropout,
            n_transformer_layers=a.n_transformer_layers, decoder_lane=a.decoder_lane,
            deconv_norm=a.deconv_norm, depth_center=depth_center,
            use_offset=a.offset in ("on", "offset_on"),
            num_assays=len(source.assays), context_length=source.context_bins,
            resolution=source.resolution, d_model=a.d_model, nhead=a.nhead,
            heads=parse_heads(a.heads),
            meta_embed_layernorm=(a.meta_embed_layernorm == "on"))
    model = model.to(device)
    model.load_state_dict(torch.load(a.ckpt, map_location=device), strict=True)
    model.eval()
    return model


def _ranking(result: dict, competitors: str) -> dict:
    """Place CANDI in a competitor field. Only ever called when a table is supplied (D4).

    **CANDI's cells are not bootstrapped and the competitors' are.** The published table carries ten
    bootstrap resamples per team per experiment; this inserts CANDI's whole-track score into all ten,
    so its ten bootstrap ranks are identical. Stage 4 takes each team's SECOND-BEST bootstrap rank,
    which rewards a team that got lucky twice — so a competitor gains from its own spread and CANDI
    gains nothing from a spread it does not have. The bias is **against** CANDI, and it is recorded
    in the output rather than left for a reader to work out.
    """
    import csv
    import gzip
    from collections import defaultdict

    from candi.bench.eic import MEASURES
    from candi.bench.harness import SCOPE_HELD_OUT, panel_of
    from candi.bench.ranking import MISSING_SCORE, aggregate_ranks

    path = Path(competitors)
    opener = gzip.open if path.suffix == ".gz" else open
    table: dict = defaultdict(lambda: defaultdict(dict))
    with opener(path, "rt") as fh:
        for row in csv.DictReader(fh):
            table[int(row["bootstrap_id"])][row["cell"] + row["assay"]][row["team_id"]] = {
                m: float(row[m]) for m in MEASURES}

    # THE HELD-OUT SCOPE, always. A placement is a claim about loci where nothing was fit, so
    # ranking the genome-wide aggregation would place a memorisation score in a field of
    # imputation scores. `result["per_track"]` IS the held-out scope by construction (see
    # `run_bench`); `result["genome_wide"]` is deliberately not read here.
    ours: dict = {}
    panels: dict = {}
    for key, arms in result["per_track"].items():
        arm = arms.get("pval") or arms.get("count")
        if arm is None or arms is None:
            continue
        fields = key.split("|")
        exp = f"{fields[1]}{fields[2]}"
        ours[exp] = {m: float(arm[m]) for m in MEASURES if m in arm}
        panels[exp] = panel_of(fields[1])
    placed, absent = 0, []
    for b in table:
        for exp, cell in table[b].items():
            if exp in ours:
                cell["CANDI"] = ours[exp]
                placed += 1
            else:
                absent.append(exp)
    agg = aggregate_ranks(table)
    agg["ranker"] = "candi.bench.ranking.aggregate_ranks"
    agg["scope"] = SCOPE_HELD_OUT
    agg["scope_chroms"] = list((result.get("provenance", {}).get("scope") or {})
                               .get("held_out_chroms", []))
    agg["panels_placed"] = {p: sum(1 for e in ours if panels.get(e) == p)
                            for p in ("V", "B") if any(panels.get(e) == p for e in ours)}
    agg["missing_score"] = MISSING_SCORE
    agg["single_ranker_note"] = (
        "plan/BENCHMARK_DESIGN.md 5.3 -- this is the ONLY ranker, for every arm, every regime and "
        "both truths. Quote the resolution limit with every placement: two methods separated by "
        "less than it are not separated. A method missing an arm scores MISSING_SCORE, equal to "
        "the cap, i.e. below-median rather than disqualified. Ranks are computed WITHIN a panel; "
        "never subtract V_ breadth from B_.")
    agg["candi_experiments_placed"] = placed // max(1, len(table))
    agg["candi_experiments_absent"] = sorted(set(absent))[:20]
    agg["candi_bootstrap_caveat"] = (
        "CANDI's whole-track score is repeated across all ten bootstraps while every competitor's "
        "cells are genuinely resampled. Stage 4 keeps each team's SECOND-BEST bootstrap rank, so "
        "competitors profit from their spread and CANDI cannot. The bias runs AGAINST CANDI. "
        "Scoring CANDI on the challenge's own position bootstraps is not implemented (layer 4 is "
        "PR evidence, not a merge gate — EVAL_PLAN.md §6).")
    agg["arm"] = "pval" if any("pval" in a for a in result["per_track"].values()) else "count"
    return agg


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    _reject_retired(argv)
    a = build_parser().parse_args(argv)
    if (a.h5 is None) == (a.store is None):
        raise SystemExit(
            "exactly one of --h5 / --store is required. --h5 <baked.h5> reads a frozen bake; "
            "--store <regime.json> reads a CANDI_STORE through a regime file.")

    kinds = tuple(k.strip() for k in a.kinds.split(",") if k.strip())
    blocks = tuple(b.strip().upper() for b in a.blocks.split(",") if b.strip())
    chroms = tuple(c.strip() for c in a.chroms.split(",")) if a.chroms else None
    held_out = (tuple(c.strip() for c in a.held_out_chroms.split(",") if c.strip())
                if a.held_out_chroms else None)
    bios = tuple(b.strip() for b in a.biosamples.split(",")) if a.biosamples else None

    kw: dict = {"chroms": chroms}
    if a.h5:
        kw["regime"] = a.regime
    else:
        kw["biosamples"] = bios
    source = open_source(h5=a.h5, store=a.store, **kw)
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    try:
        model = _build_model(a, source, device)
        if not a.quiet:
            n = sum(p.numel() for p in model.parameters())
            print(f"[bench] {n:,} params on {device}; {len(source.assays)} assays; "
                  f"eval chroms {list(source.eval_chroms)}", flush=True)
        # EVALUATION IS NEVER AUTOCAST. Every recorded number in this repo is fp32, and a metric
        # measured at a different precision than the one it is compared against is a difference
        # nobody declared.
        with no_autocast(device):
            result = run_bench(
                model, source, device, kinds=kinds, blocks=blocks, seed=a.seed,
                batch_windows=a.batch_windows, c_index_pairs=a.c_index_pairs,
                varpool_root=a.varpool, varpool_corpus=a.varpool_corpus,
                c_windows=a.c_windows, c_resamples=a.c_resamples,
                with_curve=a.with_curve, progress=not a.quiet, held_out_chroms=held_out,
                extra_provenance={"ckpt": str(a.ckpt), "arch_from": a.arch_from,
                                  "heads": a.heads, "offset": a.offset, "device": str(device)})
    finally:
        source.close()

    if a.competitors:
        result["ranking"] = _ranking(result, a.competitors)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jsonable(result), indent=2))
    if not a.quiet:
        for arm in ARMS:
            m = result["macro"].get(arm) or {}
            if m:
                head = ", ".join(f"{k}={m[k]:.4f}" for k in ("mse", "gwcorr", "crps")
                                 if k in m and np.isfinite(m[k]))
                print(f"[bench] macro {arm}: {m['n_tracks']} tracks — {head}", flush=True)
        print(f"[bench] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(main())
