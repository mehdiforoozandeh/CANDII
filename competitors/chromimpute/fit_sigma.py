"""fit_sigma — the §6.1 σ-table for a point-only rival, fitted on V-pair residuals.

    python fit_sigma.py <scores_p1.json> --out sigma.json [--method ChromImpute]

`RIVALS_PLAN.md` B1a turns a rival's point track into a homoscedastic Gaussian: one σ per
method × assay, so the pval arm can carry a forecast distribution and enter the CRPS tier at all.
§6.1 defines it exactly:

    sigma^2 = mean over V-pair tracks of (signal_mu - truth)^2,
              pooled over all bins of the P1 eval chromosomes, in -log10 p space

**That quantity is already computed.** The bench's `mse` for the pval arm is
`mean((truth - pred)^2)` (`bench/eic.py::mse`) over every bin of the track, with
`signal_target_transform="none"` — i.e. in raw `-log10 p`, which is the space §6.1 names. So the fit
reads the P1 scores file rather than re-walking the store: pooling is the `n_points`-weighted mean
of the per-track `mse`, and the σ is its square root. Re-deriving it from the arrays would compute
the same float twice and give a second number to keep in sync.

Read the **P1** scores file, not P2. §6.1 says the eval chromosomes of the declared-pair protocol,
and the B-pair run reuses whatever this writes **unchanged** — that reuse is what makes the B-pair
CRPS leak-free, and it only means something if the table is pinned to one fitting panel.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence


def fit(scores: dict) -> Dict[str, dict]:
    """`{assay: {sigma, n_tracks, n_points, mse_pooled}}` from a scores json's `per_track` block."""
    acc: Dict[str, List[tuple]] = defaultdict(list)
    for key, arms in scores["per_track"].items():
        pv = arms.get("pval")
        if not pv:
            continue
        if pv.get("signal_target_transform") != "none":
            raise ValueError(
                f"{key}: signal_target_transform={pv.get('signal_target_transform')!r}. §6.1 fits "
                f"sigma in raw -log10 p; a track scored through a transform is a different space.")
        acc[pv["assay"]].append((float(pv["mse"]), int(pv["n_points"])))
    out: Dict[str, dict] = {}
    for assay, rows in sorted(acc.items()):
        n_tot = sum(n for _, n in rows)
        mse_pooled = sum(m * n for m, n in rows) / n_tot
        out[assay] = {
            "sigma": math.sqrt(mse_pooled),
            "n_tracks": len(rows),
            "n_points": n_tot,
            "mse_pooled": mse_pooled,
        }
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fit_sigma.py", description="RIVALS_PLAN.md §6.1 sigma-table")
    p.add_argument("scores", help="the P1 scores json (declared pairs, eval chromosomes)")
    p.add_argument("--out", required=True)
    p.add_argument("--method", default=None, help="defaults to provenance.method in the scores file")
    p.add_argument("--fitted-on", default="regime.eic_val eval_pairs")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scores = json.loads(Path(args.scores).read_text(encoding="utf-8"))
    method = args.method or scores.get("provenance", {}).get("method")
    if not method:
        raise SystemExit("no method: pass --method or use a scores file with provenance.method")
    per_assay = fit(scores)

    table = {
        "method": method,
        "fitted_on": args.fitted_on,
        "sigma": {a: v["sigma"] for a, v in per_assay.items()},
        "notes": (
            "RIVALS_PLAN.md §6.1 B1a. sigma = sqrt(pooled mean squared residual) per assay, over "
            "the V-pair tracks of the P1 protocol, in -log10 p. IN-SAMPLE for the V panel: any "
            "table showing V-pair CRPS for this method must say so. The B-pair run reuses this "
            "table unchanged, and that one is leak-free."),
        "fit_detail": per_assay,
        "source_scores": str(Path(args.scores).name),
    }
    Path(args.out).write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")

    w = max(len(a) for a in per_assay)
    print(f"{'assay':<{w}} {'n_tr':>4} {'n_points':>12} {'mse_pooled':>12} {'sigma':>10}")
    for a, v in per_assay.items():
        print(f"{a:<{w}} {v['n_tracks']:>4} {v['n_points']:>12,} "
              f"{v['mse_pooled']:>12.4f} {v['sigma']:>10.4f}")
    print(f"\n{len(per_assay)} assays -> {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
