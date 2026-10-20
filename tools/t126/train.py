"""t126 — train one run of the encoder-decoder rung A task table, depth 7.5M held out.

    python tools/t126/train.py train-index <manifest.tsv> <covariates.tsv> <data_dir> <runs_dir>
        <index> [--max-steps N] [--eval-every N] [--device auto|cpu|cuda]

The index-th row of `t126.pairs.tasks()` is trained by t118's `ladder.train.run_training` with
`holdout=t126.pairs.HOLDOUT`; every other setting (schedule, sampler, chromosome split, early
stopping, run dir) is t118 row 1's. Score with t118's `ladder/score.py trained` and `law` (the
hold-out is read from the run's config.json).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[1]
for _p in (_TOOLS / "t118", _TOOLS):    # `ladder` from tools/t118, `t126` from tools
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ladder import pairs as lpairs  # noqa: E402
from ladder import train as ltrain  # noqa: E402
from t126 import pairs as tp  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train-index")
    for a in ("manifest", "covariates", "data_dir", "runs_dir"):
        p.add_argument(a)
    p.add_argument("index", type=int)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = ap.parse_args(argv)
    table = tp.tasks(lpairs.read_manifest(args.manifest))
    if not 0 <= args.index < len(table):
        raise IndexError(f"task index {args.index} outside 0..{len(table) - 1}")
    t = table[args.index]
    ltrain.run_training(args.manifest, args.covariates, args.data_dir, args.runs_dir, t["rung"],
                        t["g"], t["space"], t["model"], t["seed"], max_steps=args.max_steps,
                        eval_every=args.eval_every, device=args.device, holdout=tp.HOLDOUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
