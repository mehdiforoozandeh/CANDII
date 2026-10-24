"""t128 — train one run of the symmetric encoder-decoder task table, depth 7.5M held out.

    python tools/t128/train.py train-index <manifest.tsv> <covariates.tsv> <data_dir> <runs_dir>
        <index> [--max-steps N] [--eval-every N] [--device auto|cpu|cuda]

The index-th row of `t128.pairs.tasks()` is trained by `t128.train_ds.run_training_ds` with the
setting of its arm (`t128.pairs.SETTING_BY_ARM`) and `holdout=t128.pairs.HOLDOUT`. Score with
t118's `ladder/score.py trained` and `law` (the hold-out is read from the run's config.json).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[1]
for _p in (_TOOLS / "t118", _TOOLS):    # `ladder` from tools/t118, `t126` / `t128` from tools
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from t128 import pairs as tp  # noqa: E402
from t128 import train_ds  # noqa: E402


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
    table = tp.tasks()
    if not 0 <= args.index < len(table):
        raise IndexError(f"task index {args.index} outside 0..{len(table) - 1}")
    t = table[args.index]
    train_ds.run_training_ds(args.manifest, args.covariates, args.data_dir, args.runs_dir,
                             tp.SETTING_BY_ARM[t["rung"]], t["g"], t["model"], t["seed"],
                             holdout=tp.HOLDOUT, max_steps=args.max_steps,
                             eval_every=args.eval_every, device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
