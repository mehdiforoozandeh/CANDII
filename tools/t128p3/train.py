"""t128 phase 3 — train one run of the two-arm task table, depth 7.5M held out, 15,000 steps.

    python tools/t128p3/train.py train-index <manifest.tsv> <covariates.tsv> <data_dir> <runs_dir>
        <index> [--max-steps N] [--eval-every N] [--device auto|cpu|cuda]

The index-th row of `t128p3.pairs.tasks()`: old rung D (form `D`) trained by
`ladder.train.run_training` at the arm's sizes (`t128p3.pairs.SETTING_BY_ARM`: `g_hidden`,
`f_channels`), run name `<arm>_<g>_counts_real_s<seed>` (`arm` = Dbigg or Dbigf),
`holdout=t128p3.pairs.HOLDOUT` and the setting's `max_steps` unless `--max-steps` is given.
Score with t118's `ladder/score.py trained` and `law`; read through the base with
`tools/t128/viabase.py run`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[1]
for _p in (_TOOLS / "t118", _TOOLS):    # `ladder` from tools/t118, `t128p3` from tools
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ladder import train as ltrain  # noqa: E402
from t128p3 import pairs as tp  # noqa: E402


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
    s = tp.SETTING_BY_ARM[t["rung"]]
    ltrain.run_training(args.manifest, args.covariates, args.data_dir, args.runs_dir, "D",
                        t["g"], "counts", "real", t["seed"],
                        max_steps=args.max_steps or s["max_steps"], eval_every=args.eval_every,
                        device=args.device, holdout=tp.HOLDOUT, g_hidden=s["g_hidden"],
                        f_channels=s["f_channels"], arm=s["arm"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
