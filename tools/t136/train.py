"""t136 — train one run of the arm-to-arm grid (Darm1 / Darm2), depth 7.5M held out, 12,000 steps.

    python tools/t136/train.py train-index <manifest.tsv> <covariates.tsv> <data_dir> <runs_dir>
        <index> [--max-steps N] [--eval-every N] [--device auto|cpu|cuda]

The index-th row of `t136.pairs.tasks()`: old rung D (form `D`, g hidden 64, f 32 channels) trained
by `ladder.train.run_training` with `holdout=t136.pairs.HOLDOUT`, run name
`<arm>_<g>_counts_real_s<seed>` (`arm` = Darm1 or Darm2), and the arm's arm->arm pairs added to
training (`extra_pairs=t136.pairs.extra_pairs(rows, g, arm)`, recorded in config.json). The
setting's `max_steps` unless `--max-steps` is given. Score with t118's `ladder/score.py trained`
and `law --split`; the extra pairs are in the law set and the readings subtract them.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[1]
for _p in (_TOOLS / "t118", _TOOLS):    # `ladder` from tools/t118, `t136` from tools
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ladder import pairs as lpairs  # noqa: E402
from ladder import train as ltrain  # noqa: E402
from t136 import pairs as tp  # noqa: E402


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
    rows = lpairs.read_manifest(args.manifest)
    ltrain.run_training(args.manifest, args.covariates, args.data_dir, args.runs_dir, "D",
                        t["g"], "counts", "real", t["seed"],
                        max_steps=args.max_steps or s["max_steps"], eval_every=args.eval_every,
                        device=args.device, holdout=tp.HOLDOUT, g_hidden=s["g_hidden"],
                        f_channels=s["f_channels"], arm=s["arm"],
                        extra_pairs=tp.extra_pairs(rows, t["g"], s["arm"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
