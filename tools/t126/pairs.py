"""t126 — the encoder-decoder rung A run (plan/T126_ENCDEC_A.md): arms, hold-out, task table.

**Arms.** `A` = t118 rung A as is (g reads [c, c']), retrained under this run's training set;
`AED` = encoder-decoder rung A, beta learned; `AED1` = encoder-decoder rung A, beta = 1
(`ladder/fforms/form_aed.py`, `form_aed1.py`, `ladder.model.GEncDec`). Counts only.

**Hold-out.** `HOLDOUT` = the depth 7.5M arm of every track: every base<->7.5M pair leaves
training (`ladder.pairs.train_pairs(rows, g, HOLDOUT)`) and is scored only (kind `heldout` in
scores.json); arm->arm pairs with a 7.5M product stay in t118's law pass.

**Tasks.** 144 = ARMS x G_IDS x MODELS x SEEDS, arm slowest and seed fastest:
`index = arm_i*48 + g_i*6 + model_i*3 + seed`, `run_name = f"{arm}_{g}_counts_{model}_s{seed}"`
(t118's formula; index 18 = A_C19M16_counts_real_s0, 66 = AED_..., 114 = AED1_...). G_IDS are
t118's (7 per-track g's, then "all"); MODELS = real and the no-covariates twin.

    python tools/t126/pairs.py tasks <MANIFEST.tsv>
    python tools/t126/pairs.py train-pairs <MANIFEST.tsv> <g>
    python tools/t126/pairs.py heldout-pairs <MANIFEST.tsv> <g>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_T118 = Path(__file__).resolve().parents[1] / "t118"
if str(_T118) not in sys.path:          # import convention: tools/t118 holds `ladder`
    sys.path.insert(0, str(_T118))

from ladder import pairs as _lp  # noqa: E402

ARMS = ("A", "AED", "AED1")
G_IDS = _lp.G_IDS
MODELS = ("real", "nocov")
SEEDS = (0, 1, 2)
SPACE = "counts"
#: (arm, level) of the products held out of training, on every track (PI 2026-10-19)
HOLDOUT = (("depth", "7.5M"),)
TASK_KEYS = ("index", "rung", "g", "space", "model", "seed", "run_name")


def tasks(rows: list[dict] | None = None) -> list[dict]:
    """The 144-row task table; a pure function of the constants (rows kept for the signature)."""
    out = []
    for ai, arm in enumerate(ARMS):
        for gi, g in enumerate(G_IDS):
            for mi, model in enumerate(MODELS):
                for seed in SEEDS:
                    out.append({"index": ai * 48 + gi * 6 + mi * 3 + seed, "rung": arm, "g": g,
                                "space": SPACE, "model": model, "seed": seed,
                                "run_name": f"{arm}_{g}_{SPACE}_{model}_s{seed}"})
    assert [t["index"] for t in out] == list(range(len(out)))
    return out


def train_pairs(rows: list[dict], g: str) -> list[dict]:
    return _lp.train_pairs(rows, g, HOLDOUT)


def heldout_pairs(rows: list[dict], g: str) -> list[dict]:
    return _lp.heldout_pairs(rows, g, HOLDOUT)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("tasks", "train-pairs", "heldout-pairs"):
        p = sub.add_parser(name)
        p.add_argument("manifest")
        if name != "tasks":
            p.add_argument("g")
    args = ap.parse_args(argv)
    rows = _lp.read_manifest(args.manifest)
    if args.cmd == "tasks":
        sys.stdout.write("\t".join(TASK_KEYS) + "\n" + _lp._tsv(tasks(rows), TASK_KEYS))
    else:
        fn = train_pairs if args.cmd == "train-pairs" else heldout_pairs
        sys.stdout.write("\t".join(_lp.PAIR_KEYS) + "\n" + _lp._tsv(fn(rows, args.g), _lp.PAIR_KEYS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
