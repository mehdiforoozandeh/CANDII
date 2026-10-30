"""t136 — the arm-to-arm training grid for h28 (plan/T136_ARM2ARM_GRID.md): settings, the extra
training pairs of each arm, the held-out score set and the 42-row task table.

**Arms.** Both are phase 2's old D recipe (form rung D, g hidden 64, f 32 channels, n_theta 256,
NB loss, 12,000 steps, depth 7.5M held out; `tools/t128p3/pairs.py` `_s3b(2, "D", ...)`) plus
extra arm->arm training pairs (`ladder.train.run_training(..., extra_pairs=...)`):
  1. Darm1: the two ordered pairs between the track's `DARM1_ENDPOINTS` products
     (depth 15M -> depth 3.75M and back), taken from `ladder.pairs.law_pairs(rows, g)`.
  2. Darm2: every `law_pairs` pair of the track whose source and target are both not held out.
The base<->arm training pairs are `ladder.pairs.train_pairs(rows, g, HOLDOUT)` as in phase 2;
`train_pairs(rows, g, arm)` = those + `extra_pairs(rows, g, arm)`, in that order.

**Held-out score set.** `heldout_score_pairs(rows, g)` = every `law_pairs` pair with a held-out
product as source or target (base <-> 7.5M comes from `score_trained`'s heldout records).

**Tasks.** 42 rows: arm slowest (Darm1, Darm2), then g in `G_IDS` (the 7 per-track g), then seed;
`index = arm_i*21 + g_i*3 + seed`, `run_name = f"{arm}_{g}_counts_real_s{seed}"`.
`n_extra_pairs` = len(extra_pairs) on the manifest given (None when the table is built without
rows). Anchors: 0 = Darm1_<G_IDS[0]>_counts_real_s0, 21 = Darm2_<G_IDS[0]>_counts_real_s0.

    python tools/t136/pairs.py tasks <MANIFEST.tsv>
    python tools/t136/pairs.py train-pairs <MANIFEST.tsv> <g> <arm>
    python tools/t136/pairs.py extra-pairs <MANIFEST.tsv> <g> <arm>
    python tools/t136/pairs.py heldout-pairs <MANIFEST.tsv> <g>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[1]
for _p in (_TOOLS / "t118", _TOOLS):    # import convention: tools/t118 holds `ladder`
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ladder import pairs as _lp  # noqa: E402

#: (arm, level) of the products held out of training, on every track (as t126, t128, t128p2)
HOLDOUT = (("depth", "7.5M"),)
SEEDS = (0, 1, 2)
SPACE = "counts"
G_IDS = tuple(g for g in _lp.G_IDS if g != "all")
assert len(G_IDS) == 7, G_IDS
#: the two (arm, level) products whose ordered pairs Darm1 adds (ratio 4, never the held-out 7.5M)
DARM1_ENDPOINTS = (("depth", "15M"), ("depth", "3.75M"))
TASK_KEYS = ("index", "rung", "form_rung", "g", "space", "model", "seed", "run_name", "setting",
             "g_hidden", "f_channels", "n_extra_pairs")


def _s(number, arm):
    return {"number": number, "arm": arm, "rung": "D", "g_hidden": 64, "f_channels": 32,
            "n_theta": 256, "loss": "nb", "max_steps": 12000, "val_nll_rule": "t118 cross NLL"}


SETTINGS = (_s(1, "Darm1"), _s(2, "Darm2"))
SETTING_BY_ARM = {s["arm"]: s for s in SETTINGS}
ARMS = tuple(s["arm"] for s in SETTINGS)
ARM_ARRAYS = {"Darm1": "0-20", "Darm2": "21-41"}
#: the C19M16 seed-0 run of each arm (9, 30), as the t128p3 smoke used C19M16
SMOKE = tuple(ai * 21 + G_IDS.index("C19M16") * 3 for ai in range(2))


def _key(p: dict) -> tuple:
    return (p["arm_src"], p["level_src"]), (p["arm_tgt"], p["level_tgt"])


def extra_pairs(rows: list[dict], g: str, arm: str) -> list[dict]:
    """The arm->arm pairs `arm` adds to training on track g, in `law_pairs` order."""
    if g == "all":
        raise ValueError("t136 trains the per-track g only, not 'all'")
    law = _lp.law_pairs(rows, g)
    if arm == "Darm1":
        ends = set(DARM1_ENDPOINTS)
        out = [p for p in law if set(_key(p)) == ends]
        if len(out) != 2:
            raise ValueError(f"g {g}: {len(out)} Darm1 pairs between {DARM1_ENDPOINTS}, need 2")
        return out
    if arm == "Darm2":
        return [p for p in law if not _lp._held(p, HOLDOUT)]
    raise ValueError(f"arm {arm!r} not in {ARMS}")


def train_pairs(rows: list[dict], g: str, arm: str) -> list[dict]:
    """Phase 2's base<->arm training pairs with the hold-out, then the arm's extra pairs."""
    return _lp.train_pairs(rows, g, HOLDOUT) + extra_pairs(rows, g, arm)


def heldout_score_pairs(rows: list[dict], g: str) -> list[dict]:
    """Every arm->arm pair of g with a held-out product as source or target (the readings' set)."""
    return [p for p in _lp.law_pairs(rows, g) if _lp._held(p, HOLDOUT)]


def tasks(rows: list[dict] | None = None) -> list[dict]:
    """The 42-row task table; `n_extra_pairs` is counted on `rows` (None without rows)."""
    out = []
    for ai, s in enumerate(SETTINGS):
        for gi, g in enumerate(G_IDS):
            n = None if rows is None else len(extra_pairs(rows, g, s["arm"]))
            for seed in SEEDS:
                out.append({"index": ai * 21 + gi * 3 + seed, "rung": s["arm"],
                            "form_rung": s["rung"], "g": g, "space": SPACE, "model": "real",
                            "seed": seed, "run_name": f"{s['arm']}_{g}_{SPACE}_real_s{seed}",
                            "setting": s["number"], "g_hidden": s["g_hidden"],
                            "f_channels": s["f_channels"], "n_extra_pairs": n})
    assert [t["index"] for t in out] == list(range(len(out)))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("tasks", "train-pairs", "extra-pairs", "heldout-pairs"):
        p = sub.add_parser(name)
        p.add_argument("manifest")
        if name != "tasks":
            p.add_argument("g")
        if name in ("train-pairs", "extra-pairs"):
            p.add_argument("arm", choices=ARMS)
    args = ap.parse_args(argv)
    rows = _lp.read_manifest(args.manifest)
    if args.cmd == "tasks":
        sys.stdout.write("\t".join(TASK_KEYS) + "\n" + _lp._tsv(tasks(rows), TASK_KEYS))
        return 0
    if args.cmd == "heldout-pairs":
        table = heldout_score_pairs(rows, args.g)
    else:
        fn = train_pairs if args.cmd == "train-pairs" else extra_pairs
        table = fn(rows, args.g, args.arm)
    sys.stdout.write("\t".join(_lp.PAIR_KEYS) + "\n" + _lp._tsv(table, _lp.PAIR_KEYS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
