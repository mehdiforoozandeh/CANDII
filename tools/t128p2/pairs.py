"""t128 phase 2 — four arms at 12,000 steps (plan/T126_ENCDEC_D2.md "Phase 2: PI decisions,
2026-10-25"): settings, hold-out, task table.

**Settings.** `SETTINGS` holds the 4 phase-2 arms, in this order. Keys: `number` (1-4), `arm` (the
run-name prefix), `rung` (the form rung: "DS", "DSANC" or "D"), `cclean` ("A" free, zero init;
"F" frozen at the base's encoded knobs; None for rung D), `loss` ("L12" = L1 + L2, "L12A" = L1 +
L2 + L_anchor at weight 1 each, "nb" = t118 rung D's loss), `lam`, `vae`, `twin` (no twins in
phase 2), `max_steps` (12,000 everywhere) and `val_nll_rule` (the chr22 NLL terms that select the
best step; each arm its own loss's NLL terms, PI ruling (iii) of phase 1).

**Hold-out.** `HOLDOUT` = the depth 7.5M arm of every track, as t126 and t128.

**Tasks.** 96 rows: arm slowest, then g in `G_IDS`, then seed; `index = arm_i*24 + g_i*3 + seed`.
`rung` is the ARM name (as the t126/t128 tables); `form_rung` is the setting's form rung (the law
script reads it); `setting` is the number. Anchors: 9 = DSfree_C19M16_counts_real_s0,
33 = DSfixed_..., 57 = DSanchor_..., 81 = D_C19M16_counts_real_s0 (arm D's run names equal
`ladder.train.run_name("D", ...)` and `t128/viabase.VIABASE_RUNS`).

    python tools/t128p2/pairs.py tasks <MANIFEST.tsv>
    python tools/t128p2/pairs.py train-pairs <MANIFEST.tsv> <g>
    python tools/t128p2/pairs.py heldout-pairs <MANIFEST.tsv> <g>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_T118 = Path(__file__).resolve().parents[1] / "t118"
if str(_T118) not in sys.path:          # import convention: tools/t118 holds `ladder`
    sys.path.insert(0, str(_T118))

from ladder import pairs as _lp  # noqa: E402

G_IDS = _lp.G_IDS
SEEDS = (0, 1, 2)
SPACE = "counts"
MAX_STEPS = 12000
#: (arm, level) of the products held out of training, on every track (as t126, t128)
HOLDOUT = (("depth", "7.5M"),)
TASK_KEYS = ("index", "rung", "form_rung", "g", "space", "model", "seed", "run_name", "setting")


def _s(number, arm, rung, cclean, loss, val_nll_rule):
    return {"number": number, "arm": arm, "rung": rung, "cclean": cclean, "loss": loss,
            "lam": 0.0, "vae": False, "twin": False, "max_steps": MAX_STEPS,
            "val_nll_rule": val_nll_rule}


SETTINGS = (
    _s(1, "DSfree", "DS", "A", "L12", "L1+L2 NLL"),
    _s(2, "DSfixed", "DS", "F", "L12", "L1+L2 NLL"),
    _s(3, "DSanchor", "DSANC", "F", "L12A", "L1+L2+anchor NLL"),
    _s(4, "D", "D", None, "nb", "t118 cross NLL"),
)
SETTING_BY_ARM = {s["arm"]: s for s in SETTINGS}
ARMS = tuple(s["arm"] for s in SETTINGS)
#: one real seed-0 C19M16 run per arm
SMOKE = (9, 33, 57, 81)
#: the full submission runs in its own out dir, so it includes the smoke indices
FULL_ARRAY = "0-95"


def tasks(rows: list[dict] | None = None) -> list[dict]:
    """The 96-row task table; a pure function of the constants (`rows` is unused)."""
    out = []
    for s in SETTINGS:
        for g in G_IDS:
            for seed in SEEDS:
                out.append({"index": len(out), "rung": s["arm"], "form_rung": s["rung"], "g": g,
                            "space": SPACE, "model": "real", "seed": seed,
                            "run_name": f"{s['arm']}_{g}_{SPACE}_real_s{seed}",
                            "setting": s["number"]})
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
