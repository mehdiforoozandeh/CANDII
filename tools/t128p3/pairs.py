"""t128 phase 3 — a bigger g and a bigger f for old D, read through the base (plan/T126_ENCDEC_D2.md
"Phase 3: PI decisions, 2026-10-27"): settings, hold-out, task table.

**Settings.** `SETTINGS` holds the 2 phase-3 arms, in this order. Keys: `number` (1-2), `arm` (the
run-name prefix), `rung` (the form rung: "D" for both — old rung D, `fforms/form_d.py`, as phase 2's
arm D), `g_hidden` (g's hidden width), `f_channels` (form D's channel count), `n_theta`
(2 * 4 layers * f_channels), `loss` ("nb", t118 rung D's loss), `max_steps` (15,000 everywhere)
and `val_nll_rule` (the chr22 NLL terms that select the best step).
  1. Dbigg: g hidden 64 -> 256, f as today (32 channels, n_theta 256).
  2. Dbigf: f at 64 channels (n_theta 512), g hidden 64.

**Hold-out.** `HOLDOUT` = the depth 7.5M arm of every track, as t126, t128 and phase 2.

**Tasks.** 48 rows: arm slowest, then g in `G_IDS`, then seed; `index = arm_i*24 + g_i*3 + seed`.
`rung` is the ARM name (the run-name prefix, as the t126/t128/t128p2 tables); `form_rung` is "D";
`setting` is the number; `g_hidden` / `f_channels` are the arm's sizes. Anchors:
9 = Dbigg_C19M16_counts_real_s0, 33 = Dbigf_C19M16_counts_real_s0.

**Reference.** `REFERENCE_RUNS` = phase 2's arm-D runs `D_<g>_counts_real_s<seed>` (24, the names
of `t128/viabase.VIABASE_RUNS`), NOT rerun here.

**Phase 3b** (plan/T126_ENCDEC_D2.md "Phase 3b"). `SETTINGS_3B`: Dbigf (phase 3's arm, 15,000
steps) and D (phase 2's arm D: g hidden 64, f 32 channels, n_theta 256, 12,000 steps; run name
`D_<g>_counts_real_s<seed>`, as phase 2). `tasks_3b()`: 42 rows, arm slowest (Dbigf, D), then g in
`G_IDS_3B` (the 7 per-track g; no across-track g), then seed in `SEEDS_3B` (3, 4, 5);
`index = arm_i*21 + g_i*3 + (seed - 3)`. Anchors: 0 = Dbigf_C07M20_counts_real_s3,
21 = D_C07M20_counts_real_s3. Same TASK_KEYS as `tasks()`.

    python tools/t128p3/pairs.py tasks <MANIFEST.tsv>
    python tools/t128p3/pairs.py tasks3b <MANIFEST.tsv>
    python tools/t128p3/pairs.py train-pairs <MANIFEST.tsv> <g>
    python tools/t128p3/pairs.py heldout-pairs <MANIFEST.tsv> <g>
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
from t128 import viabase as _vb  # noqa: E402

G_IDS = _lp.G_IDS
SEEDS = (0, 1, 2)
SPACE = "counts"
MAX_STEPS = 15000
#: (arm, level) of the products held out of training, on every track (as t126, t128, t128p2)
HOLDOUT = (("depth", "7.5M"),)
TASK_KEYS = ("index", "rung", "form_rung", "g", "space", "model", "seed", "run_name", "setting",
             "g_hidden", "f_channels")


def _s(number, arm, rung, g_hidden, f_channels, n_theta, loss, val_nll_rule):
    return {"number": number, "arm": arm, "rung": rung, "g_hidden": g_hidden,
            "f_channels": f_channels, "n_theta": n_theta, "loss": loss, "max_steps": MAX_STEPS,
            "val_nll_rule": val_nll_rule}


SETTINGS = (
    _s(1, "Dbigg", "D", 256, 32, 256, "nb", "t118 cross NLL"),
    _s(2, "Dbigf", "D", 64, 64, 512, "nb", "t118 cross NLL"),
)
SETTING_BY_ARM = {s["arm"]: s for s in SETTINGS}
ARMS = tuple(s["arm"] for s in SETTINGS)
#: phase 2's arm-D runs, the reference read through the base (not rerun)
REFERENCE_RUNS = tuple(r["run_name"] for r in _vb.VIABASE_RUNS)
#: one real seed-0 C19M16 run per arm
SMOKE = (9, 33)
#: the full submission runs in its own out dir, so it includes the smoke indices
FULL_ARRAY = "0-47"
ARM_ARRAYS = {"Dbigg": "0-23", "Dbigf": "24-47"}


#: phase 3b: three new seeds, the 7 per-track g (one g across tracks is dropped, PI ruling iv)
SEEDS_3B = (3, 4, 5)
G_IDS_3B = tuple(g for g in G_IDS if g != "all")


def _s3b(number, arm, g_hidden, f_channels, n_theta, max_steps):
    return {"number": number, "arm": arm, "rung": "D", "g_hidden": g_hidden,
            "f_channels": f_channels, "n_theta": n_theta, "loss": "nb", "max_steps": max_steps,
            "val_nll_rule": "t118 cross NLL"}


SETTINGS_3B = (
    _s3b(1, "Dbigf", 64, 64, 512, 15000),   # phase 3's arm b
    _s3b(2, "D", 64, 32, 256, 12000),       # phase 2's arm D (the run_training default sizes)
)
SETTING_BY_ARM_3B = {s["arm"]: s for s in SETTINGS_3B}


def tasks_3b(rows: list[dict] | None = None) -> list[dict]:
    """The 42-row phase-3b task table; a pure function of the constants (`rows` is unused)."""
    out = []
    for s in SETTINGS_3B:
        for g in G_IDS_3B:
            for seed in SEEDS_3B:
                out.append({"index": len(out), "rung": s["arm"], "form_rung": s["rung"], "g": g,
                            "space": SPACE, "model": "real", "seed": seed,
                            "run_name": f"{s['arm']}_{g}_{SPACE}_real_s{seed}",
                            "setting": s["number"], "g_hidden": s["g_hidden"],
                            "f_channels": s["f_channels"]})
    return out


def tasks(rows: list[dict] | None = None) -> list[dict]:
    """The 48-row task table; a pure function of the constants (`rows` is unused)."""
    out = []
    for s in SETTINGS:
        for g in G_IDS:
            for seed in SEEDS:
                out.append({"index": len(out), "rung": s["arm"], "form_rung": s["rung"], "g": g,
                            "space": SPACE, "model": "real", "seed": seed,
                            "run_name": f"{s['arm']}_{g}_{SPACE}_real_s{seed}",
                            "setting": s["number"], "g_hidden": s["g_hidden"],
                            "f_channels": s["f_channels"]})
    return out


def train_pairs(rows: list[dict], g: str) -> list[dict]:
    return _lp.train_pairs(rows, g, HOLDOUT)


def heldout_pairs(rows: list[dict], g: str) -> list[dict]:
    return _lp.heldout_pairs(rows, g, HOLDOUT)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("tasks", "tasks3b", "train-pairs", "heldout-pairs"):
        p = sub.add_parser(name)
        p.add_argument("manifest")
        if name not in ("tasks", "tasks3b"):
            p.add_argument("g")
    args = ap.parse_args(argv)
    rows = _lp.read_manifest(args.manifest)
    if args.cmd in ("tasks", "tasks3b"):
        table = tasks(rows) if args.cmd == "tasks" else tasks_3b(rows)
        sys.stdout.write("\t".join(TASK_KEYS) + "\n" + _lp._tsv(table, TASK_KEYS))
    else:
        fn = train_pairs if args.cmd == "train-pairs" else heldout_pairs
        sys.stdout.write("\t".join(_lp.PAIR_KEYS) + "\n" + _lp._tsv(fn(rows, args.g), _lp.PAIR_KEYS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
