"""t128 — symmetric encoder-decoder rung D v2 (plan/T126_ENCDEC_D2.md): settings, hold-out, task table.

**Settings.** `SETTINGS` holds the 8 phase-1 settings of the spec, in its order. Keys:
`number` (1-8), `arm` (the run-name prefix), `rung` ("DS", or "DSBOX" for the boxed C_clean),
`cclean` ("A" free, "D" boxed), `loss` ("L12" = L1 + L2, "L2" = L2 alone, "L123" = L1 + L2 +
lam * L3), `lam`, `vae`, `twin` (a no-covariates twin is trained; settings 1 and 3 only), and
`val_nll_rule` (the chr22 NLL terms that select the best step; PI ruling 2026-10-21: each setting
selects on its own loss's NLL terms, deterministic forward, no KL and no lam * mse term).

**Hold-out.** `HOLDOUT` = the depth 7.5M arm of every track, as t126.

**Tasks.** 240 rows: setting slowest, then g in `G_IDS`, then model in ("real",) + ("nocov",) if
the setting has a twin, then seed; `index` = position; `rung` is the ARM name (as t126's table);
`setting` is the number. Anchors: 18 = DS_C19M16_counts_real_s0, 57 = DSbox_..., 90 = DSrt_...,
129 = DSrtvae_..., 153 = DSl0.1_..., 177 = DSl1_..., 201 = DSl10_..., 225 = DSvae_...

    python tools/t128/pairs.py tasks <MANIFEST.tsv>
    python tools/t128/pairs.py train-pairs <MANIFEST.tsv> <g>
    python tools/t128/pairs.py heldout-pairs <MANIFEST.tsv> <g>
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
#: (arm, level) of the products held out of training, on every track (as t126)
HOLDOUT = (("depth", "7.5M"),)
TASK_KEYS = ("index", "rung", "g", "space", "model", "seed", "run_name", "setting")
#: the chr22 NLL terms that pick the best step, per loss (PI ruling (iii), 2026-10-21)
VAL_NLL_RULE = {"L12": "L1+L2 NLL", "L2": "L2 NLL", "L123": "L1+L2 NLL"}


def _s(number, arm, rung, cclean, loss, lam, vae, twin):
    return {"number": number, "arm": arm, "rung": rung, "cclean": cclean, "loss": loss,
            "lam": lam, "vae": vae, "twin": twin, "val_nll_rule": VAL_NLL_RULE[loss]}


SETTINGS = (
    _s(1, "DS", "DS", "A", "L12", 0.0, False, True),
    _s(2, "DSbox", "DSBOX", "D", "L12", 0.0, False, False),
    _s(3, "DSrt", "DS", "A", "L2", 0.0, False, True),
    _s(4, "DSrtvae", "DS", "A", "L2", 0.0, True, False),
    _s(5, "DSl0.1", "DS", "A", "L123", 0.1, False, False),
    _s(6, "DSl1", "DS", "A", "L123", 1.0, False, False),
    _s(7, "DSl10", "DS", "A", "L123", 10.0, False, False),
    _s(8, "DSvae", "DS", "A", "L12", 0.0, True, False),
)
SETTING_BY_ARM = {s["arm"]: s for s in SETTINGS}
ARMS = tuple(s["arm"] for s in SETTINGS)
#: one real seed-0 C19M16 run per setting
SMOKE = (18, 57, 90, 129, 153, 177, 201, 225)
#: every index not in SMOKE
FULL_ARRAY = "0-17,19-56,58-89,91-128,130-152,154-176,178-200,202-224,226-239"


def tasks(rows: list[dict] | None = None) -> list[dict]:
    """The 240-row task table; a pure function of the constants (`rows` is unused)."""
    out = []
    for s in SETTINGS:
        for g in G_IDS:
            for model in ("real",) + (("nocov",) if s["twin"] else ()):
                for seed in SEEDS:
                    out.append({"index": len(out), "rung": s["arm"], "g": g, "space": SPACE,
                                "model": model, "seed": seed,
                                "run_name": f"{s['arm']}_{g}_{SPACE}_{model}_s{seed}",
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
