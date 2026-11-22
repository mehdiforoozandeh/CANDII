"""Task t370 (miniCANDI recipe run), chunk C4: every screen arm from one table.

    python tools/minicandi/recipe_arms.py write
        # writes configs/recipe/<family>/rung<r>/<arm>.json and configs/recipe/arms.tsv
        # (overwrites only those files)
    python tools/minicandi/recipe_arms.py list [--stage probe|s1a|s1b|s1c|all] [--arms "<arm> ..."]
                                               [--models "2 3 4"]
        # one tab-separated line per arm-model, no header:
        # <rung> <arm> <run name> <config path relative to the kit> <training time>

An arm config is the base recipe configs/rung<r>/base_heldout.json plus the arm's keys (an arm key
that the base holds replaces its value in place; a new key is appended). The run name is
rcp_<arm>_<model>, with <model> in MODELS (the names of mix_arms.ARCH).

Keys introduced by other chunks of t370, pinned in the plan (.orchestrate/plan.md, chunks C1, C2):
`skip_film`, `head_skip`, `film_placement` (C1, trunk FiLM placement and U-Net skips) and
`cac_pre`, `cac_post` (C2, covariate-correction layers before and after the trunk). The
configs use them as pinned; this module does not import the code that reads them.

Family control (C15, decision D9): `ctl_holdout`, `ctl_holdout_w4`, `ctl_holdout_separate` on all
three models; keys `control_holdout_cells`, `control_loss_weight`, `control_branch` (read by
dataset.py, losses.py and the rungs). The family is in no stage: it runs with STAGE=arms (C18).

Stages (configs/recipe/submit_recipe.sh STAGE=...): probe = family probe; s1a = families skip,
placement, nocov, halves (C29: read-disjoint training pairs, `train_halves = pairs`, decision D29); s1b = the curve arms of family cac and family mix; s1c = the CNN arms of family
cac. Training time: 08:00:00 for an arm with the average input or a CNN correction layer, else
03:00:00.

Stdlib only: submit_recipe.sh runs `list` with the login node's python3, which has no torch.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))
from minicandi import mix_arms  # noqa: E402  (stdlib only)

CONFIGS = _ROOT / "tools/minicandi/configs"
OUT = CONFIGS / "recipe"
MODELS = {2: "convae", 3: "convattn", 4: "unet"}
#: contract.META_COLUMNS (pinned interface B; the test holds the two equal). Copied, because
#: contract imports torch and `list` runs without it.
META_COLUMNS = ("log2_depth", "run_type_pe", "read_length", "dedup_on", "mapq_threshold",
                "pipeline_id", "pipeline_version_id", "lab_id", "platform_id", "antibody_id",
                "donor_id")
ALL = (2, 3, 4)
NOCOV = {"covariate_drop_fields": list(META_COLUMNS)}
TRAIN_TIME, TRAIN_TIME_LONG = "03:00:00", "08:00:00"
#: C2 keys, read here through names only: a string literal in a subscript or .get would make
#: train.check_config_keys accept the key even where no model code reads it
CAC_KEYS = ("cac_pre", "cac_post")
#: Decision D22: the cells whose ChIP control the control arms hold out of training. Computed once
#: on 2026-11-21 from Fir /project/def-maxwl/mforooz/ONION/tables/eic_v1_2/measurements.tsv and
#: eval/eic_v1_2/evalset.json: the cells of the 42 validation leave-one-assay-out ChIP-seq targets
#: (`imputation_targets`, regime leave_one_assay_out, assay_class narrow_histone or broad_histone)
#: that hold a training `chipseq-control` row; 24 cells, sorted, every third ([::3]).
#: (C02 is in the 24 through its training control; its target's own control is the one
#: validation control row.)
CONTROL_HOLDOUT = ["C02", "C10", "C16", "C20", "C25", "C31", "C36", "C46"]


def _arms() -> dict[str, dict]:
    a: dict[str, dict] = {}

    def add(name, family, models, keys):
        assert name not in a, name
        a[name] = {"family": family, "models": tuple(models), "keys": keys}

    add("skipfilm_ccp", "skip", (4,), {"skip_film": "c_cprime"})
    add("headskip_off", "skip", (4,), {"head_skip": "off"})
    add("headskip_film", "skip", (4,), {"head_skip": "film"})
    add("skipfilm_ccp_headskip_film", "skip", (4,), {"skip_film": "c_cprime", "head_skip": "film"})
    for p in ("encoder", "decoder", "bottleneck", "all"):
        add(f"film_{p}", "placement", ALL, {"film_placement": p})
    add("nocov", "nocov", ALL, dict(NOCOV))
    add("halves", "halves", ALL, {"train_halves": "pairs"})
    ctl = {"control_holdout_cells": list(CONTROL_HOLDOUT)}
    add("ctl_holdout", "control", ALL, dict(ctl))
    add("ctl_holdout_w4", "control", ALL, {**ctl, "control_loss_weight": 4.0})
    add("ctl_holdout_separate", "control", ALL, {**ctl, "control_branch": "separate"})
    for form in ("curve", "cnn"):
        add(f"cac_{form}", "cac", ALL, {"cac_pre": form, "cac_post": form})
        add(f"cac_{form}_corefilm_off", "cac", ALL, {"cac_pre": form, "cac_post": form, **NOCOV})
    for sw, (short, _src, keys) in mix_arms.SWITCHES.items():
        models = tuple(r for r in ALL if sw in mix_arms.POOL[r])
        if models:
            add(f"mix_{short}", "mix", models, dict(keys))
    add("probe_base", "probe", (4,), {})
    # Family combo (decision D35): the screen winner halves (U-Net) plus each U-Net switch that
    # raised imputation or calibration in the screen, plus headskip_off (D41, the copy-path test).
    # In no stage: runs with STAGE=arms.
    hv = dict(a["halves"]["keys"])
    for other in ("mix_avg", "mix_pearson", "mix_fixdepth", "headskip_film", "headskip_off"):
        add(f"halves_{other}", "combo", (4,), {**hv, **a[other]["keys"]})
    return a


#: arm -> {"family", "models", "keys"}
ARMS: dict[str, dict] = _arms()
#: stage -> arms, in ARMS order
STAGES = {
    "probe": tuple(n for n, x in ARMS.items() if x["family"] == "probe"),
    "s1a": tuple(n for n, x in ARMS.items() if x["family"] in ("skip", "placement", "nocov",
                                                                 "halves")),
    "s1b": tuple(n for n, x in ARMS.items()
                 if (x["family"] == "cac" and x["keys"][CAC_KEYS[0]] == "curve")
                 or x["family"] == "mix"),
    "s1c": tuple(n for n, x in ARMS.items()
                 if x["family"] == "cac" and x["keys"][CAC_KEYS[0]] == "cnn"),
}


def config(rung: int, arm: str) -> dict:
    """configs/rung<rung>/base_heldout.json plus the arm's keys."""
    if rung not in ARMS[arm]["models"]:
        raise ValueError(f"arm {arm} does not run on rung {rung} ({MODELS.get(rung)})")
    cfg = json.loads((CONFIGS / f"rung{rung}/base_heldout.json").read_text(encoding="utf-8"))
    for k, v in ARMS[arm]["keys"].items():
        cfg[k] = json.loads(json.dumps(v))
    return cfg


def run_name(rung: int, arm: str) -> str:
    return f"rcp_{arm}_{MODELS[rung]}"


def config_path(rung: int, arm: str) -> Path:
    return OUT / ARMS[arm]["family"] / f"rung{rung}" / f"{arm}.json"


def train_time(arm: str) -> str:
    k = ARMS[arm]["keys"]
    long = bool(k.get("average_input")) or "cnn" in [k.get(c) for c in CAC_KEYS]
    return TRAIN_TIME_LONG if long else TRAIN_TIME


def stages_of(arm: str) -> list[str]:
    return [s for s, names in STAGES.items() if arm in names]


def rows() -> list[dict]:
    out = []
    for arm, x in ARMS.items():
        for r in x["models"]:
            out.append({"arm": arm, "family": x["family"], "rung": str(r), "model": MODELS[r],
                        "run_name": run_name(r, arm),
                        "config": str(config_path(r, arm).relative_to(_ROOT)),
                        "keys": json.dumps(x["keys"], sort_keys=True, separators=(",", ":")),
                        "train_time": train_time(arm), "stages": ",".join(stages_of(arm))})
    return out


def files() -> dict[Path, str]:
    """{path: text} of every file `write` writes."""
    f = {config_path(r, arm): json.dumps(config(r, arm), indent=1) + "\n"
         for arm, x in ARMS.items() for r in x["models"]}
    rs = rows()
    head = list(rs[0])
    f[OUT / "arms.tsv"] = "\n".join(["\t".join(head)] + ["\t".join(x[h] for h in head)
                                                       for x in rs]) + "\n"
    return f


def select(stage: str | None = None, arms: list[str] | None = None,
           models: list[int] | None = None) -> list[tuple[int, str]]:
    """(rung, arm) pairs of a stage or an arm list, restricted to `models`."""
    if arms:
        bad = [a for a in arms if a not in ARMS]
        if bad:
            raise ValueError(f"unknown arms {bad}")
        names = list(arms)
    elif stage in (None, "all"):
        names = list(ARMS)
    else:
        names = list(STAGES[stage])
    keep = set(models) if models else set(MODELS)
    return [(r, a) for a in names for r in ARMS[a]["models"] if r in keep]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("write", "list"))
    ap.add_argument("--stage", choices=(*STAGES, "all"))
    ap.add_argument("--arms", default="", help='list: arm names, space separated')
    ap.add_argument("--models", default="", help='list: rungs, e.g. "2 3 4"')
    a = ap.parse_args(argv)
    if a.cmd == "write":
        fs = files()
        for p, body in fs.items():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(body, encoding="utf-8")
        print(f"{len(fs) - 1} arm configs + arms.tsv -> {OUT}")
        return 0
    try:
        sel = select(a.stage, a.arms.split(), [int(m) for m in a.models.split()])
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    for r, arm in sel:
        print("\t".join((str(r), arm, run_name(r, arm), str(config_path(r, arm).relative_to(_ROOT)),
                         train_time(arm))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
