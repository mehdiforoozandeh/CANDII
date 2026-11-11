"""The miniCANDI ladder's per-rung arms as data, the json files generated from them, and the
resource plan that `slurm/minicandi/submit_arms.sh` reads (chunk LK; plan table "Per-rung arms").

    python3 tools/minicandi/ladder_arms.py write            # regenerate configs/arms/rung<r>/*.json
    python3 tools/minicandi/ladder_arms.py check            # exit 1 if a file on disk differs
    python3 tools/minicandi/ladder_arms.py arms --rung R     # the rung's arm names, base first
    python3 tools/minicandi/ladder_arms.py plan --rung R --config C [--device cpu|cuda]
        [--cluster fir|nibi] [--score-time T] [--extra "<train.py args>"]
        -> one line: mode gres cpus mem_mb workers train_time score_gres score_cpus score_mem_mb
           score_time

`configs/arms/rung<r>/base.json` is `configs/rung<r>/base.json` (left untouched) with
`windows_seen = budgets.json["windows_seen_matched_params"]`. Every other arm is that base plus the
arm's one change in `ARMS[r][arm]`, plus `simpler_than_base` from `SIMPLER` when it is not null.
`matched_compute` is the base with `windows_seen = budgets.json["windows_seen_at_matched_compute"]`.

Stdlib only: the submit script runs this on a login node with the system python3.
"""
from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
from pathlib import Path

CONFIGS = Path(__file__).resolve().parent / "configs"
ARMS_DIR = CONFIGS / "arms"
BUDGETS = CONFIGS / "budgets.json"
RUNGS = (1, 2, 3, 4, 5, 6)

_COMMON = {
    "dna_off": {"dna": "off"},
    "scrambled_covariates": {"scramble_covariates": True},
    "inject_concat": {"covariate_injection": "concat"},
}
#: ARMS[rung] = {arm: {key: value}}: the one change of each arm from that rung's arms base.json.
#: `matched_compute` is filled from budgets.json by `arm_configs`.
ARMS = {
    1: {"base": {}, **_COMMON, "scramble_all": {"scramble_all": True},
        "inject_tokens": {"covariate_injection": "tokens"}, "mask_token": {"mask_mode": "token"}},
    2: {"base": {}, **_COMMON, "mask_drop": {"mask_mode": "drop"}},
    3: {"base": {}, **_COMMON, "inject_tokens": {"covariate_injection": "tokens"},
        "mask_drop": {"mask_mode": "drop"}},
    4: {"base": {}, **_COMMON, "inject_tokens": {"covariate_injection": "tokens"},
        "mask_drop": {"mask_mode": "drop"}, "skip_dna": {"skip_source": "dna"},
        "skip_counts": {"skip_source": "counts"}},
    5: {"base": {}, **_COMMON, "inject_film": {"covariate_injection": "film"},
        "mask_token": {"mask_mode": "token"}, "axis_position_only": {"axis": "position_only"},
        "axis_track_only": {"axis": "track_only"}, "axis_tied": {"axis": "tied_position"}},
    6: {"base": {}, **_COMMON, "inject_film": {"covariate_injection": "film"},
        "mask_token": {"mask_mode": "token"}, "latents_small": {"n_latents": 32},
        "latents_large": {"n_latents": 512}},
}
#: arms of the table that do not apply by construction (written as no file; the test proves it)
NOT_APPLICABLE = {2: {"inject_tokens": {"covariate_injection": "tokens"}}}
#: agent extras (PI ruling 14): d14_condition_1 = false
EXTRAS = {"scramble_all", "axis_position_only", "axis_track_only", "axis_tied", "latents_small",
          "latents_large"}

_SIMPLER_TRUE = {"dna_off", "inject_concat", "skip_dna", "skip_counts", "axis_position_only",
                 "axis_track_only", "latents_small"}
_SIMPLER_FALSE = {"latents_large", "scrambled_covariates", "scramble_all"}


def simpler(rung: int, arm: str):
    """`simpler_than_base` of an arm: True, False (the controls, so that a control within the bars
    settles as INCUMBENT; `inject_tokens` on rungs 1, 3, 4; `latents_large`) or None."""
    if arm in _SIMPLER_TRUE:
        return True
    if arm in _SIMPLER_FALSE or (arm == "inject_tokens" and rung in (1, 3, 4)):
        return False
    return None


#: measured training seconds per window at batch 8, workers 0 (Fir pilot, P2b;
#: cruxvault/results/t182/budgets.md section 2). A (rung, device) pair not measured uses the
#: rung's CPU value. Nibi has not been timed (chunk L1a's probe).
SECONDS_PER_WINDOW = {("cuda", 1): 0.0466, ("cpu", 1): 0.320, ("cpu", 2): 0.0244,
                      ("cpu", 3): 0.0244, ("cpu", 4): 0.0250, ("cpu", 5): 0.119,
                      ("cpu", 6): 0.0244}
TRAIN_TIME_FACTOR = 3
TRAIN_TIME_MIN_S = 3600
#: scoring job resources (slurm/onion_eval/score.sh header: the 2-CPU form and the GPU form)
SCORE_SLURM = {"cpu": {"gres": "none", "cpus": 2, "mem_mb": 8000},
               "cuda": {"gres": "gpu:nvidia_h100_80gb_hbm3_1g.10gb:1", "cpus": None,
                        "mem_mb": 24000}}


def budgets() -> dict:
    return json.loads(BUDGETS.read_text(encoding="utf-8"))


def rung_base(rung: int) -> dict:
    return json.loads((CONFIGS / f"rung{rung}" / "base.json").read_text(encoding="utf-8"))


def arm_names(rung: int) -> list[str]:
    return [*ARMS[rung], "matched_compute"]


def arm_change(rung: int, arm: str, b: dict | None = None) -> dict:
    """The declared change of `arm` from the rung's arms base (simpler_than_base included)."""
    b = budgets() if b is None else b
    if arm == "matched_compute":
        change = {"windows_seen": int(b["windows_seen_at_matched_compute"][str(rung)])}
    else:
        change = dict(ARMS[rung][arm])
    s = simpler(rung, arm)
    if s is not None:
        change["simpler_than_base"] = s
    return change


def arm_configs(rung: int) -> dict[str, dict]:
    """{arm: config} for one rung; the arms base first."""
    b = budgets()
    base = {**rung_base(rung), "windows_seen": int(b["windows_seen_matched_params"])}
    return {arm: {**base, **arm_change(rung, arm, b)} for arm in arm_names(rung)}


def render(cfg: dict) -> str:
    return json.dumps(cfg, indent=1) + "\n"


def expected_files() -> dict[Path, str]:
    return {ARMS_DIR / f"rung{r}" / f"{arm}.json": render(cfg)
            for r in RUNGS for arm, cfg in arm_configs(r).items()}


def write() -> list[Path]:
    out = []
    for path, text in expected_files().items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        out.append(path)
    return out


def stale() -> list[str]:
    """Paths that are missing, differ from the generated text, or are not generated at all."""
    want = expected_files()
    bad = [str(p) for p, t in want.items()
           if not p.is_file() or p.read_text(encoding="utf-8") != t]
    bad += [str(p) for p in sorted(ARMS_DIR.glob("rung*/*.json")) if p not in want]
    return bad


def _hms(seconds: float) -> str:
    m = int(math.ceil(seconds / 60.0))
    d, m = divmod(m, 24 * 60)
    h, m = divmod(m, 60)
    return f"{d}-{h:02d}:{m:02d}:00" if d else f"{h:02d}:{m:02d}:00"


def plan(rung: int, config: dict, device: str | None = None, cluster: str | None = None,
         score_time: str | None = None, extra: str = "") -> dict:
    """Slurm resources of one (arm, seed): training from budgets.json and the measured speed,
    scoring on the same device type as the rung (one device per comparison)."""
    b = budgets()
    dev = device or b["device"][str(rung)]
    if dev not in ("cpu", "cuda"):
        raise ValueError(f"device {dev!r} is not cpu or cuda")
    slurm = dict(b["slurm"][dev])
    if dev == "cuda":
        if cluster not in b["cuda_cpus"]:
            raise ValueError(f"cluster {cluster!r} unknown; a GPU job needs one of "
                             f"{sorted(b['cuda_cpus'])} (set CLUSTER)")
        slurm["cpus"] = int(b["cuda_cpus"][cluster])
    windows = int(config["windows_seen"])
    toks = shlex.split(extra or "")
    for i, t in enumerate(toks):
        if t == "--max-windows" and i + 1 < len(toks):
            windows = min(windows, int(toks[i + 1]))
        elif t.startswith("--max-windows="):
            windows = min(windows, int(t.split("=", 1)[1]))
    spw = SECONDS_PER_WINDOW.get((dev, rung), SECONDS_PER_WINDOW[("cpu", rung)])
    train_s = max(TRAIN_TIME_MIN_S, TRAIN_TIME_FACTOR * windows * spw)
    sc = dict(SCORE_SLURM[dev])
    if sc["cpus"] is None:
        sc["cpus"] = slurm["cpus"]
    return {"mode": "gpu" if dev == "cuda" else "cpu", "gres": slurm["gres"],
            "cpus": int(slurm["cpus"]), "mem_mb": int(slurm["mem_mb"]),
            "workers": int(slurm["workers"]), "train_time": _hms(train_s),
            "score_gres": sc["gres"], "score_cpus": int(sc["cpus"]),
            "score_mem_mb": int(sc["mem_mb"]),
            "score_time": score_time or b["score_time"][str(rung)]}


PLAN_FIELDS = ("mode", "gres", "cpus", "mem_mb", "workers", "train_time", "score_gres",
               "score_cpus", "score_mem_mb", "score_time")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("write")
    sub.add_parser("check")
    a = sub.add_parser("arms")
    a.add_argument("--rung", type=int, required=True, choices=RUNGS)
    p = sub.add_parser("plan")
    p.add_argument("--rung", type=int, required=True, choices=RUNGS)
    p.add_argument("--config", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--cluster", default=None)
    p.add_argument("--score-time", default=None)
    p.add_argument("--extra", default="")
    args = ap.parse_args(argv)
    if args.cmd == "write":
        print(f"wrote {len(write())} files under {ARMS_DIR}")
    elif args.cmd == "check":
        bad = stale()
        for p_ in bad:
            print(f"stale: {p_}")
        return 1 if bad else 0
    elif args.cmd == "arms":
        print(" ".join(arm_names(args.rung)))
    else:
        cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
        try:
            r = plan(args.rung, cfg, args.device or None, args.cluster or None,
                     args.score_time or None, args.extra)
        except ValueError as e:
            print(f"plan: {e}", file=sys.stderr)
            return 2
        print(" ".join(str(r[k]) for k in PLAN_FIELDS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
