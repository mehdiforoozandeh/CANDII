"""Close-out item A13, mix-and-match: the combination arms of the single ablation choices that
qualified per architecture (cruxvault/INSTRUCTIONS/peeling_the_onion/closeout/
"closeout A13 mix-and-match combinations.md", section "Specification as worked").

    python tools/minicandi/mix_arms.py write   # writes configs/ablations/mix/rung{2,3,4}/*.json
                                               # and arms.tsv (overwrites only those files)
    python tools/minicandi/mix_arms.py check   # exit 1 when the files on disk differ

A switch is the exact key set of its single-ablation config (configs/ablations/<ablation>/
rung<r>/<arm>.json minus the base recipe configs/final/rung<r>/reference_equal.json); the test
`tests/test_minicandi_mix_arms.py` holds the two equal. An arm is the base recipe plus the union of
its switches' keys; switches never share a key, so any subset is well defined.

Design per architecture (the reference arm, every switch off, is NOT written here: it is the base
recipe configs/rung<r>/base_heldout.json under the run name eic_v1_2_rung<r>_base_heldout):
  conv autoencoder (rung 2): 1 switch, full factorial (1 arm besides the reference).
  conv + attention (rung 3): 5 switches; the half fraction 2^(5-1) with the defining relation
      I = +ABCDE (product of the +-1 codes = +1: every arm with 1, 3 or 5 switches on), resolution
      V (main effects and two-switch interactions alias only with three- and four-switch
      interactions). It holds each single switch and the all-on arm; the reference (0 switches)
      lies outside it and is run beside it. 16 arms besides the reference.
  U-Net (rung 4): 3 switches, full factorial (7 arms besides the reference).
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = _ROOT / "tools/minicandi/configs"
OUT = CONFIGS / "ablations/mix"
ARCH = {2: "convae", 3: "convattn", 4: "unet"}
#: the average input of the single ablation (built for eic_v1; eic_v1_2 changes covariates only)
AVERAGE_H5 = "/scratch/mforooz/onion/aux/eic_v1_average.h5"

#: switch -> (short name in arm names, source ablation/arm, keys set)
SWITCHES = {
    "average_input": ("avg", "average_input/average_input",
                      {"average_input": True, "average_h5": AVERAGE_H5}),
    "pearson": ("pearson", "count_transforms/tf_pearson", {"count_transform": "pearson"}),
    "fixed_depth": ("fixdepth", "depth_input/input_depth_fixed", {"input_depth": "fixed"}),
    "concat": ("concat", "cov_injection_form/inject_concat", {"covariate_injection": "concat"}),
    "impute_x4": ("imp4", "losses/loss_impute_x4",
                  {"loss_weights": {"base": 1, "correct": 1, "impute": 4, "same": 1}}),
    "nb_crps": ("crps", "losses/loss_nb_crps", {"loss": "nb_crps"}),
    "same_view_only": ("svonly", "views/all_same_view_only", {"same_view_prob": 1.0}),
}
#: per rung, the qualifying switches in a fixed order (the factor order of the design)
POOL = {2: ("same_view_only",),
        3: ("concat", "fixed_depth", "impute_x4", "nb_crps", "same_view_only"),
        4: ("average_input", "pearson", "fixed_depth")}
#: rungs that run the half fraction (+ all-on) instead of the full factorial
HALF = {3}


def design(r: int) -> list[tuple[str, ...]]:
    """The arms of rung r besides the reference: tuples of switch names in POOL order."""
    pool = POOL[r]
    k = len(pool)
    arms = []
    for bits in itertools.product((0, 1), repeat=k):
        if not any(bits):
            continue                                   # the reference arm
        if r in HALF:
            prod = 1
            for b in bits:
                prod *= 1 if b else -1
            if prod != 1:                              # I = +ABCDE
                continue
        arms.append(tuple(s for s, b in zip(pool, bits) if b))
    return sorted(arms, key=lambda a: (len(a), [pool.index(s) for s in a]))


def arm_name(switches) -> str:
    return "-".join(SWITCHES[s][0] for s in switches)


def config(r: int, switches) -> dict:
    cfg = json.loads((CONFIGS / f"final/rung{r}/reference_equal.json").read_text())
    seen = set()
    for s in switches:
        for k, v in SWITCHES[s][2].items():
            if k in seen:
                raise ValueError(f"switch {s} sets {k}, already set by another switch")
            seen.add(k)
            cfg[k] = v
    cfg["simpler_than_base"] = False
    return cfg


def rows(r: int) -> list[dict]:
    out = []
    for sw in design(r):
        fraction = "full" if r not in HALF else "half_I=+ABCDE"
        out.append({"arm": arm_name(sw), "run_name": f"mix_{ARCH[r]}_{arm_name(sw)}",
                    "switches": ",".join(sw), "n_switches": str(len(sw)),
                    "keys_changed": ",".join(sorted({k for s in sw for k in SWITCHES[s][2]})),
                    "native_depth_scoring": "yes" if "fixed_depth" in sw else "no",
                    "design": fraction})
    return out


def _files(r: int) -> dict:
    d = OUT / f"rung{r}"
    files = {d / f"{arm_name(sw)}.json": json.dumps(config(r, sw), indent=1) + "\n"
             for sw in design(r)}
    rs = rows(r)
    head = list(rs[0])
    files[d / "arms.tsv"] = "\n".join(["\t".join(head)] + ["\t".join(x[h] for h in head)
                                                          for x in rs]) + "\n"
    return files


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("write", "check"))
    a = ap.parse_args(argv)
    bad = 0
    for r in POOL:
        want = _files(r)
        if a.cmd == "write":
            (OUT / f"rung{r}").mkdir(parents=True, exist_ok=True)
            for p, body in want.items():
                p.write_text(body, encoding="utf-8")
            print(f"rung{r}: {len(want) - 1} arm configs + arms.tsv -> {OUT / f'rung{r}'}")
            continue
        have = set((OUT / f"rung{r}").glob("*"))
        for p in sorted(have - set(want)):
            print(f"extra file {p}"); bad += 1
        for p, body in want.items():
            if not p.is_file() or p.read_text(encoding="utf-8") != body:
                print(f"differs or missing: {p}"); bad += 1
    if a.cmd == "check":
        print("mix configs: " + ("in sync" if not bad else f"{bad} problems"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
