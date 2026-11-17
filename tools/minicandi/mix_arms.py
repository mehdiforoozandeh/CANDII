"""Close-out item A13, mix-and-match: the combination arms of the single ablation choices that
qualified per architecture (cruxvault/INSTRUCTIONS/peeling_the_onion/closeout/
"closeout A13 mix-and-match combinations.md", section "Specification as worked").

    python tools/minicandi/mix_arms.py write   # writes configs/ablations/mix/rung{2,3,4}/*.json
                                               # and arms.tsv (overwrites only those files)
    python tools/minicandi/mix_arms.py check   # exit 1 when the files on disk differ
    python tools/minicandi/mix_arms.py qualify --summary <arm-summary tsv of 256-region scores>
        # the qualification gate: re-applies the rule to every switch of POOL; exit 0 only when
        # every switch has a 3-seed 256-region score and still qualifies (or is in by PI ruling)

Qualification gate (foreman, 2026-11-17). The pool was derived from 16-region scores, where a
correction difference under about 0.05 is not reliable; three conv + attention switches
(concatenation, fixed input depth, NB CRPS) qualified on gap closed alone at +0.049 to +0.053.
Before any submission the pool is re-derived from 256-region scores (`select_rule.py arm-summary
--sub p256_sv`, the table format of results/ablations_closeout/arm_summary_p16_sv_2026-11-17.tsv).
Rule: better than the reference `rung<r>_base_heldout` beyond the seed range (difference of means
larger than the larger of the two ranges) on at least one of imputation (`imputation_raw`), gap
closed (`correction_v3_raw`), calibration (`uncertainty_raw`), and worse beyond the range on none.
U-Net fixed input depth is in by the PI ruling of 2026-11-17 whatever the rule says. The gate can
only remove switches; the configs stay the provisional design until it passes.

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
#: rungs that run the half fraction instead of the full factorial
HALF = {3}
#: qualification gate: the switches in the pool by PI ruling, whatever the rule says
PI_RULED = {(4, "fixed_depth")}
#: the three terms of the rule, as select_rule.py arm-summary names them
TERMS = (("imputation", "imputation_raw"), ("gap closed", "correction_v3_raw"),
         ("calibration", "uncertainty_raw"))


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


def source_run(r: int, switch: str) -> str:
    """The run name of the single ablation of `switch` on rung r (runs_fir/<name>)."""
    ab, arm = SWITCHES[switch][1].split("/")
    return f"abl_{ab}_{arm}_{ARCH[r]}"


def qualify(summary: dict, r: int) -> list[dict]:
    """Per switch of POOL[r]: {"switch", "run", "status", "detail"}; status is qualifies, fails,
    in_by_pi_ruling or no_score (the switch's or the reference's row is absent, has fewer than 3
    seeds, or lacks a term)."""
    ref = summary.get(f"rung{r}_base_heldout")
    out = []
    for sw in POOL[r]:
        run = source_run(r, sw)
        row = summary.get(run)
        rec = {"switch": sw, "run": run, "status": "no_score", "detail": ""}
        try:
            if row is None or ref is None or int(row["n_seeds"]) < 3 or int(ref["n_seeds"]) < 3:
                raise KeyError("row")
            calls = []
            for name, k in TERMS:
                d = float(row[f"{k}.mean"]) - float(ref[f"{k}.mean"])
                bar = max(float(row[f"{k}.range"]), float(ref[f"{k}.range"]))
                c = "up" if d > bar else "down" if d < -bar else "within"
                calls.append(c)
                rec["detail"] += f"{name} {d:+.3f} (bar {bar:.3f}, {c}); "
        except (KeyError, ValueError):
            rec["detail"] = ("no 3-seed score for " + (run if row is None or ref is not None
                                                        else f"rung{r}_base_heldout"))
            out.append(rec)
            continue
        ok = "up" in calls and "down" not in calls
        rec["status"] = ("in_by_pi_ruling" if (r, sw) in PI_RULED else
                         "qualifies" if ok else "fails")
        out.append(rec)
    return out


def read_summary(path) -> dict:
    """{arm: row} of an arm-summary TSV (select_rule.py arm-summary)."""
    import csv
    with open(path, newline="", encoding="utf-8") as fh:
        return {row["arm"]: row for row in csv.DictReader(fh, delimiter="\t")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("write", "check", "qualify"))
    ap.add_argument("--summary", help="qualify: arm-summary TSV of 256-region scores")
    a = ap.parse_args(argv)
    if a.cmd == "qualify":
        if not a.summary:
            ap.error("qualify needs --summary")
        summ = read_summary(a.summary)
        worst = 0
        for r in POOL:
            for rec in qualify(summ, r):
                print(f"rung{r}\t{rec['switch']}\t{rec['status']}\t{rec['run']}\t{rec['detail']}")
                worst = max(worst, {"qualifies": 0, "in_by_pi_ruling": 0, "fails": 1,
                                    "no_score": 3}[rec["status"]])
        print({0: "gate: PASS (the pool stands)", 1: "gate: FAIL (rewrite POOL, then write)",
               3: "gate: WAIT (a switch has no 256-region score)"}[worst])
        return worst
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
