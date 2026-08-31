#!/usr/bin/env python3
"""The Average gate: does our re-run of the scoring path land on 001's recorded numbers?

The chain this closes, end to end:

    published `Average` rows (MOESM2, team_id 100)
      <- 001's `our_average_tv.csv`   [001 matched published: median rel. diff 2.7e-8, max 2.6e-7]
      <- our re-run on Fir            [what this script measures]

001 rebuilt the Average from the challenge's own training+validation tracks and scored it against
the challenge's blind truth. We rebuild it from the same tracks -- now staged on Fir at
DATA_EIC_SYNAPSE and md5-verified against Nibi -- and score it with the byte-identical vendored
scorer. If the two agree, every link between the staged bigwigs and a published leaderboard row is
verified, and an entrant score produced by the same path inherits that. If they disagree, the fault
is in the staging or in how we are driving the scorer, and no entrant should be scored until it is
found.

`msevar` is not compared: 001 could not reconstruct the published variance vector and excludes it
everywhere, so the gate covers eight of nine measures.

    python gate_average.py --ours <dir-of-csvs> --reference <001 our_average_tv.csv>
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

MEASURES = ["mse", "gwcorr", "gwspear", "mseprom", "msegene", "mseenh", "mse1obs", "mse1imp"]

# 001's own worst case reproducing published rows was 2.4e-5, dominated by gwspear's rank
# tie-breaking. We are re-running the same code on the same data, so we should do far better than
# that -- anything above 1e-9 means something differs and is worth finding, not waving through.
TOL_INVESTIGATE = 1e-9
# The gate fails outright above this. It is 001's published-comparison worst case: landing worse
# than 001 did against the *published* table, while merely re-running 001's own code, is a defect.
TOL_FAIL = 2.4e-5


def read_rows(path: str) -> Dict[Tuple[str, int], dict]:
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            out[(r["experiment"], int(r["bootstrap_id"]))] = r
    return out


def read_dir(path: str) -> Dict[Tuple[str, int], dict]:
    if os.path.isfile(path):
        return read_rows(path)
    out: Dict[Tuple[str, int], dict] = {}
    files = sorted(glob.glob(os.path.join(path, "*.csv")))
    if not files:
        raise SystemExit(f"no csvs under {path}")
    for f in files:
        out.update(read_rows(f))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True, help="our re-run: a directory of per-experiment CSVs")
    ap.add_argument("--reference", required=True, help="001's output/scores/our_average_tv.csv")
    ap.add_argument("--out-json")
    args = ap.parse_args()

    ours = read_dir(args.ours)
    ref = read_dir(args.reference)

    shared = sorted(set(ours) & set(ref))
    missing = sorted(set(ref) - set(ours))
    extra = sorted(set(ours) - set(ref))
    if not shared:
        raise SystemExit("no (experiment, bootstrap) key is present in both -- nothing to compare")

    per_measure: Dict[str, List[float]] = {m: [] for m in MEASURES}
    worst: Dict[str, Tuple[float, str, int]] = {}
    for key in shared:
        for m in MEASURES:
            a, b = float(ref[key][m]), float(ours[key][m])
            rel = 0.0 if a == b else abs(a - b) / max(abs(a), 1e-30)
            per_measure[m].append(rel)
            if m not in worst or rel > worst[m][0]:
                worst[m] = (rel, key[0], key[1])

    all_rel = np.array([v for m in MEASURES for v in per_measure[m]])
    overall_worst = float(all_rel.max())

    print(f"compared {len(shared)} (experiment, bootstrap) rows x {len(MEASURES)} measures")
    if missing:
        print(f"MISSING from our re-run: {len(missing)} rows, e.g. {missing[:5]}")
    if extra:
        print(f"EXTRA in our re-run: {len(extra)} rows, e.g. {extra[:5]}")
    print(f"{'measure':10s} {'median rel':>12s} {'max rel':>12s}   worst at")
    for m in MEASURES:
        v = np.array(per_measure[m])
        w, exp, b = worst[m]
        print(f"{m:10s} {np.median(v):12.3e} {v.max():12.3e}   {exp} bootstrap {b}")
    print(f"\nworst relative difference over everything: {overall_worst:.3e}")

    if missing:
        verdict = "INCOMPLETE"
        note = (f"{len(missing)} reference rows have no counterpart in our re-run; the gate cannot "
                f"pass on a partial panel (the D2 lesson).")
    elif overall_worst <= TOL_INVESTIGATE:
        verdict = "PASS"
        note = ("bit-level agreement with 001. The staged tracks, the vendored scorer and the "
                "rebuilt Average all reproduce, so the path to the published Average rows is "
                "verified end to end.")
    elif overall_worst <= TOL_FAIL:
        verdict = "PASS_WITH_DRIFT"
        note = (f"agrees with 001 to {overall_worst:.3e}, inside 001's own worst case against the "
                f"published table ({TOL_FAIL:.1e}), but not bit-level. Find the source before "
                f"scoring entrants -- re-running the same code on the same data should be exact.")
    else:
        verdict = "FAIL"
        note = (f"differs from 001 by {overall_worst:.3e}, worse than 001's own worst case against "
                f"the published table ({TOL_FAIL:.1e}). Do not score entrants.")

    print(f"\nVERDICT: {verdict}\n{note}")

    if args.out_json:
        payload = {
            "verdict": verdict, "note": note,
            "n_rows_compared": len(shared), "n_missing": len(missing), "n_extra": len(extra),
            "missing": missing[:50], "extra": extra[:50],
            "worst_relative_difference": overall_worst,
            "per_measure": {m: {"median": float(np.median(per_measure[m])),
                                "max": float(np.max(per_measure[m])),
                                "worst_experiment": worst[m][1],
                                "worst_bootstrap": worst[m][2]} for m in MEASURES},
            "msevar_excluded": ("001 could not reconstruct the published variance vector; "
                                "eight of nine measures are compared"),
            "reference": os.path.abspath(args.reference),
            "ours": os.path.abspath(args.ours),
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w") as fh:
            json.dump(payload, fh, indent=1)

    return 0 if verdict.startswith("PASS") else 1


if __name__ == "__main__":
    sys.exit(main())
