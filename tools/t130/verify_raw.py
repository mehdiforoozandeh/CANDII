"""t130 blind readings, second method (stdlib only; no pandas, no code shared with analyze.py).

Recomputes from the extract and the floor files, for the cells
  space counts x knob depth x mark_class all x rung (D, A) x g_version per_track x direction
  both x subset all:
    d_model    real model, kind trained, eval score: per run the mean crps_all over its depth
               pairs -> per seed the mean over runs -> the mean over seeds;
    d_onewarp  the pair's QuantileMatching crps_all, carried on the same records, same pooling;
    d_floor    the pinned rule (plan_gaps.md A3), by its own route: X = the target's total depth
               (a depth target's level; 50M for C12M02__base__base; else 30M); the same track's
               base product `<track>__base__base`; its floor_curve crps_all row read ALWAYS by
               the line exp(alpha + beta * log2 X) (never the value_ columns), same pooling.
               floor_pairs is read for nothing (kept in the pinned argument list). d_floor is
               context (what a fresh replicate would differ by), not the gap top;
    d_oracle   the nested-read oracle (PI ruling 2026-10-30), read from nested_oracle.tsv by its
               own route: the row with r_source `manifest`, eval `score`, space `counts` and the
               predictive `binomial` for base_to_arm / `eb_binomial` for arm_to_base, keyed by
               (arm pid, direction); its crps_all, same pooling;
    gapclosed_onewarp  (d_onewarp - d_model) / (d_onewarp - d_oracle) from the three above;
    seed_abs_delta     max - min over the three per-seed pooled d_model values;
  and for rung D the swap median: the median swap_ratio_median over the real model's counts
  swap records of depth products, g_version per_track.
Writes verify_raw.json beside records.tsv.gz; verify_compare.py compares it with blind_table.tsv.

    python3 verify_raw.py <records.tsv.gz> <baselines.tsv.gz> <floor_pairs.tsv> <floor_curve.tsv> <nested_oracle.tsv>
"""
import csv
import gzip
import json
import math
import os
import statistics as st
import sys
from collections import defaultdict

DNASE_BASE = "C12M02__base__base"
NESTED_PRED = {"base_to_arm": "binomial", "arm_to_base": "eb_binomial"}


def rows(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def depth_of(pid, level, b2a):
    if b2a and pid.split("__")[1] == "depth":
        return float(level[:-1]) * 1e6
    return 50e6 if pid == DNASE_BASE else 30e6


def floor_of(track, x, fc):
    base = track + "__base__base"
    for r in fc:
        if r["pid"] == base and r["metric"] == "crps_all" and r["alpha"] not in ("", "nan"):
            return math.exp(float(r["alpha"]) + float(r["beta"]) * math.log2(x))
    return None


def per_seed(rec, value):
    """{seed: mean over runs of (mean over pairs)}; None if any value is missing."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rec:
        v = value(r)
        if v is None:
            return None
        by[r["seed"]][r["run"]].append(v)
    return {s: sum(sum(p) / len(p) for p in runs.values()) / len(runs) for s, runs in by.items()}


def pooled(rec, value):
    s = per_seed(rec, value)
    return sum(s.values()) / len(s) if s else None


def main(argv):
    if len(argv) != 5:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    rp, bp, fpp, fcp, nop = argv
    rec, base, fp, fc = rows(rp), rows(bp), rows(fpp), rows(fcp)
    nested = {(r["pid"], r["direction"]): float(r["crps_all"]) for r in rows(nop)
              if r["space"] == "counts" and r["eval"] == "score" and r["r_source"] == "manifest"
              and NESTED_PRED.get(r["direction"]) == r["predictive"] and r["crps_all"] != "NA"}
    ow = {(b["space"], b["pid"], b["direction"]): float(b["crps_all"]) for b in base
          if b["rung"] in ("QuantileMatching", "QM") and b["crps_all"] != ""}
    out = {"d_model": {}, "d_onewarp": {}, "d_floor": {}, "d_oracle": {}, "gapclosed_onewarp": {},
           "seed_abs_delta": {}, "swap_ratio_median": {}}
    for rung in ("D", "A"):
        key = f"counts|depth|all|{rung}|per_track|both|all"
        sel = [r for r in rec if r["rung"] == rung and r["g_version"] == "per_track"
               and r["model"] == "real" and r["kind"] == "trained" and r["eval"] == "score"
               and r["space"] == "counts" and r["arm"] == "depth"]
        out["d_model"][key] = pooled(sel, lambda r: float(r["crps_all"]))
        out["d_onewarp"][key] = pooled(sel, lambda r: ow.get(("counts", r["pid"], r["direction"])))

        def fl(r):
            x = depth_of(r["target_pid"], r["level"], r["direction"] == "base_to_arm")
            return floor_of(r["track"], x, fc)
        out["d_floor"][key] = pooled(sel, fl)
        out["d_oracle"][key] = pooled(sel, lambda r: nested.get((r["pid"], r["direction"])))
        m, o, f = out["d_model"][key], out["d_onewarp"][key], out["d_oracle"][key]
        out["gapclosed_onewarp"][key] = (None if None in (m, o, f) or o == f
                                         else (o - m) / (o - f))
        ps = per_seed(sel, lambda r: float(r["crps_all"]))
        out["seed_abs_delta"][key] = (max(ps.values()) - min(ps.values())
                                      if ps and len(ps) > 1 else None)
    sw = [float(r["swap_ratio_median"]) for r in rec
          if r["rung"] == "D" and r["g_version"] == "per_track" and r["model"] == "real"
          and r["kind"] == "swap" and r["space"] == "counts" and r["arm"] == "depth"
          and r["swap_ratio_median"] != ""]
    out["swap_ratio_median"]["counts|depth|all|D|per_track|both|all"] = (
        st.median(sw) if sw else None)
    dst = os.path.join(os.path.dirname(os.path.abspath(rp)), "verify_raw.json")
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
