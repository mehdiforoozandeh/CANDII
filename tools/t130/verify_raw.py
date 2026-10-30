"""t130 blind readings, second method (stdlib only; no pandas, no code shared with analyze.py).

Recomputes from the extract and the floor files, for the cells
  space counts x knob depth x mark_class all x rung (D, A) x g_version per_track x direction
  both x subset all:
    d_model    real model, kind trained, eval score: per run the mean crps_all over its depth
               pairs -> per seed the mean over runs -> the mean over seeds;
    d_onewarp  the pair's QuantileMatching crps_all, carried on the same records, same pooling;
    d_floor    the TARGET's floor at its depth (thin_ratio 1.0 / 0.5 / 0.25 for a 15M / 7.5M /
               3.75M depth target: the mean of floor_pairs crps_all over seeds and directions;
               a base target: floor_curve value_30M_extrap of crps_all), same pooling;
  and for rung D the swap median: the median swap_ratio_median over the real model's counts
  swap records of depth products, g_version per_track.
Writes verify_raw.json beside records.tsv.gz; verify_compare.py compares it with blind_table.tsv.

    python3 verify_raw.py <records.tsv.gz> <baselines.tsv.gz> <floor_pairs.tsv> <floor_curve.tsv>
"""
import csv
import gzip
import json
import os
import statistics as st
import sys
from collections import defaultdict

RATIO = {"15M": 1.0, "7.5M": 0.5, "3.75M": 0.25}


def rows(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def floor_of(target_pid, target_level, fp, fc):
    if target_level in RATIO:
        r = RATIO[target_level]
        v = [float(x["crps_all"]) for x in fp
             if x["pid"] == target_pid and abs(float(x["thin_ratio"]) - r) < 1e-9]
        return sum(v) / len(v) if v else None
    for x in fc:
        if x["pid"] == target_pid and x["metric"] == "crps_all":
            return float(x["value_30M_extrap"])
    return None


def pooled(rec, value):
    """{seed: {run: [values]}} -> mean over seeds of (mean over runs of (mean over pairs))."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rec:
        v = value(r)
        if v is None:
            return None
        by[r["seed"]][r["run"]].append(v)
    seeds = [sum(sum(p) / len(p) for p in runs.values()) / len(runs) for runs in by.values()]
    return sum(seeds) / len(seeds) if seeds else None


def main(argv):
    if len(argv) != 4:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    rp, bp, fpp, fcp = argv
    rec, base, fp, fc = rows(rp), rows(bp), rows(fpp), rows(fcp)
    ow = {(b["space"], b["pid"], b["direction"]): float(b["crps_all"]) for b in base
          if b["rung"] in ("QuantileMatching", "QM") and b["crps_all"] != ""}
    out = {"d_model": {}, "d_onewarp": {}, "d_floor": {}, "swap_ratio_median": {}}
    for rung in ("D", "A"):
        key = f"counts|depth|all|{rung}|per_track|both|all"
        sel = [r for r in rec if r["rung"] == rung and r["g_version"] == "per_track"
               and r["model"] == "real" and r["kind"] == "trained" and r["eval"] == "score"
               and r["space"] == "counts" and r["arm"] == "depth"]
        out["d_model"][key] = pooled(sel, lambda r: float(r["crps_all"]))
        out["d_onewarp"][key] = pooled(sel, lambda r: ow.get(("counts", r["pid"], r["direction"])))

        def fl(r):
            b2a = r["direction"] == "base_to_arm"
            return floor_of(r["target_pid"], r["level"] if b2a else "base", fp, fc)
        out["d_floor"][key] = pooled(sel, fl)
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
