"""t139: put the t137 resampling floor at the arm's depth beside every knob reading.

    python tools/t139/join_floor.py <knob_out_dir> <floor_dir> <out.tsv>

Reads `<knob_out_dir>/levels.tsv` (knob_effects.py: per space, mark class, arm, level, the median
over tracks) and `<floor_dir>/floor_curve.tsv` (t137 floor.py: one row per product x metric, the
product's own curve log(metric) = alpha + beta * log2(reads_per_half) read at fixed depths, and
its thinning-seed spread by ratio, seed_spread_r0.5 / r0.25 / r0.125).

The floor rule (plan_gaps.md A3) is "the same track's BASE product's curve read at the target's
depth X". The knob table is per mark class, not per track, so here the floor of a (mark_class,
metric, X) is the MEDIAN over the class's BASE products only (pid suffix `__base__base`) of each
base's curve read at reads_per_half = X:

  X = the arm's total depth: the level of a depth arm (15M / 7.5M / 3.75M), else 30M
  value_<X> for X in 15M / 7.5M / 3.75M / 1.875M; value_30M_extrap for X = 30M;
  any other X: exp(alpha + beta * log2 X) of the base's row.

floor_seed_spread = the median over the same base products of seed_spread_r<k>, k = X / that
base's reads_per_half_full (its n_lines) rounded to the nearest of 0.5 / 0.25 / 0.125 in log2;
k >= 0.5 -> r0.5 (ratio 1 and the 30M read have no draw). above_floor = value > floor + 2 x
floor_seed_spread. floor_at_30M.tsv (the median over ALL the class's products) is not read.
The floor exists in counts only: the p-space ladder was not run, so p rows carry floor NA.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

METRICS = {"abs_log2fc_top1": "median_abs_log2fc_top1", "shape_1k": "median_shape_1k"}
LADDER = ("15M", "7.5M", "3.75M")
READ_AT = {"15M": 15e6, "7.5M": 7.5e6, "3.75M": 3.75e6, "1.875M": 1.875e6}   # floor.READ_AT
SPREAD_RATIOS = (0.5, 0.25, 0.125)
BASE_SUFFIX = "__base__base"
COLS = ("space mark_class arm level metric value floor floor_depth floor_seed_spread "
        "ratio_to_floor above_floor").split()


def _read(path: Path) -> list:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _num(v):
    return None if v in (None, "", "NA", "None", "nan") else float(v)


def depth_of(arm: str, level: str) -> str:
    return level if arm == "depth" and level in LADDER else "30M"


def reads(depth: str) -> float:
    return float(depth.rstrip("M")) * 1e6


def curve_at(row: dict, depth: str):
    """The product's curve read at reads_per_half = depth."""
    if depth in READ_AT:
        return _num(row[f"value_{depth}"])
    if depth == "30M":
        return _num(row["value_30M_extrap"])
    beta, alpha = _num(row.get("beta")), _num(row.get("alpha"))
    if beta is None:
        return None
    if alpha is None:
        v15 = _num(row.get("value_15M"))
        if v15 is None:
            return None
        alpha = math.log(v15) - beta * math.log2(15e6)
    return math.exp(alpha + beta * math.log2(reads(depth)))


def spread_label(x: float, n_lines: float) -> str:
    k = x / n_lines
    if k >= 0.5:
        return "r0.5"
    best = min(SPREAD_RATIOS, key=lambda r: abs(math.log2(k) - math.log2(r)))
    return f"r{best:g}"


def _median(v):
    v = [x for x in v if x is not None]
    return float(np.median(v)) if v else None


def floor_at(base_rows: list, depth: str):
    """(floor, seed_spread): medians over the class's base products of the curve read at depth."""
    x = reads(depth)
    vals = [curve_at(r, depth) for r in base_rows]
    sp = [_num(r[f"seed_spread_{spread_label(x, float(r['reads_per_half_full']))}"])
          for r in base_rows]
    return _median(vals), _median(sp)


def bases(floor_dir: Path) -> dict:
    """{(mark_class, metric): [floor_curve rows of the class's base products]}."""
    out: dict = {}
    for r in _read(floor_dir / "floor_curve.tsv"):
        if r["pid"].endswith(BASE_SUFFIX):
            out.setdefault((r["mark_class"], r["metric"]), []).append(r)
    return out


def join(levels: list, base_rows: dict) -> list:
    rows = []
    for lv in levels:
        for metric, col in METRICS.items():
            value = _num(lv.get(col))
            row = {"space": lv["space"], "mark_class": lv["mark_class"], "arm": lv["arm"],
                   "level": lv["level"], "metric": metric, "value": value, "floor": None,
                   "floor_depth": None, "floor_seed_spread": None, "ratio_to_floor": None,
                   "above_floor": None}
            if lv["space"] != "counts":
                row["floor_depth"] = "p-ladder-not-run"
            else:
                depth = depth_of(lv["arm"], lv["level"])
                br = base_rows.get((lv["mark_class"], metric), [])
                floor, spread = floor_at(br, depth) if br else (None, None)
                row.update(floor=floor, floor_depth=depth, floor_seed_spread=spread)
                if value is not None and floor is not None:
                    row["ratio_to_floor"] = value / floor if floor else None
                    if spread is not None:
                        row["above_floor"] = value > floor + 2 * spread
            rows.append(row)
    return rows


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 3:
        sys.exit("usage: join_floor.py <knob_out_dir> <floor_dir> <out.tsv>")
    knob, floor_dir, out = Path(argv[0]), Path(argv[1]), Path(argv[2])
    rows = join(_read(knob / "levels.tsv"), bases(floor_dir))
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(COLS)
        for r in rows:
            w.writerow(["NA" if r[c] is None else r[c] for c in COLS])
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
