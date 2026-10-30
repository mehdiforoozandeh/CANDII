"""t139: put the t137 resampling floor at the arm's depth beside every knob reading.

    python tools/t139/join_floor.py <knob_out_dir> <floor_dir> <out.tsv>

Reads `<knob_out_dir>/levels.tsv` (knob_effects.py: per space, mark class, arm, level, the median
over tracks) and, from `<floor_dir>`, `floor_at_30M.tsv` (class median of the curve read at 30M)
and `floor_curve.tsv` (per product: the ladder values and the thinning-seed spread). For each knob
row and metric (abs_log2fc_top1, shape_1k) the floor is the class median at the arm's depth:

  depth 15M / 7.5M / 3.75M  -> class median of value_15M / value_7.5M / value_3.75M
  every other arm (30M)     -> value_30M_extrap from floor_at_30M.tsv

floor_seed_spread is the class median |seed 0 - seed 1| at that depth; 15M and 30M have no thinning
draw, so they carry the 7.5M spread. above_floor = value > floor + 2 x floor_seed_spread.
The floor exists in counts only: the p-space ladder was not run, so p rows carry floor NA
(floor_p15M.tsv holds CRPS only, and no CRPS metric is in levels.tsv).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

METRICS = {"abs_log2fc_top1": "median_abs_log2fc_top1", "shape_1k": "median_shape_1k"}
LADDER = ("15M", "7.5M", "3.75M")
COLS = ("space mark_class arm level metric value floor floor_depth floor_seed_spread "
        "ratio_to_floor above_floor").split()


def _read(path: Path) -> list:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _num(v):
    return None if v in (None, "", "NA", "None", "nan") else float(v)


def _pick(row: dict, *names):
    for n in names:
        if n in row:
            return row[n]
    raise KeyError(f"none of {names} in {sorted(row)}")


def floors(floor_dir: Path) -> dict:
    """{(mark_class, metric, depth): (floor, seed_spread)} for depth in LADDER + ('30M',)."""
    curve: dict = {}
    for r in _read(floor_dir / "floor_curve.tsv"):
        curve.setdefault((r["mark_class"], r["metric"]), []).append(r)
    out = {}
    for (mc, m), rows in curve.items():
        s75 = float(np.median([_num(_pick(r, "seed_spread_7.5M", "seed_spread_15M_7.5M"))
                               for r in rows]))
        s375 = float(np.median([_num(r["seed_spread_3.75M"]) for r in rows]))
        spread = {"15M": s75, "7.5M": s75, "3.75M": s375, "30M": s75}
        for d in LADDER:
            out[(mc, m, d)] = (float(np.median([_num(r[f"value_{d}"]) for r in rows])), spread[d])
        out[(mc, m, "30M")] = (None, spread["30M"])
    for r in _read(floor_dir / "floor_at_30M.tsv"):
        key = (r["mark_class"], r["metric"], "30M")
        if key in out:
            out[key] = (_num(_pick(r, "value_30M_extrap", "median_value_30M_extrap")), out[key][1])
    return out


def join(levels: list, fl: dict) -> list:
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
                depth = lv["level"] if lv["arm"] == "depth" and lv["level"] in LADDER else "30M"
                floor, spread = fl.get((lv["mark_class"], metric, depth), (None, None))
                row.update(floor=floor, floor_depth=depth, floor_seed_spread=spread)
                if value is not None and floor is not None:
                    row["ratio_to_floor"] = value / floor if floor else None
                    row["above_floor"] = value > floor + 2 * spread
            rows.append(row)
    return rows


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 3:
        sys.exit("usage: join_floor.py <knob_out_dir> <floor_dir> <out.tsv>")
    knob, floor_dir, out = Path(argv[0]), Path(argv[1]), Path(argv[2])
    rows = join(_read(knob / "levels.tsv"), floors(floor_dir))
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(COLS)
        for r in rows:
            w.writerow(["NA" if r[c] is None else r[c] for c in COLS])
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
