"""t142 (iii): do REAL depth-arm halves sit on the corrected (whole-read) thinned floor? (PI 2026-10-30)

The depth arms (15M / 7.5M / 3.75M products) were split into pseudoreplicate halves by t117, so
their ratio-1 rows of floor_pairs.tsv are two real, independent halves at 7.5M / 3.75M / 1.875M
reads each. The track's base product (30M; two 15M halves), thinned to ratio 0.5 / 0.25 / 0.125,
gives synthetic halves at the same reads per half. If thinning mimics real lower depth, the two
agree. Per-bin thinning (`unit`, the t137 floor) makes multi-bin readings too smooth (shape_1k,
abs_log2fc_top1 too small); whole-read thinning (`read`, the t142 floor) should not.

Per depth-arm product and metric:
  real            the arm's ratio-1 rows, mean over the two directions (no draw: seed 0)
  read / unit     the base product's rows at the same reads_per_half in the read / unit floor,
                  mean over directions, then over seeds 0, 1 (`source` = ladder); when the base's
                  ladder has no such depth (the 50M DNase base: 25M per half), the base's depth
                  curve read at that depth (`value_<label>` of floor_curve.tsv, `source` = curve)
  *_seed_spread   |seed 0 - seed 1| of that floor (ladder rows only): its own noise floor
  real_over_*     real / read, real / unit (1 = the floor matches real halves)
Per mark class and metric: the median of each ratio over the class's depth arms.

stdlib only. Inputs: the read floor dir (t142, thinning "read") and the unit floor dir (t137), each
with floor_pairs.tsv, floor_curve.tsv (and config.json when present: its `thinning` is checked).

    python compare_real.py <read_floor_dir> <unit_floor_dir> <out_dir>
      -> compare_real.tsv, compare_real_class.tsv
"""
from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path

METRICS = ("shape_1k", "abs_log2fc_top1", "divergence_bin", "spearman_fine", "crps_all",
           "crps_top1")
READ_AT = {7_500_000: "7.5M", 3_750_000: "3.75M", 1_875_000: "1.875M", 15_000_000: "15M"}
COLS = ("track mark_class arm_pid level base_pid reads_per_half metric real read unit source "
        "read_seed_spread unit_seed_spread real_over_read real_over_unit").split()
CLASS_COLS = ("mark_class metric n median_real_over_read median_real_over_unit "
              "median_read_seed_spread_over_read").split()


def _tsv(path) -> list:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _mean(v) -> float:
    v = [x for x in v if not math.isnan(x)]
    return sum(v) / len(v) if v else float("nan")


def _div(a, b) -> float:
    return a / b if b and not (math.isnan(a) or math.isnan(b)) else float("nan")


def _median(v) -> float:
    v = [x for x in v if not math.isnan(x)]
    return statistics.median(v) if v else float("nan")


def check_mode(floor_dir: Path, want: str) -> None:
    cfg = floor_dir / "config.json"
    if cfg.is_file():
        got = json.loads(cfg.read_text()).get("thinning", "unit")
        if got != want:
            raise SystemExit(f"{floor_dir}: thinning {got!r}, expected {want!r}")


def floor_value(pairs: list, curve: list, base_pid: str, rph: int, metric: str):
    """(value, seed_spread, source) of the base product's floor at `rph` reads per half."""
    rows = [r for r in pairs if r["pid"] == base_pid and int(r["reads_per_half"]) == rph
            and float(r["thin_ratio"]) < 1.0]
    if rows:
        by_seed = {}
        for r in rows:
            by_seed.setdefault(int(r["seed"]), []).append(_f(r[metric]))
        dm = {s: _mean(v) for s, v in by_seed.items()}
        spread = abs(dm[0] - dm[1]) if {0, 1} <= set(dm) else float("nan")
        return _mean(list(dm.values())), spread, "ladder"
    label = READ_AT.get(rph)
    c = next((r for r in curve if r["pid"] == base_pid and r["metric"] == metric), None)
    if label and c is not None:
        return _f(c[f"value_{label}"]), float("nan"), "curve"
    return float("nan"), float("nan"), "none"


def compare(read_dir, unit_dir) -> tuple[list, list]:
    read_dir, unit_dir = Path(read_dir), Path(unit_dir)
    check_mode(read_dir, "read")
    check_mode(unit_dir, "unit")
    rp, up = _tsv(read_dir / "floor_pairs.tsv"), _tsv(unit_dir / "floor_pairs.tsv")
    rc, uc = _tsv(read_dir / "floor_curve.tsv"), _tsv(unit_dir / "floor_curve.tsv")
    base_of = {r["track"]: r["pid"] for r in rp if r["arm"] == "base"}
    arms = sorted({r["pid"] for r in rp if r["arm"] == "depth"})
    out = []
    for pid in arms:
        real_rows = [r for r in rp if r["pid"] == pid and float(r["thin_ratio"]) == 1.0]
        r0 = real_rows[0]
        base = base_of.get(r0["track"])
        rph = int(r0["reads_per_half"])
        for m in METRICS:
            real = _mean([_f(r[m]) for r in real_rows])
            if base is None:
                vr = vu = sr = su = float("nan")
                src = "no base"
            else:
                vr, sr, src = floor_value(rp, rc, base, rph, m)
                vu, su, _ = floor_value(up, uc, base, rph, m)
            out.append({"track": r0["track"], "mark_class": r0["mark_class"], "arm_pid": pid,
                        "level": r0["level"], "base_pid": base or "", "reads_per_half": rph,
                        "metric": m, "real": real, "read": vr, "unit": vu, "source": src,
                        "read_seed_spread": sr, "unit_seed_spread": su,
                        "real_over_read": _div(real, vr), "real_over_unit": _div(real, vu)})
    cls = []
    for mc in sorted({r["mark_class"] for r in out}) + ["all"]:
        for m in METRICS:
            sel = [r for r in out if r["metric"] == m and mc in ("all", r["mark_class"])]
            cls.append({"mark_class": mc, "metric": m, "n": len(sel),
                        "median_real_over_read": _median([r["real_over_read"] for r in sel]),
                        "median_real_over_unit": _median([r["real_over_unit"] for r in sel]),
                        "median_read_seed_spread_over_read": _median(
                            [_div(r["read_seed_spread"], r["read"]) for r in sel])})
    return out, cls


def _write(path: Path, rows: list, cols) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cols), delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v) for k, v in r.items()})
    tmp.replace(path)


def main(argv=None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    if len(a) != 3:
        print("usage: compare_real.py <read_floor_dir> <unit_floor_dir> <out_dir>", file=sys.stderr)
        return 2
    rows, cls = compare(a[0], a[1])
    out = Path(a[2])
    out.mkdir(parents=True, exist_ok=True)
    _write(out / "compare_real.tsv", rows, COLS)
    _write(out / "compare_real_class.tsv", cls, CLASS_COLS)
    for r in cls:
        if r["metric"] in ("shape_1k", "abs_log2fc_top1", "divergence_bin"):
            print(f"{r['mark_class']:7s} {r['metric']:16s} n={r['n']:2d} real/read="
                  f"{r['median_real_over_read']:.3f} real/unit={r['median_real_over_unit']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
