"""t139: join_floor.py puts the t137 floor at the arm's depth beside every knob reading."""
from __future__ import annotations

import csv
import math
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "t139" / "join_floor.py"

LEVELS = """space	mark_class	arm	level	n_tracks	median_log2fc_top1	median_abs_log2fc_top1	median_shape_1k	median_dfrip
counts	narrow	depth	15M	3	-1.0	1.0	0.05	0.01
counts	narrow	depth	3.75M	3	-3.0	3.0	0.30	0.02
counts	narrow	dedup	off	3	0.1	0.1	0.011	0.0
pval	narrow	depth	15M	3	-0.2	0.2	0.3
"""
CURVE_HEAD = ("pid mark_class metric reads_per_half_full alpha beta beta_sign r2 value_15M "
              "value_7.5M value_3.75M value_1.875M value_30M_extrap obs_r1 obs_r0.5 obs_r0.25 "
              "obs_r0.125 seed_spread_r0.5 seed_spread_r0.25 seed_spread_r0.125").split()
_O = [0, 0, 0, 0]  # obs_r* (not read)
CURVE = [  # two narrow BASE products (15M and 25M per half) -> class median = mean of the two;
    # one depth-arm product whose 9s must never reach a floor (base products only)
    ["C19M16__base__base", "narrow", "shape_1k", 15000000, 0, -0.5, "neg", 1,
     0.10, 0.20, 0.40, 0.8, 0.050, *_O, 0.002, 0.004, 0.010],
    ["C07M20__base__base", "narrow", "shape_1k", 25000000, 0, -0.5, "neg", 1,
     0.12, 0.22, 0.42, 0.8, 0.060, *_O, 0.004, 0.006, 0.008],
    ["C19M16__depth__15M", "narrow", "shape_1k", 7500000, 0, -0.5, "neg", 1,
     9, 9, 9, 9, 9, *_O, 9, 9, 9],
    ["C19M16__base__base", "narrow", "abs_log2fc_top1", 15000000, 0, -0.5, "neg", 1,
     0.02, 0.03, 0.05, 0.1, 0.010, *_O, 0.001, 0.002, 0.010],
    ["C07M20__base__base", "narrow", "abs_log2fc_top1", 25000000, 0, -0.5, "neg", 1,
     0.04, 0.05, 0.07, 0.1, 0.020, *_O, 0.003, 0.004, 0.006],
    ["C19M16__depth__15M", "narrow", "abs_log2fc_top1", 7500000, 0, -0.5, "neg", 1,
     9, 9, 9, 9, 9, *_O, 9, 9, 9],
]


@pytest.fixture
def table(tmp_path):
    knob, floor = tmp_path / "knob", tmp_path / "floor"
    knob.mkdir(), floor.mkdir()
    (knob / "levels.tsv").write_text(LEVELS)
    (floor / "floor_curve.tsv").write_text(
        "\n".join("\t".join(map(str, r)) for r in [CURVE_HEAD, *CURVE]) + "\n")
    out = tmp_path / "knob_vs_floor.tsv"
    subprocess.run([sys.executable, str(SCRIPT), str(knob), str(floor), str(out)], check=True,
                   capture_output=True)
    with open(out, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    return {(r["space"], r["arm"], r["level"], r["metric"]): r for r in rows}, out


def test_columns_and_count(table):
    rows, out = table
    assert out.read_text().splitlines()[0].split("\t") == (
        "space mark_class arm level metric value floor floor_depth floor_seed_spread "
        "ratio_to_floor above_floor").split()
    assert len(rows) == 4 * 2


def test_depth_rows_read_the_base_curves_at_the_arm_depth(table):
    rows, _ = table
    r = rows[("counts", "depth", "15M", "shape_1k")]
    assert (r["floor_depth"], float(r["floor"])) == ("15M", pytest.approx(0.11))
    # X / n_lines: 15/15 = 1 and 15/25 = 0.6, both >= 0.5 -> r0.5: median(0.002, 0.004)
    assert float(r["floor_seed_spread"]) == pytest.approx(0.003)
    r = rows[("counts", "depth", "3.75M", "shape_1k")]
    assert (r["floor_depth"], float(r["floor"])) == ("3.75M", pytest.approx(0.41))
    # 3.75/15 = 0.25 -> r0.25 (0.004); 3.75/25 = 0.15 -> r0.125 (0.008)
    assert float(r["floor_seed_spread"]) == pytest.approx(0.006)
    r = rows[("counts", "depth", "3.75M", "abs_log2fc_top1")]
    assert float(r["floor"]) == pytest.approx(0.06)


def test_other_arms_pick_the_base_30M_extrap(table):
    rows, _ = table
    r = rows[("counts", "dedup", "off", "shape_1k")]
    assert (r["floor_depth"], float(r["floor"])) == ("30M", pytest.approx(0.055))
    assert float(r["floor_seed_spread"]) == pytest.approx(0.003)
    assert float(r["ratio_to_floor"]) == pytest.approx(0.011 / 0.055)


def test_above_floor_two_spread_rule(table):
    rows, _ = table
    # dedup shape 0.011 vs floor 0.055 + 2 x 0.003 -> not above
    assert rows[("counts", "dedup", "off", "shape_1k")]["above_floor"] == "False"
    # dedup |lfc| 0.1 vs 0.015 + 2 x 0.002 -> above
    assert rows[("counts", "dedup", "off", "abs_log2fc_top1")]["above_floor"] == "True"
    # depth 15M shape 0.05 vs 0.11 + 0.006 -> not above
    assert rows[("counts", "depth", "15M", "shape_1k")]["above_floor"] == "False"


def test_curve_off_the_fixed_depths_uses_the_line():
    import importlib.util
    spec = importlib.util.spec_from_file_location("join_floor", SCRIPT)
    jf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(jf)
    alpha, beta = 1.0, -0.2
    row = {"alpha": str(alpha), "beta": str(beta),
           "value_15M": str(math.exp(alpha + beta * math.log2(15e6)))}
    want = math.exp(alpha + beta * math.log2(50e6))
    assert jf.curve_at(row, "50M") == pytest.approx(want)
    assert jf.curve_at({**row, "alpha": ""}, "50M") == pytest.approx(want)   # alpha from value_15M
    assert [jf.spread_label(x, 25e6) for x in (50e6, 15e6, 7.5e6, 3.75e6, 1e6)] == [
        "r0.5", "r0.5", "r0.25", "r0.125", "r0.125"]


def test_p_space_rows_carry_na(table):
    rows, _ = table
    for m in ("shape_1k", "abs_log2fc_top1"):
        r = rows[("pval", "depth", "15M", m)]
        assert r["floor"] == "NA" and r["floor_depth"] == "p-ladder-not-run"
        assert r["above_floor"] == "NA" and r["ratio_to_floor"] == "NA"
