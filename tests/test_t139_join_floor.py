"""t139: join_floor.py puts the t137 floor at the arm's depth beside every knob reading."""
from __future__ import annotations

import csv
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
CURVE_HEAD = ("pid mark_class metric alpha beta r2 value_15M value_7.5M value_3.75M value_1.875M "
              "value_30M_extrap seed_spread_7.5M seed_spread_3.75M seed_spread_1.875M").split()
CURVE = [  # two narrow products; class median = mean of the two
    ["p1", "narrow", "shape_1k", 0, 0, 1, 0.10, 0.20, 0.40, 0.8, 0.050, 0.002, 0.004, 0.01],
    ["p2", "narrow", "shape_1k", 0, 0, 1, 0.12, 0.22, 0.42, 0.8, 0.060, 0.004, 0.006, 0.01],
    ["p1", "narrow", "abs_log2fc_top1", 0, 0, 1, 0.02, 0.03, 0.05, 0.1, 0.010, 0.001, 0.002, 0.01],
    ["p2", "narrow", "abs_log2fc_top1", 0, 0, 1, 0.04, 0.05, 0.07, 0.1, 0.020, 0.003, 0.004, 0.01],
]
AT30 = """mark_class	metric	value_30M_extrap	min	max	value_15M	seed_spread_7.5M	n_products
narrow	shape_1k	0.009	0.005	0.06	0.11	0.003	2
narrow	abs_log2fc_top1	0.015	0.01	0.02	0.03	0.002	2
"""


@pytest.fixture
def table(tmp_path):
    knob, floor = tmp_path / "knob", tmp_path / "floor"
    knob.mkdir(), floor.mkdir()
    (knob / "levels.tsv").write_text(LEVELS)
    (floor / "floor_at_30M.tsv").write_text(AT30)
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


def test_depth_rows_pick_their_ladder_value(table):
    rows, _ = table
    r = rows[("counts", "depth", "15M", "shape_1k")]
    assert (r["floor_depth"], float(r["floor"])) == ("15M", pytest.approx(0.11))
    assert float(r["floor_seed_spread"]) == pytest.approx(0.003)  # 15M has no draw: the 7.5M spread
    r = rows[("counts", "depth", "3.75M", "shape_1k")]
    assert (r["floor_depth"], float(r["floor"])) == ("3.75M", pytest.approx(0.41))
    assert float(r["floor_seed_spread"]) == pytest.approx(0.005)
    r = rows[("counts", "depth", "3.75M", "abs_log2fc_top1")]
    assert float(r["floor"]) == pytest.approx(0.06)


def test_other_arms_pick_30M_extrap(table):
    rows, _ = table
    r = rows[("counts", "dedup", "off", "shape_1k")]
    assert (r["floor_depth"], float(r["floor"])) == ("30M", pytest.approx(0.009))
    assert float(r["ratio_to_floor"]) == pytest.approx(0.011 / 0.009)


def test_above_floor_two_spread_rule(table):
    rows, _ = table
    # dedup shape 0.011 vs floor 0.009 + 2 x 0.003 = 0.015 -> not above
    assert rows[("counts", "dedup", "off", "shape_1k")]["above_floor"] == "False"
    # dedup |lfc| 0.1 vs 0.015 + 2 x 0.002 -> above
    assert rows[("counts", "dedup", "off", "abs_log2fc_top1")]["above_floor"] == "True"
    # depth 15M shape 0.05 vs 0.11 + 0.006 -> not above
    assert rows[("counts", "depth", "15M", "shape_1k")]["above_floor"] == "False"


def test_p_space_rows_carry_na(table):
    rows, _ = table
    for m in ("shape_1k", "abs_log2fc_top1"):
        r = rows[("pval", "depth", "15M", m)]
        assert r["floor"] == "NA" and r["floor_depth"] == "p-ladder-not-run"
        assert r["above_floor"] == "NA" and r["ratio_to_floor"] == "NA"
