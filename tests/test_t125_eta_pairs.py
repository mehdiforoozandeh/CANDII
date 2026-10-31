"""t125 part 1 — the depth offset (`eta.offset`) and the 432-row task table (`eta.pairs`).

Pinned by plan/ETA_OFFSET_AND_H_DESIGN.md (eta_in = log(1 + X) - log d, d in millions of reads,
eps = 1; log mu' = log d' + eta_out; 2 rows x 4 designs x 8 g x 2 models x 3 seeds = 384 runs plus
48 old-spread A runs = 432) and the t125 plan's chunk t125-P1a (index formula, pinned names, the
depth-only law pairs of PI decision (i)). The table formula is written out here independently of
`eta.pairs.tasks`.
"""
from __future__ import annotations

import hashlib
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
FIXTURE_MANIFEST = REPO / "tests" / "fixtures" / "t112_meta" / "MANIFEST.tsv"
PAIRS_PY = REPO / "tools" / "t125" / "eta" / "pairs.py"
for _p in (REPO / "tools" / "t118", REPO / "tools" / "t125"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from eta import offset, pairs  # noqa: E402
from ladder import pairs as lpairs  # noqa: E402

KEYS = ("index", "rung", "g", "model", "seed", "run_name")
TRACKS = ("C07M20", "C07M29", "C12M02", "C19M16", "C19M22", "C40M17", "C40M18")
G = TRACKS + ("all",)


def _rows():
    return lpairs.read_manifest(FIXTURE_MANIFEST)


# ---- offset ------------------------------------------------------------------------------------

def test_offset_constants():
    assert offset.D_UNIT == 1e6
    assert offset.EPS == 1.0


def test_log_depth():
    assert offset.log_depth(30_000_000) == pytest.approx(math.log(30.0), abs=1e-12)
    assert offset.log_depth(1e6) == 0.0
    assert offset.log_depth(3_750_000) == pytest.approx(math.log(3.75), abs=1e-12)


def test_transform_eta_hand_case_zero_count():
    X = np.zeros(5, dtype=np.int32)
    out = offset.transform_eta(X, offset.log_depth(30_000_000))
    assert out.dtype == np.float32
    assert out.shape == (5,)
    np.testing.assert_array_equal(out, np.full(5, -math.log(30.0), dtype=np.float32))


def test_transform_eta_general():
    X = np.array([0, 1, 4, 99, 12345], dtype=np.float32)
    ld = math.log(15.0)
    want = (np.log1p(X.astype(np.float64)) - ld).astype(np.float32)
    got = offset.transform_eta(X, ld)
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, want)
    # by hand: log(1 + 99) - log 15
    assert float(got[3]) == pytest.approx(math.log(100.0) - math.log(15.0), rel=1e-6)
    # 2-d input keeps its shape
    X2 = np.arange(12, dtype=np.int64).reshape(3, 4)
    assert offset.transform_eta(X2, 0.0).shape == (3, 4)


def test_offset_loc():
    loc = np.array([-1.0, 0.0, 2.5], dtype=np.float64)
    out = offset.offset_loc(loc, math.log(7.5))
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, loc + math.log(7.5), rtol=0, atol=0)


def test_identity_map_is_the_depth_scale():
    """eta_out = eta_in gives mu' = (1 + X) d'/d: the "do nothing" output scales by d'/d."""
    X = np.array([0, 3, 10, 250], dtype=np.int32)
    ld_src, ld_tgt = offset.log_depth(30e6), offset.log_depth(7.5e6)
    mu = np.exp(offset.offset_loc(offset.transform_eta(X, ld_src).astype(np.float64), ld_tgt))
    np.testing.assert_allclose(mu, (1.0 + X) * 7.5 / 30.0, rtol=1e-6)


def test_depth_table_from_fixture_manifest():
    rows = _rows()
    table = offset.depth_table(rows)
    assert len(rows) == 130
    assert set(table) == {r["pid"] for r in rows}            # every row, not only usable
    for pid in lpairs.EXCLUDED_PIDS:
        assert pid in table
    for r in rows:
        assert table[r["pid"]] == pytest.approx(math.log(float(r["depth"]) / 1e6), abs=1e-12)
    assert table["C07M20__abproxy__f0.5"] == pytest.approx(math.log(30.0), abs=1e-12)
    base_c12 = [r["pid"] for r in rows if r["track"] == "C12M02" and r["arm"] == "base"]
    assert table[base_c12[0]] == pytest.approx(math.log(50.0), abs=1e-12)


# ---- constants ---------------------------------------------------------------------------------

def test_pairs_constants():
    assert pairs.RUNGS_ROW1 == ("A", "B", "C", "D")
    assert pairs.RUNGS_ROW2 == ("A2", "B2", "C2", "D2")
    assert pairs.RUNG_OLDSPREAD == "Aold"
    assert pairs.RUNGS == ("A", "B", "C", "D", "A2", "B2", "C2", "D2", "Aold")
    assert pairs.MODELS == ("real", "nocov")
    assert pairs.SEEDS == (0, 1, 2)
    assert pairs.SPACE == "counts"
    assert pairs.G_IDS == lpairs.G_IDS == G
    assert pairs.TASK_KEYS == KEYS
    assert pairs.ROW1_OF == {"A2": "A", "B2": "B", "C2": "C", "D2": "D"}
    assert pairs.FORM_MODULE == {
        "A": "eta.fforms.form_a", "Aold": "ladder.fforms.form_a",
        "B": "ladder.fforms.form_b", "C": "ladder.fforms.form_c", "D": "ladder.fforms.form_d",
        "A2": "ladder.fforms.form_a2", "B2": "ladder.fforms.form_b2",
        "C2": "ladder.fforms.form_c2", "D2": "ladder.fforms.form_d2"}
    assert set(pairs.FORM_MODULE) == set(pairs.RUNGS)


def test_row_of():
    for r in ("A", "B", "C", "D", "Aold"):
        assert pairs.row_of(r) == 1
    for r in ("A2", "B2", "C2", "D2"):
        assert pairs.row_of(r) == 2
    with pytest.raises(ValueError):
        pairs.row_of("E")


# ---- the task table ----------------------------------------------------------------------------

def _formula_tsv() -> str:
    """The 432-row table written out from the pinned formula, independently of `pairs.tasks`."""
    lines = ["\t".join(KEYS)]
    for ri, rung in enumerate(("A", "B", "C", "D", "A2", "B2", "C2", "D2")):
        for gi, g in enumerate(G):
            for mi, model in enumerate(("real", "nocov")):
                for seed in (0, 1, 2):
                    index = ri * 48 + gi * 6 + mi * 3 + seed
                    lines.append(f"{index}\t{rung}\t{g}\t{model}\t{seed}\t"
                                 f"{rung}_{g}_counts_{model}_s{seed}")
    for gi, g in enumerate(G):
        for mi, model in enumerate(("real", "nocov")):
            for seed in (0, 1, 2):
                index = 384 + gi * 6 + mi * 3 + seed
                lines.append(f"{index}\tAold\t{g}\t{model}\t{seed}\tAold_{g}_counts_{model}_s{seed}")
    return "\n".join(lines) + "\n"


def _md5(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def test_tasks_match_the_formula():
    t = pairs.tasks()
    assert len(t) == 432
    assert [x["index"] for x in t] == list(range(432))
    assert all(tuple(x) == KEYS for x in t)
    got = "\t".join(KEYS) + "\n" + "".join("\t".join(str(x[k]) for k in KEYS) + "\n" for x in t)
    assert _md5(got) == _md5(_formula_tsv())
    assert len({x["run_name"] for x in t}) == 432
    assert sum(x["rung"] == "Aold" for x in t) == 48
    assert pairs.tasks(_rows()) == t                       # rows kept for the signature only


def test_pinned_indices():
    t = pairs.tasks()
    assert t[18]["run_name"] == "A_C19M16_counts_real_s0"
    assert t[42]["run_name"] == "A_all_counts_real_s0"
    assert t[402]["run_name"] == "Aold_C19M16_counts_real_s0"
    assert t[402]["rung"] == "Aold" and t[402]["g"] == "C19M16"
    assert t[383]["run_name"] == "D2_all_counts_nocov_s2"
    assert t[431]["run_name"] == "Aold_all_counts_nocov_s2"


def test_f_runs_seed0():
    f = pairs.f_runs_seed0()
    assert len(f) == 64
    assert f[0] == "A_C07M20_counts_real_s0"
    assert f[-1] == "D2_all_counts_real_s0"
    assert not any(n.startswith("Aold_") for n in f)
    want = [x["run_name"] for x in pairs.tasks()
            if x["model"] == "real" and x["seed"] == 0 and x["rung"] != "Aold"]
    assert f == want


# ---- pairs -------------------------------------------------------------------------------------

def test_train_pairs_and_fit_pids_are_t118s():
    rows = _rows()
    for g in ("C19M16", "C12M02", "all"):
        assert pairs.train_pairs(rows, g) == lpairs.train_pairs(rows, g)
        assert pairs.fit_pids(rows, g) == lpairs.fit_pids(rows, g)


def test_law_pairs_depth_counts():
    rows = _rows()
    for track in TRACKS:
        lp = pairs.law_pairs_depth(rows, track)
        assert len(lp) == 6
        assert all(p["arm_src"] == p["arm_tgt"] == "depth" for p in lp)
        assert all(p["direction"] == "arm_to_arm" for p in lp)
        assert all(p["track"] == track for p in lp)
    allp = pairs.law_pairs_depth(rows, "all")
    assert len(allp) == 42
    # exactly the depth->depth subset of t118's law pairs, in t118's order
    want = [p for p in lpairs.law_pairs(rows, "all")
            if p["arm_src"] == "depth" and p["arm_tgt"] == "depth"]
    assert allp == want


def test_law_pairs_scope():
    rows = _rows()
    assert pairs.law_pairs(rows, "all") == pairs.law_pairs_depth(rows, "all")
    assert pairs.law_pairs(rows, "all", scope="depth") == pairs.law_pairs_depth(rows, "all")
    assert pairs.law_pairs(rows, "all", scope="all") == lpairs.law_pairs(rows, "all")
    assert len(pairs.law_pairs(rows, "all", scope="all")) == 2094
    with pytest.raises(ValueError):
        pairs.law_pairs(rows, "all", scope="some")


# ---- CLI ---------------------------------------------------------------------------------------

def _env() -> dict:
    e = dict(os.environ)
    e["PYTHONPATH"] = str(REPO / "src")
    return e


def test_cli_tasks_script_mode():
    out = subprocess.run([sys.executable, str(PAIRS_PY), "tasks", str(FIXTURE_MANIFEST)],
                         capture_output=True, text=True, env=_env(), check=True).stdout
    assert _md5(out) == _md5(_formula_tsv())
    lines = out.splitlines()
    assert len(lines) == 433
    picked = [ln.split("\t")[-1] for ln in lines[1:] if ln.split("\t")[0] in ("18", "402")]
    assert picked == ["A_C19M16_counts_real_s0", "Aold_C19M16_counts_real_s0"]


def test_cli_count(capsys):
    assert pairs.main(["count", str(FIXTURE_MANIFEST)]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "tasks 432", "f_runs_seed0 64", "law_pairs_depth all 42"]
