"""t118 row 2 — end to end: the smoke for a row-2 rung, and the pinned task tables.

  * the smoke (`tools/t118/ladder/smoke.py`) for rung A2, 10 steps, 1 seed, 2 jobs, on CPU, through
    every CLI of the chain on synthetic products: `aggregate.py --rows 2`, `report.py A2` and
    `report.py grid`; it prints `SMOKE OK A2` and writes the files the acceptance criteria name;
  * the task tables of both rows: `pairs.tasks()` and `pairs.py tasks` give the table the pinned
    formula gives (index = rung*144 + g*18 + space*9 + model*3 + seed), and the row-1 table keeps
    its pinned md5.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LADDER = REPO / "tools" / "t118" / "ladder"
FIXTURE_MANIFEST = REPO / "tests" / "fixtures" / "t112_meta" / "MANIFEST.tsv"
sys.path.insert(0, str(REPO / "tools" / "t118"))
from ladder import figures, pairs  # noqa: E402

#: md5 of `pairs.py tasks tests/fixtures/t112_meta/MANIFEST.tsv` (row 1), measured before row 2
ROW1_TASKS_MD5 = "110feba71ebf5f52a4e9cf9e177f443c"
KEYS = ("index", "rung", "g", "space", "model", "seed", "run_name")
TRACKS = ("C07M20", "C07M29", "C12M02", "C19M16", "C19M22", "C40M17", "C40M18")


def _env() -> dict:
    e = dict(os.environ)
    e["PYTHONPATH"] = str(REPO / "src")
    return e


# ---- the smoke of a row-2 rung ----------------------------------------------------------------

def test_smoke_rung_a2_end_to_end(tmp_path):
    work = tmp_path / "smoke_A2"
    r = subprocess.run([sys.executable, str(LADDER / "smoke.py"), str(work), "A2", "--steps", "10",
                        "--seeds", "1", "--jobs", "2"], env=_env(), capture_output=True, text=True,
                       timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert r.stdout.strip().splitlines()[-1] == "SMOKE OK A2"
    agg = work / "agg"
    res = figures.load_results(agg)
    assert figures.check_schema(res) == []
    assert res["rows"] == [2]
    assert "grid" in res and "rung_choice_row2" in res
    runs = {f"A2_{g}_{s}_{m}_s0" for g in ("T1", "all") for s in pairs.SPACES
            for m in pairs.MODELS}
    assert runs <= set(res["runs_present"])
    assert {r["run_name"] for r in res["per_pair"] if r["kind"] == "trained"} == runs
    for f in ("ckpt.pt", "scores.json", "figdata.npz", "law.json"):
        assert (work / "runs" / "A2_T1_counts_real_s0" / f).is_file()
    for f in ("report.md", "checks_A2.json"):
        assert (agg / "A2" / f).is_file()
    assert (agg / "checks_A2.json").is_file()
    assert (agg / "grid.md").read_text().startswith("# The transformation grid")
    # the aggregate saw row 2 only: no row-1 rung has a report directory
    assert not any((agg / rung).exists() for rung in pairs.RUNGS)
    log = (work / "logs" / "agg.log").read_text()
    assert "--rows 2" in log and "report.py grid" in log


# ---- the task tables ---------------------------------------------------------------------------

def _formula_tsv(rungs) -> str:
    """The task table written out from the pinned formula, independently of `pairs.tasks`."""
    lines = ["\t".join(KEYS)]
    for ri, rung in enumerate(rungs):
        for gi, g in enumerate(TRACKS + ("all",)):
            for si, space in enumerate(("counts", "pval")):
                for mi, model in enumerate(("real", "nocov", "ids")):
                    for seed in (0, 1, 2):
                        index = ri * 144 + gi * 18 + si * 9 + mi * 3 + seed
                        lines.append(f"{index}\t{rung}\t{g}\t{space}\t{model}\t{seed}\t"
                                     f"{rung}_{g}_{space}_{model}_s{seed}")
    return "\n".join(lines) + "\n"


def _md5(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def _tasks_tsv(row: int) -> str:
    rows = pairs.read_manifest(FIXTURE_MANIFEST)
    return "\t".join(KEYS) + "\n" + "".join(
        "\t".join(str(t[k]) for k in KEYS) + "\n" for t in pairs.tasks(rows, row))


@pytest.mark.parametrize("row,rungs", [(1, ("A", "B", "C", "D")), (2, ("A2", "B2", "C2", "D2"))])
def test_tasks_md5_matches_the_pinned_formula(row, rungs, capsys):
    want = _md5(_formula_tsv(rungs))
    assert _md5(_tasks_tsv(row)) == want
    argv = ["tasks", str(FIXTURE_MANIFEST)] + (["--row", "2"] if row == 2 else [])
    assert pairs.main(argv) == 0
    assert _md5(capsys.readouterr().out) == want
    if row == 1:
        assert want == ROW1_TASKS_MD5
    else:
        t = pairs.tasks(None, 2)
        assert sum(x["rung"] == "A2" for x in t) == 144
        assert t[54]["run_name"] == "A2_C19M16_counts_real_s0"
