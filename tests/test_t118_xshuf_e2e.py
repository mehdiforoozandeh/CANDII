"""t118 shuffled-bin twin — end to end: the smoke of a row-2 rung with `--xshuf`, and the task tables.

  * the smoke (`tools/t118/ladder/smoke.py`) for rung A2 with `--xshuf`, 10 steps, 1 seed, 2 jobs, on
    CPU: the twin (model `xshuf`) runs beside real, nocov and ids through every CLI, `aggregate.py`
    gets `--rows 2 --xshuf`, and the outputs carry the twin's checks, report section and grid row;
  * `--xshuf` is refused for a row-1 rung;
  * the task tables through `pairs.py tasks`: row 1 and row 2 keep their pinned md5s, and
    `--row 2 --xshuf` is the 192-row twin table with index 18 as pinned.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LADDER = REPO / "tools" / "t118" / "ladder"
FIXTURE_MANIFEST = REPO / "tests" / "fixtures" / "t112_meta" / "MANIFEST.tsv"
sys.path.insert(0, str(REPO / "tools" / "t118"))
from ladder import figures, pairs  # noqa: E402

#: md5 of `pairs.py tasks tests/fixtures/t112_meta/MANIFEST.tsv` (row 1) and `... --row 2`,
#: measured at 3c9842b (2026-09-28), before the twin existed
ROW1_TASKS_MD5 = "110feba71ebf5f52a4e9cf9e177f443c"
ROW2_TASKS_MD5 = "991cef470efbeaeedb279e62ac45de45"
ROW18 = "18\tA2\tC19M16\tcounts\txshuf\t0\tA2_C19M16_counts_xshuf_s0"


def _env() -> dict:
    e = dict(os.environ)
    e["PYTHONPATH"] = str(REPO / "src")
    return e


def _pairs_tasks(*extra) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(LADDER / "pairs.py"), "tasks", str(FIXTURE_MANIFEST),
                           *extra], env=_env(), capture_output=True, text=True, timeout=120)


# ---- the smoke of A2 with the twin -------------------------------------------------------------

def test_smoke_rung_a2_xshuf_end_to_end(tmp_path):
    work = tmp_path / "smoke_A2x"
    r = subprocess.run([sys.executable, str(LADDER / "smoke.py"), str(work), "A2", "--xshuf",
                        "--steps", "10", "--seeds", "1", "--jobs", "2"], env=_env(),
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert r.stdout.strip().splitlines()[-1] == "SMOKE OK A2"
    agg = work / "agg"
    res = figures.load_results(agg)
    assert figures.check_schema(res) == []
    assert res["rows"] == [2]
    runs = {f"A2_{g}_{s}_{m}_s0" for g in ("T1", "all") for s in pairs.SPACES
            for m in pairs.MODELS + (pairs.MODEL_XSHUF,)}
    assert runs <= set(res["runs_present"])
    assert {r["run_name"] for r in res["per_pair"] if r["kind"] == "trained"} == runs
    assert any(r["model"] == "xshuf" for r in res["per_class"])
    # the twin's own run: its config names the model and the draw rule
    for g in ("T1", "all"):
        for s in pairs.SPACES:
            run = work / "runs" / f"A2_{g}_{s}_xshuf_s0"
            for f in ("ckpt.pt", "scores.json", "law.json", "config.json"):
                assert (run / f).is_file(), run / f
            cfg = json.loads((run / "config.json").read_text())
            assert cfg["model"] == "xshuf" and cfg["row"] == 2
            assert cfg["xshuf"]["rule"] == "same_chrom_uniform"
    # both twin checks, the report section and the grid's third row
    checks = json.loads((agg / "A2" / "checks_A2.json").read_text())["checks"]
    names = {c["check"] for c in checks}
    assert {"beatsxshuf", "lawtest_xshuf"} <= names
    assert "## Against the shuffled-bin twin" in (agg / "A2" / "report.md").read_text()
    grid = (agg / "grid.md").read_text()
    assert grid.startswith("# The transformation grid")
    assert "| row 2, shuffled-bin twin" in grid
    assert any(row["row"] == 3 and row["rung"] == "A2" for row in res["grid"])
    log = (work / "logs" / "agg.log").read_text()
    assert "--rows 2 --xshuf" in log and "report.py grid" in log


def test_smoke_xshuf_refuses_a_row1_rung(tmp_path):
    r = subprocess.run([sys.executable, str(LADDER / "smoke.py"), str(tmp_path / "w"), "A",
                        "--xshuf", "--steps", "1", "--seeds", "1"], env=_env(),
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 2
    assert "row-2 rung" in r.stderr
    assert not (tmp_path / "w").exists()


# ---- the task tables ---------------------------------------------------------------------------

def test_task_tables_pinned_through_the_cli():
    for extra, want in (((), ROW1_TASKS_MD5), (("--row", "2"), ROW2_TASKS_MD5)):
        r = _pairs_tasks(*extra)
        assert r.returncode == 0, r.stderr
        assert hashlib.md5(r.stdout.encode()).hexdigest() == want
    r = _pairs_tasks("--row", "2", "--xshuf")
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    assert len(lines) == 193  # header + 192 runs
    assert lines[19] == ROW18
    assert all(line.split("\t")[4] == "xshuf" for line in lines[1:])
    assert _pairs_tasks("--xshuf").returncode == 2
