"""t396 iteration 2 chunk A3: gate (f) learning-rate pick (`tools/t396/select_lr.py`) on
synthetic run folders."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools" / "t396") not in sys.path:
    sys.path.insert(0, str(REPO / "tools" / "t396"))

import select_lr as SL  # noqa: E402


def _run(root: Path, arm: str, lr: str, seed: int, last: float, n: int = 12,
         done: bool = True, capped: bool = False) -> None:
    d = root / f"{arm}__lr{lr}" / f"seed{seed}"
    d.mkdir(parents=True)
    lines = []
    for i in range(n):                      # early entries are high: only the last 10 count
        v = 100.0 if i < n - SL.LAST_N else last
        lines.append(json.dumps({"step": i, "loss": 1.0, "heldout_loss": v}))
        lines.append(json.dumps({"step": i, "loss": 1.0}))
    (d / "log.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if done:
        (d / "DONE").write_text("1\n")
    if capped:
        (d / "CAPPED").write_text("{}")


def _grid(root: Path, arm: str, means: dict) -> None:
    for lr, base in means.items():
        for s, off in enumerate((-0.01, 0.0, 0.01)):
            _run(root, arm, lr, s, base + off)


def _rows(path: Path) -> list[dict]:
    return list(csv.DictReader(open(path, encoding="utf-8"), delimiter="\t"))


def test_picks_lowest_mean_and_flags_edges(tmp_path):
    root = tmp_path / "runs"
    _grid(root, "mid_arm", {"0.0003": 2.0, "0.001": 1.5, "0.003": 1.8})
    _grid(root, "edge_arm", {"0.0003": 1.2, "0.001": 1.5, "0.003": 1.8})
    out = tmp_path / "sweep.tsv"
    assert SL.main(["--runs-root", str(root), "--out", str(out)]) == 0
    rows = _rows(out)
    assert list(rows[0]) == list(SL.COLUMNS)
    assert len(rows) == 6
    picked = {r["arm"]: r for r in rows if r["status"] == "picked"}
    assert picked["mid_arm"]["lr"] == "0.001" and picked["mid_arm"]["edge_pick"] == "no"
    assert picked["edge_arm"]["lr"] == "0.0003" and picked["edge_arm"]["edge_pick"] == "yes"
    r = picked["mid_arm"]
    assert float(r["heldout_mean"]) == 1.5          # the 100.0 entries before the last 10 drop
    assert abs(float(r["seed_range"]) - 0.02) < 1e-9
    assert r["n_seeds"] == "3"
    assert r["heldout_per_seed"].startswith("seed0:")


def test_extension_rows_join_the_pick(tmp_path):
    root = tmp_path / "runs"
    _grid(root, "arm", {"0.0003": 1.2, "0.001": 1.5, "0.003": 1.8, "0.0001": 1.3})
    out = tmp_path / "sweep.tsv"
    assert SL.main(["--runs-root", str(root), "--out", str(out)]) == 0
    rows = _rows(out)
    assert [r["lr"] for r in rows] == ["0.0001", "0.0003", "0.001", "0.003"]
    picked = [r for r in rows if r["status"] == "picked"]
    assert len(picked) == 1 and picked[0]["lr"] == "0.0003" and picked[0]["edge_pick"] == "no"


def test_unfinished_run_blocks_the_pick(tmp_path, capsys):
    root = tmp_path / "runs"
    _grid(root, "arm", {"0.0003": 2.0, "0.001": 1.5})
    _run(root, "arm", "0.003", 0, 1.0)
    _run(root, "arm", "0.003", 1, 1.0, capped=True)
    _run(root, "arm", "0.003", 2, 1.0, n=5)
    out = tmp_path / "sweep.tsv"
    assert SL.main(["--runs-root", str(root), "--out", str(out)]) == 1
    rows = _rows(out)
    assert not any(r["status"] == "picked" for r in rows)
    assert {r["lr"]: r["n_seeds"] for r in rows}["0.003"] == "1"
    assert "no pick" in capsys.readouterr().err
