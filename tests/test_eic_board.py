"""The landing-page EIC board (tools/eic_board.py): it must refuse source tables that no longer
describe a reproduced board, and its build must be deterministic and self-contained."""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import eic_board as eb  # noqa: E402


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    dst = tmp_path / "eic"
    shutil.copytree(eb.ROOT, dst)
    return dst


def rewrite(path: Path, edit) -> None:
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    fields = list(rows[0])
    edit(rows)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def test_the_shipped_board_passes_its_gates() -> None:
    assert eb.main(["check"]) == 0


def test_each_layer_has_the_published_field_and_every_candi_arm() -> None:
    board = eb.compile_board(eb.ROOT)
    assert [l["id"] for l in board["layers"]] == ["original", "qn", "corrected"]
    for layer in board["layers"]:
        assert [p["position"] for p in layer["published"]] == list(range(1, 26))
        assert len(layer["candi"]) == len(eb.ARMS)
        assert all(c["of"] == 26 for c in layer["candi"])


def test_the_board_carries_the_reproduced_headline() -> None:
    """Max's sprint 04: CANDI + tcfloor is 1st of 26 on all three boards; 0.2287 on MOESM4."""
    layers = {l["id"]: l for l in eb.compile_board(eb.ROOT)["layers"]}
    top = {k: next(c for c in l["candi"] if c["name"] == "CANDI + tcfloor") for k, l in layers.items()}
    assert {k: c["position"] for k, c in top.items()} == {"original": 1, "qn": 1, "corrected": 1}
    assert top["corrected"]["score"] == 0.2287
    avg = {k: next(p for p in l["published"] if p["name"] == "Average")["position"]
           for k, l in layers.items()}
    assert avg == {"original": 3, "qn": 6, "corrected": 17}


def test_a_board_that_moved_an_entry_is_refused(root: Path) -> None:
    def move(rows):
        rows[0]["position_ours"], rows[0]["shift"] = "2", "1"
    rewrite(root / "source" / "qn_vs_MOESM4_board.tsv", move)
    with pytest.raises(eb.GateError, match="not in its published position"):
        eb.compile_board(root)


def test_an_exact_layer_whose_scores_drift_is_refused(root: Path) -> None:
    def drift(rows):
        rows[3]["score_mean_ours"] = str(float(rows[3]["score_mean_pub"]) * 1.001)
    rewrite(root / "source" / "pe_vs_MOESM2_board.tsv", drift)
    with pytest.raises(eb.GateError, match="reproduced"):
        eb.compile_board(root)


def test_a_missing_candi_arm_is_refused(root: Path) -> None:
    rewrite(root / "source" / "candi_placement.tsv",
            lambda rows: rows.__setitem__(slice(None), [r for r in rows if not (
                r["board"] == "published MOESM3" and r["arm"] == "CANDI + tcfloor")]))
    with pytest.raises(eb.GateError, match="0 rows"):
        eb.compile_board(root)


def test_the_site_loads_nothing_from_outside(root: Path) -> None:
    page = root / "site" / "index.html"
    page.write_text(page.read_text(encoding="utf-8").replace(
        "</head>", '<script src="https://cdn.example.org/x.js"></script></head>'), encoding="utf-8")
    with pytest.raises(eb.GateError, match="external resource"):
        eb.main(["--root", str(root), "check"])


def test_build_writes_the_page_and_its_data(tmp_path: Path) -> None:
    out = tmp_path / "_site"
    eb.main(["build", "--out", str(out)])
    assert sorted(p.name for p in out.iterdir()) == sorted(["eic.json", *eb.SITE_FILES])
    assert json.loads((out / "eic.json").read_text(encoding="utf-8"))["schema"] == "eic-board/1"
