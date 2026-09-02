"""`tools/candi_in_field.py` — CANDI placed inside the 2019 field, and what it must refuse.

The tool exists because the board cannot answer "CANDI would have placed Nth": the entrants sit in
a separate anchor block, so they never share a ranking denominator with CANDI
(`plan/BENCHMARK_DESIGN.md` §6). Almost every test here is therefore about a boundary rather than
about a number:

* it ranks with `candi.bench.ranking` and with nothing else — a second ranker hidden in a tool is
  the failure §5.3 is written to prevent, so the ranks in the report are compared against a direct
  call to `aggregate_ranks`;
* it reads the `pval` arm only — the jsons here carry a poisoned `count` arm that would invert the
  order if it were ever read;
* it refuses store truth and it refuses a `V_` panel;
* the non-independence count is counted, not read off the row count.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Sequence

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from candi.bench import ranking                                    # noqa: E402
from candi.bench.eic import MEASURES, RANK_DIRECTION               # noqa: E402


def _load():
    spec = importlib.util.spec_from_file_location(
        "candi_in_field", REPO / "tools" / "candi_in_field.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["candi_in_field"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tool():
    return _load()


# ---------------------------------------------------------------------------
# synthetic score jsons — the 25-entrant field, plus CANDI
# ---------------------------------------------------------------------------

#: The eight reported measures. `msevar` is excluded everywhere (entrants README §1).
USED = tuple(m for m in MEASURES if m != "msevar")

#: 25 entrant slugs. The three byte-identical groups of `competitors/entrants/README.md` §7 are all
#: represented, because the count they force is the thing under test.
ENTRANTS = [
    "CUImpute1", "CUWA", "ICU", "Avocado_p0", "Average",
    "Hongyang_Li_and_Yuanfang_Guan", "Hongyang_Li_and_Yuanfang_Guan_v1",
    "LiPingChun", "imp", "BrokenNodes", "Guacamole", "KKT_ENCODE_Impute",
    "NittanyLions2", "Song_Lab", "UIOWA_Michaelson", "Aug2Sep", "CostaLab",
    "Lavawizard2019", "PREDICTD_reboot", "Kongstantinos", "DrChromosome",
    "hihiclub", "tanjianeng", "BlindDate", "LadyGaga",
]
ASSAYS = ("H3K4me3", "H3K36me3", "H3K27ac", "ATAC-seq")
N_EXP = 51


def _experiments(n: int = N_EXP) -> List[str]:
    return [f"T_{i:02d}|B_{i:02d}|{ASSAYS[i % len(ASSAYS)]}" for i in range(n)]


def _q(i: int, e: int) -> float:
    """A per-(team, experiment) quality in (0, 1); lower is better on every measure.

    Jittered on purpose. With one constant quality per team, every team below the median would be
    below the median in EVERY experiment and stage 3's `min(0.5, r/n)` cap would tie the whole
    bottom half — the cap behaving exactly as its docstring says, but a degenerate fixture. The
    jitter makes per-experiment ranks move, so the aggregate separates.
    """
    return 0.20 + 0.008 * i + 0.16 * (((i * 7 + e * 11) % 11) / 10.0)


def _row(q: float, *, assay: str, keys: Sequence[str] = USED,
         extra: Dict[str, float] | None = None) -> Dict[str, object]:
    out: Dict[str, object] = {
        m: (1.0 - q if RANK_DIRECTION[m] == "ASCENDING" else q) for m in keys}
    out.update({"assay": assay, "kind": "impute", "n_bins": 1000,
                "chroms": ["chr20", "chr21", "chr22"], "bin_scope": "full",
                "pred_space": "-log10p"})
    if extra:
        out.update(extra)
    return out


def _score_json(method: str, i: int, *, truth: str = "challenge", panel: str = "B_",
                n_exp: int = N_EXP, drop_measure: str | None = None,
                with_msevar: bool = False, missing: Sequence[str] = ()) -> dict:
    """One `candi.bench.external` score json, cut to what the tool reads.

    The `count` arm is POISONED — its measures run backwards — so any test that gets the same
    ranking as a direct `aggregate_ranks` call over the pval arm has also proved the tool never
    looked at it. Under real challenge truth the count arm is absent entirely; a tool that reads it
    anyway would fail silently on real data and loudly here.
    """
    keys = tuple(m for m in USED if m != drop_measure)
    per_track: Dict[str, Dict[str, object]] = {}
    for e, key in enumerate(_experiments(n_exp)):
        assay = key.split("|")[2]
        if key in missing:
            continue
        tgt = key.split("|")[1].replace("B_", panel)
        per_track[f"{key.split('|')[0]}|{tgt}|{assay}"] = {
            "pval": _row(_q(i, e), assay=assay, keys=keys,
                         extra={"msevar": _q(i, e)} if with_msevar else None),
            "count": _row(1.0 - _q(i, e), assay=assay),
        }
    return {
        "provenance": {
            "method": method,
            "suite": "candi.bench.external",
            "truth": ({"source": "store"} if truth == "store" else
                      {"source": "challenge", "root": "/project/.../t81_truth_challenge/B_",
                       "manifest_sha256": "0" * 64}),
            "pred_root": f"/project/.../t81_pred_B/anchor/{method}/B_",
            "missing_tracks": list(missing),
            "allow_missing": bool(missing),
        },
        "tracks": sorted(per_track),
        "per_track": per_track,
        "macro": {}, "panels": {}, "panel": {}, "ranking": None,
    }


def _write(path: Path, doc: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


@pytest.fixture
def field(tmp_path):
    """25 anchors in the pinned layout plus a CANDI json, and the paths to both."""
    anchors = tmp_path / "anchor"
    for i, name in enumerate(ENTRANTS):
        _write(anchors / name / "challenge.B_.json", _score_json(name, i + 4))
    candi = _write(tmp_path / "CANDI" / "challenge.B_.json", _score_json("CANDI", 0))
    return {"anchors": anchors, "candi": candi, "out": tmp_path / "out"}


def _run(tool, field, *extra) -> int:
    return tool.main(["--candi", str(field["candi"]), "--anchors", str(field["anchors"]),
                      "--out", str(field["out"]), *extra])


def _md(field) -> str:
    return (field["out"] / "candi_in_2019_field.md").read_text(encoding="utf-8")


def _table_rows(md: str) -> List[List[str]]:
    """The ranked table's data rows, as split cells."""
    rows = []
    for line in md.splitlines():
        if not line.startswith("| ") or "|---" in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and cells[0].isdigit():
            rows.append(cells)
    return rows


def _expected_ranks(tool, field) -> Dict[str, int]:
    """The ranking, from the ONLY ranker, computed independently of the report."""
    entries = [tool.load_score(field["candi"])] + [
        tool.load_score(field["anchors"] / n / "challenge.B_.json") for n in ENTRANTS]
    table, _exps, _dropped = tool.build_table(entries)
    return ranking.aggregate_ranks(table)["final_rank"]


# ---------------------------------------------------------------------------
# 1 — the table
# ---------------------------------------------------------------------------

def test_every_method_gets_one_row_and_candi_is_marked(tool, field):
    assert _run(tool, field) == 0
    rows = _table_rows(_md(field))
    assert len(rows) == 26, "1 CANDI + 25 entrants, one row each"
    assert [int(r[0]) for r in rows] == list(range(1, 27))
    marked = [r for r in rows if "CANDI" in r[6]]
    assert len(marked) == 1 and marked[0][1] == "**CANDI**"
    names = {r[1].strip("*") for r in rows}
    assert names == {"CANDI", *ENTRANTS}


def test_the_ranks_are_the_only_rankers_ranks(tool, field):
    """No second ranker. The report's order must equal `aggregate_ranks`, method for method."""
    assert _run(tool, field) == 0
    want = _expected_ranks(tool, field)
    got = {r[1].strip("*"): int(r[0]) for r in _table_rows(_md(field))}
    assert got == want


def test_the_poisoned_count_arm_is_never_read(tool, field):
    """The fixture's count arm runs backwards; reading it would invert the table."""
    assert _run(tool, field) == 0
    order = [r[1].strip("*") for r in _table_rows(_md(field))]
    want = _expected_ranks(tool, field)
    assert order == sorted(want, key=lambda t: want[t])
    # and the count arm really would have moved it — otherwise the assertion above is vacuous
    assert order != order[::-1]
    assert order[0] != order[-1]


def test_the_placement_and_the_resolution_limit_travel_together(tool, field):
    assert _run(tool, field) == 0
    md = _md(field)
    assert f"{ranking.RESOLUTION_LIMIT_CORR} correlation units" in md
    assert f"{ranking.UNSEPARABLE_ADJACENT_PAIRS} of 24 adjacent pairs invert on ≥ 3 of the ten " \
           "chromosome subsets" in md
    want = _expected_ranks(tool, field)
    assert f"CANDI places **{want['CANDI']} of 26**" in md
    # every sentence that states the placement carries the limit in the same breath
    for line in md.splitlines():
        if "CANDI places" in line:
            assert "correlation units" in line


def test_one_bootstrap_is_declared_not_hidden(tool, field):
    assert _run(tool, field) == 0
    md = _md(field)
    assert "| bootstraps | 1 |" in md
    assert "second-best of ten" in md


# ---------------------------------------------------------------------------
# 2 — the non-independence count
# ---------------------------------------------------------------------------

def test_three_byte_identical_groups_by_default(tool, field):
    assert _run(tool, field) == 0
    md = _md(field)
    assert "Byte-identical entrant groups: 3" in md
    assert len(tool.IDENTICAL_GROUPS) == 3
    for group in tool.IDENTICAL_GROUPS:
        for member in group["members"]:
            assert f"`{member}`" in md
        assert "competitors/entrants/README.md" in group["evidence"]
    assert "CUImpute1" in md and "Non-independence" in md


def _naive_distinct(names: Sequence[str], groups) -> int:
    """An independent count: merge overlapping member sets until nothing changes."""
    comps = [{n} for n in dict.fromkeys(names)]
    for g in groups:
        members = {m for m in g["members"] if m in set(names)}
        if len(members) < 2:
            continue
        touched = [c for c in comps if c & members]
        comps = [c for c in comps if not (c & members)]
        comps.append(set().union(*touched) if touched else members)
    return len(comps)


def test_beats_n_is_counted_over_distinct_submissions(tool, field):
    assert _run(tool, field) == 0
    md = _md(field)
    rows = _table_rows(md)
    order = [r[1].strip("*") for r in rows]
    below = order[order.index("CANDI") + 1:]
    want = _naive_distinct(below, tool.IDENTICAL_GROUPS)
    assert f"rows below CANDI: **{len(below)}**" in md
    assert f"**distinct submissions below CANDI: {want}**" in md
    assert "must be counted, never read off the" in md
    assert want <= len(below)


def test_the_count_collapses_an_overlapping_group_once(tool):
    """`ICU` is in two groups; a per-group subtraction would double-count it."""
    names = ["CUImpute1", "CUWA", "ICU", "Avocado_p0", "Average", "LiPingChun"]
    n, merged = tool.count_distinct(names, tool.normalise_groups(
        [dict(g) for g in tool.IDENTICAL_GROUPS]))
    assert n == 3, "the four linked entrants collapse to one, plus Average and LiPingChun"
    assert merged == [["Avocado_p0", "CUImpute1", "CUWA", "ICU"]]
    assert n == _naive_distinct(names, tool.IDENTICAL_GROUPS)


def test_identical_groups_can_be_overridden(tool, field, tmp_path):
    override = tmp_path / "groups.json"
    override.write_text(json.dumps([["CUWA", "ICU"]]), encoding="utf-8")
    assert _run(tool, field, "--identical-groups", str(override)) == 0
    md = _md(field)
    assert "Byte-identical entrant groups: 1" in md
    assert "| A | `CUWA`, `ICU` |" in md


# ---------------------------------------------------------------------------
# 3 — the two refusals
# ---------------------------------------------------------------------------

def test_store_truth_candi_json_is_refused(tool, field):
    _write(field["candi"], _score_json("CANDI", 0, truth="store"))
    assert _run(tool, field) == tool.EXIT_REFUSED
    assert not field["out"].exists()


def test_a_v_panel_candi_json_is_refused(tool, field):
    _write(field["candi"], _score_json("CANDI", 0, panel="V_"))
    assert _run(tool, field) == tool.EXIT_REFUSED
    assert not field["out"].exists()


def test_a_store_truth_anchor_is_refused_too(tool, field):
    """The same gate on the anchors: a `store.B_.json` swept up by a glob is the real case."""
    _write(field["anchors"] / "ICU" / "challenge.B_.json",
           _score_json("ICU", 6, truth="store"))
    assert _run(tool, field) == tool.EXIT_REFUSED


def test_two_files_naming_the_same_method_are_refused(tool, field):
    _write(field["anchors"] / "ICU" / "challenge.B_.json", _score_json("CUWA", 6))
    assert _run(tool, field) == tool.EXIT_REFUSED


def test_an_empty_anchor_directory_is_an_input_error(tool, field, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert tool.main(["--candi", str(field["candi"]), "--anchors", str(empty),
                      "--out", str(field["out"])]) == tool.EXIT_INPUT


# ---------------------------------------------------------------------------
# 4 — the figure
# ---------------------------------------------------------------------------

def test_svg_is_written_and_parses_as_xml(tool, field):
    assert _run(tool, field) == 0
    svg = field["out"] / "candi_in_2019_field.svg"
    root = ET.fromstring(svg.read_text(encoding="utf-8"))
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    ns = "{http://www.w3.org/2000/svg}"
    texts = [t.text or "" for t in root.iter(f"{ns}text")]
    assert "CANDI" in texts, "the highlighted row is labelled by name"
    for name in ENTRANTS:
        assert name in texts
    # one dot plus one surface ring per method, and CANDI's dot in the accent class
    classes = [c.get("class") for c in root.iter(f"{ns}circle")]
    assert classes.count("dot") == 25 and classes.count("dot-hi") == 1
    assert classes.count("ring") == 26
    body = svg.read_text(encoding="utf-8")
    assert "prefers-color-scheme: dark" in body, "theme-neutral: both modes are defined"
    assert "lower is better" in body
    assert str(ranking.RESOLUTION_LIMIT_CORR) in body


def test_no_text_runs_off_the_canvas(tool, field):
    """An SVG has no layout engine: an unmeasured label overflows silently and stays overflowed.

    The longest entrant name is 32 characters and the caption sentences are ~200, so this is the
    check that keeps the head block, the name column and the captions inside the box.
    """
    assert _run(tool, field) == 0
    root = ET.fromstring((field["out"] / "candi_in_2019_field.svg").read_text(encoding="utf-8"))
    ns = "{http://www.w3.org/2000/svg}"
    width, height = float(root.get("width")), float(root.get("height"))
    sizes = {"title": 15.0, "sub": 11.0, "cap": 10.5, "name": 11.0, "name-hi": 11.0,
             "rank": 10.0, "tick": 10.0, "axlabel": 10.5, "val": 10.0, "tag": 8.5}
    for t in root.iter(f"{ns}text"):
        w = tool._text_w(t.text or "", sizes[t.get("class")])
        x, y = float(t.get("x")), float(t.get("y"))
        anchor = t.get("text-anchor", "start")
        left = x if anchor == "start" else (x - w / 2 if anchor == "middle" else x - w)
        assert left >= 0, (t.get("class"), t.text)
        assert left + w <= width, (t.get("class"), t.text, left + w, width)
        assert 0 < y <= height, (t.get("class"), t.text)


def _caption(field) -> str:
    """Every caption line joined back into one string — the lines are wrapped to the canvas."""
    root = ET.fromstring((field["out"] / "candi_in_2019_field.svg").read_text(encoding="utf-8"))
    ns = "{http://www.w3.org/2000/svg}"
    return " ".join(t.text or "" for t in root.iter(f"{ns}text")
                    if t.get("class") == "cap")


def test_the_figure_carries_the_group_tags_and_the_counted_n(tool, field):
    assert _run(tool, field) == 0
    body = (field["out"] / "candi_in_2019_field.svg").read_text(encoding="utf-8")
    assert "byte-identical submission group" in body
    caption = _caption(field)
    for group in tool.IDENTICAL_GROUPS:
        assert f"{group['tag']} = " + ", ".join(group["members"]) in caption
    order = [r[1].strip("*") for r in _table_rows(_md(field))]
    below = order[order.index("CANDI") + 1:]
    assert f"{len(below)} rows below it, " \
           f"{_naive_distinct(below, tool.IDENTICAL_GROUPS)} distinct submissions" in caption


# ---------------------------------------------------------------------------
# 5 — the measure set, and a panel with a hole in it
# ---------------------------------------------------------------------------

def test_msevar_is_dropped_and_a_partly_present_measure_is_reported(tool, field):
    """`msevar` never ranks; a measure one team lacks is dropped for everyone, and said out loud."""
    for i, name in enumerate(ENTRANTS):
        _write(field["anchors"] / name / "challenge.B_.json",
               _score_json(name, i + 4, with_msevar=True))
    _write(field["anchors"] / "ICU" / "challenge.B_.json",
           _score_json("ICU", 6, with_msevar=True, drop_measure="mseenh"))
    _write(field["candi"], _score_json("CANDI", 0, with_msevar=True))
    assert _run(tool, field) == 0
    md = _md(field)
    assert "| measures used | 7 —" in md
    assert "`msevar`" in md and "`mseenh`" in md
    for kept in ("mse", "gwcorr", "gwspear", "mse1obs", "mse1imp", "mseprom", "msegene"):
        assert f"`{kept}`" in md


def test_a_team_missing_an_experiment_still_gets_a_row(tool, field):
    """`UIOWA_Michaelson` never submitted `C38M18`; the ranker scores the gap at the 0.5 cap."""
    gone = _experiments()[7]
    _write(field["anchors"] / "UIOWA_Michaelson" / "challenge.B_.json",
           _score_json("UIOWA_Michaelson", 18, missing=(gone,)))
    assert _run(tool, field) == 0
    md = _md(field)
    row = [r for r in _table_rows(md) if r[1].strip("*") == "UIOWA_Michaelson"]
    assert len(row) == 1 and row[0][5] == str(N_EXP - 1)
    assert "| experiments | 51 |" in md
    assert "MISSING_SCORE" in md and "UIOWA_Michaelson" in md


def test_anchors_may_be_a_glob(tool, field):
    assert tool.main(["--candi", str(field["candi"]),
                      "--anchors", str(field["anchors"] / "*" / "challenge.B_.json"),
                      "--out", str(field["out"])]) == 0
    assert len(_table_rows(_md(field))) == 26


def test_the_candi_file_swept_up_by_the_glob_is_not_ranked_twice(tool, field, tmp_path):
    """A glob wide enough to catch CANDI's own json must not seat it twice."""
    candi_in_tree = _write(field["anchors"] / "CANDI" / "challenge.B_.json",
                           _score_json("CANDI", 0))
    assert tool.main(["--candi", str(candi_in_tree),
                      "--anchors", str(field["anchors"]),
                      "--out", str(field["out"])]) == 0
    rows = _table_rows(_md(field))
    assert len(rows) == 26
    assert [r[1].strip("*") for r in rows].count("CANDI") == 1
