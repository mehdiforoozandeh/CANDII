"""`tools/seed_floor.py` — the tool that says what a seed change moves under the NEW instrument.

Tested because of what tonight's run found twice over: a tool with no test is a tool that has never
run, and a fixture that only a human re-records goes stale in silence. This one is small enough
that the tests are mostly about what it must REFUSE to say.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location("seed_floor", REPO / "tools" / "seed_floor.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["seed_floor"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tool():
    return _load()


def _bench(crps: float, tracks: dict, *, oracle: float | None = None,
           scale_err: float = 0.0) -> dict:
    """`oracle` defaults to the track's own crps, i.e. a perfectly scaled model."""
    return {"macro": {"count": {"crps": crps, "gwspear": 0.5}},
            "per_track": {k: {"count": {"crps": v,
                                        "crps_oracle_scaled": v if oracle is None else oracle,
                                        "scale_error": scale_err}}
                          for k, v in tracks.items()}}


def _write(tmp_path, name, doc):
    p = tmp_path / name
    p.write_text(json.dumps(doc))
    return str(p)


def test_the_same_run_twice_has_a_spread_of_exactly_zero(tool, tmp_path, capsys) -> None:
    """The anchor. Any non-zero here is the tool inventing variation, which is the one thing a
    noise-floor tool may never do."""
    a = _write(tmp_path, "a.json", _bench(1.5, {"T|B|X": 1.0, "T|B|Y": 2.0}))
    tool.main([a, a])
    out = capsys.readouterr().out
    body = [l for l in out.splitlines() if l.startswith("| ") and "spread" not in l]
    assert body
    for line in body:
        cells = [c.strip().strip("*") for c in line.strip("| \n").split("|")]
        numeric = [c for c in cells if c.replace(".", "", 1).replace("-", "", 1).isdigit()]
        assert numeric, line
        # every numeric cell is a value or a spread; the spreads are the trailing ones and all
        # of them must be exactly zero when the two inputs are the same file.
        assert float(numeric[-1]) == 0.0, line


def test_a_pair_reports_the_absolute_difference_and_not_a_signed_one(tool, tmp_path, capsys) -> None:
    """A floor is a magnitude. Order of arguments must not change it."""
    a = _write(tmp_path, "a.json", _bench(1.50, {"T|B|X": 1.0}))
    b = _write(tmp_path, "b.json", _bench(1.25, {"T|B|X": 1.4}))
    tool.main([a, b])
    first = capsys.readouterr().out
    tool.main([b, a])
    second = capsys.readouterr().out
    assert "**0.25000**" in first and "**0.25000**" in second
    assert "0.40000" in first and "0.40000" in second


def test_three_runs_report_the_range_not_a_pairwise_difference(tool, tmp_path, capsys) -> None:
    runs = [_write(tmp_path, f"{i}.json", _bench(c, {"T|B|X": 1.0}))
            for i, c in enumerate((1.0, 1.7, 1.2))]
    tool.main(runs)
    assert "**0.70000**" in capsys.readouterr().out          # max - min, not |first - last|


def test_one_run_is_refused_because_it_has_no_spread(tool, tmp_path) -> None:
    a = _write(tmp_path, "a.json", _bench(1.0, {}))
    with pytest.raises(SystemExit, match="two seeds at minimum"):
        tool.main([a])


def test_the_suite_is_detected_from_the_file_rather_than_assumed(tool, tmp_path) -> None:
    assert tool.detect(_bench(1.0, {})) == "bench"
    assert tool.detect({"M1": {"imp_macro_crps": 1.0}}) == "eval"


def test_only_tracks_present_in_every_run_are_compared(tool, tmp_path, capsys) -> None:
    """A track scored in one run and not another has no spread; including it would silently
    compare a number against nothing."""
    a = _write(tmp_path, "a.json", _bench(1.0, {"T|B|X": 1.0, "T|B|ONLY_A": 9.0}))
    b = _write(tmp_path, "b.json", _bench(1.0, {"T|B|X": 1.0}))
    tool.main([a, b])
    out = capsys.readouterr().out
    assert "T|B|X" in out.replace("\\|", "|")
    assert "ONLY_A" not in out
    assert "1 `impute` tracks common to every run" in out


def test_a_raw_spread_that_is_all_scale_is_named_as_scale(tool, tmp_path, capsys) -> None:
    """`AGENTS.md` section 7.2's split, doing the work it exists for.

    A track whose RAW CRPS doubles across seeds while its oracle-scaled number barely moves has
    changed its calibration, not its ranking. Quoting the raw number alone reads as the model
    falling apart. The report must say which of the two happened.
    """
    a = _write(tmp_path, "a.json", _bench(1.0, {"T|B|X": 1.10}, oracle=1.04, scale_err=0.07))
    b = _write(tmp_path, "b.json", _bench(1.0, {"T|B|X": 2.20}, oracle=1.36, scale_err=0.84))
    tool.main([a, b])
    out = capsys.readouterr().out
    assert "most of the 1.10000 is SCALE and not ranking" in out


def test_it_never_calls_a_range_a_confidence_interval(tool, tmp_path, capsys) -> None:
    """Two or three seeds are a magnitude. `AGENTS.md` section 7.2 exists because this distinction
    was lost once already."""
    runs = [_write(tmp_path, f"{i}.json", _bench(1.0 + i / 10, {"T|B|X": 1.0}))
            for i in range(3)]
    tool.main(runs)
    out = capsys.readouterr().out.lower()
    assert "confidence interval" not in out
    assert "not a distribution" in out


def test_the_per_track_block_is_one_arm_and_not_the_union(tool, tmp_path, capsys) -> None:
    """bench keeps both arms in one `per_track` map, and denoising is the easier task with the
    smaller spread and, on the t22 panel, more than twice the track count. A median over the union
    is a median of the denoise arm wearing both names, and it is not comparable to `eval.py`'s
    imputation-only one."""
    a = _write(tmp_path, "a.json", _bench(1.0, {"T|B|X": 1.0, "T|T|Y|denoise": 1.0}))
    b = _write(tmp_path, "b.json", _bench(1.0, {"T|B|X": 2.0, "T|T|Y|denoise": 1.0}))
    tool.main([a, b])                                   # default arm=impute
    out = capsys.readouterr().out
    assert "1 `impute` tracks" in out
    assert "denoise" not in out.replace("--arm", "")
    assert "median 1.00000" in out                      # not 0.5, which the union would give

    tool.main([a, b, "--arm", "denoise"])
    assert "1 `denoise` tracks" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# --panel: one floor per panel, because the three panels are not one exam
# ---------------------------------------------------------------------------

#: Five scored tracks over two target cells. `V_aa` poses three assays, `B_bb` poses two, so
#: `V_matched` is a MEASURED subset of `V_breadth` (H3K27ac drops out) rather than a listed one —
#: which is the only arrangement in which the three panels can be told apart at all.
PANEL_TRACKS = {
    "T_x0|V_aa|ATAC-seq": 1.0,
    "T_x0|V_aa|H3K4me3": 1.2,
    "T_x0|V_aa|H3K27ac": 1.4,
    "T_x1|B_bb|ATAC-seq": 2.0,
    "T_x1|B_bb|H3K4me3": 2.2,
}

#: The assay `V_aa` poses and `B_bb` does not. `V_matched` must not carry it; `V_breadth` must.
UNMATCHED_ASSAY = "H3K27ac"


def _pblock(crps: float, *, arm: str = "count", n_exp: int = 2,
            assays=("ATAC-seq", "H3K4me3"), ranked: bool = True, drop=()) -> dict:
    """One `panels[arm][panel]` block, shaped as `harness.panel_macros` writes one.

    Hand-written so this file needs no `candi` import — the tool reads a finished score json off
    disk and should be testable wherever that json is. The fixture-vs-`panel_macros` test below is
    what stops the shape drifting into fiction.
    """
    if arm == "count":
        row = {"crps": crps, "crps_oracle_scaled": crps - 0.05, "scale_error": 0.05,
               "gwspear": 0.5, "n_points": 1000.0}
    else:
        row = {"crps": crps, "pit_ks": 0.1, "coverage_95": 0.94, "n_points": 1000.0}
    for k in drop:
        row.pop(k, None)
    row.update({f"{k}_n_tracks": n_exp for k in list(row)})
    out = {**row, "n_tracks": n_exp, "n_experiments": n_exp, "assays": list(assays),
           "ranked": ranked}
    if not ranked:                        # panel_macros stamps these two on V_matched only
        out["matched_to"] = list(assays)
        out["note"] = "NOT RANKED."
    return out


def _panels(v_breadth: float, v_matched: float, b: float, **kw) -> dict:
    """`panels[arm][panel]` for both arms. The pval arm sits 0.5 below the count arm, so a test can
    say WHICH arm a printed number came from and not merely that some number was printed."""
    return {arm: {"V_breadth": _pblock(v_breadth + off, arm=arm, n_exp=3,
                                       assays=("ATAC-seq", UNMATCHED_ASSAY, "H3K4me3"), **kw),
                  "V_matched": _pblock(v_matched + off, arm=arm, ranked=False, **kw),
                  "B": _pblock(b + off, arm=arm, **kw)}
            for arm, off in (("count", 0.0), ("pval", -0.5))}


def _panelled(macro_crps: float, tracks: dict, panels: dict, *, gw: dict | None = None) -> dict:
    """A bench score json carrying the two blocks `--panel` and `--scope` address."""
    def _pt(bump: float) -> dict:
        return {k: {"count": {"kind": "impute", "assay": k.split("|")[2], "crps": v + bump,
                              "crps_oracle_scaled": v + bump - 0.05, "scale_error": 0.05}}
                for k, v in tracks.items()}
    doc = {"macro": {"count": {"crps": macro_crps, "gwspear": 0.5}},
           "per_track": _pt(0.0), "panels": panels}
    if gw is not None:
        # A genome-wide block is the same three aggregations over more chromosomes. The +0.5 bump on
        # its per-track rows is what lets a test prove the tool read THIS block and not the top one.
        doc["genome_wide"] = {"chroms": ["chr1"], "per_track": _pt(0.5),
                              "macro": {"count": {"crps": macro_crps + 0.5, "gwspear": 0.5}},
                              "panels": gw}
    return doc


def _tracks(bump: float = 0.0) -> dict:
    return {k: v + bump for k, v in PANEL_TRACKS.items()}


def test_the_panel_fixture_is_the_block_the_harness_writes(tool) -> None:
    """The fixtures above are hand-written JSON, so every `--panel` test below runs without
    importing `candi`. This is the one test that holds them against the production function:
    `harness.panel_macros` writes the block the tool reads, and a fixture whose key set has drifted
    from it proves nothing about the tool."""
    H = pytest.importorskip("candi.bench.harness")
    per_track = {k: {"count": {"kind": "impute", "assay": k.split("|")[2], "crps": v,
                               "crps_oracle_scaled": v - 0.05, "scale_error": 0.05,
                               "gwspear": 0.5, "n_points": 1000.0}}
                 for k, v in PANEL_TRACKS.items()}
    real = H.panel_macros(per_track, "count")
    mine = _panels(1.2, 1.1, 2.1)["count"]
    assert set(real) == set(mine) == set(tool.PANELS) == set(H.PANELS)
    for name in tool.PANELS:
        assert set(real[name]) == set(mine[name]), name
    # and the fixture's own arithmetic: V_matched really is V_ minus the assay B_ never posed
    assert real["V_matched"]["matched_to"] == ["ATAC-seq", "H3K4me3"]
    assert UNMATCHED_ASSAY in real["V_breadth"]["assays"]
    assert UNMATCHED_ASSAY not in real["V_matched"]["assays"]


@pytest.mark.parametrize("panel,expect", [("V_breadth", 1.20), ("V_matched", 1.10), ("B", 2.10)])
def test_each_panel_is_read_off_its_own_block_and_no_other(tool, tmp_path, capsys, panel,
                                                           expect) -> None:
    """The whole point of the option. Three panels over one scored pass, three different means over
    three different track populations — so three different floors, and a spread read on one is not
    the resolution of another."""
    a = _write(tmp_path, "a.json", _panelled(9.9, _tracks(), _panels(1.20, 1.10, 2.10)))
    b = _write(tmp_path, "b.json", _panelled(9.9, _tracks(0.3), _panels(1.35, 1.22, 2.40)))
    tool.main([a, b, "--suite", "bench", "--panel", panel])
    out = capsys.readouterr().out
    assert f"## Panel `{panel}` — held-out scope" in out
    assert f"| `crps` | {expect:.5f} |" in out
    assert "9.90000" not in out                      # never the whole-pass macro


def test_a_panel_report_carries_both_arms_with_each_arms_own_crps_companions(
        tool, tmp_path, capsys) -> None:
    """`EVAL.md` rule 4 and the board's companion rule. A count `crps` travels with its
    oracle-scaled/scale-error split; a pval `crps` travels with `pit_ks` and `coverage_95`. The
    pval arm sits 0.5 below the count arm in the fixture, so this also proves the two tables are
    not the same table printed twice."""
    a = _write(tmp_path, "a.json", _panelled(9.9, _tracks(), _panels(1.20, 1.10, 2.10)))
    b = _write(tmp_path, "b.json", _panelled(9.9, _tracks(0.3), _panels(1.35, 1.22, 2.40)))
    tool.main([a, b, "--panel", "B"])
    out = capsys.readouterr().out
    assert "### `B` · `count` arm" in out and "### `B` · `pval` arm" in out
    for key in ("crps", "crps_oracle_scaled", "scale_error", "pit_ks", "coverage_95"):
        assert f"| `{key}` |" in out, key
    assert "| `crps` | 2.10000 |" in out             # count arm
    assert "| `crps` | 1.60000 |" in out             # pval arm, 0.5 below


def test_a_panel_crps_whose_companions_are_missing_is_withheld_rather_than_quoted_alone(
        tool, tmp_path, capsys) -> None:
    """All three or none of the three (`AGENTS.md` §7.2, `EVAL.md` rule 4). A raw CRPS on its own
    cannot say whether the seed moved the model's shape or only its scale, and a floor tool that
    prints it anyway hands a reader the exact number §7.2 forbids."""
    a = _write(tmp_path, "a.json",
               _panelled(9.9, _tracks(), _panels(1.20, 1.10, 2.10, drop=("scale_error",))))
    b = _write(tmp_path, "b.json",
               _panelled(9.9, _tracks(0.3), _panels(1.35, 1.22, 2.40, drop=("scale_error",))))
    tool.main([a, b, "--panel", "B"])
    out = capsys.readouterr().out
    assert "WITHHELD" in out
    assert "| `crps` | 2.10000 |" not in out
    assert "| `crps_oracle_scaled` |" not in out
    assert "| `gwspear` |" in out                    # the rest of the arm still reports


def test_the_per_track_block_is_cut_to_the_panel_too(tool, tmp_path, capsys) -> None:
    """Leaving the per-track block over every scored track while the headline shows one panel is
    the worse of the two available errors: the reader weighs a `B` macro against a track floor that
    is mostly `V_` tracks. Membership comes off the track key's TARGET cell."""
    a = _write(tmp_path, "a.json", _panelled(9.9, _tracks(), _panels(1.20, 1.10, 2.10)))
    b = _write(tmp_path, "b.json", _panelled(9.9, _tracks(0.3), _panels(1.35, 1.22, 2.40)))

    tool.main([a, b, "--panel", "B"])
    out = capsys.readouterr().out.replace("\\|", "|")
    assert "3 `impute` tracks" not in out
    assert "2 `impute` tracks common to every run on panel `B`" in out
    assert "B_bb" in out and "V_aa" not in out

    tool.main([a, b, "--panel", "V_breadth"])
    out = capsys.readouterr().out.replace("\\|", "|")
    assert "3 `impute` tracks common to every run on panel `V_breadth`" in out
    assert f"T_x0|V_aa|{UNMATCHED_ASSAY}" in out and "B_bb" not in out

    # the middle panel: the same V_ tracks, minus the assay B_ never posed. Measured off the B_
    # rows in front of it -- a listed assay set would go stale the first time the panel moved.
    tool.main([a, b, "--panel", "V_matched"])
    out = capsys.readouterr().out.replace("\\|", "|")
    assert "2 `impute` tracks common to every run on panel `V_matched`" in out
    assert UNMATCHED_ASSAY not in out and "T_x0|V_aa|H3K4me3" in out


@pytest.mark.parametrize("panel", [None, "B"])
def test_the_genome_wide_scope_is_read_from_the_genome_wide_block(tool, tmp_path, capsys,
                                                                  panel) -> None:
    """`per_track`/`macro`/`panels` at the top of a score json are the HELD-OUT scope — the ranked
    one — and the same three over every scored chromosome live under `genome_wide`. Reading a
    genome-wide number out of the held-out block is a different population, silently."""
    a = _write(tmp_path, "a.json",
               _panelled(1.0, _tracks(), _panels(1.20, 1.10, 2.10), gw=_panels(3.20, 3.10, 4.10)))
    b = _write(tmp_path, "b.json",
               _panelled(1.2, _tracks(0.3), _panels(1.35, 1.22, 2.40),
                         gw=_panels(3.45, 3.32, 4.50)))
    tool.main([a, b] + ([] if panel is None else ["--panel", panel]) + ["--scope", "genome-wide"])
    out = capsys.readouterr().out
    assert "the genome-wide scope" in out
    if panel is None:
        assert "`genome_wide.macro[arm]`" in out
        assert "| macro count CRPS | 1.50000 | 1.70000 |" in out       # macro_crps + 0.5
    else:
        assert "`genome_wide.panels[arm][B]`" in out
        assert "| `crps` | 4.10000 |" in out and "2.10000" not in out
    assert "| T_x1\\|B_bb\\|ATAC-seq | 2.50000 |" in out               # the +0.5 genome-wide rows


def test_a_json_with_no_panels_block_is_refused_and_names_the_file(tool, tmp_path) -> None:
    """An empty table would read as "the seeds agreed", which is the one wrong answer a floor tool
    can give. Refuse, name the file, and say what would fix it."""
    ok = _panelled(9.9, _tracks(), _panels(1.20, 1.10, 2.10))
    old = _panelled(9.9, _tracks(0.3), _panels(1.35, 1.22, 2.40))
    old.pop("panels")
    a = _write(tmp_path, "new.json", ok)
    b = _write(tmp_path, "old.json", old)
    with pytest.raises(SystemExit) as e:
        tool.main([a, b, "--panel", "B"])
    assert e.value.code != 0
    assert "old" in str(e.value) and "panels" in str(e.value)


def test_an_eval_era_json_cannot_be_read_by_panel_at_all(tool, tmp_path) -> None:
    """The three panels postdate `candi.eval`. Saying so beats reporting that its `panels` block is
    missing, which invites a reader to go looking for one."""
    a = _write(tmp_path, "a.json", {"M1": {"imp_macro_crps": 1.0}})
    with pytest.raises(SystemExit, match="candi.bench"):
        tool.main([a, a, "--panel", "V_breadth"])


def test_a_scope_the_run_never_computed_is_refused(tool, tmp_path) -> None:
    """A run scored on exactly the chromosomes it holds out has ONE scope, and under §4's blanking
    rule a method fit at every position is run that way on purpose."""
    a = _write(tmp_path, "a.json", _panelled(9.9, _tracks(), _panels(1.2, 1.1, 2.1)))
    with pytest.raises(SystemExit, match="genome_wide"):
        tool.main([a, a, "--scope", "genome-wide"])


def test_a_panel_that_scored_nothing_says_so_instead_of_printing_an_empty_table(
        tool, tmp_path, capsys) -> None:
    empty = _panels(1.2, 1.1, 2.1)
    for arm in ("count", "pval"):
        empty[arm]["B"] = {"n_experiments": 0, "assays": [], "ranked": True}
    a = _write(tmp_path, "a.json", _panelled(9.9, _tracks(), empty))
    tool.main([a, a, "--panel", "B"])
    out = capsys.readouterr().out
    assert "scored no experiments in any run" in out


def test_two_seeds_are_named_a_paired_magnitude_in_the_header_line(tool, tmp_path, capsys) -> None:
    """Two seeds are ONE paired difference. A reader who takes it for an interval will read a
    between-arm gap that does not clear it as significant, which is the failure `AGENTS.md` §7.2
    exists to prevent — and §7.2's own seed figures were measured on a different instrument and a
    different panel, so they are NOT the comparison to make here."""
    a = _write(tmp_path, "a.json", _panelled(9.9, _tracks(), _panels(1.20, 1.10, 2.10)))
    b = _write(tmp_path, "b.json", _panelled(9.9, _tracks(0.3), _panels(1.35, 1.22, 2.40)))
    tool.main([a, b, "--panel", "B"])
    out = capsys.readouterr().out
    assert "2 seeds: a paired |Δ|, not an interval" in out
    assert "paired \\|Δ\\|" in out                    # the spread column, escaped for the table
    assert "| sd |" not in out                        # two draws have no sd worth printing
    assert "confidence interval" not in out.lower()


def test_three_seeds_add_an_sd_beside_the_range(tool, tmp_path, capsys) -> None:
    """Still not an interval, and the tool says so in the same line it prints the sd."""
    runs = [_write(tmp_path, f"{i}.json",
                   _panelled(9.9, _tracks(i / 10), _panels(1.20 + i / 10, 1.10, 2.10)))
            for i in range(3)]
    tool.main(runs + ["--panel", "V_breadth"])
    out = capsys.readouterr().out
    assert "3 seeds: a range and an sd over 3 draws, not an interval" in out
    assert "| range | sd |" in out
    assert "| `crps` | 1.20000 | 1.30000 | 1.40000 | **0.20000** | 0.10000 |" in out
    assert "confidence interval" not in out.lower()


def test_a_panel_nobody_defined_is_refused_by_argparse(tool, tmp_path) -> None:
    a = _write(tmp_path, "a.json", _panelled(9.9, _tracks(), _panels(1.2, 1.1, 2.1)))
    with pytest.raises(SystemExit) as e:
        tool.main([a, a, "--panel", "V_"])
    assert e.value.code != 0


# ---------------------------------------------------------------------------
# what --panel must NOT have changed
# ---------------------------------------------------------------------------

#: The whole report the tool printed before `--panel` existed, frozen. Existing callers — the memo
#: chunk, `tools/leaderboard.py`'s caveat text, anyone with a shell history — read this document,
#: and a new option that quietly reflows it breaks every one of them. Change this string only to
#: change the default report ON PURPOSE.
GOLDEN_DEFAULT = r"""
# What a seed change moves — `candi.bench`

2 runs of one recipe at different seeds: `a`, `b`

Two seeds give ONE paired difference, not a distribution. That is the same standing as `AGENTS.md` §7.2's own seed numbers, which is what makes the two comparable — and it is a magnitude, never an interval.

## Headline scalars

| key | a | b | spread |
|---|---|---|---|
| macro count CRPS | 1.50000 | 1.25000 | **0.25000** |
| macro count gwspear | 0.50000 | 0.50000 | **0.00000** |

## Per track — CRPS across 2 `impute` tracks common to every run

The macro is a mean of these. A macro that holds still while its tracks swing is a macro whose stability is an averaging artefact, and the track spread is what an arm-vs-arm claim has to clear.

The last two columns are the split `AGENTS.md` §7.2 requires beside any raw CRPS, and here they do real work: a track whose raw spread is large while its oracle-scaled spread is small moved its SCALE, not its ranking, and the two are not the same failure.

| track | a | b | spread | oracle-scaled sp. | scale-error sp. |
|---|---|---|---|---|---|
| T\|B\|X | 1.00000 | 1.40000 | 0.40000 | 0.40000 | 0.00000 |
| T\|B\|Y | 2.00000 | 2.30000 | 0.30000 | 0.30000 | 0.00000 |

**Track spread: median 0.35000, max 0.40000 on `T|B|X`.** The macro spread above is a mean of this column, so it is smaller by construction; a claim about a single track has to clear the track number, not the macro one.

"""[1:]


def test_without_a_panel_the_report_is_byte_identical_to_the_one_it_always_printed(
        tool, tmp_path, capsys) -> None:
    a = _write(tmp_path, "a.json", _bench(1.50, {"T|B|X": 1.0, "T|B|Y": 2.0}))
    b = _write(tmp_path, "b.json", _bench(1.25, {"T|B|X": 1.4, "T|B|Y": 2.3}))
    tool.main([a, b])
    assert capsys.readouterr().out == GOLDEN_DEFAULT


def test_a_json_that_carries_panels_still_reads_the_macro_when_no_panel_is_asked_for(
        tool, tmp_path, capsys) -> None:
    """The default address does not move because a richer file arrived."""
    a = _write(tmp_path, "a.json", _panelled(9.9, _tracks(), _panels(1.20, 1.10, 2.10)))
    b = _write(tmp_path, "b.json", _panelled(9.5, _tracks(0.3), _panels(1.35, 1.22, 2.40)))
    tool.main([a, b])
    out = capsys.readouterr().out
    assert "| macro count CRPS | 9.90000 | 9.50000 | **0.40000** |" in out
    assert "## Panel" not in out
    assert "5 `impute` tracks common to every run\n" in out      # every track, no panel cut
