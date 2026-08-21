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
