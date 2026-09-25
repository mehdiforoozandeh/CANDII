"""t118 ladder — figures and report (`tools/t118/ladder/{figures,report,synth_results}.py`).

Runs in the candii env, which has no matplotlib: the report and the schema check run in-process;
the nine PNGs are drawn by a subprocess under a python that has matplotlib (`T118_MPL_PYTHON`,
default `/Users/mforooz/miniforge3/bin/python`), skipped when none is available.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "t118"))

from ladder import figures, report, synth_results  # noqa: E402

LADDER = ROOT / "tools" / "t118" / "ladder"
MPL_PYTHON = os.environ.get("T118_MPL_PYTHON", "/Users/mforooz/miniforge3/bin/python")
SECTIONS = ("Summary", "Checks", "Law test", "Depth law", "Shuffle and swap", "Figures", "Runs",
            "Choices")


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    out = tmp_path_factory.mktemp("t118_synth_results")
    res = synth_results.make_results(out, seed=0)
    return out, res


@pytest.fixture(scope="module")
def report_b(synth):
    out, _ = synth
    text = report.build_report(out, "B")
    return text


def _section(text, name):
    m = re.search(rf"^## {re.escape(name)}\n(.*?)(?=^## |\Z)", text, re.S | re.M)
    assert m, f"section {name!r} missing"
    return m.group(1)


# ---------------------------------------------------------------------------------------------
# synthetic results and the schema
# ---------------------------------------------------------------------------------------------


def test_figures_module_needs_no_matplotlib_at_import():
    src = (LADDER / "figures.py").read_text()
    top = src.split("def _mpl", 1)[0]
    assert "import matplotlib" not in top
    for banned in ("torch", "scipy", "candi", "ladder."):
        assert not re.search(rf"^\s*(import|from) {re.escape(banned)}", src, re.M), banned


def test_synth_covers_every_rung_version_space_and_seed(synth):
    _, res = synth
    assert figures.check_schema(res) == []
    pc = res["per_class"]
    assert {r["rung"] for r in pc} == set(figures.RUNGS)
    assert {r["g_version"] for r in pc} == set(figures.G_VERSIONS)
    assert {r["space"] for r in pc} == set(figures.SPACES)
    assert all(len(r["per_seed"]) == 3 for r in pc)
    assert len(res["runs_present"]) + len(res["runs_missing"]) == 576
    for r in figures.RUNGS:
        assert (synth[0] / f"checks_{r}.json").is_file()
    assert (synth[0] / "checks_main.json").is_file()
    # the seed wobble follows the pinned rule: max pairwise |delta| over the seeds
    row = next(r for r in pc if None not in r["per_seed"])
    assert row["seed_wobble"] == pytest.approx(max(row["per_seed"]) - min(row["per_seed"]))


def test_check_schema_cli_passes_on_synth(synth):
    out, _ = synth
    p = subprocess.run([sys.executable, str(LADDER / "figures.py"), "--check-schema", str(out)],
                       capture_output=True, text=True, timeout=200)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "SCHEMA OK" in p.stdout


def test_check_schema_catches_broken_results(synth):
    _, res = synth
    bad = {k: v for k, v in res.items() if k != "law_grid"}
    bad["per_class"] = [dict(res["per_class"][0], per_seed=[1.0, 2.0])] + res["per_class"][1:3]
    bad["checks"] = [dict(res["checks"][0], check="made_up")]
    bad["rung_choice"] = {"per_track-counts": {"chosen": "A"}}
    errs = "\n".join(figures.check_schema(bad))
    assert "missing key 'law_grid'" in errs
    assert "per_seed length 2" in errs
    assert "check='made_up'" in errs
    assert "rung_choice" in errs


def test_load_checks_accepts_a_list_or_a_dict(tmp_path, synth):
    out, res = synth
    rows = [c for c in res["checks"] if c["rung"] == "C"]
    (tmp_path / "checks_C.json").write_text(json.dumps(rows))
    assert len(figures.load_checks(tmp_path, "C")) == len(rows)
    assert len(figures.load_checks(out, "C")) == len(rows)


# ---------------------------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------------------------


def test_report_has_every_section_in_order(report_b):
    heads = re.findall(r"^## (.+)$", report_b, re.M)
    assert tuple(heads) == SECTIONS


def test_report_has_one_line_per_check_value_bar_and_reading(synth, report_b):
    out, _ = synth
    checks = figures.load_checks(out, "B")
    assert checks
    body = _section(report_b, "Checks")
    rows = [ln for ln in body.splitlines()
            if ln.startswith("| ") and ln.rstrip().endswith(("| met |", "| unmet |"))]
    assert len(rows) == len(checks)
    assert sum(r.endswith("| met |") for r in rows) == sum(c["met"] for c in checks)
    assert "not ticked" in body
    for c in checks[:20]:
        assert any(figures.fmt(c["value"]) in r and figures.fmt(c["bar"]) in r for r in rows)


def test_report_quotes_seed_wobble_and_crps_split(report_b):
    body = _section(report_b, "Summary")
    for ln in body.splitlines():
        if not ln.startswith("| ") or ln.startswith("| class") or ln.startswith("| g "):
            continue
        cells = [c.strip() for c in ln.strip("|").split("|")]
        numeric = [c for c in cells if re.match(r"^-?\d", c)]
        for c in numeric:
            assert "seed wobble" in c or "reference, no seed" in c, c
        if "CRPS, all bins" in cells:
            for c in numeric:
                assert "split" in c, c
        if "CRPS, top 1 % bins" in cells:
            for c in numeric:
                assert "split" in c, c           # said explicitly: not computed on this subset


def test_report_names_hypotheses_by_content(report_b):
    assert not re.search(r"\b[hqt]\d{1,3}\b", report_b)
    assert "design B, the per-bin monotone curve" in report_b
    assert not re.search(r"\b(supported|refuted|verdict:)", report_b, re.I) or \
        "not a verdict" in report_b


def test_report_embeds_the_nine_figures_and_reads_runs_and_config(synth, report_b):
    for name in figures.FIG_NAMES:
        assert f"](figures/{name}.png)" in report_b
    runs = _section(report_b, "Runs")
    assert "train: train" in runs and "score: score_trained" in runs
    choices = _section(report_b, "Choices")
    assert "| steps | 4000 |" in choices and "| learning rate | 0.001 |" in choices
    assert "| window (bins) | 2048 |" in choices and "| batch | 32 |" in choices


def test_report_cli_writes_report_md(synth):
    out, _ = synth
    p = subprocess.run([sys.executable, str(LADDER / "report.py"), str(out), "D"],
                       capture_output=True, text=True, timeout=200)
    assert p.returncode == 0, p.stderr
    text = (out / "D" / "report.md").read_text()
    assert "Missing: `D_C07M29_pval_ids_s2`, `D_all_pval_ids_s2`." in text
    assert "2 of 3 seeds" in text        # a missing seed is shown beside the number


def test_report_survives_missing_optional_inputs(tmp_path, synth):
    """No refs, no checks file, no run directories: the report still writes, saying so."""
    out, res = synth
    lite = dict(res, refs={}, runs_dir=str(tmp_path / "nowhere"), figdata={})
    (tmp_path / "results.json").write_text(json.dumps(lite))
    text = report.build_report(tmp_path, "A")
    assert "No config.json found" in text and "No timing files found" in text
    assert "| n/a |" in text


# ---------------------------------------------------------------------------------------------
# the figures (a python with matplotlib, in a subprocess)
# ---------------------------------------------------------------------------------------------


def _mpl_python():
    exe = shutil.which(MPL_PYTHON) or (MPL_PYTHON if Path(MPL_PYTHON).is_file() else None)
    if exe is None:
        pytest.skip(f"no python with matplotlib at {MPL_PYTHON}")
    p = subprocess.run([exe, "-c", "import matplotlib, numpy"], capture_output=True)
    if p.returncode != 0:
        pytest.skip(f"{MPL_PYTHON} lacks matplotlib or numpy")
    return exe


def test_figures_write_the_nine_pngs(synth):
    exe = _mpl_python()
    out, _ = synth
    p = subprocess.run([exe, str(LADDER / "figures.py"), str(out), "B", "--refs-qm",
                        str(out / "qm_curves.json")], capture_output=True, text=True,
                       timeout=230)
    assert p.returncode == 0, p.stderr[-3000:]
    pngs = sorted((out / "B" / "figures").glob("*.png"))
    assert [q.stem for q in pngs] == sorted(figures.FIG_NAMES)
    assert all(q.stat().st_size > 10_000 for q in pngs)
