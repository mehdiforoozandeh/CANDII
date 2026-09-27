"""t118 row 2 — figures, per-rung report and the 2 x 4 grid report accept the row-2 rungs A2-D2.

The synthetic results hold both rows: `synth_results.make_results` writes row 1 (A-D) and this file
lifts every row-1 row to its row-2 design (A -> A2, ...) with CRPS scaled, then adds the optional
keys the aggregator writes for row 2 (`beatsrow1` checks, `rung_choice_row2`, `grid`). Runs in the
candii env (no matplotlib): the PNGs are drawn by a subprocess under `T118_MPL_PYTHON` (default
`/Users/mforooz/miniforge3/bin/python`), skipped when that python is absent.
"""
from __future__ import annotations

import copy
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "t118"))

from ladder import figures, report, synth_results  # noqa: E402

LADDER = ROOT / "tools" / "t118" / "ladder"
MPL_PYTHON = os.environ.get("T118_MPL_PYTHON", "/Users/mforooz/miniforge3/bin/python")
ROW2_OF = {"A": "A2", "B": "B2", "C": "C2", "D": "D2"}
#: row-2 CRPS multiplier over the same row-1 design (invented)
LIFT = {"A": 0.95, "B": 0.97, "C": 0.99, "D": 1.002}
EXPLODE = ("B2", "per_track", "pval", "DNase", 3)       # rung, g_version, space, class, n records


def _crps(metric):
    return metric.startswith("crps_")


def _lift_value(metric, v, rung):
    return v * LIFT[rung] if (v is not None and _crps(metric)) else v


def _row2_describe(space, rng):
    x = np.asarray(synth_results.KNOTS_X[space])
    loc = x + 0.1 * np.tanh(x) + rng.normal(0, 0.01, len(x))
    disp = np.log(5.0) + 0.2 * np.exp(-np.abs(x)) + rng.normal(0, 0.01, len(x))
    return {"n_levels": len(x), "levels_x": [round(v, 4) for v in x],
            "response": {"loc": [round(v, 4) for v in loc], "disp": [round(v, 4) for v in disp]}}


def _wobble(v):
    v = [x for x in v if x is not None]
    return float(max(v) - min(v)) if len(v) >= 2 else 0.0


def _mean(v):
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


def add_row2(out: Path, res: dict) -> dict:
    """Lift every row-1 row of `res` to row 2 and add the row-2 keys; rewrite `out`."""
    rng = np.random.default_rng(11)
    res = copy.deepcopy(res)
    for sec in ("per_class", "per_track", "per_pair", "knob_gain", "law_grid", "depth_law"):
        new = []
        for r in res[sec]:
            if r["rung"] not in figures.ROW1:
                continue
            q = dict(r, rung=ROW2_OF[r["rung"]])
            if sec == "per_class":
                q["per_seed"] = [_lift_value(r["metric"], v, r["rung"]) for v in r["per_seed"]]
                q["mean"], q["seed_wobble"] = _mean(q["per_seed"]), _wobble(q["per_seed"])
            elif sec == "per_track":
                q["value"] = _lift_value(r["metric"], r["value"], r["rung"])
            elif sec == "per_pair":
                for m in ("crps_all", "crps_nonzero", "crps_top1", "crps_oracle_scaled_all",
                          "scale_error_all"):
                    if q.get(m) is not None:
                        q[m] = q[m] * LIFT[r["rung"]]
                if "describe" in q:
                    q["describe"] = _row2_describe(q["space"], rng)
            elif sec == "law_grid":
                for m in ("d_real", "d_nocov", "d_ids"):
                    q[m] = q[m] * LIFT[r["rung"]]
                q["gain_nocov"], q["gain_ids"] = q["d_nocov"] - q["d_real"], q["d_ids"] - q["d_real"]
            new.append(q)
        res[sec] += new
    # a few exploding (pair, seed) records in one row-2 cell
    rung, gv, space, cls, n = EXPLODE
    hits = [r for r in res["per_pair"] if r["rung"] == rung and r["g_version"] == gv
            and r["space"] == space and r["mark_class"] == cls and r["model"] == "real"
            and r["kind"] == "trained" and r["eval"] == "score"][:n]
    for r in hits:
        r["crps_all"] = 25.0 + len(hits)
    # checks: row-1 checks lifted, then beatsrow1
    idx = figures.class_index(res)
    checks = []
    for c in res["checks"]:
        if c["rung"] not in figures.ROW1:
            continue
        q = dict(c, rung=ROW2_OF[c["rung"]], components=dict(c["components"]))
        if c["check"] == "beatsbelow":
            q["components"]["rung_below"] = ROW2_OF[c["components"]["rung_below"]]
        checks.append(q)
        if c["check"] == "beatstwin":
            a = idx[(c["rung"], c["g_version"], c["space"], "real", "trained", c["mark_class"],
                     c["metric"], "all")]
            b = idx[(q["rung"], c["g_version"], c["space"], "real", "trained", c["mark_class"],
                     c["metric"], "all")]
            w = max(a["seed_wobble"], b["seed_wobble"])
            v = a["mean"] - b["mean"]
            checks.append({"rung": q["rung"], "g_version": c["g_version"], "space": c["space"],
                           "mark_class": c["mark_class"], "metric": c["metric"],
                           "check": "beatsrow1", "value": v, "bar": 2 * w, "met": v > 2 * w,
                           "seed_wobble": w,
                           "components": {"rung_row1": c["rung"], "d_row1": a["mean"],
                                          "d_row2": b["mean"], "wobble_row1": a["seed_wobble"],
                                          "wobble_row2": b["seed_wobble"]}})
    res["checks"] += checks
    # the row-2 rung choice
    res["rung_choice_row2"] = {}
    for key, v in res["rung_choice"].items():
        val = {ROW2_OF[r]: x * LIFT[r] for r, x in v["val_by_rung"].items()}
        wob = {ROW2_OF[r]: x for r, x in v["wobble_by_rung"].items()}
        best = min(val, key=val.get)
        chosen = next(r for r in figures.ROW2 if val[r] <= val[best] + wob[best])
        res["rung_choice_row2"][key] = {"chosen": chosen, "val_by_rung": val,
                                        "wobble_by_rung": wob}
    # the grid rows: real g, trained pairs, variant all
    grid = []
    for gv_ in figures.G_VERSIONS:
        for sp in figures.SPACES:
            for cl in figures.CLASSES:
                for metric in ("crps_all", "crps_top1"):
                    for col in figures.ROW1:
                        for row, rg in ((1, col), (2, ROW2_OF[col])):
                            pc = idx.get((rg, gv_, sp, "real", "trained", cl, metric, "all"))
                            if pc is None:
                                continue
                            n_exp = sum(1 for r in res["per_pair"] if r["rung"] == rg
                                        and r["g_version"] == gv_ and r["space"] == sp
                                        and r["mark_class"] == cl and r["model"] == "real"
                                        and r["kind"] == "trained" and r["eval"] == "score"
                                        and r["crps_all"] > 20)
                            grid.append({"g_version": gv_, "space": sp, "mark_class": cl,
                                         "metric": metric, "column": col, "row": row,
                                         "rung": rg, "mean": pc["mean"],
                                         "seed_wobble": pc["seed_wobble"],
                                         "n_pairs_crps_gt_20": n_exp})
    res["grid"] = grid
    # runs: the row-2 names, their run directories and figdata
    runs_dir = Path(res["runs_dir"])
    extra = []
    for name in list(res["runs_present"]):
        r1, rest = name.split("_", 1)
        name2 = f"{ROW2_OF[r1]}_{rest}"
        shutil.copytree(runs_dir / name, runs_dir / name2, dirs_exist_ok=True)
        extra.append(name2)
        if name in res["figdata"]:
            res["figdata"][name2] = str(runs_dir / name2 / "figdata.npz")
    res["runs_present"] += extra
    res["runs_missing"] += [f"{ROW2_OF[n.split('_', 1)[0]]}_{n.split('_', 1)[1]}"
                            for n in list(res["runs_missing"])]
    with open(out / "results.json", "w") as fh:
        json.dump(res, fh)
    for rg in figures.ROW2:
        with open(out / f"checks_{rg}.json", "w") as fh:
            json.dump({"rung": rg, "checks": [c for c in res["checks"] if c["rung"] == rg]}, fh)
    return res


@pytest.fixture(scope="module")
def both(tmp_path_factory):
    out = tmp_path_factory.mktemp("t118_row2_results")
    res1 = synth_results.make_results(out, seed=0)
    res = add_row2(out, res1)
    return out, res, res1


@pytest.fixture(scope="module")
def grid_md(both):
    return report.build_grid(both[0])


def _section(text, name):
    m = re.search(rf"^## {re.escape(name)}\n(.*?)(?=^## |\Z)", text, re.S | re.M)
    assert m, f"section {name!r} missing"
    return m.group(1)


# ---------------------------------------------------------------------------------------------
# the labels and the schema
# ---------------------------------------------------------------------------------------------


def test_rungs_rows_and_labels():
    assert figures.RUNGS == ("A", "B", "C", "D", "A2", "B2", "C2", "D2")
    assert figures.ROW1 == figures.RUNGS[:4] and figures.ROW2 == figures.RUNGS[4:]
    for r in figures.RUNGS:
        assert r in figures.RUNG_NAME and r in figures.RUNG_COLOUR
    assert figures.RUNG_NAME["A2"] == "design A2, the affine map chosen per bin"
    assert len(set(figures.RUNG_COLOUR.values())) == len(figures.RUNGS)
    assert "beatsrow1" in figures.CHECK_ORDER and figures.CHECK_RULE["beatsrow1"] == ">"
    # row-1 checks keep their order
    assert figures.CHECK_ORDER[:7] == ("beatstwin", "lawtest_nocov", "lawtest_ids", "depthlaw",
                                       "beatsbelow", "shuffle", "swap")


def test_both_rows_pass_the_schema_and_row1_alone_still_does(both):
    out, res, res1 = both
    assert figures.check_schema(res) == []
    assert figures.check_schema(res1) == []
    assert "grid" not in res1 and "rung_choice_row2" not in res1
    assert {r["rung"] for r in res["per_class"]} == set(figures.RUNGS)
    assert {c["check"] for c in res["checks"] if c["rung"] in figures.ROW2} == \
        {"beatstwin", "lawtest_nocov", "lawtest_ids", "depthlaw", "beatsbelow", "shuffle",
         "swap", "beatsrow1"}
    p = subprocess.run([sys.executable, str(LADDER / "figures.py"), "--check-schema", str(out)],
                       capture_output=True, text=True, timeout=300)
    assert p.returncode == 0 and "SCHEMA OK" in p.stdout, p.stdout + p.stderr


def test_schema_catches_broken_row2_keys(both):
    _, res, _ = both
    bad = {k: v for k, v in res.items() if k not in ("per_pair", "per_track")}
    g0 = res["grid"][0]
    bad["grid"] = [dict(g0, column="E"), dict(g0, row=3), dict(g0, rung="B2"),
                   {k: v for k, v in g0.items() if k != "n_pairs_crps_gt_20"}]
    b1 = next(c for c in res["checks"] if c["check"] == "beatsrow1")
    bad["checks"] = [dict(b1, components={"rung_row1": "A"})]
    bad["rung_choice_row2"] = {"per_track-counts": {"chosen": "A2"}}
    errs = "\n".join(figures.check_schema(bad))
    assert "grid[0]" in errs and "column='E'" in errs
    assert "row=3" in errs
    assert "is not the cell" in errs
    assert "n_pairs_crps_gt_20" in errs
    assert "components.d_row1" in errs and "components.wobble_row2" in errs
    assert "rung_choice_row2: key 'per_track-counts'" in errs
    assert "rung_choice_row2['per_track-counts']: missing 'val_by_rung'" in errs
    bad["grid"] = "not rows"
    assert "grid: not a list" in figures.check_schema(bad)


# ---------------------------------------------------------------------------------------------
# the per-rung report
# ---------------------------------------------------------------------------------------------


def test_row2_report_adds_the_row1_section_after_checks(both):
    out, res, _ = both
    text = report.build_report(out, "A2")
    heads = re.findall(r"^## (.+)$", text, re.M)
    assert tuple(heads) == ("Summary", "Checks", "Against the same design in row 1", "Law test",
                            "Depth law", "Shuffle and swap", "Figures", "Runs", "Choices")
    assert text.startswith("# Rung A2: design A2, the affine map chosen per bin")
    assert "source value x" in text
    assert not re.search(r"\b[hqt]\d{1,3}\b", text)
    body = _section(text, "Against the same design in row 1")
    rows = [ln for ln in body.splitlines() if ln.startswith("| one g")]
    assert len(rows) == len(figures.G_VERSIONS) * len(figures.SPACES) * len(figures.CLASSES) * 2
    b1 = {(c["g_version"], c["space"], c["mark_class"], c["metric"]): c
          for c in figures.load_checks(out, "A2") if c["check"] == "beatsrow1"}
    assert len(b1) == len(rows)
    n_met = sum(r.split("|")[-2].strip().startswith("met") for r in rows)
    assert n_met == sum(c["met"] for c in b1.values())
    c = b1[("per_track", "counts", "narrow", "crps_all")]
    line = next(r for r in rows if "| one g per track | counts | narrow | CRPS, all bins |" in r)
    assert figures.fmt(c["components"]["d_row1"]) in line
    assert figures.fmt(c["components"]["d_row2"]) in line
    assert figures.fmt(c["components"]["d_row1"] - c["components"]["d_row2"]) in line
    assert "seed wobble" in line and "split: oracle-scaled" in line
    # the checks table carries beatsrow1 and the row-2 choice sits beside the main claim's
    checks = _section(text, "Checks")
    assert "beats the same design in row 1 (design A, the per-bin affine map)" in checks
    assert "### The row-2 choice, reported beside it" in checks
    runs = _section(text, "Runs")
    assert "Missing: `A2_" not in runs and "0 missing for rung A2" in runs


def test_row1_report_is_unchanged_by_row2_results(both, tmp_path):
    """A row-1 report written from both rows equals the one written from row 1 alone."""
    out, _, res1 = both
    both_text = report.build_report(out, "B")
    solo = tmp_path / "solo"
    solo.mkdir()
    (solo / "results.json").write_text(json.dumps(res1))
    shutil.copy(out / "checks_B.json", solo / "checks_B.json")
    assert both_text == report.build_report(solo, "B")
    assert "Against the same design in row 1" not in both_text


def test_report_cli_accepts_a_row2_rung(both):
    out, _, _ = both
    p = subprocess.run([sys.executable, str(LADDER / "report.py"), str(out), "D2"],
                       capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr
    text = (out / "D2" / "report.md").read_text()
    assert "Missing: `D2_C07M29_pval_ids_s2`, `D2_all_pval_ids_s2`." in text


# ---------------------------------------------------------------------------------------------
# the grid report
# ---------------------------------------------------------------------------------------------


def test_grid_has_one_block_per_version_space_class_and_metric(grid_md):
    for gv in figures.G_VERSIONS:
        for sp in figures.SPACES:
            assert f"## {figures.GV_NAME[gv]}, {figures.SPACE_NAME[sp]}" in grid_md
    assert grid_md.count("**CRPS, all bins**") == 12
    assert grid_md.count("**CRPS, top 1 % bins**") == 12
    assert grid_md.count("| row 1: g reads (C, C') |") == 36      # 2 metrics + counts, x 12
    assert grid_md.count("| row 2: g reads (x, C, C') |") == 36
    head = [ln for ln in grid_md.splitlines() if ln.startswith("|  | A: affine")]
    assert head and all("noSolution" in h for h in head if "QuantileMatching" in h)


def test_grid_cells_quote_the_grid_rows_with_wobble_split_and_references(both, grid_md):
    _, res, _ = both
    g = next(r for r in res["grid"] if (r["g_version"], r["space"], r["mark_class"],
                                        r["metric"], r["column"], r["row"]) ==
             ("across", "pval", "broad", "crps_all", "C", 2))
    block = grid_md.split("## one g across tracks, -log10 p", 1)[1].split("### broad", 1)[1]
    table = block.split("**CRPS, all bins**", 1)[1].split("**CRPS, top 1 % bins**", 1)[0]
    row2 = next(ln for ln in table.splitlines() if ln.startswith("| row 2:"))
    cells = [c.strip() for c in row2.strip("|").split("|")]
    assert cells[3].startswith(f"{figures.fmt(g['mean'])} (seed wobble "
                               f"{figures.fmt(g['seed_wobble'])})")
    assert "split: oracle-scaled" in cells[3]
    assert "reference, no seed" in cells[5] and "reference, no seed" in cells[6]
    top = block.split("**CRPS, top 1 % bins**", 1)[1].split("(pair, seed)", 1)[0]
    assert "split not computed on this subset" in top


def test_grid_reports_the_exploding_count_only(both, grid_md):
    rung, gv, sp, cls, n = EXPLODE
    block = grid_md.split(f"## {figures.GV_NAME[gv]}, {figures.SPACE_NAME[sp]}", 1)[1]
    block = block.split(f"### {cls}", 1)[1].split("###", 1)[0]
    counts = block.split("(pair, seed) trained records with CRPS on all bins > 20", 1)[1]
    assert "(reported only)" in counts
    row2 = next(ln for ln in counts.splitlines() if ln.startswith("| row 2:"))
    assert [c.strip() for c in row2.strip("|").split("|")][1:] == ["0", str(n), "0", "0"]
    assert "reported only" in grid_md and "not a check" in grid_md
    assert not re.search(r"\b(met|unmet|passed|failed)\b", grid_md)


def test_grid_shows_both_rung_choices(both, grid_md):
    _, res, _ = both
    sec = _section(grid_md, "Which design each row chooses")
    for key, v in res["rung_choice_row2"].items():
        gv, _, sp = key.partition("|")
        line = next(ln for ln in sec.splitlines()
                    if ln.startswith(f"| {figures.GV_NAME[gv]} | {figures.SPACE_NAME[sp]} |"))
        cells = [c.strip() for c in line.strip("|").split("|")]
        assert cells[2] == res["rung_choice"][key]["chosen"]
        assert cells[4] == v["chosen"]
        assert "A2: " in cells[5] and "A: " in cells[3]


def test_grid_cli_writes_grid_md_and_survives_row1_results(both, tmp_path):
    out, _, res1 = both
    p = subprocess.run([sys.executable, str(LADDER / "report.py"), "grid", str(out)],
                       capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr
    assert (out / "grid.md").read_text().startswith("# The transformation grid")
    (tmp_path / "results.json").write_text(json.dumps(dict(res1, per_pair=[])))
    text = report.build_grid(tmp_path)
    assert "No grid rows in results.json" in text
    assert "No row-2 rung choice in results.json." in text


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


def test_figures_write_the_nine_pngs_for_a2(both):
    exe = _mpl_python()
    out, _, _ = both
    p = subprocess.run([exe, str(LADDER / "figures.py"), str(out), "A2", "--refs-qm",
                        str(out / "qm_curves.json")], capture_output=True, text=True,
                       timeout=400)
    assert p.returncode == 0, p.stderr[-3000:]
    pngs = sorted((out / "A2" / "figures").glob("*.png"))
    assert [q.stem for q in pngs] == sorted(figures.FIG_NAMES)
    assert all(q.stat().st_size > 10_000 for q in pngs)
