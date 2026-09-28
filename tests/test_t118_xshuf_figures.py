"""t118 row 2, the shuffled-bin twin — the schema, figures, per-rung report and grid accept it.

The twin (model `xshuf`) has the same row-2 design, size, training and real covariates as the
row-2 run; only g's x input is read from a random bin of the same chromosome. The synthetic
results: `synth_results.make_results` writes row 1, `add_row2` of `test_t118_row2_figures.py`
lifts it to row 2, and this file adds twin rows for A2 and B2 (the real g's rows with CRPS
scaled: A2's twin is clearly worse, B2's twin is within the seed wobble), the `beatsxshuf`
(trained pairs) and `lawtest_xshuf` (never-trained pairs) checks and grid row 3. Without twin rows every output must stay byte-identical to what the code
wrote before the twin existed: the sha256 values below were captured from the pre-change code at
8d48e9c on the same synthetic input (paths and the creation time normalised away).
"""
from __future__ import annotations

import copy
import hashlib
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
from tests.test_t118_row2_figures import add_row2  # noqa: E402

LADDER = ROOT / "tools" / "t118" / "ladder"
MPL_PYTHON = os.environ.get("T118_MPL_PYTHON", "/Users/mforooz/miniforge3/bin/python")
TW = "xshuf"
#: the twin's CRPS multiplier over the real g of the same row-2 cell (invented)
TWIN_LIFT = {"A2": 1.25, "B2": 1.00001}
TWIN_RUNGS = tuple(TWIN_LIFT)
EXPLODE_TW = ("A2", "per_track", "counts", "narrow", 2)    # rung, g_version, space, class, n
ROW3 = "row 2, shuffled-bin twin: g reads x from a random other bin"
#: sha256 of the normalised outputs WITHOUT twin rows, from the pre-change code at 8d48e9c
GOLDEN = {
    "A": "096e009cf000e401383b7c43850f2bed44d53127fc55400421c7d2b778df6831",
    "B": "74f0bad0173260cfb858f894839a5c2603500f12f975f85414f8218718b9ce29",
    "C": "5356c3403ad4f9b6d0ed140bc0e33fa41f0e344ec23d232817bae72f8109e0f6",
    "D": "d691929dab8f18da0e2886f20b13a8f6fbbe4eb659f38cee7bf759bac62a3734",
    "A2": "cdd5ce7e2a9dc11e95c98d75184de76e03d0f00775dfd1bcd0c1ec16da900eaa",
    "B2": "37ee0ef83f25c22cea0acbdfac9da1fb73159c48d5cc6809c8012c665da36828",
    "C2": "0d26a7985514fb4b7bca56715513da9761127ae1a87a43662e329556677136e4",
    "D2": "3933c249410a7d9d520a469d991d9a580410ae9da3a8a74be3fca1a5759a6758",
    "grid": "5e306086b53fafd111a65d5581d6ff5b7bd7c32382ae620dc9f26c69f3945bc3",
}


def _scale(metric, v, rung):
    return v * TWIN_LIFT[rung] if (v is not None and metric.startswith("crps_")) else v


def _wobble(v):
    v = [x for x in v if x is not None]
    return float(max(v) - min(v)) if len(v) >= 2 else 0.0


def add_twin(out: Path, res: dict) -> dict:
    """Add the shuffled-bin twin of A2 and B2 to `res` and write it to `out`."""
    res = copy.deepcopy(res)
    for sec in ("per_class", "per_track", "per_pair"):
        new = []
        for r in res[sec]:
            if r["rung"] not in TWIN_RUNGS or r["model"] != "real":
                continue
            q = dict(r, model=TW)
            if sec == "per_class":
                q["per_seed"] = [_scale(r["metric"], v, r["rung"]) for v in r["per_seed"]]
                v = [x for x in q["per_seed"] if x is not None]
                q["mean"] = float(sum(v) / len(v)) if v else None
                q["seed_wobble"] = _wobble(q["per_seed"])
            elif sec == "per_track":
                q["value"] = _scale(r["metric"], r["value"], r["rung"])
            else:
                q.pop("describe", None)
                for m in ("crps_all", "crps_nonzero", "crps_top1", "crps_oracle_scaled_all",
                          "scale_error_all"):
                    if q.get(m) is not None:
                        q[m] = q[m] * TWIN_LIFT[r["rung"]]
            new.append(q)
        res[sec] += new
    rung, gv, space, cls, n = EXPLODE_TW
    hits = [r for r in res["per_pair"] if r["rung"] == rung and r["g_version"] == gv
            and r["space"] == space and r["mark_class"] == cls and r["model"] == TW
            and r["kind"] == "trained" and r["eval"] == "score"][:n]
    for r in hits:
        r["crps_all"] = 30.0
    idx = figures.class_index(res)
    for rg in TWIN_RUNGS:
        for gv_ in figures.G_VERSIONS:
            for sp in figures.SPACES:
                for cl in figures.CLASSES:
                    for metric in ("crps_all", "crps_top1"):
                        for check, kind in (("beatsxshuf", "trained"), ("lawtest_xshuf", "law")):
                            a = idx[(rg, gv_, sp, "real", kind, cl, metric, "all")]
                            b = idx[(rg, gv_, sp, TW, kind, cl, metric, "all")]
                            w = max(a["seed_wobble"], b["seed_wobble"])
                            v = b["mean"] - a["mean"]
                            res["checks"].append(
                                {"rung": rg, "g_version": gv_, "space": sp, "mark_class": cl,
                                 "metric": metric, "check": check, "value": v, "bar": 2 * w,
                                 "met": v > 2 * w, "seed_wobble": w,
                                 "components": {"d_real": a["mean"], "d_xshuf": b["mean"],
                                                "wobble_real": a["seed_wobble"],
                                                "wobble_xshuf": b["seed_wobble"]}})
                        b = idx[(rg, gv_, sp, TW, "trained", cl, metric, "all")]
                        n_exp = sum(1 for r in res["per_pair"] if r["rung"] == rg
                                    and r["g_version"] == gv_ and r["space"] == sp
                                    and r["mark_class"] == cl and r["model"] == TW
                                    and r["kind"] == "trained" and r["eval"] == "score"
                                    and r["crps_all"] > 20)
                        res["grid"].append({"g_version": gv_, "space": sp, "mark_class": cl,
                                            "metric": metric, "column": rg[0], "row": 3,
                                            "rung": rg, "mean": b["mean"],
                                            "seed_wobble": b["seed_wobble"],
                                            "n_pairs_crps_gt_20": n_exp})
    res["runs_present"] += [f"{rg}_{g}_{sp}_{TW}_s{sd}" for rg in TWIN_RUNGS
                            for g in (*synth_results.TRACKS, "all") for sp in figures.SPACES
                            for sd in figures.SEEDS]
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.json", "w") as fh:
        json.dump(res, fh)
    for rg in figures.RUNGS:
        with open(out / f"checks_{rg}.json", "w") as fh:
            json.dump({"rung": rg, "checks": [c for c in res["checks"] if c["rung"] == rg]}, fh)
    return res


@pytest.fixture(scope="module")
def dirs(tmp_path_factory):
    """(no-twin dir, its results, twin dir, its results); both share the runs dir."""
    out = tmp_path_factory.mktemp("t118_xshuf_results")
    res1 = synth_results.make_results(out, seed=0)
    res2 = add_row2(out, res1)
    tw = out / "with_twin"
    res_tw = add_twin(tw, res2)
    return out, res2, tw, res_tw


def _norm(text, res, *paths):
    for p in sorted((str(p) for p in paths), key=len, reverse=True):
        text = text.replace(p, "<OUT>")
    return text.replace(res["created_utc"], "<UTC>")


def _section(text, name):
    m = re.search(rf"^## {re.escape(name)}\n(.*?)(?=^## |\Z)", text, re.S | re.M)
    assert m, f"section {name!r} missing"
    return m.group(1)


def _cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


# ---------------------------------------------------------------------------------------------
# the labels and the schema
# ---------------------------------------------------------------------------------------------


def test_the_pinned_names():
    assert figures.MODELS == ("real", "nocov", "ids", "xshuf")
    assert figures.ENUMS["model"] == figures.MODELS
    assert figures.TWIN_XSHUF == "xshuf"
    assert figures.MODEL_NAME["xshuf"] == "shuffled-bin twin"
    assert figures.MODEL_COLOUR["xshuf"] == "#e6a100"
    assert figures.CHECK_ORDER == ("beatstwin", "lawtest_nocov", "lawtest_ids", "depthlaw",
                                   "beatsbelow", "shuffle", "swap", "beatsrow1", "beatsxshuf",
                                   "lawtest_xshuf")
    assert figures.CHECK_NAME["beatsxshuf"] == \
        "beats the shuffled-bin twin (held-out chromosomes)"
    assert figures.CHECK_SHORT["beatsxshuf"] == "beats shuffled twin"
    assert figures.CHECK_RULE["beatsxshuf"] == ">"
    assert figures.CHECK_NAME["lawtest_xshuf"] == "law test: beats the shuffled-bin twin"
    assert figures.CHECK_SHORT["lawtest_xshuf"] == "law test vs shuffled twin"
    assert figures.CHECK_RULE["lawtest_xshuf"] == ">"
    assert figures.BEATSXSHUF_COMPONENTS == ("d_real", "d_xshuf", "wobble_real", "wobble_xshuf")
    assert figures.GRID_KEYS == ("g_version", "space", "mark_class", "metric", "column", "row",
                                 "rung", "mean", "seed_wobble", "n_pairs_crps_gt_20")
    assert report.ROW_NAME[3] == ROW3
    assert report.ROW_NAME[1] == "row 1: g reads (C, C')"
    assert report.ROW_NAME[2] == "row 2: g reads (x, C, C')"


def test_twin_results_pass_the_schema(dirs):
    _, res2, tw, res_tw = dirs
    assert figures.check_schema(res_tw) == []
    assert figures.check_schema(res2) == []
    assert {r["model"] for r in res_tw["per_class"]} == set(figures.MODELS)
    assert any(g["row"] == 3 for g in res_tw["grid"])
    assert {c["check"] for c in res_tw["checks"] if c["rung"] in TWIN_RUNGS} >= \
        {"beatsxshuf", "lawtest_xshuf"}
    p = subprocess.run([sys.executable, str(LADDER / "figures.py"), "--check-schema", str(tw)],
                       capture_output=True, text=True, timeout=300)
    assert p.returncode == 0 and "SCHEMA OK" in p.stdout, p.stdout + p.stderr


def test_schema_catches_broken_twin_rows(dirs):
    _, _, _, res_tw = dirs
    bad = {k: v for k, v in res_tw.items() if k not in ("per_pair", "per_track")}
    bx = next(c for c in res_tw["checks"] if c["check"] == "beatsxshuf")
    comp = {k: v for k, v in bx["components"].items() if k != "wobble_xshuf"}
    lx = next(c for c in res_tw["checks"] if c["check"] == "lawtest_xshuf")
    lcomp = {k: v for k, v in lx["components"].items() if k != "d_real"}
    bad["checks"] = [dict(bx, components=comp), dict(bx, components=None),
                     dict(bx, check="beatsxshuf2"), dict(lx, components=lcomp), lx]
    g3 = next(g for g in res_tw["grid"] if g["row"] == 3 and g["column"] == "A")
    bad["grid"] = [dict(g3), dict(g3, rung="A"), dict(g3, rung="B2"), dict(g3, row=4),
                   dict(g3, row="3")]
    pc = next(r for r in res_tw["per_class"] if r["model"] == TW)
    bad["per_class"] = [pc, dict(pc, model="xshuffle")]
    errs = figures.check_schema(bad)
    text = "\n".join(errs)
    assert "checks[0]: missing or invalid components.wobble_xshuf" in text
    assert "components.d_real" not in text.split("checks[1]")[0]
    assert "checks[1]: missing or invalid components.d_real, components.d_xshuf, " \
           "components.wobble_real, components.wobble_xshuf" in text
    assert "check='beatsxshuf2'" in text
    assert "checks[3]: missing or invalid components.d_real" in text
    assert not any(e.startswith("checks[4]") for e in errs)       # a valid lawtest_xshuf row
    assert not any(e.startswith("grid[0]") for e in errs)          # a valid row 3
    assert not any(e.startswith("grid[4]") for e in errs)          # row "3" as a string
    assert "grid[1]: missing or invalid rung='A' is not the cell (A, row=3" in text
    assert "grid[2]: missing or invalid rung='B2' is not the cell (A, row=3" in text
    assert "grid[3]: missing or invalid row=4" in text
    assert not any(e.startswith("per_class[0]") for e in errs)
    assert "per_class[1]: missing or invalid model='xshuffle'" in text


# ---------------------------------------------------------------------------------------------
# the per-rung report
# ---------------------------------------------------------------------------------------------


def test_report_adds_the_twin_section_after_row1(dirs):
    _, _, tw, _ = dirs
    text = report.build_report(tw, "A2")
    heads = tuple(re.findall(r"^## (.+)$", text, re.M))
    assert heads == ("Summary", "Checks", "Against the same design in row 1",
                     "Against the shuffled-bin twin", "Law test", "Depth law",
                     "Shuffle and swap", "Figures", "Runs", "Choices")
    assert not re.search(r"\b[hqt]\d{1,3}\b", text)
    body = _section(text, "Against the shuffled-bin twin")
    para = body.strip().split("\n\n", 1)[0]
    for phrase in ("same design, size, training and real covariates",
                   "uniformly random bin of the same chromosome", "every training step",
                   "one fixed seeded permutation of the bins per chromosome",
                   "f still reads the true x", "drafted, not ticked"):
        assert phrase in para, phrase
    assert "### Law test against the twin" in body
    trained, law = body.split("### Law test against the twin", 1)
    rows = [ln for ln in trained.splitlines() if ln.startswith("| one g")]
    assert len(rows) == len(figures.G_VERSIONS) * len(figures.SPACES) * len(figures.CLASSES) * 2
    head = next(ln for ln in trained.splitlines() if ln.startswith("| g |"))
    assert _cells(head) == ["g", "space", "class", "metric", "real g, rung A2",
                            "shuffled-bin twin", "gain", "drafted reading"]
    bx = {(c["g_version"], c["space"], c["mark_class"], c["metric"]): c
          for c in figures.load_checks(tw, "A2") if c["check"] == "beatsxshuf"}
    assert len(bx) == len(rows)
    for ln in rows:
        cells = _cells(ln)
        key = ({v: k for k, v in figures.GV_NAME.items()}[cells[0]],
               {v: k for k, v in figures.SPACE_NAME.items()}[cells[1]], cells[2],
               {v: k for k, v in figures.METRIC_NAME.items()}[cells[3]])
        c = bx[key]
        comp = c["components"]
        assert cells[4].startswith(f"{figures.fmt(comp['d_real'])} (seed wobble "
                                   f"{figures.fmt(comp['wobble_real'])})")
        assert cells[5].startswith(f"{figures.fmt(comp['d_xshuf'])} (seed wobble "
                                   f"{figures.fmt(comp['wobble_xshuf'])})")
        assert cells[6] == figures.fmt(comp["d_xshuf"] - comp["d_real"])
        assert cells[7] == f"{'met' if c['met'] else 'unmet'} (bar {figures.fmt(c['bar'])})"
        if key[3] == "crps_all":
            assert cells[4].count("split: oracle-scaled") == 1
            assert cells[5].count("split: oracle-scaled") == 1
        else:
            assert "split not computed on this subset" in cells[4]
    # A2's twin is 25 % worse: every check met; the planted values drive the reading
    assert all(_cells(ln)[7].startswith("met") for ln in rows)
    law_rows = [ln for ln in law.splitlines() if ln.startswith("| one g")]
    assert len(law_rows) == len(rows)
    assert "law test: beats the shuffled-bin twin" in law and "nothing gates on it" in law
    lx = {(c["g_version"], c["space"], c["mark_class"], c["metric"]): c
          for c in figures.load_checks(tw, "A2") if c["check"] == "lawtest_xshuf"}
    assert len(lx) == len(law_rows)
    for ln in law_rows:
        cells = _cells(ln)
        key = ({v: k for k, v in figures.GV_NAME.items()}[cells[0]],
               {v: k for k, v in figures.SPACE_NAME.items()}[cells[1]], cells[2],
               {v: k for k, v in figures.METRIC_NAME.items()}[cells[3]])
        c = lx[key]
        comp = c["components"]
        assert cells[4].startswith(figures.fmt(comp["d_real"]))
        assert cells[5].startswith(figures.fmt(comp["d_xshuf"]))
        assert cells[6] == figures.fmt(comp["d_xshuf"] - comp["d_real"])
        assert cells[7] == f"{'met' if c['met'] else 'unmet'} (bar {figures.fmt(c['bar'])})"
        assert cells[7].startswith("met")
    # the checks table lists the twin's check last among the rung's checks
    checks = _section(text, "Checks")
    names = [_cells(ln)[0] for ln in checks.splitlines() if ln.startswith("| beats")]
    assert names[-1] == "beats the shuffled-bin twin (held-out chromosomes)"


def test_b2_twin_within_the_wobble_reads_unmet(dirs):
    _, _, tw, _ = dirs
    body = _section(report.build_report(tw, "B2"), "Against the shuffled-bin twin")
    trained = body.split("### Law test against the twin", 1)[0]
    rows = [ln for ln in trained.splitlines() if ln.startswith("| one g")]
    assert rows and all(_cells(ln)[7].startswith("unmet") for ln in rows)
    law = body.split("### Law test against the twin", 1)[1]
    law_rows = [ln for ln in law.splitlines() if ln.startswith("| one g")]
    assert law_rows and all(_cells(ln)[7].startswith("unmet") for ln in law_rows)


def test_rungs_without_twin_rows_have_no_twin_section(dirs):
    out, res2, tw, res_tw = dirs
    for rung in ("A", "C2", "D2"):
        with_tw = report.build_report(tw, rung)
        assert "Against the shuffled-bin twin" not in with_tw
        assert _norm(with_tw, res_tw, tw, out) == _norm(report.build_report(out, rung), res2, out)


def test_no_twin_outputs_are_byte_identical_to_the_pre_change_code(dirs):
    out, res2, _, _ = dirs
    got = {rung: _norm(report.build_report(out, rung), res2, out) for rung in figures.RUNGS}
    got["grid"] = _norm(report.build_grid(out), res2, out)
    for k, text in got.items():
        assert "shuffled-bin" not in text and "xshuf" not in text, k
        assert hashlib.sha256(text.encode()).hexdigest() == GOLDEN[k], k


def test_report_cli_writes_the_twin_section(dirs):
    _, _, tw, _ = dirs
    p = subprocess.run([sys.executable, str(LADDER / "report.py"), str(tw), "A2"],
                       capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr
    assert "## Against the shuffled-bin twin" in (tw / "A2" / "report.md").read_text()


# ---------------------------------------------------------------------------------------------
# the grid report
# ---------------------------------------------------------------------------------------------


def test_grid_gains_the_third_row(dirs):
    _, _, tw, res_tw = dirs
    g = report.build_grid(tw)
    assert g.count(f"| {ROW3} |") == 36          # 2 metrics + the counts table, x 12 blocks
    assert g.count("| row 1: g reads (C, C') |") == 36
    assert g.count("| row 2: g reads (x, C, C') |") == 36
    assert len(re.findall(r"^\| row 2, shuffled", g, re.M)) >= 12
    assert "Row 3 is the shuffled-bin twin of row 2." in g
    assert "uniformly random bin of the same chromosome" in g
    assert not re.search(r"\b(met|unmet|passed|failed)\b", g)
    cell = next(r for r in res_tw["grid"] if (r["g_version"], r["space"], r["mark_class"],
                                              r["metric"], r["column"], r["row"]) ==
                ("across", "pval", "broad", "crps_all", "B", 3))
    block = g.split("## one g across tracks, -log10 p", 1)[1].split("### broad", 1)[1]
    table = block.split("**CRPS, all bins**", 1)[1].split("**CRPS, top 1 % bins**", 1)[0]
    lines = [ln for ln in table.splitlines() if ln.startswith("| row")]
    assert [_cells(ln)[0] for ln in lines] == [report.ROW_NAME[r] for r in (1, 2, 3)]
    cells = _cells(lines[2])
    assert cells[2].startswith(f"{figures.fmt(cell['mean'])} (seed wobble "
                               f"{figures.fmt(cell['seed_wobble'])})")
    split = report._split_index(res_tw, "B2")[("across", "pval", TW, "broad")]
    assert report._split_text(split) in cells[2]
    # A2's twin differs from its real g by 25 %: row 3 carries the twin's split, row 2 the real's
    split_a = report._split_index(res_tw, "A2")
    assert report._split_text(split_a[("across", "pval", TW, "broad")]) in cells[1]
    assert report._split_text(split_a[("across", "pval", "real", "broad")]) in _cells(lines[1])[1]
    assert report._split_text(split_a[("across", "pval", "real", "broad")]) not in cells[1]
    assert cells[1].startswith(figures.fmt(next(
        r for r in res_tw["grid"] if (r["g_version"], r["space"], r["mark_class"], r["metric"],
                                      r["column"], r["row"]) ==
        ("across", "pval", "broad", "crps_all", "A", 3))["mean"]))
    assert cells[3] == "not run" and cells[4] == "not run"            # no C2/D2 twin
    assert "reference, no seed" in cells[5] and "reference, no seed" in cells[6]


def test_grid_counts_the_twins_exploding_records(dirs):
    _, _, tw, _ = dirs
    rung, gv, sp, cls, n = EXPLODE_TW
    g = report.build_grid(tw)
    block = g.split(f"## {figures.GV_NAME[gv]}, {figures.SPACE_NAME[sp]}", 1)[1]
    block = block.split(f"### {cls}", 1)[1].split("###", 1)[0]
    counts = block.split("(pair, seed) trained records with CRPS on all bins > 20", 1)[1]
    row3 = next(ln for ln in counts.splitlines() if ln.startswith(f"| {ROW3}"))
    assert _cells(row3)[1:] == [str(n), "0", "n/a", "n/a"]


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


def test_checks_card_orders_the_twin_checks_last(dirs):
    _, _, tw, _ = dirs
    rows = figures.sort_checks(figures.load_checks(tw, "A2"))
    assert [c["check"] for c in rows[-48:]] == ["beatsxshuf"] * 24 + ["lawtest_xshuf"] * 24
    assert all(c["check"] not in ("beatsxshuf", "lawtest_xshuf") for c in rows[:-48])


def test_figures_write_the_nine_pngs_for_a2_with_the_twin(dirs):
    exe = _mpl_python()
    out, _, tw, _ = dirs
    shutil.copy(out / "qm_curves.json", tw / "qm_curves.json")
    p = subprocess.run([exe, str(LADDER / "figures.py"), str(tw), "A2", "--refs-qm",
                        str(tw / "qm_curves.json")], capture_output=True, text=True,
                       timeout=400)
    assert p.returncode == 0, p.stderr[-3000:]
    pngs = sorted((tw / "A2" / "figures").glob("*.png"))
    assert [q.stem for q in pngs] == sorted(figures.FIG_NAMES)
    assert all(q.stat().st_size > 10_000 for q in pngs)
