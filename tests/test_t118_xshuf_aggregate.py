"""t118 shuffled-bin twin — `tools/t118/ladder/aggregate.py` reads the twin's runs beside both rows.

Fabricated per-run scores.json/law.json (the `_fab_run` of `tests/test_t118_row2_aggregate.py`,
copied) for rungs A, B (row 1), A2, B2 (row 2) and the twin (model `xshuf`) of A2 and B2 in a
third runs dir, on track T1, counts. Planted: the A2 twin is worse than A2 by far more than
2 x the seed wobble (both twin checks met); the B2 twin is within B2's wobble (unmet); one B2 twin
trained record explodes (crps_all > 20). Checks: `beatsxshuf` and `lawtest_xshuf` values, bars,
wobbles and `met`; grid row 3; the pinned schema; `runs_missing` with and without `--xshuf`; and
that every call without twin runs writes exactly what the pre-change code writes.
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "t118"))

from ladder import aggregate, figures, pairs, score, synth  # noqa: E402

XS = pairs.MODEL_XSHUF
#: crps_all per (rung, model) per seed; crps_top1 = 2 x crps_all; law = value + 0.1
FAB = {("A", "real"): [1.00, 1.10, 0.95], ("A", "nocov"): [2.0, 2.0, 2.0],
       ("A", "ids"): [1.05, 1.05, 1.05],
       ("B", "real"): [0.60, 0.62, 0.61], ("B", "nocov"): [2.0, 2.1, 2.0],
       ("B", "ids"): [0.9, 0.9, 0.9],
       ("A2", "real"): [0.40, 0.42, 0.41], ("A2", "nocov"): [2.0, 2.0, 2.0],
       ("A2", "ids"): [1.0, 1.0, 1.0],
       ("B2", "real"): [0.50, 0.70, 0.55], ("B2", "nocov"): [2.0, 2.0, 2.1],
       ("B2", "ids"): [0.8, 0.8, 0.8],
       # the twin: A2's is far worse than A2 (met), B2's is within B2's wobble (unmet)
       ("A2", XS): [0.80, 0.82, 0.81], ("B2", XS): [0.55, 0.60, 0.62]}
FAB_VAL = {"A": [1.0, 1.05, 1.02], "B": [0.985, 1.0, 1.015],
           "A2": [1.0, 1.05, 1.02], "B2": [0.90, 0.91, 0.92]}
#: (run, crps_all) of the one trained pair that "explodes" (crps_all > 20), per run
EXPLODE = {"B2_T1_counts_real_s2": 30.0, "B2_T1_counts_xshuf_s0": 25.0}
FIXED_NOW = datetime(2026, 9, 28, 12, 0, 0, tzinfo=timezone.utc)
ROW1 = ("A", "B")
ROW2 = ("A2", "B2")
#: the tree the twin aggregation was built on (aggregate.py before this chunk)
PRE_CHANGE = "0a8b0df"


def _fab_run(runs_dir: Path, rows, rung, model, seed):
    space, g = "counts", "T1"
    name = f"{rung}_{g}_{space}_{model}_s{seed}"
    d = runs_dir / name
    d.mkdir(parents=True)
    v = FAB[(rung, model)][seed]

    def rec(kind, ev, p, value, **extra):
        r = {"kind": kind, "eval": ev, **{k: p[k] for k in pairs.PAIR_KEYS},
             "cov_src_pid": p["source_pid"], "cov_tgt_pid": p["target_pid"]}
        for m in score.METRICS:
            r[m] = (2 * value if m == "crps_top1" else value) if m.startswith("crps") else 0.5
        r.update(extra)
        return r

    recs = []
    for i, p in enumerate(pairs.train_pairs(rows, g)):
        r = rec("trained", "score", p, v, crps_oracle_scaled_all=0.9 * v, scale_error_all=0.1 * v,
                c_star_all=0.0, pit_hist=[1] * 20, describe={"a": 0.0, "b": 1.0})
        if name in EXPLODE and i == 0:
            r["crps_all"] = EXPLODE[name]
        recs.append(r)
        recs.append(rec("trained", "val", p, FAB_VAL[rung][seed] if model == "real" else 9.0))
    law = [rec("law", "score", p, v + 0.1) for p in pairs.law_pairs(rows, g)]
    run = {"run_name": name, "rung": rung, "g": g, "g_version": "per_track", "space": space,
           "model": model, "seed": seed, "steps": 10, "best_step": 10, "val_nll": 1.0}
    (d / "scores.json").write_text(json.dumps({"run": run, "records": recs, "snippets": []}))
    (d / "law.json").write_text(json.dumps({"run": run, "records": law}))


class _FixedDatetime:
    @staticmethod
    def now(tz=None):
        return FIXED_NOW


@pytest.fixture(scope="module")
def fab(tmp_path_factory):
    base = tmp_path_factory.mktemp("twinagg")
    manifest, cov = synth.make_products(base / "data", seed=0)
    rows = pairs.read_manifest(manifest)
    runs1, runs2, runsx = base / "runs_row1", base / "runs_row2", base / "runs_xshuf"
    for runs, rungs, models in ((runs1, ROW1, pairs.MODELS), (runs2, ROW2, pairs.MODELS),
                                (runsx, ROW2, (XS,))):
        runs.mkdir()
        for rung in rungs:
            for model in models:
                for seed in pairs.SEEDS:
                    _fab_run(runs, rows, rung, model, seed)
    refs = base / "refs.tsv"
    refs.write_text("track\tspace\trung\teval\tarm_pid\tspread_crps_all\tspread_crps_top1\n"
                    "T1\tcounts\tnoSolution\tscore\tT1__pe__pe\t3.0\t6.0\n")
    return {"base": base, "manifest": manifest, "cov": cov, "rows": rows, "runs1": runs1,
            "runs2": runs2, "runsx": runsx, "refs": refs}


def _args(fab, agg, *extra):
    return [str(fab["manifest"]), str(fab["cov"]), str(fab["runs1"]), str(fab["refs"]), str(agg),
            *extra]


@pytest.fixture(scope="module")
def twin(fab):
    """Both rows and the twin through the CLI, exactly as `slurm/t118/xshuf_agg.sh` calls it."""
    agg = fab["base"] / "agg_twin"
    assert aggregate.main(_args(fab, agg, "--also-runs", str(fab["runs2"]), "--also-runs",
                                str(fab["runsx"]), "--rows", "1,2", "--xshuf")) == 0
    return json.loads((agg / "results.json").read_text()), agg


@pytest.fixture(scope="module")
def no_twin(fab):
    agg = fab["base"] / "agg_no_twin"
    assert aggregate.main(_args(fab, agg, "--also-runs", str(fab["runs2"]), "--rows", "1,2")) == 0
    return json.loads((agg / "results.json").read_text()), agg


def _snapshot(agg: Path) -> dict:
    """Every output file's bytes (the gzip decompressed: its header carries an mtime)."""
    out = {}
    for f in sorted(agg.rglob("*")):
        if f.is_file():
            data = f.read_bytes()
            out[str(f.relative_to(agg))] = gzip.decompress(data) if f.suffix == ".gz" else data
    return out


def _check(res, name, rung, metric="crps_all", mc="narrow"):
    hits = [c for c in res["checks"] if c["check"] == name and c["rung"] == rung
            and c["metric"] == metric and c["mark_class"] == mc]
    assert len(hits) == 1, (name, rung, metric, len(hits))
    return hits[0]


def _pc(res, rung, model, kind="trained", metric="crps_all"):
    return next(r for r in res["per_class"] if r["rung"] == rung and r["model"] == model
                and r["kind"] == kind and r["metric"] == metric and r["variant"] == "all"
                and r["mark_class"] == "narrow")


def _mean(v):
    return float(np.mean(v))


def _wob(v):
    return float(max(v) - min(v))


# ---------------------------------------------------------------------------------------------


def test_schema_runs_and_twin_rows(fab, twin):
    res, agg = twin
    assert figures.check_schema(res) == []
    assert len(res["runs_present"]) == 36 + 6
    assert "A2_T1_counts_xshuf_s0" in res["runs_present"]
    assert {r["model"] for r in res["per_class"]} == {"real", "nocov", "ids", XS}
    assert {r["rung"] for r in res["per_track"] if r["model"] == XS} == set(ROW2)
    assert {r["kind"] for r in res["per_class"] if r["model"] == XS} == {"trained", "law"}
    # the twin's trained/score records are kept in per_pair like every other model's
    assert any(r["model"] == XS and r["kind"] == "trained" for r in res["per_pair"])
    for rung in ROW2:
        names = {c["check"] for c in json.loads((agg / f"checks_{rung}.json").read_text())["checks"]}
        assert {"beatsxshuf", "lawtest_xshuf"} <= names
        names = {c["check"] for c in
                 json.loads((agg / rung / f"checks_{rung}.json").read_text())["checks"]}
        assert {"beatsxshuf", "lawtest_xshuf"} <= names
    for rung in ROW1:
        assert not [c for c in res["checks"] if c["rung"] == rung and "xshuf" in c["check"]]


def test_beatsxshuf_met_and_unmet(twin):
    res, _ = twin
    a2, a2x = FAB[("A2", "real")], FAB[("A2", XS)]
    for metric, k in (("crps_all", 1), ("crps_top1", 2)):
        c = _check(res, "beatsxshuf", "A2", metric)
        wmax = k * max(_wob(a2), _wob(a2x))
        assert c["value"] == pytest.approx(k * (_mean(a2x) - _mean(a2)))   # positive = real better
        assert c["bar"] == pytest.approx(2 * wmax) and c["seed_wobble"] == pytest.approx(wmax)
        assert c["met"] is True                                             # 0.40 > 0.04
        comp = c["components"]
        assert set(figures.BEATSXSHUF_COMPONENTS) <= set(comp)
        assert comp["d_real"] == pytest.approx(k * _mean(a2))
        assert comp["d_xshuf"] == pytest.approx(k * _mean(a2x))
        assert comp["wobble_real"] == pytest.approx(k * _wob(a2))
        assert comp["wobble_xshuf"] == pytest.approx(k * _wob(a2x))
    # B2 top1 (no exploding pair in crps_top1): the twin is within the wobble -> unmet
    b2, b2x = FAB[("B2", "real")], FAB[("B2", XS)]
    c = _check(res, "beatsxshuf", "B2", "crps_top1")
    assert c["value"] == pytest.approx(2 * (_mean(b2x) - _mean(b2)))
    assert c["bar"] == pytest.approx(4 * max(_wob(b2), _wob(b2x)))
    assert c["met"] is False and "met_reason" not in c["components"]
    # B2 crps_all carries both exploding pairs: read the per-class values back
    real, xs = _pc(res, "B2", "real"), _pc(res, "B2", XS)
    c = _check(res, "beatsxshuf", "B2")
    assert c["value"] == pytest.approx(xs["mean"] - real["mean"])
    assert c["bar"] == pytest.approx(2 * max(real["seed_wobble"], xs["seed_wobble"]))
    assert c["met"] is (c["value"] > c["bar"])


def test_lawtest_xshuf_reads_the_law_rows(twin):
    res, _ = twin
    for rung, met in (("A2", True), ("B2", False)):
        r, x = FAB[(rung, "real")], FAB[(rung, XS)]
        for metric, k in (("crps_all", 1), ("crps_top1", 2)):
            c = _check(res, "lawtest_xshuf", rung, metric)
            lr, lx = _pc(res, rung, "real", "law", metric), _pc(res, rung, XS, "law", metric)
            assert lr["mean"] == pytest.approx(k * (_mean(r) + 0.1))
            assert lx["mean"] == pytest.approx(k * (_mean(x) + 0.1))
            wmax = max(lr["seed_wobble"], lx["seed_wobble"])
            assert c["value"] == pytest.approx(lx["mean"] - lr["mean"])
            assert c["bar"] == pytest.approx(2 * wmax) and c["seed_wobble"] == pytest.approx(wmax)
            assert c["met"] is met
            assert c["components"] == {"d_real": lr["mean"], "d_xshuf": lx["mean"],
                                       "wobble_real": lr["seed_wobble"],
                                       "wobble_xshuf": lx["seed_wobble"]}


def test_twin_checks_order_after_beatsrow1(twin):
    res, _ = twin
    for rung in ROW2:
        names = [c["check"] for c in res["checks"] if c["rung"] == rung
                 and c["metric"] == "crps_all" and c["mark_class"] == "narrow"]
        i = names.index("beatsrow1")
        assert names[i + 1:i + 3] == ["beatsxshuf", "lawtest_xshuf"]


def test_one_seed_twin_has_no_bar(fab, tmp_path):
    runsx = tmp_path / "runsx"
    shutil.copytree(fab["runsx"] / "A2_T1_counts_xshuf_s0", runsx / "A2_T1_counts_xshuf_s0")
    res = aggregate.aggregate(fab["manifest"], fab["cov"], fab["runs1"], fab["refs"],
                              tmp_path / "agg", also_runs=[fab["runs2"], runsx], rows_set=(1, 2))
    assert figures.check_schema(res) == []
    c = _check(res, "beatsxshuf", "A2")
    assert c["bar"] is None and c["seed_wobble"] is None and c["met"] is False
    assert c["components"]["met_reason"].startswith("no bar")
    # B2 has real runs and no twin run: the check is there, with no value
    c = _check(res, "beatsxshuf", "B2")
    assert c["value"] is None and c["met"] is False and c["components"]["met_reason"] == "no value"


def test_grid_row_3(twin, no_twin):
    res, _ = twin
    grid = res["grid"]
    assert figures._check_grid(grid) == []
    cell = {(r["metric"], r["column"], r["row"]): r for r in grid
            if (r["g_version"], r["space"], r["mark_class"]) == ("per_track", "counts", "narrow")}
    assert set(cell) == {(m, c, row) for m in ("crps_all", "crps_top1") for c in ("A", "B")
                         for row in (1, 2, 3)}
    for (m, col, row), r in cell.items():
        assert r["rung"] == (col if row == 1 else f"{col}2")
        pc = _pc(res, r["rung"], XS if row == 3 else "real", metric=m)
        assert r["mean"] == pytest.approx(pc["mean"])
        assert r["seed_wobble"] == pytest.approx(pc["seed_wobble"])
    assert cell[("crps_all", "A", 3)]["mean"] == pytest.approx(_mean(FAB[("A2", XS)]))
    assert cell[("crps_top1", "B", 3)]["mean"] == pytest.approx(2 * _mean(FAB[("B2", XS)]))
    # the explode count is keyed by model: the twin's own planted record, and the real one's
    assert {k: r["n_pairs_crps_gt_20"] for k, r in cell.items() if k[0] == "crps_all"} == \
        {("crps_all", "A", 1): 0, ("crps_all", "B", 1): 0, ("crps_all", "A", 2): 0,
         ("crps_all", "B", 2): 1, ("crps_all", "A", 3): 0, ("crps_all", "B", 3): 1}
    # rows 1 and 2 are the same as without the twin
    assert [r for r in grid if r["row"] != 3] == no_twin[0]["grid"]


def test_twin_ignored_where_it_should_be(twin, no_twin):
    res, ref = twin[0], no_twin[0]
    for k in ("law_grid", "knob_gain", "rung_choice", "rung_choice_row2", "depth_law"):
        assert res[k] == ref[k], k
    assert "d_xshuf" not in json.dumps(res["law_grid"])
    # every non-twin check is unchanged, in the same order
    assert [c for c in res["checks"] if "xshuf" not in c["check"]] == ref["checks"]
    assert [r for r in res["per_class"] if r["model"] != XS] == ref["per_class"]


def test_runs_missing_lists_twin_only_with_flag(fab, twin, tmp_path):
    res, _ = twin
    base = aggregate.expected_runs(fab["rows"], (1, 2))
    exp = aggregate.expected_runs(fab["rows"], (1, 2), xshuf=True)
    assert exp[:len(base)] == base
    xnames = exp[len(base):]
    g_ids = tuple(pairs.track_ids(fab["rows"])) + ("all",)
    assert len(xnames) == len(pairs.RUNGS_ROW2) * len(g_ids) * len(pairs.SPACES) * len(pairs.SEEDS)
    assert all(n.split("_")[3] == XS for n in xnames)
    assert set(res["runs_missing"]) == set(exp) - set(res["runs_present"])
    assert "C2_all_pval_xshuf_s2" in res["runs_missing"]
    assert "A2_T1_counts_xshuf_s0" not in res["runs_missing"]
    # the twin dir read without --xshuf: runs present and checks written, none expected
    agg = tmp_path / "agg"
    assert aggregate.main(_args(fab, agg, "--also-runs", str(fab["runs2"]), "--also-runs",
                                str(fab["runsx"]), "--rows", "1,2")) == 0
    res2 = json.loads((agg / "results.json").read_text())
    assert not [n for n in res2["runs_missing"] if XS in n]
    assert "A2_T1_counts_xshuf_s0" in res2["runs_present"]
    assert res2["checks"] == res["checks"] and res2["grid"] == res["grid"]


def test_expected_twin_names_are_the_task_table():
    rows = pairs.read_manifest(ROOT / "tests" / "fixtures" / "t112_meta" / "MANIFEST.tsv")
    base = aggregate.expected_runs(rows, (1, 2))
    xnames = aggregate.expected_runs(rows, (1, 2), xshuf=True)[len(base):]
    assert xnames == [t["run_name"] for t in pairs.tasks(rows, 2, xshuf=True)]
    assert len(xnames) == 192
    assert aggregate.expected_runs(rows, (1, 2), xshuf=False) == base


def test_xshuf_needs_row_2(fab, tmp_path, capsys):
    for rows in ([], ["--rows", "1"]):
        with pytest.raises(SystemExit) as e:
            aggregate.main(_args(fab, tmp_path / "agg", *rows, "--xshuf"))
        assert e.value.code == 2
    assert "--xshuf needs 2 in --rows" in capsys.readouterr().err
    with pytest.raises(ValueError):
        aggregate.aggregate(fab["manifest"], fab["cov"], fab["runs1"], fab["refs"],
                            tmp_path / "agg2", rows_set=(1,), xshuf=True)
    assert aggregate.main(_args(fab, tmp_path / "agg3", "--rows", "2", "--also-runs",
                                str(fab["runs2"]), "--xshuf")) == 0


# ---------------------------------------------------------------------------------------------
# without twin runs, every output is what the pre-change code writes
# ---------------------------------------------------------------------------------------------


def _pre_change_module(tmp_path):
    """aggregate.py as it was before the twin (from git); None when git or the commit is absent."""
    try:
        src = subprocess.run(["git", "-C", str(ROOT), "show",
                              f"{PRE_CHANGE}:tools/t118/ladder/aggregate.py"],
                             capture_output=True, check=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    path = tmp_path / "aggregate_pre.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("aggregate_pre_xshuf", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("extra", [(), ("--rows", "1,2", "--also-runs", "RUNS2")],
                         ids=["default", "rows12"])
def test_no_twin_output_equals_pre_change(fab, tmp_path, monkeypatch, extra):
    pre = _pre_change_module(tmp_path)
    if pre is None:
        pytest.skip(f"git show {PRE_CHANGE} unavailable")
    extra = [str(fab["runs2"]) if a == "RUNS2" else a for a in extra]
    monkeypatch.setattr(aggregate, "datetime", _FixedDatetime)
    monkeypatch.setattr(pre, "datetime", _FixedDatetime)
    agg = tmp_path / "agg"             # one path for both: results.json names the agg dir
    assert pre.main(_args(fab, agg, *extra)) == 0
    old = _snapshot(agg)
    shutil.rmtree(agg)
    assert aggregate.main(_args(fab, agg, *extra)) == 0
    new = _snapshot(agg)
    assert set(old) == set(new)
    for k in old:
        assert old[k] == new[k], k
        assert b"xshuf" not in new[k], k


def test_empty_twin_dir_changes_nothing_but_also_runs(fab, tmp_path, monkeypatch):
    monkeypatch.setattr(aggregate, "datetime", _FixedDatetime)
    empty = tmp_path / "runs_twin_empty"
    empty.mkdir()
    agg = tmp_path / "agg"             # one path for both: results.json names the agg dir
    assert aggregate.main(_args(fab, agg, "--also-runs", str(fab["runs2"]), "--rows", "1,2")) == 0
    a = _snapshot(agg)
    shutil.rmtree(agg)
    assert aggregate.main(_args(fab, agg, "--also-runs", str(fab["runs2"]),
                                "--also-runs", str(empty), "--rows", "1,2")) == 0
    b = _snapshot(agg)
    assert set(a) == set(b)
    for k in a:
        assert b"xshuf" not in a[k] and b"xshuf" not in b[k], k
        if k == "results.json":
            ra, rb = json.loads(a[k]), json.loads(b[k])
            assert rb.pop("also_runs") == [str(fab["runs2"]), str(empty)]
            ra.pop("also_runs")
            assert ra == rb
        else:
            assert a[k] == b[k], k
