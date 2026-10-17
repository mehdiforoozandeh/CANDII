"""t118 row 2 — `tools/t118/ladder/aggregate.py` reads row-2 runs beside row-1 runs.

Fabricated per-run scores.json/law.json (as `tests/test_t118_ladder_score.py` fabricates them) for
rungs A, B (row 1) and A2, B2 (row 2) on track T1, counts. Checks: the default call (`--rows 1`, no
`--also-runs`) ignores row-2 runs byte for byte; with both rows, the `beatsrow1` check, beatsbelow
within row 2, `rung_choice_row2`, the `grid` rows and the pinned schema (`figures.check_schema`).
"""
from __future__ import annotations

import gzip
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "t118"))

from ladder import aggregate, figures, pairs, score, synth  # noqa: E402

#: crps_all per (rung, model) per seed; crps_top1 = 2 x crps_all
FAB = {("A", "real"): [1.00, 1.10, 0.95], ("A", "nocov"): [2.0, 2.0, 2.0],
       ("A", "ids"): [1.05, 1.05, 1.05],
       ("B", "real"): [0.60, 0.62, 0.61], ("B", "nocov"): [2.0, 2.1, 2.0],
       ("B", "ids"): [0.9, 0.9, 0.9],
       ("A2", "real"): [0.40, 0.42, 0.41], ("A2", "nocov"): [2.0, 2.0, 2.0],
       ("A2", "ids"): [1.0, 1.0, 1.0],
       ("B2", "real"): [0.50, 0.70, 0.55], ("B2", "nocov"): [2.0, 2.0, 2.1],
       ("B2", "ids"): [0.8, 0.8, 0.8]}
#: chr22 crps_all (real). Row 1: A within B's wobble -> A. Row 2: B2 best, A2 outside -> B2
FAB_VAL = {"A": [1.0, 1.05, 1.02], "B": [0.985, 1.0, 1.015],
           "A2": [1.0, 1.05, 1.02], "B2": [0.90, 0.91, 0.92]}
#: (run, crps_all) of the one trained pair that "explodes" (crps_all > 20)
EXPLODE = ("B2_T1_counts_real_s2", 30.0)
FIXED_NOW = datetime(2026, 9, 26, 12, 0, 0, tzinfo=timezone.utc)
ROW1 = ("A", "B")
ROW2 = ("A2", "B2")


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
        if name == EXPLODE[0] and i == 0:
            r["crps_all"] = EXPLODE[1]
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
    base = tmp_path_factory.mktemp("row2agg")
    manifest, cov = synth.make_products(base / "data", seed=0)
    rows = pairs.read_manifest(manifest)
    runs1, runs2 = base / "runs_row1", base / "runs_row2"
    for runs, rungs in ((runs1, ROW1), (runs2, ROW2)):
        runs.mkdir()
        for rung in rungs:
            for model in pairs.MODELS:
                for seed in pairs.SEEDS:
                    _fab_run(runs, rows, rung, model, seed)
    refs = base / "refs.tsv"
    refs.write_text("track\tspace\trung\teval\tarm_pid\tspread_crps_all\tspread_crps_top1\n"
                    "T1\tcounts\tnoSolution\tscore\tT1__pe__pe\t3.0\t6.0\n")
    return {"base": base, "manifest": manifest, "cov": cov, "rows": rows, "runs1": runs1,
            "runs2": runs2, "refs": refs}


@pytest.fixture(scope="module")
def both(fab):
    """Both rows through the CLI: row-1 runs in runs_dir, row-2 runs via --also-runs."""
    agg = fab["base"] / "agg_both"
    assert aggregate.main([str(fab["manifest"]), str(fab["cov"]), str(fab["runs1"]),
                           str(fab["refs"]), str(agg), "--also-runs", str(fab["runs2"]),
                           "--rows", "1,2"]) == 0
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
    return next(c for c in res["checks"] if c["check"] == name and c["rung"] == rung
                and c["metric"] == metric and c["mark_class"] == mc)


def _mean(v):
    return float(np.mean(v))


def _wob(v):
    return float(max(v) - min(v))


def test_default_ignores_row2_runs_byte_for_byte(fab, tmp_path, monkeypatch):
    monkeypatch.setattr(aggregate, "datetime", _FixedDatetime)
    runs, agg = tmp_path / "runs", tmp_path / "agg"
    shutil.copytree(fab["runs1"], runs)
    aggregate.aggregate(fab["manifest"], fab["cov"], runs, fab["refs"], agg)
    before = _snapshot(agg)
    for d in fab["runs2"].iterdir():
        shutil.copytree(d, runs / d.name)
    shutil.rmtree(agg)
    assert aggregate.main([str(fab["manifest"]), str(fab["cov"]), str(runs), str(fab["refs"]),
                           str(agg)]) == 0
    after = _snapshot(agg)
    assert set(before) == set(after)
    assert {"results.json", "results_summary.tsv", "checks_A.json", "checks_B.json",
            "checks_main.json"} <= set(after)
    for k in before:
        assert before[k] == after[k], k
    res = json.loads(after["results.json"])
    assert not {"rung_choice_row2", "grid", "rows", "also_runs"} & set(res)
    assert not any(r.startswith(("A2", "B2")) for r in res["runs_present"] + res["runs_missing"])
    assert figures.check_schema(res) == []


def test_both_rows_schema_and_runs(fab, both):
    res, agg = both
    assert figures.check_schema(res) == []
    assert len(res["runs_present"]) == 36
    assert "A2_T1_counts_real_s0" in res["runs_present"]
    exp = aggregate.expected_runs(fab["rows"], (1, 2))
    assert len(exp) == 2 * len(aggregate.expected_runs(fab["rows"]))
    assert exp[:len(exp) // 2] == aggregate.expected_runs(fab["rows"])
    assert len(res["runs_missing"]) == len(exp) - 36
    assert "C2_all_pval_ids_s2" in res["runs_missing"]
    assert res["rows"] == [1, 2] and res["also_runs"] == [str(fab["runs2"])]
    for rung in ("A", "B", "A2", "B2"):
        assert (agg / f"checks_{rung}.json").is_file()
        assert (agg / rung / f"checks_{rung}.json").is_file()
    ca2 = json.loads((agg / "checks_A2.json").read_text())
    assert ca2["rung"] == "A2"
    assert {c["check"] for c in ca2["checks"]} == {"beatstwin", "lawtest_nocov", "lawtest_ids",
                                                  "beatsrow1"}
    cb2 = json.loads((agg / "checks_B2.json").read_text())
    assert {c["check"] for c in cb2["checks"]} == {"beatstwin", "lawtest_nocov", "lawtest_ids",
                                                  "beatsbelow", "beatsrow1"}


def test_beatsrow1_value_bar_and_wobble(both):
    res, _ = both
    a, a2 = FAB[("A", "real")], FAB[("A2", "real")]
    c = _check(res, "beatsrow1", "A2")
    wmax = max(_wob(a), _wob(a2))
    assert c["value"] == pytest.approx(_mean(a) - _mean(a2))      # positive = row 2 better
    assert c["bar"] == pytest.approx(2 * wmax) and c["seed_wobble"] == pytest.approx(wmax)
    assert c["met"] is True                                        # 0.607 > 0.30
    comp = c["components"]
    assert comp["rung_row1"] == "A"
    assert comp["d_row1"] == pytest.approx(_mean(a)) and comp["d_row2"] == pytest.approx(_mean(a2))
    assert comp["wobble_row1"] == pytest.approx(_wob(a))
    assert comp["wobble_row2"] == pytest.approx(_wob(a2))
    c = _check(res, "beatsrow1", "A2", metric="crps_top1")
    assert c["value"] == pytest.approx(2 * (_mean(a) - _mean(a2)))
    assert c["bar"] == pytest.approx(4 * wmax)
    # B2 against B: B2's per-class values include the exploding pair, so read them back
    b = FAB[("B", "real")]
    pc = next(r for r in res["per_class"] if r["rung"] == "B2" and r["model"] == "real"
              and r["kind"] == "trained" and r["metric"] == "crps_all" and r["variant"] == "all"
              and r["mark_class"] == "narrow")
    c = _check(res, "beatsrow1", "B2")
    assert c["components"]["rung_row1"] == "B"
    assert c["value"] == pytest.approx(_mean(b) - pc["mean"])
    assert c["bar"] == pytest.approx(2 * max(_wob(b), pc["seed_wobble"]))
    assert c["met"] is False
    assert not [x for x in res["checks"] if x["check"] == "beatsrow1" and x["rung"] in ROW1]


def test_beatsbelow_stays_within_row2(both):
    res, _ = both
    c = _check(res, "beatsbelow", "B2", metric="crps_top1")
    assert c["components"]["rung_below"] == "A2"
    assert c["components"]["d_below"] == pytest.approx(2 * _mean(FAB[("A2", "real")]))
    assert not [x for x in res["checks"] if x["check"] == "beatsbelow" and x["rung"] == "A2"]
    c = _check(res, "beatsbelow", "B")
    assert c["components"]["rung_below"] == "A"


def test_rung_choice_row2_and_main_unchanged(fab, both, tmp_path):
    res, agg = both
    ch = res["rung_choice_row2"]["per_track|counts"]
    assert set(ch["val_by_rung"]) == {"A2", "B2"}
    assert ch["best"] == "B2" and ch["chosen"] == "B2"
    assert ch["wobble_by_rung"]["B2"] == pytest.approx(0.02)
    assert ch["val_by_rung"]["A2"] == pytest.approx(_mean(FAB_VAL["A2"]))
    assert set(res["rung_choice"]["per_track|counts"]["val_by_rung"]) == {"A", "B"}
    assert res["rung_choice"]["per_track|counts"]["chosen"] == "A"
    default = tmp_path / "agg"
    aggregate.aggregate(fab["manifest"], fab["cov"], fab["runs1"], fab["refs"], default)
    ref = json.loads((default / "checks_main.json").read_text())
    main = json.loads((agg / "checks_main.json").read_text())
    assert main["rung_choice"] == ref["rung_choice"] and main["checks"] == ref["checks"]
    assert json.loads((default / "results.json").read_text())["rung_choice"] == res["rung_choice"]


def test_grid_rows(both):
    res, _ = both
    grid = res["grid"]
    assert all(set(r) == set(figures.GRID_KEYS) for r in grid)
    cell = {(r["metric"], r["column"], r["row"]): r for r in grid
            if (r["g_version"], r["space"], r["mark_class"]) == ("per_track", "counts", "narrow")}
    assert set(cell) == {(m, c, row) for m in ("crps_all", "crps_top1") for c in ("A", "B")
                         for row in (1, 2)}
    for (m, col, row), r in cell.items():
        assert r["rung"] == (col if row == 1 else f"{col}2")
        pc = next(x for x in res["per_class"] if x["rung"] == r["rung"] and x["model"] == "real"
                  and x["kind"] == "trained" and x["metric"] == m and x["variant"] == "all"
                  and x["mark_class"] == "narrow")
        assert r["mean"] == pytest.approx(pc["mean"])
        assert r["seed_wobble"] == pytest.approx(pc["seed_wobble"])
        assert r["n_pairs_crps_gt_20"] == (1 if r["rung"] == "B2" else 0)
    assert cell[("crps_all", "A", 1)]["mean"] == pytest.approx(_mean(FAB[("A", "real")]))
    # the count is reported only: no check carries it
    assert not [c for c in res["checks"] if "crps_gt_20" in json.dumps(c)]


def test_rows_2_alone(fab, tmp_path):
    agg = tmp_path / "agg"
    res = aggregate.aggregate(fab["manifest"], fab["cov"], fab["runs1"], fab["refs"], agg,
                              also_runs=[fab["runs2"]], rows_set=(2,))
    assert figures.check_schema(res) == []
    assert len(res["runs_present"]) == 18
    assert all(r.startswith(("A2", "B2")) for r in res["runs_present"])
    assert all(n[:2] in ("A2", "B2", "C2", "D2") for n in res["runs_missing"])
    assert res["rung_choice"] == {} and res["rung_choice_row2"]["per_track|counts"]["chosen"] == "B2"
    assert {r["row"] for r in res["grid"]} == {2}
    assert not [c for c in res["checks"] if c["check"] == "beatsrow1"]
    assert json.loads((agg / "checks_main.json").read_text())["checks"] == []


def test_duplicate_run_across_dirs_is_read_once(fab, tmp_path, capsys):
    dup = tmp_path / "dup"
    shutil.copytree(fab["runs2"] / "A2_T1_counts_real_s0", dup / "A2_T1_counts_real_s0")
    res = aggregate.aggregate(fab["manifest"], fab["cov"], fab["runs2"], fab["refs"],
                              tmp_path / "agg", also_runs=[dup], rows_set=(2,))
    assert res["runs_present"].count("A2_T1_counts_real_s0") == 1
    assert "already read" in capsys.readouterr().err


def test_rows_argument():
    assert aggregate._rows_arg("1") == (1,)
    assert aggregate._rows_arg("2,1") == (1, 2)
    for bad in ("3", "", "a", "1,3"):
        with pytest.raises(Exception):
            aggregate._rows_arg(bad)
