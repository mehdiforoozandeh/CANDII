"""t118 row 2 — `score.py trained` and `score.py law` on per-bin (row-2) runs.

Real 5-step CPU runs on `synth.make_products`, scored through the CLI (`score.main`), so the
checkpoint loader, the Predictor dispatch and the row-2 `describe` dict are all on the path.
A row-1 A run is scored once before the row-2 form modules are imported and once after; the two
record sets must be equal (the row-1 records do not move).
"""
from __future__ import annotations

import collections
import importlib
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "t118"))

from ladder import pairs, score, synth, train  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:.*meant for synthetic data only")
FAST = {"window": 256, "batch": 16, "knot_sample": 20_000, "knot_block": 256}
G = "T1"
ROW2_MODS = tuple(f"ladder.fforms.form_{r.lower()}" for r in pairs.RUNGS_ROW2)
VOLATILE = ("created_utc", "timing")


@pytest.fixture(scope="module")
def kit(tmp_path_factory):
    root = tmp_path_factory.mktemp("row2score")
    manifest, cov = synth.make_products(root / "products", 0)
    bed = root / "blacklist.bed"
    bed.write_text("chr19\t0\t2500\nchr21\t5000\t7500\nchr22\t0\t250\n")
    return {"root": root, "products": root / "products", "manifest": manifest, "cov": cov,
            "bed": bed, "rows": pairs.read_manifest(manifest)}


def _train(kit, rung, model_name="real", space="counts", sub="runs"):
    return train.run_training(kit["manifest"], kit["cov"], kit["products"], kit["root"] / sub,
                              rung, G, space, model_name, 0, max_steps=5, eval_every=5,
                              device="cpu", lr=1e-2, **FAST)


def _score(kit, run_dir, cmd):
    argv = [cmd, str(kit["manifest"]), str(kit["cov"]), str(kit["products"]), str(kit["bed"]),
            str(run_dir), "--workers", "1"]
    assert score.main(argv) == 0
    name = "scores.json" if cmd == "trained" else "law.json"
    return json.loads((run_dir / name).read_text())


def _score_both(kit, run_dir):
    t0 = time.time()
    s = _score(kit, run_dir, "trained")
    law = _score(kit, run_dir, "law")
    with np.load(run_dir / "figdata.npz") as z:
        figs = {k: z[k] for k in z.files}
    return {"scores": s, "law": law, "figs": figs, "seconds": time.time() - t0}


def _kinds(recs):
    return collections.Counter((r["kind"], r["eval"]) for r in recs)


def _stable(doc):
    return {k: v for k, v in doc.items() if k not in VOLATILE}


@pytest.fixture(scope="module")
def row1_a(kit):
    """Row-1 A, trained and scored while no row-2 form module is imported; then a copy of the
    run is re-scored after importing A2-D2. The saved module objects are put back afterwards,
    so other test files keep the classes they imported."""
    saved = {m: sys.modules.pop(m) for m in ROW2_MODS if m in sys.modules}
    pkg = sys.modules.get("ladder.fforms")
    saved_attrs = {m: getattr(pkg, m.rsplit(".", 1)[1]) for m in saved
                   if pkg is not None and hasattr(pkg, m.rsplit(".", 1)[1])}
    for m in saved_attrs:
        delattr(pkg, m.rsplit(".", 1)[1])
    try:
        rd = _train(kit, "A")
        after_dir = kit["root"] / "rescore" / rd.name
        shutil.copytree(rd, after_dir)                  # unscored copy: same name, same ckpt
        before = _score_both(kit, rd)
        clean = not any(m in sys.modules for m in ROW2_MODS)
        for m in ROW2_MODS:
            importlib.import_module(m)
        after = _score_both(kit, after_dir)
    finally:
        sys.modules.update(saved)
        for m, mod in saved_attrs.items():
            setattr(pkg, m.rsplit(".", 1)[1], mod)
    return {"run_dir": rd, "before": before, "after": after, "clean": clean}


@pytest.fixture(scope="module")
def row2_runs(kit, row1_a):          # row1_a first: its "before" pass must precede the imports
    out = {}
    for rung, model_name in (("A2", "real"), ("D2", "real"), ("A2", "nocov")):
        rd = _train(kit, rung, model_name)
        out[(rung, model_name)] = {"run_dir": rd, **_score_both(kit, rd)}
    return out


# ---------------------------------------------------------------------------------------------


def test_row1_records_unchanged_by_row2_imports(row1_a):
    assert row1_a["clean"], "a row-2 form module was imported before the row-1 'before' pass"
    b, a = row1_a["before"], row1_a["after"]
    assert _stable(a["scores"]) == _stable(b["scores"])
    assert _stable(a["law"]) == _stable(b["law"])
    assert sorted(a["figs"]) == sorted(b["figs"])
    for k in b["figs"]:
        assert np.array_equal(a["figs"][k], b["figs"][k]), k
    # the row-1 describe is the form's own dict, with no row-2 keys
    rec = next(r for r in b["scores"]["records"] if r["kind"] == "trained")
    assert set(rec["describe"]).isdisjoint({"n_levels", "levels_x", "response"})


@pytest.mark.parametrize("key", [("A2", "real"), ("D2", "real"), ("A2", "nocov")])
def test_row2_record_kinds_and_counts_match_row1(kit, row1_a, row2_runs, key):
    r1, r2 = row1_a["before"], row2_runs[key]
    assert _kinds(r2["scores"]["records"]) == _kinds(r1["scores"]["records"])
    assert _kinds(r2["law"]["records"]) == _kinds(r1["law"]["records"])
    kinds = {r["kind"] for r in r2["scores"]["records"]}
    assert {"trained", "shuffle", "swap"} <= kinds
    assert len([r for r in r2["scores"]["records"] if r["kind"] == "shuffle"]) \
        == len(pairs.train_pairs(kit["rows"], G))
    assert len(r2["law"]["records"]) == len(r1["law"]["records"]) > 0
    assert r2["scores"]["skipped"] == r1["scores"]["skipped"]
    # run_info: same keys, the rung as trained (the row is derived downstream)
    assert set(r2["scores"]["run"]) == set(r1["scores"]["run"])
    assert r2["scores"]["run"]["rung"] == key[0] and pairs.row_of(key[0]) == 2


@pytest.mark.parametrize("key", [("A2", "real"), ("D2", "real"), ("A2", "nocov")])
def test_row2_describe_crps_and_figdata(row1_a, row2_runs, key):
    r2 = row2_runs[key]
    recs = r2["scores"]["records"] + r2["law"]["records"]
    per_form = {"A2": {"a", "b", "disp"}, "D2": {"film_gamma_abs_dev", "film_beta_abs"}}[key[0]]
    described = [r for r in recs if "describe" in r]
    assert described and all(r["kind"] == "trained" for r in described)
    for r in described:
        d = r["describe"]
        assert set(d) == {"n_levels", "levels_x", "response"} | per_form
        assert d["n_levels"] == 12 and len(d["levels_x"]) == 12
        assert all(len(d[k]) == 12 for k in per_form)
        assert set(d["response"]) == {"loc", "disp"}
        assert all(len(v) == 12 and all(math.isfinite(x) for x in v)
                   for v in d["response"].values())
    for r in recs:
        assert r["n_crps_nonfinite"] == 0
        for s in score.SUBSETS:
            assert r[f"crps_{s}"] is not None and math.isfinite(r[f"crps_{s}"]), (r["kind"], s)
    assert r2["scores"]["n_records_crps_nonfinite"] == 0
    assert r2["law"]["n_records_crps_nonfinite"] == 0
    swaps = [r for r in r2["scores"]["records"] if r["kind"] == "swap"]
    assert swaps and all(r[score.SWAP_KEY] is not None for r in swaps)
    assert (r2["run_dir"] / "figdata.npz").is_file()
    assert (r2["run_dir"] / "SCORE_DONE").is_file() and (r2["run_dir"] / "LAW_DONE").is_file()
    if key[1] == "real":             # the real model writes the same figure arrays as row 1
        assert sorted(r2["figs"]) == sorted(row1_a["before"]["figs"])
        assert all(np.isfinite(v).all() for v in r2["figs"].values())
    else:
        assert r2["figs"] == {}


def test_row2_nocov_describe_ignores_covariates(row2_runs):
    """The nocov twin's averaged map: every trained record carries one and the same describe."""
    recs = [r for r in row2_runs[("A2", "nocov")]["scores"]["records"] if "describe" in r]
    assert len({json.dumps(r["describe"], sort_keys=True) for r in recs}) == 1
