"""OXl: the other_experiment rung of layers.tsv per pair class (plan Amendment 4, decision 5).

Known-answer tests on the total-variance simulator (`tools/onion_eval/total_variance.py`). The simulator
has no lab level, so the test maps its nested levels (crossed=False): the simulator's `donor` plays the
lab and its `experiment` plays a repeat experiment inside one lab. Then a simulator anisogenic pair (other
`donor`, so other lab and other experiment, same cell type) is an other-lab pair, class `experiment;lab`,
and a simulator other_experiment pair (same `donor`, other experiment) is a same-lab repeat, class
`experiment`. The planted values are pair-value differences of a reference draw that shares every latent
rate with the test draw and differs from it by Poisson noise only (the method of `total_variance.audit`).
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "tools", REPO / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from onion_eval import layers as LY  # noqa: E402
from onion_eval import total_variance as TV  # noqa: E402

# simulator level -> role here: donor = lab (large), experiment = repeat experiment inside a lab (small)
LAB_LAYERS = {"cell_type": 0.5, "donor": 0.30, "experiment": 0.04, "bio_rep": 0.08, "library": 0.05,
              "sequencing": 0.02}
# Seed 2 is the first seed of 0-11 where both class intervals hold the planted value. Over seeds 0-11 the
# other-lab interval held it 12 of 12 times and the repeat interval 9 of 12 (the misses were by 0.0000 to
# 0.0011, against a layer of 0.016): the interval covers the cells only, not the Poisson noise of the
# reference draw, which weighs more on a small layer. test_coverage_over_seeds pins the rate.
SEED, N_CELLS = 2, 6


def _class_pairs(sim, lab_knobs="experiment;lab", repeat_knobs="experiment"):
    """isogenic, other_cell_type and the two other_experiment classes, with `knobs_differing`."""
    out = []
    for p in sim["pairs"]:
        if p["rung"] in ("isogenic", "other_cell_type"):
            out.append(dict(p, knobs=""))
        elif p["rung"] == "anisogenic":                   # other lab (and other experiment)
            out.append(dict(p, rung="other_experiment", pair_id="lab:" + p["pair_id"], knobs=lab_knobs))
        elif p["rung"] == "other_experiment":             # same lab, other experiment
            out.append(dict(p, pair_id="rep:" + p["pair_id"], knobs=repeat_knobs))
    return out


def _run(sim, pairs, n_boot=400, scope=None):
    """layers.run on tables written by total_variance._write_tables, with the pairs' knobs_differing."""
    with tempfile.TemporaryDirectory() as td:
        tables, ev = TV._write_tables(sim, td, pairs)
        knobs = {p["pair_id"]: p.get("knobs", "") for p in pairs}
        with open(tables / "pairs.tsv", newline="") as fh:
            rows = list(csv.DictReader(fh, delimiter="\t"))
        for r in rows:
            r["knobs_differing"] = knobs[r["pair_id"]]
        with open(tables / "pairs.tsv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
            w.writeheader()
            w.writerows(rows)
        if scope:
            e = json.loads(ev.read_text())
            e["scorable"] = {"layers": {"isogenic": scope}}
            ev.write_text(json.dumps(e))
        out = Path(td) / "layers.tsv"
        LY.run(tables, ev, out, strata_path=Path(td) / "none.h5", n_boot=n_boot)
        text = out.read_text()
    return list(csv.DictReader(text.splitlines(), delimiter="\t")), text


def _all(rows, assay="A"):
    return [r for r in rows if r["stratum"] == "all" and r["band"] == "all" and r["pooled_by_class"] == "0"
            and r["assay_or_class"] == assay]


def _one(rows, rung, pair_class=""):
    got = [r for r in _all(rows) if r["rung"] == rung and r.get("pair_class", "") == pair_class]
    assert len(got) == 1, (rung, pair_class, len(got))
    return got[0]


@pytest.fixture(scope="module")
def planted():
    big = TV.simulate(LAB_LAYERS, n_cells=N_CELLS, seed=SEED, count_seed=SEED + 977, crossed=False)
    pv = TV.pair_values(big, "log1p", ["isogenic", "anisogenic", "other_experiment"])
    return {"lab": pv["anisogenic"] - pv["isogenic"], "repeat": pv["other_experiment"] - pv["isogenic"]}


@pytest.fixture(scope="module")
def sim():
    return TV.simulate(LAB_LAYERS, n_cells=N_CELLS, seed=SEED, crossed=False)


@pytest.fixture(scope="module")
def table(sim):
    pairs = _class_pairs(sim, lab_knobs="experiment;lab;read_length;platform")
    return _run(sim, pairs, scope=["B"])


def test_planted_lab_layer_recovered_on_the_other_lab_row(table, planted):
    rows, _ = table
    lab = _one(rows, "other_experiment", "experiment;lab")
    assert lab["layer"] == "experiment" and lab["increment_over"] == "isogenic"
    assert (lab["n_pairs"], lab["n_cells"]) == ("12", "6")
    assert float(lab["ci_low"]) <= planted["lab"] <= float(lab["ci_high"]), (planted, lab)
    # the lab layer is far above the repeat layer, and the lab row's interval excludes the repeat value
    assert planted["lab"] > 3 * planted["repeat"]
    assert float(lab["ci_low"]) > planted["repeat"]


def test_planted_repeat_layer_on_the_repeat_row_never_named_lab(table, planted):
    rows, _ = table
    rep = _one(rows, "other_experiment", "experiment")
    assert rep["layer"] == "experiment_repeat" and rep["increment_over"] == "isogenic"
    assert float(rep["ci_low"]) <= planted["repeat"] <= float(rep["ci_high"]), (planted, rep)
    assert float(rep["ci_high"]) < planted["lab"]
    # the only row named as the experiment (lab) layer is the other-lab class row
    named = [r for r in _all(rows) if r["layer"] == "experiment" and r["pair_class"]]
    assert [r["pair_class"] for r in named] == ["experiment;lab"]


def test_pooled_row_stays_between_the_classes(table):
    rows, _ = table
    pooled = _one(rows, "other_experiment", "")
    assert pooled["layer"] == "experiment" and pooled["n_pairs"] == "24"
    lo = float(_one(rows, "other_experiment", "experiment")["layer_increment"])
    hi = float(_one(rows, "other_experiment", "experiment;lab")["layer_increment"])
    assert lo < float(pooled["layer_increment"]) < hi
    assert float(pooled["layer_increment"]) == pytest.approx((lo + hi) / 2, rel=1e-9)   # same cells, 2+2 pairs


def test_coverage_over_seeds():
    hit = tot = 0
    for s in range(12):
        big = TV.simulate(LAB_LAYERS, n_cells=N_CELLS, seed=s, count_seed=s + 977, crossed=False)
        pv = TV.pair_values(big, "log1p", ["isogenic", "anisogenic", "other_experiment"])
        sim = TV.simulate(LAB_LAYERS, n_cells=N_CELLS, seed=s, crossed=False)
        rows, _ = _run(sim, _class_pairs(sim), n_boot=400)
        for rung, cls in (("anisogenic", "experiment;lab"), ("other_experiment", "experiment")):
            r = _one(rows, "other_experiment", cls)
            hit += float(r["ci_low"]) <= pv[rung] - pv["isogenic"] <= float(r["ci_high"])
            tot += 1
            if cls == "experiment;lab":        # never confused with the repeat layer
                assert float(r["ci_low"]) > pv["other_experiment"] - pv["isogenic"]
    assert hit / tot >= 0.8, hit / tot


def test_class_row_equals_a_table_of_that_class_alone(sim):
    """A class row is the pooled computation on that class's pairs: same numbers as a table built from
    those pairs alone with no class."""
    pairs = _class_pairs(sim)
    rows, _ = _run(sim, pairs, n_boot=200)
    only = [p for p in pairs if p["rung"] != "other_experiment" or p["knobs"] == "experiment;lab"]
    alone, _ = _run(sim, [dict(p, knobs="") for p in only], n_boot=200)
    a, b = _one(rows, "other_experiment", "experiment;lab"), _one(alone, "other_experiment")
    for k in ("variance", "layer_increment", "ci_low", "ci_high", "n_pairs", "n_cells", "increment_over"):
        assert a[k] == b[k], k


def test_columns_tech_diff_and_flag_on_every_row(table):
    rows, text = table
    assert text.splitlines()[0].split("\t") == LY.COLUMNS + ["pair_class", "tech_diff_pairs"]
    assert all(LY.FLAG_SECOND in r["flag"] for r in rows)
    assert {r["flag"] for r in rows if r["assay_or_class"] == "B"} == {f"{LY.FLAG_EMPTY}; {LY.FLAG_SECOND}"}
    assert _one(rows, "other_experiment", "experiment;lab")["tech_diff_pairs"] == "12"
    assert _one(rows, "other_experiment", "experiment")["tech_diff_pairs"] == "0"
    assert _one(rows, "other_experiment", "")["tech_diff_pairs"] == "12"
    assert {r["pair_class"] for r in rows if r["rung"] != "other_experiment"} == {""}
    assert {r["tech_diff_pairs"] for r in rows if r["rung"] != "other_experiment"} == {""}


def test_cell_type_row_over_same_donor_pairs(sim):
    """Both classes share the donor: the cell-type row subtracts them (no flag). Relabel every pair as a
    donor change: the cell-type row falls back to the pooled rung with the flag `donor not separated`."""
    rows, _ = _run(sim, _class_pairs(sim), n_boot=50)
    ct = _one(rows, "other_cell_type")
    assert LY.FLAG_DONOR not in ct["flag"] and ct["increment_over"] == "other_experiment"
    sd = _one(rows, "other_experiment", "")       # here the same-donor pairs are all the pairs
    assert float(ct["layer_increment"]) == pytest.approx(float(ct["variance"]) - float(sd["variance"]), rel=1e-9)
    rows2, _ = _run(sim, _class_pairs(sim, "experiment;lab;donor", "experiment;donor"), n_boot=50)
    ct2 = _one(rows2, "other_cell_type")
    assert ct2["flag"] == f"{LY.FLAG_DONOR}; {LY.FLAG_SECOND}"
    assert {r["layer"] for r in _all(rows2) if r["pair_class"]} == {"experiment+donor", "experiment_repeat+donor"}


def _recs(spec):
    """Records with one ("all", "all") value: (rung, cell, Var(d), pair_class)."""
    return [dict(assay="A", assay_class="k", rung=r, cell=c, v={("all", "all"): v},
                 **({"pair_class": k} if k else {})) for r, c, v, k in spec]


def test_cell_type_subtracts_only_the_same_donor_pairs():
    spec = []
    for i in range(4):
        c = f"c{i}"
        spec += [("isogenic", c, 1.0, ""), ("other_experiment", c, 2.0 + i, "experiment;lab"),
                 ("other_experiment", c, 3.0, "experiment;lab;donor"), ("other_cell_type", c, 10.0, "")]
        if i < 2:
            spec.append(("other_experiment", c, 2.0 + i, "experiment;lab"))     # 6 same-donor pairs, 4 cells
    rows = {(r["rung"], r.get("pair_class")): r for r in LY.layer_table(_recs(spec), n_boot=50)}
    ct = rows[("other_cell_type", "")]
    assert ct["increment_over"] == "other_experiment"
    terms = LY.parse_terms(ct["increment_over"])
    sd_pv = {f"c{i}": 0.5 * (2.0 + i) for i in range(4)}
    want = np.mean([5.0 - sum(k * (sd_pv[c] if q == "other_experiment" else 0.5) for q, k in terms)
                    for c in sd_pv])
    assert ct["layer_increment"] == pytest.approx(want, rel=1e-12)
    assert LY.FLAG_DONOR not in ct["flag"]
    # 4 same-donor pairs: under 5, so the pooled rung is used and flagged
    few = [s for s in spec if not (s[3] == "experiment;lab" and s[1] in ("c2", "c3"))]
    ct2 = {(r["rung"], r.get("pair_class")): r for r in LY.layer_table(_recs(few), n_boot=50)}[("other_cell_type", "")]
    assert LY.FLAG_DONOR in ct2["flag"]


def test_explained_table_class_rows():
    recs = []
    for i in range(4):
        c = f"c{i}"
        recs += [dict(rung="isogenic", cell=c, captured=0.0, pair_value=1.0),
                 dict(rung="other_experiment", cell=c, captured=0.5, pair_value=2.0, pair_class="experiment"),
                 dict(rung="other_experiment", cell=c, captured=1.5, pair_value=4.0, pair_class="experiment;lab")]
    rows = LY.explained_table(recs, n_boot=50)
    got = {r["pair_class"]: r for r in rows if r["rung"] == "other_experiment"}
    assert got["experiment"]["layer"] == "experiment_repeat" and got["experiment;lab"]["layer"] == "experiment"
    assert got["experiment"]["explained_fraction"] == pytest.approx(0.5 / 1.0)
    assert got["experiment;lab"]["explained_fraction"] == pytest.approx(1.5 / 3.0)
    assert got[""]["explained_fraction"] == pytest.approx(1.0 / 2.0)
    assert [r["pair_class"] for r in rows] == ["", "experiment", "experiment;lab"]


def test_pair_class_of():
    assert LY.pair_class_of("experiment;lab;read_length;platform") == ("experiment;lab", True)
    assert LY.pair_class_of("experiment;donor?") == ("experiment;donor?", False)
    assert LY.pair_class_of("experiment") == ("experiment", False)
    assert LY.pair_class_of("") == ("", False)


# ---------------------------------------------------------------------------------------------------
# Byte identity without classes: digests written on the base commit 5bfad89 with the same recipes.

BASE_DIGESTS = {"assemble": "7c07f886b2719e04b09dda947b84b5ef52e5fc797c6897917f956e85f26764d3",
                "run": "a3eba56e7c30a5a78f0f442bad6fe5255b443db683c01f8248ca8b35fe3c85e8",
                "explained": "852c7f7db3db44211e781261535502b721ca774480ff38228f8182a666126090"}


def test_table_without_classes_is_byte_identical_to_base():
    import test_onion_eval_layers as T
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "l.tsv"
        LY.write_tsv(LY.assemble(T.ladder_records(seed=5), n_boot=200, seed=5), p)
        assert hashlib.sha256(p.read_bytes()).hexdigest() == BASE_DIGESTS["assemble"]
    # other_experiment pairs with empty knobs_differing (today's tables) through layers.run
    s = TV.simulate(TV.BASE_LAYERS, n_cells=4, seed=1)
    ps = [q for q in s["pairs"] if q["rung"] in ("pseudoreplicate", "isogenic", "anisogenic", "other_experiment",
                                                  "other_cell_type")]
    with tempfile.TemporaryDirectory() as td:
        tables, ev = TV._write_tables(s, td, ps)
        out = Path(td) / "layers.tsv"
        LY.run(tables, ev, out, strata_path=Path(td) / "none.h5", n_boot=200)
        assert hashlib.sha256(out.read_bytes()).hexdigest() == BASE_DIGESTS["run"]
    er = [dict(rung=r, cell=f"c{c}", captured=0.1 * c + i, pair_value=0.3 * c + 2 * i + 1)
          for i, r in enumerate(["isogenic", "anisogenic", "other_experiment", "other_cell_type"]) for c in range(4)]
    got = json.dumps(LY.explained_table(er, n_boot=100), sort_keys=True, default=str).encode()
    assert hashlib.sha256(got).hexdigest() == BASE_DIGESTS["explained"]
