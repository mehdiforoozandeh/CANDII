"""Close-out item E4: the replicate ladder (tools/onion_eval/ladder.py) and its two switches
(layers.py --pairs-out, mixed_model.py --second-experiments), each off by default."""
from __future__ import annotations

import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from onion_eval import ladder as LD  # noqa: E402
from onion_eval import layers as LY  # noqa: E402
from onion_eval import mixed_model as MM  # noqa: E402
from onion_eval import total_variance as TV  # noqa: E402


# ---------------------------------------------------------------------------------------------------
# layers.py --pairs-out

@pytest.fixture(scope="module")
def sim():
    return TV.simulate({"cell_type": 0.5, "donor": 0.1, "experiment": 0.05, "bio_rep": 0.05, "library": 0.02,
                        "sequencing": 0.01}, n_cells=4, seed=3)


def test_pairs_out_off_leaves_layers_tsv_identical_and_values_match(sim, tmp_path):
    pairs = [p for p in sim["pairs"] if p["rung"] in ("pseudoreplicate", "isogenic", "other_cell_type")]
    tables, ev = TV._write_tables(sim, tmp_path, pairs)
    off, on = tmp_path / "off.tsv", tmp_path / "on.tsv"
    pv = tmp_path / "pv.tsv.gz"
    LY.run(tables, ev, off, strata_path=tmp_path / "none.h5", n_boot=50)
    rows = LY.run(tables, ev, on, strata_path=tmp_path / "none.h5", n_boot=50, pairs_out=pv)
    assert off.read_bytes() == on.read_bytes()
    got = LD._read_tsv(pv)
    assert list(got[0]) == LY.PAIR_COLUMNS
    n_keys = len({(r["stratum"], r["band"]) for r in got})
    assert len(got) == len(pairs) * n_keys
    # the per-pair values give back each rung's variance (mean over cells of the cell's mean pair value)
    for rung in ("pseudoreplicate", "isogenic", "other_cell_type"):
        cells = {}
        for r in got:
            if r["rung"] == rung and r["stratum"] == "all" and r["band"] == "all":
                cells.setdefault(r["cell"], []).append(0.5 * float(r["var_d"]))
        want = [x for x in rows if x["rung"] == rung and x["stratum"] == "all" and x["band"] == "all"
                and x["pooled_by_class"] == 0][0]["variance"]
        assert np.mean([np.mean(v) for v in cells.values()]) == pytest.approx(want, rel=1e-12)
    # byte-reproducible
    pv2 = tmp_path / "pv2.tsv.gz"
    LY.run(tables, ev, tmp_path / "x.tsv", strata_path=tmp_path / "none.h5", n_boot=50, pairs_out=pv2)
    assert pv.read_bytes() == pv2.read_bytes()


# ---------------------------------------------------------------------------------------------------
# mixed_model.py --second-experiments

def _meas(mid, exp, cell, in_eic, donor, lab, unit="pooled"):
    return dict(measurement_id=mid, experiment=exp, assay="A", assay_class="narrow_histone", cell=cell,
                in_eic=in_eic, unit=unit, bio_rep="", arm_knob="base", layer="uniform", depth_reads="1000",
                depth_fragments="", donor=donor, lab=lab, counts_path=f"h5:/x/{mid}/counts.h5",
                has_pseudoreps="1")


MEAS = ([_meas(f"e{c}", f"E{c}", f"C{c}", "train", f"D{c}", "L1") for c in range(4)]
        + [_meas(f"s{c}", f"S{c}", f"C{c}", "evalonly", f"D{c}" if c % 2 else f"X{c}", "L2") for c in range(3)])


def test_second_experiments_switch():
    off, _ = MM.select_observations(MEAS, "A")
    off2, _ = MM.select_observations(MEAS, "A", None, (), False)
    on, _ = MM.select_observations(MEAS, "A", None, (), True)
    assert off == off2 and {o["experiment"] for o in off} == {"E0", "E1", "E2", "E3"}
    assert {o["experiment"] for o in on} == {"E0", "E1", "E2", "E3", "S0", "S1", "S2"}


def test_design_rows_count_contrasts_inside_cells():
    no = {r["factor"]: r for r in LD.design_rows(MEAS, "A", False)}
    se = {r["factor"]: r for r in LD.design_rows(MEAS, "A", True)}
    assert no["experiment"]["estimable"] == 0 and "confounded with cell" in no["experiment"]["note"]
    assert no["lab"]["estimable"] == 0                         # one lab
    assert se["experiment"]["estimable"] == 1 and se["experiment"]["cells_with_contrast"] == 3
    assert se["replicate"]["cells_with_contrast"] == 2         # C0, C2 got another donor
    assert se["lab"]["estimable"] == 1 and se["lab"]["cells_with_contrast"] == 3


# ---------------------------------------------------------------------------------------------------
# pairlist

def _p(pid, rung, cell, split="train", a="m", b="n"):
    return dict(pair_id=pid, rung=rung, cell=cell, split=split, a=a, b=b, assay="A")


def test_pairlist_adds_pseudoreplicates_drops_one_cell_sibling_and_never_blind():
    rows = ([_p(f"ps{i}", "pseudoreplicate", f"C{i}", a="m", b="m") for i in range(3)]
            + [_p("psb", "pseudoreplicate", "C9", "blind", a="m", b="m")]
            + [_p(f"an{i}", "anisogenic", "C9") for i in range(4)]
            + [_p(f"oe{i}", "other_experiment", f"C{i}") for i in range(3)]
            + [_p("iso", "isogenic", "C1")])
    ev = {"replicate_pairs": ["iso", "an0", "an1", "an2", "an3", "oe0", "oe1", "oe2"]}
    ids, changes = LD.build_pairlist(ev, rows)
    assert ids == ["iso", "oe0", "oe1", "oe2", "ps0", "ps1", "ps2"]
    assert {c["pair_id"] for c in changes if c["role"] == "dropped"} == {"an0", "an1", "an2", "an3"}
    with pytest.raises(ValueError):
        LD.build_pairlist({"replicate_pairs": ["psb"]}, rows)


# ---------------------------------------------------------------------------------------------------
# report

def _pv(rows, cls, assay, rung, cell, value, pc="", n=1):
    for i in range(n):
        rows.append(dict(pair_id=f"{assay}{rung}{pc}{cell}{i}", assay=assay, assay_class=cls, rung=rung,
                         pair_class=pc, tech_diff="0", cell=cell, split="train", depth="1.0", stratum="all",
                         band="all", var_d=repr(2.0 * value)))


def _planted():
    """Two assays with different baselines (A 1.0, B 0.2). Every cell has isogenic pairs in A and B; the
    lab-with-donor pairs exist in assay A only, 0.05 above A's isogenic value. The pure-lab class has 2 cells."""
    rows = []
    for c in ("C1", "C2", "C3", "C4"):
        _pv(rows, "broad_histone", "A", "pseudoreplicate", c, 0.9)
        _pv(rows, "broad_histone", "B", "pseudoreplicate", c, 0.15)
        _pv(rows, "broad_histone", "A", "isogenic", c, 1.0)
        _pv(rows, "broad_histone", "B", "isogenic", c, 0.2)
        _pv(rows, "broad_histone", "A", "other_experiment", c, 1.05, "experiment;lab;donor", n=2)
        _pv(rows, "broad_histone", "A", "other_cell_type", c, 2.0)
    for c in ("C1", "C2"):
        _pv(rows, "broad_histone", "A", "other_experiment", c, 1.02, "experiment;lab", n=4)
    _pv(rows, "narrow_histone", "H", "pseudoreplicate", "C1", 0.3)
    return rows


def test_report_assay_matched_flags_and_no_cross_class_pool():
    rows = LD.ladder_rows(_planted(), n_boot=200)
    bh = {(r["rung"], r["pair_class"]): r for r in rows if r["assay_class"] == "broad_histone"}
    ld = bh[("other_experiment", "experiment;lab;donor")]
    assert ld["flag"] == "ok" and ld["layer"] == "lab with donor"
    assert ld["layer_increment"] == pytest.approx(0.05, abs=1e-12)   # matched on assay A
    # the cell means over all assays (the pooled-by-class form) would give 1.05 - 0.6 = 0.45
    assert (ld["n_pairs"], ld["n_cells"]) == (8, 4)
    lab = bh[("other_experiment", "experiment;lab")]
    assert lab["flag"].startswith("not estimable: 8 pairs from 2 cells")
    assert not np.isfinite(lab["layer_increment"]) and lab["layer"] == "lab"
    assert bh[("isogenic", "")]["layer_increment"] == pytest.approx(0.075)  # (0.1 + 0.05) / 2
    assert bh[("sequencing", "")]["flag"] == LD.REASON_UNBUILT
    assert bh[("anisogenic", "")]["flag"] == LD.REASON_EMPTY
    assert ld["share"] == pytest.approx(0.05 / 2.0)
    assert {r["assay_class"] for r in rows} == {"broad_histone", "narrow_histone"}
    assert len(rows) == 2 * len(LD.LADDER)


def test_report_cli_writes_tables(tmp_path):
    pv = tmp_path / "pv.tsv.gz"
    with gzip.open(pv, "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LY.PAIR_COLUMNS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(_planted())
    assert LD.main(["report", "--pairs", str(pv), "--out-dir", str(tmp_path / "o"), "--n-boot", "50"]) == 0
    md = (tmp_path / "o" / "ladder.md").read_text()
    assert "## broad_histone" in md and "not estimable" in md and "lab with donor" in md
    assert len(LD._read_tsv(tmp_path / "o" / "ladder.tsv")) == 2 * len(LD.LADDER)


def test_pairlist_cli(tmp_path):
    rows = [_p("ps0", "pseudoreplicate", "C0", a="m", b="m"), _p("iso", "isogenic", "C0")]
    with open(tmp_path / "pairs.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    ev = tmp_path / "ev.json"
    ev.write_text(json.dumps({"replicate_pairs": ["iso"], "regions": []}))
    assert LD.main(["pairlist", "--evalset", str(ev), "--tables", str(tmp_path), "--out",
                    str(tmp_path / "pl.json"), "--changes", str(tmp_path / "ch.tsv")]) == 0
    out = json.loads((tmp_path / "pl.json").read_text())
    assert out["replicate_pairs"] == ["iso", "ps0"] and out["layers_pair_list"]["added"] == 1


def test_pairlist_assay_filter_for_smoke():
    rows = [_p("ps0", "pseudoreplicate", "C0", a="m", b="m"), dict(_p("ps1", "pseudoreplicate", "C0", a="k", b="k"),
                                                                  assay="B")]
    ids, _ = LD.build_pairlist({"replicate_pairs": []}, rows, assays=["B"])
    assert ids == ["ps1"]


def test_mixed_model_assays_filter(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(MM, "select_observations", lambda meas, a, *x: (seen.append(a) or ([], "uniform")))
    meas = [dict(_meas("e0", "E0", "C0", "train", "D0", "L1")), dict(_meas("e1", "E1", "C1", "train", "D1", "L1"),
                                                                  assay="B")]
    with open(tmp_path / "measurements.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(meas[0]), delimiter="\t")
        w.writeheader()
        w.writerows(meas)
    ev = tmp_path / "ev.json"
    ev.write_text(json.dumps({"regions": []}))
    MM.run(tmp_path, ev, "fine", tmp_path / "mm.tsv", assays=["B"])
    assert seen == ["B"]
    seen.clear()
    MM.run(tmp_path, ev, "fine", tmp_path / "mm.tsv")
    assert seen == ["A", "B"]


# ---------------------------------------------------------------------------------------------------
# review fixes: quote rule, ladder telescoping, matched share, negative flag

def test_mixed_model_quote_rule_blanks_factor_without_contrast_in_three_cells():
    labels = {"cell": ["C1", "C1", "C2", "C2", "C3", "C3"],
              "experiment": ["E1", "F1", "E2", "F2", "E3", "E3"],    # varies inside C1, C2: 2 cells
              "library": ["a", "b", "c", "d", "e", "f"]}             # varies inside 3 cells
    rows = [dict(factor="cell", variance_fraction=0.3, ci_low=0.1, ci_high=0.5, n_levels=3, estimable=1, note=""),
            dict(factor="experiment", variance_fraction=0.2, ci_low=0.0, ci_high=0.4, n_levels=5, estimable=1,
                 note=""),
            dict(factor="library", variance_fraction=1e-9, ci_low=0.0, ci_high=1e-8, n_levels=6, estimable=1,
                 note=""),
            dict(factor="residual", variance_fraction=0.5, ci_low=0.4, ci_high=0.6, n_levels=6, estimable=1,
                 note="REML")]
    out = {r["factor"]: r for r in MM.quote_rule(rows, labels, 3)}
    assert np.isnan(out["experiment"]["variance_fraction"]) and np.isnan(out["experiment"]["ci_low"])
    assert out["experiment"]["note"].startswith(MM.NOT_IDENTIFIED + ": varies inside 2 cells")
    assert out["library"]["variance_fraction"] == 1e-9 and "boundary" in out["library"]["note"]
    assert out["cell"]["variance_fraction"] == 0.3 and out["residual"]["note"] == "REML"
    assert rows[1]["variance_fraction"] == 0.2                       # input untouched


def test_report_quote_rule_from_design():
    mixed = [dict(assay="A", band="fine", factor="lab", variance_fraction="0.002", ci_low="0", ci_high="0.05",
                  n_levels="4", estimable="1", note=""),
             dict(assay="A", band="fine", factor="experiment", variance_fraction="0.01", ci_low="0", ci_high="0.02",
                  n_levels="9", estimable="1", note=""),
             dict(assay="A", band="fine", factor="residual", variance_fraction="0.9", ci_low="0.8", ci_high="1",
                  n_levels="20", estimable="1", note="REML; second experiments included")]
    design = [dict(assay="A", second_experiments="1", factor="lab", cells_with_contrast="2", n_obs="20", n_cells="6", estimable="1"),
              dict(assay="A", second_experiments="1", factor="experiment", cells_with_contrast="5", n_obs="20", n_cells="6", estimable="1"),
              dict(assay="A", second_experiments="0", factor="experiment", cells_with_contrast="0", n_obs="10", n_cells="6", estimable="0")]
    out = {m["factor"]: m for m in LD.quote_rule(mixed, design)}
    assert out["lab"]["variance_fraction"] == "" and LD.NOT_IDENTIFIED in out["lab"]["note"]
    assert out["experiment"]["variance_fraction"] == "0.01"
    md = LD.markdown([], mixed=list(out.values()), design=design)
    assert "| A | fine | lab |  |" in md


def test_ladder_chain_check_telescopes_on_common_units():
    rows = _planted()
    _pv(rows, "broad_histone", "B", "other_cell_type", "C1", 0.4)
    got = [c for c in LD.chain_check(rows) if c["assay_class"] == "broad_histone"][0]
    assert got["ok"] is True and got["n_units"] == 5 and got["n_cells"] == 4
    assert got["sum_of_increments"] == pytest.approx(got["other_cell_type_value"], abs=1e-12)


def test_share_is_matched_and_negative_increment_flagged():
    rows = _planted()
    for c in ("C1", "C2", "C3", "C4"):
        _pv(rows, "broad_histone", "B", "other_cell_type", c, 0.4)   # would lower a class-wide denominator
        _pv(rows, "broad_histone", "A", "other_experiment", c, 0.98, "experiment;donor", n=2)
    bh = {(r["rung"], r["pair_class"]): r for r in LD.ladder_rows(rows, n_boot=100)
          if r["assay_class"] == "broad_histone"}
    ld = bh[("other_experiment", "experiment;lab;donor")]
    assert ld["share"] == pytest.approx(0.05 / 2.0) and ld["share_units"] == 4
    neg = bh[("other_experiment", "experiment;donor")]
    assert neg["layer_increment"] == pytest.approx(-0.02) and "negative increment" in neg["flag"]
