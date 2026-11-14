"""OXr: metric 7 with two bounds, the add-on score and the explained fraction in the arm report.

Hand-written scores.json fixtures (no model, no data). The digests pin that, on fixtures without an
add-on score, every output of the base commit 5bfad89 is unchanged (headline: the base columns)."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from onion_eval import arm_report as AR  # noqa: E402

SHA, XSHA = "E" * 64, "X" * 64
H7 = AR.H7
UB = f"{H7}.per_rung.other_experiment"
PR = f"{H7}.per_rung.pseudoreplicate"
PASS = f"{H7}.pass_rate"
NB = f"{H7}.n_targets_with_both_bounds"

# sha256 of each output on the frozen fixture, computed with tools/onion_eval/arm_report.py at 5bfad89
BASE_DIGESTS = {
    "headline": "fb3a94ab3f6a1b01a88490ede2385055294330dc36dfcfe0f7529ae6da49b161",
    "dev-evals": "9df9361452084b408fe19052a28da554923f3f92bfdc1805e4e82f2d75c2292d",
    "accessibility": "b6c417d4270284e0f0ba2b37689f8945918ac88624c960d58736491ff4e37dd8",
    "per-knob": "6db1d42db31ac0385aa5f92d72eb547f4c00e4ad3d56d0e8f571db59565eccf2",
    "shared-specific": "3176d93f389e089a105d9dd3de4a01ace100cba7c95b08aece00981701452392",
}

PATHS_1 = ("count_crps", "count_crps_oracle_scaled", "count_crps_scale_error",
           "correction_gap_closed_vs_depth_only_never_trained", "correction_crps_model",
           "correction_crps_model_oracle_scaled", "correction_crps_depth_only",
           "correction_crps_pseudoreplicate_floor", "subset_pairs_gap_closed",
           "depth_law_abs_log2_error_never_trained", "pit_deviation_pooled",
           "pit_deviation_worst_stratum", "count_crps_skill_vs_position_average")
SPACES = ("counts_log1p", "signal_arcsinh", "signal_arcsinh_plug_in")


def m(macro, per=None, non=None):
    return {"macro": macro, "per_assay": per or {}, "non_scorable_per_assay": non or {},
            "ci": [None, None]}


def h7_block(lo, up=None, pass_rate=None, per_target=None):
    per_rung = {"pseudoreplicate": m(lo, {"H3K4me3": lo})}
    if up is not None:
        per_rung["other_experiment"] = m(up, {"H3K4me3": up})
    blk = m(lo, {"H3K4me3": lo})
    blk.update(per_rung=per_rung, pass_rate=m(pass_rate), per_target=per_target or {},
               upper_bound_rung={"imputation": "other_experiment"})
    return blk


def frozen(v, sha=SHA):
    """A frozen-evalset scores.json; written explicitly so the fixture does not follow AR.HEADLINE."""
    s = {"evalset_sha256": sha, "tables_sha256": "T" * 64, "split": "val",
         "regime": "leave_one_assay_out"}
    for k in PATHS_1:
        s[k] = m(v, {"H3K4me3": v})
    for k in ("specific_variance_recovered_above_noise", "r_spec"):
        s[k] = {sp: m(v, {"H3K4me3": v}) for sp in SPACES}
    s["correction_crps_model_oracle_scaled"] = m(v - 0.1)
    s["correction_gap_closed_vs_depth_only_never_trained"]["per_knob"] = {"crop36": v, "se": 2 * v}
    s[H7] = h7_block(v + 1.0, per_target={"t_eic": {"per_rung": {"pseudoreplicate": v + 1.0},
                                                    "pass": None, "reason": "no upper rung"}})
    b = {k: v for k in AR.BUDGET_KEYS}
    b["per_assay"] = {"H3K4me3": {k: v + 1 for k in AR.BUDGET_KEYS}}
    s["error_budget"] = b
    a = {"regime": "accessibility_only"}
    for path in AR.IMPUTATION:
        blk = m(v / 2, {"H3K4me3": v / 2}, {"ATAC-seq": v / 4})
        if len(path) == 2:
            a.setdefault(path[0], {})[path[1]] = blk
        else:
            a[path[0]] = blk
    s["accessibility_only"] = a
    return s


def ef_rows(x, flag="ok"):
    return [{"rung": "other_experiment", "layer": "experiment", "increment_over": "isogenic",
             "explained_fraction": x, "ci_low": x - 0.1, "ci_high": x + 0.1, "captured": x / 2,
             "layer_term": 0.5, "n_pairs": 4, "n_cells": 3, "flag": flag}]


def addon(lo, up, pass_rate, per_target, sha=XSHA, ef=0.4):
    s = {"evalset_sha256": sha, "tables_sha256": "U" * 64, "split": "val"}
    s[H7] = h7_block(lo, up, pass_rate, per_target)
    s["explained_fraction_per_layer"] = {"per_assay": {
        "H3K4me3": {"rows": ef_rows(ef), "reason": ""},
        "DNase-seq": {"rows": [], "reason": "no other_experiment or other_cell_type pair"}}}
    return s


def put(d: Path, s: dict):
    d.mkdir(parents=True, exist_ok=True)
    (d / "scores.json").write_text(json.dumps(s), encoding="utf-8")


def make_tree(root: Path):
    runs, base = root / "runs", root / "baselines"
    for k, v in enumerate((0.2, 0.3, 0.4)):
        put(runs / "rung1_base" / f"seed{k}", frozen(v))
        put(runs / "rung1_dna_off" / f"seed{k}", frozen(v + 0.5))
    put(base / "marginal_nb" / "seed0", frozen(0.1))
    put(base / "position_average" / "seed0", frozen(0.25))
    return runs, base


def pt(lo, up, ok):
    return {"per_rung": {"pseudoreplicate": lo, "other_experiment": up}, "pass": ok, "reason": ""}


def add_addon(runs, base):
    """Add-on scores: rung1_base has all three seeds, rung1_dna_off none; marginal_nb has one."""
    for k, up in enumerate((0.8, 0.9, 1.0)):
        put(runs / "rung1_base" / f"seed{k}" / "ox", addon(
            1.2, up, 0.5, {"t_lab": pt(1.2, up, 1.0), "t_rep": pt(1.1, 1.3, 0.0),
                           "s_lab": pt(1.0, 0.7, 1.0), "t_none": pt(1.2, None, None)},
            ef=0.3 + 0.1 * k))
    put(base / "marginal_nb" / "seed0" / "ox", addon(
        1.0, 1.5, 0.25, {"t_lab": pt(1.0, 1.5, 0.0)}, ef=0.0))


def write_pairs(p: Path):
    rows = [("p1", "t_lab", "s_lab", "other_experiment", "experiment;lab;read_length"),
            ("p2", "t_rep", "s_rep", "other_experiment", "experiment"),
            ("p3", "t_lab", "t_lab_pr2", "pseudoreplicate", "")]
    p.write_text("pair_id\ta\tb\trung\tassay\tcell\tknobs_differing\tsubset_pair\tsplit\tnever_trained\n"
                 + "".join(f"{i}\t{a}\t{b}\t{r}\tH3K4me3\tC01\t{k}\t0\tval\t0\n"
                           for i, a, b, r, k in rows), encoding="utf-8")
    return p


def read(p):
    with open(p, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def f(x):
    return float(x) if x not in ("", None) else None


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def outputs(mod, runs, base, out: Path, base_cols=None) -> dict:
    """{command: sha256 of its TSV}; headline projected on `base_cols` when given."""
    got = {}
    for cmd in ("headline", "dev-evals", "accessibility", "per-knob"):
        p = out / f"{cmd}.tsv"
        assert mod.main([cmd, "--runs", str(runs), "--baselines", str(base), "--out", str(p)]) == 0
        text = p.read_text(encoding="utf-8")
        if cmd == "headline" and base_cols is not None:
            rows = list(csv.DictReader(text.splitlines(), delimiter="\t"))
            text = "\t".join(base_cols) + "\n" + "".join(
                "\t".join(r[c] for c in base_cols) + "\n" for r in rows)
        got[cmd] = digest_text(text)
    p = out / "ss.tsv"
    assert mod.main(["shared-specific", "--runs", str(runs), "--baselines", str(base),
                     "--arms", "rung1_base,rung1_dna_off,marginal_nb", "--out", str(p)]) == 0
    got["shared-specific"] = digest_text(p.read_text(encoding="utf-8"))
    return got


def base_headline_columns():
    cols = ["arm", "kind", "n_seeds"]
    for path, _ in AR.HEADLINE:
        q = ".".join(path)
        cols += [f"{q}.mean", f"{q}.spread", f"{q}.n_defined", f"{q}.reference",
                 f"{q}.reference_source"]
    return cols


# ------------------------------------------------------------------------------------------ tests

def test_without_addon_outputs_match_base_commit(tmp_path):
    runs, base = make_tree(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    assert outputs(AR, runs, base, out, base_headline_columns()) == BASE_DIGESTS


def test_without_addon_three_columns_not_populated(tmp_path):
    runs, base = make_tree(tmp_path)
    rows, cols = AR.headline(runs, "", base)
    for c in (PR, UB, PASS, NB):
        assert f"{c}.mean" in cols and f"{c}.target" in cols
    r = {x["arm"]: x for x in rows}["rung1_base"]
    for c in (UB, PASS, NB):
        assert r[f"{c}.mean"] == r[f"{c}.spread"] == r[f"{c}.n_defined"] == AR.NOT_POPULATED
    assert r[f"{PR}.mean"] == pytest.approx(1.3) and r[f"{PR}.spread"] == pytest.approx(0.2)
    assert (r[f"{PR}.target"], r[f"{UB}.target"], r[f"{PASS}.target"]) == (">= 1", "<= 1", "high")
    assert r[f"{H7}.target"] == AR.H7_OLD_TARGET
    assert r[f"{H7}.mean"] == pytest.approx(1.3)                      # the old scalar is kept


def test_addon_bounds_pass_rate_and_count(tmp_path):
    runs, base = make_tree(tmp_path)
    add_addon(runs, base)
    out = tmp_path / "h.tsv"
    assert AR.main(["headline", "--runs", str(runs), "--baselines", str(base), "--out", str(out)]) == 0
    rows = {r["arm"]: r for r in read(out)}
    r = rows["rung1_base"]
    assert f(r[f"{UB}.mean"]) == pytest.approx(0.9) and f(r[f"{UB}.spread"]) == pytest.approx(0.2)
    assert f(r[f"{PASS}.mean"]) == pytest.approx(0.5) and r[f"{PASS}.n_defined"] == "3"
    assert f(r[f"{NB}.mean"]) == 3.0                                  # t_none has no upper bound
    assert f(r[f"{PR}.mean"]) == pytest.approx(1.3)                   # read from the frozen score
    assert r[f"{PASS}.reference_source"] == "marginal_nb"
    assert f(r[f"{PASS}.reference"]) == pytest.approx(0.25)
    assert rows["rung1_dna_off"][f"{UB}.mean"] == AR.NOT_POPULATED
    b = rows["marginal_nb"]
    assert f(b[f"{UB}.mean"]) == pytest.approx(1.5) and f(b[f"{UB}.spread"]) == 0.0


def test_addon_missing_for_one_seed_gives_empty_mean(tmp_path):
    runs, base = make_tree(tmp_path)
    add_addon(runs, base)
    (runs / "rung1_base" / "seed1" / "ox" / "scores.json").unlink()
    rows, _ = AR.headline(runs, "rung1_base", None)
    r = rows[0]
    assert r[f"{UB}.mean"] is None and r[f"{UB}.n_defined"] == 2


def test_addon_evalset_may_differ_from_frozen_but_not_among_addons(tmp_path):
    runs, base = make_tree(tmp_path)
    add_addon(runs, base)
    AR.headline(runs, "", base)                                       # XSHA != SHA: accepted
    put(runs / "rung1_base" / "seed2" / "ox", addon(1.2, 1.0, 0.5, {}, sha="Y" * 64))
    for cmd in ("headline", "explained"):
        out = tmp_path / f"{cmd}.tsv"
        assert AR.main([cmd, "--runs", str(runs), "--baselines", str(base), "--out", str(out)]) == 2
        assert not out.exists()
    put(runs / "rung1_base" / "seed2" / "ox", addon(1.2, 1.0, 0.5, {}, sha=XSHA))
    put(base / "marginal_nb" / "seed0" / "ox", addon(1.0, 1.5, 0.25, {}, sha=""))
    with pytest.raises(AR.Refused):
        AR.headline(runs, "", base)


def test_addon_subdir_option(tmp_path):
    runs, base = make_tree(tmp_path)
    put(runs / "rung1_base" / "seed0" / "other", addon(1.2, 0.6, 1.0, {}))
    rows, _ = AR.headline(runs, "rung1_base", None, addon="other")
    assert rows[0][f"{UB}.n_defined"] == 1 and rows[0][f"{UB}.mean"] is None
    rows, _ = AR.headline(runs, "rung1_base", None)
    assert rows[0][f"{UB}.mean"] == AR.NOT_POPULATED


def test_pairs_split_by_class_and_second_experiment_left_out(tmp_path):
    runs, base = make_tree(tmp_path)
    add_addon(runs, base)
    pairs = write_pairs(tmp_path / "pairs.tsv")
    assert AR.pair_class("experiment;lab;read_length;platform") == "experiment;lab"
    assert AR.pair_class("experiment;donor?;platform") == "experiment;donor?"
    rows, cols = AR.headline(runs, "rung1_", base, pairs=str(pairs))
    r = {x["arm"]: x for x in rows}["rung1_base"]
    lab, rep = f"{H7}.by_class.experiment;lab", f"{H7}.by_class.experiment"
    assert r[f"{lab}.upper_bound.mean"] == pytest.approx(0.9)        # s_lab (b side) is left out
    assert r[f"{lab}.upper_bound.spread"] == pytest.approx(0.2)
    assert r[f"{lab}.pass_rate.mean"] == pytest.approx(1.0)
    assert r[f"{lab}.n_targets.mean"] == 1.0
    assert r[f"{rep}.upper_bound.mean"] == pytest.approx(1.3)
    assert r[f"{rep}.pass_rate.mean"] == pytest.approx(0.0)
    assert r[f"{NB}.mean"] == 2.0                                     # t_lab, t_rep (s_lab dropped)
    assert {x["arm"]: x for x in rows}["rung1_dna_off"][f"{lab}.upper_bound.mean"] == AR.NOT_POPULATED
    assert AR.main(["headline", "--runs", str(runs), "--baselines", str(base), "--out",
                    str(tmp_path / "h.tsv"), "--pairs", str(tmp_path / "nope.tsv")]) == 2


def test_class_mean_undefined_when_a_target_is_undefined(tmp_path):
    runs, base = make_tree(tmp_path)
    put(runs / "rung1_base" / "seed0" / "ox", addon(
        1.2, 0.8, 0.5, {"t_lab": pt(1.2, None, None), "t_rep": pt(1.1, 1.3, 0.0)}))
    pairs = write_pairs(tmp_path / "pairs.tsv")
    rows, _ = AR.headline(runs, "rung1_base", None, pairs=str(pairs))
    lab = f"{H7}.by_class.experiment;lab"
    assert rows[0][f"{lab}.upper_bound.n_defined"] == 0
    assert rows[0][f"{lab}.n_targets.n_defined"] == 1


def test_explained_one_row_per_arm_and_layer(tmp_path):
    runs, base = make_tree(tmp_path)
    add_addon(runs, base)
    out = tmp_path / "e.tsv"
    assert AR.main(["explained", "--runs", str(runs), "--prefix", "rung1_", "--baselines", str(base),
                    "--out", str(out)]) == 0
    rows = read(out)
    by = {(r["arm"], r["assay"], r["layer"]): r for r in rows}
    r = by[("rung1_base", "H3K4me3", "experiment")]
    assert r["rung"] == "other_experiment" and r["increment_over"] == "isogenic"
    assert f(r["explained_fraction.mean"]) == pytest.approx(0.4)
    assert f(r["explained_fraction.spread"]) == pytest.approx(0.2)
    assert r["n_seeds"] == "3" and r["n_pairs"] == "4" and r["flag"] == "ok"
    d = by[("rung1_base", "DNase-seq", "")]
    assert d["flag"] == "no other_experiment or other_cell_type pair"
    assert by[("rung1_dna_off", "", "")]["flag"] == AR.NOT_POPULATED
    assert f(by[("marginal_nb", "H3K4me3", "experiment")]["explained_fraction.mean"]) == 0.0


def test_explained_keeps_pair_class_rows(tmp_path):
    runs, base = make_tree(tmp_path)
    s = addon(1.2, 0.8, 0.5, {})
    rows = ef_rows(0.4) + [dict(ef_rows(0.6)[0], pair_class="experiment;lab")]
    s["explained_fraction_per_layer"]["per_assay"]["H3K4me3"]["rows"] = rows
    put(runs / "rung1_base" / "seed0" / "ox", s)
    got, _ = AR.explained(runs, "rung1_base")
    vals = {(r["pair_class"]): r["explained_fraction.mean"] for r in got if r.get("layer")}
    assert vals == {"": pytest.approx(0.4), "experiment;lab": pytest.approx(0.6)}


def test_keys_exist_in_scorer():
    src = (REPO / "tools" / "onion_eval" / "score.py").read_text()
    for k in ("per_rung", "pass_rate", "per_target", "explained_fraction_per_layer", "rows",
              "pseudoreplicate", "other_experiment"):
        assert f'"{k}"' in src, k
    lsrc = (REPO / "tools" / "onion_eval" / "layers.py").read_text()
    for k in ("explained_fraction", "captured", "layer_term", "n_pairs", "n_cells", "flag",
              "increment_over"):
        assert f"{k}=" in lsrc, k
