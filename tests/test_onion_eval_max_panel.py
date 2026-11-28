"""C17: tools/onion_eval/max_panel.py, the PI's metric panel (D15, D23)."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from onion_eval import max_panel as MP  # noqa: E402
from onion_eval import metrics as Mx  # noqa: E402
from onion_eval import select_rule as SR  # noqa: E402

M = 2048


def _tracks(rng, cells, noise):
    """truth and prediction in log1p space for targets in `cells` (one assay)."""
    base = np.log1p(rng.poisson(3.0, M).astype(float))
    T = np.stack([base + rng.normal(0, 0.6, M) for _ in cells])
    P = 0.7 * T + 0.4 + rng.normal(0, noise, T.shape)
    return T, P


def _r2_direct(T, P):
    return [1 - np.sum((t - p) ** 2) / np.sum((t - t.mean()) ** 2) for t, p in zip(T, P)]


def _write_score(d: Path, assays: dict, *, evalset="ev1", foreign=None, imp=0.4, unc=0.6,
                 r_spec=None):
    """A score dir as score.py writes it: per_target level and the Gram suffstats."""
    d.mkdir(parents=True, exist_ok=True)
    per, suff, direct = {}, {}, []
    for a, (T, P, cells) in assays.items():
        ss = Mx.shared_specific(T, P, np.array(cells))
        for i, c in enumerate(cells):
            per[f"{a}.{c}.{i}"] = {"assay": a, "cell": c, "level": float(ss["level"][i])}
        for nm in ("V", "P", "C", "E"):
            for part in ("sh", "sp"):
                suff[f"gram/{MP.REGIME}/{a}/{nm}_{part}_by_level"] = ss["by_level"][f"{nm}_{part}"]
        direct += _r2_direct(T, P)
    s = {"selection": {"imputation_gap_closed": imp, "correction_gap_closed": 0.5,
                       "uncertainty_gap_closed": unc,
                       "denominators": {"uncertainty_marginal_pit_deviation": 1.0,
                                        "correction_baseline_minus_floor_relative_min": 1.0}},
         "copy_leak": {"copy_leak_pass": True, "part1_wiring_pass": True},
         "evalset_sha256": evalset, "tables_sha256": "tb", "split": "val",
         "regime": MP.REGIME, "scorable": {"imputation": sorted(assays)},
         MP.H1_KEY: {MP.SPACE: {"per_target": per}},
         "r_spec": {MP.SPACE: {"per_assay": r_spec or {a: 0.5 for a in assays}}}}
    if foreign:
        s["foreign_covariates"] = {"mode": foreign, "seed": 0,
                                   "n_changed_by_column": {"out": {"lab_id": 3}}}
    (d / "scores.json").write_text(json.dumps(s), encoding="utf-8")
    np.savez_compressed(d / "suffstats.npz", **suff)
    (d / "SCORED").write_text("")
    return float(np.mean(direct))


def _assays(seed, noise=0.5):
    rng = np.random.default_rng(seed)
    a = _tracks(rng, ["c1", "c2", "c3"], noise)
    b = _tracks(rng, ["c1", "c4"], noise)
    return {"H3K4me3": (*a, ["c1", "c2", "c3"]), "DNase-seq": (*b, ["c1", "c4"])}


def test_r2_from_suffstats_equals_direct_sse_over_sst(tmp_path):
    want = _write_score(tmp_path / "s", _assays(0))
    got = MP.r2_from_score(tmp_path / "s")
    assert got["n_targets"] == 5 and got["n_targets_not_derivable"] == 0
    assert got["r2"] == pytest.approx(want, abs=1e-9)


def test_single_cell_assay_is_not_derivable_and_counted(tmp_path):
    rng = np.random.default_rng(1)
    asy = _assays(1)
    T, P = _tracks(rng, ["c9"], 0.5)
    asy["CTCF"] = (T, P, ["c9"])
    _write_score(tmp_path / "s", asy)
    got = MP.r2_from_score(tmp_path / "s")
    assert got["n_targets"] == 5 and got["n_targets_not_derivable"] == 1
    assert "CTCF" in got["assays_not_derivable"]
    want = np.mean(sum((_r2_direct(t, p) for a, (t, p, _) in asy.items() if a != "CTCF"), []))
    assert got["r2"] == pytest.approx(want, abs=1e-9)
    assert got["r_spec_sq"] == pytest.approx(0.25)


def _arm(root: Path, name, sub, noise, imp, **kw):
    vals = []
    for k in range(3):
        vals.append(_write_score(root / name / f"seed{k}" / sub, _assays(10 * k, noise),
                                 imp=imp + 0.01 * k, **kw))
    return vals


def _read(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def test_panel_rows_differences_and_cli(tmp_path):
    runs = tmp_path / "runs"
    r2_rec = _arm(runs, "rcp_film", "p256_sv", 0.3, 0.45)
    _arm(runs, "rcp_film_x", "p256_sv", 0.3, 0.9)               # a prefix sibling is ignored
    r2_nc = _arm(runs, "rcp_nocov", "p256_sv", 0.8, 0.40)
    _arm(runs, "rcp_film", "p16_sv", 0.3, 0.50)
    r2_fo = _arm(runs, "rcp_film", "p16_foreign_out_keep_depth", 0.9, 0.30,
                 foreign="out_keep_depth")
    out = tmp_path / "o" / "panel.tsv"
    assert MP.main(["--recipe", str(runs / "rcp_film"), "--nocov", str(runs / "rcp_nocov"),
                    "--foreign-sub", "p16_foreign_out_keep_depth", "--sub", "p256_sv",
                    "--out", str(out)]) == 0
    rows = _read(out)
    assert list(rows[0]) == list(MP.COLUMNS)
    by = {(r["panel_row"], r["quantity"]): r for r in rows}
    r1 = by[("r2_test_tracks", MP.R2)]
    assert float(r1["value"]) == pytest.approx(np.mean(r2_rec), rel=1e-5)
    assert float(r1["seed_range"]) == pytest.approx(max(r2_rec) - min(r2_rec), rel=1e-4)
    assert r1["n_seeds"] == "3" and r1["sub"] == "p256_sv" and r1["n_targets"] == "5"
    g = by[("covariates_gain", MP.R2)]
    assert float(g["value"]) == pytest.approx(np.mean(r2_rec) - np.mean(r2_nc), rel=1e-4)
    rng_ = max(max(r2_rec) - min(r2_rec), max(r2_nc) - min(r2_nc))
    assert float(g["seed_range"]) == pytest.approx(rng_, rel=1e-4)
    gi = by[("covariates_gain", "imputation_raw")]
    assert float(gi["value"]) == pytest.approx(0.05, abs=1e-9)
    assert gi["n_seeds_ref"] == "3" and gi["arm_ref"] == "rcp_nocov"
    assert by[("covariates_gain", "correction_v3_raw")]["value"] == ""   # no oracle reference
    f = by[("real_vs_foreign", "imputation_raw")]
    assert f["sub"] == "p16_sv" and f["sub_ref"] == "p16_foreign_out_keep_depth"
    assert float(f["value"]) == pytest.approx(0.20, abs=1e-9)
    assert json.loads(f["n_changed_by_column"]) == {"out": {"lab_id": 3}}
    assert float(by[("real_vs_foreign", MP.R2)]["value"]) > 0 > np.mean(r2_fo) - np.mean(r2_rec)
    assert MP.FALLBACK not in {r["quantity"] for r in rows}


def test_not_comparable_and_fallback_rows(tmp_path):
    runs = tmp_path / "runs"
    _arm(runs, "a", "p256_sv", 0.3, 0.4)
    for k in range(3):
        rng = np.random.default_rng(k)
        T, P = _tracks(rng, ["c9"], 0.5)
        _write_score(runs / "n" / f"seed{k}" / "p256_sv", {"CTCF": (T, P, ["c9"])}, evalset="ev2")
    rows = MP.panel(runs / "a", "p256_sv", nocov=runs / "n")
    g = [r for r in rows if r["panel_row"] == "covariates_gain"]
    assert all(r.get("value") is None and "evalset_sha256 differs" in r["note"] for r in g)
    assert MP.FALLBACK in {r["quantity"] for r in g}
    assert {r["quantity"] for r in rows if r["panel_row"] == "r2_test_tracks"} == {MP.R2}


def test_terms_come_from_select_rule_table(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    _arm(runs, "a", "p256_sv", 0.3, 0.4)
    real = SR.table

    def fake(*a, **k):
        rows = real(*a, **k)
        return [dict(r, correction_gap_closed_oracle_no_nodedup_side=0.6 + 0.1 * r["seed"])
                for r in rows]
    monkeypatch.setattr(SR, "table", fake)
    side = MP.arm_side(runs / "a", "p256_sv")
    assert side["vals"]["correction_v3_raw"] == pytest.approx([0.6, 0.7, 0.8])
    assert MP._agg(side["vals"]["correction_v3_raw"]) == pytest.approx((0.7, 0.2))


def test_foreign_ref_sub_default():
    assert MP.foreign_ref_sub("p16_foreign_out_keep_depth", None) == "p16_sv"
    assert MP.foreign_ref_sub("p16_foreign_x", "p256_sv") == "p256_sv"
    with pytest.raises(SystemExit):
        MP.foreign_ref_sub("foreign_x", None)
