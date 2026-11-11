"""RP arm report on hand-written scores.json fixtures (no model, no data)."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from onion_eval import arm_report as AR  # noqa: E402

SHA = "E" * 64


def m(macro, per=None, non=None):
    return {"macro": macro, "per_assay": per or {}, "non_scorable_per_assay": non or {},
            "ci": [None, None]}


def budget(x):
    """An error budget whose every key is x (macro) and x + 1 for assay H3K4me3."""
    b = {k: x for k in AR.BUDGET_KEYS}
    b["per_assay"] = {"H3K4me3": {k: x + 1 for k in AR.BUDGET_KEYS}}
    return b


def scores(v, sha=SHA, knob=None, acc=None):
    """A scores.json where every headline macro is v, except the ones the tests pin."""
    s = {"evalset_sha256": sha, "tables_sha256": "T" * 64, "split": "val",
         "regime": "leave_one_assay_out"}
    for path, _ in AR.HEADLINE:
        if path[0] in AR.DERIVED_SCALE_ERROR:
            continue
        if len(path) == 2:
            s.setdefault(path[0], {})[path[1]] = m(v, {"H3K4me3": v})
        else:
            s[path[0]] = m(v, {"H3K4me3": v})
    s["correction_crps_model_oracle_scaled"] = m(v - 0.1)
    s["correction_gap_closed_vs_depth_only_never_trained"]["per_knob"] = knob or {"crop36": v,
                                                                                  "se": 2 * v}
    s["error_budget"] = budget(v)
    a = {}
    for path in AR.IMPUTATION:
        blk = m(acc if acc is not None else v / 2, {"H3K4me3": v / 2}, {"ATAC-seq": v / 4})
        if len(path) == 2:
            a.setdefault(path[0], {})[path[1]] = blk
        else:
            a[path[0]] = blk
    a["regime"] = "accessibility_only"
    s["accessibility_only"] = a
    return s


def put(d: Path, s: dict):
    d.mkdir(parents=True, exist_ok=True)
    (d / "scores.json").write_text(json.dumps(s), encoding="utf-8")


@pytest.fixture
def tree(tmp_path):
    runs, base = tmp_path / "runs", tmp_path / "baselines"
    for k, v in enumerate((0.2, 0.3, 0.4)):
        put(runs / "rung1_base" / f"seed{k}", scores(v))
        put(runs / "rung1_dna_off" / f"seed{k}", scores(v + 0.5))
        put(runs / "rung2_base" / f"seed{k}", scores(9.0))
    (runs / "settlements").mkdir()
    (runs / "settlements" / "settlements.tsv").write_text("x\n")
    put(base / "marginal_nb" / "seed0", scores(0.1))
    put(base / "position_average" / "seed0", scores(0.25))
    put(base / "depth_only_scaling" / "seed0", scores(1.2))
    return runs, base


def read(p):
    with open(p, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def f(x):
    return float(x) if x != "" else None


def test_headline_means_spreads_and_references(tree, tmp_path):
    runs, base = tree
    out = tmp_path / "h.tsv"
    assert AR.main(["headline", "--runs", str(runs), "--prefix", "rung1_", "--baselines", str(base),
                    "--out", str(out)]) == 0
    rows = {r["arm"]: r for r in read(out)}
    assert set(rows) == {"rung1_base", "rung1_dna_off", "marginal_nb", "position_average",
                         "depth_only_scaling"}
    r = rows["rung1_base"]
    assert r["kind"] == "arm" and r["n_seeds"] == "3"
    q = "specific_variance_recovered_above_noise.counts_log1p"
    assert f(r[f"{q}.mean"]) == pytest.approx(0.3)
    assert f(r[f"{q}.spread"]) == pytest.approx(0.2)
    assert r[f"{q}.reference_source"] == "depth_only_scaling"         # highest is best
    assert f(r[f"{q}.reference"]) == pytest.approx(1.2)
    assert r["count_crps.reference_source"] == "marginal_nb"          # lowest is best
    assert r["predicted_over_replicate_variance.reference_source"] == "depth_only_scaling"  # |1.2-1|
    assert r["correction_crps_model.reference_source"] == "correction_crps_pseudoreplicate_floor"
    assert f(r["correction_crps_model.reference"]) == pytest.approx(0.3)
    # every CRPS with its oracle split; the correction scale error derived as crps - oracle_scaled
    for c in ("count_crps_oracle_scaled.mean", "count_crps_scale_error.mean",
              "signal_arcsinh_plug_in", "r_spec.signal_arcsinh.mean"):
        assert any(c in k for k in r)
    assert f(r["correction_crps_model_scale_error.mean"]) == pytest.approx(0.1)
    b = rows["marginal_nb"]
    assert b["kind"] == "baseline" and f(b[f"{q}.spread"]) == 0.0


def test_headline_undefined_seed_gives_empty_mean(tree, tmp_path):
    runs, base = tree
    p = runs / "rung1_base" / "seed1" / "scores.json"
    s = json.loads(p.read_text())
    s["count_crps"]["macro"] = None
    p.write_text(json.dumps(s))
    rows, _ = AR.headline(runs, "rung1_base", base)
    r = [x for x in rows if x["arm"] == "rung1_base"][0]
    assert r["count_crps.mean"] is None and r["count_crps.n_defined"] == 2


@pytest.mark.parametrize("where", ["arm", "baseline", "empty"])
def test_mixed_evalset_refused(tree, tmp_path, where):
    runs, base = tree
    if where == "arm":
        put(runs / "rung1_base" / "seed2", scores(0.4, sha="F" * 64))
    elif where == "baseline":
        put(base / "marginal_nb" / "seed0", scores(0.1, sha="F" * 64))
    else:
        put(runs / "rung1_base" / "seed2", scores(0.4, sha=""))
    for cmd in ("headline", "dev-evals", "accessibility", "per-knob"):
        out = tmp_path / f"{cmd}.tsv"
        assert AR.main([cmd, "--runs", str(runs), "--prefix", "rung1_", "--baselines", str(base),
                        "--out", str(out)]) == 2
        assert not out.exists()
    assert AR.main(["shared-specific", "--runs", str(runs), "--baselines", str(base),
                    "--arms", "rung1_base,marginal_nb"]) == 2


def test_dev_evals_per_assay_and_macro(tree, tmp_path):
    runs, base = tree
    out = tmp_path / "d.tsv"
    assert AR.main(["dev-evals", "--runs", str(runs), "--prefix", "rung1_base",
                    "--out", str(out)]) == 0
    rows = read(out)
    assert [(r["arm"], r["assay"], r["band"]) for r in rows] == [
        ("rung1_base", "H3K4me3", "all"), ("rung1_base", AR.MACRO, "all")]
    for k in AR.BUDGET_KEYS:
        assert f(rows[0][f"{k}.mean"]) == pytest.approx(1.3)
        assert f(rows[1][f"{k}.mean"]) == pytest.approx(0.3)
        assert f(rows[1][f"{k}.spread"]) == pytest.approx(0.2)


def test_accessibility_beside_leave_one_assay_out(tree, tmp_path):
    runs, base = tree
    rows, _ = AR.accessibility(runs, "rung1_base", None)
    by = {(r["assay"], r["quantity"]): r for r in rows}
    r = by[("H3K4me3", "count_crps")]
    assert r["scorable"] == 1
    assert r["accessibility_only.mean"] == pytest.approx(0.15)
    assert r["leave_one_assay_out.mean"] == pytest.approx(0.3)
    a = by[("ATAC-seq", "r_spec.counts_log1p")]
    assert a["scorable"] == 0 and a["accessibility_only.mean"] == pytest.approx(0.075)
    assert a["leave_one_assay_out.mean"] is None                      # no such assay in that regime
    assert by[(AR.MACRO, "count_crps")]["accessibility_only.mean"] == pytest.approx(0.15)


def test_per_knob_with_subset_pairs_apart(tree, tmp_path):
    runs, base = tree
    rows, _ = AR.per_knob(runs, "rung1_base")
    got = [(r["pair_kind"], r["knob"]) for r in rows]
    assert got == [("never_trained_non_subset", "crop36"), ("never_trained_non_subset", "se"),
                   ("never_trained_non_subset", AR.MACRO), ("subset", AR.MACRO)]
    assert rows[1]["mean"] == pytest.approx(0.6) and rows[1]["spread"] == pytest.approx(0.4)


def test_shared_specific_gain_against_first_arm(tree, tmp_path):
    runs, base = tree
    out = tmp_path / "s.tsv"
    assert AR.main(["shared-specific", "--runs", str(runs), "--baselines", str(base),
                    "--arms", "rung1_base,rung1_dna_off,marginal_nb", "--out", str(out)]) == 0
    rows = {(r["arm"], r["assay"]): r for r in read(out)}
    ref = rows[("rung1_base", AR.MACRO)]
    assert f(ref["shared.gain"]) == 0.0
    d = rows[("rung1_dna_off", AR.MACRO)]
    assert f(d["specific.gain"]) == pytest.approx(-0.5)               # more error: negative gain
    assert f(d["specific.gain_spread"]) == pytest.approx(0.2)
    b = rows[("marginal_nb", "H3K4me3")]
    assert f(b["shared.gain"]) == pytest.approx(0.2) and b["n_seeds"] == "1"
    assert AR.main(["shared-specific", "--runs", str(runs), "--arms", "rung1_base"]) == 2
    assert AR.main(["shared-specific", "--runs", str(runs), "--arms", "rung1_base,nope"]) == 2


def test_blind_sub_reads_blind_scores(tree, tmp_path):
    runs, base = tree
    for k in range(3):
        put(runs / "rung1_base" / f"seed{k}" / "blind", scores(5.0, sha="B" * 64))
    rows, _ = AR.headline(runs, "rung1_base", None, sub="blind")
    assert rows[0]["count_crps.mean"] == pytest.approx(5.0)
    put(runs / "rung1_base" / "seed2" / "blind", scores(5.0, sha="C" * 64))
    with pytest.raises(AR.Refused):
        AR.headline(runs, "rung1_base", None, sub="blind")


def test_keys_exist_in_scorer():
    """Every top-level key the report reads is a key score.py writes (string present in its source)."""
    src = (REPO / "tools" / "onion_eval" / "score.py").read_text()
    keys = {p[0] for p, _ in AR.HEADLINE if p[0] not in AR.DERIVED_SCALE_ERROR}
    keys |= {p[0] for p in AR.IMPUTATION} | {"error_budget", "accessibility_only", "per_knob",
                                               "evalset_sha256"}
    assert all(f'"{k}"' in src for k in keys), [k for k in keys if f'"{k}"' not in src]
    assert tuple(AR.BUDGET_KEYS) == ("level", "shared", "specific", "explained", "unexplained",
                                     "noise", "added")
