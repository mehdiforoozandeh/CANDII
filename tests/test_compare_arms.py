"""Tests for the paired arm-vs-arm comparison.

The sign convention is the thing most likely to be got backwards by a reader in a hurry, and getting
it backwards would report the losing arm as the winner, so it is pinned from both directions.

There are TWO conventions in this module, deliberately opposite: `control - case` for the CRPS family
(a loss) and `case - control` for the M2 ablation steering (a magnitude). Both are pinned below.
"""
from __future__ import annotations

import json
import warnings

import pytest

from candi.compare_arms import (ABLATION_METRICS, _ablation_key, _stratum, compare,
                                    compare_ablation, run)


def _run_json(per_target: dict, *, tag: str, block: str = "imp") -> dict:
    return {"config": {"tag": tag, "cell_cond": tag},
            "M1": {f"{block}_per_target": per_target}}


def _targets(crps: dict, *, oracle: dict | None = None, n: int = 1000) -> dict:
    out = {}
    for k, v in crps.items():
        o = (oracle or {}).get(k, v)
        out[k] = dict(crps=v, crps_oracle_scaled=o, scale_error=v - o, n_points=n)
    return out


KEYS_V = [f"T_c{i}|V_c{i}|H3K4me3" for i in range(6)]
KEYS_B = [f"T_d{i}|B_d{i}|H3K27ac" for i in range(4)]


def test_stratum_reads_the_imputation_target_prefix():
    assert _stratum("T_x|V_x|H3K4me3") == "V"
    assert _stratum("T_x|B_x|H3K4me3") == "B"


def test_positive_delta_means_the_case_arm_won():
    """CRPS is a loss. delta = control - case, so a case arm with LOWER crps must score positive."""
    case = _run_json(_targets({k: 1.0 for k in KEYS_V}), tag="id")
    ctrl = _run_json(_targets({k: 1.5 for k in KEYS_V}), tag="off")
    r = compare(case, ctrl, block="imp", metric="crps", n_boot=200)
    assert r["mean"] == pytest.approx(0.5)
    assert r["supports_direction"] is True
    assert r["n_pos"] == len(KEYS_V) and r["n_neg"] == 0


def test_negative_delta_when_the_case_arm_lost():
    case = _run_json(_targets({k: 2.0 for k in KEYS_V}), tag="id")
    ctrl = _run_json(_targets({k: 1.0 for k in KEYS_V}), tag="off")
    r = compare(case, ctrl, block="imp", metric="crps", n_boot=200)
    assert r["mean"] < 0
    assert r["supports_direction"] is False
    assert r["hi"] < 0, "a real loss must have an interval entirely below zero"


def test_a_tied_arm_is_not_reported_as_significant():
    """The bug `_cluster_bootstrap_ci` documents: ties scored as negatives once made a perfectly
    null arm print the most significant p-value on the page."""
    case = _run_json(_targets({k: 1.0 for k in KEYS_V}), tag="id")
    ctrl = _run_json(_targets({k: 1.0 for k in KEYS_V}), tag="off")
    r = compare(case, ctrl, block="imp", metric="crps", n_boot=200)
    assert r["mean"] == pytest.approx(0.0)
    assert r["n_tied"] == len(KEYS_V) and r["n_pos"] == 0 and r["n_neg"] == 0
    assert not r["supports_direction"]


def test_V_and_B_are_reported_separately():
    """A win confined to V_ must not be smeared over the B_ targets, which is the whole reason the
    strata are never pooled."""
    crps_case = {**{k: 1.0 for k in KEYS_V}, **{k: 2.0 for k in KEYS_B}}
    crps_ctrl = {**{k: 1.5 for k in KEYS_V}, **{k: 2.0 for k in KEYS_B}}
    case = _run_json(_targets(crps_case), tag="id")
    ctrl = _run_json(_targets(crps_ctrl), tag="off")
    v = compare(case, ctrl, metric="crps", stratum="V", n_boot=200)
    b = compare(case, ctrl, metric="crps", stratum="B", n_boot=200)
    allr = compare(case, ctrl, metric="crps", n_boot=200)
    assert v["mean"] == pytest.approx(0.5) and v["n_clusters"] == len(KEYS_V)
    assert b["mean"] == pytest.approx(0.0) and b["n_clusters"] == len(KEYS_B)
    assert 0.0 < allr["mean"] < 0.5, "the pooled number sits between — which is why it is not the headline"


def test_a_scale_only_win_shows_in_crps_but_not_in_the_oracle_scaled_metric():
    """The PI ruling in one test: raw crps is primary, and the decomposition is what says whether the
    gain was shape or merely scale."""
    case = _run_json(_targets({k: 1.0 for k in KEYS_V}, oracle={k: 0.8 for k in KEYS_V}), tag="id")
    ctrl = _run_json(_targets({k: 1.5 for k in KEYS_V}, oracle={k: 0.8 for k in KEYS_V}), tag="off")
    raw = compare(case, ctrl, metric="crps", n_boot=200)
    orc = compare(case, ctrl, metric="crps_oracle_scaled", n_boot=200)
    assert raw["mean"] == pytest.approx(0.5) and raw["supports_direction"]
    assert orc["mean"] == pytest.approx(0.0) and not orc["supports_direction"]


def test_only_targets_present_in_both_arms_are_compared():
    case = _run_json(_targets({k: 1.0 for k in KEYS_V + KEYS_B}), tag="id")
    ctrl = _run_json(_targets({k: 1.5 for k in KEYS_V}), tag="off")
    r = compare(case, ctrl, metric="crps", n_boot=200)
    assert r["n_clusters"] == len(KEYS_V)


def test_nan_targets_are_dropped_and_counted():
    case = _run_json(_targets({k: 1.0 for k in KEYS_V}), tag="id")
    ctrl = _run_json(_targets({k: 1.5 for k in KEYS_V}), tag="off")
    ctrl["M1"]["imp_per_target"][KEYS_V[0]]["crps"] = float("nan")
    r = compare(case, ctrl, metric="crps", n_boot=200)
    assert r["n_clusters"] == len(KEYS_V) - 1
    assert r["n_targets_dropped_nan"] == 1


def test_weighting_uses_the_smaller_point_count():
    case = _run_json(_targets({k: 1.0 for k in KEYS_V}, n=10), tag="id")
    ctrl = _run_json(_targets({k: 1.5 for k in KEYS_V}, n=10_000), tag="off")
    r = compare(case, ctrl, metric="crps", n_boot=200)
    assert r["n_clusters"] == len(KEYS_V)   # weights are equal, so the mean is the plain mean
    assert r["mean"] == pytest.approx(0.5)


def test_a_run_json_without_per_target_says_what_to_do(tmp_path):
    old = {"config": {"tag": "old"}, "M1": {"imp_per_assay": {}}}
    p = tmp_path / "old.json"
    p.write_text(json.dumps(old))
    with pytest.raises(ValueError, match="predating per-target reporting"):
        run(str(p), [str(p)], blocks=("imp",))


def test_run_emits_every_metric_and_stratum():
    case = _run_json(_targets({k: 1.0 for k in KEYS_V + KEYS_B}), tag="id")
    ctrl = _run_json(_targets({k: 1.5 for k in KEYS_V + KEYS_B}), tag="off")
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        pa, pb = f"{d}/a.json", f"{d}/b.json"
        open(pa, "w").write(json.dumps(case))
        open(pb, "w").write(json.dumps(ctrl))
        out = run(pa, [pb], blocks=("imp",), n_boot=200)
    # 3 metrics x 3 strata
    assert len(out["results"]) == 9
    assert "case better" in out["text"]


# ---------------------------------------------------------------------------
# M2.ablation steering. The subtraction here is the OPPOSITE way round from the CRPS path -- d_eta is
# a steering magnitude, not a loss -- so the sign is pinned from both directions again, and the
# caveats (mode, sentinel skips, purity fallbacks) are asserted to actually reach the output.
# ---------------------------------------------------------------------------

def _abl_recs(spec: dict, *, fired: frozenset = frozenset(), max_scale: float = 10.0) -> list:
    """`{target_key: [(d_eta, n_fg), ...]}` -> the flat per-EVAL-UNIT record list a real run emits."""
    out = []
    for k, units in spec.items():
        for d_eta, n_fg in units:
            out.append(dict(target=k.split("|"), d_eta=d_eta, d_eta_max=d_eta * max_scale,
                            d_crps=0.0, d_mu=0.0, n_fg=n_fg, fg_frac_realized=0.02,
                            purity_fallback_fired=(k in fired), true_value=1.0, swapped_value=2.0))
    return out


def _abl_json(spec: dict, *, tag: str, row: str = "assay_id", mode: str = "cross_target",
              n_sentinel_skipped: int = 0, fired: frozenset = frozenset()) -> dict:
    recs = _abl_recs(spec, fired=fired)
    return {"config": {"tag": tag, "cell_cond": tag},
            "M1": {"imp_per_target": {}},
            "M2": {"ablation": {row: dict(row=1, covariate=row, mode=mode, per_target=recs,
                                          n_targets=len(recs),
                                          n_sentinel_skipped=n_sentinel_skipped,
                                          mean_abs_d_eta=0.0, max_abs_d_eta=0.0, mean_d_crps=0.0,
                                          frac_true_better=0.5, uses_covariate=True)}}}


def _flat(keys, v, n_fg=100):
    return {k: [(v, n_fg)] for k in keys}


def test_ablation_key_accepts_both_serializations():
    """`_jsonable` writes a tuple VALUE as a list; a tuple KEY becomes 'a|b|c'. Both are read."""
    assert _ablation_key({"target": ["T_x", "V_x", "H3K4me3"]}) == "T_x|V_x|H3K4me3"
    assert _ablation_key({"target": "T_x|V_x|H3K4me3"}) == "T_x|V_x|H3K4me3"
    assert _ablation_key({"target": ["T_x", "V_x"]}) is None


def test_ablation_paired_delta_is_computable_by_hand():
    """Known answer, end to end: per-unit records collapse n_fg-weighted to one value per target,
    then the paired deltas are n_fg-weighted across targets.

    target A: case (1.0 x100, 2.0 x300) -> 1.75 ; control (1.0 x200, 1.0 x200) -> 1.00 ; d=+0.75 w=400
    target B: case (0.5 x100)           -> 0.50 ; control (1.5 x100)           -> 1.50 ; d=-1.00 w=100
    overall  = (400*0.75 + 100*-1.00) / 500 = +0.40
    """
    A, B = "T_a|V_a|H3K4me3", "T_b|V_b|H3K4me3"
    case = _abl_json({A: [(1.0, 100), (2.0, 300)], B: [(0.5, 100)]}, tag="case")
    ctrl = _abl_json({A: [(1.0, 200), (1.0, 200)], B: [(1.5, 100)]}, tag="ctrl")
    r = compare_ablation(case, ctrl, "assay_id", metric="d_eta", n_boot=200)
    assert r["n_clusters"] == 2 and r["n_targets_paired"] == 2
    assert r["mean"] == pytest.approx(0.40, abs=1e-12)
    assert sorted(r["cluster_values"]) == [pytest.approx(-1.0), pytest.approx(0.75)]
    assert r["n_pos"] == 1 and r["n_neg"] == 1 and r["n_tied"] == 0
    # d_eta_max is a fixed multiple of d_eta in the fixture, so its delta must scale exactly.
    rm = compare_ablation(case, ctrl, "assay_id", metric="d_eta_max", n_boot=200)
    assert rm["mean"] == pytest.approx(4.0, abs=1e-9)


def test_ablation_sign_is_inverted_more_steering_is_the_case_arm_winning():
    """d_eta is NOT a loss: delta = case - control, so uniformly LARGER d_eta must read POSITIVE."""
    case = _abl_json(_flat(KEYS_V, 1.5), tag="case")
    ctrl = _abl_json(_flat(KEYS_V, 1.0), tag="ctrl")
    r = compare_ablation(case, ctrl, "assay_id", n_boot=200)
    assert r["mean"] == pytest.approx(0.5)
    assert r["supports_direction"] is True
    assert r["n_pos"] == len(KEYS_V) and r["n_neg"] == 0
    assert "case - control" in r["sign_convention"]


def test_ablation_sign_flips_when_the_case_arm_steers_less():
    case = _abl_json(_flat(KEYS_V, 1.0), tag="case")
    ctrl = _abl_json(_flat(KEYS_V, 1.5), tag="ctrl")
    r = compare_ablation(case, ctrl, "assay_id", n_boot=200)
    assert r["mean"] == pytest.approx(-0.5)
    assert r["supports_direction"] is False and r["hi"] < 0


def test_ablation_unmatched_targets_are_dropped_and_counted_not_zero_filled():
    """A target only one arm scored has no paired delta. Zero-filling it would dilute a real effect
    toward zero; averaging it in would invent a number."""
    case = _abl_json(_flat(KEYS_V + KEYS_B, 1.5), tag="case")
    ctrl = _abl_json(_flat(KEYS_V, 1.0), tag="ctrl")
    r = compare_ablation(case, ctrl, "assay_id", n_boot=200)
    assert r["n_targets_case"] == len(KEYS_V) + len(KEYS_B)
    assert r["n_targets_control"] == len(KEYS_V)
    assert r["n_targets_paired"] == len(KEYS_V) and r["n_clusters"] == len(KEYS_V)
    assert r["n_targets_dropped_unmatched"] == len(KEYS_B)
    assert r["n_targets_dropped_case_only"] == len(KEYS_B)
    assert r["n_targets_dropped_control_only"] == 0
    # not zero-filled: every paired delta is the real +0.5, none pulled toward 0
    assert r["mean"] == pytest.approx(0.5)


def test_ablation_within_batch_mode_warns_loudly():
    """within_batch is the identity swap — a structural null. Comparing two arms on it compares two
    zeros, so it must never pass silently."""
    case = _abl_json(_flat(KEYS_V, 1.5), tag="case", mode="within_batch")
    ctrl = _abl_json(_flat(KEYS_V, 1.0), tag="ctrl", mode="within_batch")
    with pytest.warns(UserWarning, match="STRUCTURAL NULL"):
        r = compare_ablation(case, ctrl, "assay_id", n_boot=200)
    assert r["mode_ok"] is False
    assert r["mode_case"] == "within_batch" and r["mode_control"] == "within_batch"

    # and the printed table must carry the banner too — a reader of the .md never sees the warning
    import tempfile
    for d in (case, ctrl):
        d["M1"]["imp_per_target"] = _targets({k: 1.0 for k in KEYS_V})
    with tempfile.TemporaryDirectory() as t:
        pa, pb = f"{t}/a.json", f"{t}/b.json"
        open(pa, "w").write(json.dumps(case))
        open(pb, "w").write(json.dumps(ctrl))
        with pytest.warns(UserWarning, match="STRUCTURAL NULL"):
            out = run(pa, [pb], blocks=("imp",), n_boot=200, ablation_rows=("assay_id",))
    assert "!! WARNING" in out["text"] and "STRUCTURAL NULL" in out["text"]


def test_ablation_cross_target_mode_does_not_warn():
    case = _abl_json(_flat(KEYS_V, 1.5), tag="case")
    ctrl = _abl_json(_flat(KEYS_V, 1.0), tag="ctrl")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = compare_ablation(case, ctrl, "assay_id", n_boot=200)
    assert r["mode_ok"] is True


def test_ablation_V_and_B_are_stratified_never_pooled():
    spec_case = {**_flat(KEYS_V, 1.5), **_flat(KEYS_B, 1.0)}
    spec_ctrl = {**_flat(KEYS_V, 1.0), **_flat(KEYS_B, 1.0)}
    case, ctrl = _abl_json(spec_case, tag="case"), _abl_json(spec_ctrl, tag="ctrl")
    v = compare_ablation(case, ctrl, "assay_id", stratum="V", n_boot=200)
    b = compare_ablation(case, ctrl, "assay_id", stratum="B", n_boot=200)
    allr = compare_ablation(case, ctrl, "assay_id", n_boot=200)
    assert v["n_clusters"] == len(KEYS_V) and v["mean"] == pytest.approx(0.5)
    assert b["n_clusters"] == len(KEYS_B) and b["mean"] == pytest.approx(0.0)
    assert v["stratum"] == "V" and b["stratum"] == "B"
    assert 0.0 < allr["mean"] < 0.5, "the pooled number sits between — which is why it is not the headline"


def test_ablation_purity_fallback_and_sentinel_counts_reach_the_output():
    """~18% of real records fire the purity fallback and it is invisible unless counted."""
    fired = frozenset(KEYS_V[:2])
    case = _abl_json(_flat(KEYS_V, 1.5), tag="case", fired=fired, n_sentinel_skipped=7)
    ctrl = _abl_json(_flat(KEYS_V, 1.0), tag="ctrl", fired=fired, n_sentinel_skipped=3)
    r = compare_ablation(case, ctrl, "assay_id", n_boot=200)
    assert r["n_purity_fallback_fired_case"] == 2
    assert r["n_purity_fallback_fired_control"] == 2
    assert r["n_sentinel_skipped_case"] == 7 and r["n_sentinel_skipped_control"] == 3
    # surfaced, not dropped, by default
    assert r["purity_fallback_excluded"] is False and r["n_clusters"] == len(KEYS_V)
    dropped = compare_ablation(case, ctrl, "assay_id", n_boot=200, exclude_purity_fallback=True)
    assert dropped["purity_fallback_excluded"] is True
    assert dropped["n_clusters"] == len(KEYS_V) - 2


def test_ablation_nan_records_are_dropped():
    case = _abl_json(_flat(KEYS_V, 1.5), tag="case")
    ctrl = _abl_json(_flat(KEYS_V, 1.0), tag="ctrl")
    case["M2"]["ablation"]["assay_id"]["per_target"][0]["d_eta"] = float("nan")
    r = compare_ablation(case, ctrl, "assay_id", n_boot=200)
    assert r["n_records_nan_case"] == 1
    assert r["n_clusters"] == len(KEYS_V) - 1


def test_ablation_missing_block_says_what_to_do():
    case = _abl_json(_flat(KEYS_V, 1.5), tag="case")
    ctrl = _abl_json(_flat(KEYS_V, 1.0), tag="ctrl")
    del ctrl["M2"]["ablation"]
    with pytest.raises(ValueError, match="predating the sentinel-free metadata ablation"):
        compare_ablation(case, ctrl, "assay_id", control_path="ctrl.json")
    with pytest.raises(ValueError, match="rows present"):
        compare_ablation(case, case, "run_type")


def test_run_is_purely_additive_and_carries_the_ablation_into_the_outputs():
    """No --ablation-row => byte-identical CRPS behaviour. With it => 2 metrics x 3 strata more."""
    import tempfile
    spec = {**_flat(KEYS_V, 1.5), **_flat(KEYS_B, 1.5)}
    case = _abl_json(spec, tag="case")
    ctrl = _abl_json({**_flat(KEYS_V, 1.0), **_flat(KEYS_B, 1.0)}, tag="ctrl")
    for d in (case, ctrl):
        d["M1"]["imp_per_target"] = _targets({k: 1.0 for k in KEYS_V + KEYS_B})
    with tempfile.TemporaryDirectory() as t:
        pa, pb = f"{t}/a.json", f"{t}/b.json"
        open(pa, "w").write(json.dumps(case))
        open(pb, "w").write(json.dumps(ctrl))
        base = run(pa, [pb], blocks=("imp",), n_boot=200)
        out = run(pa, [pb], blocks=("imp",), n_boot=200, ablation_rows=("assay_id",))
    assert len(base["results"]) == 9 and not base["ablation"]
    assert len(out["results"]) == 9 + len(ABLATION_METRICS) * 3
    assert len(out["ablation"]) == len(ABLATION_METRICS) * 3
    assert all(k in out["results"] for k in out["ablation"])
    # the inverted sign and both invisible counts must be in the printed table, not just the json
    assert "delta = case - control" in out["text"]
    assert "purity_fallback_fired=" in out["text"] and "n_sentinel_skipped=" in out["text"]
    assert "mode=cross_target/cross_target" in out["text"]
