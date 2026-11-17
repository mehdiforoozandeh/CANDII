"""E2: the checks of the counts-to-signal route (tools/onion_eval/nb_signal_checks.py)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from candi.store.pval_from_counts import log10_upper_tail  # noqa: E402
from onion_eval import nb_signal_checks as NC  # noqa: E402
from onion_eval import signal_from_counts as SG  # noqa: E402


def test_chip_lambda_is_the_lambda_of_signal_track():
    rng = np.random.default_rng(1)
    mu = rng.gamma(1.0, 2.0, 3000) + 1e-3
    ctl = rng.poisson(1.5, 3000).astype(float)
    cov = dict(depth_reads=2.0e7, read_length=36, frag_len=180, control_depth_reads=3.1e7,
               control_read_length=50)
    sig, src = SG.signal_track(mu, np.ones_like(mu), "narrow_histone", ctl, cov, mode="plug_in",
                               pileup="inbin")
    lam = NC.chip_lambda(ctl, depth=2.0e7, frag_len=180, ctl_depth=3.1e7, ctl_read_length=50)
    want = log10_upper_tail(np.floor(mu * 180 / (36 + 24)).astype(np.int64), lam)
    assert src == "observed_control"
    np.testing.assert_allclose(sig, want.astype(np.float32), rtol=0, atol=0)


def test_agreement_of_a_track_with_itself():
    rng = np.random.default_rng(2)
    x = rng.exponential(1.0, 2 * 400)
    got = NC.agreement(x, x, rb=400)
    assert abs(got["pearson_raw"] - 1) < 1e-12 and abs(got["slope"] - 1) < 1e-12
    assert got["median_abs_diff"] == 0 and got["tile_f1"] == 1 and got["bin_jaccard"] == 1


def test_agreement_peak_counts():
    x = np.zeros(40)
    y = np.zeros(40)
    x[[0, 25]] = 3.0       # formula calls tile 0 and tile 1
    y[[1]] = 3.0           # pipeline calls tile 0 only
    got = NC.agreement(x, y, rb=40)
    assert got["tile_precision"] == 0.5 and got["tile_recall"] == 1.0
    assert got["bin_precision"] == 0.0


def test_lambda_compare():
    obs = np.array([1.0, 2.0, 4.0, 8.0])
    got = NC.lambda_compare(obs * np.array([1.0, 1.05, 1.5, 0.5]), obs, 1.0, 1.0)
    assert got["frac_within_10pct"] == 0.5
    assert got["frac_bg_obs"] == 0.25 and got["within_10pct_median"] == 1


def test_agreement_slope_is_pipeline_on_formula():
    """y = 2x + noise-free offset: slope 2 (pipeline on formula), not 0.5 (the reverse)."""
    x = np.linspace(0.0, 5.0, 800)
    got = NC.agreement(x, 2.0 * x + 1.0, rb=400)
    assert abs(got["slope"] - 2.0) < 1e-9
    assert abs(got["mean_diff"] - float(np.mean(x - (2.0 * x + 1.0)))) < 1e-12


def test_agreement_masks_non_finite_bins():
    rng = np.random.default_rng(3)
    x = rng.exponential(1.0, 800)
    y = x.copy()
    y[[5, 17]] = np.nan
    x[33] = np.inf
    got = NC.agreement(x, y, rb=400)
    assert got["n_masked"] == 3
    assert np.isfinite(got["median_abs_diff"]) and np.isfinite(got["p99_abs_diff"])
    assert abs(got["pearson_raw"] - 1.0) < 1e-12


def test_lambda_ratio_is_predicted_over_observed():
    """predicted = 2 x observed: ratio 2 (pred / obs), not 0.5; and the background fractions
    belong to their own side."""
    obs = np.array([1.0, 2.0, 4.0, 8.0])
    got = NC.lambda_compare(2.0 * obs, obs, 2.0, 1.0)
    assert got["median_ratio"] == 2.0 and got["within_10pct_median"] == 0
    assert got["frac_bg_obs"] == 0.25 and got["frac_bg_pred"] == 0.25
    got = NC.lambda_compare(obs, 2.0 * obs, 1.0, 1.0)
    assert got["median_ratio"] == 0.5 and got["frac_bg_pred"] == 0.25 and got["frac_bg_obs"] == 0.0


def test_lambda_refuses_a_non_validation_dump(tmp_path):
    import json
    import pytest
    (tmp_path / "tracks.json").write_text(json.dumps({"split": "blind", "regime": "x",
                                                      "targets": []}))

    class Sc:
        rb, regions = 4, []
    with pytest.raises(SystemExit, match="validation dumps only"):
        NC._lambda_one(Sc(), tmp_path)


def test_table_common_macro_uses_one_assay_set(tmp_path):
    import json
    blk = {"counts_log1p": {"macro": 0.5, "per_assay": {"A": 0.4, "B": 0.6}, "missing_scorable": []},
           "signal_arcsinh": {"macro": None, "per_assay": {"A": 0.2, "B": None},
                              "missing_scorable": ["C"]}}
    s = {"specific_variance_recovered_above_noise": blk, "r_spec": blk}
    p = tmp_path / "s.json"
    p.write_text(json.dumps(s))
    assert NC.main(["table", "--run", f"r={p}", "--out", str(tmp_path / "o")]) == 0
    import csv
    rows = list(csv.DictReader(open(tmp_path / "o" / "signal_space_macro.tsv"), delimiter="\t"))
    r = {(x["block"], x["space"]): x for x in rows if x["regime"] == "leave_one_assay_out"}
    c = r[("r_spec", "counts_log1p")]
    g = r[("r_spec", "signal_arcsinh")]
    assert float(c["macro_common"]) == 0.4 and float(g["macro_common"]) == 0.2
    assert c["n_assays_common"] == g["n_assays_common"] == "1"
    assert float(c["macro_defined"]) == 0.5 and g["missing_scorable"] == "C"
    assert g["n_missing_scorable"] == "1"


# ---------------------------------------------------------------- E2 formula extension (pileups)

def _row(**kw):
    r = dict(assay_class="narrow_histone", assay="H3K4me3", depth_reads=2.0e7, read_length=36,
             frag_len=180)
    r.update(kw)
    return r


def test_inbin_variant_is_the_scorer_route():
    rng = np.random.default_rng(4)
    c = rng.poisson(2.0, 2000).astype(float)
    ctl = rng.poisson(1.0, 2000).astype(float)
    crow = dict(depth_reads=3.0e7, read_length=50)
    got = NC.variant_signal_region(c, _row(), crow, ctl, "inbin")
    mu = np.where(c > 0, c, NC.ZERO_COUNT)
    want, _ = SG.signal_track(mu, np.ones_like(mu), "narrow_histone", ctl,
                              dict(depth_reads=2.0e7, read_length=36, frag_len=180,
                                   control_depth_reads=3.0e7, control_read_length=50),
                              mode="plug_in", pileup="inbin")
    np.testing.assert_array_equal(got.astype(np.float32), want)
    row = _row(assay_class="accessibility_dnase", assay="DNase-seq", read_length=76)
    got = NC.variant_signal_region(c, row, None, None, "inbin")
    want, _ = SG.signal_track(mu, np.ones_like(mu), "accessibility_dnase", None,
                              dict(depth_reads=2.0e7, read_length=76), mode="plug_in",
                              pileup="inbin")
    # the scorer route reads its llocal lambda off mu, where a zero count is 1e-9: 1e-7 relative
    np.testing.assert_allclose(got.astype(np.float32), want, rtol=1e-6, atol=0)


def test_box_weights_and_widths():
    np.testing.assert_array_equal(NC.box_weights(25), [1.0])
    w = NC.box_weights(75)                       # exactly three bins
    np.testing.assert_allclose(w, [1 / 3] * 3)
    w = NC.box_weights(100)                      # 4 bins: 0.5, 1, 1, 1, 0.5 over 4
    np.testing.assert_allclose(w, np.array([0.5, 1, 1, 1, 0.5]) / 4)
    assert NC.kernel_width_bp("inbin", 180, 36) == 0
    assert NC.kernel_width_bp("box_d", 180, 36) == 180
    assert NC.kernel_width_bp("box_ext", 180, 36) == 144
    assert NC.kernel_width_bp("box_ext", 30, 36) == 25


def test_box_variants_spread_one_count_and_keep_the_mass():
    c = np.zeros(101)
    c[50] = 6.0
    for v, width in (("box_d", 180.0), ("box_ext", 144.0)):
        p = NC.pileup_variant(c, 180, 36, v)
        assert abs(p.sum() - 6.0 * 180 / 60) < 1e-9          # mass count * d / (rl + 24)
        assert np.count_nonzero(p) == NC.box_weights(width).size
        assert p[50] == p.max() and np.allclose(p, p[::-1])  # centred
    assert np.count_nonzero(NC.pileup_variant(c, 180, 36, "inbin")) == 1


def test_submean_of_a_flat_track_is_the_plain_score():
    from candi.store.pval_from_counts import log10_upper_tail
    pile, lam = np.full(50, 7.3), np.full(50, 2.0)
    want = log10_upper_tail(np.full(50, 7, dtype=np.int64), lam)
    np.testing.assert_allclose(NC._score_submean(pile, lam), want, rtol=1e-12)
    pile[25] = 30.0                               # a peak: the mean of the scores is below the
    got = NC._score_submean(pile, lam)            # score of the bin's mean pileup at its centre
    assert got[25] < log10_upper_tail(np.array([30]), np.array([2.0]))[0]


def test_agreement_masks_the_edge_bins():
    """E2 review: bins outside `keep` leave every statistic and their tiles leave the tile counts."""
    rng = np.random.default_rng(12)
    x = rng.exponential(1.0, 800)
    y = x.copy()
    y[:3] = 50.0                    # a disagreement confined to the masked edge of region 0
    x[:3] = 0.0
    keep = SG.edge_keep(2, 400, 3)
    got = NC.agreement(x, y, rb=400, keep=keep)
    assert got["n_edge_masked"] == 12 and abs(got["pearson_raw"] - 1.0) < 1e-12
    assert got["tile_f1"] == 1.0 or np.isnan(got["tile_f1"])
    assert NC.agreement(x, y, rb=400)["pearson_raw"] < 0.99
