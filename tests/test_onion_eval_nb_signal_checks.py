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
    sig, src = SG.signal_track(mu, np.ones_like(mu), "narrow_histone", ctl, cov, mode="plug_in")
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
