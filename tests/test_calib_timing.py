"""t31 — the statistic that decides mid-training coverage.

`gain_vs_jitter` is the whole reason calibration (a) produces a decision rather than a table. It
answers one question per coverage level: does the selection metric move MORE between two epochs
than it wobbles across three? Below 1, ordering two nearby checkpoints is not information, and
running the check more often buys nothing.

The measure is the median absolute SECOND difference, deliberately, and that choice is what these
tests pin. A standard deviation would be dominated by the trend — a steeply but perfectly smoothly
descending curve would look enormously noisy, and the ratio would say "unusable" about the best
possible case. A second difference is zero for any straight line however steep, so it sees
roughness and nothing else.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.calib_timing import gain_vs_jitter                          # noqa: E402


def test_a_perfectly_smooth_descent_has_no_jitter_however_steep_it_is():
    """The case a standard deviation would get exactly backwards."""
    steep = gain_vs_jitter([10.0, 8.0, 6.0, 4.0, 2.0])
    assert steep["gain"] == pytest.approx(2.0)
    assert steep["jitter"] == pytest.approx(0.0)
    assert math.isinf(steep["ratio"])                 # no wobble at all -> perfectly usable
    # and a gentle straight line is just as usable, which is the point of the measure
    gentle = gain_vs_jitter([1.0, 0.99, 0.98, 0.97, 0.96])
    assert gentle["jitter"] == pytest.approx(0.0) and math.isinf(gentle["ratio"])


def test_a_curve_that_only_wobbles_is_reported_as_unusable():
    """No trend, pure alternation. Selection here would be picking noise."""
    r = gain_vs_jitter([1.0, 1.1, 1.0, 1.1, 1.0, 1.1])
    assert abs(r["gain"]) < 0.11
    assert r["jitter"] > 0.19                          # |c - 2c' + c''| = 0.2 every step
    assert r["ratio"] < 1.0


def test_a_real_descent_with_noise_on_it_lands_above_one():
    """Trend 0.05 per epoch, wobble an order of magnitude smaller -- the shape we want to find."""
    base = [1.0 - 0.05 * i for i in range(10)]
    noisy = [v + (0.002 if i % 2 else -0.002) for i, v in enumerate(base)]
    r = gain_vs_jitter(noisy)
    assert r["gain"] > 0.04
    assert r["ratio"] > 1.0


def test_the_sign_convention_is_lower_is_better():
    """CRPS falls when the model improves, so `gain` must be POSITIVE on a falling curve.

    A sign error here would silently invert the recommendation, and nothing downstream would catch
    it -- the table would simply advise the wrong coverage.
    """
    assert gain_vs_jitter([1.0, 0.9, 0.8, 0.7])["gain"] > 0
    assert gain_vs_jitter([0.7, 0.8, 0.9, 1.0])["gain"] < 0


def test_too_few_points_reports_nan_rather_than_inventing_a_verdict():
    """A second difference needs three points. Two epochs cannot say whether a curve is rough."""
    for curve in ([], [1.0], [1.0, 0.9]):
        r = gain_vs_jitter(curve)
        assert math.isnan(r["gain"]) and math.isnan(r["ratio"])
        assert r["n"] == len(curve)


def test_non_finite_points_are_dropped_not_propagated():
    """A NaN check -- no imputation target scored that epoch -- must not erase the whole level."""
    r = gain_vs_jitter([1.0, float("nan"), 0.9, 0.8, float("inf"), 0.7])
    assert r["n"] == 4
    assert math.isfinite(r["gain"]) and math.isfinite(r["jitter"])


def test_the_tool_opens_its_source_through_the_real_constructor():
    """A guessed classmethod name fails only on the cluster, an hour into a queue.

    `DataSource.resolve` is the one entry point that enforces exactly-one-of h5/store and parses the
    regime; `from_flags` never existed. Nothing else in this tool touches the training internals, so
    this one line is the whole surface it can get wrong that way.
    """
    import inspect

    from candi.train import DataSource
    from tools import calib_timing

    assert hasattr(DataSource, "resolve")
    src = inspect.getsource(calib_timing.main)
    assert "DataSource.resolve(" in src
    assert "from_flags" not in src
