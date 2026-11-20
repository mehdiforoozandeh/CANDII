"""miniCANDI ChIP control arms (chunk A4): the output-only smooth lambda head and the dataset arms
`control_output_only` and `control_none`. The wrapper is built directly, not through the registry."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

from minicandi import contract as C  # noqa: E402
from minicandi import control_head as CH  # noqa: E402
from minicandi import registry as R  # noqa: E402
from minicandi import synth  # noqa: E402
from minicandi.dataset import OnionDataset  # noqa: E402

L = 64
W = CH.DEFAULT_WINDOW_BINS
CI = C.CONTROL_INDEX
LAMBDA = {"control_head": "lambda_smooth", "control_input": False}


class _Stub(C.MiniCandi):
    """Inner model whose output is a fixed log-mean [L, A] plus a learned offset per track."""

    def __init__(self, log_mu: torch.Tensor):
        super().__init__()
        self.register_buffer("log_mu", log_mu.clone())
        self.bias = torch.nn.Parameter(torch.zeros(C.A))

    def forward(self, batch):
        B = batch["x_counts"].shape[0]
        lm = (self.log_mu + self.bias).expand(B, -1, -1)
        return {"n": torch.exp(0.1 * lm), "mu": torch.exp(lm)}


def _haar_detail(levels, L: int, seed: int) -> torch.Tensor:
    """Random sum of dyadically aligned Haar wavelets at the given levels (support 2^j bins)."""
    g = torch.Generator().manual_seed(seed)
    out = torch.zeros(L, dtype=torch.float64)
    for j in levels:
        s = 2 ** j
        for k in range(L // s):
            c = torch.randn((), generator=g, dtype=torch.float64)
            out[k * s:k * s + s // 2] += c
            out[k * s + s // 2:(k + 1) * s] -= c
    return out


def _dummy_batch(B: int = 2, L_: int = L) -> dict:
    return {"x_counts": torch.zeros(B, L_, C.A, dtype=torch.int32)}


def _cfg(r: int, **over) -> dict:
    base = json.loads((_ROOT / f"tools/minicandi/configs/rung{r}/base.json").read_text())
    return {**base, "context_bins": L, "param_budget": 3.0e4, "control_input": False, **over}


def _perturb(model, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    return model.eval()


@pytest.fixture(scope="module")
def tables(tmp_path_factory) -> Path:
    td = synth.make_synthetic_tables(tmp_path_factory.mktemp("onion_ctl"))
    synth.make_evalset(td, n_regions=2, region_bins=4 * L)
    return td


@pytest.fixture(scope="module")
def batch(tables) -> dict:
    b = OnionDataset(tables, "train", _cfg(1), 0).batch(4, np.random.default_rng(0))
    C.validate_batch(b)
    assert b["x_bin_mask"].any() and (~b["x_avail"]).any()
    return b


# ------------------------------------------------------------------------------- config rules


def test_channel_is_the_identity():
    inner = _Stub(torch.zeros(L, C.A))
    assert CH.wrap(inner, {}) is inner
    assert CH.wrap(inner, {"control_head": "channel", "control_input": True}) is inner


@pytest.mark.parametrize("cfg", [{"control_head": "lambda_smooth"},
                                 {"control_head": "lambda_smooth", "control_input": True}])
def test_lambda_with_control_input_raises(cfg):
    with pytest.raises(ValueError, match="control_input"):
        CH.wrap(_Stub(torch.zeros(L, C.A)), cfg)


@pytest.mark.parametrize("cfg", [{"control_head": "lambda"},
                                 {**LAMBDA, "lambda_window_bins": 0},
                                 {**LAMBDA, "lambda_window_bins": 40.0},
                                 {**LAMBDA, "lambda_window_bins": True}])
def test_bad_values_raise(cfg):
    with pytest.raises(ValueError):
        CH.wrap(_Stub(torch.zeros(L, C.A)), cfg)


def test_window_default_and_override():
    assert CH.wrap(_Stub(torch.zeros(L, C.A)), LAMBDA).window_bins == 40
    assert CH.wrap(_Stub(torch.zeros(L, C.A)), {**LAMBDA, "lambda_window_bins": 8}).window_bins == 8


# ------------------------------------------------------------------------------- smoothness


@pytest.mark.parametrize("L_", [64, 1024])
def test_haar_levels_1_to_3_never_reach_the_output(L_):
    """The 40-bin log-space average removes all Haar detail at levels 1 to 3 (supports 2, 4,
    8 bins): adding it to the inner control log-mean changes no output bin. Detail at level 5
    (32 bins) does change it, so the check is not vacuous."""
    g = torch.Generator().manual_seed(0)
    base = torch.randn(L_, C.A, generator=g, dtype=torch.float64)
    ref = CH.wrap(_Stub(base), LAMBDA)(_dummy_batch(L_=L_))["mu"]
    fine = base.clone()
    fine[:, CI] += _haar_detail((1, 2, 3), L_, seed=1)
    got = CH.wrap(_Stub(fine), LAMBDA)(_dummy_batch(L_=L_))["mu"]
    torch.testing.assert_close(got, ref, rtol=1e-10, atol=0)
    coarse = base.clone()
    coarse[:, CI] += _haar_detail((5,), L_, seed=2)
    moved = CH.wrap(_Stub(coarse), LAMBDA)(_dummy_batch(L_=L_))["mu"]
    assert (moved[..., CI] - ref[..., CI]).abs().max() > 1e-3


@pytest.mark.parametrize("L_", [64, 1024])
def test_output_is_constant_within_no_window_and_piecewise_linear(L_):
    g = torch.Generator().manual_seed(3)
    base = torch.randn(L_, C.A, generator=g, dtype=torch.float64)
    lm = torch.log(CH.wrap(_Stub(base), LAMBDA)(_dummy_batch(L_=L_))["mu"][0, :, CI])
    for s in range(0, L_, W):
        seg = lm[s:s + W]
        assert seg.max() - seg.min() > 1e-6, f"window at bin {s} is constant"
    # the second difference is 0 except where a window centre (a knot) sits between bins
    starts = list(range(0, L_, W))
    centres = [s + (min(W, L_ - s) - 1) / 2 for s in starts]
    d2 = lm[2:] - 2 * lm[1:-1] + lm[:-2]          # centred at bins 1 .. L-2
    near_knot = torch.zeros(L_ - 2, dtype=torch.bool)
    for c in centres[1:-1]:
        for i in range(L_ - 2):
            if abs((i + 1) - c) < 1.0:
                near_knot[i] = True
    assert d2[~near_knot].abs().max() < 1e-9
    if near_knot.any():                            # L = 64 has 2 knots, so one straight line
        assert d2[near_knot].abs().max() > 1e-6    # the slope does change at the knots


def test_window_means_are_the_inner_log_means():
    """Linear interpolation through the centre of a full window, read at the two bins around the
    centre, averages back to that window's log-space mean."""
    g = torch.Generator().manual_seed(4)
    base = torch.randn(L, C.A, generator=g, dtype=torch.float64)
    lm = torch.log(CH.wrap(_Stub(base), LAMBDA)(_dummy_batch())["mu"][0, :, CI])
    # window 0 = bins 0..39, centre 19.5: the mean of bins 19 and 20 is the knot value
    torch.testing.assert_close((lm[19] + lm[20]) / 2, base[:W, CI].mean(), rtol=0, atol=1e-12)


def test_control_size_is_one_learned_scalar_and_gradients_flow():
    inner = _Stub(torch.zeros(L, C.A))
    m = CH.wrap(inner, LAMBDA)
    assert m.param_count() == inner.param_count() + 1
    out = m(_dummy_batch())
    assert (out["n"][..., CI] == 1.0).all()            # log_size starts at 0
    loss = out["mu"][..., CI].sum() + out["n"][..., CI].sum()
    loss.backward()
    assert m.log_size.grad is not None and m.log_size.grad.abs() > 0
    assert inner.bias.grad[CI].abs() > 0
    assert (inner.bias.grad[:CI] == 0).all()


# ------------------------------------------------------------------------------- real rungs


@pytest.mark.parametrize("r", [1, 2])
def test_real_rung_other_35_untouched_and_wiring_check(r, batch):
    torch.manual_seed(0)
    inner = _perturb(R.build_model(_cfg(r)))
    wrapped = CH.wrap(inner, _cfg(r, **LAMBDA)).eval()
    with torch.no_grad():
        a, b = inner(batch), wrapped(batch)
    for k in ("n", "mu"):
        assert torch.equal(a[k][..., :CI], b[k][..., :CI])
        assert b[k].shape == a[k].shape and (b[k] > 0).all()
    assert not torch.equal(a["mu"][..., CI], b["mu"][..., CI])
    res = C.wiring_check(wrapped, batch)
    assert res["pass"], res
    assert res["n_perturbed"] > 0


def test_state_dict_round_trip(batch):
    torch.manual_seed(0)
    m1 = _perturb(CH.wrap(R.build_model(_cfg(1)), _cfg(1, **LAMBDA)))
    with torch.no_grad():
        m1.log_size.fill_(0.7)
    torch.manual_seed(5)
    m2 = CH.wrap(R.build_model(_cfg(1)), _cfg(1, **LAMBDA)).eval()
    m2.load_state_dict(m1.state_dict())
    with torch.no_grad():
        o1, o2 = m1(batch), m2(batch)
    for k in ("n", "mu"):
        assert torch.equal(o1[k], o2[k])


# ------------------------------------------------------------------------------- dataset arms


def test_control_output_only_gives_control_as_imputation_target(tables):
    ds = OnionDataset(tables, "train", {"context_bins": L, "control_input": False,
                                        "predict_control": True}, seed=7)
    seen = 0
    for i in range(10):
        b = ds.batch(16, np.random.default_rng(i))
        C.validate_batch(b)
        assert not b["x_avail"][:, CI].any()
        tgt = b["y_avail"][:, CI]
        seen += int(tgt.sum())
        assert (b["y_kind"][:, CI][tgt] == C.KIND_IMPUTE).all()
    assert seen > 0


def test_control_none_has_no_control_target(tables):
    ds = OnionDataset(tables, "train", {"context_bins": L, "control_input": False,
                                        "predict_control": False}, seed=7)
    for i in range(10):
        b = ds.batch(16, np.random.default_rng(i))
        C.validate_batch(b)
        assert not b["x_avail"][:, CI].any()
        assert not b["y_avail"][:, CI].any()
        assert (b["y_kind"][:, CI] == C.KIND_NONE).all()
