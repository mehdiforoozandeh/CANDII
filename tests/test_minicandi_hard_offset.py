"""The `hard_offset` arm at rung 4 (PI ruling 48): with `hide_depth_covariate` on, the network
reads no depth covariate (C, C', input count transform), and depth enters the NB mean only through
the offset add-back mu = s * 2**eta. With the key off the model is today's, output for output."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "tools",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from minicandi import contract as C  # noqa: E402
from minicandi import dataset as D  # noqa: E402
from minicandi import synth  # noqa: E402
from minicandi.components import REF_LOG2_DEPTH  # noqa: E402
from minicandi.covariates import model_covariates  # noqa: E402
from minicandi.rungs import unet  # noqa: E402

L = 64
B = 4
CFG_DIR = _ROOT / "tools/minicandi/configs"
ARM = json.loads((CFG_DIR / "rung4/hard_offset.json").read_text())
ARMS_BASE = json.loads((CFG_DIR / "arms/rung4/base.json").read_text())


@pytest.fixture(scope="module")
def batch(tmp_path_factory):
    tables = synth.make_synthetic_tables(tmp_path_factory.mktemp("onion_ho"))
    cfg = dict(ARMS_BASE, context_bins=L)
    b = D.OnionDataset(tables, "train", cfg, seed=0).batch(B, np.random.default_rng(0))
    C.validate_batch(b)
    return b


def _model(hide: bool | None, seed: int = 0):
    cfg = dict(ARMS_BASE, context_bins=L)
    if hide is not None:
        cfg["hide_depth_covariate"] = hide
    torch.manual_seed(seed)
    m = unet.Model(cfg).eval()
    # move the zero-initialised head off its constant start so eta depends on the features
    with torch.no_grad():
        m.nb.lin.weight.normal_(0.0, 0.3, generator=torch.Generator().manual_seed(1))
    return m


def _shift_depth(b: dict, key: str, avail: str, delta: float) -> dict:
    out = dict(b)
    c = b[key].clone()
    c[..., C.LOG2_DEPTH_COL] = torch.where(b[avail].bool(), c[..., C.LOG2_DEPTH_COL] + delta,
                                           c[..., C.LOG2_DEPTH_COL])
    out[key] = c
    return out


def _eta(out: dict, b: dict) -> torch.Tensor:
    d = b["c_out"][..., C.LOG2_DEPTH_COL].float()[:, None, :]
    return torch.log2(out["mu"].double()) - (d.double() - REF_LOG2_DEPTH)


def test_arm_config_is_the_arms_base_plus_one_key():
    extra = {k: v for k, v in ARM.items() if ARMS_BASE.get(k, object()) != v}
    assert extra == {"hide_depth_covariate": True}
    assert set(ARM) - set(ARMS_BASE) == {"hide_depth_covariate"}
    assert ARM["count_transform"] == "offset"


def test_eta_ignores_the_depth_covariate_with_counts_fixed(batch):
    m = _model(True)
    y = batch["y_avail"].bool()[:, None, :].expand(-1, L, -1)
    with torch.no_grad():
        o0 = m(batch)
        for delta in (-3.0, 2.0):
            b1 = _shift_depth(_shift_depth(batch, "c_in", "x_avail", delta), "c_out", "y_avail",
                              delta)
            o1 = m(b1)
            torch.testing.assert_close(_eta(o1, b1)[y], _eta(o0, batch)[y], rtol=0, atol=1e-5)
            assert torch.equal(o1["n"], o0["n"])
    # the same shift moves eta when the key is off: the check has power
    m_off = _model(False)
    with torch.no_grad():
        b1 = _shift_depth(_shift_depth(batch, "c_in", "x_avail", 2.0), "c_out", "y_avail", 2.0)
        assert (_eta(m_off(b1), b1) - _eta(m_off(batch), batch))[y].abs().max() > 1e-3


def test_mean_scales_exactly_with_target_depth_through_the_offset(batch):
    m = _model(True)
    y = batch["y_avail"].bool()[:, None, :].expand(-1, L, -1)
    with torch.no_grad():
        o0 = m(batch)
        # input depth alone: nothing moves (the input transform reads no depth either)
        oi = m(_shift_depth(batch, "c_in", "x_avail", 1.5))
        assert torch.equal(oi["mu"], o0["mu"]) and torch.equal(oi["n"], o0["n"])
        # target depth +1 log2: mu doubles, n unchanged
        o1 = m(_shift_depth(batch, "c_out", "y_avail", 1.0))
    torch.testing.assert_close(o1["mu"][y].double() / o0["mu"][y].double(),
                               torch.full_like(o0["mu"][y].double(), 2.0), rtol=1e-6, atol=0)
    assert torch.equal(o1["n"], o0["n"])


def test_model_covariates_keeps_true_target_depth_and_zeros_of_absent_tracks(batch):
    e_in, e_out, t_in, t_out = model_covariates(batch, "none", False, 0.0, True)
    xa, ya = batch["x_avail"].bool(), batch["y_avail"].bool()
    for c, a in ((e_in, xa), (e_out, ya), (t_in, xa)):
        assert torch.all(c[..., 0][a] == 24.0)
        assert torch.all(c[..., 0][~a] == 0.0)
        assert torch.equal(c[..., 1:], (batch["c_in"] if c is not e_out else batch["c_out"])[..., 1:])
    assert torch.equal(t_out, batch["c_out"])


def test_key_off_is_todays_model(batch):
    # model_covariates with the flag off returns the batch's own tensors, as before the change
    r = model_covariates(batch, "none", False, 0.0, False)
    assert r[0] is batch["c_in"] and r[1] is batch["c_out"]
    assert r[2] is batch["c_in"] and r[3] is batch["c_out"]
    for mode in (False, True):
        a, b = _model(None), _model(False)
        a.train(mode)
        b.train(mode)
        assert [k for k, _ in a.named_parameters()] == [k for k, _ in b.named_parameters()]
        with torch.no_grad():
            torch.manual_seed(7)
            oa = a(batch)
            torch.manual_seed(7)
            ob = b(batch)
        assert torch.equal(oa["mu"], ob["mu"]) and torch.equal(oa["n"], ob["n"])


def test_arm_passes_the_wiring_check(batch):
    m = _model(True)
    for i in range(2):
        assert C.wiring_check(m, batch, seed=i)["pass"]



# ------------------------------------------------------------------ split arms (PI ruling 49)
# `"output"` hides only C' column 0; `"input"` hides only C column 0. In both, the input count
# transform and the offset add-back keep the true depths.


def _shift_input_depth_keep_transform(b: dict, delta: int) -> dict:
    """c_in log2 depth + delta (an integer >= 1) on available tracks, and visible counts y -> (y + 1) * 2**delta - 1,
    so the offset transform log2((y + 1) / s) is unchanged bit for bit: only the depth covariate
    the C encoder reads moves. (The input transform reads the true c_in depth in the split arms,
    so the input depth cannot change with the counts held fixed without moving the transform.)"""
    assert int(delta) == delta >= 1
    out = _shift_depth(b, "c_in", "x_avail", float(delta))
    vis = b["x_avail"].bool()[:, None, :] & ~b["x_bin_mask"].bool()
    y = b["x_counts"]
    out["x_counts"] = torch.where(vis, (y + 1) * 2 ** delta - 1, y).to(y.dtype)
    return out


def _hidden_shift(value: str, b: dict, delta: int) -> dict:
    if value == "output":
        return _shift_depth(b, "c_out", "y_avail", float(delta))
    return _shift_input_depth_keep_transform(b, delta)


def _other_shift(value: str, b: dict, delta: int) -> dict:
    if value == "output":
        return _shift_input_depth_keep_transform(b, delta)
    return _shift_depth(b, "c_out", "y_avail", float(delta))


@pytest.mark.parametrize("value,name", [("output", "hide_depth_output"),
                                        ("input", "hide_depth_input")])
def test_split_arm_config_is_hard_offset_with_the_key_value_changed(value, name):
    cfg = json.loads((CFG_DIR / f"rung4/{name}.json").read_text())
    assert cfg == dict(ARM, hide_depth_covariate=value)


@pytest.mark.parametrize("value", ["output", "input"])
def test_model_covariates_split_hides_one_encoder_path_only(batch, value):
    e_in, e_out, t_in, t_out = model_covariates(batch, "none", False, 0.0, value)
    hidden, kept, key, avail = ((e_out, e_in, "c_in", "y_avail") if value == "output"
                                else (e_in, e_out, "c_out", "x_avail"))
    a = batch[avail].bool()
    assert torch.all(hidden[..., 0][a] == 24.0) and torch.all(hidden[..., 0][~a] == 0.0)
    assert torch.equal(hidden[..., 1:], batch["c_out" if value == "output" else "c_in"][..., 1:])
    assert kept is batch[key]
    assert t_in is batch["c_in"] and t_out is batch["c_out"]


def test_shifting_input_depth_with_matched_counts_keeps_the_transform(batch):
    m = _model(False)
    vis = batch["x_avail"].bool()[:, None, :] & ~batch["x_bin_mask"].bool()
    b1 = _shift_input_depth_keep_transform(batch, 1)
    t0 = m.transform(batch["x_counts"], batch["c_in"].float()[..., 0], vis)
    t1 = m.transform(b1["x_counts"], b1["c_in"].float()[..., 0], vis)
    assert torch.equal(t0[vis], t1[vis])


def _perturbed(value, seed: int = 0):
    """Noise on every parameter: the covariate injectors start at the identity, so an untrained
    model ignores C and C' and the "other side moves eta" check would fail for want of power."""
    m = _model(value, seed)
    g = torch.Generator().manual_seed(seed + 2)
    with torch.no_grad():
        for q in m.parameters():
            q.add_(0.05 * torch.randn(q.shape, generator=g))
    return m


@pytest.mark.parametrize("value", ["output", "input"])
def test_split_eta_ignores_the_hidden_side_and_reads_the_other(batch, value):
    m = _perturbed(value)
    y = batch["y_avail"].bool()[:, None, :].expand(-1, L, -1)
    with torch.no_grad():
        o0 = m(batch)
        for delta in (1, 2, 3):
            b1 = _hidden_shift(value, batch, delta)
            o1 = m(b1)
            torch.testing.assert_close(_eta(o1, b1)[y], _eta(o0, batch)[y], rtol=0, atol=1e-5)
            assert torch.equal(o1["n"], o0["n"])
        # the other side's depth covariate is read: eta moves
        b2 = _other_shift(value, batch, 2)
        assert (_eta(m(b2), b2) - _eta(o0, batch))[y].abs().max() > 1e-3
    # power: the same hidden-side shift moves eta when the key is off
    m_off = _perturbed(False)
    with torch.no_grad():
        b1 = _hidden_shift(value, batch, 2)
        assert (_eta(m_off(b1), b1) - _eta(m_off(batch), batch))[y].abs().max() > 1e-3


@pytest.mark.parametrize("value", ["output", "input"])
def test_split_mean_scales_with_target_depth_through_the_offset(batch, value, monkeypatch):
    m = _perturbed(value)
    y = batch["y_avail"].bool()[:, None, :].expand(-1, L, -1)
    if value == "input":
        # C' still reads the target depth in this arm; pin the C' encoder copy to the unshifted
        # batch so that only the offset add-back (c_out_t) sees the +1 log2 shift
        real = unet.model_covariates

        def pinned(b, *a):
            e_in, _, t_in, t_out = real(b, *a)
            return e_in, batch["c_out"], t_in, t_out
        monkeypatch.setattr(unet, "model_covariates", pinned)
    with torch.no_grad():
        o0 = m(batch)
        o1 = m(_shift_depth(batch, "c_out", "y_avail", 1.0))
    torch.testing.assert_close(o1["mu"][y].double() / o0["mu"][y].double(),
                               torch.full_like(o0["mu"][y].double(), 2.0), rtol=1e-6, atol=0)
    assert torch.equal(o1["n"], o0["n"])


@pytest.mark.parametrize("value", ["yes", "both", 1, 0, "Output"])
def test_other_key_values_are_rejected(batch, value):
    with pytest.raises(ValueError, match="hide_depth_covariate"):
        model_covariates(batch, "none", False, 0.0, value)
    # (a null key is dropped by unet._cfg and means the default, false, as for every key)
    cfg = dict(ARMS_BASE, context_bins=L, hide_depth_covariate=value)
    with pytest.raises(ValueError, match="hide_depth_covariate"):
        unet.Model(cfg).eval()(batch)


@pytest.mark.parametrize("value", ["output", "input"])
def test_split_arm_passes_the_wiring_check(batch, value):
    m = _perturbed(value)
    for i in range(2):
        assert C.wiring_check(m, batch, seed=i)["pass"]
