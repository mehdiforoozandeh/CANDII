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
