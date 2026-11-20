"""Rung 6 round-2 tuning keys `decoder_norm` and `unpatch_hidden` (default off).

(i) With the keys absent, or present at their defaults, the model is the pre-change model: the
parameter count, the state-dict names and the output on a fixed batch are pinned to values computed
with `perceiver.py` at commit 0e9abfc. (ii) Each key builds, runs and changes what it should.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

from minicandi import synth  # noqa: E402
from minicandi.dataset import OnionDataset  # noqa: E402
from minicandi.rungs import perceiver as R6  # noqa: E402

L = 64
BASE = {**json.loads((_ROOT / "tools/minicandi/configs/rung6/base.json").read_text()),
        "context_bins": L}
# computed with perceiver.py at 0e9abfc (before the tuning keys), same seeds and batch
PIN_PARAMS = 1_025_120
PIN_N_SUM, PIN_MU_SUM = 14796.8940876722, 782.0436826581


def _model(cfg: dict) -> R6.Model:
    torch.manual_seed(0)
    m = R6.Model(cfg)
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():                 # the NB head starts at zero: add noise so the output
        for p in m.parameters():          # depends on every parameter
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    return m.eval()


@pytest.fixture(scope="module")
def batch(tmp_path_factory) -> dict:
    tables = synth.make_synthetic_tables(tmp_path_factory.mktemp("tune6"))
    return OnionDataset(tables, "train", BASE, seed=0).batch(3)


def _n(m) -> int:
    return sum(p.numel() for p in m.parameters())


@pytest.mark.parametrize("extra", [{}, {"decoder_norm": False, "unpatch_hidden": 0}])
def test_default_is_the_pre_change_model(batch, extra):
    m = _model({**BASE, **extra})
    assert _n(m) == PIN_PARAMS
    assert m.dec_norm is None and isinstance(m.unpatch, torch.nn.Linear)
    with torch.no_grad():
        o = m(batch)
    assert o["n"].double().sum().item() == pytest.approx(PIN_N_SUM, rel=1e-6)
    assert o["mu"].double().sum().item() == pytest.approx(PIN_MU_SUM, rel=1e-6)
    ref = _model(BASE)
    assert list(m.state_dict()) == list(ref.state_dict())


def test_decoder_norm(batch):
    m = _model({**BASE, "decoder_norm": True})
    assert _n(m) == PIN_PARAMS + 2 * BASE["d_model"]
    with torch.no_grad():
        o, ref = m(batch), _model(BASE)(batch)
    assert o["mu"].shape == ref["mu"].shape and torch.isfinite(o["mu"]).all()
    assert not torch.equal(o["mu"], ref["mu"])


def test_unpatch_hidden(batch):
    d, P, dout, h = BASE["d_model"], BASE["patch_bins"], BASE["d_bin_out"], 128
    m = _model({**BASE, "unpatch_hidden": h})
    assert _n(m) == PIN_PARAMS - (d * P * dout + P * dout) + (d * h + h) + (h * P * dout + P * dout)
    with torch.no_grad():
        o = m(batch)
    assert o["n"].shape == (3, L, o["n"].shape[-1]) and torch.isfinite(o["n"]).all()
    with pytest.raises(ValueError):
        R6.Model({**BASE, "unpatch_hidden": -1})
