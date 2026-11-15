"""Rung 4 round-2 tuning keys `n_levels` and `attn_dropout`: default-off, and they work."""
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

from minicandi import dataset as D  # noqa: E402
from minicandi import synth  # noqa: E402
from minicandi.rungs import unet  # noqa: E402

BASE = json.loads((_ROOT / "tools/minicandi/configs/rung4/base.json").read_text())
TUNE = _ROOT / "tools/minicandi/configs/tune/rung4"
# Recorded with unet.py before the keys existed (commit af677fa), by the recipe of `_out`.
REF_PARAMS, REF_WIDTH = 993_599, 25
REF_MU_SUM, REF_N_SUM = 997.7176046455279, 18553.313170194626


def _cfg(**kw) -> dict:
    return dict(BASE, context_bins=64, **kw)


@pytest.fixture(scope="module")
def batch(tmp_path_factory):
    tables = synth.make_synthetic_tables(tmp_path_factory.mktemp("onion_tune_r4"))
    return D.OnionDataset(tables, "train", _cfg(), seed=0).batch(4, np.random.default_rng(0))


def _out(cfg: dict, batch, train: bool = False):
    torch.manual_seed(0)
    m = unet.Model(cfg)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.01 * torch.randn_like(p))
    m.train(train)
    with torch.no_grad():
        return m, m(batch)


@pytest.mark.parametrize("extra", [{}, {"n_levels": 4, "attn_dropout": 0.0}])
def test_default_is_the_pre_change_model(batch, extra):
    m, o = _out(_cfg(**extra), batch)
    assert (m.param_count(), m.width) == (REF_PARAMS, REF_WIDTH)
    assert o["mu"].double().sum().item() == pytest.approx(REF_MU_SUM, rel=1e-6)
    assert o["n"].double().sum().item() == pytest.approx(REF_N_SUM, rel=1e-6)


@pytest.mark.parametrize("nl", [3, 5])
def test_n_levels_builds_within_budget(batch, nl):
    m, o = _out(_cfg(n_levels=nl), batch)
    assert len(m.count_tower.blocks) == nl and len(m.dec1) == nl
    assert abs(m.param_count() - 1.0e6) <= 1.0e5
    assert o["mu"].shape == (4, 64, 36) and torch.isfinite(o["mu"]).all()


def test_attn_dropout_acts_in_train_mode_only(batch):
    m, _ = _out(_cfg(attn_dropout=0.1), batch)
    assert m.param_count() == REF_PARAMS
    assert all(layer.dropout.p == 0.1 for layer in m.attn.layers)
    m.eval()
    with torch.no_grad():
        a, b = m(batch)["mu"], m(batch)["mu"]
    assert torch.equal(a, b)
    m.train()
    torch.manual_seed(1)
    with torch.no_grad():
        c = m(batch)["mu"]
    assert not torch.equal(a, c)


@pytest.mark.parametrize("bad", [{"n_levels": 1}, {"n_levels": 6}, {"attn_dropout": 1.0}])
def test_bad_values_refused(bad):
    with pytest.raises(ValueError):
        unet.Model(_cfg(**bad))


def test_tune_configs_build_within_budget():
    for p in sorted(TUNE.glob("*.json")):
        if p.name.endswith(".template.json"):
            continue
        cfg = json.loads(p.read_text())
        assert abs(unet.Model(cfg).param_count() - 1.0e6) <= 1.0e5, p.name
