"""Rung 1 round-2 shape keys `mlp_depth` and `trunk_blocks` (default off).

(i) With the keys absent, or set to their defaults, the model is the pre-change model: the same
parameter count, state dict and output for a fixed seed (values pinned before the change).
(ii) The keys work: they change the layer structure, refit the width to the budget, keep the
receptive field at one bin, and every tune/rung1 config builds within 10% of 1.0 M.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

from minicandi import registry as R  # noqa: E402
from minicandi import synth  # noqa: E402
from minicandi import train as T  # noqa: E402
from minicandi.components import MaskHandler  # noqa: E402
from minicandi.dataset import OnionDataset  # noqa: E402

CFG = _ROOT / "tools" / "minicandi" / "configs"
ARMS_BASE = CFG / "arms" / "rung1" / "base.json"
SMALL_BASE = CFG / "rung1" / "base.json"
TUNE = CFG / "tune" / "rung1"
L = 64
# pinned with the code at 0e9abfc (before the shape keys existed)
PINNED_FULL = (328, 1001418, "4877250bf0d89d6a53e852fa0535a03b2e6b2bdbeddbc9fa86ecc899e1da10f2")
PINNED_SMALL = (24, 138058, {"n": 21800038.47496448, "mu": 1226279749.0808682})


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    return synth.make_synthetic_tables(tmp_path_factory.mktemp("onion_tune_r1"))


def _state_hash(m) -> str:
    h = hashlib.sha256()
    for k, v in m.state_dict().items():
        h.update(k.encode())
        h.update(v.numpy().tobytes())
    return h.hexdigest()


def _small(**kw):
    cfg = json.loads(SMALL_BASE.read_text())
    cfg.update(context_bins=L, param_budget=1.4e5, **kw)
    return cfg


def _jiggled_output(cfg, batch, scale=0.2):
    torch.manual_seed(0)
    m = R.build_model(cfg)
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(scale * torch.randn(p.shape, generator=g))
    m.eval()
    with torch.no_grad():
        return m, m(batch)


@pytest.mark.parametrize("kw", [{}, {"mlp_depth": 2, "trunk_blocks": 1}])
def test_default_full_model_is_the_pre_change_model(kw):
    cfg = dict(json.loads(ARMS_BASE.read_text()), **kw)
    torch.manual_seed(0)
    m = R.build_model(cfg)
    assert (m.d_model, m.param_count(), _state_hash(m)) == PINNED_FULL


@pytest.mark.parametrize("kw", [{}, {"mlp_depth": 2, "trunk_blocks": 1}])
def test_default_output_is_the_pre_change_output(tables, kw):
    cfg = _small(**kw)
    batch = OnionDataset(tables, "train", cfg, 0).batch(2, np.random.default_rng(2))
    m, out = _jiggled_output(cfg, batch)
    d, n, sums = PINNED_SMALL
    assert (m.d_model, m.param_count()) == (d, n)
    for k, v in sums.items():
        assert float(out[k].double().sum()) == pytest.approx(v, rel=1e-9)


def test_mlp_depth_adds_hidden_layers_and_refits_width():
    base = json.loads(ARMS_BASE.read_text())
    m = R.build_model(dict(base, mlp_depth=3))
    for blk in (m.phi, m.psi, m.trunk[2]):
        assert len(blk.mid) == 1
    assert m.d_model < PINNED_FULL[0]
    assert abs(m.param_count() - 1e6) <= 1e5


def test_trunk_blocks_zero_drops_the_trunk_block_and_widens():
    base = json.loads(ARMS_BASE.read_text())
    m = R.build_model(dict(base, trunk_blocks=0))
    assert len(m.trunk) == 2
    assert m.d_model > PINNED_FULL[0]
    assert abs(m.param_count() - 1e6) <= 1e5


@pytest.mark.parametrize("kw", [{"mlp_depth": 1}, {"trunk_blocks": -1}])
def test_bad_shape_values_are_refused(kw):
    with pytest.raises(ValueError):
        R.build_model(_small(**kw))


@pytest.mark.parametrize("kw", [{"mlp_depth": 3}, {"trunk_blocks": 0}])
def test_shape_keys_keep_the_one_bin_receptive_field(tables, kw):
    cfg = _small(**kw)
    batch = OnionDataset(tables, "train", cfg, 0).batch(1, np.random.default_rng(3))
    _, out = _jiggled_output(cfg, batch, 0.02)   # small: 0.2 saturates the mu clamp
    b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    vis = MaskHandler.visible(batch["x_avail"], batch["x_bin_mask"])[0].any(-1)   # [L]
    i = int(torch.nonzero(vis)[len(torch.nonzero(vis)) // 2])     # a bin with a visible track
    b2["x_counts"] = b2["x_counts"].clone()
    b2["x_counts"][:, i] = b2["x_counts"][:, i] + 50
    _, out2 = _jiggled_output(cfg, b2, 0.02)
    changed = (out["mu"] != out2["mu"]).any(-1)[0]
    assert changed[i]
    assert not changed[:i].any() and not changed[i + 1:].any()


def test_every_tune_config_builds_within_budget():
    paths = sorted(p for p in TUNE.glob("*.json") if not p.name.endswith(".template.json"))
    assert len(paths) == 8
    for p in paths:
        cfg = T.full_config(json.loads(p.read_text()))
        assert cfg["windows_seen"] == 40000 and cfg["heldout_loss"] == {"batches": 16, "every": 20}
        n = R.build_model(cfg).param_count()
        assert 900_000 <= n <= 1_100_000, p.name
