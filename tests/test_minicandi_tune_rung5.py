"""Rung 5 round-2 tuning: the default-off shape keys `d_model`, `n_layers`, `n_heads` of rungs/axial.py.

(i) With the keys absent (or set to the module constants) the model is the pre-change model: the same
parameter count and the same output for a fixed seed, against axial.py at commit 0e9abfc.
(ii) The keys change the shape, keep the 10 % parameter band, and run a forward pass.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "tools"))

from minicandi import synth  # noqa: E402
from minicandi.dataset import OnionDataset  # noqa: E402
from minicandi.rungs import axial  # noqa: E402

L = 64
PRE_CHANGE = "0e9abfc"
BASE = json.loads((_ROOT / "tools/minicandi/configs/arms/rung5/base.json").read_text())


def cfg(**over) -> dict:
    return {**BASE, "context_bins": L, **over}


@pytest.fixture(scope="module")
def batch(tmp_path_factory):
    tables = synth.make_synthetic_tables(tmp_path_factory.mktemp("tune5"))
    ds = OnionDataset(tables, "train", {**cfg(), "batch_size": 4}, seed=0)
    return next(iter(ds))


@pytest.fixture(scope="module")
def pre_module(tmp_path_factory):
    try:
        src = subprocess.run(["git", "-C", str(_ROOT), "show",
                              f"{PRE_CHANGE}:tools/minicandi/rungs/axial.py"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"git show {PRE_CHANGE} unavailable")
    path = tmp_path_factory.mktemp("pre") / "axial_pre.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("axial_pre", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _out(module, config, batch):
    torch.manual_seed(0)
    m = module.Model(config).eval()
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():                       # NBHead starts at zero: make the output live
        for p in m.parameters():
            p.add_(torch.randn(p.shape, generator=g) * 0.05)
        out = m(batch)
    return sum(p.numel() for p in m.parameters()), out


@pytest.mark.parametrize("over", [{}, {"d_model": 128, "n_layers": 4, "n_heads": 8}])
def test_default_is_pre_change_model(over, batch, pre_module):
    n_old, o_old = _out(pre_module, cfg(), batch)
    n_new, o_new = _out(axial, cfg(**over), batch)
    assert n_new == n_old
    for k in ("n", "mu"):
        assert torch.equal(o_new[k], o_old[k])


@pytest.mark.parametrize("over,layers,d", [({"n_layers": 3}, 3, 128),
                                           ({"d_model": 96, "n_layers": 6}, 6, 96),
                                           ({"d_model": 160, "n_layers": 2}, 2, 160)])
def test_shape_keys_work(over, layers, d, batch):
    full = cfg(context_bins=1024, **over)
    torch.manual_seed(0)
    m = axial.Model(full)
    n = sum(p.numel() for p in m.parameters())
    assert len(m.layers) == layers and m.d == d
    assert 900_000 <= n <= 1_100_000
    assert m.d_ff != axial.Model(cfg(context_bins=1024)).d_ff
    small = axial.Model(cfg(**over)).eval()
    with torch.no_grad():
        out = small(batch)
    assert out["mu"].shape == batch["x_counts"].shape[:2] + (out["mu"].shape[-1],)
    assert bool((out["mu"] > 0).all()) and bool((out["n"] > 0).all())


def test_n_heads_must_divide_d_model():
    with pytest.raises(ValueError):
        axial.Model(cfg(d_model=100, n_heads=8))
