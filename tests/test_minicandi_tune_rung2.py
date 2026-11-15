"""Rung 2 tuning structure keys (`n_levels`, `kernel`, `blocks_per_level`), default-off.

(i) With the keys absent, conv_ae.Model has the same parameter count, the same parameters and the
same output for a fixed seed as the module before the change (commit 0e9abfc, read from git).
(ii) Each key builds what it names, stays inside the 10% parameter band, and runs forward.
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
for _p in (_ROOT / "tools", _ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from minicandi import contract as C  # noqa: E402
from minicandi.rungs import conv_ae  # noqa: E402
from test_minicandi_rung2 import _perturb, _toy_batch  # noqa: E402

BASE = json.loads((_ROOT / "tools/minicandi/configs/arms/rung2/base.json").read_text())
PRE_CHANGE = "0e9abfc"


def _old_module(tmp_path):
    try:
        src = subprocess.run(["git", "-C", str(_ROOT), "show",
                              f"{PRE_CHANGE}:tools/minicandi/rungs/conv_ae.py"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"git or commit {PRE_CHANGE} not available")
    path = tmp_path / "conv_ae_pre_change.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("conv_ae_pre_change", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build(mod, cfg, seed=0):
    torch.manual_seed(seed)
    return mod.Model(cfg)


def test_default_is_bit_identical_to_pre_change(tmp_path):
    old = _old_module(tmp_path)
    m_old, m_new = _build(old, BASE), _build(conv_ae, BASE)
    assert m_new.param_count() == m_old.param_count() == 1042514
    assert m_new.width == m_old.width
    so, sn = m_old.state_dict(), m_new.state_dict()
    assert list(so) == list(sn)
    for k in so:
        assert torch.equal(so[k], sn[k]), k
    _perturb(m_old), _perturb(m_new)
    m_old.eval(), m_new.eval()
    b = _toy_batch()
    with torch.no_grad():
        oo, on = m_old(b), m_new(b)
    for k in ("mu", "n"):
        assert torch.equal(oo[k], on[k]), k


@pytest.mark.parametrize("change,check", [
    ({"n_levels": 3}, lambda m: len(m.enc) == 3 and len(m.dec) == 3),
    ({"n_levels": 5}, lambda m: len(m.enc) == 5 and len(m.up) == 5),
    ({"kernel": 3}, lambda m: m.enc[0][0].conv.kernel_size == (3,)
     and m.mid[0].conv.kernel_size == (3,)),
    ({"kernel": 9}, lambda m: m.dec[-1][1].conv.kernel_size == (9,)),
    ({"blocks_per_level": 1}, lambda m: all(len(b) == 1 for b in m.enc)),
    ({"blocks_per_level": 3}, lambda m: [blk.conv.dilation[0] for blk in m.dec[0]] == [1, 2, 4]),
])
def test_structure_key_builds_what_it_names(change, check):
    m = _build(conv_ae, {**BASE, **change})
    assert check(m)
    assert abs(m.param_count() - 1.0e6) <= 0.1e6
    m.eval()
    b = _toy_batch()                                   # L = 48: exercises the padding path
    with torch.no_grad():
        out = m(b)
    assert out["mu"].shape == (3, 48, C.A) and bool((out["mu"] > 0).all())


@pytest.mark.parametrize("bad", [{"kernel": 4}, {"n_levels": 0}, {"blocks_per_level": 5}])
def test_bad_structure_value_is_refused(bad):
    with pytest.raises(ValueError):
        conv_ae.Model({**BASE, **bad}, width=60)
