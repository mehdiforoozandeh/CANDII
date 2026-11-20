"""Rung-3 tuning knobs (`n_down`, `attn_layers`, `attn_heads`, `attn_dropout`): (i) absent keys
build today's model, parameter for parameter and output for output; (ii) each knob takes effect;
(iii) every round-2 tuning config builds within 10% of 1.0 M parameters."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
for p in (_ROOT / "tools", _ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from minicandi import registry  # noqa: E402
from minicandi.rungs import conv_attn_ae as R  # noqa: E402
from test_minicandi_rung3 import _cfg, _perturb, _toy_batch  # noqa: E402

ARMS_BASE = json.loads((_ROOT / "tools/minicandi/configs/arms/rung3/base.json").read_text())
TUNE = sorted((_ROOT / "tools/minicandi/configs/tune/rung3").glob("*.json"))
TUNE = [p for p in TUNE if not p.name.endswith(".template.json")]
BASE_COUNT = 1_038_018      # run.json param_count of rung3_base (all three seeds)


def _out(cfg, seed=0):
    torch.manual_seed(seed)
    m = R.Model(cfg)
    _perturb(m, seed)
    with torch.no_grad():
        o = m(_toy_batch())
    return m, o


def test_defaults_unchanged():
    m0, o0 = _out(_cfg())
    explicit = _cfg(n_down=R.N_DOWN, attn_layers=R.N_LAYERS, attn_heads=R.N_HEADS,
                    attn_dropout=0.0)
    m1, o1 = _out(explicit)
    assert m0.param_count() == m1.param_count() == BASE_COUNT
    assert m0.width == m1.width == 104
    for k in ("n", "mu"):
        assert torch.equal(o0[k], o1[k])
    sd0, sd1 = m0.state_dict(), m1.state_dict()
    assert sd0.keys() == sd1.keys() and all(torch.equal(sd0[k], sd1[k]) for k in sd0)


def test_knobs_take_effect():
    m, _ = _out(_cfg(attn_layers=4))
    assert len(m.attn.layers) == 4 and m.width < 104
    m, _ = _out(_cfg(n_down=3))
    assert len(m.enc) == len(m.down) == len(m.up) == len(m.dec) == 3
    m, _ = _out(_cfg(attn_heads=8))
    assert m.attn.layers[0].self_attn.num_heads == 8
    m, _ = _out(_cfg(attn_dropout=0.1))
    assert m.attn.layers[0].dropout.p == 0.1 and m.attn.layers[0].self_attn.dropout == 0.1
    # dropout acts in training mode only
    b = _toy_batch()
    m.train()
    torch.manual_seed(1)
    a = m(b)["mu"]
    torch.manual_seed(2)
    assert not torch.equal(a, m(b)["mu"])
    with pytest.raises(ValueError):
        R.Model(_cfg(attn_dropout=1.0))


@pytest.mark.parametrize("path", TUNE, ids=lambda p: p.stem)
def test_tune_config_builds_in_budget(path):
    cfg = json.loads(path.read_text())
    changed = {k for k in cfg if cfg[k] != ARMS_BASE.get(k)}
    assert "heldout_loss" in changed and changed - {"heldout_loss"}, changed
    assert cfg["windows_seen"] == 80000
    torch.manual_seed(0)
    n = registry.build_model(dict(cfg, context_bins=64)).param_count()
    assert 900_000 <= n <= 1_100_000
