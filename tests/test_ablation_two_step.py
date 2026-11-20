"""Ablation 4 (two_step): the two-step wrapper on conv + attention (rung 3) and U-Net (rung 4),
which tests/test_minicandi_two_step.py does not cover (it covers rungs 1, 2 and 5), and the
written ablation configs (configs/ablations/two_step/rung{2,3,4})."""
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
from minicandi import losses  # noqa: E402
from minicandi import registry as R  # noqa: E402
from minicandi import synth  # noqa: E402
from minicandi import two_step as TS  # noqa: E402
from minicandi.dataset import OnionDataset  # noqa: E402

L = 64
ABL = _ROOT / "tools/minicandi/configs/ablations/two_step"
FINAL = _ROOT / "tools/minicandi/configs/final"
ARMS = {"two_step_native": {"two_step": True},
        "two_step_fixed": {"two_step": True, "base_depth": "fixed",
                           "base_ref_log2_depth": {"accessibility_dnase": 24.5,
                                                   "accessibility_atac": 25.0,
                                                   "broad_histone": 24.0, "narrow_histone": 23.5,
                                                   "control": 23.5}}}


def _small(r: int, arm: str) -> dict:
    cfg = json.loads((ABL / f"rung{r}/{arm}.json").read_text())
    cfg.update(context_bins=L, param_budget=3e5, batch_size=4)
    return cfg


@pytest.fixture(scope="module")
def tables(tmp_path_factory) -> Path:
    td = synth.make_synthetic_tables(tmp_path_factory.mktemp("abl_two_step"))
    synth.make_evalset(td, n_regions=2, region_bins=4 * L)
    return td


@pytest.mark.parametrize("r", (2, 3, 4))
@pytest.mark.parametrize("arm", tuple(ARMS))
def test_config_differs_from_reference_only_in_the_arm_keys(r, arm):
    ref = json.loads((FINAL / f"rung{r}/reference_equal.json").read_text())
    cfg = json.loads((ABL / f"rung{r}/{arm}.json").read_text())
    diff = {k for k in set(ref) | set(cfg) if ref.get(k) != cfg.get(k)}
    assert diff == set(ARMS[arm]) | {"simpler_than_base"}
    assert all(cfg[k] == v for k, v in ARMS[arm].items())


@pytest.mark.parametrize("r", (3, 4))
@pytest.mark.parametrize("arm", tuple(ARMS))
def test_wrapper_on_rungs_3_and_4(r, arm, tables):
    cfg = _small(r, arm)
    b = OnionDataset(tables, "train", cfg, 0).batch(4, np.random.default_rng(0))
    assert (b["c_base"][..., C.LOG2_DEPTH_COL] != 0).any()
    torch.manual_seed(0)
    m = R.build_model(cfg)
    assert isinstance(m, TS.TwoStep) and type(m.inner) is R.rung_module(r).Model
    assert m.param_count() == m.inner.param_count()          # shared weights
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    m.eval()
    out = m(b)
    assert set(out) == {"n", "mu", "n_base", "mu_base"}
    assert not torch.allclose(out["mu"], out["mu_base"])
    loss, parts = losses.total_loss(out, b, cfg)
    assert torch.isfinite(loss) and "base" in parts
    w = C.wiring_check(m, b)
    assert w["pass"] and w["n_perturbed"] > 0, w


def test_per_class_base_reference_depth(tables):
    """`base_ref_log2_depth` sets the fixed base depth per assay class; without it the base leg is
    byte-identical to the constant FIXED_BASE_LOG2_DEPTH; a map that misses a class is refused."""
    from minicandi import dataset as D
    base = {"context_bins": L, "two_step": True, "base_depth": "fixed"}
    rng = lambda: np.random.default_rng(3)  # noqa: E731
    old = OnionDataset(tables, "train", base, 3).batch(8, rng())
    same = OnionDataset(tables, "train", dict(base, base_ref_log2_depth=None), 3).batch(8, rng())
    const = dict(D.FIXED_BASE_LOG2_DEPTH)
    explicit = OnionDataset(tables, "train", dict(base, base_ref_log2_depth=const), 3).batch(8, rng())
    for b in (same, explicit):
        assert torch.equal(b["y_base"], old["y_base"]) and torch.equal(b["c_base"], old["c_base"])
    low = {k: 10.0 for k in C.ASSAY_CLASSES}
    low["narrow_histone"] = 12.0
    ds = OnionDataset(tables, "train", dict(base, base_ref_log2_depth=low), 3)
    b = ds.batch(8, rng())
    for bi, m in enumerate(b["meta"]):
        for t in np.nonzero(b["y_avail"][bi].numpy())[0]:
            want = low[C.assay_class(C.TRACKS[t])]
            assert float(b["c_base"][bi, t, C.LOG2_DEPTH_COL]) == pytest.approx(want)
    assert b["y_base"].sum() < old["y_base"].sum()
    with pytest.raises(ValueError):
        OnionDataset(tables, "train", dict(base, base_ref_log2_depth={"control": 20.0}), 3)
