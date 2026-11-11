"""Rung 4 (U-Net with skip-source arms): shapes, budget, copy-leak wiring, learning, gradients."""
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
from minicandi.components import MaskHandler, NotApplicable  # noqa: E402
from minicandi.rungs import unet  # noqa: E402
from onion_eval.guardrail import copy_leak  # noqa: E402

L = 64
B = 4
BASE = json.loads((_ROOT / "tools/minicandi/configs/rung4/base.json").read_text())
SKIPS = ("both", "dna", "counts")
ARMS = {
    "inject_concat": {"covariate_injection": "concat"},
    "inject_tokens": {"covariate_injection": "tokens"},
    "mask_drop": {"mask_mode": "drop"},
    "dna_off": {"dna": "off"},
    "dna_shuffled": {"dna": "shuffled"},
    "qc_on_input": {"qc_on_input": True},
    "average_input": {"average_input": True},
    **{f"transform_{t}": {"count_transform": t}
       for t in ("log1p", "asinh", "anscombe", "pearson", "local_ratio", "quantile", "raw")},
}


def _cfg(**kw) -> dict:
    cfg = dict(BASE, context_bins=L)
    cfg.update(kw)
    return cfg


@pytest.fixture(scope="module")
def tables(tmp_path_factory) -> Path:
    return synth.make_synthetic_tables(tmp_path_factory.mktemp("onion_r4"))


def _batches(tables, cfg, n=2, seed=0):
    ds = D.OnionDataset(tables, "train", cfg, seed=seed)
    rng = np.random.default_rng(seed)
    out = [ds.batch(B, rng) for _ in range(n)]
    for b in out:
        C.validate_batch(b)
    return out


@pytest.fixture(scope="module")
def batch(tables):
    return _batches(tables, _cfg(), n=1)[0]


def _model(**kw) -> unet.Model:
    torch.manual_seed(0)
    return unet.Model(_cfg(**kw))


def _jitter(m: unet.Model) -> unet.Model:
    """Add noise to every parameter. At initialisation the NB head's Linear is zero, so the
    output ignores every feature and a wiring check would pass vacuously."""
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    return m


def test_base_config_is_rung4_with_both_skips():
    assert BASE["rung"] == 4 and BASE["skip_source"] == "both"
    assert BASE["covariate_injection"] == "film" and BASE["mask_mode"] == "token"


@pytest.mark.parametrize("skip", SKIPS)
def test_output_shapes_positive_and_in_budget(batch, skip):
    m = _model(skip_source=skip).eval()
    out = m(batch)
    assert set(out) == {"n", "mu"}
    for k in ("n", "mu"):
        assert out[k].shape == (B, L, C.A) and out[k].dtype == torch.float32
        assert torch.isfinite(out[k]).all() and (out[k] > 0).all()
    budget = BASE["param_budget"]
    assert abs(m.param_count() - budget) <= 0.10 * budget, m.param_count()


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_every_arm_is_in_budget(arm):
    m = _model(**ARMS[arm])
    assert abs(m.param_count() - BASE["param_budget"]) <= 0.10 * BASE["param_budget"]


def _wiring(tables, cfg):
    m = _jitter(unet.Model(cfg)).eval()
    bs = _batches(tables, cfg, n=2, seed=3)
    for i, b in enumerate(bs):
        assert b["x_bin_mask"].any() and (~b["x_avail"]).any()
        w = C.wiring_check(m, b, seed=i)
        assert w["pass"], w
    g = copy_leak(m, bs, seed=0)
    assert g["part1_wiring_pass"], g
    assert g["hidden_input_max_abs_change"] <= 1e-6
    assert g["unavailable_input_max_abs_change"] <= 1e-6


@pytest.mark.parametrize("skip", SKIPS)
def test_guardrail_part1_every_skip_source(tables, skip):
    torch.manual_seed(1)
    _wiring(tables, _cfg(skip_source=skip))


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_guardrail_part1_every_arm_with_count_skips(tables, arm, tmp_path):
    cfg = _cfg(**ARMS[arm])
    if cfg.get("average_input"):
        cfg["average_h5"] = str(D.build_average_h5(tables, tmp_path / "avg.h5"))
    torch.manual_seed(2)
    _wiring(tables, cfg)


def test_wiring_check_has_power_against_an_unmasked_count_skip(batch):
    """Bypass the mask: the count skip then carries hidden values and the check must fail."""
    m = _jitter(_model(skip_source="counts")).eval()
    assert C.wiring_check(m, batch)["pass"]
    m.mask.forward = lambda h, a, mk: (h, MaskHandler.visible(a, mk))
    assert not C.wiring_check(m, batch)["pass"]


def test_skip_dna_with_dna_off_is_not_applicable():
    with pytest.raises(NotApplicable):
        unet.Model(_cfg(skip_source="dna", dna="off"))
    m = _model(skip_source="both", dna="off")       # the dna_off arm of the base
    assert m.skip_counts and not m.skip_dna


def test_bad_keys_raise():
    with pytest.raises(ValueError):
        unet.Model(_cfg(skip_source="latent"))
    with pytest.raises(ValueError):
        unet.Model(_cfg(rung=3))


def _train(m, batches, steps, lr=1e-3):
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    m.train()
    losses = []
    for s in range(steps):
        b = batches[s % len(batches)]
        out = m(b)
        loss = C.nb_nll(out["n"], out["mu"], b["y_counts"], C.loss_mask(b))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


@pytest.mark.parametrize("skip", SKIPS)
def test_optimizer_steps_lower_the_nb_loss(tables, skip):
    bs = _batches(tables, _cfg(), n=2, seed=5)
    m = _model(skip_source=skip)
    m.eval()

    def held(b):
        with torch.no_grad():
            o = m(b)
            return C.nb_nll(o["n"], o["mu"], b["y_counts"], C.loss_mask(b)).item()

    before = np.mean([held(b) for b in bs])
    _train(m, bs, 30)
    m.eval()
    after = np.mean([held(b) for b in bs])
    assert after < 0.9 * before, (before, after)


@pytest.mark.parametrize("injection", ("film", "concat", "tokens"))
def test_gradients_reach_covariate_encoders_and_dna_tower(batch, injection):
    m = _model(covariate_injection=injection)
    _train(m, [batch], 2)                    # zero-initialised heads open after one step
    out = m(batch)
    C.nb_nll(out["n"], out["mu"], batch["y_counts"], C.loss_mask(batch)).backward()

    def gnorm(mod):
        return sum(float(p.grad.abs().sum()) for p in mod.parameters() if p.grad is not None)

    assert gnorm(m.in_inj.embed) > 0
    assert gnorm(m.out_inj.embed) > 0
    assert gnorm(m.dna) > 0 and gnorm(m.dna_tower) > 0


def test_dna_off_has_no_dna_parameters(batch):
    m = _model(dna="off")
    assert not list(m.dna.parameters()) and not hasattr(m, "dna_tower")
    assert m(batch)["mu"].shape == (B, L, C.A)


@pytest.mark.parametrize("Lx", (50, 37, 16))
def test_context_not_a_multiple_of_the_stride_works(batch, Lx):
    b = dict(batch)
    for k in ("x_counts", "x_bin_mask", "y_counts"):
        b[k] = batch[k][:, :Lx].contiguous()
    b["x_dna"] = batch["x_dna"][:, :Lx * C.RES].contiguous()
    m = _jitter(_model()).eval()
    out = m(b)
    assert out["n"].shape == (B, Lx, C.A) and (out["mu"] > 0).all()
    assert C.wiring_check(m, b)["pass"]


def test_explicit_width_and_registry_free_build():
    m = unet.build(_cfg(width=8))
    assert m.width == 8 and isinstance(m, C.MiniCandi)
    assert unet.fit_width(_cfg()) == _model().width
