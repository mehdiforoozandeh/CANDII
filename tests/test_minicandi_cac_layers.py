"""Pre and post covariate-correction layers (t370 chunk C2, decision D5): `minicandi.cac_layers`
and its place in `registry.build_model`."""
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

from minicandi import cac_layers as CAC  # noqa: E402
from minicandi import contract as C  # noqa: E402
from minicandi import registry as R  # noqa: E402
from minicandi import synth  # noqa: E402
from minicandi import train as T  # noqa: E402
from minicandi import two_step as TS  # noqa: E402
from minicandi.dataset import OnionDataset  # noqa: E402

L = 64
RUNGS = (2, 3, 4)
SIDES = ("cac_pre", "cac_post")


def _cfg(r: int, **over) -> dict:
    cfg = json.loads((_ROOT / f"tools/minicandi/configs/rung{r}/base.json").read_text())
    cfg.update(context_bins=L, param_budget=5e5, batch_size=4)
    cfg.update(over)
    return cfg


def _heldout(r: int, **over) -> dict:
    cfg = json.loads((_ROOT / f"tools/minicandi/configs/rung{r}/base_heldout.json").read_text())
    cfg.update(over)
    return cfg


def _layers(r: int, **keys):
    """CacLayers around a small rung; built directly, since the small budget is not the rule."""
    torch.manual_seed(0)
    inner = R.build_model(_cfg(r))
    torch.manual_seed(1)
    return inner, CAC.CacLayers(inner, keys)


@pytest.fixture(scope="module")
def tables(tmp_path_factory) -> Path:
    td = synth.make_synthetic_tables(tmp_path_factory.mktemp("onion_cac"))
    synth.make_evalset(td, n_regions=2, region_bins=4 * L)
    return td


@pytest.fixture(scope="module")
def batch(tables) -> dict:
    b = OnionDataset(tables, "train", _cfg(2, two_step=True), 0).batch(
        4, np.random.default_rng(0))
    C.validate_batch(b)
    assert b["x_bin_mask"].any() and (~b["x_avail"]).any()
    return b


# ------------------------------------------------------------------- defaults off


@pytest.mark.parametrize("r", RUNGS)
def test_keys_absent_or_null_build_todays_model(r, batch):
    torch.manual_seed(0)
    a = R.build_model(_cfg(r)).eval()
    torch.manual_seed(0)
    b = R.build_model(_cfg(r, cac_pre=None, cac_post=None)).eval()
    assert type(a) is type(b) is R.rung_module(r).Model
    assert list(a.state_dict()) == list(b.state_dict())
    with torch.no_grad():
        oa, ob = a(batch), b(batch)
    for k in ("n", "mu"):
        assert torch.equal(oa[k], ob[k])


@pytest.mark.parametrize("r", RUNGS)
def test_base_heldout_parameter_counts_unchanged(r):
    want = {2: 1_042_514, 3: 1_038_018, 4: 993_599}[r]
    torch.manual_seed(0)
    assert R.build_model(_heldout(r)).param_count() == want


def test_unknown_value_is_refused():
    inner = R.build_model(_cfg(2))
    with pytest.raises(ValueError, match="cac_pre"):
        CAC.wrap(inner, {"cac_pre": "spline"})


# ------------------------------------------------------------------- identity at the start


@pytest.mark.parametrize("r", RUNGS)
@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("form", CAC.FORMS)
def test_each_form_on_each_side_is_the_identity_at_start(r, side, form, batch):
    inner, m = _layers(r, **{side: form})
    inner.eval(), m.eval()
    with torch.no_grad():
        want, got = inner(batch), m(batch)
    for k in ("n", "mu"):
        rel = ((got[k] - want[k]).abs() / want[k].abs()).max().item()
        assert rel <= 1e-5, (k, rel)


def test_side_sizes_are_pinned():
    side = lambda f, k: sum(p.numel() for p in CAC.CacSide(f, k).parameters())  # noqa: E731
    assert side("curve", C.K_META) == 15_784
    assert side("cnn", C.K_META) == 23_234


# ------------------------------------------------------------------- guardrails


@pytest.mark.parametrize("r", RUNGS)
def test_wiring_check_passes_with_pre_cnn_after_training_like_perturbation(r, batch):
    _, m = _layers(r, cac_pre="cnn", cac_post="cnn")
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    m.eval()
    w = C.wiring_check(m, batch)
    assert w["pass"] and w["n_perturbed"] > 0, w


@pytest.mark.parametrize("form", CAC.FORMS)
def test_g_reads_no_counts_and_layers_change_the_output_after_training(form, batch):
    _, m = _layers(2, cac_pre=form, cac_post=form)
    seen = []
    for s in (m.pre, m.post):
        s.g.register_forward_hook(lambda mod, inp, out: seen.append(out.detach().clone()))
    g = torch.Generator().manual_seed(4)
    with torch.no_grad():
        for p in list(m.pre.parameters()) + list(m.post.parameters()):
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    m.eval()
    pert = dict(batch)
    pert["x_counts"] = torch.randint(0, 500, batch["x_counts"].shape, generator=g,
                                     dtype=torch.int32)
    with torch.no_grad():
        m(batch)
        m(pert)
    assert torch.equal(seen[0], seen[2]) and torch.equal(seen[1], seen[3])
    # the perturbed layers move the output away from the inner model's
    with torch.no_grad():
        assert not torch.allclose(m(batch)["mu"], m.inner(batch)["mu"])


def test_g_input_is_the_raw_covariates(batch):
    _, m = _layers(2, cac_pre="curve", cac_post="curve")
    with torch.no_grad():
        th_in = m.pre.theta(batch["c_in"])
        th_out = m.post.theta(batch["c_out"])
    assert th_in.shape == (batch["c_in"].shape[0], C.A, m.pre.f.n_theta)
    assert th_out.shape == (batch["c_out"].shape[0], C.A, m.post.f.n_theta)


# ------------------------------------------------------------------- registry and composition


@pytest.mark.parametrize("r", RUNGS)
def test_registry_wraps_in_order_and_composes_with_two_step(r, batch):
    torch.manual_seed(0)
    m = R.build_model(_cfg(r, cac_pre="curve", cac_post="curve", two_step=True))
    assert isinstance(m, TS.TwoStep) and isinstance(m.inner, CAC.CacLayers)
    assert type(m.inner.inner) is R.rung_module(r).Model
    m.eval()
    with torch.no_grad():
        out = m(batch)
    assert set(out) == {"n", "mu", "n_base", "mu_base"}
    assert all(torch.isfinite(v).all() for v in out.values())


def test_check_config_keys_accepts_both_keys():
    T.check_config_keys(_cfg(2, cac_pre="cnn", cac_post="curve"))


def test_load_run_rebuilds_the_wrapped_model(tmp_path, batch):
    cfg = _cfg(2, cac_pre="cnn", cac_post="curve")
    torch.manual_seed(0)
    m = R.build_model(cfg)
    torch.save(C.make_ckpt(m, cfg, 0, "x", "y"), tmp_path / R.CKPT_NAME)
    m2, _ = R.load_run(tmp_path)
    assert isinstance(m2, CAC.CacLayers)
    with torch.no_grad():
        assert torch.equal(m.eval()(batch)["mu"], m2(batch)["mu"])


# ------------------------------------------------------------------- the parameter rule


@pytest.mark.parametrize("r", RUNGS)
@pytest.mark.parametrize("form", CAC.FORMS)
def test_both_sides_fit_the_budget_on_base_heldout(r, form):
    torch.manual_seed(0)
    cfg = _heldout(r, cac_pre=form, cac_post=form)
    m = R.build_model(cfg)
    assert isinstance(m, CAC.CacLayers)
    assert m.param_count() == m.inner.param_count() + sum(
        p.numel() for s in (m.pre, m.post) for p in s.parameters())
    assert m.param_count() <= 1.10 * cfg["param_budget"]


def test_over_budget_raises():
    torch.manual_seed(0)
    with pytest.raises(ValueError, match="param_budget"):
        R.build_model(_cfg(2, cac_pre="cnn", cac_post="cnn", param_budget=3e5))


def test_curve_extrapolation_overflow_gives_finite_gradients():
    """t370: FormB extrapolates past its last knot; an uncapped expm1/exp overflowed to inf and the
    clamp then gave a 0 x inf = NaN gradient (U-Net curve runs died at steps 6,505-9,765)."""
    loc = torch.full((4,), 200.0, requires_grad=True)
    disp = torch.full((4,), 200.0, requires_grad=True)
    mu = torch.expm1(loc.clamp(max=CAC.LOC_CAP)).clamp(C.MU_MIN, C.MU_MAX)
    n = (torch.ones(4) * torch.exp(disp.clamp(-CAC.DISP_CAP, CAC.DISP_CAP))).clamp(C.N_MIN, C.N_MAX)
    (mu.sum() + n.sum()).backward()
    assert torch.isfinite(mu).all() and torch.isfinite(n).all()
    assert torch.isfinite(loc.grad).all() and torch.isfinite(disp.grad).all()
    assert abs(CAC.LOC_CAP - np.log1p(C.MU_MAX)) < 1e-12
