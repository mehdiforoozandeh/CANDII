"""miniCANDI rung 5 (axial transformer) on synthetic tables: shapes, positivity, parameter budget,
guardrail part 1 for the base and every supported arm, loss falls, gradients reach C and DNA."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

from minicandi import contract as C  # noqa: E402
from minicandi import synth  # noqa: E402
from minicandi.components import NotApplicable, TRANSFORMS  # noqa: E402
from minicandi.dataset import OnionDataset  # noqa: E402
from minicandi.rungs import axial  # noqa: E402
from onion_eval.guardrail import copy_leak  # noqa: E402

L = 64
BASE = json.loads((_ROOT / "tools/minicandi/configs/rung5/base.json").read_text())

ARMS = {
    "base": {},
    "concat": {"covariate_injection": "concat"},
    "film": {"covariate_injection": "film"},
    "mask_token": {"mask_mode": "token"},
    "dna_off": {"dna": "off"},
    "dna_shuffled": {"dna": "shuffled"},
    "axis_position_only": {"axis": "position_only"},
    "axis_track_only": {"axis": "track_only"},
    "axis_tied": {"axis": "tied_position"},
    "qc_on_input": {"qc_on_input": True},
    **{f"transform_{t}": {"count_transform": t} for t in TRANSFORMS if t != "offset"},
}


def cfg(**over) -> dict:
    return {**BASE, "context_bins": L, **over}


@pytest.fixture(scope="module")
def tables(tmp_path_factory) -> Path:
    return synth.make_synthetic_tables(tmp_path_factory.mktemp("rung5"))


def _batches(tables, n=2, **over):
    ds = OnionDataset(tables, "train", {**cfg(**over), "batch_size": 4}, seed=0)
    it = iter(ds)
    return [next(it) for _ in range(n)]


@pytest.fixture(scope="module")
def batches(tables):
    bs = _batches(tables)
    assert any(b["x_bin_mask"].any() for b in bs), "no hidden bins: guardrail would be vacuous"
    assert any((~b["x_avail"]).any() for b in bs), "no missing tracks: guardrail vacuous"
    return bs


@pytest.fixture(scope="module")
def qc_batches(tables):
    return _batches(tables, qc_on_input=True)


def _model(**over):
    torch.manual_seed(0)
    return axial.Model(cfg(**over)).eval()


def _live(**over):
    """NBHead's output layer is zero-initialised, so at initialisation every output is constant and
    a wiring check would pass vacuously. Give the head random weights first."""
    m = _model(**over)
    with torch.no_grad():
        torch.nn.init.normal_(m.head.lin.weight, std=0.5)
    return m


def test_base_config_is_rung5():
    assert BASE["rung"] == 5 and BASE["mask_mode"] == "drop"
    assert BASE["covariate_injection"] == "tokens" and BASE["axis"] == "both"
    assert BASE["context_bins"] % axial.PATCH == 0


def test_shapes_and_positivity(batches):
    m = _model()
    with torch.no_grad():
        out = m(batches[0])
    B = batches[0]["x_counts"].shape[0]
    for k in ("n", "mu"):
        assert out[k].shape == (B, L, C.A) and out[k].dtype == torch.float32
        assert torch.isfinite(out[k]).all() and (out[k] > 0).all()


@pytest.mark.parametrize("arm", list(ARMS))
def test_param_budget(arm):
    m = axial.Model(cfg(**ARMS[arm]))
    budget = BASE["param_budget"]
    assert abs(m.param_count() - budget) <= 0.10 * budget, (arm, m.param_count())


def test_param_budget_at_full_context():
    m = axial.Model({**BASE})
    assert BASE["context_bins"] == 1024
    assert abs(m.param_count() - BASE["param_budget"]) <= 0.10 * BASE["param_budget"]


@pytest.mark.parametrize("arm", list(ARMS))
def test_guardrail_part1(arm, batches, qc_batches):
    over = ARMS[arm]
    bs = qc_batches if over.get("qc_on_input") else batches
    m = _live(**over)
    for b in bs:
        w = C.wiring_check(m, b)
        assert w["pass"] and w["n_perturbed"] > 0, (arm, w)
    g = copy_leak(m, bs, seed=1)
    assert g["part1_wiring_pass"], (arm, g)
    assert g["hidden_input_max_abs_change"] <= 1e-6 and g["unavailable_input_max_abs_change"] <= 1e-6


def test_guardrail_detects_a_planted_leak(batches):
    """Negative control: with the bin mask bypassed, hidden counts reach the patch tokens and the
    check fails (so the passing checks above are not vacuous)."""
    m = _live()
    m.mask.forward = lambda h, avail, bmask: (h, bmask)
    w = C.wiring_check(m, batches[0])
    assert not w["pass"], w


def test_guardrail_part1_after_training(batches):
    """The wiring must hold for trained weights too, not only at initialisation."""
    for over in ({}, {"mask_mode": "token"}, {"axis": "tied_position"}):
        m = _model(**over).train()
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        for _ in range(3):
            b = batches[0]
            out = m(b)
            loss = C.nb_nll(out["n"], out["mu"], b["y_counts"], C.loss_mask(b))
            opt.zero_grad()
            loss.backward()
            opt.step()
        m.eval()
        g = copy_leak(m, batches, seed=2)
        assert g["part1_wiring_pass"], (over, g)


@pytest.mark.parametrize("arm", ["base", "axis_tied", "mask_token"])
def test_loss_falls(arm, batches):
    m = _model(**ARMS[arm]).train()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    b = batches[0]
    mask = C.loss_mask(b)
    losses = []
    for _ in range(15):
        out = m(b)
        loss = C.nb_nll(out["n"], out["mu"], b["y_counts"], mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < 0.9 * losses[0], losses


@pytest.mark.parametrize("inj", ["tokens", "film", "concat"])
def test_gradients_reach_covariates_and_dna(inj, batches):
    """NBHead's output layer is zero-initialised, so upstream gradients are exactly 0 before the
    first step; take one step, then every covariate encoder and the DNA tower must get gradient."""
    m = _model(covariate_injection=inj).train()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    b = batches[0]
    mask = C.loss_mask(b)
    for _ in range(2):
        out = m(b)
        loss = C.nb_nll(out["n"], out["mu"], b["y_counts"], mask)
        opt.zero_grad()
        loss.backward()
        if _ == 0:
            opt.step()
    for name, mod in (("cov_in", m.cov_in), ("cov_out", m.cov_out), ("dna", m.dna)):
        g = sum(float(p.grad.abs().sum()) for p in mod.parameters() if p.grad is not None)
        assert g > 0, (inj, name)


def test_dna_off_has_no_tower():
    m = axial.Model(cfg(dna="off"))
    assert sum(p.numel() for p in m.dna.parameters()) == 0


@pytest.mark.parametrize("over", [{"average_input": True}, {"two_step": True}])
def test_not_applicable(over):
    with pytest.raises(NotApplicable):
        axial.Model(cfg(**over))


def test_bad_axis_and_context():
    with pytest.raises(ValueError):
        axial.Model(cfg(axis="diagonal"))
    with pytest.raises(ValueError):
        axial.Model(cfg(context_bins=100))
