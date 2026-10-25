"""t127 — rung A0 (`fforms/form_a0.py`) and the ppml loss (`model.Ladder(loss="ppml")`).

A0: mu = e^a X^b + e^c on the raw count X = expm1(x). ppml: Poisson NLL for the mean, NB NLL given
the detached mean for the dispersion. The nb default must be unchanged.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "t118"))

from ladder import base, model, pairs  # noqa: E402
from ladder.fforms import form_a0  # noqa: E402

STATS = {"knots_x": np.linspace(0.0, 5.0, 12), "n0": 5.0, "sigma0": 0.5}


def _x(X):
    return torch.log1p(torch.as_tensor(X, dtype=torch.float32))


def test_a0_contract():
    f = base.load_form("A0", "counts", STATS)
    assert isinstance(f, form_a0.FormA0)
    assert (f.rung, f.n_theta, f.context) == ("A0", 4, 0)
    assert torch.allclose(f.init_theta(), torch.tensor([0.0, 1.0, math.log(0.01), math.log(5.0)]))
    with pytest.raises(ValueError):
        form_a0.build("pval", STATS)
    with pytest.raises(ValueError):
        f(torch.zeros(2, 5), torch.zeros(2, 3))


def test_a0_identity_at_init_up_to_floor():
    f = form_a0.build("counts", STATS)
    X = torch.tensor([[0.0, 1.0, 2.0, 7.0, 150.0, 3e4]])
    loc, disp = f(_x(X), f.init_theta()[None])
    assert torch.allclose(loc.exp(), X + 0.01, rtol=1e-5)
    assert torch.allclose(disp, torch.full_like(X, math.log(5.0)))


def test_a0_power_law_gives_r_times_x():
    f = form_a0.build("counts", STATS)
    r = 2.5
    X = torch.tensor([[0.0, 1.0, 3.0, 40.0, 900.0]])
    theta = torch.tensor([[math.log(r), 1.0, -60.0, 0.0]])
    loc, _ = f(_x(X), theta)
    mu = loc.exp()
    assert torch.allclose(mu[:, 1:], r * X[:, 1:], rtol=1e-5)
    assert mu[0, 0] < 1e-20                    # X = 0: only the (vanishing) floor


def test_a0_gradient_finite_at_zero_counts():
    f = form_a0.build("counts", STATS)
    theta = f.init_theta()[None].clone().requires_grad_(True)
    loc, disp = f(_x([[0.0, 0.0, 5.0]]), theta)
    model.nb_nll(loc, disp, torch.tensor([[0.0, 1.0, 4.0]])).sum().backward()
    assert torch.isfinite(theta.grad).all()


def _ladder(loss, rung="A"):
    return model.Ladder(rung, "counts", STATS, loss=loss)


def test_ppml_loc_gradient_is_poisson():
    lad = _ladder("ppml")
    y = torch.tensor([0.0, 1.0, 4.0, 30.0, 2.0], dtype=torch.float64)
    loc = torch.tensor([0.3, -1.0, 1.2, 3.0, 0.0], dtype=torch.float64, requires_grad=True)
    disp = torch.full((5,), 0.7, dtype=torch.float64, requires_grad=True)
    lad.nll_bins(loc, disp, y).sum().backward()
    assert torch.allclose(loc.grad, loc.detach().exp() - y)
    assert disp.grad.abs().sum() > 0


def test_ppml_nb_term_sends_no_gradient_to_loc():
    y = torch.tensor([0.0, 3.0, 9.0], dtype=torch.float64)
    loc = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64, requires_grad=True)
    disp = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    per = _ladder("ppml").nll_bins(loc, disp, y) - model.poisson_nll(loc, y)
    per.sum().backward()
    assert torch.equal(loc.grad, torch.zeros(3, dtype=torch.float64))
    # the value is the NB NLL; the disp gradient is the NB one
    assert torch.allclose(per, model.nb_nll(loc.detach(), disp.detach(), y))
    d2 = disp.detach().clone().requires_grad_(True)
    model.nb_nll(loc.detach(), d2, y).sum().backward()
    assert torch.allclose(disp.grad, d2.grad)


def test_ppml_terms_and_counts_only():
    lad = _ladder("ppml")
    y = torch.tensor([[0.0, 2.0, 5.0]])
    loc, disp = torch.tensor([[0.1, 0.5, 1.5]]), torch.tensor([[1.0, 1.0, 1.0]])
    t = lad.nll_terms(loc, disp, y)
    assert math.isclose(t["loss_pois"] + t["loss_nb"], float(lad.nll(loc, disp, y)), rel_tol=1e-6)
    with pytest.raises(ValueError):
        model.Ladder("A", "pval", STATS, loss="ppml")
    with pytest.raises(ValueError):
        model.Ladder("A", "counts", STATS, loss="poisson")


def test_nb_default_unchanged():
    torch.manual_seed(0)
    lad, lad_nb = model.Ladder("A", "counts", STATS), _ladder("nb")
    assert lad.loss == "nb"
    loc, disp = torch.randn(3, 50), torch.randn(3, 50)
    y = torch.poisson(torch.full((3, 50), 3.0))
    mask = torch.rand(3, 50) > 0.3
    per = model.nb_nll(loc, disp, y)
    want = (per * mask.float()).sum() / mask.float().sum()
    assert torch.equal(lad.nll(loc, disp, y, mask), want)
    assert torch.equal(lad_nb.nll(loc, disp, y), per.mean())


def test_ppml_task_table():
    t = pairs.tasks_ppml()
    assert len(t) == 126
    assert {r["rung"] for r in t} == {"A", "A0", "D"}
    assert t[0]["run_name"] == "A_C07M20_counts_real_s0"
    assert t[21]["run_name"] == "A_C07M20_counts_real_ppml_s0"
    assert t[42 + 21 + 3 * 3 + 2]["run_name"] == "A0_C19M16_counts_real_ppml_s2"
    assert all(r["space"] == "counts" and r["model"] == "real" and r["g"] in pairs.TRACKS for r in t)
