"""t118 rung X (exploratory) — `tools/t118/ladder/fforms/form_x.py`: g reads the bin value.

The unit tests train theta and X's own MLP directly with a local log-normal NLL (no harness). The
last test runs the real `train.py train` + `score.py trained` CLIs in-process on synthetic products
(`smoke.py` accepts only rungs A-D, so it is not used).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "t118"))

from ladder import base  # noqa: E402
from ladder.fforms import form_x  # noqa: E402

STATS = {"knots_x": np.linspace(0.0, 5.0, 12), "n0": 5.0, "sigma0": 0.5}


def _form(space="counts", seed=0):
    torch.manual_seed(seed)
    return base.load_form("X", space, STATS)


def _randomise_last(form, seed=1):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        form.h[4].weight.copy_(0.3 * torch.randn(form.h[4].weight.shape, generator=g))
        form.h[4].bias.copy_(0.1 * torch.randn(form.h[4].bias.shape, generator=g))


def _lognormal_nll(loc, disp, t):
    """Per-bin log-normal NLL on t = log y (up to the Jacobian term, which is constant in theta)."""
    return disp + 0.5 * ((t - loc) * torch.exp(-disp)) ** 2 + 0.5 * math.log(2 * math.pi)


# ---- contract --------------------------------------------------------------------------------

@pytest.mark.parametrize("space", ["counts", "pval"])
def test_load_form_and_attributes(space):
    f = _form(space)
    assert isinstance(f, form_x.FormX) and isinstance(f, base.FForm)
    assert (f.rung, f.space, f.n_theta, f.context) == ("X", space, 32, 0)
    assert torch.equal(f.init_theta(), torch.zeros(32))
    names = [type(m).__name__ for m in f.h]
    assert names == ["Linear", "GELU", "Linear", "GELU", "Linear"]
    assert [(m.in_features, m.out_features) for m in f.h if isinstance(m, torch.nn.Linear)] == \
        [(33, 64), (64, 64), (64, 2)]
    assert torch.count_nonzero(f.h[4].weight) == 0 and torch.count_nonzero(f.h[4].bias) == 0


@pytest.mark.parametrize("space", ["counts", "pval"])
def test_shapes_and_bad_input(space):
    f = _form(space)
    x = torch.randn(3, 500)
    loc, disp = f(x, torch.randn(3, 32))
    assert loc.shape == (3, 500) and disp.shape == (3, 500)
    with pytest.raises(ValueError):
        f(x, torch.randn(3, 31))
    with pytest.raises(ValueError):
        f(x.unsqueeze(0), torch.randn(3, 32))


@pytest.mark.parametrize("space", ["counts", "pval"])
def test_identity_at_init(space):
    f = _form(space)
    x = torch.randn(4, 777) * 2
    theta = torch.randn(4, 32)          # any theta: the zero last layer makes it the identity
    loc, disp = f(x, theta)
    d0 = math.log(STATS["n0"] if space == "counts" else STATS["sigma0"])
    assert torch.equal(loc, x)
    assert torch.allclose(disp, torch.full_like(disp, d0), atol=0, rtol=0)
    loc0, disp0 = f(x, f.init_theta().expand(4, -1))
    assert torch.equal(loc0, x) and torch.allclose(disp0, torch.full_like(disp0, d0))


def test_matches_mlp_on_concatenation_and_is_per_bin():
    """The split first layer equals h applied to [x_i, theta]; a change in one bin moves only it."""
    f = _form("pval")
    _randomise_last(f)
    x, theta = torch.randn(2, 64), torch.randn(2, 32)
    loc, disp = f(x, theta)
    inp = torch.cat([x.unsqueeze(-1), theta.unsqueeze(1).expand(-1, 64, -1)], dim=-1)
    out = f.h(inp)
    assert torch.allclose(loc, x + out[..., 0], atol=1e-6)
    assert torch.allclose(disp, f.disp0 + out[..., 1], atol=1e-6)
    x2 = x.clone()
    x2[0, 10] += 1.0
    loc2, disp2 = f(x2, theta)
    changed = ((loc2 - loc).abs() + (disp2 - disp).abs()) > 0
    assert changed[0, 10] and changed.sum() == 1
    # theta changes the map
    loc3, _ = f(x, theta + 1.0)
    assert (loc3 - loc).abs().max() > 1e-4


def test_gradients_reach_theta_and_h():
    f = _form("counts")
    theta = torch.zeros(2, 32, requires_grad=True)
    x = torch.randn(2, 300)
    target = x + 0.7 * torch.tanh(x)
    opt = torch.optim.Adam([theta, *f.parameters()], lr=1e-2)

    def step():
        opt.zero_grad()
        loc, disp = f(x, theta)
        (((loc - target) ** 2).mean() + (disp ** 2).mean()).backward()

    step()                               # step 1: only the zero last layer receives gradient
    assert f.h[4].weight.grad.abs().sum() > 0
    assert theta.grad.abs().sum() == 0
    opt.step()
    step()                               # step 2: everything does
    assert theta.grad.abs().sum() > 0
    for name, p in f.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name
    assert {n for n, _ in f.named_parameters()} == {
        f"h.{i}.{k}" for i in (0, 2, 4) for k in ("weight", "bias")}


def test_describe():
    f = _form("pval")
    t = torch.arange(32, dtype=torch.float32) / 10
    d = f.describe(t)
    assert set(d) == {"theta_norm", "theta"}
    assert len(d["theta"]) == 32 and all(isinstance(v, float) for v in d["theta"])
    assert d["theta_norm"] == pytest.approx(float(t.double().norm()))
    json.dumps(d)
    with pytest.raises(ValueError):
        f.describe(torch.zeros(31))


# ---- heteroscedastic recovery ------------------------------------------------------------------

def _sigma_true(x):
    """log sigma linear in x: sigma 0.1 at x = -3, 1.5 at x = 3."""
    return 0.1 * np.exp((x + 3.0) / 6.0 * math.log(15.0))


def test_heteroscedastic_sigma_recovery():
    """pval-like data, t = log target = x + 0.3 + sigma(x) * eps. X learns sigma(x); a constant-sigma
    fit (rung A's disp) cannot."""
    rng = np.random.default_rng(0)
    n = 40_000
    xs = rng.uniform(-3.0, 3.0, n)
    ts = xs + 0.3 + _sigma_true(xs) * rng.standard_normal(n)
    x = torch.tensor(xs, dtype=torch.float32).unsqueeze(0)
    t = torch.tensor(ts, dtype=torch.float32).unsqueeze(0)

    f = _form("pval", seed=0)
    theta = f.init_theta().unsqueeze(0).clone().requires_grad_(True)
    opt = torch.optim.Adam([theta, *f.parameters()], lr=3e-3)
    for _ in range(800):
        opt.zero_grad()
        loc, disp = f(x, theta)
        loss = _lognormal_nll(loc, disp, t).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        loc, disp = f(x, theta)
        nll_x = float(_lognormal_nll(loc, disp, t).mean())

    # constant-sigma fit, closed form: loc = x + a, sigma = the residual std (the MLE)
    r = ts - xs
    a, s = r.mean(), r.std()
    nll_const = float(np.mean(np.log(s) + 0.5 * ((r - a) / s) ** 2 + 0.5 * math.log(2 * math.pi)))

    grid = np.linspace(-2.7, 2.7, 55)
    with torch.no_grad():
        gl, gd = f(torch.tensor(grid, dtype=torch.float32).unsqueeze(0), theta)
    sig_hat = np.exp(gd.numpy()[0])
    log_err = np.abs(np.log(sig_hat) - np.log(_sigma_true(grid)))
    loc_err = np.abs(gl.numpy()[0] - (grid + 0.3))
    # tolerance: learned sigma(x) within 20 % (|Δ log sigma| < 0.18) of the truth over the grid
    assert log_err.max() < 0.18, (log_err.max(), sig_hat[[0, 27, 54]])
    # and it spans the range: the fitted high-x / low-x sigma ratio is within 30 % of the true ratio (15^0.9)
    ratio = sig_hat[-1] / sig_hat[0]
    true_ratio = _sigma_true(grid[-1]) / _sigma_true(grid[0])
    assert abs(math.log(ratio / true_ratio)) < math.log(1.3), (ratio, true_ratio)
    assert loc_err.max() < 0.15, loc_err.max()
    # beats the constant-sigma fit on NLL by a wide margin (nats per bin)
    assert nll_x < nll_const - 0.3, (nll_x, nll_const)


# ---- end to end through the real CLIs ----------------------------------------------------------

@pytest.mark.parametrize("space", ["counts", "pval"])
def test_cli_train_and_score_rung_x(tmp_path, space):
    from ladder import score, synth, train
    products, runs = tmp_path / "products", tmp_path / "runs"
    manifest, cov = synth.make_products(products, seed=0)
    blacklist = tmp_path / "blacklist.bed"
    blacklist.write_text("chr19\t10000\t12500\nchr21\t5000\t5100\nchr22\t0\t1000\n")
    rc = train.main(["train", str(manifest), str(cov), str(products), str(runs), "X", "T1", space,
                     "real", "0", "--max-steps", "20", "--device", "cpu"])
    assert rc in (0, None)
    run_dir = runs / f"X_T1_{space}_real_s0"
    for fn in ("ckpt.pt", "config.json", "train_log.tsv", "TRAIN_DONE"):
        assert (run_dir / fn).is_file(), fn
    ck = torch.load(run_dir / "ckpt.pt", map_location="cpu", weights_only=False)
    assert {"h.0.weight", "h.4.weight"} <= set(ck["form"])
    rc = score.main(["trained", str(manifest), str(cov), str(products), str(blacklist),
                     str(run_dir), "--workers", "1"])
    assert rc in (0, None)
    assert (run_dir / "scores.json").is_file()
    sc = json.loads((run_dir / "scores.json").read_text("utf-8"))
    trained = [r for r in sc["records"] if r["kind"] == "trained"]
    assert trained and all(np.isfinite(r["crps_all"]) for r in trained)
    assert all(len(r["describe"]["theta"]) == 32 for r in trained)
