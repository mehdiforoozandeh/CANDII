"""t125-P2d: the learned h (models, standardiser, losses), tested against the plan's pinned
interface (plan_t125.md "### t125-P2d") and design note Part 2, not against the implementation.

Feature widths are pinned in t125-P2a: rung 1 = 9 columns (log1p_k, log1p_ctl_bin, assay one-hot 7),
rung 5 = 19 columns (rung 1 + 9 continuous + has_control); the formula variants take one extra
column, log1p of the formula's P, last.
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
sys.path.insert(0, str(ROOT / "tools" / "t125"))

from hmap import model  # noqa: E402

N_R1, N_R5 = 9, 19
ASSAY_COLS = list(range(2, 9))          # one-hot, both rungs
HALF_LOG_2PI = 0.5 * math.log(2 * math.pi)


def _feat(n, rung, formula=False, seed=0, assay=None):
    rng = np.random.default_rng(seed)
    F = N_R1 if rung == 1 else N_R5
    x = rng.normal(size=(n, F)).astype(np.float32)
    a = rng.integers(0, 7, size=n) if assay is None else np.full(n, assay)
    x[:, 2:9] = 0.0
    x[np.arange(n), 2 + a] = 1.0
    if rung == 5:
        x[:, 18] = rng.integers(0, 2, size=n)        # has_control flag
    if formula:
        x = np.concatenate([x, rng.uniform(0, 5, size=(n, 1)).astype(np.float32)], axis=1)
    return torch.from_numpy(x)


# --- HModel: shapes, init ----------------------------------------------------------------

LEARNED = [("affine", 1), ("affine", 5), ("mlp", 1), ("mlp", 5),
           ("formula_affine", 5), ("formula_mlp", 5)]


def test_hidden_is_64():
    assert model.HIDDEN == 64


@pytest.mark.parametrize("variant,rung", LEARNED)
@pytest.mark.parametrize("noise", [False, True])
def test_output_shapes(variant, rung, noise):
    n_in = N_R1 if rung == 1 else N_R5
    h = model.HModel(variant, rung, noise, n_in)
    feat = _feat(37, rung, formula=variant.startswith("formula"))
    mu, ls = h(feat)
    assert mu.shape == (37,)
    if noise:
        assert ls.shape == (37,)
    else:
        assert ls is None


@pytest.mark.parametrize("variant,rung", LEARNED)
def test_leading_dims(variant, rung):
    # [N, K, F] rows (the enumerated count axis of the chain / fnb training) map to [N, K]
    n_in = N_R1 if rung == 1 else N_R5
    h = model.HModel(variant, rung, True, n_in)
    feat = _feat(4 * 6, rung, formula=variant.startswith("formula")).reshape(4, 6, -1)
    mu, ls = h(feat)
    assert mu.shape == (4, 6) and ls.shape == (4, 6)


@pytest.mark.parametrize("variant,rung", [("affine", 1), ("affine", 5), ("mlp", 1), ("mlp", 5)])
@pytest.mark.parametrize("noise", [False, True])
def test_zero_init(variant, rung, noise):
    torch.manual_seed(3)
    h = model.HModel(variant, rung, noise, N_R1 if rung == 1 else N_R5)
    mu, ls = h(_feat(50, rung))
    assert torch.all(mu == 0)
    if noise:
        assert torch.all(ls == 0)        # sigma = 1


@pytest.mark.parametrize("variant", ["formula_affine", "formula_mlp"])
@pytest.mark.parametrize("noise", [False, True])
def test_formula_identity_at_init(variant, noise):
    torch.manual_seed(5)
    h = model.HModel(variant, 5, noise, N_R5)
    feat = _feat(64, 5, formula=True)
    mu, ls = h(feat)
    assert torch.equal(mu, feat[:, -1])
    if noise:
        assert torch.all(ls == 0)


def test_mlp_architecture():
    h = model.HModel("mlp", 5, True, N_R5)
    lin = [m for m in h.modules() if isinstance(m, torch.nn.Linear)]
    assert [(m.in_features, m.out_features) for m in lin] == [(19, 64), (64, 64), (64, 2)]
    assert sum(isinstance(m, torch.nn.ReLU) for m in h.modules()) == 2
    h1 = model.HModel("mlp", 1, False, N_R1)
    lin = [m for m in h1.modules() if isinstance(m, torch.nn.Linear)]
    assert [(m.in_features, m.out_features) for m in lin] == [(9, 64), (64, 64), (64, 1)]
    hf = model.HModel("formula_mlp", 5, False, N_R5)
    lin = [m for m in hf.modules() if isinstance(m, torch.nn.Linear)]
    assert lin[0].in_features == 20                    # n_in + the formula column


def test_affine_is_affine():
    h = model.HModel("affine", 5, False, N_R5)
    with torch.no_grad():
        for p in h.parameters():
            p.normal_()
    a, b = _feat(10, 5, seed=1), _feat(10, 5, seed=2)
    ma, _ = h(a)
    mb, _ = h(b)
    mab, _ = h(0.5 * a + 0.5 * b)
    assert torch.allclose(mab, 0.5 * ma + 0.5 * mb, atol=1e-5)


LOG_DEPTH_C = 15          # rung 5: 9 log1p_k_frag ... 14 log_depth_t, 15 log_depth_c, ... 18 has_control


@pytest.mark.parametrize("noise", [False, True])
def test_formula_affine_is_formula_plus_affine_over_all_columns(noise):
    """AMENDED 2026-10-11 (PI): mu = f + w . feat + b over all n_in + 1 = 20 columns, f included;
    with noise log_sigma = w' . feat + b'. No per-assay-only parameters."""
    h = model.HModel("formula_affine", 5, noise, N_R5)
    n_out = 2 if noise else 1
    assert sum(p.numel() for p in h.parameters()) == n_out * (20 + 1)
    torch.manual_seed(0)
    W = torch.randn(n_out, 20)
    B = torch.randn(n_out)
    with torch.no_grad():
        for p in h.parameters():
            p.copy_(W if p.dim() == 2 else B)
    feat = _feat(30, 5, formula=True, seed=4)
    mu, ls = h(feat)
    lin = feat @ W.T + B
    assert torch.allclose(mu, feat[:, -1] + lin[:, 0], atol=1e-5)
    if noise:
        assert torch.allclose(ls, lin[:, 1].clamp(math.log(1e-3), math.log(1e3)), atol=1e-5)
    # a non-formula column moves mu (the correction sees every feature, not only f and the assay)
    feat2 = feat.clone()
    feat2[:, LOG_DEPTH_C] += 1.0
    mu2, _ = h(feat2)
    assert torch.allclose(mu2 - mu, torch.full_like(mu, W[0, LOG_DEPTH_C].item()), atol=1e-5)


def test_formula_affine_learns_a_non_formula_correction():
    """Target log1p P = f + 0.7 * log_depth_c - 0.2: the correction must come from column 15."""
    torch.manual_seed(2)
    feat = _feat(2000, 5, formula=True, seed=7)
    t = feat[:, -1] + 0.7 * feat[:, LOG_DEPTH_C] - 0.2
    y = torch.expm1(t)
    h = model.HModel("formula_affine", 5, False, N_R5)
    mu0, _ = h(feat)
    assert torch.equal(mu0, feat[:, -1])              # the formula at init
    opt = torch.optim.Adam(h.parameters(), lr=0.05)
    first = model.loss_push_true(mu0, y).item()
    for _ in range(400):
        opt.zero_grad()
        mu, _ = h(feat)
        loss = model.loss_push_true(mu, y)
        loss.backward()
        opt.step()
    assert loss.item() < 1e-3 * first
    w = h.lin.weight.detach()[0]
    assert abs(w[LOG_DEPTH_C].item() - 0.7) < 0.02
    # b trades off with the one-hot weights (they sum to 1): the intercept per assay is b + w_assay
    icpt = h.lin.bias.detach()[0] + w[2:9]
    assert torch.allclose(icpt, torch.full_like(icpt, -0.2), atol=0.02)


def test_log_sigma_clamped():
    h = model.HModel("affine", 1, True, N_R1)
    with torch.no_grad():
        for p in h.parameters():
            p.fill_(100.0)
    _, ls = h(_feat(8, 1).abs() + 1)
    assert torch.allclose(ls, torch.full_like(ls, math.log(1e3)))
    with torch.no_grad():
        for p in h.parameters():
            p.fill_(-100.0)
    _, ls = h(_feat(8, 1).abs() + 1)
    assert torch.allclose(ls, torch.full_like(ls, math.log(1e-3)))


def test_bad_args():
    with pytest.raises(ValueError):
        model.HModel("quadratic", 1, False, N_R1)
    with pytest.raises(ValueError):
        model.HModel("affine", 3, False, N_R1)
    with pytest.raises(ValueError):
        model.HModel("affine", 1, False, 10)               # rung 1 is 9 wide
    with pytest.raises(ValueError):
        model.HModel("formula_mlp", 1, False, N_R1)        # formula needs rung 5
    h = model.HModel("mlp", 5, False, N_R5)
    with pytest.raises(ValueError):
        h(_feat(3, 5, formula=True))                        # 20 wide into a 19-wide h


def test_P_hat():
    mu = torch.tensor([0.0, math.log(11.0), -0.5])
    assert torch.allclose(model.P_hat(mu), torch.tensor([0.0, 10.0, 0.0]), atol=1e-5)


# --- losses (hand cases) -----------------------------------------------------------------

def test_loss_push_true():
    mu = torch.tensor([0.0, 1.0])
    y = torch.tensor([math.e - 1, 0.0])          # log1p(y) = 1, 0
    assert math.isclose(model.loss_push_true(mu, y).item(), (1.0 + 1.0) / 2, rel_tol=1e-6)


def test_loss_noise_true():
    mu = torch.tensor([0.0, 2.0])
    ls = torch.tensor([0.0, math.log(2.0)])
    y = torch.tensor([math.e - 1, 0.0])          # t = 1, 0
    nll0 = 0.0 + HALF_LOG_2PI + 0.5 * 1.0
    nll1 = math.log(2.0) + HALF_LOG_2PI + 0.5 * (2.0 / 2.0) ** 2
    got = model.loss_noise_true(mu, ls, y).item()
    assert math.isclose(got, (nll0 + nll1) / 2, rel_tol=1e-6)


def test_loss_push_fnb_hand_K2():
    mu = torch.tensor([[0.0, 2.0], [1.0, 3.0]])
    w = torch.tensor([[0.25, 0.75], [1.0, 0.0]])
    y = torch.tensor([math.e - 1, math.e - 1])   # t = 1
    row0 = 0.25 * 1.0 + 0.75 * 1.0
    row1 = 1.0 * 0.0
    assert math.isclose(model.loss_push_fnb(mu, w, y).item(), (row0 + row1) / 2, rel_tol=1e-6)


def _gauss_nll(t, m, s):
    return math.log(s) + HALF_LOG_2PI + 0.5 * ((t - m) / s) ** 2


def test_loss_noise_fnb_hand_K2():
    mu = torch.tensor([[0.0, 2.0]])
    ls = torch.tensor([[0.0, math.log(0.5)]])
    w = torch.tensor([[0.3, 0.7]])
    y = torch.tensor([math.e - 1])               # t = 1
    n0, n1 = _gauss_nll(1, 0, 1), _gauss_nll(1, 2, 0.5)
    expected = 0.3 * n0 + 0.7 * n1
    mixture = -math.log(0.3 * math.exp(-n0) + 0.7 * math.exp(-n1))
    got_e = model.loss_noise_fnb(mu, ls, w, y).item()               # default rule
    got_e2 = model.loss_noise_fnb(mu, ls, w, y, rule="expected").item()
    got_m = model.loss_noise_fnb(mu, ls, w, y, rule="mixture").item()
    assert math.isclose(got_e, expected, rel_tol=1e-6)
    assert got_e == got_e2
    assert math.isclose(got_m, mixture, rel_tol=1e-6)
    assert abs(got_e - got_m) > 1e-3            # the two rules differ
    assert got_m <= got_e + 1e-9                # Jensen: -log E[p] <= E[-log p]
    with pytest.raises(ValueError):
        model.loss_noise_fnb(mu, ls, w, y, rule="sampled")


def test_fnb_losses_reduce_to_true_at_K1():
    rng = np.random.default_rng(0)
    mu = torch.from_numpy(rng.normal(size=11))
    ls = torch.from_numpy(rng.normal(scale=0.3, size=11))
    y = torch.from_numpy(rng.exponential(3.0, size=11))
    w = torch.ones(11, 1, dtype=torch.float64)
    assert torch.allclose(model.loss_push_fnb(mu[:, None], w, y), model.loss_push_true(mu, y))
    for rule in ("expected", "mixture"):
        assert torch.allclose(model.loss_noise_fnb(mu[:, None], ls[:, None], w, y, rule=rule),
                              model.loss_noise_true(mu, ls, y))


def test_losses_mask():
    mu = torch.tensor([0.0, 5.0, 1.0])
    y = torch.tensor([0.0, 0.0, math.e - 1])
    mask = torch.tensor([True, False, True])
    assert model.loss_push_true(mu, y, mask=mask).item() == 0.0
    w = torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.5, 0.5]])      # a padded row of zero weight
    mk = torch.stack([mu, mu], 1)
    ls = torch.zeros_like(mk, requires_grad=True)
    out = model.loss_noise_fnb(mk, ls, w, y, rule="mixture", mask=mask)
    assert torch.isfinite(out)
    out.backward()
    assert torch.all(torch.isfinite(ls.grad))
    assert torch.all(ls.grad[1] == 0)


def test_losses_have_gradients():
    h = model.HModel("formula_mlp", 5, True, N_R5)
    feat = _feat(16, 5, formula=True).reshape(4, 4, -1)
    mu, ls = h(feat)
    w = torch.full((4, 4), 0.25)
    y = torch.rand(4) * 10
    loss = model.loss_noise_fnb(mu, ls, w, y, rule="mixture")
    loss.backward()
    last = [m for m in h.modules() if isinstance(m, torch.nn.Linear)][-1]
    assert last.weight.grad is not None and torch.any(last.weight.grad != 0)


# --- standardiser ------------------------------------------------------------------------

def test_standardiser_scales_only_continuous():
    feat = _feat(500, 5).numpy().astype(np.float64)
    feat[:, 0] = feat[:, 0] * 4 + 7
    feat[:, 12] = 2.5                                 # a constant column: std 0 -> 0
    cont = [0, 1] + list(range(9, 18))
    s = model.Standardiser().fit(feat, cont)
    out = s.apply(feat)
    assert out.shape == feat.shape
    for c in cont:
        if c == 12:
            assert np.all(out[:, c] == 0)
        else:
            assert abs(out[:, c].mean()) < 1e-5
            assert abs(out[:, c].std() - 1) < 1e-4    # population std, as ladder.encoding
    for c in ASSAY_COLS + [18]:
        assert np.array_equal(out[:, c], feat[:, c])  # one-hot and the flag untouched


def test_standardiser_leading_dims_torch_and_state():
    feat = _feat(200, 1).numpy()
    s = model.Standardiser().fit(feat, [0, 1])
    s2 = model.Standardiser.from_state(s.state())
    x = _feat(30, 1, seed=9)
    a = s.apply(x.numpy())
    b = s2.apply(x.reshape(5, 6, -1))                 # torch in -> torch out, any leading shape
    assert isinstance(b, torch.Tensor) and b.shape == (5, 6, N_R1)
    assert np.allclose(a, b.reshape(30, -1).numpy(), atol=1e-6)
    with pytest.raises(ValueError):
        s.apply(np.zeros((3, N_R5), dtype=np.float32))


def test_continuous_columns():
    assert model.continuous_cols(1) == [0, 1]
    assert model.continuous_cols(5) == [0, 1] + list(range(9, 18))
    # the formula column is never scaled: the formula variants add it to mu as is
    assert model.continuous_cols(5, formula=True) == [0, 1] + list(range(9, 18))


# --- checkpoint --------------------------------------------------------------------------

@pytest.mark.parametrize("variant,rung,noise", [("mlp", 5, True), ("formula_affine", 5, False),
                                                ("affine", 1, True)])
def test_ckpt_round_trip(tmp_path, variant, rung, noise):
    torch.manual_seed(1)
    n_in = N_R1 if rung == 1 else N_R5
    h = model.HModel(variant, rung, noise, n_in)
    with torch.no_grad():
        for p in h.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    feat = _feat(40, rung, formula=variant.startswith("formula"))
    s = model.Standardiser().fit(feat.numpy(), model.continuous_cols(rung))
    cfg = {"h_name": "h_x", "variant": variant, "rung": rung, "noise": int(noise), "seed": 0}
    path = tmp_path / "ckpt.pt"
    model.save_ckpt(path, h, s, cfg, 750, 0.125)
    raw = torch.load(path, map_location="cpu", weights_only=True)
    assert set(raw) == {"h", "std", "config", "step", "val_loss"}
    h2, s2, cfg2, step, val = model.load_ckpt(path)
    assert (h2.variant, h2.rung, h2.noise) == (variant, rung, noise)
    assert cfg2["h_name"] == "h_x" and step == 750 and val == 0.125
    x = torch.as_tensor(s.apply(feat.numpy()))
    m1, l1 = h(x)
    m2, l2 = h2(torch.as_tensor(s2.apply(feat.numpy())))
    assert torch.equal(m1, m2)
    if noise:
        assert torch.equal(l1, l2)


def test_ckpt_config_mismatch(tmp_path):
    h = model.HModel("mlp", 5, False, N_R5)
    s = model.Standardiser().fit(_feat(10, 5).numpy(), model.continuous_cols(5))
    with pytest.raises(ValueError):
        model.save_ckpt(tmp_path / "c.pt", h, s, {"variant": "affine"}, 0, 1.0)
