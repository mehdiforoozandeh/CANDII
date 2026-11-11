"""miniCANDI rung 2 (convolutional autoencoder): shapes, budget, guardrail part 1, learning, grads.

The registry may not exist in this tree, so the model class is built directly.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "tools",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from minicandi import components as M  # noqa: E402
from minicandi import contract as C  # noqa: E402
from minicandi.dataset import OnionDataset  # noqa: E402
from minicandi.rungs import conv_ae  # noqa: E402
from minicandi.synth import make_synthetic_tables  # noqa: E402
from onion_eval.guardrail import copy_leak  # noqa: E402

BASE_JSON = _ROOT / "tools" / "minicandi" / "configs" / "rung2" / "base.json"
BASE = json.loads(BASE_JSON.read_text())
PINNED_KEYS = ("rung", "context_bins", "covariate_injection", "mask_mode", "dna",
               "scramble_covariates", "count_transform", "qc_on_input", "two_step", "base_depth",
               "average_input", "control_input", "predict_control", "param_budget",
               "windows_seen", "lr", "batch_size", "simpler_than_base")

#: every arm rung 2 supports (tokens injection is NotApplicable)
ARMS = {
    "base": {},
    "concat": {"covariate_injection": "concat"},
    "drop": {"mask_mode": "drop"},
    "dna_off": {"dna": "off"},
    "dna_shuffled": {"dna": "shuffled"},
    "scrambled_covariates": {"scramble_covariates": True},
    "qc_on_input": {"qc_on_input": True},
    "average_input": {"average_input": True},
    "concat_drop": {"covariate_injection": "concat", "mask_mode": "drop"},
    **{f"transform_{t}": {"count_transform": t} for t in M.TRANSFORMS if t != "offset"},
}


def _cfg(arm: str) -> dict:
    return {**BASE, **ARMS[arm]}


def _toy_batch(B=3, L=48, K=C.K_META, seed=0, avg=False):
    """A contract batch with all three target kinds, hidden bins, and 33 missing tracks.
    L = 48 is not a multiple of 16, so the padding path is exercised."""
    g = torch.Generator().manual_seed(seed)
    x_avail = torch.zeros(B, C.A, dtype=torch.bool)
    x_avail[:, [1, 15, 35]] = True
    y_kind = torch.zeros(B, C.A, dtype=torch.uint8)
    y_kind[:, 1] = C.KIND_SAME
    y_kind[:, 15] = C.KIND_CORRECT
    y_kind[:, 21] = C.KIND_IMPUTE
    y_avail = y_kind > 0
    bm = torch.rand(B, L, C.A, generator=g) < 0.3
    bm &= (y_kind == C.KIND_SAME)[:, None, :]
    x = torch.randint(0, 60, (B, L, C.A), generator=g, dtype=torch.int32)
    x[bm | ~x_avail[:, None, :]] = 0
    c_in = torch.zeros(B, C.A, K)
    c_in[x_avail] = 1.0
    c_in[..., 0][x_avail] = 23.0 + torch.rand(int(x_avail.sum()), generator=g) * 2
    c_out = torch.zeros(B, C.A, C.K_META)
    c_out[y_avail] = 1.0
    c_out[..., 0][y_avail] = 23.0 + torch.rand(int(y_avail.sum()), generator=g) * 2
    b = {"x_counts": x, "x_avail": x_avail, "x_bin_mask": bm,
         "x_dna": torch.randint(0, 5, (B, L * C.RES), generator=g).to(torch.uint8),
         "c_in": c_in, "c_out": c_out,
         "y_counts": torch.randint(0, 60, (B, L, C.A), generator=g, dtype=torch.int32),
         "y_avail": y_avail, "y_kind": y_kind,
         "meta": [{"cell": "x", "chrom": "chr1", "start_bin": 0}] * B}
    if avg:
        b["avg"] = torch.rand(B, L, C.A, generator=g) * 5
    C.validate_batch(b)
    return b


def _perturb(model: torch.nn.Module, seed: int = 0) -> torch.nn.Module:
    """Add noise to every parameter. The NB head and the FiLM generators start at zero, so an
    untrained model ignores its inputs and a wiring check would pass vacuously."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    return model


@pytest.fixture(scope="module")
def synth_batches(tmp_path_factory):
    tables = make_synthetic_tables(tmp_path_factory.mktemp("synth"), seed=0)
    ds = OnionDataset(tables, "train", {"context_bins": 64, "batch_size": 4}, seed=0)
    it = iter(ds)
    return [next(it) for _ in range(3)]


# ------------------------------------------------------------------------------- build


def test_base_json_carries_every_pinned_key_and_the_plan_base():
    assert set(PINNED_KEYS) <= set(BASE)
    assert BASE["rung"] == 2
    assert BASE["mask_mode"] == "token" and BASE["covariate_injection"] == "film"
    assert BASE["simpler_than_base"] is None


def test_tokens_injection_is_not_applicable():
    with pytest.raises(M.NotApplicable):
        conv_ae.Model({**BASE, "covariate_injection": "tokens"})


def test_two_step_is_not_applicable():
    with pytest.raises(M.NotApplicable, match="later ablation on the winning design"):
        conv_ae.Model({**BASE, "two_step": True})


def test_wrong_rung_is_refused():
    with pytest.raises(ValueError):
        conv_ae.Model({**BASE, "rung": 3})


@pytest.mark.parametrize("arm", ["base", "concat", "drop", "dna_off", "qc_on_input",
                                 "average_input", "concat_drop"])
def test_param_count_within_budget(arm):
    cfg = _cfg(arm)
    m = conv_ae.Model(cfg)
    budget = cfg["param_budget"]
    assert abs(m.param_count() - budget) <= 0.10 * budget, (arm, m.width, m.param_count())


def test_param_budget_moves_the_width():
    small = conv_ae.Model({**BASE, "param_budget": 5.0e5})
    big = conv_ae.Model({**BASE, "param_budget": 2.0e6})
    assert small.width < big.width
    for m, b in ((small, 5.0e5), (big, 2.0e6)):
        assert abs(m.param_count() - b) <= 0.10 * b


def test_construction_does_not_move_the_global_seed():
    torch.manual_seed(1)
    a = conv_ae.Model(BASE).state_dict()
    torch.manual_seed(1)
    b = conv_ae.Model(BASE).state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_shapes_and_positivity_on_synthetic_batches(synth_batches):
    torch.manual_seed(0)
    model = _perturb(conv_ae.Model(BASE)).eval()
    for batch in synth_batches:
        C.validate_batch(batch)
        out = model(batch)
        shape = tuple(batch["x_counts"].shape)
        assert set(out) >= {"n", "mu"}
        for k in ("n", "mu"):
            assert tuple(out[k].shape) == shape and out[k].dtype == torch.float32
            assert torch.isfinite(out[k]).all() and (out[k] > 0).all()


# ------------------------------------------------------------------------------- guardrail


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_guardrail_part1_holds_for_every_arm(arm):
    cfg = _cfg(arm)
    torch.manual_seed(0)
    model = _perturb(conv_ae.Model(cfg)).eval()
    K = C.k_input(cfg)
    batches = [_toy_batch(K=K, seed=s, avg=cfg["average_input"]) for s in (0, 1)]
    out = model(batches[0])
    assert tuple(out["n"].shape) == tuple(out["mu"].shape) == (3, 48, C.A)
    assert (out["n"] > 0).all() and (out["mu"] > 0).all()
    for s, b in enumerate(batches):
        r = C.wiring_check(model, b, seed=s)
        assert r["pass"] and r["n_perturbed"] > 0, (arm, r)
    g = copy_leak(model, batches, seed=0)
    assert g["part1_wiring_pass"], (arm, g)
    assert g["hidden_input_max_abs_change"] <= 1e-6
    assert g["unavailable_input_max_abs_change"] <= 1e-6
    assert g["n_hidden_bins"] > 0 and g["n_unavailable_bins"] > 0
    # not vacuous: a change of VISIBLE counts must move the outputs (random, not a constant
    # shift, which the rank-based quantile transform rightly ignores)
    b = batches[0]
    vis = b["x_avail"][:, None, :] & ~b["x_bin_mask"]
    noise = torch.randint(1, 50, b["x_counts"].shape, generator=torch.Generator().manual_seed(3),
                          dtype=torch.int32)
    moved = dict(b, x_counts=torch.where(vis, b["x_counts"] + noise, b["x_counts"]))
    with torch.no_grad():
        mu0 = model(b)["mu"]
        delta = ((model(moved)["mu"] - mu0).abs() / mu0).max().item()
    assert delta > 1e-3, (arm, delta)


def test_guardrail_part1_on_synthetic_batches(synth_batches):
    torch.manual_seed(0)
    model = _perturb(conv_ae.Model(BASE)).eval()
    g = copy_leak(model, synth_batches, seed=0)
    assert g["part1_wiring_pass"], g
    for b in synth_batches:
        assert C.wiring_check(model, b)["pass"]


@pytest.mark.parametrize("arm", ["base", "drop"])
def test_wiring_check_fails_with_the_mask_bypassed(arm, monkeypatch):
    """Planted negative control: with every bin and track called visible, hidden and missing
    counts reach the perturbed model and the check must fail (so the passes above are real)."""
    torch.manual_seed(0)
    model = _perturb(conv_ae.Model(_cfg(arm))).eval()
    b = _toy_batch()
    assert C.wiring_check(model, b)["pass"]
    monkeypatch.setattr(M.MaskHandler, "visible",
                        staticmethod(lambda a, mk: torch.ones_like(mk)))
    assert not C.wiring_check(model, b)["pass"]
    assert not copy_leak(model, [b])["part1_wiring_pass"]


def test_average_input_needs_avg():
    model = conv_ae.Model({**BASE, "average_input": True}).eval()
    with pytest.raises(ValueError, match="avg"):
        model(_toy_batch())


# ------------------------------------------------------------------------------- learning


def test_optimizer_steps_lower_the_nb_loss(synth_batches):
    torch.manual_seed(0)
    model = conv_ae.Model(BASE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    def loss_on(batches):
        tot = 0.0
        for b in batches:
            o = model(b)
            tot = tot + C.nb_nll(o["n"], o["mu"], b["y_counts"], C.loss_mask(b))
        return tot / len(batches)

    model.eval()
    with torch.no_grad():
        before = float(loss_on(synth_batches))
    model.train()
    for _ in range(25):
        for b in synth_batches:
            o = model(b)
            loss = C.nb_nll(o["n"], o["mu"], b["y_counts"], C.loss_mask(b))
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        after = float(loss_on(synth_batches))
    assert after < 0.9 * before, (before, after)


@pytest.mark.parametrize("arm", ["base", "concat", "dna_shuffled", "drop"])
def test_gradients_reach_covariate_encoders_and_dna_tower(arm, synth_batches):
    cfg = _cfg(arm)
    torch.manual_seed(0)
    model = _perturb(conv_ae.Model(cfg)).train()
    b = synth_batches[0]
    o = model(b)
    C.nb_nll(o["n"], o["mu"], b["y_counts"], C.loss_mask(b)).backward()

    def grad_norm(mod):
        return sum(float(p.grad.abs().sum()) for p in mod.parameters() if p.grad is not None)

    assert grad_norm(model.inj_in.embed) > 0, arm
    assert grad_norm(model.inj_out.embed) > 0, arm
    assert grad_norm(model.dna) > 0, arm


def test_dna_off_has_no_dna_parameters():
    m = conv_ae.Model({**BASE, "dna": "off"})
    assert sum(p.numel() for p in m.dna.parameters()) == 0


# ------------------------------------------------------------------------------- import


def test_module_imports_without_the_registry():
    code = ("import sys; sys.path.insert(0, 'tools'); "
            "import minicandi.rungs.conv_ae as m; "
            "assert 'minicandi.registry' not in sys.modules; print(m.RUNG, m.NAME)")
    r = subprocess.run([sys.executable, "-c", code], cwd=_ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["2", "conv_ae"]
