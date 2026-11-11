"""miniCANDI rung 3 (conv plus attention autoencoder): shapes, budget, guardrail part 1 per arm,
attention reach, training signal, gradient paths."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

from minicandi import components as M  # noqa: E402
from minicandi import contract as C  # noqa: E402
from minicandi import synth  # noqa: E402
from minicandi.dataset import OnionDataset  # noqa: E402
from minicandi.rungs import conv_attn_ae as R  # noqa: E402
from onion_eval.guardrail import copy_leak  # noqa: E402

L = 64
BASE = json.loads((_ROOT / "tools/minicandi/configs/rung3/base.json").read_text())
ARMS = [{}] + [{"covariate_injection": m} for m in ("concat", "tokens")] \
    + [{"mask_mode": "drop"}] + [{"dna": d} for d in ("off", "shuffled")] \
    + [{"qc_on_input": True}, {"average_input": True}]


def _cfg(**over):
    cfg = dict(BASE, context_bins=L)
    cfg.update(over)
    return cfg


def _toy_batch(B=2, K=C.K_META, seed=0, avg=False, L=L):
    g = torch.Generator().manual_seed(seed)
    x_avail = torch.zeros(B, C.A, dtype=torch.bool)
    x_avail[:, [1, 15, 21, 35]] = True
    y_kind = torch.zeros(B, C.A, dtype=torch.uint8)
    y_kind[:, 1] = C.KIND_SAME
    y_kind[:, 15] = C.KIND_CORRECT
    y_kind[:, 16] = C.KIND_IMPUTE
    y_kind[:, 21] = C.KIND_SAME
    y_avail = y_kind > 0
    bm = torch.zeros(B, L, C.A, dtype=torch.bool)
    bm[:, ::3, 1] = True
    bm[:, 5::4, 21] = True
    bm &= x_avail[:, None, :]
    x = torch.randint(0, 50, (B, L, C.A), generator=g, dtype=torch.int32)
    x[bm | ~x_avail[:, None, :]] = 0
    c_in = torch.zeros(B, C.A, K)
    c_in[x_avail] = 1.0
    c_in[..., 0][x_avail] = 24.0
    c_out = torch.zeros(B, C.A, C.K_META)
    c_out[y_avail] = 1.0
    c_out[..., 0][y_avail] = 23.0
    b = {"x_counts": x, "x_avail": x_avail, "x_bin_mask": bm,
         "x_dna": torch.randint(0, 5, (B, L * C.RES), generator=g).to(torch.uint8),
         "c_in": c_in, "c_out": c_out,
         "y_counts": torch.randint(0, 50, (B, L, C.A), generator=g, dtype=torch.int32),
         "y_avail": y_avail, "y_kind": y_kind,
         "meta": [{"cell": "x", "chrom": "chr1", "start_bin": 0}] * B}
    if avg:
        b["avg"] = torch.rand(B, L, C.A, generator=g) * 5
    C.validate_batch(b)
    return b


def _perturb(model, seed: int = 0):
    """Add noise to every parameter. The NB head and FiLM start at zero, so an untrained model
    ignores its inputs and a wiring check would pass vacuously."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    return model.eval()


def _batch_for(cfg):
    return _toy_batch(K=C.k_input(cfg), avg=bool(cfg.get("average_input")))


@pytest.mark.parametrize("arm", ARMS, ids=lambda a: ",".join(f"{k}={v}" for k, v in a.items())
                         or "base")
def test_shapes_positivity_budget_and_guardrail_part1(arm):
    cfg = _cfg(**arm)
    torch.manual_seed(0)
    model = _perturb(R.Model(cfg))
    n = model.param_count()
    assert abs(n - cfg["param_budget"]) <= 0.10 * cfg["param_budget"], n
    b = _batch_for(cfg)
    out = model(b)
    assert out["n"].shape == out["mu"].shape == (2, L, C.A)
    assert out["n"].dtype == out["mu"].dtype == torch.float32
    assert (out["n"] > 0).all() and (out["mu"] > 0).all()
    assert torch.isfinite(out["n"]).all() and torch.isfinite(out["mu"]).all()
    r = C.wiring_check(model, b)
    assert r["pass"] and r["n_perturbed"] > 0, r
    g = copy_leak(model, [b, _batch_for(cfg)])
    assert g["part1_wiring_pass"], g
    assert g["n_hidden_bins"] > 0 and g["n_unavailable_bins"] > 0


@pytest.mark.parametrize("transform", M.TRANSFORMS)
@pytest.mark.parametrize("mask_mode", M.MASK_MODES)
def test_every_count_transform_passes_part1(transform, mask_mode):
    cfg = _cfg(count_transform=transform, mask_mode=mask_mode)
    model = _perturb(R.Model(cfg, width=32))
    b = _toy_batch()
    out = model(b)
    assert (out["n"] > 0).all() and (out["mu"] > 0).all()
    assert C.wiring_check(model, b)["pass"]


def _attn_io(model, batch):
    """The tensors that enter and leave the latent attention (a forward hook)."""
    seen = {}

    def hook(_m, args, kwargs, out):
        seen["in"] = (args[0] if args else kwargs["src"]).detach().clone()
        seen["out"] = out.detach().clone()

    handle = model.attn.register_forward_hook(hook, with_kwargs=True)
    try:
        with torch.no_grad():
            model(batch)
    finally:
        handle.remove()
    return seen["in"], seen["out"]


@pytest.mark.parametrize("inj", ["film", "concat", "tokens"])
@pytest.mark.parametrize("mask_mode", M.MASK_MODES)
def test_attention_carries_no_hidden_bin_content(inj, mask_mode):
    """The attention's input and output are bit-identical when hidden bins and missing tracks
    change, while a change of visible content does move it (so the check looks at a live
    path). 512 bins, 32 latent positions."""
    Lw = 512
    model = _perturb(R.Model(_cfg(covariate_injection=inj, mask_mode=mask_mode,
                                 context_bins=Lw), width=32))
    b = _toy_batch(B=1, L=Lw)
    hid = dict(b)
    x = b["x_counts"].clone()
    x[b["x_bin_mask"] | ~b["x_avail"][:, None, :]] = 1000
    hid["x_counts"] = x
    i0, o0 = _attn_io(model, b)
    i1, o1 = _attn_io(model, hid)
    assert torch.equal(i0, i1) and torch.equal(o0, o1)
    far = dict(b)
    x = b["x_counts"].clone()
    tail = x[:, Lw - 16:, 15]
    x[:, Lw - 16:, 15] = torch.where(~b["x_bin_mask"][:, Lw - 16:, 15], tail + 40, tail)
    far["x_counts"] = x
    i2, o2 = _attn_io(model, far)
    assert not torch.equal(o0, o2)


def test_a_leaky_variant_fails_the_wiring_check():
    model = _perturb(R.Model(_cfg(), width=32))

    class _NoMask(torch.nn.Module):            # forwards raw features at hidden bins
        def forward(self, h, x_avail, x_bin_mask):
            return h, M.MaskHandler.visible(x_avail, x_bin_mask)

    model.mask = _NoMask()
    assert not C.wiring_check(model, _toy_batch())["pass"]


@pytest.mark.parametrize("mask_mode", M.MASK_MODES)
def test_wiring_check_fails_with_the_mask_bypassed(mask_mode, monkeypatch):
    """Planted negative control: with every bin and track called visible, hidden and missing
    counts reach the perturbed model and the check must fail (so the passes above are real)."""
    model = _perturb(R.Model(_cfg(mask_mode=mask_mode), width=32))
    b = _toy_batch()
    assert C.wiring_check(model, b)["pass"]
    monkeypatch.setattr(M.MaskHandler, "visible",
                        staticmethod(lambda a, mk: torch.ones_like(mk)))
    assert not C.wiring_check(model, b)["pass"]
    assert not copy_leak(model, [b])["part1_wiring_pass"]


def test_not_applicable_and_bad_configs():
    with pytest.raises(M.NotApplicable, match="later ablation on the winning design"):
        R.Model(_cfg(two_step=True))
    with pytest.raises(ValueError):
        R.Model(_cfg(context_bins=L + 3))
    with pytest.raises(ValueError):
        R.Model(_cfg(rung=2))
    with pytest.raises(ValueError):
        R.Model(_cfg(covariate_injection="cross"))


def test_base_config_keys_and_budget_at_full_context():
    for key in ("rung", "context_bins", "covariate_injection", "mask_mode", "dna",
                "scramble_covariates", "count_transform", "qc_on_input", "two_step",
                "base_depth", "average_input", "control_input", "predict_control",
                "param_budget", "windows_seen", "lr", "batch_size", "simpler_than_base"):
        assert key in BASE, key
    assert BASE["rung"] == 3 and BASE["mask_mode"] == "token"
    assert BASE["covariate_injection"] == "film"
    n = R.Model(BASE).param_count()
    assert abs(n - BASE["param_budget"]) <= 0.10 * BASE["param_budget"]


@pytest.fixture(scope="module")
def synth_batch(tmp_path_factory):
    td = synth.make_synthetic_tables(tmp_path_factory.mktemp("onion_r3"))
    ds = OnionDataset(td, "train", {"context_bins": L, "batch_size": 4}, seed=0)
    import numpy as np
    b = ds.batch(4, np.random.default_rng(0))
    C.validate_batch(b)
    return b


@pytest.mark.parametrize("inj", ["film", "concat", "tokens"])
def test_training_lowers_nb_loss_and_grads_reach_covariates_and_dna(synth_batch, inj):
    torch.manual_seed(0)
    model = R.Model(_cfg(covariate_injection=inj)).train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    mask = C.loss_mask(synth_batch)
    assert mask.any()
    losses = []
    for _ in range(25):
        out = model(synth_batch)
        loss = C.nb_nll(out["n"], out["mu"], synth_batch["y_counts"], mask)
        opt.zero_grad()
        loss.backward()
        losses.append(loss.item())
        opt.step()
    assert losses[-1] < 0.9 * losses[0], losses
    cov_grad = sum(p.grad.abs().sum().item() for p in model.cov_in.embed.parameters()
                   if p.grad is not None)
    out_grad = sum(p.grad.abs().sum().item() for p in model.cov_out.embed.parameters()
                   if p.grad is not None)
    dna_grad = sum(p.grad.abs().sum().item() for p in model.dna.parameters()
                   if p.grad is not None)
    assert cov_grad > 0 and out_grad > 0 and dna_grad > 0, (cov_grad, out_grad, dna_grad)


def test_dna_off_has_no_dna_parameters():
    model = R.Model(_cfg(dna="off"), width=32)
    assert sum(p.numel() for p in model.dna.parameters()) == 0
