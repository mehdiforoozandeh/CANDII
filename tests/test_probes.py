"""Known-answer tests for the metadata-pathway probe (`candi.probes`).

The probe is the mechanism instrument for h75, so shape checks are not enough: every scalar it emits
is checked against an INDEPENDENT brute-force computation — an explicit Python loop over prompts,
written from the definition rather than from the implementation. Where the two agree, the number the
experiment will be read from is the number it claims to be.

Synthetic modules only. Nothing here builds the real model.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from candi.probes import MetaPathProbe, effective_rank, make_meta_path_probe

B, A, E, C = 3, 5, 8, 6            # prompts = B*A = 15, fusion width E, FiLM output 2*C


# ---------------------------------------------------------------------------
# tiny stand-in for `CandiModel`: only `.decoder.meta_embedding.fusion` and
# `.decoder.film_proj` are what the probe reaches for.
# ---------------------------------------------------------------------------

class _MetaEmbed(nn.Module):
    def __init__(self, layernorm: bool):
        super().__init__()
        layers = [nn.Linear(E, E), nn.GELU(), nn.Linear(E, E)]
        if layernorm:
            layers.append(nn.LayerNorm(E))
        self.fusion = nn.Sequential(*layers)

    def forward(self, x):
        return self.fusion(x)


class _Decoder(nn.Module):
    def __init__(self, layernorm: bool):
        super().__init__()
        self.meta_embedding = _MetaEmbed(layernorm)
        self.film_proj = nn.Linear(E, 2 * C)

    def forward(self, x):
        memb = self.meta_embedding(x)
        gamma, beta = self.film_proj(memb).chunk(2, dim=-1)
        return gamma, beta


class _Model(nn.Module):
    def __init__(self, layernorm: bool = True, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.decoder = _Decoder(layernorm)

    def forward(self, x):
        return self.decoder(x)


def _inputs(seed: int = 7) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    # scaled up so the fusion output is not trivially tiny and the norms are comfortably nonzero
    return torch.randn(B, A, E, generator=g) * 3.0


# ---------------------------------------------------------------------------
# independent (loop-based) reference implementations of the two magnitudes
# ---------------------------------------------------------------------------

def brute_norm(x: torch.Tensor) -> float:
    """Mean over prompts of the per-prompt L2 norm, one prompt at a time."""
    prompts = x.detach().reshape(-1, x.shape[-1]).double()
    tot = 0.0
    for i in range(prompts.shape[0]):
        tot += math.sqrt(float((prompts[i] ** 2).sum()))
    return tot / prompts.shape[0]


def brute_spread(x: torch.Tensor) -> float:
    """Mean L2 norm of each prompt minus the mean prompt, one prompt at a time."""
    prompts = x.detach().reshape(-1, x.shape[-1]).double()
    mean = prompts.mean(dim=0)
    tot = 0.0
    for i in range(prompts.shape[0]):
        tot += math.sqrt(float(((prompts[i] - mean) ** 2).sum()))
    return tot / prompts.shape[0]


# ---------------------------------------------------------------------------
# 1. known answers on the LayerNorm arm
# ---------------------------------------------------------------------------

def test_known_answer_pre_post_and_attenuation():
    m = _Model(layernorm=True)
    x = _inputs()
    fusion = m.decoder.meta_embedding.fusion
    assert isinstance(fusion[-1], nn.LayerNorm)

    with torch.no_grad():
        pre = fusion[:3](x)                       # Linear -> GELU -> Linear  (the LayerNorm's input)
        post = fusion(x)                          # ... -> LayerNorm
        fout = m.decoder.film_proj(post)

    probe = MetaPathProbe(m)
    probe.arm()
    with torch.no_grad():
        m(x)
    r = probe.read()
    probe.close()

    assert r["meta/ln_present"] == 1.0
    assert r["meta/pre_ln_norm"] == pytest.approx(brute_norm(pre), rel=1e-6)
    assert r["meta/post_ln_norm"] == pytest.approx(brute_norm(post), rel=1e-6)
    assert r["meta/pre_ln_spread"] == pytest.approx(brute_spread(pre), rel=1e-6)
    assert r["meta/post_ln_spread"] == pytest.approx(brute_spread(post), rel=1e-6)

    assert r["meta/ln_gain"] == pytest.approx(brute_norm(post) / brute_norm(pre), rel=1e-6)
    assert r["meta/perturbation_attenuation"] == pytest.approx(
        brute_spread(pre) / brute_spread(post), rel=1e-6)
    assert r["meta/pre_ln_snr"] == pytest.approx(brute_spread(pre) / brute_norm(pre), rel=1e-6)
    assert r["meta/post_ln_snr"] == pytest.approx(brute_spread(post) / brute_norm(post), rel=1e-6)

    # LayerNorm pins every prompt to sqrt(E); the ambient norm is therefore a known constant.
    assert r["meta/post_ln_norm"] == pytest.approx(math.sqrt(E), rel=1e-4)

    # film_proj: input is the fusion output, output is [gamma | beta]
    assert r["meta/film_in_norm"] == pytest.approx(brute_norm(post), rel=1e-6)
    assert r["meta/film_out_norm"] == pytest.approx(brute_norm(fout), rel=1e-6)
    assert r["meta/film_gain"] == pytest.approx(brute_norm(fout) / brute_norm(post), rel=1e-6)
    assert r["meta/film_in_spread"] == pytest.approx(brute_spread(post), rel=1e-6)
    assert r["meta/film_out_spread"] == pytest.approx(brute_spread(fout), rel=1e-6)
    assert r["meta/film_attenuation"] == pytest.approx(
        brute_spread(post) / brute_spread(fout), rel=1e-6)

    gamma, beta = fout.chunk(2, dim=-1)
    assert r["meta/gamma_absmax"] == pytest.approx(float(gamma.abs().max()), rel=1e-6)
    assert r["meta/beta_absmax"] == pytest.approx(float(beta.abs().max()), rel=1e-6)
    assert all(math.isfinite(v) for v in r.values())


def test_zero_film_proj_gives_nan_attenuation_not_a_zero_division():
    """adaLN-zero starts `film_proj` at exactly zero, so its output spread is 0. The guarded ratio
    must read nan; an unguarded one would raise or report inf."""
    m = _Model(layernorm=True)
    nn.init.zeros_(m.decoder.film_proj.weight)
    nn.init.zeros_(m.decoder.film_proj.bias)
    probe = MetaPathProbe(m)
    probe.arm()
    with torch.no_grad():
        m(_inputs())
    r = probe.read()
    probe.close()
    assert math.isnan(r["meta/film_attenuation"])          # 0 denominator -> nan, never a raise
    assert math.isnan(r["meta/gamma_effrank"])             # an all-zero matrix has no rank to report
    assert r["meta/film_gain"] == pytest.approx(0.0, abs=1e-12)   # numerator is 0, denominator is not
    assert r["meta/film_out_norm"] == pytest.approx(0.0, abs=1e-12)
    assert r["meta/gamma_absmax"] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 2. effective rank
# ---------------------------------------------------------------------------

def test_effective_rank_rank_one_is_one():
    g = torch.Generator().manual_seed(1)
    a = torch.randn(40, 1, generator=g)
    b = torch.randn(1, 9, generator=g)
    assert effective_rank(a @ b) == pytest.approx(1.0, abs=1e-4)


def test_effective_rank_orthogonal_is_full_dimension():
    d = 7
    g = torch.Generator().manual_seed(2)
    q, _ = torch.linalg.qr(torch.randn(d, d, generator=g))     # all singular values == 1
    assert effective_rank(q) == pytest.approx(float(d), rel=1e-4)
    assert effective_rank(torch.eye(d)) == pytest.approx(float(d), rel=1e-6)


def test_effective_rank_degenerate_inputs_are_nan_not_exceptions():
    assert math.isnan(effective_rank(torch.zeros(5, 4)))
    assert math.isnan(effective_rank(torch.zeros(0, 4)))


# ---------------------------------------------------------------------------
# 3. arming discipline
# ---------------------------------------------------------------------------

def test_unarmed_probe_captures_nothing():
    m = _Model(layernorm=True)
    probe = MetaPathProbe(m)
    x = _inputs()
    with torch.no_grad():
        m(x)
        m(x)
    assert probe.read() == {}
    probe.close()


def test_arm_captures_exactly_one_forward():
    m = _Model(layernorm=True)
    probe = MetaPathProbe(m)
    probe.arm()
    with torch.no_grad():
        m(_inputs(seed=1))
        m(_inputs(seed=2))                       # must NOT overwrite the first capture
    first = probe.read()
    assert first
    assert first["meta/pre_ln_norm"] == pytest.approx(
        brute_norm(m.decoder.meta_embedding.fusion[:3](_inputs(seed=1))), rel=1e-6)
    assert probe.read() == {}                    # buffer cleared by the first read
    probe.close()


def test_close_removes_hooks():
    m = _Model(layernorm=True)
    probe = MetaPathProbe(m)
    probe.close()
    probe.arm()
    with torch.no_grad():
        m(_inputs())
    assert probe.read() == {}


# ---------------------------------------------------------------------------
# 4. strict no-op on the forward output
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("layernorm", [True, False])
def test_probe_does_not_change_the_forward_output(layernorm):
    x = _inputs()
    clean = _Model(layernorm=layernorm, seed=3)
    with torch.no_grad():
        g0, b0 = clean(x)

    hooked = _Model(layernorm=layernorm, seed=3)
    probe = MetaPathProbe(hooked)
    with torch.no_grad():
        g1, b1 = hooked(x)                       # unarmed
        probe.arm()
        g2, b2 = hooked(x)                       # armed
    probe.close()

    assert torch.equal(g0, g1) and torch.equal(b0, b1)
    assert torch.equal(g0, g2) and torch.equal(b0, b2)


def test_probe_does_not_touch_autograd_or_gradients():
    x = _inputs()
    clean, hooked = _Model(seed=4), _Model(seed=4)
    probe = MetaPathProbe(hooked)
    probe.arm()

    def grads(model):
        model.zero_grad()
        g, b = model(x)
        (g.sum() + b.sum()).backward()
        return {k: p.grad.clone() for k, p in model.named_parameters()}

    a, c = grads(clean), grads(hooked)
    probe.close()
    assert set(a) == set(c)
    assert all(torch.equal(a[k], c[k]) for k in a)


# ---------------------------------------------------------------------------
# 5. the LN-absent arm
# ---------------------------------------------------------------------------

def test_layernorm_absent_emits_the_same_keys():
    on, off = _Model(layernorm=True, seed=5), _Model(layernorm=False, seed=5)
    x = _inputs()
    out = {}
    for name, m in (("on", on), ("off", off)):
        probe = MetaPathProbe(m)
        probe.arm()
        with torch.no_grad():
            m(x)
        out[name] = probe.read()
        probe.close()

    assert set(out["on"]) == set(out["off"])         # directly comparable across arms
    assert out["on"]["meta/ln_present"] == 1.0
    assert out["off"]["meta/ln_present"] == 0.0
    # with no LayerNorm the "pre" and "post" tensors are the same object, by construction
    assert out["off"]["meta/ln_gain"] == pytest.approx(1.0, rel=1e-12)
    assert out["off"]["meta/perturbation_attenuation"] == pytest.approx(1.0, rel=1e-12)
    assert out["off"]["meta/pre_ln_norm"] == pytest.approx(out["off"]["meta/post_ln_norm"], rel=1e-12)
    assert out["off"]["meta/pre_ln_spread"] == pytest.approx(out["off"]["meta/post_ln_spread"],
                                                             rel=1e-12)
    # and the reported "post" side really is the fusion output
    with torch.no_grad():
        fused = off.decoder.meta_embedding.fusion(x)
    assert out["off"]["meta/post_ln_norm"] == pytest.approx(brute_norm(fused), rel=1e-6)


def test_layernorm_attenuates_the_perturbation_and_removing_it_does_not():
    """The h75 reading in miniature: the LN arm attenuates the across-prompt perturbation, the LN-off
    arm reports exactly 1.0 because there is nothing between the two capture points."""
    on, off = _Model(layernorm=True, seed=6), _Model(layernorm=False, seed=6)
    x = _inputs() * 40.0                             # a large ambient scale is what LayerNorm removes
    vals = {}
    for name, m in (("on", on), ("off", off)):
        probe = MetaPathProbe(m)
        probe.arm()
        with torch.no_grad():
            m(x)
        vals[name] = probe.read()["meta/perturbation_attenuation"]
        probe.close()
    assert vals["on"] > 2.0
    assert vals["off"] == pytest.approx(1.0, rel=1e-12)


# ---------------------------------------------------------------------------
# 6. the best-effort constructor
# ---------------------------------------------------------------------------

def test_make_meta_path_probe_returns_none_on_a_model_without_the_pathway():
    assert make_meta_path_probe(nn.Linear(2, 2)) is None
    assert isinstance(make_meta_path_probe(_Model()), MetaPathProbe)
