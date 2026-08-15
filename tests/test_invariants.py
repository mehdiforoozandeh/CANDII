"""The invariants that define the CANDI decoder, tested on the model itself.

Three claims are load-bearing and each has a test that would fail loudly if it stopped being true:

  1. the deconv trunk is grouped, so no channel of assay `a` ever reaches assay `b`;
  2. conditioning is per-assay — moving one assay's covariates moves that assay and nothing else;
  3. every decoder FiLM is adaLN-ZERO, so at init the target metadata moves nothing at all.

Claim 3 is why a run can be compared to its own control: the model is born un-conditioned and
steering has to be learned, so it cannot lose by being born mis-conditioned.
"""
from __future__ import annotations

import pytest
import torch

from candi._vendored import CLOZE
from candi.decoder import LaneDeconvBlock, LaneNorm, SymmetricDecoder
from candi.model import build_model

A, CTX, RES = 6, 24, 25
READLENS = torch.tensor([36.0, 50.0, 76.0, 100.0, 150.0])
DECONV_NORMS_UNDER_TEST = ["lane", "group"]


def _meta(B, F, seed=0, num_assays=A):
    g = torch.Generator().manual_seed(seed)
    m = torch.zeros(B, 4, F)
    for c in range(F):
        m[:, 0, c] = 22.0 + 6.0 * torch.rand(B, generator=g)
        m[:, 1, c] = float(c if c < num_assays else num_assays)
        m[:, 2, c] = READLENS[torch.randint(0, 5, (B,), generator=g)]
        m[:, 3, c] = torch.randint(0, 2, (B,), generator=g).float()
    return m


def _batch(B=2, seed=0, num_assays=A):
    g = torch.Generator().manual_seed(seed)
    G = CTX * RES
    x_sig = (torch.rand(B, CTX, num_assays, generator=g) * 50).round()
    ctrl = (torch.rand(B, CTX, 1, generator=g) * 50).round()
    x_dna = torch.zeros(B, G, 4)
    x_dna.scatter_(2, torch.randint(0, 4, (B, G), generator=g).unsqueeze(-1), 1.0)
    return dict(x_data=torch.cat([x_sig, ctrl], dim=2), x_dna=x_dna,
                x_meta=_meta(B, num_assays + 1, seed, num_assays),
                y_meta=_meta(B, num_assays, seed + 1, num_assays))


def _cloze(batch, a: int):
    """Cloze assay `a` in BOTH signal and metadata, the way `prepare_masked_batch` does.

    The encoder cross-checks the two derivations and aborts if they disagree, so a test that masked
    only one of them would be testing the abort, not the substitution.
    """
    b = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    b["x_data"][:, :, a] = CLOZE
    b["x_meta"][:, :, a] = CLOZE
    return b


def _build(**kw):
    torch.manual_seed(0)
    m = build_model(embed_dim=16, num_assays=A, context_length=CTX, nhead=2, decoder_lane=4, **kw)
    m.eval()
    return m


# ---------------------------------------------------------------------------
# it builds, runs, and produces gradient
# ---------------------------------------------------------------------------

def test_forward_and_backward():
    m = _build()
    b = _batch()
    out = m(b["x_data"], b["x_dna"], b["x_meta"], b["y_meta"])
    assert out["mu"].shape == (2, CTX, A)
    for k in ("p", "n", "eta", "log2_mu", "mu"):
        assert torch.isfinite(out[k]).all(), f"non-finite {k}"
    out["mu"].sum().backward()
    assert [p for p in m.parameters() if p.grad is not None and p.grad.abs().max() > 0]


def test_decode_latent_matches_forward():
    """The eval path and the training path must be the SAME function."""
    m = _build()
    b = _batch()
    direct = m(b["x_data"], b["x_dna"], b["x_meta"], b["y_meta"])
    z = m.encode(b["x_data"], b["x_dna"], b["x_meta"])
    staged = m.decode_latent(z, b["y_meta"])
    for k in direct:
        assert torch.allclose(direct[k], staged[k], atol=1e-6), f"{k} diverges"


def test_masked_assay_still_predicts():
    """Cloze one assay and check the model still emits a finite prediction for it."""
    m = _build()
    b = _cloze(_batch(), a=1)
    out = m(b["x_data"], b["x_dna"], b["x_meta"], b["y_meta"])
    assert torch.isfinite(out["mu"][:, :, 1]).all()


# ---------------------------------------------------------------------------
# claim 1 — the lanes are disjoint
# ---------------------------------------------------------------------------

def test_grouped_deconv_does_not_mix_lanes():
    torch.manual_seed(0)
    blk = LaneDeconvBlock(num_assays=A, lane=4).eval()
    x = torch.randn(2, 6, A, 4)
    base = blk(x)
    for a in range(A):
        bumped = x.clone()
        bumped[:, :, a, :] += 3.0
        moved = (blk(bumped) - base).abs().amax(dim=-1).amax(dim=1).amax(dim=0)   # [A]
        assert moved[a] > 1e-6
        others = torch.cat([moved[:a], moved[a + 1:]])
        assert others.max() < 1e-6, f"assay {a} leaked across the grouped deconv"


@pytest.mark.parametrize("deconv_norm", DECONV_NORMS_UNDER_TEST)
def test_deconv_norm_never_mixes_lanes(deconv_norm):
    torch.manual_seed(0)
    n = LaneNorm(8, deconv_norm).eval()
    z = torch.randn(2, 6, A, 8)
    base = n(z)
    for a in range(A):
        bumped = z.clone()
        bumped[:, :, a, :] += 5.0
        moved = (n(bumped) - base).abs().amax(dim=-1).amax(dim=1).amax(dim=0)
        others = torch.cat([moved[:a], moved[a + 1:]])
        assert others.max() < 1e-6, f"{deconv_norm}: lane {a} leaked into another lane"


def test_group_norm_preserves_the_profile_along_the_sequence():
    """The distinction that makes `group` a real option rather than a restyling of `lane`.

    `lane` divides each POSITION by its own scale, so two positions differing only in amplitude
    become identical. `group` divides the whole lane by one scale, so their ratio survives — and
    that ratio is the peak structure the model is being asked to predict.
    """
    z = torch.zeros(1, 4, 1, 8)
    z[0, 0, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0])
    z[0, 3, 0] = z[0, 0, 0] * 10.0
    lane = LaneNorm(8, "lane").eval()(z)
    group = LaneNorm(8, "group").eval()(z)
    assert torch.allclose(lane[0, 0, 0], lane[0, 3, 0], atol=1e-4), \
        "'lane' should erase the amplitude difference between the two positions"
    assert not torch.allclose(group[0, 0, 0], group[0, 3, 0], atol=1e-2), \
        "'group' must KEEP the amplitude difference — that is the whole point of the option"


def test_deconv_norm_reaches_every_deconv_block():
    for mode in DECONV_NORMS_UNDER_TEST:
        m = _build(deconv_norm=mode)
        modes = {blk.norm.mode for blk in m.decoder.blocks}
        assert modes == {mode}, f"deconv blocks on {modes}, expected {mode}"


# ---------------------------------------------------------------------------
# claims 2 and 3 — conditioning is per-assay, and starts as an exact no-op
# ---------------------------------------------------------------------------

def test_film_is_per_assay_not_pooled():
    """Changing ONE assay's covariate row must move that assay's output and no other's.

    `PerLaneFiLM` is zero-initialised, so the projection is trained here first — otherwise gamma and
    beta are identically zero and the test would pass for the wrong reason.
    """
    torch.manual_seed(0)
    dec = SymmetricDecoder(num_assays=A, lane=4, meta_embed_dim=16, in_dense_dim=8)
    for f in dec.film_layers:
        torch.nn.init.normal_(f.proj.weight, std=0.1)
    dec.eval()
    ym = _meta(2, A, seed=3)
    z = torch.randn(2, 6, 8)
    base = dec(z, ym)["eta"]
    a = 2
    ym2 = ym.clone()
    ym2[:, 0, a] = ym2[:, 0, a] + 4.0          # move only assay a's depth
    moved = (dec(z, ym2)["eta"] - base).abs().amax(dim=(0, 1))
    assert moved[a] > 1e-6, "the assay whose covariate moved did not respond"
    others = torch.cat([moved[:a], moved[a + 1:]])
    assert others.max() < 1e-6, ("a covariate change on one assay moved another assay's output — "
                                 "the FiLM is pooled, not per-assay")


def test_conditioning_starts_as_an_exact_no_op():
    """Every decoder FiLM is adaLN-ZERO, so at init y_meta must move NOTHING.

    `use_offset=False` removes the depth offset, which is a separate, non-FiLM y_meta path and is
    live from step 0.
    """
    m = _build(use_offset=False)
    b = _batch()
    z = m.encode(b["x_data"], b["x_dna"], b["x_meta"])
    base = m.decode_latent(z, b["y_meta"])["eta"]
    ym = b["y_meta"].clone()
    ym[:, 0, :] += 3.0
    assert torch.allclose(m.decode_latent(z, ym)["eta"], base, atol=1e-6), \
        "conditioning is live at init — some FiLM projection is not zero-initialised"


# ---------------------------------------------------------------------------
# the anchor
# ---------------------------------------------------------------------------

def test_param_count_is_anchored():
    """The shipped configuration is 2,353,634 parameters on the 35-assay EIC panel.

    Anchored because it is the number every recorded run was produced at, and because a silent
    change to it means the model is no longer the model those results describe.
    """
    torch.manual_seed(0)
    m = build_model(embed_dim=32, num_assays=35, context_length=768, d_model=288, nhead=4,
                    n_transformer_layers=2, decoder_lane=8, dropout=0.1)
    assert sum(p.numel() for p in m.parameters()) == 2_353_634
