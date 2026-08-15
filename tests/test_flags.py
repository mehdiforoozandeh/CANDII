"""The tunable surface: every default is a no-op, and every flag actually does something.

A configuration flag has two ways to be wrong, and only one of them is loud.

  1. IT CHANGES THE DEFAULT MODEL. Caught by `test_defaults_are_a_no_op` and, at full scale, by
     `tools/golden.py check`. Every recorded number was produced at the defaults, so a default that
     drifts silently re-dates every result in the repo.
  2. IT DOES NOTHING WHEN FLIPPED. This is the quiet one, and it has already happened here once:
     `--dna-dim` was exposed, documented and inert on the arms that used it, because the tower it
     fed had its width fixed elsewhere. `test_every_flag_changes_the_model` is the guard — it flips
     each flag off its default and demands the model change in a way that is externally visible.

Both have already earned their keep. `test_every_flag_changes_the_model` is what found that the
pinned x-transformers does not accept `resi_dual` at all — it had been agreed as a
`--transformer-norm-placement` choice, and without this test it would have shipped as a documented
option that raises the moment anyone selects it. `test_post_head_tap_is_an_exact_no_op_at_init` is
what found that slicing a stacked pair hands the head arithmetic a STRIDED tensor, which moved `p` by
one ULP through nothing but memory layout.
"""
from __future__ import annotations

import hashlib

import pytest
import torch

from candi.decoder import DECODER_FILM_TAPS, NoFiLM, PerLaneFiLM, PerLaneHead
from candi.model import (
    ALL_FILM_TAPS,
    DEFAULT_FILM_TAPS,
    build_model,
    build_model_from_arch,
    arch_keys,
    film_mode_from_taps,
    parse_film_taps,
)
from candi.train import CLIP_NORM, LR_SCHEDULES, make_lr_schedule

A, CTX, RES = 6, 24, 25

# Small enough to build in a fraction of a second, large enough that every axis under test is real:
# 3 conv layers, 3 deconv layers, 2 transformer layers, 6 assays + control.
BASE = dict(embed_dim=16, num_assays=A, context_length=CTX, nhead=2, decoder_lane=4)

# Every new flag, with the value it is documented to default to. Passing these explicitly must be
# indistinguishable from passing none of them — that is what "a no-op by default" means.
DOCUMENTED_DEFAULTS = dict(
    resolution=25, n_cnn_layers=3, conv_kernel_size=3, pool_size=2, expansion_factor=2,
    n_deconv_layers=3, deconv_upsample=2, deconv_kernel_size=3,
    conv_norm="layer", deconv_norm="lane", transformer_norm="layer",
    transformer_norm_placement="pre", attn_qk_norm=False,
    film_taps=DEFAULT_FILM_TAPS, film_init_encoder="xavier", film_init_decoder="zero",
    head_sharing="shared", head_hidden=0,
)

# One non-default value per flag. Each must produce a model that differs from the default in its
# parameters, its state_dict keys, or its outputs.
NON_DEFAULTS = [
    ("n_cnn_layers", dict(n_cnn_layers=2, n_deconv_layers=2)),      # paired: the geometry guard
    ("conv_kernel_size", dict(conv_kernel_size=5)),
    ("pool_size", dict(pool_size=4, n_cnn_layers=1, deconv_upsample=4, n_deconv_layers=1)),
    ("expansion_factor", dict(expansion_factor=3)),
    ("n_deconv_layers", dict(n_cnn_layers=2, n_deconv_layers=2)),
    ("deconv_upsample", dict(pool_size=8, n_cnn_layers=1, deconv_upsample=8, n_deconv_layers=1)),
    ("deconv_kernel_size", dict(deconv_kernel_size=5)),
    ("conv_norm", dict(conv_norm="instance")),
    ("conv_norm_lane", dict(conv_norm="lane")),
    ("conv_norm_group", dict(conv_norm="group")),
    ("deconv_norm", dict(deconv_norm="group")),
    ("transformer_norm", dict(transformer_norm="rmsnorm")),
    ("transformer_norm_simple", dict(transformer_norm="simple_rmsnorm")),
    ("transformer_norm_scale", dict(transformer_norm="scalenorm")),
    ("transformer_norm_placement", dict(transformer_norm_placement="post")),
    ("transformer_norm_sandwich", dict(transformer_norm_placement="sandwich")),
    ("attn_qk_norm", dict(attn_qk_norm=True)),
    ("film_taps_post_head", dict(film_taps=DEFAULT_FILM_TAPS + ("post_head",))),
    ("film_taps_drop_pre_deconv", dict(film_taps=("per_conv", "per_deconv"))),
    ("film_taps_none", dict(film_taps=())),
    ("film_taps_transformer", dict(film_taps=DEFAULT_FILM_TAPS + ("per_transformer",))),
    ("film_init_encoder", dict(film_init_encoder="zero")),
    ("film_init_decoder", dict(film_init_decoder="xavier")),
    ("head_sharing", dict(head_sharing="per_assay")),
    ("head_hidden", dict(head_hidden=16)),
]


# ---------------------------------------------------------------------------
# fingerprinting
# ---------------------------------------------------------------------------

def _inputs(model, seed: int = 1234):
    """Fixed inputs sized to whatever geometry `model` was built with."""
    g = torch.Generator().manual_seed(seed)
    ctx = model.encoder.l1
    res = model.encoder.resolution
    x_data = torch.rand(2, ctx, A + 1, generator=g) * 5.0
    x_meta = torch.zeros(2, 4, A + 1)
    x_meta[:, 0, :] = 26.0
    x_meta[:, 1, :] = torch.arange(A + 1).float()
    x_meta[:, 2, :] = 100.0
    x_meta[:, 3, :] = 1.0
    y_meta = x_meta[:, :, :A].clone()
    x_dna = torch.zeros(2, 4, ctx * res)
    x_dna[:, 0, :] = 1.0
    return x_data, x_dna, x_meta, y_meta


def _fingerprint(**kw):
    """`(n_params, sd_sha, keys, outputs)` for a model built from one seed on fixed inputs."""
    torch.manual_seed(0)
    model = build_model(**{**BASE, **kw}).eval()
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    with torch.no_grad():
        out = model(*_inputs(model))
    return (sum(p.numel() for p in model.parameters()), h.hexdigest(), sorted(model.state_dict()),
            {k: v.clone() for k, v in out.items()})


def _differs(a, b) -> bool:
    """True when two fingerprints are distinguishable at all — params, keys, weights or outputs."""
    if a[0] != b[0] or a[1] != b[1] or a[2] != b[2]:
        return True
    if set(a[3]) != set(b[3]):
        return True
    return any((a[3][k] - b[3][k]).abs().max().item() != 0.0 for k in a[3])


# ---------------------------------------------------------------------------
# 1. the defaults are a no-op
# ---------------------------------------------------------------------------

def test_defaults_are_a_no_op():
    """Naming every flag at its documented default must be bit-identical to naming none of them.

    This is `tools/golden.py` in miniature: same claim, small enough to run in the suite. If it
    fails, a default moved and every recorded number in the repo now describes a different model.
    """
    bare = _fingerprint()
    named = _fingerprint(**DOCUMENTED_DEFAULTS)
    assert bare[0] == named[0], f"parameter count moved: {bare[0]:,} -> {named[0]:,}"
    assert bare[2] == named[2], "state_dict keys moved"
    assert bare[1] == named[1], "state_dict VALUES moved — the RNG stream shifted"
    for k in bare[3]:
        worst = (bare[3][k] - named[3][k]).abs().max().item()
        assert worst == 0.0, f"{k} differs by {worst:.3e} — the defaults are not a no-op"


def test_the_shipped_tap_set_is_the_historical_conditioning():
    """The default tap set must resolve to the encoder mode the recorded runs used."""
    assert film_mode_from_taps(DEFAULT_FILM_TAPS) == "per_conv"
    m = build_model(**BASE)
    assert m.encoder.signal_tower.per_conv_film_layers is not None
    assert m.encoder.transformer_film_layers is None            # per_transformer is OFF by default
    assert all(isinstance(f, PerLaneFiLM) for f in m.decoder.film_layers)
    assert m.decoder.film_head is None                          # post_head is OFF by default


# ---------------------------------------------------------------------------
# 2. every flag actually does something
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,kw", NON_DEFAULTS, ids=[n for n, _ in NON_DEFAULTS])
def test_every_flag_changes_the_model(label, kw):
    """A flag that is a no-op when FLIPPED is worse than no flag: it documents a control nobody has.

    `--dna-dim` shipped in exactly that state. The assertion is deliberately weak about HOW the model
    changes — parameters, keys or outputs, any of them counts — because the point is only that the
    setting reaches the model at all.
    """
    assert _differs(_fingerprint(), _fingerprint(**kw)), \
        f"{label}: {kw} produced an identical model — the flag is inert"


# ---------------------------------------------------------------------------
# 3. the guards
# ---------------------------------------------------------------------------

def test_geometry_mismatch_is_refused_before_anything_is_built():
    with pytest.raises(ValueError, match="geometry disagree"):
        build_model(**BASE, n_cnn_layers=3, pool_size=2, n_deconv_layers=2, deconv_upsample=2)


def test_dna_pool_is_derived_from_resolution_not_chosen():
    """The DNA tower's two large pools must collapse exactly `resolution` bp into one bin."""
    m = build_model(**BASE, resolution=25)
    assert m.encoder.cfg.dna_pool_size == 5
    assert m.encoder.resolution == 25
    m = build_model(**BASE, resolution=100)
    assert m.encoder.cfg.dna_pool_size == 10
    assert m.encoder.resolution == 100


def test_a_non_square_resolution_is_refused():
    with pytest.raises(ValueError, match="perfect square"):
        build_model(**BASE, resolution=30)


@pytest.mark.parametrize("bad,msg", [
    ("per_conv,pre_conv", "alternative placements"),
    ("per_transformer", "requires 'per_conv'"),
    ("per_deconv,nonsense", "unknown film tap"),
])
def test_bad_tap_sets_are_named_not_silently_resolved(bad, msg):
    with pytest.raises(ValueError, match=msg):
        parse_film_taps(bad)


def test_tap_parsing_is_order_and_spelling_insensitive():
    """Two spellings of one set must compare equal, or two identical arms look different."""
    assert parse_film_taps("per_deconv,per_conv,pre_deconv") == parse_film_taps(DEFAULT_FILM_TAPS)
    assert parse_film_taps(" per_conv , per_conv ") == ("per_conv",)
    assert parse_film_taps("") == ()
    assert set(DEFAULT_FILM_TAPS) <= set(ALL_FILM_TAPS)


def test_a_disabled_tap_keeps_its_index():
    """Dropping `pre_deconv` must not renumber `per_deconv`, or the FiLM gradient logs mislabel."""
    m = build_model(**BASE, film_taps=("per_conv", "per_deconv"))
    assert isinstance(m.decoder.film_layers[0], NoFiLM)
    assert all(isinstance(f, PerLaneFiLM) for f in m.decoder.film_layers[1:])
    assert len(m.decoder.film_layers) == 4          # 1 + n_deconv_layers, unchanged


def test_no_film_is_an_exact_identity_and_owns_nothing():
    z = torch.randn(2, 6, A, 4)
    tap = NoFiLM()
    assert torch.equal(tap(z, torch.randn(2, A, 16)), z)
    assert list(tap.parameters()) == []


# ---------------------------------------------------------------------------
# 4. the alternatives behave as advertised
# ---------------------------------------------------------------------------

def test_per_assay_head_does_not_mix_assays():
    """The whole point of the alternative: each assay is read by its own weights."""
    torch.manual_seed(0)
    head = PerLaneHead(num_assays=A, lane=4, hidden=4).eval()
    x = torch.randn(2, 6, A, 4)
    base = head(x)
    for a in range(A):
        bumped = x.clone()
        bumped[:, :, a, :] += 3.0
        moved = (head(bumped) - base).abs().amax(dim=-1).amax(dim=1).amax(dim=0)
        assert moved[a] > 1e-6
        others = torch.cat([moved[:a], moved[a + 1:]])
        assert others.max() < 1e-6, f"per-assay head leaked assay {a} into another lane"


def test_per_assay_head_costs_a_times_the_parameters():
    shared = _fingerprint()[0]
    per_assay = _fingerprint(head_sharing="per_assay")[0]
    assert per_assay > shared


def test_post_head_tap_is_an_exact_no_op_at_init():
    """It is adaLN-zero like every other decoder tap, so switching it on cannot move a number."""
    off = _fingerprint(use_offset=False)
    on = _fingerprint(use_offset=False, film_taps=DEFAULT_FILM_TAPS + ("post_head",))
    for k in off[3]:
        worst = (off[3][k] - on[3][k]).abs().max().item()
        assert worst == 0.0, f"post_head moved {k} by {worst:.3e} at init — it is not zero-inited"
    assert on[0] > off[0], "post_head added no parameters"


def test_film_init_choices_reach_the_projection():
    zero = build_model(**BASE, film_init_encoder="zero")
    w = zero.encoder.signal_tower.per_conv_film_layers[0].proj.weight
    assert float(w.abs().max()) == 0.0
    live = build_model(**BASE, film_init_encoder="xavier")
    w = live.encoder.signal_tower.per_conv_film_layers[0].proj.weight
    assert float(w.abs().max()) > 0.0


# ---------------------------------------------------------------------------
# 5. the optimisation knobs
# ---------------------------------------------------------------------------

def _lrs(total=100, **kw):
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=1.0)
    sched = make_lr_schedule(opt, total, **kw)
    out = []
    for _ in range(total):
        out.append(sched.get_last_lr()[0])
        opt.step()
        sched.step()
    return out


def test_the_default_schedule_is_the_frozen_cosine():
    """Warmup over the first 10%, then cosine from 1.0 down to the 0.1 floor."""
    lrs = _lrs()
    assert lrs[0] == pytest.approx(0.0)
    assert lrs[10] == pytest.approx(1.0)               # peak at the end of warmup
    assert lrs[-1] == pytest.approx(0.1, abs=2e-3)     # floor = lr_min_ratio
    assert lrs == pytest.approx(_lrs(schedule="cosine", warmup_frac=0.1, min_ratio=0.1))


@pytest.mark.parametrize("schedule", LR_SCHEDULES)
def test_every_schedule_warms_up_and_only_cosine_is_the_default(schedule):
    lrs = _lrs(schedule=schedule)
    assert lrs[0] == pytest.approx(0.0)
    assert lrs[10] == pytest.approx(1.0)
    if schedule == "constant":
        assert lrs[-1] == pytest.approx(1.0)
    else:
        assert lrs[-1] < 0.5
    if schedule != "cosine":
        assert lrs != pytest.approx(_lrs())


def test_warmup_and_floor_are_not_inert():
    assert _lrs(warmup_frac=0.5)[10] < _lrs(warmup_frac=0.1)[10]
    assert _lrs(min_ratio=0.5)[-1] > _lrs(min_ratio=0.1)[-1]


def test_an_unknown_schedule_is_refused():
    with pytest.raises(ValueError, match="lr_schedule must be one of"):
        _lrs(schedule="triangular")


def test_the_clip_default_is_still_one():
    """Frozen: every recorded run clipped at 1.0, and the flag defaults to it."""
    assert CLIP_NORM == 1.0


# ---------------------------------------------------------------------------
# 6. an architecture can be rebuilt from its own run JSON
# ---------------------------------------------------------------------------

def test_arch_round_trips_through_build_model_from_arch():
    """`--arch-from` is only worth having if it rebuilds the model bit-for-bit."""
    arch = dict(BASE, **DOCUMENTED_DEFAULTS)
    arch["film_taps"] = list(arch["film_taps"])          # as JSON would store it
    torch.manual_seed(0)
    a = build_model_from_arch(dict(arch)).eval()
    torch.manual_seed(0)
    b = build_model(**BASE).eval()
    assert sorted(a.state_dict()) == sorted(b.state_dict())
    for k, v in a.state_dict().items():
        assert torch.equal(v, b.state_dict()[k]), f"{k} differs"


def test_arch_keys_covers_every_flag_this_test_file_knows_about():
    """A flag that never lands in `config.arch` produces an un-rescorable checkpoint."""
    keys = set(arch_keys())
    missing = sorted(set(DOCUMENTED_DEFAULTS) - keys)
    assert not missing, f"{missing} are build_model arguments but not in arch_keys()"


def test_an_unknown_arch_key_raises_rather_than_being_dropped():
    with pytest.raises(ValueError, match="does not accept"):
        build_model_from_arch(dict(BASE, from_the_future="whatever"))


def test_decoder_taps_are_a_subset_of_the_whole_set():
    assert set(DECODER_FILM_TAPS) <= set(ALL_FILM_TAPS)


# ---------------------------------------------------------------------------
# 7. the alternatives are trainable, not merely constructible
# ---------------------------------------------------------------------------
# Everything above runs a FORWARD pass, so it catches a shape bug. It does not catch a setting that
# builds and predicts but produces no gradient — a parameter stack the autograd graph never reaches,
# a norm that divides by a constant, a tap wired outside the loss path. A flag nobody can train
# through is inert in the only way that matters, and it would pass every other test in this file.

TRAINABLE = [
    ("per_assay head", dict(head_sharing="per_assay")),
    ("instance conv norm", dict(conv_norm="instance")),
    ("lane conv norm", dict(conv_norm="lane")),
    ("group deconv norm", dict(deconv_norm="group")),
    ("post_head tap", dict(film_taps=DEFAULT_FILM_TAPS + ("post_head",))),
    ("per_transformer tap", dict(film_taps=DEFAULT_FILM_TAPS + ("per_transformer",))),
    ("rmsnorm transformer", dict(transformer_norm="rmsnorm")),
    ("post-norm transformer", dict(transformer_norm_placement="post")),
    ("shallow geometry", dict(n_cnn_layers=2, n_deconv_layers=2)),
    ("wide head", dict(head_hidden=16)),
]


@pytest.mark.parametrize("label,kw", TRAINABLE, ids=[n for n, _ in TRAINABLE])
def test_every_alternative_produces_gradient(label, kw):
    torch.manual_seed(0)
    model = build_model(**{**BASE, **kw})
    out = model(*_inputs(model))
    for k in ("p", "n", "eta", "log2_mu", "mu"):
        assert torch.isfinite(out[k]).all(), f"{label}: non-finite {k}"
    out["mu"].sum().backward()
    live = [n for n, p in model.named_parameters()
            if p.grad is not None and float(p.grad.abs().max()) > 0]
    assert live, f"{label}: nothing in the model received gradient"
    # The parameters this setting ADDS must be among the live ones, not just the ones it left alone.
    if kw.get("head_sharing") == "per_assay":
        assert any(".w1" in n or ".w2" in n for n in live), "the per-assay head weights are dead"
    if "post_head" in kw.get("film_taps", ()):
        # zero-inited, so gamma/beta are 0 and the tap is an identity — but the PROJECTION must
        # still see gradient, or it can never leave that init and the tap is decorative forever.
        assert any("film_head" in n for n in live), "the post_head projection receives no gradient"
