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
from candi.precision import (DEFAULT_PRECISION, PRECISION_HELP, PRECISIONS, assert_no_grad_scaler,
                             autocast_region, fp32_fence, no_autocast)
from candi.train import CLIP_NORM, LR_SCHEDULES, _elem_nb_nll, make_lr_schedule

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

# `precision` IS DELIBERATELY ABSENT FROM BOTH DICTS, and a reader following AGENTS.md §4.6 step 3
# should know why before wondering. Both are `build_model` keyword sets, and precision is not a
# `build_model` keyword: it changes no module, no shape and no state_dict key, so it is a
# training-loop setting rather than an architecture one. Section 8 below is its equivalent pair —
# the same two claims (the default is a no-op, flipping it is not inert), tested against the thing
# precision actually changes.


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


# ---------------------------------------------------------------------------
# 8. --precision, and the fp32 fences that make bf16 safe
# ---------------------------------------------------------------------------
# The section-1/section-2 pair, for a flag that is not a `build_model` keyword. Everything here runs
# on CPU: `torch.autocast` is a dispatcher feature, not a hardware one, so CPU bf16 demotes the same
# ops CUDA bf16 does and every dtype claim below is real. What CPU CANNOT tell you is anything about
# throughput or memory — those are the reasons to want the flag, and they are not tested here.

DEV = "cpu"


def test_the_default_precision_is_fp32_and_is_an_exact_no_op():
    """Section 1's claim, for precision: the shipped default cannot move a recorded number.

    `tools/golden.py` makes the same claim at full scale and at 0 ULP. This is the version that runs
    in the suite, and it is the reason the fences could be written as unconditional `with` blocks
    rather than as `if precision == ...` branches — an unconditional fence that is not exactly inert
    at the default would re-date every number in the repo.
    """
    assert DEFAULT_PRECISION == "fp32"
    torch.manual_seed(0)
    model = build_model(**BASE).eval()
    xs = _inputs(model)
    with torch.no_grad():
        bare = model(*xs)
        with autocast_region(DEV, DEFAULT_PRECISION):
            named = model(*xs)
    for k in bare:
        worst = (bare[k] - named[k]).abs().max().item()
        assert worst == 0.0, f"{k} moved by {worst:.3e} at --precision fp32 — the default is not inert"


def test_bf16_actually_reaches_the_arithmetic():
    """The section-2 claim. Without this every fence test below would be vacuously green.

    A `--precision` that quietly resolved to fp32 — because the region was gated to CUDA, say —
    would pass every dtype assertion in this file by never demoting anything in the first place.
    """
    lin = torch.nn.Linear(4, 4)
    x = torch.randn(2, 4)
    with autocast_region(DEV, "bf16"):
        assert lin(x).dtype == torch.bfloat16
    with autocast_region(DEV, DEFAULT_PRECISION):
        assert lin(x).dtype == torch.float32


def test_fp16_is_not_offered_and_the_reason_is_arithmetic_not_taste():
    """fp16 is absent because the decoder's own clamp ceiling overflows it — silently.

    The second assertion is the argument, not a restatement of the first: `log2_mu` is clamped at 30,
    so `mu` reaches 2**30, and fp16's largest finite value is 65504. The overflow would then be
    laundered by `p = n / (n + mu)` into a finite, plausible probability rather than raised.
    """
    assert "fp16" not in PRECISIONS
    with pytest.raises(ValueError, match="precision must be one of"):
        autocast_region(DEV, "fp16")
    ceiling = build_model(**BASE).decoder.clamp_hi
    assert 2.0 ** ceiling > 65504.0, "fp16 could represent this model's mu; re-read the audit"
    assert float(torch.tensor(2.0 ** ceiling, dtype=torch.float16)) == float("inf")
    # ...and the laundering, demonstrated: the inf leaves no trace in p.
    n16 = torch.tensor(5.0, dtype=torch.float16)
    p16 = (n16 / (n16 + torch.tensor(float("inf"), dtype=torch.float16))).clamp(1e-6, 1 - 1e-6)
    assert torch.isfinite(p16) and float(p16) > 0.0


@pytest.mark.parametrize("precision", list(PRECISIONS))
def test_every_head_output_stays_fp32_whatever_the_region_says(precision):
    """THE FENCE TEST. The NB head is fp32 or the distribution it parameterises is not the one scored.

    Runs the whole model under the region, so it exercises the metadata embedders, the conv-tower
    FiLM, LaneNorm, the decoder taps and the head as one path rather than five unit tests.
    """
    torch.manual_seed(0)
    model = build_model(**BASE).eval()
    with torch.no_grad(), autocast_region(DEV, precision):
        out = model(*_inputs(model))
    for k, v in out.items():
        assert v.dtype == torch.float32, f"{k} came back {v.dtype} under --precision {precision}"
        assert torch.isfinite(v).all(), f"{k} is non-finite under --precision {precision}"


def test_the_fence_disables_autocast_and_upcasts_what_already_arrived_demoted():
    """Both halves, because doing only the first is the failure wearing the disguise of the fix."""
    with autocast_region(DEV, "bf16"):
        assert torch.is_autocast_enabled(DEV)
        x = torch.randn(4, 4, dtype=torch.bfloat16)
        with fp32_fence(x) as (y,):
            assert not torch.is_autocast_enabled(DEV), "the fence left autocast on"
            assert y.dtype == torch.float32, "the fence let an already-demoted tensor through"
        assert torch.is_autocast_enabled(DEV), "the fence did not restore the outer region"


def test_the_fence_leaves_non_float_arguments_alone():
    """Casting an index tensor to float would corrupt an embedding lookup, not just round it."""
    idx, flag = torch.arange(4), torch.tensor([True, False])
    with fp32_fence(idx, flag, None, 3.5) as (i, f, none, scalar):
        assert i.dtype == idx.dtype and torch.equal(i, idx)
        assert f.dtype == torch.bool
        assert none is None and scalar == 3.5


def test_the_fence_is_free_in_fp32():
    """`Tensor.float()` on an fp32 tensor must return the SAME object, not a copy.

    This is why the unconditional fences cost no arithmetic at the default — and why the golden gate
    still reads 0 ULP with a dozen of them on the forward path.
    """
    x = torch.randn(4, 4)
    with fp32_fence(x) as (y,):
        assert y is x


def test_no_autocast_covers_a_whole_call_tree():
    """`eval.evaluate` / `eval.quick_eval` use this form: the guarantee is 'nothing under here'."""
    with autocast_region(DEV, "bf16"):
        with no_autocast(DEV):
            assert not torch.is_autocast_enabled(DEV)
            assert torch.nn.Linear(4, 4)(torch.randn(2, 4)).dtype == torch.float32


# ---- the adaLN-zero pivot ------------------------------------------------------------------------

def test_the_adaln_pivot_rewrite_widens_the_live_band_but_does_not_rescue_it():
    """The arithmetic claim behind the rewrite, MEASURED — including where it does not help.

    `1.0 + gamma` measures the addend against ulp(1.0) = 2**-8 = 3.9e-3, a hard uniform threshold.
    `z + z*gamma` measures it against ulp(z), whose position in the mantissa varies per element, so
    the threshold becomes a soft per-element 2**-9..2**-8. That is a factor of two, not a cure — and
    this test pins BOTH ends of that, because a comment claiming the rewrite fixes small gammas
    would be wrong and the fp32 fence is what actually does the work.
    """
    torch.manual_seed(0)
    z = torch.randn(20000, dtype=torch.bfloat16)
    # 121 of these 20000 draws are EXACTLY 0.0 — bf16 randn is coarse near the mean — and for z == 0
    # both forms are exactly right (`0*(1+g) == 0` and `0 + 0*g == 0`), so neither can ever "move"
    # them. That is the whole of the 0.61% shortfall from 100%: it is not a rounding residual and not
    # a tolerance to widen, so the ceiling asserted below is the NONZERO count, exactly.
    live = int((z != 0).sum())
    assert live == 19879, f"the fixture drifted: {live} nonzero of {len(z)}"

    def moved(g):
        gam = torch.full_like(z, g)
        return int((z * (1.0 + gam) != z).sum()), int((z + z * gam != z).sum())

    # Measured on this build at seed 0 (old = z*(1+gamma), new = z + z*gamma):
    #   gamma     old              new
    #   1e-3          0 ( 0.00%)       0 ( 0.00%)   <- BOTH dead. The rewrite is not a rescue.
    #   2e-3          0 ( 0.00%)     602 ( 3.01%)
    #   3e-3          0 ( 0.00%)   12293 (61.47%)
    #   3.9e-3        0 ( 0.00%)   19685 (98.42%)   <- still below ulp(1.0); old form still asleep
    #   4e-3      19879 (99.39%)   19879 (99.39%)   <- 1+gamma finally rounds up; 99.39% = all nonzero
    assert moved(1e-3) == (0, 0), "at 1e-3 BOTH forms are dead — the rewrite is not a rescue"
    assert moved(2e-3) == (0, 602)
    assert moved(3e-3) == (0, 12293)
    assert moved(3.9e-3) == (0, 19685), "just under ulp(1.0) the old form must still be dead"
    assert moved(4e-3) == (live, live), "past ulp(1.0) both forms move every NONZERO element"


def test_the_fence_not_the_rewrite_is_what_saves_a_small_gamma():
    """The honest division of labour, stated as a test rather than left in a comment.

    A gamma of 1e-3 is dead in bf16 under EITHER algebraic form (the test above). It survives only
    because `PerLaneFiLM` runs its arithmetic in fp32, where ulp(1.0) is 6e-8.
    """
    torch.manual_seed(0)
    tap = PerLaneFiLM(embed_dim=8, lane_width=4, init="zero")
    torch.nn.init.constant_(tap.proj.bias[:tap.C], 1e-3)      # gamma = 1e-3, beta stays 0
    z, memb = torch.randn(2, 6, 3, 4), torch.randn(2, 3, 8)
    with autocast_region(DEV, "bf16"):
        out = tap(z, memb)
    assert out.dtype == torch.float32, "the tap came back demoted"
    assert not torch.equal(out, z), "a gamma of 1e-3 was rounded away — the fence is not holding"


def test_the_shipped_tap_computes_the_rewritten_pivot():
    """Not merely that the rewrite is correct — that the module actually uses it.

    Float multiplication does not distribute over addition, so with live gammas the two forms differ
    in fp32 by a rounding. That difference is what makes this test able to tell them apart, and it is
    also why the rewrite is bit-identical ONLY at the zero init (the next test).
    """
    torch.manual_seed(0)
    tap = PerLaneFiLM(embed_dim=8, lane_width=4, init="xavier")
    z, memb = torch.randn(2, 6, 3, 4), torch.randn(2, 3, 8)
    gamma, beta = tap.proj(memb).chunk(2, dim=-1)
    new = z + z * gamma.unsqueeze(1) + beta.unsqueeze(1)
    old = z * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
    assert not torch.equal(old, new), "the two forms are indistinguishable here — test is vacuous"
    assert torch.equal(tap(z, memb), new), "the tap still computes z * (1 + gamma) + beta"


def test_the_rewritten_pivot_is_an_exact_identity_at_the_zero_init():
    """Why the golden gate survives the rewrite: at gamma == beta == 0 both forms reduce to z."""
    torch.manual_seed(0)
    tap = PerLaneFiLM(embed_dim=8, lane_width=4, init="zero")
    z, memb = torch.randn(2, 6, 3, 4), torch.randn(2, 3, 8)
    assert torch.equal(tap(z, memb), z)


def test_the_decoder_taps_still_steer_under_a_bf16_region():
    """The end-to-end version: a hand-set small gamma must move the model's output under autocast.

    Set at 1e-3 — dead in bf16 under either algebraic form — so this fails if the fences are dropped,
    which is the failure that matters. It passes on the strength of the fence, not the rewrite.
    """
    torch.manual_seed(0)
    model = build_model(**BASE).eval()
    xs = _inputs(model)
    with torch.no_grad(), autocast_region(DEV, "bf16"):
        base = model(*xs)["eta"].clone()
        for tap in model.decoder.film_layers:
            if not hasattr(tap, "proj"):
                continue                      # a NoFiLM tap; nothing to steer with
            # GAMMA ONLY. `proj` emits [gamma | beta], and setting the whole bias would move the
            # output through beta no matter how gamma was written — the test would pass on the old
            # form and prove nothing. Beta is left at zero so the only live path is the pivot.
            torch.nn.init.zeros_(tap.proj.bias)
            tap.proj.bias[:tap.C] = 1e-3
        moved = model(*xs)["eta"]
    assert (moved - base).abs().max().item() > 0.0, \
        "a gamma of 1e-3 changed nothing — the conditioning pathway was rounded away"


# ---- the loss ------------------------------------------------------------------------------------

def _unfenced_nb_nll(p, n, target, eps: float = 1e-6):
    """`train._elem_nb_nll` with the fence REMOVED — arithmetic stays in the input dtype.

    Kept here, in the test file, so the tests below can compare against what this code did before
    the fence and would do again if anyone deleted it. It is the only reason those tests can fail
    for the right reason.
    """
    probs = (1.0 - p).clamp(eps, 1.0 - eps)
    total = n.clamp_min(eps)
    dist = torch.distributions.NegativeBinomial(total_count=total, probs=probs)
    return -dist.log_prob(target.clamp_min(0.0))


def test_the_loss_fence_does_not_perturb_the_real_call_path():
    """The fence must be invisible when the inputs are fp32 — which is how the model actually calls it.

    The head is itself fenced, so `p` and `n` reach the loss in fp32 and this is the path that runs
    every step. Feeding fp32 under an autocast region isolates the fence without destroying the
    inputs first, and the answer must be bit-identical to the unfenced fp32 computation.
    """
    p, n, y = torch.full((64,), 1.0 - 1e-4), torch.full((64,), 5.0), torch.zeros(64)
    base = _unfenced_nb_nll(p, n, y)
    with autocast_region(DEV, "bf16"):
        got = _elem_nb_nll(p, n, y)
    assert got.dtype == torch.float32
    assert torch.equal(base, got), "the fence moved a number on the fp32 path"


def test_the_loss_fence_changes_the_answer_when_the_inputs_arrive_demoted():
    """The test that would FAIL if the fence were deleted. Everything else here would not notice.

    A caller that hands the loss bf16 tensors — which is what an unfenced head would do — cannot
    have the precision of `p` restored; that is gone at the call. What the fence still buys is that
    every operation AFTER it runs in fp32, and the numbers show how much that is worth:

        p = 1 - 1e-4   fenced 5.000002e-06     unfenced 3.312778e-02    (a factor of 6600)

    The mechanism is the clamp BOUND, not just the subtraction. `1.0 - eps` with eps = 1e-6 is a
    value bf16 cannot hold as anything but 1.0, so an unfenced `probs.clamp(eps, 1 - eps)` clamps
    against exactly 1.0 — and `probs = 1.0` is outside NegativeBinomial's support, so for small `p`
    the unfenced path does not merely lose accuracy, it leaves the distribution entirely.
    """
    n, y = torch.full((8,), 5.0), torch.zeros(8)
    p16 = torch.full((8,), 1.0 - 1e-4).bfloat16()
    assert float(p16[0]) == 1.0, "the hazard did not reproduce: p did not round to 1.0 in bf16"

    fenced = _elem_nb_nll(p16, n.bfloat16(), y)
    unfenced = _unfenced_nb_nll(p16, n.bfloat16(), y)
    assert fenced.dtype == torch.float32 and torch.isfinite(fenced).all()
    ratio = float(unfenced[0]) / float(fenced[0])
    assert ratio > 1e3, f"fenced and unfenced agree to within {ratio:.1f}x — the fence is not biting"

    # `1.0 - eps` collapses to 1.0 in bf16, so probs clamps onto the open bound of the support.
    assert float(torch.tensor(1.0 - 1e-6, dtype=torch.bfloat16)) == 1.0
    p_small = torch.full((8,), 1e-6).bfloat16()
    with pytest.raises(ValueError):
        _unfenced_nb_nll(p_small, n.bfloat16(), y)
    assert torch.isfinite(_elem_nb_nll(p_small, n.bfloat16(), y)).all(), \
        "the fenced path must stay inside the support where the unfenced one leaves it"


def test_the_loss_fence_matters_most_where_the_lgamma_terms_are_large():
    """The other half of what the fence buys: the reductions, not just the clamp.

    NB log-prob is differences of lgammas, and those grow with `n`. Measured fenced-vs-unfenced
    relative disagreement on this build: n=5 -> 2.2e-03, n=50 -> 1.0e-02, n=5000 -> 2.3e-02. The
    error grows with the count, which is the direction that matters — deep tracks are where the
    likelihood carries the most information.
    """
    y = torch.full((8,), 10.0)
    p16 = torch.full((8,), 0.3).bfloat16()
    rel = []
    for nv in (5.0, 50.0, 5000.0):
        n16 = torch.full((8,), nv).bfloat16()
        f, u = float(_elem_nb_nll(p16, n16, y)[0]), float(_unfenced_nb_nll(p16, n16, y)[0])
        rel.append(abs(f - u) / abs(f))
    assert rel[0] < rel[1] < rel[2], f"the error should grow with n; got {rel}"
    assert rel[2] > 1e-2, f"bf16 lgamma should cost >1% at n=5000; got {rel[2]:.2e}"


# ---- the grad probe ------------------------------------------------------------------------------

def test_a_grad_scaler_is_refused_rather_than_silently_rescaling_the_probe():
    """bf16 needs no scaler; one added quietly would corrupt every `grad/*` number.

    `train._grad_norms` reads `p.grad` between `backward()` and `clip_grad_norm_`. A scaler that has
    not been `unscale_`d at that point multiplies every norm — and every `*_over_trunk` ratio — by a
    dynamically changing scale factor, which is exactly the drifting common factor that would hide a
    covariate unable to steer the model.
    """
    assert assert_no_grad_scaler(None) is None
    with pytest.raises(ValueError, match="unscale_"):
        assert_no_grad_scaler(object())


def test_the_train_step_refuses_a_scaler_at_its_own_door():
    """The guard has to be on the step, not only in the helper, or it guards nothing."""
    import inspect
    from candi.train import _train_step
    params = inspect.signature(_train_step).parameters
    assert params["scaler"].default is None
    assert params["precision"].default == DEFAULT_PRECISION
    # No `device` parameter, deliberately: the step reads it off the model, so a caller cannot pass
    # a device that disagrees with where the weights are and turn bf16 into a silent no-op.
    assert "device" not in params
    with pytest.raises(ValueError, match="unscale_"):
        _train_step(None, None, None, None, None, scaler=object())


# ---- the flag surface ----------------------------------------------------------------------------

def test_precision_is_not_an_arch_key():
    """It must NOT be, or `--arch-from` could not re-score a bf16 checkpoint in fp32.

    Precision builds no module and changes no state_dict key — autocast keeps master weights in fp32,
    so a bf16 run's checkpoint is byte-for-byte the same format as an fp32 run's. In `config.arch` it
    would become a rebuild argument that `build_model` does not accept, and since evaluation is never
    autocast, fp32 is the ONLY precision a checkpoint is ever scored at.
    """
    assert "precision" not in set(arch_keys())
    build_model(**BASE)                       # and it is not a build_model keyword at all
    with pytest.raises(TypeError):
        build_model(**BASE, precision="bf16")


def test_precision_is_recorded_so_a_reader_knows_which_one_produced_a_json():
    """The flag leaves no trace in the weights, so the run config is the only place it can live."""
    import inspect
    from candi.train import train, train_and_eval
    for fn in (train, train_and_eval):
        assert inspect.signature(fn).parameters["precision"].default == DEFAULT_PRECISION
    # It must land in BOTH config blocks: `run_cfg` is what W&B shows, `cfg_block` is what the run
    # JSON keeps, and a reader has only one of the two in front of them.
    src = inspect.getsource(train_and_eval)
    assert src.count("precision=str(precision)") == 2, \
        "precision must reach both the W&B config and the run JSON"


def test_an_unknown_precision_is_refused_by_train_before_it_opens_anything():
    """Before the h5 is opened and the model is built — not after an hour of training."""
    from candi.train import train
    with pytest.raises(ValueError, match="precision must be one of"):
        train(None, "/nonexistent.h5", DEV, precision="fp8")


def test_an_unknown_precision_is_refused_by_train_and_eval_before_it_opens_anything():
    from candi.train import train_and_eval
    with pytest.raises(ValueError, match="precision must be one of"):
        train_and_eval(h5_path="/nonexistent.h5", out_dir="/tmp", precision="fp8")


def test_the_cli_offers_exactly_the_two_precisions_and_says_what_to_expect():
    """A user who reads `--help` must not read a null speed result as a bug."""
    import argparse
    import candi.train as T
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", default=DEFAULT_PRECISION, choices=list(PRECISIONS),
                    help=T.PRECISION_HELP)
    assert ap.parse_args([]).precision == "fp32"
    assert ap.parse_args(["--precision", "bf16"]).precision == "bf16"
    with pytest.raises(SystemExit):
        ap.parse_args(["--precision", "fp16"])
    low = PRECISION_HELP.lower()
    assert "memory" in low and "not speed" in low
    assert "fp16" in low, "the help text must say why fp16 is absent, not merely omit it"
