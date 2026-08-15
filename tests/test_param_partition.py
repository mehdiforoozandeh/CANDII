"""h51 V3: the no-decay / decay parameter partition, ASSERTED rather than eyeballed.

h51's own wording: "the partition must be ASSERTED IN A TEST, not eyeballed. The test enumerates
`named_parameters()`, checks every parameter lands in exactly one group, and fails on any unclassified
name — a silent name-filter miss would put trunk parameters in the wd=0 group and quietly turn this
into a different experiment."

Four things are proved here, and the last two are the ones that actually protect the node:

  1. EXHAUSTIVE + DISJOINT on the real model, by tensor count AND element count.
  2. POSITIVE MEMBERSHIP — film_proj, the embedding tables, every bias and every norm affine are in
     the no-decay group; trunk and head weights are in the decay group.
  3. NO SILENT FALLBACK — an unrecognized module or parameter name RAISES. Every one of the negative
     tests below fails the moment someone adds an `else: decay` branch to the classifier.
  4. THE CONTROL ARM'S ASSUMPTION — `AdamW` with every group at `weight_decay=0.0` produces
     bit-identical updates to the frozen default `Adam(..., weight_decay=0.0)`. h51 asserts this in
     prose ("the optimizer swap is therefore free of confound by construction"); this proves it.

Construction only — no h5, no forward pass through the big model, no GPU.
"""
from __future__ import annotations

import copy
import inspect

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from candi.model import build_model
from candi.param_groups import (
    DECAY,
    EMBEDDING_TYPES,
    NORM_TYPES,
    NO_DECAY,
    ParamClassificationError,
    build_adamw,
    build_param_groups,
    classify_parameters,
    validate_optimizer_config,
)

LR = 5e-4


@pytest.fixture(scope="module")
def model():
    """Small but structurally complete: both metadata towers, FiLM, x-transformers blocks, deconv."""
    torch.manual_seed(0)
    return build_model(embed_dim=16, n_transformer_layers=1, decoder_lane=8)


@pytest.fixture(scope="module")
def part(model):
    return classify_parameters(model)


# ---------------------------------------------------------------------------
# 1 — exhaustive and disjoint
# ---------------------------------------------------------------------------

def test_partition_is_exhaustive_and_disjoint_by_tensor_count(model, part):
    named = list(model.named_parameters())
    assert len(part.no_decay) + len(part.decay) == len(named)
    assert not (part.no_decay_names & part.decay_names)
    assert part.no_decay_names | part.decay_names == {n for n, _ in named}


def test_partition_is_exhaustive_by_element_count(model, part):
    total = sum(p.numel() for _, p in model.named_parameters())
    grouped = (sum(p.numel() for _, p in part.no_decay)
               + sum(p.numel() for _, p in part.decay))
    assert grouped == total, f"{grouped:,} grouped != {total:,} in named_parameters()"


def test_every_parameter_is_classified_exactly_once(model, part):
    """Identity-level check: `id(p)`, not name, so a tied tensor cannot be double-counted."""
    ids = [id(p) for _, p in part.no_decay] + [id(p) for _, p in part.decay]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for p in model.parameters()}


def test_both_groups_are_non_empty(part):
    assert part.no_decay and part.decay


@pytest.mark.parametrize("num_assays,context_length", [(8, 768), (35, 1200)])
def test_partition_holds_at_production_scale(num_assays, context_length):
    """The 35-assay `panel.eic_full` model is what h51 actually runs; assert it, don't extrapolate."""
    torch.manual_seed(0)
    m = build_model(num_assays=num_assays, context_length=context_length)
    p = classify_parameters(m)
    named = list(m.named_parameters())
    assert len(p.no_decay) + len(p.decay) == len(named)
    assert (sum(t.numel() for _, t in p.no_decay) + sum(t.numel() for _, t in p.decay)
            == sum(t.numel() for _, t in named))
    # the split is meaningful, not 99/1 in either direction
    s = p.summary()
    assert s["decay_params"] > s["no_decay_params"], "the trunk must dominate the decayed group"


# ---------------------------------------------------------------------------
# 2 — positive membership
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "decoder.film_layers.0.proj.weight",
    "decoder.film_layers.3.proj.bias",
])
def test_film_proj_in_no_decay(part, name):
    assert name in part.no_decay_names, part.reasons.get(name)


def test_every_decoder_film_projection_is_no_decay(model, part):
    """By type, not by name: renaming a FiLM attribute must not silently start decaying it.

    Decoder only — h51 pre-registered the decoder's FiLM as conditioning and left the encoder
    tower's projections in the decay group. That asymmetry is deliberate; see `param_groups`.
    """
    from candi.param_groups import _FILM_TYPES
    films = [n for n, m in model.named_modules() if isinstance(m, _FILM_TYPES)]
    assert films, "no FiLM modules found — the type rule is looking for the wrong classes"
    for f in films:
        for pn, _ in model.get_submodule(f).named_parameters():
            full = f"{f}.{pn}"
            assert full in part.no_decay_names, f"{full} is conditioning and must not be decayed"


def test_metadata_embedding_tables_in_no_decay(model, part):
    tables = [f"{n}.weight" for n, m in model.named_modules() if isinstance(m, EMBEDDING_TYPES)]
    # both towers carry assay_id + run_type tables; the h51 filter says all of them are conditioning
    assert len(tables) == 4, tables
    for t in tables:
        assert t in part.no_decay_names, f"{t} is an embedding table and must not be decayed"


@pytest.mark.parametrize("name", [
    "decoder.input_proj.weight",
    "decoder.blocks.0.deconv.weight",
    "decoder.head_eta.0.weight",
    "decoder.head_n.2.weight",
])
def test_trunk_and_head_weights_in_decay(part, name):
    assert name in part.decay_names, part.reasons.get(name)


def test_every_bias_is_in_no_decay(model, part):
    biases = [n for n, _ in model.named_parameters() if n.endswith(".bias")]
    assert biases
    leaked = [n for n in biases if n not in part.no_decay_names]
    assert not leaked, f"biases in the decay group: {leaked}"


def test_every_norm_affine_is_in_no_decay_found_by_module_type(model, part):
    """The load-bearing test. Norm modules are found BY TYPE; their affines are then checked by name.

    A LayerNorm's scale is `.weight` — the same name a Linear's matrix carries — so this set can only
    be built from the module graph. If someone replaces the type walk with a name filter, the
    x-transformers norms (whose affine is `.gamma`) drop out of the no-decay group and this fails.
    """
    affines = []
    for mod_name, module in model.named_modules():
        if isinstance(module, NORM_TYPES):
            for p_name, _ in module.named_parameters(recurse=False):
                affines.append(f"{mod_name}.{p_name}" if mod_name else p_name)
    assert len(affines) >= 20, f"only {len(affines)} norm affines found — the type walk broke"
    leaked = [n for n in affines if n not in part.no_decay_names]
    assert not leaked, f"norm affines in the decay group: {leaked}"


def test_norm_affines_are_not_separable_by_name(model, part):
    """Documents WHY the type walk exists: both failure modes of a name filter are present here."""
    norm_params = set()
    for mod_name, module in model.named_modules():
        if isinstance(module, NORM_TYPES):
            for p_name, _ in module.named_parameters(recurse=False):
                norm_params.add(f"{mod_name}.{p_name}" if mod_name else p_name)
    # (a) norm affines called `.weight`, indistinguishable from a Linear's matrix by name
    assert any(n.endswith(".weight") for n in norm_params)
    # (b) norm affines NOT called `.weight` at all (x-transformers uses `.gamma`)
    gammas = [n for n in norm_params if n.endswith(".gamma")]
    assert gammas, "expected x-transformers LayerNorm `.gamma` affines"
    for g in gammas:
        assert g in part.no_decay_names


def test_conditioning_sentinels_and_mask_token_in_no_decay(part):
    """The bare `nn.Parameter`s: the four continuous-field MISSING/CLOZE rows, and the mask token."""
    for tower in ("encoder.metadata_embedding", "decoder.meta_embedding"):
        for field in ("depth_missing_emb", "depth_cloze_emb",
                      "readlen_missing_emb", "readlen_cloze_emb"):
            assert f"{tower}.{field}" in part.no_decay_names
    assert "encoder.mask_injector.mask_embedding" in part.no_decay_names


def test_metadata_fusion_matmuls_are_decayed(part):
    """A DELIBERATE reading of h51's filter, pinned so it cannot drift silently.

    h51 lists embeddings / film_proj / norm affines / biases as the no-decay group — the
    GPT-3 / LLaMA convention — so the conditioning embedder's own matmuls (`depth_proj`,
    `read_length_proj`, the two `fusion` Linears) ARE decayed, and so are the encoder's per-conv FiLM
    projections. If the PI wants the wider "whole conditioning pathway" reading, this test is the
    thing that must be updated with it.
    """
    for name in ("encoder.metadata_embedding.depth_proj.weight",
                 "encoder.metadata_embedding.read_length_proj.weight",
                 "encoder.metadata_embedding.fusion.0.weight",
                 "encoder.signal_tower.per_conv_film_layers.0.proj.weight"):
        assert name in part.decay_names, part.reasons.get(name)


# ---------------------------------------------------------------------------
# 3 — no silent fallback
# ---------------------------------------------------------------------------

class _UnknownModule(nn.Module):
    """A module type `param_groups` has never heard of, holding a parameter with a novel name."""

    def __init__(self) -> None:
        super().__init__()
        self.mystery_parameter = nn.Parameter(torch.zeros(4))


def test_unknown_module_type_raises_instead_of_defaulting(model):
    """THE test that protects the experiment. Adding `else: decay` to the classifier fails it."""
    m = copy.deepcopy(model)
    m.add_module("injected", _UnknownModule())
    with pytest.raises(ParamClassificationError) as exc:
        classify_parameters(m)
    msg = str(exc.value)
    assert "injected.mystery_parameter" in msg, msg
    assert "_UnknownModule" in msg, msg


def test_unknown_parameter_name_on_a_known_module_raises(model):
    """Recognizing the CLASS is not enough: a new parameter on a known class must also raise."""
    m = copy.deepcopy(model)
    m.decoder.head_eta[0].register_parameter("extra_scale", nn.Parameter(torch.zeros(3)))
    with pytest.raises(ParamClassificationError) as exc:
        classify_parameters(m)
    assert "extra_scale" in str(exc.value)


def test_new_raw_parameter_on_the_metadata_embedder_raises(model):
    """The conditioning embedder holds bare Parameters; a fifth one must not be assumed no-decay."""
    m = copy.deepcopy(model)
    m.encoder.metadata_embedding.register_parameter("newfield_emb", nn.Parameter(torch.zeros(3)))
    with pytest.raises(ParamClassificationError) as exc:
        classify_parameters(m)
    assert "newfield_emb" in str(exc.value)


def test_unclassified_parameter_never_reaches_an_optimizer(model):
    """Belt and braces: the failure happens at group construction, not silently at `.step()`."""
    m = copy.deepcopy(model)
    m.add_module("injected", _UnknownModule())
    with pytest.raises(ParamClassificationError):
        build_param_groups(m, trunk_wd=0.1)
    with pytest.raises(ParamClassificationError):
        build_adamw(m, lr=LR, trunk_wd=0.1)


def test_tied_parameter_is_counted_once():
    net = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    net[1].weight = net[0].weight                     # tie
    p = classify_parameters(net)
    assert len(p.no_decay) + len(p.decay) == len(list(net.named_parameters()))
    assert not (p.no_decay_names & p.decay_names)


def test_tie_across_the_two_groups_raises():
    """One tensor reached as both a trunk matmul and a norm affine is ambiguous, not silently fine."""
    net = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
    net[1].weight = net[0].weight
    with pytest.raises(ParamClassificationError, match="two decay groups"):
        classify_parameters(net)


# ---------------------------------------------------------------------------
# 4 — the control arm's assumption, proved on a synthetic problem
# ---------------------------------------------------------------------------

def _synthetic_problem(seed: int = 0):
    """A tiny net that exercises BOTH groups: Linear matmuls (decay) + biases and a LayerNorm (no)."""
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(6, 8), nn.GELU(), nn.LayerNorm(8),
                        nn.Linear(8, 4), nn.GELU(), nn.Linear(4, 2))
    g = torch.Generator().manual_seed(seed + 1)
    x = torch.randn(16, 6, generator=g)
    y = torch.randn(16, 2, generator=g)
    return net, x, y


def _step(net, opt, x, y):
    opt.zero_grad()
    F.mse_loss(net(x), y).backward()
    opt.step()


def test_adamw_all_groups_zero_is_bit_identical_to_adam():
    """h51's control arm rests on this: at wd=0 coupled L2 and decoupled shrink are the same update.

    AdamW's decoupled term is `p.mul_(1 - lr*wd)`, which at wd=0 is a multiply by exactly 1.0 (an
    identity in IEEE-754), and Adam skips its coupled term entirely. So the control arm is the frozen
    default, numerically — proved over several steps rather than cited.
    """
    net_adam, x, y = _synthetic_problem()
    net_adamw = copy.deepcopy(net_adam)

    opt_adam = torch.optim.Adam(net_adam.parameters(), LR, weight_decay=0.0)
    groups, p = build_param_groups(net_adamw, trunk_wd=0.0)
    assert p.no_decay and p.decay, "a one-group 'split' would make this test vacuous"
    opt_adamw = torch.optim.AdamW(groups, lr=LR)

    for step in range(8):
        _step(net_adam, opt_adam, x, y)
        _step(net_adamw, opt_adamw, x, y)
        for (na, pa), (nw, pw) in zip(net_adam.named_parameters(), net_adamw.named_parameters()):
            assert na == nw
            assert torch.equal(pa, pw), (
                f"step {step}: {na} diverged, max|delta|={float((pa - pw).abs().max()):.3e}")


def test_nonzero_trunk_wd_moves_the_decay_group_and_only_it():
    """One step, so gradients are still identical: the decay group must move, the no-decay group not."""
    net_adam, x, y = _synthetic_problem()
    net_split = copy.deepcopy(net_adam)

    opt_adam = torch.optim.Adam(net_adam.parameters(), LR, weight_decay=0.0)
    groups, p = build_param_groups(net_split, trunk_wd=0.1)
    opt_split = torch.optim.AdamW(groups, lr=LR)

    _step(net_adam, opt_adam, x, y)
    _step(net_split, opt_split, x, y)

    ref = dict(net_adam.named_parameters())
    got = dict(net_split.named_parameters())
    changed = [n for n in p.decay_names if not torch.equal(ref[n], got[n])]
    assert changed, "trunk_wd=0.1 changed nothing — the knob is inert"
    unchanged = [n for n in p.no_decay_names if not torch.equal(ref[n], got[n])]
    assert not unchanged, f"no-decay group was decayed after all: {unchanged}"


# ---------------------------------------------------------------------------
# the ambiguity guard + the default-off contract
# ---------------------------------------------------------------------------

def test_adamw_with_nonzero_weight_decay_raises():
    with pytest.raises(ValueError, match="double-specifies"):
        validate_optimizer_config("adamw", 0.1, 0.0)


def test_adamw_with_nonzero_weight_decay_raises_at_build(model):
    with pytest.raises(ValueError, match="double-specifies"):
        build_adamw(model, lr=LR, weight_decay=0.1, trunk_wd=0.1)


def test_train_and_eval_rejects_adamw_with_weight_decay(tmp_path):
    """The guard fires BEFORE the h5 is opened — a nonexistent path still gets the decay error."""
    from candi.train import train_and_eval
    with pytest.raises(ValueError, match="double-specifies"):
        train_and_eval(h5_path=str(tmp_path / "does-not-exist.h5"), out_dir=str(tmp_path),
                       optimizer="adamw", weight_decay=0.1, trunk_wd=0.1)


def test_nonzero_trunk_wd_under_adam_raises():
    with pytest.raises(ValueError, match="inert"):
        validate_optimizer_config("adam", 0.0, 0.1)


def test_unknown_optimizer_raises():
    with pytest.raises(ValueError, match="adam"):
        validate_optimizer_config("sgd", 0.0, 0.0)


@pytest.mark.parametrize("optimizer,weight_decay,trunk_wd", [
    ("adam", 0.0, 0.0), ("adam", 0.1, 0.0), ("adamw", 0.0, 0.0), ("adamw", 0.0, 0.1),
])
def test_valid_configurations_are_accepted(optimizer, weight_decay, trunk_wd):
    assert validate_optimizer_config(optimizer, weight_decay, trunk_wd) == optimizer


def test_new_flags_are_default_off():
    from candi.train import train, train_and_eval
    for fn in (train, train_and_eval):
        sig = inspect.signature(fn)
        assert sig.parameters["optimizer"].default == "adam", fn.__name__
        assert sig.parameters["trunk_wd"].default == 0.0, fn.__name__


def test_default_adam_construction_line_is_verbatim():
    """`train.py` is FROZEN. The default path must be the historical line, not a param-group rewrite.

    If this fails because someone "unified" the two branches, the goldens are not what proves the
    default is unchanged — this is.
    """
    from candi import train as train_mod
    src = inspect.getsource(train_mod.train)
    assert "opt = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)" in src


def test_group_weight_decays_are_exactly_zero_and_trunk_wd(model):
    groups, _ = build_param_groups(model, trunk_wd=0.1)
    by_name = {g["name"]: g for g in groups}
    assert by_name[NO_DECAY]["weight_decay"] == 0.0
    assert by_name[DECAY]["weight_decay"] == 0.1
