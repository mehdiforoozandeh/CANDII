"""h51 optimizer parameter groups: a no-decay conditioning group + a decayed trunk.

NEW FILE. Nothing here runs on the default `--optimizer adam` path. `train.py` imports this module
only inside its `adamw` branch, so a run that passes none of the new flags still constructs its
optimizer through the unchanged `torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)`
line and is bit-identical to the pre-h51 kit.

WHY THIS IS NOT A NAME FILTER
-----------------------------
h51 pre-registers the split as "`*_embedding`, `film_proj`, LayerNorm affines and biases get
`weight_decay=0`; trunk and heads get the decay". Implementing that by matching parameter NAMES is
the silent failure this experiment cannot have:

  * a `nn.LayerNorm`'s scale parameter is called `.weight` — the same name a `nn.Linear`'s matrix
    carries. No substring rule separates them.
  * x-transformers' own `LayerNorm` calls its affine `.gamma`, so a rule looking for `"norm"` or
    `"weight"` misses all six of them in this model.

A miss in either direction moves trunk parameters into the zero-decay group and quietly turns the
`split` arm into a different experiment than the one that was pre-registered. So classification walks
`model.named_modules()` and assigns each module's DIRECT parameters by the module's TYPE, and — for
the two things h51 names explicitly — by module IDENTITY. Group membership is tracked by `id(p)`, so
a shared/tied parameter can never land in both groups.

THERE IS NO FALLBACK. A module class this file does not positively recognize, or a parameter name a
recognized class does not declare, raises `ParamClassificationError` naming the offender. Adding a
module to the model without deciding where its parameters belong is meant to break the run loudly,
not to be swept into the decay group.

THE PARTITION, EXPLICITLY (this is the part a reader should check against h51)
-----------------------------------------------------------------------------
NO-DECAY:
  * every `nn.Embedding` table                      (assay_embedding, runtype_embedding, both towers)
  * `film_proj`, resolved as a module               (`decoder.film_proj` weight AND bias)
  * every normalization affine, resolved by type    (nn.LayerNorm, x-transformers LayerNorm, ...)
  * every bias of every Linear / Conv / ConvTranspose
  * the metadata embedder's four sentinel vectors   (`depth_missing_emb`, ... — these are the
    continuous fields' MISSING/CLOZE embedding ROWS; the categorical tables carry theirs inside the
    `nn.Embedding` above, so classifying them apart would be inconsistent)
  * `mask_injector.mask_embedding`                  (a learned per-assay token table; h51's own
    `*_embedding` filter catches it, and it is an embedding by role)
DECAY (trunk + heads):
  * every Linear / Conv1d / ConvTranspose1d `weight` that is not `film_proj`

TWO CONSEQUENCES WORTH SEEING RATHER THAN DISCOVERING LATER, both of which follow h51's own filter
and are deliberate, not oversights:
  * `metadata_embedding.depth_proj.weight`, `read_length_proj.weight` and the two `fusion` Linear
    weights are DECAYED. They sit in the conditioning pathway, but they are matmuls, and h51's
    pre-registered filter lists only embeddings / film_proj / norms / biases. This is the
    GPT-3 / LLaMA convention the node cites.
  * the encoder's per-conv FiLM projections (`signal_tower.per_conv_film_layers.*.proj`) are
    DECAYED for the same reason: h51 names `film_proj`, which is the decoder module.
Both are one-line flips (`_FILM_PROJ_ATTRS`, or a conditioning-module rule) if the PI wants the wider
reading — but they must be flipped deliberately, not by accident.
"""
from __future__ import annotations

from typing import Dict, List, NamedTuple, Sequence, Tuple

import torch
import torch.nn as nn

from candi.decoder import LaneNorm, PerLaneFiLM
from candi.encoder import MaskTokenInjector, MetadataEmbedding, RMSNormSeq

__all__ = [
    "NO_DECAY",
    "DECAY",
    "ParamClassificationError",
    "ParamPartition",
    "classify_parameters",
    "build_param_groups",
    "build_adamw",
    "validate_optimizer_config",
    "partition_banner",
]

NO_DECAY = "no_decay"
DECAY = "decay"


class ParamClassificationError(ValueError):
    """A parameter or module the partition does not positively recognize.

    Raised INSTEAD OF defaulting into a group. If you are reading this from a traceback: decide where
    the named parameter belongs and register it below — do not add an `else` branch.
    """


# ---------------------------------------------------------------------------
# recognized module types
# ---------------------------------------------------------------------------
# Normalization modules, BY TYPE. `_BatchNorm` / `_InstanceNorm` are the private bases so every
# 1d/2d/3d variant and SyncBatchNorm are covered by one entry.
_NORM_TYPES: List[type] = [
    nn.LayerNorm,
    nn.GroupNorm,
    nn.modules.batchnorm._BatchNorm,
    nn.modules.instancenorm._InstanceNorm,
    RMSNormSeq,                       # this repo's own, encoder.py (affine is `.weight`)
    LaneNorm,                         # the decoder's per-assay normaliser (affine is `.weight`/`.bias`)
]
if hasattr(nn, "RMSNorm"):            # torch >= 2.4
    _NORM_TYPES.append(nn.RMSNorm)

# x-transformers ships its own norm classes; the one this model actually builds calls its affine
# `.gamma`. Registered by TYPE, looked up defensively so a version bump that renames one of them
# produces a loud ParamClassificationError rather than a silent misclassification.
try:                                                        # pragma: no cover - import shape only
    import x_transformers.x_transformers as _xt

    for _name in ("LayerNorm", "RMSNorm", "SimpleRMSNorm", "ScaleNorm", "MultiheadRMSNorm",
                  "AdaptiveLayerNorm", "AdaptiveRMSNorm"):
        _cls = getattr(_xt, _name, None)
        if isinstance(_cls, type) and issubclass(_cls, nn.Module):
            _NORM_TYPES.append(_cls)
except ImportError:                                         # pragma: no cover
    pass

NORM_TYPES: Tuple[type, ...] = tuple(_NORM_TYPES)

# Embedding TABLES, resolved as modules (never as a `*_embedding` substring).
EMBEDDING_TYPES: Tuple[type, ...] = (nn.Embedding, nn.EmbeddingBag)

# Modules whose `weight` is a trunk matmul/kernel and whose `bias` is a bias.
WEIGHT_AND_BIAS_TYPES: Tuple[type, ...] = (
    nn.Linear, nn.Bilinear,
    nn.Conv1d, nn.Conv2d, nn.Conv3d,
    nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
)

# Modules that hold BARE `nn.Parameter`s. Each entry declares every direct parameter it may own; a
# parameter name absent from the mapping raises, which is how a newly added Parameter gets noticed.
_RAW_PARAM_RULES: Sequence[Tuple[type, Dict[str, str]]] = (
    (MetadataEmbedding, {
        "depth_missing_emb": NO_DECAY,
        "depth_cloze_emb": NO_DECAY,
        "readlen_missing_emb": NO_DECAY,
        "readlen_cloze_emb": NO_DECAY,
    }),
    (MaskTokenInjector, {"mask_embedding": NO_DECAY}),
)

# Attribute names under which a FiLM projection hangs. Resolved as `getattr(parent, attr)` and matched
# by `id(module)`, so a parameter is in this set because of WHICH MODULE OWNS IT, not its name.
_FILM_PROJ_ATTRS: Tuple[str, ...] = ("film_proj",)

# Resolved by TYPE as well as by attribute name, because the decoder module h51 named `film_proj`
# is now a ModuleList of `PerLaneFiLM`, whose projections are called `film_layers.N.proj`. The
# attribute rule matched the old name only, so after the decoder rebuild the very module h51
# pre-registered as conditioning was falling through to "Linear weight -> decay". Invisible while
# `--weight-decay` stays at its 0.0 default, wrong the moment it does not.
#
# The ENCODER's `FiLMLayer` is deliberately NOT here. h51 named the decoder module, and the encoder
# tower's projections have always been decayed — see the note at the top of this file. Moving them
# would change a pre-registered split, which is a decision, not a refactor.
_FILM_TYPES: Tuple[type, ...] = (PerLaneFiLM,)


class ParamPartition(NamedTuple):
    """The two groups, plus why each parameter landed where it did."""

    no_decay: List[Tuple[str, nn.Parameter]]
    decay: List[Tuple[str, nn.Parameter]]
    reasons: Dict[str, str]

    @property
    def no_decay_names(self) -> set:
        return {n for n, _ in self.no_decay}

    @property
    def decay_names(self) -> set:
        return {n for n, _ in self.decay}

    def group_of(self, name: str) -> str:
        if name in self.no_decay_names:
            return NO_DECAY
        if name in self.decay_names:
            return DECAY
        raise KeyError(name)

    def summary(self) -> Dict[str, int]:
        return {
            "no_decay_tensors": len(self.no_decay),
            "no_decay_params": sum(p.numel() for _, p in self.no_decay),
            "decay_tensors": len(self.decay),
            "decay_params": sum(p.numel() for _, p in self.decay),
        }


def _film_proj_module_ids(model: nn.Module) -> Dict[int, str]:
    """`id(module) -> path` for every FiLM projection, resolved as a MODULE attribute."""
    found: Dict[int, str] = {}
    for mod_name, module in model.named_modules():
        for attr in _FILM_PROJ_ATTRS:
            child = getattr(module, attr, None)
            if isinstance(child, nn.Module):
                found[id(child)] = f"{mod_name}.{attr}" if mod_name else attr
        if isinstance(module, _FILM_TYPES):
            # the FiLM wrapper AND every module it owns: all of it is conditioning
            found[id(module)] = mod_name or "<root>"
            for sub_name, sub in module.named_modules():
                if sub is not module:
                    found[id(sub)] = f"{mod_name}.{sub_name}" if mod_name else sub_name
    return found


def _classify_one(module: nn.Module, param_name: str, full_name: str,
                  film_ids: Dict[int, str]) -> Tuple[str, str]:
    """Return `(group, reason)` for one DIRECT parameter, or raise.

    Order matters only for `film_proj`, which is a `nn.Linear` and would otherwise be classified as a
    trunk matmul. Everything else is disjoint by construction.
    """
    cls = type(module).__name__

    # 1. film_proj — by module IDENTITY. Both its weight and its bias are conditioning.
    if id(module) in film_ids:
        return NO_DECAY, f"film_proj({film_ids[id(module)]})"

    # 2. normalization affines — BY TYPE, never by name. This is the rule that makes the partition
    #    trustworthy: `.weight` on a LayerNorm and `.weight` on a Linear are indistinguishable by name.
    if isinstance(module, NORM_TYPES):
        # qualified, because torch's LayerNorm and x-transformers' LayerNorm share a class NAME and
        # differ in their affine's parameter name — the `--full` dump has to tell them apart.
        return NO_DECAY, f"norm:{type(module).__module__.split('.')[0]}.{cls}"

    # 3. embedding tables — resolved as modules.
    if isinstance(module, EMBEDDING_TYPES):
        return NO_DECAY, f"embedding:{cls}"

    # 4. bare nn.Parameter holders — an explicit (class, parameter name) allowlist.
    for owner_cls, rules in _RAW_PARAM_RULES:
        if isinstance(module, owner_cls):
            if param_name in rules:
                return rules[param_name], f"raw:{cls}.{param_name}"
            raise ParamClassificationError(
                f"unclassified parameter {full_name!r}: {cls} declares "
                f"{sorted(rules)} but owns a parameter named {param_name!r}. Decide which group it "
                f"belongs to and register it in _RAW_PARAM_RULES — do NOT let it default.")

    # 5. trunk matmuls / conv kernels.
    if isinstance(module, WEIGHT_AND_BIAS_TYPES):
        if param_name == "bias":
            return NO_DECAY, f"bias:{cls}"
        if param_name == "weight":
            return DECAY, f"weight:{cls}"
        raise ParamClassificationError(
            f"unclassified parameter {full_name!r}: {cls} is a recognized weight/bias module but "
            f"owns a parameter named {param_name!r}, which the partition does not know about.")

    raise ParamClassificationError(
        f"unclassified parameter {full_name!r}: module type "
        f"{type(module).__module__}.{cls} is not recognized by candi.param_groups. Register it "
        f"as a norm / embedding / weight-and-bias module, or add its parameters to "
        f"_RAW_PARAM_RULES. There is deliberately no default group.")


def classify_parameters(model: nn.Module) -> ParamPartition:
    """Split `model.named_parameters()` into exactly two disjoint, exhaustive groups.

    Raises `ParamClassificationError` on anything unrecognized, and `AssertionError` if the resulting
    groups are not a partition (both checks run every time the optimizer is built, not only in tests).
    """
    film_ids = _film_proj_module_ids(model)
    groups: Dict[str, List[Tuple[str, nn.Parameter]]] = {NO_DECAY: [], DECAY: []}
    reasons: Dict[str, str] = {}
    assigned: Dict[int, Tuple[str, str]] = {}          # id(p) -> (group, first name seen)

    for mod_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            full = f"{mod_name}.{param_name}" if mod_name else param_name
            group, reason = _classify_one(module, param_name, full, film_ids)
            prev = assigned.get(id(param))
            if prev is not None:
                # A shared/tied tensor reached through a second module. Counting it twice would break
                # exhaustiveness; letting the two sites disagree would put it in both groups.
                if prev[0] != group:
                    raise ParamClassificationError(
                        f"tied parameter reached as both {prev[1]!r} ({prev[0]}) and {full!r} "
                        f"({group}); one tensor cannot be in two decay groups.")
                continue
            assigned[id(param)] = (group, full)
            groups[group].append((full, param))
            reasons[full] = reason

    part = ParamPartition(no_decay=groups[NO_DECAY], decay=groups[DECAY], reasons=reasons)
    _assert_is_partition(model, part)
    return part


def _assert_is_partition(model: nn.Module, part: ParamPartition) -> None:
    """Exhaustive AND disjoint, by tensor identity, tensor count and element count."""
    nd_ids = {id(p) for _, p in part.no_decay}
    d_ids = {id(p) for _, p in part.decay}
    both = nd_ids & d_ids
    assert not both, f"{len(both)} parameter(s) landed in BOTH groups"

    all_named = list(model.named_parameters())
    all_ids = {id(p) for _, p in all_named}
    covered = nd_ids | d_ids

    missing = [n for n, p in all_named if id(p) not in covered]
    assert not missing, f"{len(missing)} parameter(s) in no group: {missing[:8]}"
    extra = covered - all_ids
    assert not extra, f"{len(extra)} grouped tensor(s) are not in model.named_parameters()"

    n_group = len(part.no_decay) + len(part.decay)
    assert n_group == len(all_named), (
        f"tensor count mismatch: {n_group} grouped vs {len(all_named)} named_parameters()")

    numel_group = (sum(p.numel() for _, p in part.no_decay)
                   + sum(p.numel() for _, p in part.decay))
    numel_all = sum(p.numel() for _, p in all_named)
    assert numel_group == numel_all, (
        f"element count mismatch: {numel_group:,} grouped vs {numel_all:,} in named_parameters()")


# ---------------------------------------------------------------------------
# optimizer construction
# ---------------------------------------------------------------------------

def validate_optimizer_config(optimizer: str, weight_decay: float, trunk_wd: float) -> str:
    """Normalize + reject the configurations whose meaning would be ambiguous.

    `--optimizer adamw` with a non-zero `--weight-decay` is REJECTED rather than merged: the two would
    double-specify decay (a global coupled term plus a per-group decoupled one) and the arms would
    stop meaning what h51's table says they mean. `--trunk-wd` is the only decay knob in the adamw
    path, and a non-zero `--trunk-wd` on the adam path is rejected too — it would be silently inert
    while still being written into the run config, which is a run that lies about itself.
    """
    name = str(optimizer).lower()
    if name not in ("adam", "adamw"):
        raise ValueError(f"--optimizer must be 'adam' or 'adamw'; got {optimizer!r}")
    wd, twd = float(weight_decay), float(trunk_wd)
    if twd < 0.0:
        raise ValueError(f"--trunk-wd must be >= 0; got {twd}")
    if name == "adamw" and wd != 0.0:
        raise ValueError(
            f"--optimizer adamw with --weight-decay {wd} double-specifies weight decay. In the adamw "
            "path --trunk-wd is the ONLY decay knob (the no-decay group is pinned at 0.0 and the "
            "decay group takes --trunk-wd); a non-zero --weight-decay on top would add a second, "
            "coupled term and the arms would no longer differ in decay alone. Drop --weight-decay "
            "(or set it to 0) and put the value in --trunk-wd.")
    if name == "adam" and twd != 0.0:
        raise ValueError(
            f"--optimizer adam with --trunk-wd {twd} is inert: the adam path is the frozen single-group "
            "line and never reads --trunk-wd, so the run would record a decay it never applied. Use "
            "--optimizer adamw for the split, or --weight-decay for a global coupled L2.")
    return name


def build_param_groups(model: nn.Module, *, trunk_wd: float) -> Tuple[List[dict], ParamPartition]:
    """`[no-decay group @ 0.0, decay group @ trunk_wd]` + the partition that produced them."""
    part = classify_parameters(model)
    groups = [
        {"name": NO_DECAY, "params": [p for _, p in part.no_decay], "weight_decay": 0.0},
        {"name": DECAY, "params": [p for _, p in part.decay], "weight_decay": float(trunk_wd)},
    ]
    return groups, part


def partition_banner(part: ParamPartition, *, optimizer: str, trunk_wd: float) -> str:
    s = part.summary()
    return (f"[train] optimizer={optimizer} trunk_wd={float(trunk_wd)} | "
            f"no_decay: {s['no_decay_tensors']} tensors / {s['no_decay_params']:,} params @ wd=0.0 | "
            f"decay: {s['decay_tensors']} tensors / {s['decay_params']:,} params @ wd={float(trunk_wd)}")


def build_adamw(model: nn.Module, *, lr: float, weight_decay: float = 0.0, trunk_wd: float = 0.0,
                verbose: bool = True) -> torch.optim.AdamW:
    """`torch.optim.AdamW` over the two groups. At `trunk_wd=0.0` this is the h51 CONTROL arm.

    The control is numerically the current default: at `weight_decay=0`, AdamW's decoupled shrink is
    `param.mul_(1 - lr*0)` — a multiply by exactly 1.0 — and Adam's coupled term is skipped, so the
    two produce identical updates. `tests/test_param_partition.py` proves that rather than citing it.
    """
    validate_optimizer_config("adamw", weight_decay, trunk_wd)
    groups, part = build_param_groups(model, trunk_wd=trunk_wd)
    opt = torch.optim.AdamW(groups, lr=lr)
    if verbose:
        print(partition_banner(part, optimizer="adamw", trunk_wd=trunk_wd), flush=True)
    return opt


# ---------------------------------------------------------------------------
# CLI: dump the partition for a given panel size so it can be eyeballed once
# ---------------------------------------------------------------------------

def main() -> None:                                          # pragma: no cover - operator tool
    import argparse

    from candi.model import build_model

    ap = argparse.ArgumentParser(description="dump the h51 parameter partition")
    ap.add_argument("--num-assays", type=int, default=35)
    ap.add_argument("--context-length", type=int, default=1200)
    ap.add_argument("--full", action="store_true", help="list every parameter, not just the summary")
    a = ap.parse_args()

    torch.manual_seed(0)
    model = build_model(num_assays=a.num_assays, context_length=a.context_length)
    part = classify_parameters(model)
    if a.full:
        for group, rows in ((NO_DECAY, part.no_decay), (DECAY, part.decay)):
            print(f"--- {group} ({len(rows)} tensors) ---")
            for name, p in rows:
                print(f"  {name:<62} {str(tuple(p.shape)):<20} {part.reasons[name]}")
    s = part.summary()
    print(f"num_assays={a.num_assays} context_length={a.context_length}")
    print(f"  no_decay: {s['no_decay_tensors']} tensors / {s['no_decay_params']:,} params")
    print(f"  decay   : {s['decay_tensors']} tensors / {s['decay_params']:,} params")
    print(f"  total   : {s['no_decay_tensors'] + s['decay_tensors']} tensors / "
          f"{s['no_decay_params'] + s['decay_params']:,} params")


if __name__ == "__main__":                                   # pragma: no cover
    main()
