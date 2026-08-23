"""CANDI — the model.

One architecture, not a registry. The encoder is a grouped Conv1D tower over the assay tracks and a
dense Conv1D tower over DNA, fused linearly and passed through a RoPE transformer; the decoder is a
grouped deconv mirror of the signal tower with per-assay, per-layer FiLM and a depth-offset,
log-linked Negative Binomial head.

    x_data  [B, L, A+1]      counts, A assays + 1 control        (CLOZE = -2, MISSING = -1)
    x_dna   [B, 4, L*res]    one-hot DNA
    x_meta  [B, 4, A+1]      (log2_depth, assay_id, read_length, run_type) for the INPUT tracks
    y_meta  [B, 4, A]        the same 4 rows for the TARGET tracks
      -> {p, n, eta, log2_mu, mu}, each [B, L, A]
      -> + {signal_mu, signal_var} and/or {peak_logit} when `heads` names them (default: count only)

WHY THE DECODER IS GROUPED
--------------------------
The decoder this replaces ran an UNGROUPED deconv trunk at `num_assays * feat_per_assay * 2**3`
= 4480 channels, against the production decoder's `num_assays * 2**3` = 280. The 16x width was a
vendoring accident, and it put 93.7% of the model's parameters in a dense conv tower. Mirroring the
encoder instead — grouped by assay, constant lane width — drops the model from 56.2 M parameters to
2.35 M and improves both imputation and denoising.

`obs` and `imp` everywhere in this package mean UNMASKED and MASKED positions. They do not mean
biologically observed and imputed.
"""
from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn

from candi.config import EncoderConfig
from candi.decoder import DECODER_FILM_TAPS, DEFAULT_HEADS, HEADS, SymmetricDecoder, parse_heads
from candi.encoder import MetadataEmbedding, V2Encoder

__all__ = ["ALL_FILM_TAPS", "DEFAULT_FILM_TAPS", "DEFAULT_HEADS", "ENCODER_FILM_TAPS", "HEADS",
           "CandiModel", "build_model", "parse_film_taps", "parse_heads", "film_mode_from_taps",
           "forward_full", "encode_latent", "nb_mean", "nb_nll"]


# ---------------------------------------------------------------------------
# FiLM taps — one flag naming every place conditioning may enter, both towers
# ---------------------------------------------------------------------------
# The encoder's `film_mode` is an enum of mutually exclusive placements and the decoder's taps are an
# independent set, which made "where is the conditioning?" a two-flag question with no single answer.
# `film_taps` is that one answer: a SET, validated as a whole, from which the encoder's mode is
# derived. Combinations the encoder cannot express are rejected by name rather than silently resolved
# to whichever branch happens to be checked first.

ENCODER_FILM_TAPS: Tuple[str, ...] = ("pre_conv", "per_conv", "post_conv", "per_transformer")
ALL_FILM_TAPS: Tuple[str, ...] = ENCODER_FILM_TAPS + DECODER_FILM_TAPS

# The shipped model: FiLM after every conv, after the input projection, and after every deconv.
# `post_conv`, `per_transformer` and `post_head` are available and off.
DEFAULT_FILM_TAPS: Tuple[str, ...] = ("per_conv", "pre_deconv", "per_deconv")

# At most one of these may be present — they are alternative placements of the SAME conv-tower FiLM,
# not additive taps, because that is how `SignalConvTower` is built.
_EXCLUSIVE_CONV_TAPS = ("pre_conv", "per_conv", "post_conv")


def parse_film_taps(taps) -> Tuple[str, ...]:
    """Normalise a comma string or iterable of tap names into a validated tuple.

    Raises on an unknown name, on two mutually exclusive conv placements, and on `per_transformer`
    without `per_conv` — the encoder has no transformer-FiLM-only mode, so asking for one would
    otherwise quietly give you per-conv FiLM as well.
    """
    if isinstance(taps, str):
        names = [t.strip() for t in taps.split(",") if t.strip()]
    else:
        names = [str(t).strip() for t in taps]
    unknown = [t for t in names if t not in ALL_FILM_TAPS]
    if unknown:
        raise ValueError(f"unknown film tap(s) {unknown}; valid taps are {ALL_FILM_TAPS}")
    chosen = [t for t in _EXCLUSIVE_CONV_TAPS if t in names]
    if len(chosen) > 1:
        raise ValueError(
            f"film_taps names {chosen}, but pre_conv / per_conv / post_conv are alternative "
            "placements of the one conv-tower FiLM, not additive taps; pick exactly one")
    if "per_transformer" in names and "per_conv" not in names:
        raise ValueError("film_taps 'per_transformer' requires 'per_conv': the encoder has no "
                         "transformer-only FiLM mode")
    # de-duplicated, and ordered as ALL_FILM_TAPS so two spellings of the same set compare equal
    return tuple(t for t in ALL_FILM_TAPS if t in names)


def film_mode_from_taps(taps: Sequence[str]) -> str:
    """The encoder's `film_mode` enum implied by a validated tap set."""
    if "per_transformer" in taps:
        return "per_conv_and_transformer"
    for tap, mode in (("per_conv", "per_conv"), ("pre_conv", "pre_conv"),
                      ("post_conv", "post_conv")):
        if tap in taps:
            return mode
    return "off"


class CandiModel(nn.Module):
    """`V2Encoder` (grouped signal tower, per-conv FiLM, arcsinh, 4-row metadata) + `SymmetricDecoder`."""

    def __init__(self, *, embed_dim: int = 32, dropout: float = 0.1, decoder_lane: int = 8,
                 depth_center: float = 25.1, use_offset: bool = True, num_assays: int = 35,
                 context_length: int = 768, d_model: int = 0, nhead: int = 4,
                 n_transformer_layers: int = 2, num_cells: int = 0,
                 meta_embed_layernorm: bool = True, meta_gain: float = 1.0,
                 deconv_norm: str = "lane",
                 # -- geometry ---------------------------------------------------------------
                 resolution: int = 25, n_cnn_layers: int = 3, conv_kernel_size: int = 3,
                 pool_size: int = 2, expansion_factor: int = 2,
                 n_deconv_layers: int = 3, deconv_upsample: int = 2, deconv_kernel_size: int = 3,
                 # -- normalisation ----------------------------------------------------------
                 conv_norm: str = "layer", transformer_norm: str = "layer",
                 transformer_norm_placement: str = "pre", attn_qk_norm: bool = False,
                 # -- conditioning -----------------------------------------------------------
                 film_taps: Sequence[str] = DEFAULT_FILM_TAPS,
                 film_init_encoder: str = "xavier", film_init_decoder: str = "zero",
                 # -- heads ------------------------------------------------------------------
                 head_sharing: str = "shared", head_hidden: int = 0,
                 heads: Sequence[str] = DEFAULT_HEADS) -> None:
        super().__init__()
        taps = parse_film_taps(film_taps)
        # Parsed here, next to the tap set and before the geometry guard, for the same reason: neither
        # consumes RNG, so validating both up front costs nothing and turns a typo into a message that
        # names the flag rather than a shape error deep inside a module.
        head_set = parse_heads(heads)
        # THE GEOMETRY GUARD. The decoder has to return the encoder's downsampling exactly, or the
        # output is a different length from the target and the loss indexes into the wrong bins. It
        # is checked here, before any module is built, because the alternative is a shape error deep
        # inside a deconv that names neither flag responsible.
        enc_factor = int(pool_size) ** int(n_cnn_layers)
        dec_factor = int(deconv_upsample) ** int(n_deconv_layers)
        if enc_factor != dec_factor:
            raise ValueError(
                f"encoder and decoder geometry disagree: pool_size**n_cnn_layers = "
                f"{pool_size}**{n_cnn_layers} = {enc_factor}, but deconv_upsample**n_deconv_layers "
                f"= {deconv_upsample}**{n_deconv_layers} = {dec_factor}. The decoder must undo the "
                "encoder's downsampling exactly.")
        # `dna_pool_size` is DERIVED, never passed: the DNA tower's two large pools must collapse
        # exactly `resolution` bp into one bin, so the pool is isqrt(resolution). A resolution that
        # is not a perfect square has no such tower and is refused rather than rounded.
        dna_pool = int(math.isqrt(int(resolution)))
        if dna_pool * dna_pool != int(resolution):
            raise ValueError(
                f"resolution={resolution} is not a perfect square, so the DNA tower's two large "
                "pooling steps cannot collapse it into one bin. The panel's resolution and the "
                "tower geometry are the same fact; change the panel, not this.")
        cfg = EncoderConfig(
            num_assays=num_assays, context_length=context_length, metadata_embed_dim=embed_dim,
            signal_transform="arcsinh", missing_data_mode="mask_token", dropout=dropout,
            n_transformer_layers=int(n_transformer_layers), d_model=d_model, nhead=nhead,
            film_mode=film_mode_from_taps(taps), num_cells=num_cells,
            meta_embed_layernorm=bool(meta_embed_layernorm),
            n_cnn_layers=int(n_cnn_layers), conv_kernel_size=int(conv_kernel_size),
            pool_size=int(pool_size), expansion_factor=int(expansion_factor),
            conv_norm=str(conv_norm), dna_pool_size=dna_pool,
            transformer_norm=str(transformer_norm),
            transformer_norm_placement=str(transformer_norm_placement),
            attn_qk_norm=bool(attn_qk_norm), film_init=str(film_init_encoder))
        self.encoder = V2Encoder(cfg)
        # CONSTRUCTION ORDER IS LOAD-BEARING, and this line is why. `V2Encoder` already builds a
        # `MetadataEmbedding` from the same cfg; replacing it with an identically-configured one is
        # functionally a no-op but consumes a block of RNG draws, so every module built afterwards —
        # the whole decoder — lands on different weights without it. Removing it is a legitimate
        # cleanup, but it re-samples the model and must be its own labelled change, not a side
        # effect of moving code between files.
        self.encoder.metadata_embedding = MetadataEmbedding(
            num_assays=num_assays, embed_dim=embed_dim,
            use_layernorm=bool(meta_embed_layernorm), num_cells=num_cells)
        self.decoder = SymmetricDecoder(
            num_assays=num_assays, lane=decoder_lane, meta_embed_dim=embed_dim,
            in_dense_dim=self.encoder.d_model, deconv_norm=deconv_norm,
            use_offset=use_offset, depth_center=depth_center,
            use_layernorm=bool(meta_embed_layernorm), depth_row=0, num_cells=num_cells,
            meta_gain=meta_gain,
            n_deconv_layers=int(n_deconv_layers), upsample=int(deconv_upsample),
            conv_kernel_size=int(deconv_kernel_size),
            film_taps=tuple(t for t in taps if t in DECODER_FILM_TAPS),
            film_init=str(film_init_decoder),
            head_sharing=str(head_sharing), head_hidden=int(head_hidden),
            heads=head_set)

    def forward(self, x_data, x_dna, x_meta, y_meta, log_ref=None) -> Dict[str, torch.Tensor]:
        z = self.encoder.encode(x_data, x_dna, x_meta, return_meta=False)
        return self.decoder(z, y_meta, log_ref)

    def encode(self, x_data, x_dna, x_meta) -> torch.Tensor:
        """Encoder latent `[B, L2, d_model]`, for `depthblind`'s invariance readout."""
        return self.encoder.encode(x_data, x_dna, x_meta, return_meta=False)

    def decode_latent(self, z, y_meta, log_ref=None) -> Dict[str, torch.Tensor]:
        """Decode a latent produced by `encode()` — see `eval.decode_latent` for why this exists."""
        return self.decoder(z, y_meta, log_ref)


def build_model(**kw) -> CandiModel:
    """The one entry point. Kept as a function so callers never construct the class directly."""
    return CandiModel(**kw)


def arch_keys() -> Tuple[str, ...]:
    """Every keyword that defines the architecture, read off the constructor so it cannot drift."""
    import inspect
    return tuple(p for p in inspect.signature(CandiModel.__init__).parameters if p != "self")


def build_model_from_arch(arch: dict) -> CandiModel:
    """Rebuild a model from a run JSON's `config.arch` block.

    THE POINT IS RE-SCORING. Every geometry and normalisation flag changes the state_dict, so a
    checkpoint can only be reloaded by a model built from the same arguments. Retyping them on the
    eval command line is how a checkpoint becomes unscorable six weeks later; reading them back from
    the run's own JSON is how it stays scorable forever.

    Unknown keys RAISE: an arch block written by a newer checkout names something this code cannot
    honour, and quietly dropping it would build a different model and then load the weights into it
    anyway. Missing keys take the current default, which is the safe direction — the run predates the
    flag, so the default is what it ran with.
    """
    known = set(arch_keys())
    unknown = sorted(set(arch) - known)
    if unknown:
        raise ValueError(
            f"config.arch names {unknown}, which this checkout's model does not accept. The "
            "checkpoint was written by a different version of candi; check it out to re-score.")
    return CandiModel(**arch)


# ---------------------------------------------------------------------------
# Batch-level helpers shared by train, eval and healthcheck
# ---------------------------------------------------------------------------

def forward_full(model: CandiModel, batch: dict) -> Dict[str, torch.Tensor]:
    """Full head output dict (p, n, eta, log2_mu, mu, + any optional head) — the covariate block
    reads it.

    `log_ref` rides in the prep dict rather than as an argument so every existing caller keeps
    working: a batch without the key is the pre-reference model, exactly.
    """
    return model(batch["x_data"], batch["x_dna"], batch["x_meta"], batch["y_meta"],
                 batch.get("log_ref"))


def encode_latent(model: CandiModel, batch: dict) -> torch.Tensor:
    """Encoder latent z [B, L2, d_model] (for `depthblind`); depends only on x_data/x_dna/x_meta."""
    return model.encode(batch["x_data"], batch["x_dna"], batch["x_meta"])


def nb_mean(p: torch.Tensor, n: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return n * (1.0 - p) / (p + eps)


def nb_nll(p: torch.Tensor, n: torch.Tensor, target: torch.Tensor, avail: torch.Tensor,
           eps: float = 1e-6) -> torch.Tensor:
    """Masked NB negative log-likelihood over available assays. target = integer counts [B,L,A]."""
    probs = (1.0 - p).clamp(eps, 1.0 - eps)
    total = n.clamp_min(eps)
    dist = torch.distributions.NegativeBinomial(total_count=total, probs=probs)
    ll = dist.log_prob(target.clamp_min(0.0))               # [B, L, A]
    m = avail.unsqueeze(1).expand_as(ll)                    # [B, L, A]
    denom = m.sum().clamp_min(1.0)
    return -(ll * m).sum() / denom
