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

from typing import Dict

import torch
import torch.nn as nn

from candi.config import EncoderConfig
from candi.decoder import SymmetricDecoder
from candi.encoder import MetadataEmbedding, V2Encoder

__all__ = ["CandiModel", "build_model", "forward_full", "encode_latent", "nb_mean", "nb_nll"]


class CandiModel(nn.Module):
    """`V2Encoder` (grouped signal tower, per-conv FiLM, arcsinh, 4-row metadata) + `SymmetricDecoder`."""

    def __init__(self, *, embed_dim: int = 32, dropout: float = 0.1, decoder_lane: int = 8,
                 depth_center: float = 25.1, use_offset: bool = True, num_assays: int = 35,
                 context_length: int = 768, d_model: int = 0, nhead: int = 4,
                 n_transformer_layers: int = 2, num_cells: int = 0,
                 meta_embed_layernorm: bool = True, meta_gain: float = 1.0,
                 lane_norm: str = "lane") -> None:
        super().__init__()
        cfg = EncoderConfig(
            num_assays=num_assays, context_length=context_length, metadata_embed_dim=embed_dim,
            signal_transform="arcsinh", missing_data_mode="mask_token", dropout=dropout,
            n_transformer_layers=int(n_transformer_layers), d_model=d_model, nhead=nhead,
            film_mode="per_conv", num_cells=num_cells,
            meta_embed_layernorm=bool(meta_embed_layernorm))
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
            in_dense_dim=self.encoder.d_model, lane_norm=lane_norm,
            use_offset=use_offset, depth_center=depth_center,
            use_layernorm=bool(meta_embed_layernorm), depth_row=0, num_cells=num_cells,
            meta_gain=meta_gain)

    def forward(self, x_data, x_dna, x_meta, y_meta, log_ref=None) -> Dict[str, torch.Tensor]:
        z = self.encoder.encode(x_data, x_dna, x_meta, return_meta=False)
        return self.decoder(z, y_meta, log_ref)

    def encode(self, x_data, x_dna, x_meta) -> torch.Tensor:
        """Encoder latent `[B, L2, d_model]`, for M3's invariance readout."""
        return self.encoder.encode(x_data, x_dna, x_meta, return_meta=False)

    def decode_latent(self, z, y_meta, log_ref=None) -> Dict[str, torch.Tensor]:
        """Decode a latent produced by `encode()` — see `eval.decode_latent` for why this exists."""
        return self.decoder(z, y_meta, log_ref)


def build_model(**kw) -> CandiModel:
    """The one entry point. Kept as a function so callers never construct the class directly."""
    return CandiModel(**kw)


# ---------------------------------------------------------------------------
# Batch-level helpers shared by train, eval and healthcheck
# ---------------------------------------------------------------------------

def forward_full(model: CandiModel, batch: dict) -> Dict[str, torch.Tensor]:
    """Full head output dict (p, n, eta, log2_mu, mu) — used by the distributional M2 readout.

    `log_ref` rides in the prep dict rather than as an argument so every existing caller keeps
    working: a batch without the key is the pre-reference model, exactly.
    """
    return model(batch["x_data"], batch["x_dna"], batch["x_meta"], batch["y_meta"],
                 batch.get("log_ref"))


def encode_latent(model: CandiModel, batch: dict) -> torch.Tensor:
    """Encoder latent z [B, L2, d_model] (for M3); depends only on x_data/x_dna/x_meta."""
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
