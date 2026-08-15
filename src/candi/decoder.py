"""The CANDI decoder — grouped deconv trunk, per-assay per-layer FiLM, NB head.

The decoder mirrors the encoder's conv tower: every transposed convolution is grouped by assay, so
no channel of assay `a` ever reaches assay `b`, and conditioning is applied *while* each assay's
profile is forming rather than once at the end.

    z            [B, 96, d_model]
      input_proj                        -> [B, 96, A, lane]
      FiLM (0)
      deconv x2, groups=A, + FiLM (1)   -> [B, 192, A, lane]
      deconv x2, groups=A, + FiLM (2)   -> [B, 384, A, lane]
      deconv x2, groups=A, + FiLM (3)   -> [B, 768, A, lane]
      weight-shared head lane -> 1      -> eta, raw_n   [B, 768, A]

WHY FOUR FiLM TAPS AND NOT ONE AT THE END
-----------------------------------------
FiLM is a POSITION-CONSTANT affine: gamma and beta do not vary along the 768 positions. Applied
after the deconv it can raise, lower or rescale a finished profile but it cannot move a peak, narrow
a peak, or turn a sharp ATAC profile into a broad H3K36me3 domain. Assay-specific SHAPE forms INSIDE
the deconv, so the conditioning has to be there while it forms. Widening the lane does not fix that
— C=16 was never too few channels, the affine simply arrived too late.

Tap 0 (right after `input_proj`) is NOT redundant with tap 1: `input_proj` is a dense `Linear`, so
every output sees every input and the lanes are a labelling with no assay identity until FiLM gives
them one. The first grouped deconv would otherwise run on anonymous lanes.

WHY CONSTANT LANE WIDTH RATHER THAN AN 8 -> 4 -> 2 -> 1 MIRROR
--------------------------------------------------------------
The exact mirror shrinks conditioning capacity precisely as resolution grows: at 8->4->2->1 the last
FiLM tap modulates a single channel per assay, so target metadata collapses to one scalar gain at
full resolution, and the head degenerates to `Linear(1, 1)`. Once the trunk is grouped, width is
nearly free, so the lane is held constant and both FiLM and the head keep something to act on.
This is a reasoned choice, not a measured one — no run has yet varied it.

`LaneNorm`, `PerLaneFiLM` and `_init_grouped_weight` live here rather than in a separate module
because the decoder is now the only consumer of the per-lane primitives.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from candi._vendored import CLOZE, MISSING
from candi.encoder import MetadataEmbedding

__all__ = ["LANE_NORMS", "LaneNorm", "PerLaneFiLM", "LaneDeconvBlock", "SymmetricDecoder"]


LANE_NORMS = ("lane", "group")


class LaneNorm(nn.Module):
    """Per-assay normalisation on `[B, L, T, C]`. Neither mode moves information across lanes.

    TWO SEPARATE PROBLEMS LIVE IN THE WORD "LAYERNORM" HERE, AND ONLY ONE OF THEM IS THE LEAK.

    `mode="lane"` normalises the C channels of one lane AT EACH POSITION. It cannot leak — the lane's
    channels are the whole population — but it fixes each lane's energy at every position to 1, so a
    peak bin and a background bin survive only as a PATTERN across C, not as a difference in
    amplitude. Amplitude is the thing being predicted.

    `mode="group"` normalises the C channels of one lane ACROSS ALL L POSITIONS — the `nn.GroupNorm`
    statistic, at the granularity of one assay. It removes the lane's overall location and scale and
    leaves the profile along the genome intact. Depth is already carried by the decoder's log-link
    offset, so removing a lane's global scale costs nothing the model needs.

    Affine is per-CHANNEL in both modes, so each assay keeps an independent scale and shift; only the
    pooling of the statistic changes.

    WHY THIS IS NOT DECIDABLE FROM THEORY, and is therefore an arm rather than a fix: `ConvTower` is
    `gelu(norm(conv(x)) + rconv(x))`, and the residual branch is an UN-NORMALISED 1x1, so amplitude
    reaches the next layer under either mode. The argument above says `group` should be better; the
    residual says the gap may be small. Measure it.
    """

    def __init__(self, lane_width: int, mode: str = "lane", eps: float = 1e-5) -> None:
        super().__init__()
        if mode not in LANE_NORMS:
            raise ValueError(f"lane_norm must be one of {LANE_NORMS}; got {mode!r}")
        self.C = int(lane_width)
        self.mode = str(mode)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.C))
        self.bias = nn.Parameter(torch.zeros(self.C))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.mode == "lane":
            dims = (-1,)                 # C only, at each position
        else:
            dims = (1, -1)               # C and L, per lane  (z is [B, L, T, C])
        mean = z.mean(dim=dims, keepdim=True)
        var = z.var(dim=dims, keepdim=True, unbiased=False)
        return (z - mean) * torch.rsqrt(var + self.eps) * self.weight + self.bias


def _init_grouped_weight(w: torch.Tensor, fan_in: int, fan_out: int) -> None:
    """Xavier-uniform a `[T, fan_in, fan_out]` stack as T independent `Linear(fan_in, fan_out)`."""
    bound = math.sqrt(6.0 / (fan_in + fan_out))
    nn.init.uniform_(w, -bound, bound)


class PerLaneFiLM(nn.Module):
    """adaLN-zero FiLM on `[B, L, T, C]`, one (gamma, beta) per track from that track's covariates.

    Shared projection weights, per-track inputs — the same construction `encoder.FiLMLayer` already
    uses in the conv tower. Zero-init so the arm starts as an exact no-op and steering grows.
    """

    def __init__(self, embed_dim: int, lane_width: int) -> None:
        super().__init__()
        self.C = int(lane_width)
        self.proj = nn.Linear(int(embed_dim), 2 * self.C)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, z: torch.Tensor, meta_embed: torch.Tensor) -> torch.Tensor:
        if meta_embed.shape[1] != z.shape[2]:
            raise ValueError(f"PerLaneFiLM track mismatch: z has {z.shape[2]} lanes, "
                             f"meta_embed has {meta_embed.shape[1]}")
        gamma, beta = self.proj(meta_embed).chunk(2, dim=-1)      # [B, T, C] each
        return z * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)




class LaneDeconvBlock(nn.Module):
    """Grouped transposed conv + 1x1 grouped residual + per-lane LayerNorm — mirror of `ConvTower`.

    `groups=num_assays`, so no channel of assay a ever reaches assay b. The per-lane LayerNorm
    normalises within a lane only, for the same reason.
    """

    def __init__(self, num_assays: int, lane: int, kernel_size: int = 3, upsample: int = 2,
                 lane_norm: str = "lane") -> None:
        super().__init__()
        self.A = int(num_assays)
        self.lane = int(lane)
        ch = self.A * self.lane
        pad = (int(kernel_size) - 1) // 2
        self.deconv = nn.ConvTranspose1d(ch, ch, int(kernel_size), stride=int(upsample),
                                         padding=pad, output_padding=int(upsample) - 1,
                                         groups=self.A)
        self.rdeconv = nn.ConvTranspose1d(ch, ch, 1, stride=int(upsample),
                                          output_padding=int(upsample) - 1, groups=self.A)
        self.norm = LaneNorm(self.lane, lane_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`[B, L, A, lane] -> [B, L*upsample, A, lane]`."""
        B, L, A, C = x.shape
        h = x.reshape(B, L, A * C).transpose(1, 2)          # [B, A*lane, L]
        y = self.deconv(h)
        Lo = y.shape[-1]
        # norm in lane space, before the residual add — mirrors ConvBlock(norm, apply_act=False)
        y = self.norm(y.transpose(1, 2).reshape(B, Lo, A, C))
        r = self.rdeconv(h).transpose(1, 2).reshape(B, Lo, A, C)
        return F.gelu(y + r)


class SymmetricDecoder(nn.Module):
    """Grouped deconv trunk + per-assay per-layer FiLM + depth-offset log-linked NB head.

    Accepts either a dense latent `[B, L2, d]` (set `in_dense_dim`, used by the non-lane arms) or a
    lane latent `[B, L2, T, C]` (set `in_lane_width`). On the lane path the input projection is
    per-lane, so `input_proj` stops being a cross-assay mixer; on the dense path it still is, and
    tap 0 is what re-establishes assay identity afterwards.

    Head arithmetic is unchanged from `candi_kit.model.DualCondDecoder` so the objective, the depth
    offset and the h74 reference offset are all comparable across arms:
        log2_mu = (d - depth_center) + eta   [offset ON and depth not a sentinel]
        log2_mu = eta                        [otherwise]
        log2_mu += log_ref                   [when a reference is supplied]
        mu = 2^clamp;  n = softplus(raw_n) + eps;  p = n / (n + mu)
    """

    def __init__(self, *, num_assays: int, lane: int = 8, meta_embed_dim: int = 32,
                 in_dense_dim: Optional[int] = None, in_lane_width: Optional[int] = None,
                 n_deconv_layers: int = 3, upsample: int = 2, conv_kernel_size: int = 3,
                 use_offset: bool = True, depth_center: float = 25.1, mu_eps: float = 1e-6,
                 log2_mu_clamp: tuple = (-15.0, 30.0), use_layernorm: bool = True,
                 depth_row: int = 0, num_cells: int = 0, meta_gain: float = 1.0,
                 lane_norm: str = "lane") -> None:
        super().__init__()
        if (in_dense_dim is None) == (in_lane_width is None):
            raise ValueError("give exactly one of in_dense_dim (dense latent) or in_lane_width "
                             "(lane latent)")
        self.A = int(num_assays)
        self.lane = int(lane)
        self.use_offset = bool(use_offset)
        self.depth_center = float(depth_center)
        self.eps = float(mu_eps)
        self.clamp_lo, self.clamp_hi = float(log2_mu_clamp[0]), float(log2_mu_clamp[1])
        self.depth_row = int(depth_row)
        self.meta_gain = float(meta_gain)
        if not (self.meta_gain > 0.0) or not math.isfinite(self.meta_gain):
            raise ValueError(f"meta_gain must be finite and > 0; got {meta_gain!r}")

        self.dense_in = in_dense_dim is not None
        if self.dense_in:
            self.input_proj = nn.Linear(int(in_dense_dim), self.A * self.lane)
        else:
            self.w_in = nn.Parameter(torch.empty(self.A, int(in_lane_width), self.lane))
            self.b_in = nn.Parameter(torch.zeros(self.A, self.lane))
            _init_grouped_weight(self.w_in, int(in_lane_width), self.lane)

        self.blocks = nn.ModuleList([
            LaneDeconvBlock(self.A, self.lane, int(conv_kernel_size), int(upsample), lane_norm)
            for _ in range(int(n_deconv_layers))
        ])
        # one tap after input_proj + one after each deconv
        self.film_layers = nn.ModuleList([
            PerLaneFiLM(int(meta_embed_dim), self.lane)
            for _ in range(int(n_deconv_layers) + 1)
        ])
        self.head_eta = nn.Sequential(nn.Linear(self.lane, self.lane), nn.GELU(),
                                      nn.Linear(self.lane, 1))
        self.head_n = nn.Sequential(nn.Linear(self.lane, self.lane), nn.GELU(),
                                    nn.Linear(self.lane, 1))
        self.meta_embedding = MetadataEmbedding(
            num_assays=self.A, embed_dim=int(meta_embed_dim),
            use_layernorm=bool(use_layernorm), num_cells=int(num_cells))

    def forward(self, z: torch.Tensor, y_meta: torch.Tensor,
                log_ref: Optional[torch.Tensor] = None,
                memb: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # `memb` lets the split-conditioning arms (a3/a5) share ONE y_meta embedding between the
        # decoder-side axial blocks and this trunk, so the two are conditioned on the same vector
        # and the covariate gradient has a single path to attribute. None = embed here, which is
        # what the non-lane arms do and what keeps them comparable to candi_kit.
        if memb is None:
            memb = self.meta_embedding(y_meta.float())          # [B, A, E]
        if self.meta_gain != 1.0:
            memb = memb * self.meta_gain

        if self.dense_in:
            B, L, _ = z.shape
            feat = self.input_proj(z).view(B, L, self.A, self.lane)
        else:
            # lane latent may still carry the control track; the decoder predicts signal assays only,
            # so drop it by SLICING — never by a dense re-projection, which would re-mix the lanes.
            z = z[:, :, :self.A, :]
            feat = torch.einsum("bltc,tcd->bltd", z, self.w_in) + self.b_in

        feat = self.film_layers[0](feat, memb)
        for i, block in enumerate(self.blocks):
            feat = block(feat)
            feat = self.film_layers[i + 1](feat, memb)

        eta = self.head_eta(feat).squeeze(-1)                    # [B, L, A]
        raw_n = self.head_n(feat).squeeze(-1)

        depth = y_meta[:, self.depth_row, :]                     # [B, A]
        valid = (depth != MISSING) & (depth != CLOZE)
        if self.use_offset:
            d_off = (depth - self.depth_center).unsqueeze(1)     # [B, 1, A]
            log2_mu = torch.where(valid.unsqueeze(1), d_off + eta, eta)
        else:
            log2_mu = eta
        if log_ref is not None:
            log2_mu = log2_mu + log_ref
        log2_mu = log2_mu.clamp(self.clamp_lo, self.clamp_hi)
        mu = torch.pow(2.0, log2_mu).clamp_min(self.eps)
        n = F.softplus(raw_n) + self.eps
        p = (n / (n + mu)).clamp(self.eps, 1.0 - self.eps)
        return dict(p=p, n=n, eta=eta, log2_mu=log2_mu, mu=mu)
