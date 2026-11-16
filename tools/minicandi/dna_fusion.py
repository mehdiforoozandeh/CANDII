"""DNA fusion arms (close-out item A11): how the DNA features join the count features.

Config key `dna_fusion` (absent or null = the base join of the rung, bit for bit). The base joins:

  conv_ae, conv_attn_ae  early concatenation: the per-bin DNA features [B, L, 32] are concatenated
                         with the flattened per-track count features, and one linear map (the 1x1
                         stem conv, or the track-fusion Linear) projects them to the trunk width.
  unet                   late concatenation: a DNA tower of its own (the count tower's level
                         stack on its own weights) is concatenated with the count bottleneck and
                         a 1x1 conv projects it (`fuse`); the DNA tower also feeds the DNA skips.

Arms (one change from the base; the DNA source, the skips and the trunk stay):

  film   the DNA features scale and shift the count features per position, at the base join
         point: (gamma, beta) = Conv1d(d_dna, 2C, 1)(DNA), z * (1 + gamma) + beta, initialized to
         the identity. conv models: after the stem (the stem reads counts only). U-Net: the DNA
         bottleneck modulates the count bottleneck in place of the concatenation and `fuse`.
  late   conv models only: DNA joins at the bottleneck instead of the input. A learned patch
         pooling Conv1d(32, C_bottleneck, f, stride f), f = the downsampling factor, is added to
         the bottleneck features. The encoder sees counts only.
  xattn  conv + attention and U-Net: the count bottleneck tokens cross-attend to the DNA tokens of
         the same resolution (one residual pre-norm cross-attention layer; the same sinusoidal
         positions on queries and keys). conv + attention: the DNA tokens come from the patch
         pooling of `late`, and the layer sits before the self-attention. U-Net: the DNA tokens
         are the DNA tower's bottleneck; the layer replaces the concatenation and `fuse`.
  early  U-Net only: the DNA features (width C0) are concatenated into the count stem; the DNA
         tower, `fuse` and the DNA skips are gone, so every skip carries the joint features.

Not built, with the reason (`NOT_APPLICABLE`):
  addition, every rung: a linear map of a concatenation equals the sum of two linear maps, so
      "project and add" is the base's own function class; only the initialization differs.
  early on conv models and late on U-Net: that is the base join.
  xattn on the conv autoencoder: the model has no attention; one cross-attention layer would add
      the global context that defines the conv + attention model.

`dna = off` with `dna_fusion` set raises NotApplicable (no DNA features to join). Rungs 1, 5 and 6
do not read the key; `registry.build_model` refuses it there.
"""
from __future__ import annotations

import torch
from torch import nn

from minicandi.components import NotApplicable, sinusoidal_positions

FUSIONS = ("film", "late", "xattn", "early")
#: rung module -> the arms it builds
APPLIES = {"conv_ae": ("film", "late"), "conv_attn_ae": ("film", "late", "xattn"),
           "unet": ("film", "xattn", "early")}
_EARLY_BASE = "the base already joins DNA at the input (concatenation + linear projection)"
NOT_APPLICABLE = {
    ("conv_ae", "early"): _EARLY_BASE,
    ("conv_attn_ae", "early"): _EARLY_BASE,
    ("conv_ae", "xattn"): ("the conv autoencoder has no attention; one cross-attention layer "
                           "would add the global context that defines the conv + attention model"),
    ("unet", "late"): ("the base already joins DNA at the bottleneck (concatenation + 1x1 conv) "
                       "and through the DNA skips"),
}


def fusion_of(cfg: dict, name: str) -> str | None:
    """The `dna_fusion` arm of `cfg` for rung module `name`; None = the base join."""
    f = cfg.get("dna_fusion")
    if f is None:
        return None
    if f not in FUSIONS:
        raise ValueError(f"dna_fusion {f!r} not in {FUSIONS}")
    if name not in APPLIES:
        raise NotApplicable(f"dna_fusion is built in conv_ae, conv_attn_ae and unet only, not "
                            f"{name}")
    if f not in APPLIES[name]:
        raise NotApplicable(f"{name}: dna_fusion = {f}: {NOT_APPLICABLE[(name, f)]}")
    if cfg.get("dna", "own") == "off":
        raise NotApplicable("dna_fusion needs DNA features; dna = off has none")
    return f


class DnaFiLM(nn.Module):
    """z [B, C, L] * (1 + gamma) + beta, (gamma, beta) = Conv1d(d_dna, 2C, 1)(dna [B, d_dna, L]).
    Zero-initialized, so the layer starts as the identity on z."""

    def __init__(self, d_dna: int, c: int):
        super().__init__()
        self.proj = nn.Conv1d(d_dna, 2 * c, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, z: torch.Tensor, dna: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(dna.to(z.dtype)).chunk(2, dim=1)
        return z * (1 + gamma) + beta


class DnaPatchDown(nn.Module):
    """Per-bin DNA features [B, d_dna, L] -> [B, c, L / f]: Conv1d(d_dna, c, f, stride f), a
    learned pooling of each block of f bins (L must be a multiple of f)."""

    def __init__(self, d_dna: int, c: int, f: int):
        super().__init__()
        self.f = int(f)
        self.conv = nn.Conv1d(d_dna, c, self.f, stride=self.f)

    def forward(self, dna: torch.Tensor) -> torch.Tensor:
        if dna.shape[-1] % self.f:
            raise ValueError(f"DnaPatchDown: length {dna.shape[-1]} is not a multiple of {self.f}")
        return self.conv(dna)


class DnaCrossAttn(nn.Module):
    """q [B, L, d] + CrossAttention(LN(q) + pos, LN(kv) + pos, LN(kv)), kv [B, L, d]: one residual
    pre-norm layer; the same sinusoidal positions mark queries and keys, so a count token can find
    the DNA token of its own position."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.norm_q = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        if q.shape[1] != kv.shape[1]:
            raise ValueError(f"DnaCrossAttn: {q.shape[1]} queries but {kv.shape[1]} keys")
        pos = sinusoidal_positions(q.shape[1], q.shape[-1], q.device).to(q.dtype)
        k = self.norm_kv(kv.to(q.dtype))
        a, _ = self.attn(self.norm_q(q) + pos, k + pos, k, need_weights=False)
        return q + a


__all__ = ["APPLIES", "DnaCrossAttn", "DnaFiLM", "DnaPatchDown", "FUSIONS", "NOT_APPLICABLE",
           "fusion_of"]
