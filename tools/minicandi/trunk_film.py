"""Trunk FiLM for miniCANDI rungs 2, 3 and 4 (t370 recipe run, chunk C1).

Today every model reads the covariates at two places only: a per-track FiLM on the input
features with C and a per-track FiLM on the head features with C'. The config key
`film_placement` adds FiLM inside the trunk as well:

  ends        today's model: no trunk FiLM (default)
  encoder     every encoder ConvBlock of the count path
  decoder     every decoder ConvBlock
  bottleneck  conv autoencoder: the `mid` ConvBlocks; conv + attention and U-Net: gamma * seq +
              beta on the input of each transformer layer
  all         the three together

The two per-track FiLMs stay in every value. Every trunk site reads one window-level condition,
`pooled_condition`: the mean C embedding over the available input tracks concatenated with the
mean C' embedding over the target tracks. `TrunkFiLM` maps it to one (gamma, beta) per site; each
site starts as the identity (weight 0, bias (ones, zeros)).

`covariate_injection = tokens` with a non-default placement raises NotApplicable. The U-Net-only
keys `skip_film` and `head_skip` are read in `rungs/unet.py`; `check_unet_only` refuses them on
the other rungs.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from minicandi.components import NotApplicable

PLACEMENTS = ("ends", "encoder", "decoder", "bottleneck", "all")
UNET_ONLY = {"skip_film": "cprime", "head_skip": "plain"}   # key: default (today's model)


def placement_of(cfg: dict) -> str:
    """The `film_placement` value of `cfg` (default "ends"); refuses unknown values and tokens."""
    p = cfg.get("film_placement", "ends")
    if p not in PLACEMENTS:
        raise ValueError(f"film_placement {p!r} not in {PLACEMENTS}")
    if p != "ends" and cfg.get("covariate_injection") == "tokens":
        raise NotApplicable(f"film_placement = {p} needs the covariate embeddings of a "
                            f"concat / film / adaln_zero injector; covariate_injection = tokens "
                            f"does not apply")
    return p


def sites_of(placement: str) -> frozenset:
    """The trunk parts that get FiLM under `placement`."""
    if placement == "all":
        return frozenset(("encoder", "decoder", "bottleneck"))
    return frozenset() if placement == "ends" else frozenset((placement,))


def check_unet_only(cfg: dict, name: str) -> None:
    """Refuse a non-default U-Net-only key (`skip_film`, `head_skip`) on another rung."""
    for key, default in UNET_ONLY.items():
        if key in cfg and cfg[key] != default:
            raise NotApplicable(f"{key} = {cfg[key]!r} is a U-Net (rung 4) key; {name} has no "
                                f"U-Net skips")


def pooled_condition(e_in: torch.Tensor, x_avail: torch.Tensor, e_out: torch.Tensor,
                     y_avail: torch.Tensor) -> torch.Tensor:
    """[B, 2 d]: mean of `e_in` [B, A, d] over the available input tracks, then mean of `e_out`
    [B, A, d] over the target tracks. A window with no such track gives zeros for that half."""
    def mean(e, avail):
        w = avail.to(e.dtype)[..., None]
        return (e * w).sum(1) / w.sum(1).clamp_min(1.0)
    return torch.cat([mean(e_in, x_avail), mean(e_out, y_avail)], dim=-1)


class TrunkFiLM(nn.Module):
    """cond [B, d_cond] -> [(gamma, beta) per site], each [B, ch_i, 1].

    One shared MLP d_cond -> hidden -> hidden (ReLU) and one linear head per site with weight 0
    and bias (ones, zeros), so every site is the identity at the start."""

    def __init__(self, d_cond: int, channels: Sequence[int], hidden: int = 64):
        super().__init__()
        self.channels = [int(c) for c in channels]
        self.shared = nn.Sequential(nn.Linear(d_cond, hidden), nn.ReLU(),
                                    nn.Linear(hidden, hidden), nn.ReLU())
        self.heads = nn.ModuleList(nn.Linear(hidden, 2 * c) for c in self.channels)
        with torch.no_grad():
            for head, c in zip(self.heads, self.channels):
                head.weight.zero_()
                head.bias.copy_(torch.cat([torch.ones(c), torch.zeros(c)]))

    def forward(self, cond: torch.Tensor) -> list:
        z = self.shared(cond)
        out = []
        for head, c in zip(self.heads, self.channels):
            t = head(z)
            out.append((t[:, :c, None], t[:, c:, None]))
        return out


__all__ = ["PLACEMENTS", "TrunkFiLM", "UNET_ONLY", "check_unet_only", "placement_of",
           "pooled_condition", "sites_of"]
