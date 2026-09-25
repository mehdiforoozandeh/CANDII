"""t118 rung X (exploratory) — g reads the bin value: f(x_i) = h(x_i, g'(C, C')).

The PI's idea: g sees the bin value as well as the covariates, so each bin value gets its own
transformation. Realised as a shared per-bin MLP h over [x_i, theta], where theta = g'(C, C') is the
32-d output of the harness's g (n_theta = 32, context = 0):

  h: [x_i, theta] (33) -> Linear(33, 64) -> GELU -> Linear(64, 64) -> GELU -> Linear(64, 2)
  loc_i  = x_i + h_0                (counts: log NB mean; pval: log-normal mu)
  disp_i = disp0 + h_1              (counts: log NB n;    pval: log sigma)

with disp0 = log n0 (counts) / log sigma0 (pval). The last layer of h is zero-initialised, so the
map is the identity at initialisation whatever theta and the other weights are; `init_theta` is
zeros(32). h's weights are this module's own parameters (`form.parameters()`), shared across pairs
and trained jointly with g (as for rung D). Because the last layer starts at zero, the first
gradient step reaches only that layer; theta and the earlier layers get gradient from step 2 on.

Memory: the first layer is applied as `x_i * W[:, 0] + (theta @ W[:, 1:].T + b)`, which is exactly
`Linear(33, 64)` on the concatenation but never builds the [B, L, 33] input; the hidden
activations are [B, L, 64] (32 x 2048 x 64 float32 = 16 MB per tensor at the training defaults).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ladder.base import FForm

N_THETA = 32
HIDDEN = 64


class FormX(FForm):
    rung = "X"
    n_theta = N_THETA
    context = 0

    def __init__(self, space: str, stats: dict):
        super().__init__(space, stats)
        self.h = nn.Sequential(nn.Linear(1 + N_THETA, HIDDEN), nn.GELU(),
                               nn.Linear(HIDDEN, HIDDEN), nn.GELU(),
                               nn.Linear(HIDDEN, 2))
        nn.init.zeros_(self.h[4].weight)
        nn.init.zeros_(self.h[4].bias)
        s = self.stats["n0"] if space == "counts" else self.stats["sigma0"]
        self.disp0 = math.log(s)

    def init_theta(self) -> torch.Tensor:
        return torch.zeros(N_THETA, dtype=torch.float32)

    def forward(self, x: torch.Tensor, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 2 or theta.dim() != 2 or theta.shape != (x.shape[0], self.n_theta):
            raise ValueError(f"form X: x {tuple(x.shape)} must be [B, L] and theta "
                             f"{tuple(theta.shape)} must be [B, {self.n_theta}]")
        first = self.h[0]
        w = first.weight.to(x.dtype)                                   # [64, 33]
        per_pair = F.linear(theta.to(x.dtype), w[:, 1:], first.bias.to(x.dtype))   # [B, 64]
        pre = x.unsqueeze(-1) * w[:, 0] + per_pair.unsqueeze(1)        # [B, L, 64]
        out = self.h[1:](pre)                                          # [B, L, 2]
        loc = x + out[..., 0]
        disp = self.disp0 + out[..., 1]
        return loc, disp

    def describe(self, theta: torch.Tensor) -> dict:
        """{"theta_norm", "theta": [32]}: h is shared across pairs, so theta is all that is per pair."""
        t = theta.detach().reshape(-1).double().cpu()
        if t.numel() != self.n_theta:
            raise ValueError(f"form X: theta has {t.numel()} values, not {self.n_theta}")
        return {"theta_norm": float(t.norm()), "theta": [float(v) for v in t]}


def build(space: str, stats: dict) -> FormX:
    return FormX(space, stats)
