"""t118 rung D — FiLM-modulated dilated CNN.

Trunk: four `nn.Conv1d` layers, 32 channels, kernel 5, dilations 1, 2, 4, 8 ('same' zero padding),
input one channel (x). After each conv the layer is modulated per channel by g's output (FiLM,
`h = gamma * h + beta`) and passed through GELU. Receptive field 1 + 4 * (1 + 2 + 4 + 8) = 61 bins
(about 1.5 kb), so `context = 30`: an output bin sees exactly 30 bins of x on each side, all of
them inside the halo, so the 'same' padding never reaches a returned bin.

Head: `nn.Conv1d(32, 2, 1)`, weight and bias zero-initialised, read per bin:
  loc  = x_centre + head[0]
  disp = log n0 (counts) / log sigma0 (pval) + head[1]
so the map is the identity at initialisation whatever the trunk weights and theta are.

theta (n_theta = 256) = gamma[4][32] then beta[4][32], layer-major. `init_theta` = gamma 1, beta 0.
The trunk and head weights are this module's own parameters (`form.parameters()`), shared across
pairs and trained jointly with g. Because the head starts at zero, the first gradient step reaches
only the head; gamma, beta and the trunk receive gradient from the second step on.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ladder.base import FForm

N_LAYERS = 4
CHANNELS = 32
KERNEL = 5
DILATIONS = (1, 2, 4, 8)
#: bins of context each side: (KERNEL - 1) // 2 * sum(DILATIONS)
CONTEXT = (KERNEL - 1) // 2 * sum(DILATIONS)


class FormD(FForm):
    rung = "D"
    n_theta = 2 * N_LAYERS * CHANNELS
    context = CONTEXT

    def __init__(self, space: str, stats: dict):
        super().__init__(space, stats)
        self.convs = nn.ModuleList(
            nn.Conv1d(1 if i == 0 else CHANNELS, CHANNELS, KERNEL, dilation=d,
                      padding=(KERNEL - 1) // 2 * d)
            for i, d in enumerate(DILATIONS))
        self.head = nn.Conv1d(CHANNELS, 2, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        s = self.stats["n0"] if space == "counts" else self.stats["sigma0"]
        self.disp0 = math.log(s)

    def init_theta(self) -> torch.Tensor:
        half = N_LAYERS * CHANNELS
        return torch.cat([torch.ones(half), torch.zeros(half)])

    def _film(self, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = theta.shape[0]
        half = N_LAYERS * CHANNELS
        gamma = theta[:, :half].reshape(b, N_LAYERS, CHANNELS, 1)
        beta = theta[:, half:].reshape(b, N_LAYERS, CHANNELS, 1)
        return gamma, beta

    def forward(self, x: torch.Tensor, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 2 or theta.dim() != 2 or theta.shape[1] != self.n_theta:
            raise ValueError(f"x {tuple(x.shape)}, theta {tuple(theta.shape)}: "
                             f"want [B, L + {2 * CONTEXT}], [B, {self.n_theta}]")
        if x.shape[1] <= 2 * CONTEXT:
            raise ValueError(f"x length {x.shape[1]} leaves no bins after the ±{CONTEXT} halo")
        gamma, beta = self._film(theta.to(x.dtype))
        h = x.unsqueeze(1)
        for i, conv in enumerate(self.convs):
            h = F.gelu(gamma[:, i] * conv(h) + beta[:, i])
        out = self.head(h)[:, :, CONTEXT:x.shape[1] - CONTEXT]
        loc = x[:, CONTEXT:x.shape[1] - CONTEXT] + out[:, 0]
        disp = self.disp0 + out[:, 1]
        return loc, disp

    def describe(self, theta: torch.Tensor) -> dict:
        t = theta.detach().reshape(1, self.n_theta).double().cpu()
        gamma, beta = self._film(t)
        return {"film_gamma": gamma[0, :, :, 0].tolist(), "film_beta": beta[0, :, :, 0].tolist()}


def build(space: str, stats: dict) -> FormD:
    return FormD(space, stats)
