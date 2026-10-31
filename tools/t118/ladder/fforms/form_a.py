"""Rung A (h12) — per-bin affine f.

theta = (a, b, d), n_theta = 3, context = 0:
  loc_i  = a + b * x_i          (counts: log NB mean; pval: log-normal mu)
  disp_i = d                    (counts: log NB n;    pval: log sigma), one value for every bin
`init_theta` = (0, 1, log n0) for counts / (0, 1, log sigma0) for pval: the identity map.
Magnitude only — a depth shift (a) and a change of dynamic range (b); no shape.
"""
from __future__ import annotations

import math

import torch

from ladder.base import FForm


class FormA(FForm):
    rung = "A"
    n_theta = 3
    context = 0

    def init_theta(self) -> torch.Tensor:
        d0 = self.stats["n0"] if self.space == "counts" else self.stats["sigma0"]
        return torch.tensor([0.0, 1.0, math.log(d0)], dtype=torch.float32)

    def forward(self, x: torch.Tensor, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 2 or theta.dim() != 2 or theta.shape != (x.shape[0], self.n_theta):
            raise ValueError(f"form A: x {tuple(x.shape)} must be [B, L] and theta "
                             f"{tuple(theta.shape)} must be [B, {self.n_theta}]")
        a, b, d = theta[:, 0:1], theta[:, 1:2], theta[:, 2:3]
        loc = a + b * x
        disp = d.expand_as(loc)
        return loc, disp

    def describe(self, theta: torch.Tensor) -> dict:
        """{"a", "b", "disp"}: disp is d as the form emits it (log n for counts, log sigma for pval)."""
        t = theta.detach().reshape(-1).double().cpu()
        if t.numel() != self.n_theta:
            raise ValueError(f"form A: theta has {t.numel()} values, not {self.n_theta}")
        return {"a": float(t[0]), "b": float(t[1]), "disp": float(t[2])}


def build(space: str, stats: dict) -> FormA:
    return FormA(space, stats)
