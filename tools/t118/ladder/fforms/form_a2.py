"""Rung A2 (row 2) — affine per bin, with theta chosen per bin (spec plan/T118_ROW2_SPEC.md §2).

Row-2 contract: g reads the bin's own x_i, so theta is `[B, L, 3]`, one (a_i, b_i, d_i) per bin;
n_theta = 3, context = 0 (W = L):
  loc_i  = a_i + b_i * x_i     (counts: log NB mean; pval: log-normal mu)
  disp_i = d_i                 (counts: log NB n;    pval: log sigma)
`init_theta` = (0, 1, log n0) for counts / (0, 1, log sigma0) for pval, as form A: broadcast to
every bin it is the identity map. Because (a, b, d) vary with x_i, x -> loc and x -> disp are any
smooth function of x; the form stays per-bin (no neighbours) and magnitude only.
"""
from __future__ import annotations

import math

import torch

from ladder.base import FForm


class FormA2(FForm):
    rung = "A2"
    n_theta = 3
    context = 0
    per_bin = True  # set here so the form works whether or not base.FForm declares it

    def init_theta(self) -> torch.Tensor:
        d0 = self.stats["n0"] if self.space == "counts" else self.stats["sigma0"]
        return torch.tensor([0.0, 1.0, math.log(d0)], dtype=torch.float32)

    def forward(self, x: torch.Tensor, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 2 or theta.dim() != 3 or theta.shape != (*x.shape, self.n_theta):
            raise ValueError(f"form A2: x {tuple(x.shape)} must be [B, L] and theta "
                             f"{tuple(theta.shape)} must be [B, L, {self.n_theta}]")
        a, b, d = theta[..., 0], theta[..., 1], theta[..., 2]
        loc = a + b * x
        return loc, d

    def describe(self, theta: torch.Tensor) -> dict:
        """theta [K, 3] (one row per x level) -> {"n_levels": K, "a": [K], "b": [K], "disp": [K]};
        disp is d as the form emits it (log n for counts, log sigma for pval)."""
        t = theta.detach().double().cpu()
        if t.dim() != 2 or t.shape[1] != self.n_theta:
            raise ValueError(f"form A2: theta {tuple(t.shape)} must be [K, {self.n_theta}]")
        return {"n_levels": int(t.shape[0]),
                "a": [float(v) for v in t[:, 0]],
                "b": [float(v) for v in t[:, 1]],
                "disp": [float(v) for v in t[:, 2]]}


def build(space: str, stats: dict) -> FormA2:
    return FormA2(space, stats)
