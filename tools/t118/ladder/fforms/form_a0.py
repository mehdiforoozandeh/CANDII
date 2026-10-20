"""Rung A0 (t127) — rung A without the +1 pseudocount: a power law on the raw count, plus a floor.

theta = (a, b, c, d), n_theta = 4, context = 0, counts only:
  mu_i   = e^a * X_i^b + e^c      X_i = expm1(x_i), the raw source count (x = log1p(X))
  loc_i  = log mu_i = logaddexp(a + b * log X_i, c), and = c where X_i = 0
  disp_i = d                      (log NB n), one value for every bin
The ideal depth law is a = ln(d'/d), b = 1, c small. `init_theta` = (0, 1, C0, log n0) with
C0 = log(0.01): mu = X + 0.01, the identity up to the floor. The X = 0 bins take the c branch through
`torch.where` with a finite placeholder in the other branch, so no -inf * 0 reaches a gradient.
"""
from __future__ import annotations

import math

import torch

from ladder.base import FForm

#: the floor's initial value: e^c = 0.01 counts
C0 = math.log(0.01)


class FormA0(FForm):
    rung = "A0"
    n_theta = 4
    context = 0

    def __init__(self, space: str, stats: dict):
        super().__init__(space, stats)
        if space != "counts":
            raise ValueError(f"form A0 is counts only (it reads the raw count), not {space!r}")

    def init_theta(self) -> torch.Tensor:
        return torch.tensor([0.0, 1.0, C0, math.log(self.stats["n0"])], dtype=torch.float32)

    def forward(self, x: torch.Tensor, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 2 or theta.dim() != 2 or theta.shape != (x.shape[0], self.n_theta):
            raise ValueError(f"form A0: x {tuple(x.shape)} must be [B, L] and theta "
                             f"{tuple(theta.shape)} must be [B, {self.n_theta}]")
        a, b, c, d = theta[:, 0:1], theta[:, 1:2], theta[:, 2:3], theta[:, 3:4]
        pos = x > 0
        log_x = torch.log(torch.expm1(torch.where(pos, x, torch.ones_like(x))))   # finite everywhere
        loc = torch.where(pos, torch.logaddexp(a + b * log_x, c.expand_as(x)), c.expand_as(x))
        disp = d.expand_as(loc)
        return loc, disp

    def describe(self, theta: torch.Tensor) -> dict:
        """{"a", "b", "c", "disp"}: disp is d as the form emits it (log n)."""
        t = theta.detach().reshape(-1).double().cpu()
        if t.numel() != self.n_theta:
            raise ValueError(f"form A0: theta has {t.numel()} values, not {self.n_theta}")
        return {"a": float(t[0]), "b": float(t[1]), "c": float(t[2]), "disp": float(t[3])}


def build(space: str, stats: dict) -> FormA0:
    return FormA0(space, stats)
