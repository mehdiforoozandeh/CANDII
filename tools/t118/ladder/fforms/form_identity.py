"""A trivial f-form for the training-harness tests — NOT a rung.

n_theta = 1, context = 0: loc = x + 0*theta, disp = log n0 (counts) / log sigma0 (pval) + 0*theta.
The output never depends on theta, so a model built on it stays at noSolution whatever it trains.
"""
from __future__ import annotations

import math

import torch

from ladder.base import FForm


class IdentityForm(FForm):
    rung = "identity"
    n_theta = 1
    context = 0

    def __init__(self, space: str, stats: dict):
        super().__init__(space, stats)
        self.disp0 = math.log(self.stats["n0"] if space == "counts" else self.stats["sigma0"])

    def init_theta(self) -> torch.Tensor:
        return torch.zeros(1)

    def forward(self, x: torch.Tensor, theta: torch.Tensor):
        z = 0.0 * theta[:, :1]
        return x + z, torch.full_like(x, self.disp0) + z

    def describe(self, theta: torch.Tensor) -> dict:
        return {"theta0": float(theta.reshape(-1)[0])}


def build(space: str, stats: dict) -> FForm:
    return IdentityForm(space, stats)
