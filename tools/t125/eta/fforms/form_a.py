"""t125 rung A — per-bin affine f, with the NB spread set by level (plan/ETA_OFFSET_AND_H_DESIGN.md).

theta = (a, b, d_0..d_11), n_theta = 14, context = 0, row 1 (`per_bin = False`):
  loc_i  = a + b * x_i                         (counts: log NB mean; pval: log-normal mu)
  disp_i = lerp of d_k over `stats["knots_x"]` (counts: log NB n;    pval: log sigma),
           clamped to d_0 / d_11 outside the knot range
The spread rule is form B's (`ladder/fforms/form_b.py`), re-implemented here so the file stands
alone. With every d_k equal it is t118 form A exactly. In t125 `x` is eta_in = log1p(X) - log d.
`init_theta` = (0, 1, log n0 x 12) for counts / (0, 1, log sigma0 x 12) for pval: the identity map.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ladder.base import FForm

N_KNOTS = 12


class FormA(FForm):
    rung = "A"
    n_theta = 2 + N_KNOTS
    context = 0
    per_bin = False

    def __init__(self, space: str, stats: dict):
        super().__init__(space, stats)
        kx = self.stats["knots_x"]
        if kx.shape != (N_KNOTS,) or not np.all(np.diff(kx) > 0):
            raise ValueError(f"t125 rung A needs {N_KNOTS} strictly increasing knots_x, got {kx}")
        # non-persistent: rebuilt from stats, so the state_dict stays empty like t118 rung A's
        self.register_buffer("knots_x", torch.as_tensor(kx, dtype=torch.float32), persistent=False)

    def init_theta(self) -> torch.Tensor:
        d0 = self.stats["n0"] if self.space == "counts" else self.stats["sigma0"]
        return torch.tensor([0.0, 1.0] + [math.log(d0)] * N_KNOTS, dtype=torch.float32)

    def forward(self, x: torch.Tensor, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 2 or theta.dim() != 2 or theta.shape != (x.shape[0], self.n_theta):
            raise ValueError(f"t125 form A: x {tuple(x.shape)} must be [B, L] and theta "
                             f"{tuple(theta.shape)} must be [B, {self.n_theta}]")
        theta = theta.to(dtype=x.dtype)
        a, b, d = theta[:, 0:1], theta[:, 1:2], theta[:, 2:]
        loc = a + b * x
        kx = self.knots_x.to(dtype=x.dtype)
        # segment index 0..10; x below knot 0 uses segment 0, above knot 11 uses segment 10
        idx = torch.searchsorted(kx, x.detach().contiguous(), right=True) - 1
        idx = idx.clamp(0, N_KNOTS - 2)
        x_lo, x_hi = kx[idx], kx[idx + 1]
        t = ((x - x_lo) / (x_hi - x_lo)).clamp(0.0, 1.0)            # clamped: end values outside
        d_lo, d_hi = torch.gather(d, 1, idx), torch.gather(d, 1, idx + 1)
        disp = torch.lerp(d_lo, d_hi, t)
        return loc, disp

    def describe(self, theta: torch.Tensor) -> dict:
        """{"a", "b", "disp", "knots_x"}: disp is d_k as the form emits it at knot k (log n for
        counts, log sigma for pval)."""
        t = theta.detach().reshape(-1).double().cpu()
        if t.numel() != self.n_theta:
            raise ValueError(f"t125 form A: theta has {t.numel()} values, not {self.n_theta}")
        return {"a": float(t[0]), "b": float(t[1]),
                "disp": [float(v) for v in t[2:]],
                "knots_x": [float(v) for v in self.stats["knots_x"]]}


def build(space: str, stats: dict) -> FormA:
    return FormA(space, stats)
