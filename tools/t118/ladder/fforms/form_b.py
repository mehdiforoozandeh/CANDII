"""t118 ladder rung B — per-bin monotone curve (h17).

12 knots at fixed source quantiles `stats["knots_x"]` (from `ladder.base.knot_stats`). theta (24):
  theta[0]       y0         curve value at knot 0
  theta[1:12]    s_1..s_11  raw steps; the curve rises softplus(s_j) from knot j-1 to knot j
  theta[12:24]   d_0..d_11  dispersion at each knot
Curve value at knot k: `y0 + sum_{j<=k} softplus(s_j)` (strictly increasing for finite s).
`loc(x)` = linear interpolation of the curve between the knots, continued with the first / last
segment's slope below / above the knot range. `disp(x)` = linear interpolation of d_k over the
knots, clamped to d_0 / d_11 outside. `init_theta` is the identity: y0 = knots_x[0],
s_j = softplus^-1(knots_x[j] - knots_x[j-1]), d_k = log n0 (counts) / log sigma0 (pval).

Rung C re-implements this curve inside `form_c.py` (wave 1 files stand alone).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from ladder.base import FForm

N_KNOTS = 12


def _softplus_inv(v: np.ndarray) -> np.ndarray:
    """float64 inverse of softplus for v > 0: log(expm1(v)), stable for large v."""
    v = np.asarray(v, dtype=np.float64)
    return np.where(v > 20.0, v + np.log(-np.expm1(-v)), np.log(np.expm1(np.minimum(v, 20.0))))


class FormB(FForm):
    rung = "B"
    n_theta = 2 * N_KNOTS
    context = 0

    def __init__(self, space: str, stats: dict):
        super().__init__(space, stats)
        kx = self.stats["knots_x"]
        if kx.shape != (N_KNOTS,) or not np.all(np.diff(kx) > 0):
            raise ValueError(f"rung B needs {N_KNOTS} strictly increasing knots_x, got {kx}")
        # non-persistent: rebuilt from stats, so the state_dict stays empty like rung A's
        self.register_buffer("knots_x", torch.as_tensor(kx, dtype=torch.float32), persistent=False)

    def _disp0(self) -> float:
        return math.log(self.stats["n0"] if self.space == "counts" else self.stats["sigma0"])

    def init_theta(self) -> torch.Tensor:
        kx = self.stats["knots_x"]
        th = np.empty(self.n_theta, dtype=np.float64)
        th[0] = kx[0]
        th[1:N_KNOTS] = _softplus_inv(np.diff(kx))
        th[N_KNOTS:] = self._disp0()
        return torch.as_tensor(th, dtype=torch.float32)

    def curve_y(self, theta: torch.Tensor) -> torch.Tensor:
        """theta [B, 24] -> curve value at each knot [B, 12]."""
        steps = F.softplus(theta[:, 1:N_KNOTS])
        zero = torch.zeros_like(theta[:, :1])
        return theta[:, :1] + torch.cat([zero, torch.cumsum(steps, dim=1)], dim=1)

    def forward(self, x: torch.Tensor, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        kx = self.knots_x.to(dtype=x.dtype)
        y = self.curve_y(theta.to(dtype=x.dtype))                    # [B, 12]
        d = theta[:, N_KNOTS:].to(dtype=x.dtype)                     # [B, 12]
        # segment index 0..10; x below knot 0 uses segment 0, above knot 11 uses segment 10
        idx = torch.searchsorted(kx, x.detach().contiguous(), right=True) - 1
        idx = idx.clamp(0, N_KNOTS - 2)
        x_lo, x_hi = kx[idx], kx[idx + 1]
        t = (x - x_lo) / (x_hi - x_lo)                               # unclamped: slope continues
        y_lo, y_hi = torch.gather(y, 1, idx), torch.gather(y, 1, idx + 1)
        loc = torch.lerp(y_lo, y_hi, t)
        tc = t.clamp(0.0, 1.0)                                       # dispersion: end values
        d_lo, d_hi = torch.gather(d, 1, idx), torch.gather(d, 1, idx + 1)
        disp = torch.lerp(d_lo, d_hi, tc)
        return loc, disp

    def describe(self, theta: torch.Tensor) -> dict:
        th = theta.detach().reshape(1, -1).to(torch.float32).cpu()
        return {"knots_x": [float(v) for v in self.stats["knots_x"]],
                "curve_y": [float(v) for v in self.curve_y(th)[0]],
                "disp": [float(v) for v in th[0, N_KNOTS:]]}


def build(space: str, stats: dict) -> FormB:
    return FormB(space, stats)
