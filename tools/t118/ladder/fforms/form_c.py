"""Rung C — a 33-bin kernel, then rung B's monotone curve.

g outputs theta = (w_0..w_32, y0, s_1..s_11, d_0..d_11), n_theta = 57:

* **kernel** w (33 bins = 825 bp), unconstrained — it can smooth (broaden) or sharpen. It is the
  same at every position and differs per sample (per pair). Applied as a cross-correlation over
  the haloed input: `x_c[i] = sum_k w[k] * x[i + k]` for the haloed x of width L + 32, i.e.
  `w[16]` is the centre bin and `w[k]` weighs the bin `k - 16` to the right. This is
  `conv1d(x, w, padding=16)` sliced to the centre L bins.
* **curve** (rung B's, re-implemented here so the file stands alone): 12 knots at the fixed source
  quantiles `stats["knots_x"]`; curve value at knot k is `y0 + sum_{j<k} softplus(s_j)` (monotone
  non-decreasing); `loc(x_c)` interpolates it linearly between the knots and continues the first /
  last segment's slope outside; `disp(x_c)` interpolates `d_k` over the knots, clamped to the end
  values.

`init_theta()`: kernel one-hot at 16, `y0 = knots_x[0]`, `softplus(s_j) = knots_x[j+1] - knots_x[j]`,
`d_k = log n0` (counts) or `log sigma0` (pval) -> loc = centre of x, the identity.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from ladder.base import FForm

N_KERNEL = 33
HALF = N_KERNEL // 2  # 16
N_KNOTS = 12


def _softplus_inv(v: np.ndarray) -> np.ndarray:
    """float64: the s with softplus(s) = v (v > 0), stable for large and small v."""
    v = np.asarray(v, dtype=np.float64)
    return v + np.log(-np.expm1(-v))


class FormC(FForm):
    rung = "C"
    n_theta = N_KERNEL + 1 + (N_KNOTS - 1) + N_KNOTS  # 57
    context = HALF

    def __init__(self, space: str, stats: dict):
        super().__init__(space, stats)
        knots = self.stats["knots_x"]
        if knots.shape != (N_KNOTS,) or not np.all(np.diff(knots) > 0):
            raise ValueError(f"knots_x must be {N_KNOTS} strictly increasing values, got {knots}")
        self.register_buffer("knots_x", torch.as_tensor(knots, dtype=torch.float32), persistent=False)
        self.disp0 = math.log(self.stats["n0"] if space == "counts" else self.stats["sigma0"])

    # -- theta layout -------------------------------------------------------------------------
    @staticmethod
    def _split(theta: torch.Tensor):
        w = theta[..., :N_KERNEL]
        y0 = theta[..., N_KERNEL:N_KERNEL + 1]
        s = theta[..., N_KERNEL + 1:N_KERNEL + N_KNOTS]
        d = theta[..., N_KERNEL + N_KNOTS:]
        return w, y0, s, d

    def init_theta(self) -> torch.Tensor:
        knots = self.stats["knots_x"]
        w = np.zeros(N_KERNEL)
        w[HALF] = 1.0
        theta = np.concatenate([w, knots[:1], _softplus_inv(np.diff(knots)),
                                np.full(N_KNOTS, self.disp0)])
        return torch.as_tensor(theta, dtype=torch.float32)

    # -- the map -------------------------------------------------------------------------------
    def _curve_y(self, y0: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """[..., 12] curve values at the knots."""
        steps = F.softplus(s)
        return torch.cat([y0, y0 + torch.cumsum(steps, dim=-1)], dim=-1)

    def forward(self, x: torch.Tensor, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 2 or theta.dim() != 2 or x.shape[0] != theta.shape[0]:
            raise ValueError(f"x [B, L+{2 * HALF}] and theta [B, {self.n_theta}] expected, "
                             f"got {tuple(x.shape)} and {tuple(theta.shape)}")
        if theta.shape[1] != self.n_theta:
            raise ValueError(f"theta has {theta.shape[1]} columns, expected {self.n_theta}")
        b, width = x.shape
        if width <= 2 * HALF:
            raise ValueError(f"x width {width} leaves no centre after a halo of {HALF} each side")
        w, y0, s, d = self._split(theta.to(x.dtype))

        # per-sample kernel: grouped conv, one group per sample; no padding = the centre L bins
        x_c = F.conv1d(x.unsqueeze(0), w.unsqueeze(1), groups=b).squeeze(0)  # [B, L]

        knots = self.knots_x.to(x.dtype)
        idx = (torch.searchsorted(knots, x_c.contiguous(), right=True) - 1).clamp(0, N_KNOTS - 2)
        k_lo, k_hi = knots[idx], knots[idx + 1]
        t = (x_c - k_lo) / (k_hi - k_lo)  # unclamped: end segments extrapolate with their slope

        y = self._curve_y(y0, s)
        y_lo, y_hi = torch.gather(y, 1, idx), torch.gather(y, 1, idx + 1)
        loc = y_lo + t * (y_hi - y_lo)

        tc = t.clamp(0.0, 1.0)  # dispersion is clamped to the end values outside the knots
        d_lo, d_hi = torch.gather(d, 1, idx), torch.gather(d, 1, idx + 1)
        disp = d_lo + tc * (d_hi - d_lo)
        return loc, disp

    def describe(self, theta: torch.Tensor) -> dict:
        theta = torch.as_tensor(theta).detach().to(torch.float64).reshape(-1)
        if theta.numel() != self.n_theta:
            raise ValueError(f"theta has {theta.numel()} entries, expected {self.n_theta}")
        w, y0, s, d = self._split(theta)
        return {"kernel": [float(v) for v in w],
                "knots_x": [float(v) for v in self.stats["knots_x"]],
                "curve_y": [float(v) for v in self._curve_y(y0, s)],
                "disp": [float(v) for v in d]}


def build(space: str, stats: dict) -> FormC:
    return FormC(space, stats)
