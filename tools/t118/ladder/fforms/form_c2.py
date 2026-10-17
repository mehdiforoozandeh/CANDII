"""Rung C2 — row 2 of rung C: a 33-bin kernel chosen at each bin, then rung B's curve chosen at
each bin.

Row-2 contract (`per_bin = True`): `forward(x [B, W], theta [B, W, 57])` with W = L + 2 * 16 and
`theta[:, j]` belonging to input position j. Only the centre L positions' theta are used,
`theta[:, 16:W-16]`; the halo positions' theta are ignored (the kernel is the centre bin's).

Per position i of the centre (haloed index i + 16), theta_i = (w_i[0..32], y0_i, s_i[1..11],
d_i[0..11]) — the same layout as form C:

* **kernel** w_i (33 bins = 825 bp), unconstrained. Cross-correlation over the haloed input,
  `x_c[i] = sum_k w_i[k] * x[i + k]`, i.e. `w_i[16]` is the centre bin and `w_i[k]` weighs the bin
  `k - 16` to the right. Computed as `x.unfold(1, 33, 1)` -> `[B, L, 33]`, weighted by w_i and
  summed. With theta constant over bins this is form C's `conv1d` exactly (up to float rounding).
* **curve** (rung B's, re-implemented here so the file stands alone), with the per-bin knot values:
  12 knots at the fixed source quantiles `stats["knots_x"]`; the curve value at knot k is
  `y0_i + sum_{j<k} softplus(s_i[j])`; `loc` interpolates it linearly at x_c[i] and continues the
  first / last segment's slope outside; `disp` interpolates `d_i` over the knots, clamped to the end
  values. Each bin's curve is monotone; across bins it is the literal per-bin lift, so x -> loc
  need not be monotone (PI ruling).

`init_theta()` is form C's: kernel one-hot at 16, `y0 = knots_x[0]`,
`softplus(s_j) = knots_x[j+1] - knots_x[j]`, `d_k = log n0` (counts) or `log sigma0` (pval) ->
loc = centre of x, the identity, at every bin.
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


class FormC2(FForm):
    rung = "C2"
    n_theta = N_KERNEL + 1 + (N_KNOTS - 1) + N_KNOTS  # 57
    context = HALF
    per_bin = True

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
    @staticmethod
    def _curve_y(y0: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """[..., 12] curve values at the knots."""
        steps = F.softplus(s)
        return torch.cat([y0, y0 + torch.cumsum(steps, dim=-1)], dim=-1)

    def forward(self, x: torch.Tensor, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 2 or theta.dim() != 3 or tuple(theta.shape[:2]) != tuple(x.shape):
            raise ValueError(f"x [B, W] and theta [B, W, {self.n_theta}] expected (W = L + {2 * HALF}), "
                             f"got {tuple(x.shape)} and {tuple(theta.shape)}")
        if theta.shape[2] != self.n_theta:
            raise ValueError(f"theta has {theta.shape[2]} columns, expected {self.n_theta}")
        width = x.shape[1]
        if width <= 2 * HALF:
            raise ValueError(f"x width {width} leaves no centre after a halo of {HALF} each side")
        theta_c = theta[:, HALF:width - HALF].to(x.dtype)  # [B, L, 57]: the centre bins' own theta
        w, y0, s, d = self._split(theta_c)

        # per-bin kernel: patch i covers haloed x[i .. i+32], centre x[i+16]
        patches = x.unfold(1, N_KERNEL, 1)  # [B, L, 33]
        x_c = (patches * w).sum(dim=-1)  # [B, L]

        knots = self.knots_x.to(x.dtype)
        idx = (torch.searchsorted(knots, x_c.contiguous(), right=True) - 1).clamp(0, N_KNOTS - 2)
        k_lo, k_hi = knots[idx], knots[idx + 1]
        t = (x_c - k_lo) / (k_hi - k_lo)  # unclamped: end segments extrapolate with their slope

        y = self._curve_y(y0, s)  # [B, L, 12]
        i_lo, i_hi = idx.unsqueeze(-1), (idx + 1).unsqueeze(-1)
        y_lo, y_hi = torch.gather(y, 2, i_lo).squeeze(-1), torch.gather(y, 2, i_hi).squeeze(-1)
        loc = torch.lerp(y_lo, y_hi, t)

        tc = t.clamp(0.0, 1.0)  # dispersion is clamped to the end values outside the knots
        d_lo, d_hi = torch.gather(d, 2, i_lo).squeeze(-1), torch.gather(d, 2, i_hi).squeeze(-1)
        disp = torch.lerp(d_lo, d_hi, tc)
        return loc, disp

    def describe(self, theta: torch.Tensor) -> dict:
        """theta [K, 57] (a 1-D [57] is read as K = 1) -> per-level lists of length K."""
        theta = torch.as_tensor(theta).detach().to(torch.float64)
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)
        if theta.dim() != 2 or theta.shape[1] != self.n_theta:
            raise ValueError(f"theta [K, {self.n_theta}] expected, got {tuple(theta.shape)}")
        w, y0, s, d = self._split(theta)
        return {"n_levels": int(theta.shape[0]),
                "kernel": w.tolist(),
                "knots_x": [float(v) for v in self.stats["knots_x"]],
                "curve_y": self._curve_y(y0, s).tolist(),
                "disp": d.tolist()}


def build(space: str, stats: dict) -> FormC2:
    return FormC2(space, stats)
