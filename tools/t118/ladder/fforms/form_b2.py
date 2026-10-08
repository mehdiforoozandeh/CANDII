"""t118 ladder rung B2 — rung B's 12-knot curve with a per-bin theta (row 2, spec §2 "B2").

Row-2 contract (`per_bin = True`): `forward(x [B, L], theta [B, L, 24])`, theta[:, j] belongs to
bin j (context 0, so W = L). Per bin, the theta layout is rung B's:
  theta[..., 0]       y0         curve value at knot 0
  theta[..., 1:12]    s_1..s_11  raw steps; the curve rises softplus(s_j) from knot j-1 to knot j
  theta[..., 12:24]   d_0..d_11  dispersion at each knot
Bin i's curve is `y0_i + cumsum(softplus(s_i))` at the knots `stats["knots_x"]`, evaluated at that
bin's own x_i: linear interpolation continued with the end segments' slopes for loc, and d_k
interpolated with clamped ends for disp. Each bin's curve is monotone over the knots, but bins at
different levels use different curves, so x -> loc(x) across bins is NOT forced monotone (PI ruling:
the literal lift). With theta constant over bins, B2 equals rung B exactly. `init_theta` is rung B's
identity.

Rung B's arithmetic is restated here rather than imported (row files stand alone, as form_c does).
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


class FormB2(FForm):
    rung = "B2"
    n_theta = 2 * N_KNOTS
    context = 0
    per_bin = True

    def __init__(self, space: str, stats: dict):
        super().__init__(space, stats)
        kx = self.stats["knots_x"]
        if kx.shape != (N_KNOTS,) or not np.all(np.diff(kx) > 0):
            raise ValueError(f"rung B2 needs {N_KNOTS} strictly increasing knots_x, got {kx}")
        # non-persistent: rebuilt from stats, so the state_dict stays empty like rung B's
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
        """theta [..., 24] -> curve value at each knot [..., 12]."""
        steps = F.softplus(theta[..., 1:N_KNOTS])
        zero = torch.zeros_like(theta[..., :1])
        return theta[..., :1] + torch.cat([zero, torch.cumsum(steps, dim=-1)], dim=-1)

    def forward(self, x: torch.Tensor, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if theta.shape != (*x.shape, self.n_theta):
            raise ValueError(f"rung B2: theta {tuple(theta.shape)} must be x {tuple(x.shape)} "
                             f"+ ({self.n_theta},)")
        kx = self.knots_x.to(dtype=x.dtype)
        y = self.curve_y(theta.to(dtype=x.dtype))                    # [B, L, 12]
        d = theta[..., N_KNOTS:].to(dtype=x.dtype)                   # [B, L, 12]
        # segment index 0..10; x below knot 0 uses segment 0, above knot 11 uses segment 10
        idx = torch.searchsorted(kx, x.detach().contiguous(), right=True) - 1
        idx = idx.clamp(0, N_KNOTS - 2)
        x_lo, x_hi = kx[idx], kx[idx + 1]
        t = (x - x_lo) / (x_hi - x_lo)                               # unclamped: slope continues
        i0, i1 = idx.unsqueeze(-1), (idx + 1).unsqueeze(-1)          # [B, L, 1]: per-bin gather
        y_lo, y_hi = torch.gather(y, -1, i0).squeeze(-1), torch.gather(y, -1, i1).squeeze(-1)
        loc = torch.lerp(y_lo, y_hi, t)
        tc = t.clamp(0.0, 1.0)                                       # dispersion: end values
        d_lo, d_hi = torch.gather(d, -1, i0).squeeze(-1), torch.gather(d, -1, i1).squeeze(-1)
        disp = torch.lerp(d_lo, d_hi, tc)
        return loc, disp

    def describe(self, theta: torch.Tensor) -> dict:
        """theta [K, 24] (one row per level) -> per-level curves; a 1-D theta is read as K = 1."""
        th = theta.detach().reshape(-1, self.n_theta).to(torch.float32).cpu()
        return {"n_levels": int(th.shape[0]),
                "knots_x": [float(v) for v in self.stats["knots_x"]],
                "curve_y": [[float(v) for v in row] for row in self.curve_y(th)],
                "disp": [[float(v) for v in row] for row in th[:, N_KNOTS:]]}


def build(space: str, stats: dict) -> FormB2:
    return FormB2(space, stats)
