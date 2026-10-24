"""t118 rung D2 — FiLM per bin (row 2 of rung D; spec `plan/T118_ROW2_SPEC.md` §2).

Same trunk and head as rung D (`form_d.FormD`, whose `__init__` and `init_theta` this class
inherits, so the parameter names and `state_dict` match D's exactly): four `nn.Conv1d` layers, 32
channels, kernel 5, dilations 1, 2, 4, 8 ('same' zero padding), a zero-initialised
`nn.Conv1d(32, 2, 1)` head, receptive field 61 bins, `context = 30`.

The only change: theta is given per haloed input position, `[B, W, 256]` with W = L + 2 * context and
theta[:, j] belonging to input position j (every position, since a layer's modulation inside the
halo reaches the centre outputs through later layers). Each position carries gamma[4][32] then
beta[4][32], layer-major, as in D. Layer i computes

    h = GELU(gamma[:, i] * conv_i(h) + beta[:, i])     gamma, beta [B, 4, 32, W]

where gamma and beta are strided views of theta (no copy). Head, slicing and the dispersion offset
are D's: loc = x_centre + head[0], disp = log n0 / log sigma0 + head[1]. The map is the identity at
initialisation whatever theta and the trunk weights are, because the head starts at zero. With theta
constant over positions, D2 equals D exactly given the same trunk and head weights.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ladder.fforms.form_d import CHANNELS, CONTEXT, N_LAYERS, FormD


class FormD2(FormD):
    rung = "D2"
    n_theta = 2 * N_LAYERS * CHANNELS
    context = CONTEXT
    per_bin = True

    def _film_bins(self, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """theta [B, W, 256] -> (gamma, beta), each a [B, 4, 32, W] view of theta."""
        b, w, _ = theta.shape
        half = N_LAYERS * CHANNELS
        gamma = theta[:, :, :half].reshape(b, w, N_LAYERS, CHANNELS).permute(0, 2, 3, 1)
        beta = theta[:, :, half:].reshape(b, w, N_LAYERS, CHANNELS).permute(0, 2, 3, 1)
        return gamma, beta

    def forward(self, x: torch.Tensor, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if (x.dim() != 2 or theta.dim() != 3 or theta.shape[0] != x.shape[0]
                or theta.shape[1] != x.shape[1] or theta.shape[2] != self.n_theta):
            raise ValueError(f"x {tuple(x.shape)}, theta {tuple(theta.shape)}: "
                             f"want [B, W], [B, W, {self.n_theta}] with W = L + {2 * CONTEXT}")
        if x.shape[1] <= 2 * CONTEXT:
            raise ValueError(f"x length {x.shape[1]} leaves no bins after the ±{CONTEXT} halo")
        gamma, beta = self._film_bins(theta.to(x.dtype))
        h = x.unsqueeze(1)
        for i, conv in enumerate(self.convs):
            h = F.gelu(gamma[:, i] * conv(h) + beta[:, i])
        out = self.head(h)[:, :, CONTEXT:x.shape[1] - CONTEXT]
        loc = x[:, CONTEXT:x.shape[1] - CONTEXT] + out[:, 0]
        disp = self.disp0 + out[:, 1]
        return loc, disp

    def describe(self, theta: torch.Tensor) -> dict:
        """theta [K, 256] (K levels) -> per level and layer, the channel mean of |gamma - 1| and of
        |beta|: {"n_levels": K, "film_gamma_abs_dev": [K][4], "film_beta_abs": [K][4]}."""
        t = theta.detach().double().cpu()
        if t.dim() != 2 or t.shape[1] != self.n_theta:
            raise ValueError(f"theta {tuple(t.shape)}: want [K, {self.n_theta}]")
        k = t.shape[0]
        half = N_LAYERS * CHANNELS
        gamma = t[:, :half].reshape(k, N_LAYERS, CHANNELS)
        beta = t[:, half:].reshape(k, N_LAYERS, CHANNELS)
        return {"n_levels": k,
                "film_gamma_abs_dev": (gamma - 1.0).abs().mean(dim=2).tolist(),
                "film_beta_abs": beta.abs().mean(dim=2).tolist()}


def build(space: str, stats: dict) -> FormD2:
    return FormD2(space, stats)
