"""Rung DS (t128, plan/T126_ENCDEC_D2.md) — the symmetric encoder-decoder rung D, form A.

ONE old-D CNN (`form_d.FormD`, unchanged: 4 dilated layers, 32 channels, kernel 5, dilations
1, 2, 4, 8, FiLM, 1x1 head, skip loc = x_centre + head[0]) is both the encoder and the decoder,
with the same weights. Only the FiLM differs:
  encode(x, theta_e)   theta_e = g(c, C_clean)   -> Gaussian (mu_z, log sigma_z) per bin
                       mu_z = loc (the skip: Z = x at init), log sigma_z = head[1] (sigma_z = 1)
  decode(z, theta_d)   theta_d = g(C_clean, c')  -> NB (loc, disp) of X'
Z has 1 channel at 25 bp. Two f's in series need 2 * 30 = 60 bins of context each side.

theta (n_theta 512) = cat(theta_e [256], theta_d [256]), each in old D's layout. `model.GSym`
builds it (g_kind "sym_film"); `forward` is the deterministic map decode(mu_z), so `Ladder`,
`Predictor` and the no-covariates twin's mean theta work as for any row-1 rung. At init the whole
map is the identity (loc = x centre, disp = disp0).
"""
from __future__ import annotations

import torch

from ladder import encoding
from ladder.base import FForm
from ladder.fforms import form_d

FILM = form_d.FormD.n_theta
CTX1 = form_d.CONTEXT
CONTEXT = 2 * CTX1
N_KNOB = 13
N_ASSAY = len(encoding.ASSAYS)
KNOB_SLICE = slice(0, N_KNOB)
ASSAY_SLICE = slice(N_KNOB, N_KNOB + N_ASSAY)


class FormDS(FForm):
    rung = "DS"
    n_theta = 2 * FILM
    context = CONTEXT
    per_bin = False
    #: `model.Ladder` builds `GSym` for this kind
    g_kind = "sym_film"
    #: False: C_clean is free (form A); True: boxed by a sigmoid (form D, `form_dsbox`)
    cclean_box = False

    def __init__(self, space: str, stats: dict):
        super().__init__(space, stats)
        self.f = form_d.FormD(space, stats)

    def init_theta(self) -> torch.Tensor:
        return torch.cat([self.f.init_theta(), self.f.init_theta()])

    @staticmethod
    def split_theta(theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """theta [B, 512] -> (theta_e [B, 256], theta_d [B, 256])."""
        return theta[:, :FILM], theta[:, FILM:]

    def encode(self, x: torch.Tensor, theta_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x [B, M], theta_e [B, 256] -> (mu_z, log_sigma_z) [B, M - 60]."""
        loc, disp = self.f(x, theta_e)
        return loc, disp - self.f.disp0

    def decode(self, z: torch.Tensor, theta_d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """z [B, M], theta_d [B, 256] -> NB (loc, disp) [B, M - 60]."""
        return self.f(z, theta_d)

    def forward(self, x: torch.Tensor, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 2 or theta.dim() != 2 or theta.shape[1] != self.n_theta:
            raise ValueError(f"x {tuple(x.shape)}, theta {tuple(theta.shape)}: "
                             f"want [B, L + {2 * CONTEXT}], [B, {self.n_theta}]")
        if x.shape[1] <= 2 * CONTEXT:
            raise ValueError(f"x length {x.shape[1]} leaves no bins after the ±{CONTEXT} halo")
        theta_e, theta_d = self.split_theta(theta)
        return self.decode(self.encode(x, theta_e)[0], theta_d)

    def describe(self, theta: torch.Tensor) -> dict:
        t = theta.detach().reshape(1, self.n_theta)
        e = self.f.describe(t[0, :FILM])
        d = self.f.describe(t[0, FILM:])
        return {"film_gamma_enc": e["film_gamma"], "film_beta_enc": e["film_beta"],
                "film_gamma_dec": d["film_gamma"], "film_beta_dec": d["film_beta"]}


def build(space: str, stats: dict) -> FormDS:
    return FormDS(space, stats)
