"""Rung DSANC (t128 phase 2, plan/T126_ENCDEC_D2.md "Phase 2: PI decisions, 2026-10-25") — the
anchored symmetric encoder-decoder, arm `DSanchor`.

Rung DS (`form_ds.FormDS`) with one change: the encoder half is old D's NB head predicting X_base,
and the middle passed to the decoder is log1p of that NB mean, in training and at test:
  encode_nb(x, theta_e)   theta_e = g(c, C_clean)   -> NB (loc, disp) of X_base   (= f(x, theta_e))
  middle(loc)             z = log1p(clamp(exp(loc), MU_MIN, MU_MAX))   (the decoder's input
                          convention, log1p of counts; what `t128/viabase.py` does with hop 1)
  encode(x, theta_e)      -> (z, disp): the middle, and the NB log-size (NOT a log sigma)
  decode(z, theta_d)      theta_d = g(C_clean, c')  -> NB (loc, disp) of X'
C_clean is fixed at the base's encoded knobs (setting "F"; the training script freezes it), so
L_anchor = NB NLL of X_base under `encode_nb` is well defined. `latent.encode_chrom` reads
`encode()[0]` and so gets the middle unchanged.

At init the map is NOT the identity: loc = log1p X, so the mean is X + 1 and z = log1p(X + 1).
"""
from __future__ import annotations

import torch

from ladder.fforms import form_ds

#: equal to `model.MU_MIN, model.MU_MAX` (a test asserts it; no form module imports `model`)
MU_MIN, MU_MAX = 1e-4, 1e6


class FormDSAnc(form_ds.FormDS):
    rung = "DSANC"
    g_kind = "sym_film"
    cclean_box = False

    def encode_nb(self, x: torch.Tensor, theta_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x [B, M], theta_e [B, 256] -> NB (loc, disp) of X_base [B, M - 60]."""
        return self.f(x, theta_e)

    @staticmethod
    def middle(loc: torch.Tensor) -> torch.Tensor:
        """NB log-mean -> z = log1p of the mean (clamped to [MU_MIN, MU_MAX])."""
        return torch.log1p(torch.exp(loc).clamp(MU_MIN, MU_MAX))

    def encode(self, x: torch.Tensor, theta_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x [B, M], theta_e [B, 256] -> (z, disp) [B, M - 60]; disp is the NB log-size."""
        loc, disp = self.encode_nb(x, theta_e)
        return self.middle(loc), disp


def build(space: str, stats: dict) -> FormDSAnc:
    return FormDSAnc(space, stats)
