"""Rung AED (t126, plan/T126_ENCDEC_A.md) — rung A's f with the encoder-decoder g, beta learned.

f is rung A's (theta = (a, b, d), loc_i = a + b x_i, disp_i = d). Only g differs: `model.GEncDec`,
a shared MLP phi(c) -> (alpha, log beta) applied to the source and the target covariates
separately, and psi(c') -> d on the target side only:
  b = beta(c') / beta(c),  a = alpha(c') - b alpha(c),  d = psi(c').
So x -> z = (x - alpha(c)) / beta(c) -> loc = alpha(c') + beta(c') z; c' = c is the identity and
arm -> arm composes through the latent. The `g_kind` class attribute is what `model.Ladder` reads.
"""
from __future__ import annotations

from ladder.fforms.form_a import FormA


class FormAED(FormA):
    rung = "AED"
    #: `model.Ladder` builds `GEncDec(beta_learned=True)` for this kind
    g_kind = "encdec"


def build(space: str, stats: dict) -> FormAED:
    return FormAED(space, stats)
