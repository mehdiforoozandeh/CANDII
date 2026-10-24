"""Rung AED1 (t126, plan/T126_ENCDEC_A.md) — the encoder-decoder rung A with beta = 1.

As `form_aed`, but phi(c) emits alpha only and beta is fixed to 1: b = 1 exactly and
a = alpha(c') - alpha(c), a pure shift. d = psi(c'), the target side only.
"""
from __future__ import annotations

from ladder.fforms.form_a import FormA


class FormAED1(FormA):
    rung = "AED1"
    #: `model.Ladder` builds `GEncDec(beta_learned=False)` for this kind
    g_kind = "encdec_beta1"


def build(space: str, stats: dict) -> FormAED1:
    return FormAED1(space, stats)
