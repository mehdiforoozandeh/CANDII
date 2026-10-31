"""Rung DSBOX (t128, plan/T126_ENCDEC_D2.md) — rung DS with C_clean form D.

The same f as `form_ds.FormDS`; only `cclean_box` differs, so `model.GSym` maps each learned
knob through a sigmoid onto the [min, max] seen in training (`GSym.set_box`), initialised at the
base's covariates (`GSym.set_cclean_init`).
"""
from __future__ import annotations

from ladder.fforms.form_ds import FormDS


class FormDSBox(FormDS):
    rung = "DSBOX"
    cclean_box = True


def build(space: str, stats: dict) -> FormDSBox:
    return FormDSBox(space, stats)
