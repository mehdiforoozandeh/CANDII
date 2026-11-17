"""E2 (PI ruling 2026-11-17): the in-bin signal pileup for tests that run an OLD score.py copy.

An old score.py loaded from `git show` imports today's `signal_from_counts` and `ag2_space`, whose
default pileup is now `box_d`; the old scorer never passes `pileup`, so its signal space would use
the new form. `inbin_defaults()` sets the keyword defaults of the signal functions to `inbin` for
the duration, so the old copy reproduces the scorer as it was."""
from __future__ import annotations

import contextlib


@contextlib.contextmanager
def inbin_defaults():
    from onion_eval import ag2_space as AS
    from onion_eval import signal_from_counts as SG
    funcs = [SG.signal_track, SG.signal_from_nb, AS.native_from_nb, AS.peak_probability_from_nb,
             AS.fold_change_track]
    saved = [dict(f.__kwdefaults__ or {}) for f in funcs]
    try:
        for f in funcs:
            if f.__kwdefaults__ and "pileup" in f.__kwdefaults__:
                f.__kwdefaults__["pileup"] = "inbin"
        yield
    finally:
        for f, kw in zip(funcs, saved):
            f.__kwdefaults__ = kw
