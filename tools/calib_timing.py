"""Calibration (a) — what a mid-training check COSTS, and whether it can tell epochs apart.

**RETIRED as a runner.** `main()` refuses: it drove `eval.quick_eval` at several `batches_per_pair`
levels, and `candi.eval` is deleted (D15) while `candi.monitor` scores every 25 bp bin and has no
coverage knob to price. The prose below is the record of what calibration (a) asked and how; the
one piece still live and still tested is `gain_vs_jitter`.

Calibration (b) (`tools/window_content.py`) priced coverage from the data alone: on chr21 the
shipped 8 windows per target carry a relative standard error of 1.79, and half of all windows hold
no foreground bin at all. What (b) could not supply is the constant — it scored a surrogate
(predict-the-mean), not a trained model — nor the two numbers a coverage choice actually needs:

1. **What does a check cost?** Seconds, at each coverage, on the real store and the real model.
2. **Can it tell epochs apart?** A selection metric is only worth running if its epoch-to-epoch
   improvement is larger than its own jitter. Below that line, best-checkpoint selection is
   choosing noise, and no amount of it helps.

ONE training run answers both. At every eval epoch the hook scores the same model state at every
coverage level in turn, so the seconds-vs-coverage curve comes from one set of weights and the
epoch-to-epoch curve comes from one run. Training six times at six coverages would spend six times
the GPU to learn less, because the six runs would also differ from each other.

The instrument is `train()` itself, driven with a custom `eval_hook` — not a reimplementation of
the loop. A calibration measured against a copy of the training loop would be calibrating the copy.

    python tools/calib_timing.py --regime /…/regime_eic.json \
        --out /scratch/$USER/candi_kit/calib_a --epochs 10 --levels 1 2 4 8 16
"""
from __future__ import annotations

import numpy as np


def gain_vs_jitter(curve: list) -> dict:
    """Does the metric move more between epochs than it wobbles?

    `gain` is the typical epoch-to-epoch improvement — the median of `c[i] - c[i+1]`, positive when
    the metric is falling, which for CRPS is better.

    `jitter` is the median absolute SECOND difference, `|c[i] - 2c[i+1] + c[i+2]|`. A curve that
    descends smoothly has a second difference near zero however steeply it falls, so this measures
    roughness WITHOUT being fooled by a strong trend — which a plain standard deviation would be.

    `ratio` is gain / jitter. Below 1 the step between two epochs is smaller than the wobble
    between three, so the ordering of two nearby checkpoints is not information. This is the
    number the coverage choice turns on, and it is reported per level rather than argued about.
    """
    c = [x for x in curve if np.isfinite(x)]
    if len(c) < 3:
        return {"n": len(c), "gain": float("nan"), "jitter": float("nan"),
                "ratio": float("nan")}
    d1 = [c[i] - c[i + 1] for i in range(len(c) - 1)]
    d2 = [abs(c[i] - 2 * c[i + 1] + c[i + 2]) for i in range(len(c) - 2)]
    gain, jit = float(np.median(d1)), float(np.median(d2))
    return {"n": len(c), "gain": gain, "jitter": jit,
            "ratio": (gain / jit if jit > 0 else float("inf"))}


def main(argv=None) -> int:
    """RETIRED — the instrument it drove no longer exists, so it refuses rather than crashes.

    This priced `eval.quick_eval` at several `batches_per_pair` levels. `candi.eval` is deleted
    (D15) and the mid-training scorer is `candi.monitor`, which scores every 25 bp bin of the
    regime's eval chromosomes and has no coverage knob at all — so the curve this tool measured is
    not a curve any more, it is a single point. `gain_vs_jitter` above is kept because the question
    it answers ("does the selection metric move more between epochs than it wobbles?") is a
    property of any selection metric, this one included.
    """
    raise SystemExit(
        "tools/calib_timing.py is RETIRED. It priced `candi.eval.quick_eval` at several "
        "`batches_per_pair` coverage levels, and both are gone: `candi.eval` was deleted (D15) and "
        "`candi.monitor` scores every bin of the eval chromosomes, so there is no coverage to "
        "price. The t31 result stands as recorded; nothing re-runs it. `gain_vs_jitter` in this "
        "file is still live and still tested.")


if __name__ == "__main__":
    raise SystemExit(main())
