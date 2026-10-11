"""The peak caller's constants and scaling rules, as Phase 0 found them (`plan/T125_STATUS.md`).

ChIP: chip-seq-pipeline2 v2.2.2, `macs2 callpeak --nomodel --shift 0 --extsize FRAGLEN --keep-dup all
-B --SPMR -g hs`, then `bdgcmp -m ppois`; the control is scaled to the smaller set
(`--scale-to small`), or by `--ratio V` on the ratio arms. DNase: atac-seq-pipeline v2.2.3 in DNase
mode, `--shift -75 --extsize 150`, no control. The local rate is the max of the genome background
and the window rates (fragment, 1 kb, 10 kb from the control; 10 kb of the treatment without one).

The 25 bp products count read-span overlaps: a read of length rl touches on average
(rl + 24) / 25 bins, so the tags behind a binned count are about count * 25 / (rl + 24), and the
caller's pileup (tags extended to the fragment) is about count * fraglen / (rl + 24).
"""
from __future__ import annotations

import numpy as np

GENOME_SIZE = 2.7e9          # MACS2 `-g hs`
BIN = 25
SPAN_EXTRA = BIN - 1         # = 24: a read of length rl touches (rl + 24) / 25 bins on average
#: control windows of the local rate; "frag" = the product's fraglen
WINDOWS_CTL_BP = {"frag": None, "1k": 1000, "10k": 10000}
#: the treatment window of the local rate when there is no control
WINDOW_NOCTL_BP = 10000
PILEUP_RULE = "count * fraglen / (read_length + 24)"
#: P = -log10 P(X > floor(pileup)) at the local rate (MACS2 ppois)
TAIL = "strict"
PSEUDOCOUNT = 0


def scale(N_T: float, N_C: float, ratio_V: float | None = None) -> tuple[float, float]:
    """(s_T, s_C), MACS2 `--scale-to small`: the larger set is scaled down to the smaller one.

    r = ratio_V when given (the ratio arms' `--ratio`), else N_T / N_C; r <= 1 -> (1, r),
    r > 1 -> (1 / r, 1).
    """
    if ratio_V is not None:
        r = float(ratio_V)
    else:
        if N_C <= 0:
            raise ValueError(f"control depth {N_C} <= 0: no scaling without a control")
        r = float(N_T) / float(N_C)
    if not r > 0:
        raise ValueError(f"treatment/control ratio {r} is not positive")
    if r <= 1.0:
        return 1.0, r
    return 1.0 / r, 1.0


def lambda_bg(N_scaled: float, fraglen: float) -> float:
    """The genome-background rate: fraglen * N / GENOME_SIZE (N already scaled).

    MACS2 takes N from the TREATMENT in every arm: N_scaled = N_T * s_T (s_T = 1 without a
    control); `--ratio` scales the control windows only. Same definition as `formula.lambda_bg`.
    """
    return float(fraglen) * float(N_scaled) / GENOME_SIZE


def lambda_window(tags_in_window, window_bp: float, fraglen: float, s: float):
    """A window rate: tags * fraglen / window_bp * s (scalar or array tags)."""
    if isinstance(tags_in_window, np.ndarray):
        tags_in_window = tags_in_window.astype(np.float64)
    else:
        tags_in_window = float(tags_in_window)
    return tags_in_window * float(fraglen) / float(window_bp) * float(s)
