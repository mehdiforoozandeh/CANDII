"""t118 ladder — the f-form interface every rung implements, the form loader, and the knot statistics.

**Parameterisation** (pinned). Input `x`: counts `log1p(X)`, pval `log(max(X, 1e-3))`. A form maps
`(x, theta)` to `(loc, disp)` per bin:
  counts  NB with mean mu = exp(loc) and size n = exp(disp)
  pval    log-normal with mu = loc (log median) and sigma = exp(disp)
`init_theta()` is the identity map: loc = x, disp = log n0 (counts) / log sigma0 (pval), so every
rung starts at noSolution.

**Context.** `x` carries `context` bins of halo on each side: `x` is `[B, L + 2*context]`, `loc`
and `disp` are `[B, L]` (A, B: context 0; C: 16; D: 30).
"""
from __future__ import annotations

import importlib

import numpy as np
import torch

SPACES = ("counts", "pval")
#: the source-quantile probabilities of the 12 knots (rungs B and C)
KNOT_PROBS = (0.0, 0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995, 0.9999)


class FForm(torch.nn.Module):
    """Base of every f-form. Subclasses set the class attributes and implement the three methods."""

    rung: str = ""
    n_theta: int = 0
    context: int = 0

    def __init__(self, space: str, stats: dict):
        super().__init__()
        if space not in SPACES:
            raise ValueError(f"space {space!r} not in {SPACES}")
        self.space = space
        self.stats = {"knots_x": np.asarray(stats["knots_x"], dtype=np.float64),
                      "n0": float(stats["n0"]), "sigma0": float(stats["sigma0"])}

    def init_theta(self) -> torch.Tensor:
        """Tensor[n_theta]: the identity map (loc = x, disp = log n0 / log sigma0)."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x [B, L + 2*context], theta [B, n_theta] -> (loc [B, L], disp [B, L])."""
        raise NotImplementedError

    def describe(self, theta: torch.Tensor) -> dict:
        """theta [n_theta] -> a dict of plain floats / lists."""
        raise NotImplementedError


def load_form(rung: str, space: str, stats: dict) -> FForm:
    """`ladder.fforms.form_<rung.lower()>.build(space, stats)`."""
    mod = importlib.import_module(f"ladder.fforms.form_{rung.lower()}")
    form = mod.build(space, stats)
    if not isinstance(form, FForm):
        raise TypeError(f"form_{rung.lower()}.build returned {type(form).__name__}, not an FForm")
    if form.space != space:
        raise ValueError(f"form_{rung.lower()}.build: space {form.space!r} != {space!r}")
    return form


def knot_stats(x_sample: np.ndarray, n_knots: int = 12) -> np.ndarray:
    """float64[n_knots], strictly increasing: the quantiles of `x_sample` at KNOT_PROBS, `np.unique`,
    then filled back to `n_knots` by inserting the midpoint of the largest gap, one at a time.

    Heavily tied data (most bins 0) collapses many quantiles into one value; the fill restores the
    count. A sample with a single distinct value gets a second knot at that value + 1 first (there
    is no gap to split otherwise).
    """
    x = np.asarray(x_sample, dtype=np.float64).ravel()
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError("knot_stats needs a non-empty, finite sample")
    k = np.unique(np.quantile(x, KNOT_PROBS))
    if k.size > n_knots:
        raise ValueError(f"{k.size} distinct quantiles > n_knots {n_knots}")
    if k.size == 1:
        k = np.array([k[0], k[0] + 1.0])
    while k.size < n_knots:
        i = int(np.argmax(np.diff(k)))
        k = np.insert(k, i + 1, 0.5 * (k[i] + k[i + 1]))
    if not np.all(np.diff(k) > 0):
        raise ValueError(f"knots not strictly increasing: {k}")
    return k
