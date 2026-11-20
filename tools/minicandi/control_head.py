"""ChIP control arms (plan pinned interface E, "Control arms"): the output-only smooth lambda head.

`wrap(inner, config) -> MiniCandi` is applied by `registry.build_model` when
`config["control_head"] != "channel"`.

  control_head = channel        (default) track 35 is an input channel that can be masked, and an
                                output like any track: `wrap` returns `inner` unchanged.
  control_head = lambda_smooth  output only; needs `control_input = false` (else ValueError). The
                                control track's output mean is the inner model's control mean,
                                averaged in log space over windows of `lambda_window_bins` bins
                                (default 40 bins = 1 kb), then upsampled linearly. Its NB size is
                                one learned scalar. Tracks 0 to 34 are passed through untouched.

Smoothing, exactly. Let L be the context length and W the window. The context is cut into windows
[kW, (k+1)W) from bin 0; a last partial window holds the remaining bins. Each window's value is
the mean of log(mu_control) over its bins, placed at the window's centre (bin index
start + (len - 1) / 2). The output at every bin is the piecewise-linear interpolation through
these knots, extended linearly beyond the first and last knot (so no bin range is a constant
step); with one window the output is that window's constant. Then mu is clamped to the contract's
[MU_MIN, MU_MAX]. Because each window mean is a mean of whole Haar supports when W is a multiple
of 8, Haar detail of the inner control log-mean at levels 1 to 3 never reaches the output.

The arms of interface E that use this file: `control_lambda_head` (`control_head =
lambda_smooth`, `control_input = false`). `control_output_only` (`control_input = false`) and
`control_none` (`control_input = false`, `predict_control = false`) act in the dataset only.

Copy-leak guardrail part 1 holds: the wrapper reads only the inner model's output, so any input
the inner model ignores, the wrapper ignores too.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from minicandi import contract as C  # noqa: E402

CONTROL_HEADS = ("channel", "lambda_smooth")
DEFAULT_WINDOW_BINS = 40


def smooth_log_mean(log_mu: torch.Tensor, window: int) -> torch.Tensor:
    """[B, L] log-mean -> [B, L]: window means in log space, linearly interpolated between window
    centres and extrapolated linearly beyond the end knots."""
    B, L = log_mu.shape
    starts = list(range(0, L, window))
    lens = [min(window, L - s) for s in starts]
    knots = torch.stack([log_mu[:, s:s + n].mean(dim=1) for s, n in zip(starts, lens)], dim=1)
    if knots.shape[1] == 1:
        return knots.expand(B, L)
    centres = torch.tensor([s + (n - 1) / 2.0 for s, n in zip(starts, lens)],
                           dtype=log_mu.dtype, device=log_mu.device)
    pos = torch.arange(L, dtype=log_mu.dtype, device=log_mu.device)
    # segment index: the left knot of the segment that holds (or, at the ends, extends to) pos
    seg = torch.searchsorted(centres, pos, right=True) - 1
    seg = seg.clamp(0, len(starts) - 2)
    x0, x1 = centres[seg], centres[seg + 1]
    t = (pos - x0) / (x1 - x0)
    y0, y1 = knots[:, seg], knots[:, seg + 1]
    return y0 + t * (y1 - y0)


class LambdaControlHead(C.MiniCandi):
    """The inner model, with the control track's output replaced by the smooth lambda head."""

    def __init__(self, inner: C.MiniCandi, window_bins: int = DEFAULT_WINDOW_BINS):
        super().__init__()
        self.inner = inner
        self.window_bins = int(window_bins)
        #: log of the control's NB size; one learned scalar (starts at size 1)
        self.log_size = nn.Parameter(torch.zeros(()))

    def forward(self, batch: dict) -> dict:
        out = dict(self.inner(batch))
        n, mu = out["n"], out["mu"]
        ci = C.CONTROL_INDEX
        log_mu = torch.log(mu[..., ci].clamp_min(C.MU_MIN))
        mu_c = torch.exp(smooth_log_mean(log_mu, self.window_bins)).clamp(C.MU_MIN, C.MU_MAX)
        n_c = torch.exp(self.log_size).clamp(C.N_MIN, C.N_MAX).to(n.dtype).expand_as(mu_c)
        out["mu"] = torch.cat([mu[..., :ci], mu_c[..., None], mu[..., ci + 1:]], dim=-1)
        out["n"] = torch.cat([n[..., :ci], n_c[..., None], n[..., ci + 1:]], dim=-1)
        return out


def wrap(inner: C.MiniCandi, config: dict) -> C.MiniCandi:
    """Apply `config["control_head"]` to `inner`; `channel` (or absent) returns `inner` itself."""
    head = config.get("control_head", "channel")
    if head not in CONTROL_HEADS:
        raise ValueError(f"control_head {head!r} not in {CONTROL_HEADS}")
    if head == "channel":
        return inner
    if config.get("control_input", True):
        raise ValueError("control_head = lambda_smooth is output only: it needs "
                         "control_input = false")
    w = config.get("lambda_window_bins", DEFAULT_WINDOW_BINS)
    if isinstance(w, bool) or not isinstance(w, int) or w < 1:
        raise ValueError(f"lambda_window_bins must be a positive int; got {w!r}")
    return LambdaControlHead(inner, w)


__all__ = ["CONTROL_HEADS", "DEFAULT_WINDOW_BINS", "LambdaControlHead", "smooth_log_mean", "wrap"]
