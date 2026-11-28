"""The kernel head (t396 chunk C13): start input, count target.

A read-start model (`read_start: true`) predicts per 25 bp bin, track and strand the NB of the
read starts, `mu_sub`, `n_sub` [B, L, A, 2]. With `kernel_head: true` the loss is not taken on the
starts: the fixed span kernel of each target measurement (`y_kernel`, not learned) maps the start
NB to the NB of overlap counts by `read_start.convert_nb`, and `losses.total_loss` scores that NB
against the observed overlap counts of the count store (`y_counts_overlap`).

  convert_nb_batched(mu_sub, n_sub, kernel) -> (mu, n) [B, L, A]   `convert_nb` per window and
                                         track, with a kernel per window and track [B, A, 2, 17]
  kernel_head_loss(out, batch, config) -> (loss, parts)

NB size: the moment match of `read_start.convert_nb` (sum of thinned independent NB terms; size
mu^2 / sum(c^2 / n_sub), N_MAX when that sum is 0, clamped to N_MIN .. N_MAX).

Open edges. The converted bin i reads the starts of bins i - REACH .. i + REACH. The first and the
last REACH bins of a window would read starts outside it (convert_nb fills them from the edge
bin), so those bins are outside the loss: the loss is `total_loss` on bins REACH .. L - REACH - 1.
Their starts still get gradient through the interior bins they cover.

Unstranded tracks (`read_start.UNSTRANDED_TRACKS`). `y_kernel[:, t, 0]` holds the read-weighted mix
of the two kernel rows and `y_kernel[:, t, 1]` is 0, so channel 1 adds nothing: per track this is
`convert_nb(..., stranded=False)` with the mix as row 0.

The dataset side is `rs_dataset.ReadStartDataset.window` (keys `y_counts_overlap`, `y_kernel`);
`res_dataset.train` selects this loss when the config sets `kernel_head`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from minicandi import contract as C  # noqa: E402
from minicandi.losses import total_loss  # noqa: E402
from minicandi.read_start import KERNEL_WIDTH, REACH, STRANDS  # noqa: E402


def check_config(config: dict | None) -> None:
    """ValueError when `kernel_head` is set without `read_start`."""
    cfg = config or {}
    if cfg.get("kernel_head") and not cfg.get("read_start"):
        raise ValueError("kernel_head needs read_start: true (the head predicts read starts)")


def convert_nb_batched(mu_sub: torch.Tensor, n_sub: torch.Tensor, kernel: torch.Tensor):
    """(mu, n) [B, L, A] of overlap counts from read-start (mu_sub, n_sub) [B, L, A, 2] and one
    span kernel per window and track, `kernel` [B, A, 2, 17]. The terms are summed in the order of
    `read_start.convert_nb` (offset, then strand), so each (window, track) equals that function's
    result with the kernel `kernel[b, t]`; edge fill and clamps are the same."""
    if mu_sub.shape != n_sub.shape or mu_sub.dim() != 4 or mu_sub.shape[-1] != STRANDS:
        raise ValueError(f"mu_sub {tuple(mu_sub.shape)} and n_sub {tuple(n_sub.shape)} must be "
                         f"equal and [B, L, A, {STRANDS}]")
    B, L, A, _ = mu_sub.shape
    w = torch.as_tensor(kernel).to(dtype=mu_sub.dtype, device=mu_sub.device)
    if tuple(w.shape) != (B, A, STRANDS, KERNEL_WIDTH):
        raise ValueError(f"kernel {tuple(w.shape)}, want {(B, A, STRANDS, KERNEL_WIDTH)}")
    pos = torch.arange(L, device=mu_sub.device)
    n_sub = n_sub.clamp_min(C.N_MIN)
    mu = torch.zeros(mu_sub.shape[:-1], dtype=mu_sub.dtype, device=mu_sub.device)
    v = torch.zeros_like(mu)
    for k in range(-REACH, REACH + 1):
        src = (pos - k).clamp(0, L - 1)                      # edge fill outside the array
        for s in range(STRANDS):
            wk = w[:, None, :, s, REACH + k]                 # [B, 1, A]
            c = wk * mu_sub[:, src, :, s]
            mu = mu + c
            v = v + c * c / n_sub[:, src, :, s]
    n_max = torch.full_like(mu, float(C.N_MAX))
    n = torch.where(v > 0, mu * mu / torch.where(v > 0, v, torch.ones_like(v)), n_max)
    return mu, n.clamp(float(C.N_MIN), float(C.N_MAX))


def kernel_head_loss(out: dict, batch: dict, config: dict | None):
    """(loss, parts): `losses.total_loss` of the converted overlap NB against
    `batch["y_counts_overlap"]` on bins REACH .. L - REACH - 1 (module docstring)."""
    if "mu_base" in out:
        raise ValueError("kernel_head_loss does not support two_step")
    for key in ("mu_sub", "n_sub"):
        if key not in out:
            raise KeyError(f"kernel_head_loss needs out[{key!r}] [B, L, A, 2] (read_start)")
    for key in ("y_counts_overlap", "y_kernel"):
        if key not in batch:
            raise KeyError(f"kernel_head_loss needs batch[{key!r}] (config kernel_head)")
    mu, n = convert_nb_batched(out["mu_sub"], out["n_sub"], batch["y_kernel"])
    L = mu.shape[1]
    if L <= 2 * REACH:
        raise ValueError(f"a window of {L} bins has no bin outside the {REACH}-bin open edges")
    inner = slice(REACH, L - REACH)
    o = {k: v for k, v in out.items() if k not in ("n", "mu", "n_sub", "mu_sub")}
    o.update(mu=mu[:, inner], n=n[:, inner])
    b = dict(batch, y_counts=batch["y_counts_overlap"][:, inner],
             x_bin_mask=batch["x_bin_mask"][:, inner])
    return total_loss(o, b, config)


__all__ = ["check_config", "convert_nb_batched", "kernel_head_loss"]
