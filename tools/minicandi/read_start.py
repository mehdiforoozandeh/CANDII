"""Strand channels of read-start counts, and their conversion to overlap counts (t392 chunk C4).

A read-start model predicts, per 25 bp bin and track, an NB over the reads that START in the bin,
in two strand channels: channel 0 plus-strand starts, channel 1 minus-strand starts. ChIP-seq
tracks and the ChIP-seq control use both channels. ATAC-seq and DNase-seq (`UNSTRANDED_TRACKS`,
decision D11) use channel 0 alone, and channel 1 of those tracks is never read or trained.

  STRANDS = 2, UNSTRANDED_TRACKS, channel_mask() -> bool [A, 2]   which channels carry a track
  unstranded_kernel(kernel, n_reads_strand) -> [17]   read-weighted mix of the two kernel rows
  convert_nb(mu_sub, n_sub, kernel, stranded) -> (mu, n)   read-start NB -> overlap-count NB
  stitch(metas, L) -> list of index runs   windows of a batch converted as one array
  strand_output(out) -> {"mu", "n"} [B, L, A]   the read-start NB of a bin (both strands pooled)
  read_start_loss(out, batch, config) -> (loss, parts)   `losses.total_loss` on the strands

Span kernel: `kernel[s, REACH + k]` is the average number of reads of strand s, per read start in
bin j, that cover bin j + k (k from -REACH to REACH). Overlap counts are map (1) of INSTRUCTIONS
section 2: the read starts of each strand convolved with the strand's kernel, summed over strands.

The NB of the converted bin i is the moment match of the sum of independent terms: each start
count X ~ NB(mu_sub, n_sub) of bin i - k and strand s, thinned by w = kernel[s, REACH + k],
contributes mean c = w * mu_sub and variance c + c^2 / n_sub. With mu = sum c and var = sum of the
variances, the size is r = mu^2 / (var - mu) = mu^2 / v, v = sum c^2 / n_sub. v <= 0 (all terms
Poisson) gives N_MAX; r is clamped to contract.N_MIN .. N_MAX. Bins outside the array take the
values of the edge bin (an open edge: the starts there are not predicted).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_TOOLS = Path(__file__).resolve().parents[1]
for _p in (_TOOLS / "t118", _TOOLS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from minicandi import contract as C  # noqa: E402
from minicandi.components import sub_to_fine  # noqa: E402
from minicandi.losses import pool_nb, total_loss  # noqa: E402

STRANDS = 2
UNSTRANDED_TRACKS = ("ATAC-seq", "DNase-seq")
REACH = 8
KERNEL_WIDTH = 2 * REACH + 1


def channel_mask() -> torch.Tensor:
    """bool [A, 2]: True where a track carries the channel; False at channel 1 of an unstranded
    track."""
    m = torch.ones(C.A, STRANDS, dtype=torch.bool)
    for t in UNSTRANDED_TRACKS:
        m[C.TRACK_INDEX[t], 1] = False
    return m


def _unstranded_tracks() -> torch.Tensor:
    """bool [A]: True at the tracks of UNSTRANDED_TRACKS."""
    return ~channel_mask()[:, 1]


def unstranded_kernel(kernel, n_reads_strand) -> torch.Tensor:
    """[17]: the two rows of a [2, 17] span kernel mixed by the read count of each strand."""
    k = torch.as_tensor(kernel, dtype=torch.float64)
    nr = torch.as_tensor(n_reads_strand, dtype=torch.float64)
    if tuple(k.shape) != (STRANDS, KERNEL_WIDTH) or tuple(nr.shape) != (STRANDS,):
        raise ValueError(f"kernel {tuple(k.shape)} and reads {tuple(nr.shape)}, "
                         f"want ({STRANDS}, {KERNEL_WIDTH}) and ({STRANDS},)")
    tot = float(nr.sum())
    if tot <= 0:
        raise ValueError("n_reads_strand sums to 0: no reads to weight the kernel rows")
    return (nr[:, None] * k).sum(0) / tot


def convert_nb(mu_sub: torch.Tensor, n_sub: torch.Tensor, kernel, stranded: bool):
    """(mu, n) [..., L] of overlap counts from read-start (mu_sub, n_sub) [..., L, 2].

    `kernel` is [2, 17] (row s for strand s). For an unstranded track (`stranded` False) only
    channel 0 and the kernel's row 0 are used; pass the `unstranded_kernel` mix as that row, or a
    [17] kernel. Formula, edge rule and clamp: module docstring."""
    if mu_sub.shape != n_sub.shape or mu_sub.shape[-1] != STRANDS:
        raise ValueError(f"mu_sub {tuple(mu_sub.shape)} and n_sub {tuple(n_sub.shape)} must be "
                         f"equal and end in {STRANDS}")
    w = torch.as_tensor(kernel).to(dtype=mu_sub.dtype, device=mu_sub.device)
    if w.dim() == 1:
        w = w[None, :]
    if w.shape[-1] != KERNEL_WIDTH or w.shape[0] not in (1, STRANDS):
        raise ValueError(f"kernel {tuple(w.shape)}, want ({STRANDS}, {KERNEL_WIDTH}) or "
                         f"({KERNEL_WIDTH},)")
    strands = range(STRANDS) if stranded else range(1)
    if stranded and w.shape[0] != STRANDS:
        raise ValueError("a stranded conversion needs a [2, 17] kernel")
    L = mu_sub.shape[-2]
    pos = torch.arange(L, device=mu_sub.device)
    n_sub = n_sub.clamp_min(C.N_MIN)
    mu = torch.zeros(mu_sub.shape[:-1], dtype=mu_sub.dtype, device=mu_sub.device)
    v = torch.zeros_like(mu)
    for k in range(-REACH, REACH + 1):
        src = (pos - k).clamp(0, L - 1)                      # edge fill outside the array
        for s in strands:
            wk = w[s, REACH + k]
            c = wk * mu_sub[..., src, s]
            mu = mu + c
            v = v + c * c / n_sub[..., src, s]
    n_max = torch.full_like(mu, float(C.N_MAX))
    n = torch.where(v > 0, mu * mu / torch.where(v > 0, v, torch.ones_like(v)), n_max)
    return mu, n.clamp(float(C.N_MIN), float(C.N_MAX))


def stitch(metas, L: int) -> list[list[int]]:
    """Index runs of a batch's windows: windows that share cell, chromosome and targets and whose
    `start_bin` values follow each other by L, in genome order. Runs are ordered by their first
    window in the batch; every window is in exactly one run."""
    groups: dict = {}
    for i, m in enumerate(metas):
        key = (m["cell"], m["chrom"], tuple(m["targets"]))
        groups.setdefault(key, []).append(i)
    runs = []
    for idx in groups.values():
        idx = sorted(idx, key=lambda i: int(metas[i]["start_bin"]))
        run = [idx[0]]
        for i in idx[1:]:
            if int(metas[i]["start_bin"]) == int(metas[run[-1]]["start_bin"]) + L:
                run.append(i)
            else:
                runs.append(run)
                run = [i]
        runs.append(run)
    return sorted(runs, key=lambda r: min(r))


def strand_output(out: dict) -> dict:
    """{"mu", "n"} [B, L, A] in read-start space from `mu_sub`, `n_sub` [B, L, A, 2]: both
    channels pooled by `losses.pool_nb` for a stranded track, channel 0 alone for an unstranded
    one."""
    mu_sub, n_sub = out["mu_sub"], out["n_sub"]
    mu_p, n_p = pool_nb(mu_sub, n_sub, STRANDS, dim=-1)
    un = _unstranded_tracks().to(mu_sub.device)
    mu = torch.where(un, mu_sub[..., 0], mu_p[..., 0])
    n = torch.where(un, n_sub[..., 0], n_p[..., 0])
    return {"mu": mu, "n": n}


_INEXACT = ("loss_assay_weight", "loss_shape_weight", "loss_multiscale", "loss_bin_equal")


def read_start_loss(out: dict, batch: dict, config: dict | None):
    """(loss, parts): `losses.total_loss` on the [B, 2L, A] strand layout of
    `res_dataset.res_total_loss`, with channel 1 of every unstranded track outside the loss.

    Two calls of `total_loss`: the stranded tracks on the [B, 2L, A] layout (the unstranded
    tracks set to KIND_NONE), and the unstranded tracks on channel 0, [B, L, A] (the stranded
    tracks set to KIND_NONE). Their main terms are combined weighted by the number of bins each
    masks, which equals one mean over the union of the two masks. The config options whose term
    is not such a mean (assay weights, shape, multiscale, bin-equal) and two-step outputs raise
    ValueError."""
    config = config or {}
    bad = [k for k in _INEXACT if config.get(k) not in (None, 0, False, "none", [])]
    if bad or "mu_base" in out:
        raise ValueError(f"read_start_loss does not support {bad or ['two_step']}")
    if "y_counts_sub" not in batch:
        raise KeyError("read_start_loss needs batch['y_counts_sub'] [B, L, A, 2]")
    un = _unstranded_tracks().to(batch["y_kind"].device)
    none = torch.full_like(batch["y_kind"], C.KIND_NONE)
    rest = {k: v for k, v in out.items() if k not in ("n", "mu", "n_sub", "mu_sub")}

    # stranded tracks, both channels, [B, 2L, A]
    s_out = dict(rest, n=sub_to_fine(out["n_sub"]), mu=sub_to_fine(out["mu_sub"]))
    s_batch = dict(batch, y_counts=sub_to_fine(batch["y_counts_sub"]),
                   x_bin_mask=batch["x_bin_mask"].repeat_interleave(STRANDS, dim=1),
                   y_kind=torch.where(un, none, batch["y_kind"]))
    # unstranded tracks, channel 0, [B, L, A]
    u_out = dict(rest, n=out["n_sub"][..., 0], mu=out["mu_sub"][..., 0])
    u_batch = dict(batch, y_counts=batch["y_counts_sub"][..., 0],
                   y_kind=torch.where(un, batch["y_kind"], none))

    calls = []
    for o, b in ((s_out, s_batch), (u_out, u_batch)):
        k = int(C.loss_mask(b).sum())
        if k:
            calls.append((k,) + total_loss(o, b, config))
    if not calls:
        return total_loss(s_out, s_batch, config)        # an empty mask: a zero with a graph
    ktot = sum(k for k, _, _ in calls)
    loss = sum(l * (k / ktot) for k, l, _ in calls)
    parts: dict = {}
    for key in {key for _, _, p in calls for key in p}:
        have = [(k, p[key]) for k, _, p in calls if key in p]
        kk = sum(k for k, _ in have)
        parts[key] = sum(k * x for k, x in have) / kk
    parts["main"] = float(loss.detach())
    return loss, parts


__all__ = ["KERNEL_WIDTH", "REACH", "STRANDS", "UNSTRANDED_TRACKS", "channel_mask", "convert_nb",
           "read_start_loss", "stitch", "strand_output", "unstranded_kernel"]
