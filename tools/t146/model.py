"""t146 — the shared design-D model with optional extra input channels and an auxiliary head.

`FormDX` is old rung D (`ladder/fforms/form_d.py`: 4 Conv1d layers x 32 channels, kernel 5,
dilations 1/2/4/8, FiLM `h = gamma * conv(h) + beta`, GELU, zero-initialised 1x1 head, loc =
x_centre + head[0], disp = log n0 + head[1]) built by `FormD.__init__` itself, so with no extras
its parameters are created in the same order as old D's. Extras, all created AFTER old D's modules:

  inputs "map" / "mapshift" / "mask"  one more input channel each (the first conv is rebuilt with
                                      1 + k input channels)
  inputs "dna"                        `DNAEncoder`: one-hot [B, 4, 25 * W] -> Conv1d(4, 16, 15) ->
                                      GELU -> AvgPool1d(25) -> Conv1d(16, 4, 1): 4 more channels per
                                      25 bp bin
  aux head                            a second zero-initialised 1x1 head (h29 imputation / denoising)
                                      on the same trunk and FiLM; the residual form is the same

loc stays `x_centre + head`, so every arm is the identity map at initialisation. `T146Model` =
the form + old D's g (`ladder.model.G`, 40 -> 64 -> 64 -> 256, last layer zero, bias gamma 1 /
beta 0), built form first, as `ladder.model.Ladder` builds it.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

_TOOLS = Path(__file__).resolve().parents[1]
for _p in (_TOOLS / "t118", _TOOLS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ladder import encoding  # noqa: E402
from ladder.fforms.form_d import CONTEXT, DILATIONS, KERNEL, FormD  # noqa: E402
from ladder.model import G, MU_MAX, MU_MIN, N_MAX, N_MIN, nb_nll, transform_x  # noqa: E402

RES = 25
DNA_CH = 4
CHUNK = 65_536
SCALAR_INPUTS = ("map", "mapshift", "mask")


class DNAEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 16, 15, padding=7)
        self.pool = nn.AvgPool1d(RES, RES)
        self.conv2 = nn.Conv1d(16, DNA_CH, 1)

    def forward(self, onehot: torch.Tensor) -> torch.Tensor:
        """[B, 4, 25 W] -> [B, 4, W]."""
        return self.conv2(self.pool(F.gelu(self.conv1(onehot))))


def onehot(codes: torch.Tensor) -> torch.Tensor:
    """uint8/long codes [B, n] (0-3 ACGT, 4 N) -> float [B, 4, n], N = all zero."""
    c = codes.long()
    oh = F.one_hot(c.clamp(max=4), 5)[..., :4]
    return oh.permute(0, 2, 1).float()


class FormDX(FormD):
    def __init__(self, space: str, stats: dict, inputs=(), aux_head: bool = False):
        super().__init__(space, stats)                       # old D's modules, old D's order
        self.inputs = tuple(inputs)
        bad = [i for i in self.inputs if i not in SCALAR_INPUTS + ("dna",)]
        if bad:
            raise ValueError(f"unknown inputs {bad}")
        self.scalar = tuple(i for i in self.inputs if i in SCALAR_INPUTS)
        self.has_dna = "dna" in self.inputs
        n_in = 1 + len(self.scalar) + (DNA_CH if self.has_dna else 0)
        if n_in != 1:
            self.convs[0] = nn.Conv1d(n_in, self.channels, KERNEL, dilation=DILATIONS[0],
                                      padding=(KERNEL - 1) // 2 * DILATIONS[0])
        if self.has_dna:
            self.dna_enc = DNAEncoder()
        self.aux_head = None
        if aux_head:
            self.aux_head = nn.Conv1d(self.channels, 2, 1)
            nn.init.zeros_(self.aux_head.weight)
            nn.init.zeros_(self.aux_head.bias)

    def forward(self, x, theta, extra=None, dna=None, head: str = "corr"):
        """x [B, W] (W = L + 2 CONTEXT), theta [B, n_theta], extra [B, k, W] (the scalar inputs in
        `self.scalar` order), dna codes [B, 25 W] -> (loc [B, L], disp [B, L])."""
        if x.dim() != 2 or theta.shape != (x.shape[0], self.n_theta):
            raise ValueError(f"x {tuple(x.shape)}, theta {tuple(theta.shape)}")
        W = x.shape[1]
        parts = [x.unsqueeze(1)]
        if self.scalar:
            if extra is None or extra.shape != (x.shape[0], len(self.scalar), W):
                raise ValueError(f"extra must be [B, {len(self.scalar)}, {W}]")
            parts.append(extra.to(x.dtype))
        if self.has_dna:
            if dna is None or dna.shape != (x.shape[0], W * RES):
                raise ValueError(f"dna must be [B, {W * RES}]")
            parts.append(self.dna_enc(onehot(dna)).to(x.dtype))
        h = torch.cat(parts, 1) if len(parts) > 1 else parts[0]
        gamma, beta = self._film(theta.to(x.dtype))
        for i, conv in enumerate(self.convs):
            h = F.gelu(gamma[:, i] * conv(h) + beta[:, i])
        hd = self.head if head == "corr" else self.aux_head
        if hd is None:
            raise ValueError("no aux head")
        out = hd(h)[:, :, CONTEXT:W - CONTEXT]
        loc = x[:, CONTEXT:W - CONTEXT] + out[:, 0]
        disp = self.disp0 + out[:, 1]
        return loc, disp


class T146Model(nn.Module):
    def __init__(self, inputs=(), aux_head=False, n_cov: int = encoding.N_COV,
                 stats: dict | None = None):
        super().__init__()
        stats = stats or {"knots_x": np.arange(12.0), "n0": 5.0, "sigma0": 0.5}
        self.form = FormDX("counts", stats, inputs=inputs, aux_head=aux_head)
        self.g = G(2 * n_cov, self.form.n_theta, self.form.init_theta())
        self.context = CONTEXT
        self.inputs = self.form.inputs

    def theta(self, cov):
        return self.g(cov)

    def forward(self, cov, x, extra=None, dna=None, head="corr"):
        return self.form(x, self.g(cov), extra=extra, dna=dna, head=head)

    @staticmethod
    def nll(loc, disp, y, mask=None):
        per = nb_nll(loc, disp, y)
        if mask is None:
            return per.mean()
        m = mask.to(per.dtype)
        return (per * m).sum() / m.sum().clamp(min=1.0)


def mean_of(loc) -> np.ndarray:
    """NB mean exp(loc), clamped as the loss clamps it."""
    return np.exp(np.clip(np.asarray(loc, np.float64), math.log(MU_MIN), math.log(MU_MAX)))


def size_of(disp) -> np.ndarray:
    return np.exp(np.clip(np.asarray(disp, np.float64), math.log(N_MIN), math.log(N_MAX)))


@torch.no_grad()
def predict_chrom(model: T146Model, X: np.ndarray, theta: torch.Tensor, device,
                  extra_full: np.ndarray | None = None, dna_full: np.ndarray | None = None,
                  transformed: bool = False, chunk: int = CHUNK, head: str = "corr"):
    """(loc, disp) float32[n] over a whole chromosome. X: raw counts (or, `transformed`, already
    log1p'd — the chained route's leg-1 mean goes in raw). extra_full [k, n] float32 (the scalar
    inputs), dna_full uint8[n * 25]. Chunks of `chunk` bins with CONTEXT bins of halo; outside the
    chromosome x = 0, extras 0, DNA N (as the training windows pad)."""
    c = CONTEXT
    n = int(np.asarray(X).shape[0])
    chunk = max(1, min(int(chunk), n))
    n_chunks = max(1, -(-n // chunk))
    L = n_chunks * chunk + 2 * c
    xp = np.zeros(L, dtype=np.float32)
    xp[c:c + n] = X if transformed else transform_x(np.asarray(X), "counts")
    ep = None
    if extra_full is not None and extra_full.shape[0]:
        ep = np.zeros((extra_full.shape[0], L), dtype=np.float32)
        ep[:, c:c + n] = extra_full
    dp = None
    if dna_full is not None:
        dp = np.full(L * RES, 4, dtype=np.uint8)
        dp[c * RES:(c + n) * RES] = dna_full[:n * RES]
    theta = theta.reshape(1, -1).to(device=device, dtype=torch.float32)
    per_batch = 1 if dp is not None else 8
    locs, disps = [], []
    for b0 in range(0, n_chunks, per_batch):
        idx = range(b0, min(b0 + per_batch, n_chunks))
        xb = torch.from_numpy(np.stack([xp[i * chunk:i * chunk + chunk + 2 * c] for i in idx])).to(device)
        eb = (None if ep is None else torch.from_numpy(
            np.stack([ep[:, i * chunk:i * chunk + chunk + 2 * c] for i in idx])).to(device))
        db = (None if dp is None else torch.from_numpy(np.stack(
            [dp[i * chunk * RES:(i * chunk + chunk + 2 * c) * RES] for i in idx])).to(device))
        loc, disp = model.form(xb, theta.expand(xb.shape[0], -1), extra=eb, dna=db, head=head)
        locs.append(loc.reshape(-1).float().cpu().numpy())
        disps.append(disp.reshape(-1).float().cpu().numpy())
    return (np.concatenate(locs)[:n].astype(np.float32),
            np.concatenate(disps)[:n].astype(np.float32))
