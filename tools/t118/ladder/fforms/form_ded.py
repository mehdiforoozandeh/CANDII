"""Rung DED (t126, plan/T126_ENCDEC_A.md "Rung D") — encoder-decoder rung D.

Today's D trunk (`form_d.py`) split in two, with no skip from x to the output:
  encode(x, g_enc(c))   2 conv layers (dilations 1, 2; 32 channels; kernel 5; 'same' zero padding),
                        each FiLM-modulated (h = gamma * h + beta) then GELU -> latent Z [B, 32, W]
  decode(Z, g_dec(c'))  2 conv layers (dilations 4, 8), FiLM then GELU, then D's 1x1 head ->
                        loc = head[0], disp = log n0 + head[1], on the centre W - 2*30 bins
Receptive field 61 bins, context 30, as D. theta has D's layout (n_theta 256: gamma[4][32] then
beta[4][32], layer-major): layers 0, 1 come from g_enc(c), layers 2, 3 from g_dec(c')
(`model.GEncDecD`). `init_theta` = gamma 1, beta 0, as D.

Init (changed from D): the head is zero-initialised as in D, but with no skip the map at
initialisation is the constant loc = 0 (mu = 1), disp = log n0, not the identity.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ladder.base import FForm

CHANNELS = 32
KERNEL = 5
ENC_DILATIONS = (1, 2)
DEC_DILATIONS = (4, 8)
N_LAYERS = len(ENC_DILATIONS) + len(DEC_DILATIONS)
#: bins of context each side of the whole map (30, as D) and of the encoder alone (6)
CONTEXT = (KERNEL - 1) // 2 * (sum(ENC_DILATIONS) + sum(DEC_DILATIONS))
ENC_CONTEXT = (KERNEL - 1) // 2 * sum(ENC_DILATIONS)
#: one side's FiLM vector: gamma[2][32] then beta[2][32]
N_SIDE = 2 * 2 * CHANNELS


def _conv(c_in: int, d: int) -> nn.Conv1d:
    return nn.Conv1d(c_in, CHANNELS, KERNEL, dilation=d, padding=(KERNEL - 1) // 2 * d)


def _side(film: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[B, 128] (gamma[2][32], beta[2][32]) -> gamma, beta [B, 2, 32, 1]."""
    b = film.shape[0]
    half = 2 * CHANNELS
    return (film[:, :half].reshape(b, 2, CHANNELS, 1), film[:, half:].reshape(b, 2, CHANNELS, 1))


class FormDED(FForm):
    rung = "DED"
    n_theta = 2 * N_LAYERS * CHANNELS
    context = CONTEXT
    #: `model.Ladder` builds `GEncDecD` for this kind
    g_kind = "encdec_d"

    def __init__(self, space: str, stats: dict):
        super().__init__(space, stats)
        self.enc = nn.ModuleList(_conv(1 if i == 0 else CHANNELS, d)
                                 for i, d in enumerate(ENC_DILATIONS))
        self.dec = nn.ModuleList(_conv(CHANNELS, d) for d in DEC_DILATIONS)
        self.head = nn.Conv1d(CHANNELS, 2, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        s = self.stats["n0"] if space == "counts" else self.stats["sigma0"]
        self.disp0 = math.log(s)

    def init_theta(self) -> torch.Tensor:
        half = N_LAYERS * CHANNELS
        return torch.cat([torch.ones(half), torch.zeros(half)])

    @staticmethod
    def split_theta(theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """D-layout theta [B, 256] -> (enc film [B, 128], dec film [B, 128])."""
        q = CHANNELS
        g, b = theta[:, :4 * q], theta[:, 4 * q:]
        return (torch.cat([g[:, :2 * q], b[:, :2 * q]], 1), torch.cat([g[:, 2 * q:], b[:, 2 * q:]], 1))

    @staticmethod
    def join_theta(enc: torch.Tensor, dec: torch.Tensor) -> torch.Tensor:
        """(enc film [B, 128], dec film [B, 128]) -> D-layout theta [B, 256]."""
        h = 2 * CHANNELS
        return torch.cat([enc[:, :h], dec[:, :h], enc[:, h:], dec[:, h:]], 1)

    def encode(self, x: torch.Tensor, film_enc: torch.Tensor) -> torch.Tensor:
        """x [B, W], film_enc [B, 128] -> Z [B, 32, W]."""
        gamma, beta = _side(film_enc.to(x.dtype))
        h = x.unsqueeze(1)
        for i, conv in enumerate(self.enc):
            h = F.gelu(gamma[:, i] * conv(h) + beta[:, i])
        return h

    def decode(self, z: torch.Tensor, film_dec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Z [B, 32, W], film_dec [B, 128] -> (loc, disp) [B, W - 2*CONTEXT]: the centre bins."""
        if z.shape[2] <= 2 * CONTEXT:
            raise ValueError(f"Z length {z.shape[2]} leaves no bins after the ±{CONTEXT} halo")
        gamma, beta = _side(film_dec.to(z.dtype))
        h = z
        for i, conv in enumerate(self.dec):
            h = F.gelu(gamma[:, i] * conv(h) + beta[:, i])
        out = self.head(h)[:, :, CONTEXT:z.shape[2] - CONTEXT]
        return out[:, 0], self.disp0 + out[:, 1]

    def forward(self, x: torch.Tensor, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 2 or theta.dim() != 2 or theta.shape[1] != self.n_theta:
            raise ValueError(f"x {tuple(x.shape)}, theta {tuple(theta.shape)}: "
                             f"want [B, L + {2 * CONTEXT}], [B, {self.n_theta}]")
        enc, dec = self.split_theta(theta)
        return self.decode(self.encode(x, enc), dec)

    def describe(self, theta: torch.Tensor) -> dict:
        t = theta.detach().reshape(1, self.n_theta).double().cpu()
        half = N_LAYERS * CHANNELS
        gamma = t[:, :half].reshape(N_LAYERS, CHANNELS)
        beta = t[:, half:].reshape(N_LAYERS, CHANNELS)
        return {"film_gamma": gamma.tolist(), "film_beta": beta.tolist()}


def build(space: str, stats: dict) -> FormDED:
    return FormDED(space, stats)
