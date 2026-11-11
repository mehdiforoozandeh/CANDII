"""miniCANDI rung 5: the axial transformer.

Token layout (one window, L bins, patch P = 16 bins, N = L / P patch columns):

  rows 0..35    input track rows: track a, patch j -> Linear(P x [x, visible]) of the transformed
                counts of its P bins, + a track embedding + a position embedding
  row  36       the DNA row: DnaTower features of the patch's P bins -> Linear, + a row embedding
  rows 37..72   output query rows, one per track: a learned track embedding + the position
                embedding; the output covariates C' of that track condition the row
  column 0      only with covariate_injection = tokens: one covariate token per row (C for an input
                row, C' for a query row, a learned token for the DNA row); the patch columns follow

So a batch is a grid [B, R = 73, N (+1), d]. For L = 1024: N = 64, 73 x 65 = 4745 tokens per window.
Each layer runs attention along positions (within a row) and attention across rows (within a
column), then an FFN. A query row decodes through Linear(d, P x 32) to per-bin features, adds a
per-bin DNA skip, one ConvBlock along the bins of its track, and the shared NBHead (depth add-back
from C' column 0).

Masking (guardrail part 1 holds by construction):
  mask_mode = drop (base)  hidden bins and missing tracks are exact zeros in the per-bin features,
                           the second per-bin channel says which bins are visible; a missing track's
                           row is ABSENT: no row may attend to it across rows, and it is left out of
                           the tied position map
  mask_mode = token        MaskHandler's learned hidden token on hidden bins; a missing track's row
                           is a learned mask row (constant, + track and position embeddings) that
                           other rows may read
Across rows, an input row reads the valid input rows, the DNA row and itself; a query row reads
the valid input rows, the DNA row and itself. No row reads a query row other than itself.

axis (agent extra, PI ruling 14; arms axis_position_only, axis_track_only, axis_tied):
  both            position attention + row attention in every layer (base)
  position_only   position attention only in every layer; the tracks mix once, in one row
                  attention after the stack (without it the query rows would read no input)
  track_only      row attention only; positions mix only inside a patch and in the decoder conv
  tied_position   as both, but the position attention map is shared by all rows (MSA Transformer
                  tied row attention, Rao et al. 2021): logits summed over the valid input rows
                  and the DNA row, scaled by 1/sqrt(rows), before the softmax

Parameters are matched to `param_budget` (default 1.0e6) by sizing the FFN width after every other
module is built, so every arm lands within the 10 % band of interface C.

Config keys read: context_bins, covariate_injection (concat|film|tokens), mask_mode (drop|token),
dna (own|off|shuffled), count_transform, qc_on_input, param_budget, axis. `scramble_covariates` is
applied by the train loop (dataset.py). `average_input` and `two_step` are not built in this rung:
the constructor raises NotApplicable for them.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

_TOOLS = Path(__file__).resolve().parents[2]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from minicandi.components import (ConvBlock, CountTransform, CovariateInjector,  # noqa: E402
                                  DnaTower, MaskHandler, NBHead, NotApplicable, _check,
                                  sinusoidal_positions)
from minicandi.contract import A, K_META, LOG2_DEPTH_COL, MiniCandi, k_input  # noqa: E402

RUNG = 5
AXES = ("both", "position_only", "track_only", "tied_position")
PATCH = 16          # bins per patch (400 bp)
D_MODEL = 128
N_HEADS = 8
N_LAYERS = 4
D_DNA = 32          # per-bin DNA features
D_DEC = 32          # per-bin decoder features
D_FF_MIN = 64


class AxialAttention(nn.Module):
    """Pre-LN multi-head attention along one axis of x [B, R, N, d], residual added by the caller.

    axis = "position": each row attends over its N columns (no mask; every column is content-free
                       or visible-only by construction). `tie` [B, R] bool: share one map over the
                       rows where tie is True (tied row attention); every row uses that map.
    axis = "row":      each column attends over the R rows; `allow` [B, R, R] bool, True = the
                       query row may read the key row.
    """

    def __init__(self, d: int, heads: int, axis: str):
        super().__init__()
        self.axis, self.h, self.dh = axis, heads, d // heads
        self.norm = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x: torch.Tensor, allow: torch.Tensor | None = None,
                tie: torch.Tensor | None = None) -> torch.Tensor:
        B, R, N, d = x.shape
        q, k, v = self.qkv(self.norm(x)).view(B, R, N, 3, self.h, self.dh).unbind(3)
        if self.axis == "position" and tie is not None:
            w = tie.to(x.dtype)[:, :, None, None, None]                     # [B, R, 1, 1, 1]
            n_rows = tie.sum(1).clamp_min(1).to(x.dtype)[:, None, None, None]
            logits = torch.einsum("brnhe,brmhe->bhnm", q * w, k)
            logits = logits / (math.sqrt(self.dh) * n_rows.sqrt())
            att = logits.softmax(-1)
            o = torch.einsum("bhnm,brmhe->brnhe", att, v)
        elif self.axis == "position":
            qp, kp, vp = (t.reshape(B * R, N, self.h, self.dh).transpose(1, 2) for t in (q, k, v))
            o = F.scaled_dot_product_attention(qp, kp, vp)
            o = o.transpose(1, 2).reshape(B, R, N, self.h, self.dh)
        else:
            qr, kr, vr = (t.permute(0, 2, 3, 1, 4).reshape(B * N, self.h, R, self.dh)
                          for t in (q, k, v))
            m = allow[:, None, None].expand(B, N, 1, R, R).reshape(B * N, 1, R, R)
            o = F.scaled_dot_product_attention(qr, kr, vr, attn_mask=m)
            o = o.reshape(B, N, self.h, R, self.dh).permute(0, 3, 1, 2, 4)
        return self.out(o.reshape(B, R, N, d))


class FFN(nn.Module):
    def __init__(self, d: int, d_ff: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d_ff), nn.GELU(),
                                 nn.Linear(d_ff, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Model(MiniCandi):
    """Rung 5. `Model(config)`; forward(batch) -> {"n", "mu"} [B, L, A], both > 0."""

    def __init__(self, config: dict):
        super().__init__()
        cfg = dict(config or {})
        self.config = cfg
        if cfg.get("average_input"):
            raise NotApplicable("rung 5 does not build the average_input channel")
        if cfg.get("two_step"):
            raise NotApplicable("rung 5 does not build the two-step (C_base) head")
        self.inj_mode = _check(cfg.get("covariate_injection", "tokens"),
                               ("concat", "film", "tokens"), "covariate_injection")
        self.mask_mode = _check(cfg.get("mask_mode", "drop"), ("token", "drop"), "mask_mode")
        self.axis = _check(cfg.get("axis", "both"), AXES, "axis")
        self.dna_mode = cfg.get("dna", "own")
        self.L = int(cfg.get("context_bins", 1024))
        if self.L % PATCH:
            raise ValueError(f"context_bins {self.L} is not a multiple of the patch {PATCH}")
        d = D_MODEL
        self.n_patch = self.L // PATCH
        k_in = k_input(cfg)

        # inputs
        self.transform = CountTransform(cfg.get("count_transform", "offset"))
        self.mask = MaskHandler(self.mask_mode, 2)
        self.patch_in = nn.Linear(2 * PATCH, d)
        self.dna = DnaTower(self.dna_mode, D_DNA)
        if self.dna_mode != "off":
            self.patch_dna = nn.Linear(D_DNA * PATCH, d)
            self.dna_skip = nn.Linear(D_DNA, D_DEC)
        self.row_in = nn.Parameter(torch.randn(A, d) * 0.02)       # input track rows
        self.row_dna = nn.Parameter(torch.randn(d) * 0.02)
        self.row_out = nn.Parameter(torch.randn(A, d) * 0.02)      # query rows
        if self.mask_mode == "token":
            self.mask_row = nn.Parameter(torch.randn(d) * 0.02)
        self.register_buffer("pos", sinusoidal_positions(self.n_patch, d), persistent=False)

        # covariates: C on input rows, C' on query rows
        self.cov_in = CovariateInjector(self.inj_mode, d, k_in)
        self.cov_out = CovariateInjector(self.inj_mode, d, K_META)
        if self.inj_mode == "tokens":
            self.dna_cov_token = nn.Parameter(torch.randn(d) * 0.02)

        # attention stack
        self.layers = nn.ModuleList()
        for _ in range(N_LAYERS):
            blk = nn.ModuleDict()
            if self.axis != "track_only":
                blk["pos"] = AxialAttention(d, N_HEADS, "position")
            if self.axis != "position_only":
                blk["row"] = AxialAttention(d, N_HEADS, "row")
            self.layers.append(blk)
        self.readout = AxialAttention(d, N_HEADS, "row") if self.axis == "position_only" else None

        # decoder
        self.dec_norm = nn.LayerNorm(d)
        self.unpatch = nn.Linear(d, PATCH * D_DEC)
        self.refine = ConvBlock(D_DEC, D_DEC, kernel=5)
        self.head = NBHead(D_DEC, self.transform)

        # FFN width from the parameter budget: every FFN adds d_ff (2d + 1) + 3d parameters
        budget = float(cfg.get("param_budget", 1.0e6))
        p0 = self.param_count()
        per = N_LAYERS * (2 * d + 1)
        self.d_ff = max(D_FF_MIN, int(round((budget - p0 - N_LAYERS * 3 * d) / per)))
        for blk in self.layers:
            blk["ffn"] = FFN(d, self.d_ff)

    # ------------------------------------------------------------------------------------------

    def _rows(self, batch: dict):
        """-> grid x [B, R, N(+1), d], row-attention allow [B, R, R], tie rows [B, R], dna [B, L, D_DNA]."""
        xc, avail, bmask = batch["x_counts"], batch["x_avail"].bool(), batch["x_bin_mask"].bool()
        B, L, _ = xc.shape
        if L != self.L:
            raise ValueError(f"window has {L} bins; the model was built for {self.L}")
        N, d = self.n_patch, D_MODEL
        vis = MaskHandler.visible(avail, bmask)
        z = self.transform(xc, batch["c_in"][..., LOG2_DEPTH_COL], vis)            # [B, L, A]
        feat = torch.stack([z, torch.ones_like(z)], dim=-1)                        # [B, L, A, 2]
        feat, _ = self.mask(feat, avail, bmask)        # hidden and missing replaced, never read
        feat = feat.view(B, N, PATCH, A, 2).permute(0, 3, 1, 2, 4).reshape(B, A, N, 2 * PATCH)
        h_in = self.patch_in(feat) + self.row_in[None, :, None] + self.pos
        if self.mask_mode == "token":
            mrow = (self.mask_row + self.row_in)[None, :, None] + self.pos         # [1, A, N, d]
            h_in = torch.where(avail[:, :, None, None], h_in, mrow.expand_as(h_in))
        else:
            h_in = torch.where(avail[:, :, None, None], h_in, torch.zeros_like(h_in))

        dna = self.dna(batch["x_dna"])                                             # [B, L, D_DNA]
        h_dna = (self.row_dna + self.pos).expand(B, 1, N, d)
        if self.dna_mode != "off":
            h_dna = h_dna + self.patch_dna(dna.reshape(B, N, PATCH * D_DNA))[:, None]
        h_out = (self.row_out[:, None] + self.pos).expand(B, A, N, d)

        c_in, c_out = batch["c_in"], batch["c_out"]
        if self.inj_mode == "tokens":
            t_in = self.cov_in.tokens(c_in)[:, :, None]                            # [B, A, 1, d]
            t_in = torch.where(avail[:, :, None, None], t_in, torch.zeros_like(t_in))
            h_in = torch.cat([t_in, h_in], dim=2)
            h_dna = torch.cat([self.dna_cov_token.expand(B, 1, 1, d), h_dna], dim=2)
            h_out = torch.cat([self.cov_out.tokens(c_out)[:, :, None], h_out], dim=2)
        else:
            h_in = self.cov_in(h_in, c_in[:, :, None])
            if self.mask_mode == "drop":
                h_in = torch.where(avail[:, :, None, None], h_in, torch.zeros_like(h_in))
            h_out = self.cov_out(h_out, c_out[:, :, None])
        x = torch.cat([h_in, h_dna, h_out], dim=1)                                 # [B, R, N', d]

        R = 2 * A + 1
        key_in = avail if self.mask_mode == "drop" else torch.ones_like(avail)
        ones = torch.ones(B, 1, dtype=torch.bool, device=avail.device)
        key = torch.cat([key_in, ones, torch.zeros(B, A, dtype=torch.bool,
                                                   device=avail.device)], dim=1)   # [B, R]
        eye = torch.eye(R, dtype=torch.bool, device=avail.device)
        allow = key[:, None, :] | eye[None]
        tie = torch.cat([key_in, ones, torch.zeros_like(key_in)], dim=1)
        return x, allow, tie, dna

    def forward(self, batch: dict) -> dict:
        x, allow, tie, dna = self._rows(batch)
        tied = tie if self.axis == "tied_position" else None
        for blk in self.layers:
            if "pos" in blk:
                x = x + blk["pos"](x, tie=tied)
            if "row" in blk:
                x = x + blk["row"](x, allow=allow)
            x = x + blk["ffn"](x)
        if self.readout is not None:
            x = x + self.readout(x, allow=allow)
        B = x.shape[0]
        q = x[:, A + 1:, -self.n_patch:]                                           # [B, A, N, d]
        f = self.unpatch(self.dec_norm(q)).view(B, A, self.L, D_DEC)
        if self.dna_mode != "off":
            f = f + self.dna_skip(dna)[:, None]
        f = self.refine(f.reshape(B * A, self.L, D_DEC).transpose(1, 2))
        f = f.transpose(1, 2).reshape(B, A, self.L, D_DEC).transpose(1, 2)        # [B, L, A, D]
        return self.head(f, batch["c_out"])


def build(config: dict) -> Model:
    return Model(config)


__all__ = ["AXES", "AxialAttention", "Model", "NotApplicable", "PATCH", "RUNG", "build"]
