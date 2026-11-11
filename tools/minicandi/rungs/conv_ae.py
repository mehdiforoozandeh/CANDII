"""miniCANDI rung 2: the convolutional autoencoder (plan chunk R2).

Conv towers downsample the counts and the DNA to a coarse latent; a transposed-conv decoder
upsamples back to full resolution. No skip connection (that is rung 4), so the latent is the only
path from input to output, and context is local with a receptive field that grows with depth.

Data flow, shapes for B windows, L bins, A = 36 tracks:

  x_counts [B,L,A] --CountTransform(c_in depth, visible)--> [B,L,A]
    --per-track lift (Linear(1,d_t) per track, + log1p(avg) lift if `average_input`)--> [B,L,A,d_t]
    --MaskHandler(mask_mode)--> hidden bins and missing tracks replaced (token) or zeroed (drop)
    --CovariateInjector(covariate_injection) with C = c_in, per track--> [B,L,A,d_t]
    (drop: invisible positions re-zeroed after injection, plus one visibility channel per track)
  flatten tracks into channels, concat the DnaTower(dna) features [B,L,d_dna]
    --stem Conv1d(k=1): the one linear fusion of tracks--> [B,W,L]
    --encoder: N_LEVELS x (2 ConvBlocks, Downsample x2)--> bottleneck ConvBlocks at L / 2**N_LEVELS
    --decoder: N_LEVELS x (Upsample x2, 2 ConvBlocks)--> [B,W,L]
  --Linear(W, A*d_o) + track embedding--> [B,L,A,d_o]
    --CovariateInjector(covariate_injection) with C' = c_out, per track--> MLP --> NBHead
  NBHead decodes mu through the transform's depth add-back at the target's own log2 depth.

Config keys read here: `covariate_injection` (concat | film; `tokens` raises NotApplicable:
no attention reads tokens), `mask_mode` (token | drop), `dna` (own | off | shuffled),
`count_transform`, `scramble_covariates`, `qc_on_input` (K of c_in), `average_input` (needs the
batch key `avg`), `param_budget` (the conv width W is chosen so that the parameter count is the
closest to it; default 1.0e6). Keys the dataset or the train loop handle (`context_bins`,
`control_input`, `predict_control`, `two_step`, `base_depth`, `windows_seen`, `lr`,
`batch_size`, `simpler_than_base`) do not change the network.

`scramble_covariates` permutes the covariate rows fed to the two covariate embeddings across the
windows of the batch (`covariates.scramble_covariates`; seeded in eval mode, fresh per call in
training). The count transform keeps each track's own log2 depth on both sides, so the scrambled
control removes what the network learns from C and C', not the size-factor normalisation.

Guardrail part 1 holds by construction: the only reads of `x_counts` go through the
count transform, whose window statistics use visible bins only, and the MaskHandler replaces every
hidden or missing position before any cross-bin or cross-track mixing.

Registration: `tools/minicandi/registry.py` imports this module lazily by name (`conv_ae`) and
builds `Model(config)`. This module never imports the registry, so it imports without it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

_TOOLS = Path(__file__).resolve().parents[2]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from minicandi import contract as C  # noqa: E402
from minicandi.components import (ConvBlock, CountTransform, CovariateInjector,  # noqa: E402
                                  DnaTower, Downsample, MaskHandler, NBHead, NotApplicable,
                                  Upsample)
from minicandi.covariates import scramble_covariates  # noqa: E402

RUNG = 2
NAME = "conv_ae"
DEFAULTS = {"covariate_injection": "film", "mask_mode": "token", "dna": "own",
            "count_transform": "offset", "scramble_covariates": False, "qc_on_input": False,
            "average_input": False, "param_budget": 1.0e6}
D_TRACK = 16          # per-track input features d_t
D_OUT = 16            # per-track output features d_o
D_DNA = 32            # DNA tower output channels
D_COV = 32            # covariate embedding width
N_LEVELS = 4          # downsampling steps of 2: the latent is at L / 16
KERNEL = 5
WIDTH_STEP = 4        # W and every level width are multiples of 4
WIDTH_RANGE = (16, 1024)
SCRAMBLE_SEED = 0


def _level_widths(W: int) -> list[int]:
    """Channels per encoder level: W, 1.25 W, 1.5 W, ... rounded to a multiple of WIDTH_STEP."""
    return [max(WIDTH_STEP, int(round(W * (1 + 0.25 * i) / WIDTH_STEP)) * WIDTH_STEP)
            for i in range(N_LEVELS + 1)]


class Model(C.MiniCandi):
    """Rung 2. `Model(config)`; `forward(batch) -> {"n", "mu"}` with both [B, L, A] > 0."""

    def __init__(self, config: dict | None = None, width: int | None = None):
        super().__init__()
        cfg = {**DEFAULTS, **(config or {})}
        if cfg.get("rung", RUNG) != RUNG:
            raise ValueError(f"conv_ae is rung {RUNG}; config has rung {cfg['rung']}")
        if cfg["covariate_injection"] == "tokens":
            raise NotApplicable("rung 2 (conv autoencoder) has no attention to read covariate "
                                "tokens; covariate_injection = tokens does not apply")
        self.config = cfg
        if width is None:
            width = choose_width(cfg)
        self.width = int(width)
        self._build(cfg, self.width)

    # ------------------------------------------------------------------ construction

    def _build(self, cfg: dict, W: int) -> None:
        A = C.A
        inj, mask_mode = cfg["covariate_injection"], cfg["mask_mode"]
        self.mask_mode = mask_mode
        self.scramble = bool(cfg["scramble_covariates"])
        self.use_avg = bool(cfg["average_input"])
        k_in = C.k_input(cfg)

        self.transform = CountTransform(cfg["count_transform"])
        # per-track lift: Linear(1, d_t) with its own weight and bias per track
        self.lift_w = nn.Parameter(torch.randn(A, D_TRACK) * 0.5)
        self.lift_b = nn.Parameter(torch.zeros(A, D_TRACK))
        if self.use_avg:
            self.avg_w = nn.Parameter(torch.randn(A, D_TRACK) * 0.5)
        self.mask = MaskHandler(mask_mode, D_TRACK)
        self.inj_in = CovariateInjector(inj, D_TRACK, k_in, D_COV)
        self.dna = DnaTower(cfg["dna"], D_DNA)

        widths = _level_widths(W)
        c_stem = A * D_TRACK + (A if mask_mode == "drop" else 0) \
            + (D_DNA if cfg["dna"] != "off" else 0)
        self.stem = nn.Conv1d(c_stem, widths[0], 1)
        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        for i in range(N_LEVELS):
            self.enc.append(nn.ModuleList([ConvBlock(widths[i], widths[i], KERNEL),
                                           ConvBlock(widths[i], widths[i], KERNEL, dilation=2)]))
            self.down.append(Downsample(widths[i], widths[i + 1]))
        wb = widths[N_LEVELS]
        self.mid = nn.ModuleList([ConvBlock(wb, wb, KERNEL, dilation=2),
                                  ConvBlock(wb, wb, KERNEL, dilation=4)])
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in reversed(range(N_LEVELS)):
            self.up.append(Upsample(widths[i + 1], widths[i]))
            self.dec.append(nn.ModuleList([ConvBlock(widths[i], widths[i], KERNEL),
                                           ConvBlock(widths[i], widths[i], KERNEL, dilation=2)]))

        self.out_proj = nn.Linear(widths[0], A * D_OUT)
        self.out_track = nn.Parameter(torch.randn(A, D_OUT) * 0.02)
        self.inj_out = CovariateInjector(inj, D_OUT, C.K_META, D_COV)
        self.out_mlp = nn.Sequential(nn.GELU(), nn.Linear(D_OUT, D_OUT), nn.GELU())
        self.head = NBHead(D_OUT, self.transform)

    # ------------------------------------------------------------------ forward

    def _covariates(self, batch: dict) -> tuple:
        c_in, c_out = batch["c_in"], batch["c_out"]
        if self.scramble:
            gen = None if self.training else torch.Generator().manual_seed(SCRAMBLE_SEED)
            c_in = scramble_covariates(c_in.cpu(), batch["x_avail"].cpu(), gen).to(c_in.device)
            c_out = scramble_covariates(c_out.cpu(), batch["y_avail"].cpu(), gen) \
                .to(c_out.device)
        return c_in, c_out

    def forward(self, batch: dict) -> dict:
        x_counts, x_avail, bmask = batch["x_counts"], batch["x_avail"], batch["x_bin_mask"]
        B, L, A = x_counts.shape
        vis = MaskHandler.visible(x_avail, bmask)                         # [B, L, A]
        # depth removal uses each track's TRUE depth (never scrambled)
        x = self.transform(x_counts, batch["c_in"][..., C.LOG2_DEPTH_COL], vis)
        h = x[..., None] * self.lift_w + self.lift_b                     # [B, L, A, d_t]
        h, vis = self.mask(h, x_avail, bmask)
        c_in, c_out = self._covariates(batch)
        h = self.inj_in(h, c_in[:, None])
        if self.use_avg:
            if "avg" not in batch:
                raise ValueError("average_input is set but the batch has no `avg`")
            h = h + torch.log1p(batch["avg"].float().clamp_min(0.0))[..., None] * self.avg_w
        if self.mask_mode == "drop":
            h = torch.where(vis[..., None], h, torch.zeros_like(h))
        feats = [h.reshape(B, L, A * D_TRACK)]
        if self.mask_mode == "drop":
            feats.append(vis.to(h.dtype))
        if self.dna.mode != "off":
            feats.append(self.dna(batch["x_dna"]).to(h.dtype))
        z = torch.cat(feats, dim=-1).transpose(1, 2)                       # [B, C, L]

        f = 2 ** N_LEVELS
        pad = (-L) % f
        if pad:
            z = F.pad(z, (0, pad))
        z = self.stem(z)
        for blocks, down in zip(self.enc, self.down):
            for blk in blocks:
                z = blk(z)
            z = down(z)
        for blk in self.mid:
            z = blk(z)
        for up, blocks in zip(self.up, self.dec):
            z = up(z)
            for blk in blocks:
                z = blk(z)
        z = z[..., :L].transpose(1, 2)                                     # [B, L, W]

        o = self.out_proj(z).reshape(B, L, A, D_OUT) + self.out_track
        o = self.inj_out(o, c_out[:, None])
        o = self.out_mlp(o)
        return self.head(o, batch["c_out"])


def choose_width(config: dict) -> int:
    """The multiple of WIDTH_STEP in WIDTH_RANGE whose parameter count is the closest to
    `config["param_budget"]` (the count rises with the width; bisection). Built under a forked
    RNG, so the search does not move the global torch seed."""
    cfg = {**DEFAULTS, **(config or {})}
    budget = float(cfg["param_budget"])

    def count(w: int) -> int:
        with torch.random.fork_rng(devices=[]):
            return Model(cfg, width=w).param_count()

    lo, hi = WIDTH_RANGE[0] // WIDTH_STEP, WIDTH_RANGE[1] // WIDTH_STEP
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if count(mid * WIDTH_STEP) <= budget:
            lo = mid
        else:
            hi = mid
    return min((lo * WIDTH_STEP, hi * WIDTH_STEP), key=lambda w: abs(count(w) - budget))


__all__ = ["DEFAULTS", "Model", "NAME", "NotApplicable", "RUNG", "choose_width"]
