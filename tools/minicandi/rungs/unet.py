"""miniCANDI rung 4: U-Net (rung 3 plus skips from each encoder level to the matching decoder level).

Layout of one forward pass (B windows, L bins, A = 36 tracks):

  per-track input   x = CountTransform(x_counts, log2 depth of C, visible)          [B, L, A]
                    h = MLP(x) + track embedding; input covariates C by `concat`/`film`
                    h = MaskHandler(h)   <- every hidden bin and every unavailable track is
                        replaced HERE, before the first conv and before any skip is taken
                    h = cat(h, visible flag)                                         [B, L, A, d+1]
  count tower       fuse tracks (1x1 conv over A*(d+1)) -> n_levels x (ConvBlock, Downsample)
  DNA tower         DnaTower(own|shuffled) -> the same level stack on its own weights (absent
                    when dna = off)
  bottleneck        cat(count, DNA) -> 1x1 conv -> transformer encoder on the coarse sequence
                    (sinusoidal positions; under `tokens` injection the C and C' tokens join the
                    sequence, a token of an absent track is a padded key)
  decoder           per level: Upsample, cat(skip features of the chosen source), 2 ConvBlocks.
                    The output covariates steer every skip: a FiLM (gamma, beta) from the pooled
                    C' embedding of the windows' targets (identity at initialisation).
  head              decoder features broadcast to every track + track embedding; with count skips
                    the track's own MASKED per-track input features join at full resolution;
                    C' per track by `concat`/`film` (or the attended C' token under `tokens`);
                    NBHead decodes at the target's own depth (C' column 0).

Config keys read: `skip_source` (dna | counts | both; base both), `covariate_injection`,
`mask_mode`, `dna`, `count_transform`, `qc_on_input`, `average_input`, `param_budget`, and the
optional `width` (channels of the finest level; when absent the constructor picks the width whose
parameter count is closest to `param_budget`), `scramble_covariates` and `scramble_all`
(`model_covariates`, the rule shared by all six rungs). Keys that act outside the model
(base_depth, control_input, predict_control, windows_seen, lr, batch_size) are accepted and
ignored here.

Not applicable (`NotApplicable`): `skip_source = dna` with `dna = off` (no DNA skip exists), and
`two_step = true`, as in every rung.
`skip_source = both` with `dna = off` keeps only the count skips; that is the `dna_off` arm of the
base.

Component arm keys (pinned interface E, chunk A3) reach every component through
`component_kwargs(cfg)`: `covariate_injection = adaln_zero`, `covariate_dropout` (passed to
`model_covariates`), `covariate_embedding`, `dna = precomputed` (the DnaTower reads
`batch["x_dna_emb"]`; needs `dna_embedding_dim`), `offset_eps`, `count_transform = soft_clip` and
`head = nb_shared_size`. With every key at its default the model is today's, parameter for
parameter and output for output (tests/test_minicandi_rung_arms.py). `dna = precomputed` feeds the DNA skips too.

A window whose length is not a multiple of 2**n_levels is zero-padded at its right end inside the
network (features already masked) and the output is cropped back to L.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

_TOOLS = Path(__file__).resolve().parents[2]
for _p in (_TOOLS / "t118", _TOOLS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ladder.model import G  # noqa: E402
from minicandi.components import (ConvBlock, CountTransform, CovariateInjector,  # noqa: E402
                                  DnaTower, Downsample, MaskHandler, NBHead, NotApplicable,
                                  Upsample, component_kwargs, sinusoidal_positions)
from minicandi.contract import A, K_META, LOG2_DEPTH_COL, MiniCandi, k_input  # noqa: E402
from minicandi.covariates import model_covariates, scramble_mode  # noqa: E402

RUNG = 4
NAME = "unet"
SKIP_SOURCES = ("dna", "counts", "both")
N_LEVELS = 4                 # stride 16 at the bottleneck
LEVEL_MULT = (1, 2, 2, 4, 4)  # channels per level as multiples of `width` (level N_LEVELS = bottom)
D_TRACK = 8                  # per-track input feature width
D_HEAD = 32                  # per-track head width
D_COV = 32                   # covariate embedding width
N_HEADS = 4
N_ATTN_LAYERS = 2
WIDTH_STEP = 1                # 4*width (the attention width) is always divisible by N_HEADS
DEFAULTS = {"skip_source": "both", "covariate_injection": "film", "mask_mode": "token",
            "dna": "own", "count_transform": "offset", "qc_on_input": False,
            "average_input": False, "param_budget": 1.0e6}


# ---------------------------------------------------------------------------------------------
# Rule shared by all six rungs, kept identical in each module (tests/test_minicandi_all_rungs.py
# proves it); the scramble rule is `covariates.model_covariates`. Registration: the registry imports
# this module by name and builds `Model(config)`.
# ---------------------------------------------------------------------------------------------
TWO_STEP_MESSAGE = ("two_step (D15) is not built in any rung: two-step correction is a later "
                    "ablation on the winning design")


def check_rung(cfg: dict) -> None:
    """Refuse a config that names another rung (`rung` may be the number or the module name)."""
    r = cfg.get("rung", RUNG)
    if r not in (RUNG, str(RUNG), NAME):
        raise ValueError(f"{NAME} is rung {RUNG}; config has rung {r!r}")
    if cfg.get("two_step"):
        raise NotApplicable(TWO_STEP_MESSAGE)


def _cfg(config: dict) -> dict:
    out = dict(DEFAULTS)
    out.update({k: v for k, v in (config or {}).items() if v is not None})
    return out


class _Tower(nn.Module):
    """[N, C0, L] -> skips [level i: [N, C_i, L / 2**i]] and the bottom [N, C_n, L / 2**n]."""

    def __init__(self, chans: tuple):
        super().__init__()
        self.blocks = nn.ModuleList(ConvBlock(chans[i], chans[i]) for i in range(N_LEVELS))
        self.down = nn.ModuleList(Downsample(chans[i], chans[i + 1]) for i in range(N_LEVELS))

    def forward(self, x: torch.Tensor):
        skips = []
        for blk, down in zip(self.blocks, self.down):
            x = blk(x)
            skips.append(x)
            x = down(x)
        return skips, x


class Model(MiniCandi):
    """Rung 4 U-Net; `forward(batch) -> {"n", "mu"}` per pinned interface C."""

    rung = RUNG

    def __init__(self, config: dict):
        super().__init__()
        cfg = _cfg(config)
        _validate(cfg)
        self.config = dict(config or {})
        width = cfg.get("width")
        if width is None:
            width = fit_width(cfg)
        self._build(cfg, int(width))
        budget = float(cfg["param_budget"])
        if "width" not in (config or {}) and abs(self.param_count() - budget) > 0.10 * budget:
            raise ValueError(f"rung 4: no width brings the parameter count within 10% of "
                             f"param_budget {budget:.0f} (got {self.param_count()})")
        # extra inputs (chunk XI), constructed last (after the budget check, which they do not
        # enter); dna = off leaves no DNA features
        self.build_extras(cfg, self.chans[0] if self.dna_mode != "off" else None,
                          ("in_inj", "out_inj"))

    # ------------------------------------------------------------------ construction
    def _build(self, cfg: dict, width: int) -> None:
        if width % WIDTH_STEP or width <= 0:
            raise ValueError(f"width {width} must be a positive multiple of {WIDTH_STEP}")
        self.width = width
        self.skip_source = cfg["skip_source"]
        self.injection = cfg["covariate_injection"]
        self.dna_mode = cfg["dna"]
        self.average_input = bool(cfg["average_input"])
        self.scramble = scramble_mode(cfg)
        self.k_in = k_input(cfg)
        kw = component_kwargs(cfg)
        self.cov_dropout = kw["covariate_dropout"]
        self.skip_counts = self.skip_source in ("counts", "both")
        self.skip_dna = self.skip_source in ("dna", "both") and self.dna_mode != "off"
        chans = tuple(width * m for m in LEVEL_MULT[:N_LEVELS + 1])
        self.chans = chans
        tokens = self.injection == "tokens"
        d_model = chans[-1]

        # per-track input path
        self.transform = CountTransform(cfg["count_transform"], **kw["transform"])
        self.track_in = nn.Sequential(nn.Linear(1, D_TRACK), nn.GELU(),
                                      nn.Linear(D_TRACK, D_TRACK))
        self.track_emb_in = nn.Parameter(torch.randn(A, D_TRACK) * 0.02)
        if self.average_input:
            self.avg_in = nn.Linear(1, D_TRACK)
        self.in_inj = CovariateInjector(self.injection, d_model if tokens else D_TRACK,
                                        self.k_in, D_COV, **kw["injector"])
        self.mask = MaskHandler(cfg["mask_mode"], D_TRACK)

        # towers
        self.stem = nn.Conv1d(A * (D_TRACK + 1), chans[0], 1)
        self.count_tower = _Tower(chans)
        if self.dna_mode != "off":
            self.dna = DnaTower(self.dna_mode, chans[0], **kw["dna"])
            self.dna_tower = _Tower(chans)
            self.fuse = nn.Conv1d(2 * d_model, d_model, 1)
        else:
            self.dna = DnaTower("off", chans[0])

        # bottleneck attention
        layer = nn.TransformerEncoderLayer(d_model, N_HEADS, 2 * d_model, dropout=0.0,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.attn = nn.TransformerEncoder(layer, N_ATTN_LAYERS, enable_nested_tensor=False)

        # decoder with skips steered by the output covariates
        self.out_inj = CovariateInjector(self.injection, d_model if tokens else D_HEAD, K_META,
                                         D_COV, **kw["injector"])
        if tokens:
            self.tok_to_cov = nn.Linear(d_model, D_COV)
            self.tok_to_head = nn.Linear(d_model, D_HEAD)
        skip_w = [chans[i] * (int(self.skip_counts) + int(self.skip_dna)) for i in range(N_LEVELS)]
        self.up = nn.ModuleList(Upsample(chans[i + 1], chans[i]) for i in range(N_LEVELS))
        self.steer = nn.ModuleList(
            G(D_COV, 2 * skip_w[i], torch.cat([torch.ones(skip_w[i]), torch.zeros(skip_w[i])]))
            for i in range(N_LEVELS))
        self.dec1 = nn.ModuleList(ConvBlock(chans[i] + skip_w[i], chans[i])
                                  for i in range(N_LEVELS))
        self.dec2 = nn.ModuleList(ConvBlock(chans[i], chans[i]) for i in range(N_LEVELS))

        # head
        self.to_head = nn.Linear(chans[0], D_HEAD)
        self.track_emb_out = nn.Parameter(torch.randn(A, D_HEAD) * 0.02)
        if self.skip_counts:
            self.head_skip = nn.Linear(D_HEAD + D_TRACK + 1, D_HEAD)
        self.head_mlp = nn.Sequential(nn.GELU(), nn.Linear(D_HEAD, D_HEAD), nn.GELU())
        self.nb = NBHead(D_HEAD, self.transform, **kw["head"])

    # ------------------------------------------------------------------ forward
    def forward(self, batch: dict) -> dict:
        self.begin_extras(batch)
        x_counts, x_avail = batch["x_counts"], batch["x_avail"].bool()
        x_bin_mask = batch["x_bin_mask"].bool()
        c_in, c_out, c_in_t, c_out_t = model_covariates(batch, self.scramble, self.training,
                                                         self.cov_dropout)
        c_in, c_out = c_in.float(), c_out.float()
        y_avail = batch["y_avail"].bool()
        B, L, _ = x_counts.shape
        tokens = self.injection == "tokens"

        # per-track input features; the mask is applied before anything mixes bins or tracks
        vis = MaskHandler.visible(x_avail, x_bin_mask)
        x = self.transform(x_counts, c_in_t.float()[..., LOG2_DEPTH_COL], vis)
        h = self.track_in(x[..., None]) + self.track_emb_in
        if not tokens:
            h = self.in_inj(h, c_in[:, None])
        h, vis = self.mask(h, x_avail, x_bin_mask)
        if self.average_input:
            h = h + self.avg_in(torch.log1p(batch["avg"].float().clamp_min(0))[..., None])
        h_track = torch.cat([h, vis[..., None].to(h.dtype)], dim=-1)      # [B, L, A, d+1]

        # pad L to a multiple of 2**N_LEVELS (zeros carry no input value)
        stride = 2 ** N_LEVELS
        Lp = -(-L // stride) * stride
        fused = h_track.reshape(B, L, -1).transpose(1, 2)                   # [B, A*(d+1), L]
        fused = F.pad(fused, (0, Lp - L))
        c_skips, bottom = self.count_tower(self.stem(fused))
        d_skips = None
        if self.dna_mode != "off":
            x_dna = batch["x_dna_emb" if self.dna_mode == "precomputed" else "x_dna"]
            dna = F.pad(self.add_extra_track(self.dna(x_dna), batch).transpose(1, 2),
                        (0, Lp - L))
            d_skips, d_bottom = self.dna_tower(dna)
            bottom = self.fuse(torch.cat([bottom, d_bottom], dim=1))

        # bottleneck attention, with covariate tokens under `tokens`
        seq = bottom.transpose(1, 2)                                        # [B, Lb, d_model]
        Lb = seq.shape[1]
        seq = seq + sinusoidal_positions(Lb, seq.shape[-1], seq.device)
        tok_out = None
        if tokens:
            t_in, t_out = self.in_inj.tokens(c_in), self.out_inj.tokens(c_out)
            seq = torch.cat([seq, t_in, t_out], dim=1)
            pad = torch.cat([torch.zeros(B, Lb, dtype=torch.bool, device=seq.device),
                             ~x_avail, ~y_avail], dim=1)
            seq = self.attn(seq, src_key_padding_mask=pad)
            tok_out = seq[:, Lb + A:]                                        # [B, A, d_model]
            seq = seq[:, :Lb]
        else:
            seq = self.attn(seq)
        z = seq.transpose(1, 2)

        # global output condition that steers the skips
        w = y_avail.float()[..., None]
        if tokens:
            per = self.tok_to_cov(tok_out)
        else:
            per = self.out_inj.embed(c_out)
        g_out = (per * w).sum(1) / w.sum(1).clamp_min(1.0)                  # [B, D_COV]

        for i in reversed(range(N_LEVELS)):
            z = self.up[i](z)
            parts = []
            if self.skip_counts:
                parts.append(c_skips[i])
            if self.skip_dna:
                parts.append(d_skips[i])
            if parts:
                s = torch.cat(parts, dim=1)
                theta = self.steer[i](g_out)
                k = s.shape[1]
                s = theta[:, :k, None] * s + theta[:, k:, None]
                z = torch.cat([z, s], dim=1)
            z = self.dec2[i](self.dec1[i](z))
        z = z[..., :L].transpose(1, 2)                                      # [B, L, C0]

        # per-track head
        hd = self.to_head(z)[:, :, None, :] + self.track_emb_out          # [B, L, A, D_HEAD]
        if self.skip_counts:
            hd = self.head_skip(torch.cat([hd, h_track], dim=-1))
        if tokens:
            hd = hd + self.tok_to_head(tok_out)[:, None]
        else:
            hd = self.out_inj(hd, c_out[:, None])
        return self.nb(self.head_mlp(hd), c_out_t.float())


def _validate(cfg: dict) -> None:
    check_rung(cfg)
    if cfg["skip_source"] not in SKIP_SOURCES:
        raise ValueError(f"skip_source {cfg['skip_source']!r} not in {SKIP_SOURCES}")
    if cfg["skip_source"] == "dna" and cfg["dna"] == "off":
        raise NotApplicable("rung 4: skip_source = dna needs a DNA tower; dna = off has none")


def fit_width(cfg: dict) -> int:
    """Width (multiple of WIDTH_STEP) whose parameter count is closest to `param_budget`.

    The count grows monotonically with the width, so a bisection over width steps suffices.
    """
    cfg = _cfg(cfg)
    budget = float(cfg["param_budget"])

    def count(w: int) -> int:
        m = Model.__new__(Model)
        MiniCandi.__init__(m)
        m._build(cfg, w)
        return m.param_count()

    lo, hi = 1, 2
    while count(hi * WIDTH_STEP) < budget:
        lo, hi = hi, hi * 2
        if hi > 4096:
            raise ValueError(f"rung 4: param_budget {budget:.0f} is out of reach")
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if count(mid * WIDTH_STEP) < budget:
            lo = mid
        else:
            hi = mid
    best = min((lo, hi), key=lambda k: abs(count(k * WIDTH_STEP) - budget))
    return best * WIDTH_STEP


__all__ = ["DEFAULTS", "Model", "NAME", "N_LEVELS", "RUNG", "SKIP_SOURCES", "TWO_STEP_MESSAGE",
           "fit_width"]
