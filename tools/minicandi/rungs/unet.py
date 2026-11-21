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
(`model_covariates`, the rule shared by all six rungs), and `hide_depth_covariate` (rung 4 only,
default false; the `hard_offset` arm, configs/rung4/hard_offset.json): depth leaves C, C' and the
input count transform, and enters only the NB head's offset add-back; `"output"` / `"input"` hide
depth from C' alone / C alone (`model_covariates`). Keys that act outside the
model
(base_depth, control_input, predict_control, windows_seen, lr, batch_size) are accepted and
ignored here.

Round-2 tuning keys (default-off; absent = today's model, parameter for parameter and output for
output, tests/test_minicandi_tune_rung4.py): `n_levels` (2 to 5, default N_LEVELS = 4; the
bottleneck stride is 2**n_levels, channels follow LEVEL_MULT, and the width is refitted to
`param_budget`) and `attn_dropout` (default 0.0; the dropout of the bottleneck transformer layers).

Recipe switches (t370 C1, default = today's model, which builds nothing new):
`film_placement` (ends | encoder | decoder | bottleneck | all, `tools/minicandi/trunk_film.py`):
FiLM from the pooled (C, C') embedding on `count_tower.blocks` (not the DNA tower), on `dec1` and
`dec2`, and/or on the input of each transformer layer; `tokens` with a non-default placement
raises NotApplicable. `skip_film` (cprime | c_cprime): `c_cprime` feeds the pooled (C, C')
vector (width 2 D_COV) to every `steer[i]`. `head_skip` (plain | off | film): `off` removes
`head_skip` and `h_track` does not join the head; `film` modulates the D_TRACK features of
`h_track` per track by a FiLM from cat(C embedding, C' embedding) of that track (identity at the
start; the visibility flag is not modulated). Either non-default with `skip_source = dna` raises
NotApplicable.

Control branch (t370 C15): `control_branch` (shared | separate; default shared = today's model,
which builds nothing). `separate` gives track 35 (the ChIP control) its own copy of `head_mlp` and
of the NB head (`ctl_mlp`, `ctl_nb`, same shapes), built last in `_build` (so the width search
counts them) and fed by the same head features; tracks 0 to 34 keep the shared head.

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

DNA fusion (item A11, `tools/minicandi/dna_fusion.py`): `dna_fusion = film` (the DNA bottleneck
FiLMs the count bottleneck in place of cat + `fuse`), `xattn` (the count bottleneck tokens
cross-attend to the DNA bottleneck tokens in place of cat + `fuse`) or `early` (the DNA features
join the count stem; no DNA tower, no `fuse`, no DNA skip). Under film and xattn the DNA skips stay
and a 1x1 conv (`count_proj`, the count half of `fuse`) stays on the count bottleneck. Every arm
keeps the base width (the search reads the config without `dna_fusion`; the 10% budget check is
not applied to an arm). Absent = the base, bit for bit (tests/test_ablation_dna_fusion.py).

Resolution keys (chunk RM): `res_bp` reaches the DnaTower and the count transform through
`component_kwargs`. `res_in_sub = 5`: the per-track input MLP reads the five 5 bp sub-bin counts
of each 25 bp token (batch key `x_counts_sub` [B, L, A, 5], each through the count transform), its
first layer Linear(5, d) in place of Linear(1, d). `res_out_sub = 5`: after the count skip a
learned sub-bin embedding expands each token's head features to five sub-bins, the head emits
(mu, n) per sub-bin, and the output holds `mu_sub`, `n_sub` [B, L, A, 5] and their pooled NB at
25 bp as `mu`, `n` (`components.sub_output`). At 1 (default) neither builds anything.

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
                                  Upsample, component_kwargs, fine_to_sub, res_subs_of,
                                  sinusoidal_positions, sub_output, sub_to_fine)
from minicandi.contract import (A, CONTROL_INDEX, K_META, LOG2_DEPTH_COL, MiniCandi,  # noqa: E402
                                k_input)
from minicandi.covariates import drop_field_indices, model_covariates, scramble_mode  # noqa: E402
from minicandi.dna_fusion import DnaCrossAttn, DnaFiLM, fusion_of, without_fusion  # noqa: E402
from minicandi.trunk_film import TrunkFiLM, placement_of, pooled_condition, sites_of  # noqa: E402

RUNG = 4
NAME = "unet"
SKIP_SOURCES = ("dna", "counts", "both")
SKIP_FILMS = ("cprime", "c_cprime")      # t370 C1 `skip_film`; cprime = today
HEAD_SKIPS = ("plain", "off", "film")    # t370 C1 `head_skip`; plain = today
CONTROL_BRANCHES = ("shared", "separate")  # t370 C15 `control_branch`; shared = today
N_LEVELS = 4                 # stride 16 at the bottleneck
LEVEL_MULT = (1, 2, 2, 4, 4, 4)  # channels per level as multiples of `width` (level n = bottom)
N_LEVELS_RANGE = (2, 5)      # tuning key `n_levels` (round-2 tuning, default-off)
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
        n = len(chans) - 1
        self.blocks = nn.ModuleList(ConvBlock(chans[i], chans[i]) for i in range(n))
        self.down = nn.ModuleList(Downsample(chans[i], chans[i + 1]) for i in range(n))

    def forward(self, x: torch.Tensor, films: list | None = None):
        skips = []
        for i, (blk, down) in enumerate(zip(self.blocks, self.down)):
            x = blk(x) if films is None else blk(x, films[i])
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
            width = fit_width(without_fusion(cfg))         # A11: the base width in every arm
        self._build(cfg, int(width))
        budget = float(cfg["param_budget"])
        if "width" not in (config or {}) and cfg.get("dna_fusion") is None \
                and abs(self.param_count() - budget) > 0.10 * budget:
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
        self.hide_depth = cfg.get("hide_depth_covariate", False)
        self.k_in = k_input(cfg)
        kw = component_kwargs(cfg)
        self.cov_dropout = kw["covariate_dropout"]
        _drop = drop_field_indices(cfg)   # A12: None unless covariate_drop_fields
        self.cov_drop = {} if _drop is None else {"drop": _drop}   # off: today's call
        self.fusion = fusion_of(cfg, NAME)    # item A11; None = late concatenation + DNA skips
        if self.fusion == "early" and self.skip_source == "dna":
            raise NotApplicable("rung 4: dna_fusion = early has no DNA tower, so no DNA skip")
        self.skip_counts = self.skip_source in ("counts", "both")
        self.skip_dna = self.skip_source in ("dna", "both") and self.dna_mode != "off" \
            and self.fusion != "early"
        # round-2 tuning knobs, default-off (absent = N_LEVELS and dropout 0.0, today's model)
        self.n_levels = nl = int(cfg.get("n_levels", N_LEVELS))
        attn_dropout = float(cfg.get("attn_dropout", 0.0))
        if not N_LEVELS_RANGE[0] <= nl <= N_LEVELS_RANGE[1] or not 0.0 <= attn_dropout < 1.0:
            raise ValueError(f"rung 4: n_levels {nl} not in {N_LEVELS_RANGE} or attn_dropout "
                             f"{attn_dropout} not in [0, 1)")
        chans = tuple(width * m for m in LEVEL_MULT[:nl + 1])
        self.chans = chans
        tokens = self.injection == "tokens"
        d_model = chans[-1]
        # t370 C1 switches (defaults = today's model)
        self.film_sites = sites_of(placement_of(cfg))
        self.skip_film = cfg.get("skip_film", "cprime")
        self.head_skip_mode = cfg.get("head_skip", "plain")

        # per-track input path
        self.transform = CountTransform(cfg["count_transform"], **kw["transform"])
        self.in_sub, self.out_sub = res_subs_of(cfg)
        self.track_in = nn.Sequential(nn.Linear(self.in_sub, D_TRACK), nn.GELU(),
                                      nn.Linear(D_TRACK, D_TRACK))
        self.track_emb_in = nn.Parameter(torch.randn(A, D_TRACK) * 0.02)
        if self.average_input:
            self.avg_in = nn.Linear(1, D_TRACK)
        self.in_inj = CovariateInjector(self.injection, d_model if tokens else D_TRACK,
                                        self.k_in, D_COV, **kw["injector"])
        self.mask = MaskHandler(cfg["mask_mode"], D_TRACK)

        # towers
        early = self.fusion == "early"
        self.stem = nn.Conv1d(A * (D_TRACK + 1) + (chans[0] if early else 0), chans[0], 1)
        self.count_tower = _Tower(chans)
        if early:
            self.dna = DnaTower(self.dna_mode, chans[0], **kw["dna"])
        elif self.dna_mode != "off":
            self.dna = DnaTower(self.dna_mode, chans[0], **kw["dna"])
            self.dna_tower = _Tower(chans)
            if self.fusion is None:
                self.fuse = nn.Conv1d(2 * d_model, d_model, 1)
            else:                    # the count half of `fuse` stays: the arm changes the join
                self.count_proj = nn.Conv1d(d_model, d_model, 1)
                if self.fusion == "film":
                    self.dna_film = DnaFiLM(d_model, d_model)
                else:
                    self.dna_xattn = DnaCrossAttn(d_model, N_HEADS)
        else:
            self.dna = DnaTower("off", chans[0])

        # bottleneck attention
        layer = nn.TransformerEncoderLayer(d_model, N_HEADS, 2 * d_model, dropout=attn_dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.attn = nn.TransformerEncoder(layer, N_ATTN_LAYERS, enable_nested_tensor=False)

        # decoder with skips steered by the output covariates
        self.out_inj = CovariateInjector(self.injection, d_model if tokens else D_HEAD, K_META,
                                         D_COV, **kw["injector"])
        if tokens:
            self.tok_to_cov = nn.Linear(d_model, D_COV)
            self.tok_to_head = nn.Linear(d_model, D_HEAD)
        skip_w = [chans[i] * (int(self.skip_counts) + int(self.skip_dna)) for i in range(nl)]
        self.up = nn.ModuleList(Upsample(chans[i + 1], chans[i]) for i in range(nl))
        d_steer = 2 * D_COV if self.skip_film == "c_cprime" else D_COV
        self.steer = nn.ModuleList(
            G(d_steer, 2 * skip_w[i], torch.cat([torch.ones(skip_w[i]), torch.zeros(skip_w[i])]))
            for i in range(nl))
        self.dec1 = nn.ModuleList(ConvBlock(chans[i] + skip_w[i], chans[i])
                                  for i in range(nl))
        self.dec2 = nn.ModuleList(ConvBlock(chans[i], chans[i]) for i in range(nl))

        # head
        self.to_head = nn.Linear(chans[0], D_HEAD)
        self.track_emb_out = nn.Parameter(torch.randn(A, D_HEAD) * 0.02)
        if self.skip_counts and self.head_skip_mode != "off":
            self.head_skip = nn.Linear(D_HEAD + D_TRACK + 1, D_HEAD)
        self.head_mlp = nn.Sequential(nn.GELU(), nn.Linear(D_HEAD, D_HEAD), nn.GELU())
        self.nb = NBHead(D_HEAD, self.transform, **kw["head"])
        if self.out_sub > 1:
            self.out_sub_emb = nn.Parameter(torch.randn(self.out_sub, D_HEAD) * 0.02)
        # t370 C1 modules, built last so the base modules keep their draws; the defaults build
        # nothing
        if self.head_skip_mode == "film":
            self.head_skip_film = G(2 * D_COV, 2 * D_TRACK,
                                    torch.cat([torch.ones(D_TRACK), torch.zeros(D_TRACK)]))
        if self.film_sites:
            ch = []
            if "encoder" in self.film_sites:
                ch += list(chans[:nl])
            if "bottleneck" in self.film_sites:
                ch += [d_model] * N_ATTN_LAYERS
            if "decoder" in self.film_sites:
                ch += [chans[i] for i in reversed(range(nl)) for _ in range(2)]
            self.trunk_film = TrunkFiLM(2 * D_COV, ch)
        # t370 C15 control branch, built after everything; shared (default) builds nothing
        self.control_branch = cfg.get("control_branch", "shared")
        if self.control_branch == "separate":
            self.ctl_mlp = nn.Sequential(nn.GELU(), nn.Linear(D_HEAD, D_HEAD), nn.GELU())
            self.ctl_nb = NBHead(D_HEAD, self.transform, **kw["head"])

    # ------------------------------------------------------------------ forward
    def forward(self, batch: dict) -> dict:
        self.begin_extras(batch)
        if self.out_sub > 1 and self.training and "y_counts_sub" not in batch:
            raise ValueError("res_out_sub is set and the training batch has no `y_counts_sub`: "
                             "train through tools/minicandi/res_dataset.py, whose loss is at "
                             "the sub-bins")
        x_counts, x_avail = batch["x_counts"], batch["x_avail"].bool()
        x_bin_mask = batch["x_bin_mask"].bool()
        c_in, c_out, c_in_t, c_out_t = model_covariates(batch, self.scramble, self.training,
                                                         self.cov_dropout, self.hide_depth,
                                                         **self.cov_drop)
        c_in, c_out = c_in.float(), c_out.float()
        y_avail = batch["y_avail"].bool()
        B, L, _ = x_counts.shape
        tokens = self.injection == "tokens"

        # per-track input features; the mask is applied before anything mixes bins or tracks
        vis = MaskHandler.visible(x_avail, x_bin_mask)
        if self.in_sub == 1:
            x = self.transform(x_counts, c_in_t.float()[..., LOG2_DEPTH_COL], vis)[..., None]
        else:
            if "x_counts_sub" not in batch:
                raise ValueError("res_in_sub is set but the batch has no `x_counts_sub`")
            x = fine_to_sub(self.transform(sub_to_fine(batch["x_counts_sub"]),
                                           c_in_t.float()[..., LOG2_DEPTH_COL],
                                           vis.repeat_interleave(self.in_sub, dim=1)),
                            self.in_sub)
        h = self.track_in(x) + self.track_emb_in
        if not tokens:
            h = self.in_inj(h, c_in[:, None])
        h, vis = self.mask(h, x_avail, x_bin_mask)
        if self.average_input:
            h = h + self.avg_in(torch.log1p(batch["avg"].float().clamp_min(0))[..., None])
        h_track = torch.cat([h, vis[..., None].to(h.dtype)], dim=-1)      # [B, L, A, d+1]

        # t370 C1: the pooled (C, C') condition of the trunk FiLM and of skip_film = c_cprime
        cond = None
        if self.film_sites or self.skip_film == "c_cprime":
            cond = pooled_condition(self.in_inj.embed(c_in), x_avail,
                                    self.out_inj.embed(c_out), y_avail)
        films = iter(self.trunk_film(cond)) if self.film_sites else iter(())
        enc_f, mid_f, dec_f = (part in self.film_sites
                               for part in ("encoder", "bottleneck", "decoder"))

        # pad L to a multiple of 2**n_levels (zeros carry no input value)
        stride = 2 ** self.n_levels
        Lp = -(-L // stride) * stride
        fused = h_track.reshape(B, L, -1).transpose(1, 2)                   # [B, A*(d+1), L]
        fused = F.pad(fused, (0, Lp - L))
        if self.fusion == "early":                       # DNA joins the counts at the stem
            x_dna = batch["x_dna_emb" if self.dna_mode == "precomputed" else "x_dna"]
            dna = F.pad(self.add_extra_track(self.dna(x_dna), batch).transpose(1, 2),
                        (0, Lp - L))
            fused = torch.cat([fused, dna.to(fused.dtype)], dim=1)
        c_skips, bottom = self.count_tower(
            self.stem(fused), [next(films) for _ in range(self.n_levels)] if enc_f else None)
        d_skips = None
        if self.dna_mode != "off" and self.fusion != "early":
            x_dna = batch["x_dna_emb" if self.dna_mode == "precomputed" else "x_dna"]
            dna = F.pad(self.add_extra_track(self.dna(x_dna), batch).transpose(1, 2),
                        (0, Lp - L))
            d_skips, d_bottom = self.dna_tower(dna)
            if self.fusion is None:
                bottom = self.fuse(torch.cat([bottom, d_bottom], dim=1))
            elif self.fusion == "film":
                bottom = self.dna_film(self.count_proj(bottom), d_bottom)
            else:
                bottom = self.dna_xattn(self.count_proj(bottom).transpose(1, 2),
                                        d_bottom.transpose(1, 2)).transpose(1, 2)

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
        elif mid_f:                   # FiLM on the input of each transformer layer
            for layer in self.attn.layers:
                gamma, beta = next(films)
                seq = layer(gamma.transpose(1, 2) * seq + beta.transpose(1, 2))
            if self.attn.norm is not None:
                seq = self.attn.norm(seq)
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
        if self.skip_film == "c_cprime":
            g_out = cond                                                    # [B, 2 D_COV]

        for i in reversed(range(self.n_levels)):
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
            if dec_f:
                z = self.dec2[i](self.dec1[i](z, next(films)), next(films))
            else:
                z = self.dec2[i](self.dec1[i](z))
        z = z[..., :L].transpose(1, 2)                                      # [B, L, C0]

        # per-track head
        hd = self.to_head(z)[:, :, None, :] + self.track_emb_out          # [B, L, A, D_HEAD]
        if self.skip_counts and self.head_skip_mode != "off":
            hs = h_track
            if self.head_skip_mode == "film":         # FiLM(C, C') per track; flag untouched
                theta = self.head_skip_film(torch.cat([self.in_inj.embed(c_in),
                                                       self.out_inj.embed(c_out)], dim=-1))
                theta = theta[:, None]                                      # [B, 1, A, 2 d]
                hs = torch.cat([theta[..., :D_TRACK] * h_track[..., :D_TRACK]
                                + theta[..., D_TRACK:], h_track[..., D_TRACK:]], dim=-1)
            hd = self.head_skip(torch.cat([hd, hs], dim=-1))
        if self.out_sub > 1:                                                # [B, L*s, A, D]
            hd = (hd[:, :, None] + self.out_sub_emb[:, None, :]).reshape(
                B, L * self.out_sub, A, D_HEAD)
        if tokens:
            hd = hd + self.tok_to_head(tok_out)[:, None]
        else:
            hd = self.out_inj(hd, c_out[:, None])
        out = self.nb(self.head_mlp(hd), c_out_t.float())
        if self.control_branch == "separate":                               # t370 C15
            ctl, i = self.ctl_nb(self.ctl_mlp(hd), c_out_t.float()), CONTROL_INDEX
            out = {**out, **{k: torch.cat([out[k][..., :i], ctl[k][..., i:]], dim=-1)
                             for k in ("n", "mu")}}
        return sub_output(out, self.out_sub) if self.out_sub > 1 else out


def _validate(cfg: dict) -> None:
    check_rung(cfg)
    if cfg["skip_source"] not in SKIP_SOURCES:
        raise ValueError(f"skip_source {cfg['skip_source']!r} not in {SKIP_SOURCES}")
    placement_of(cfg)
    sf, hs = cfg.get("skip_film", "cprime"), cfg.get("head_skip", "plain")
    if sf not in SKIP_FILMS:
        raise ValueError(f"skip_film {sf!r} not in {SKIP_FILMS}")
    if hs not in HEAD_SKIPS:
        raise ValueError(f"head_skip {hs!r} not in {HEAD_SKIPS}")
    cb = cfg.get("control_branch", "shared")
    if cb not in CONTROL_BRANCHES:
        raise ValueError(f"control_branch {cb!r} not in {CONTROL_BRANCHES}")
    if cfg["skip_source"] == "dna" and (sf != "cprime" or hs != "plain"):
        raise NotApplicable(f"rung 4: skip_film = {sf} / head_skip = {hs} act on the count skips; "
                            f"skip_source = dna has none")
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
