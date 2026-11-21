"""miniCANDI rung 3: the conv plus attention autoencoder (rung 2 plus a transformer encoder on the
downsampled latent; `miniCANDI architecture ablations.md`, "Rung 3").

Data flow, B windows, L bins, A = 36 tracks:

  x_counts --CountTransform(c_in depth, visible)--> x [B, L, A]
  per-track lift  Linear(1, D_TRACK)                 h [B, L, A, D_TRACK]
  MaskHandler(mask_mode)  hidden bins and missing tracks replaced (token) or zeroed (drop)
  input covariates  concat | film on h with c_in; tokens: c_in becomes A attention tokens
  [average_input]  + Linear(1, D_TRACK)(log1p(avg))  (avg holds no hidden-bin content)
  track fusion  Linear(A * D_TRACK [+ D_DNA], W): the one place tracks mix ("tracks mix once")
  DnaTower(dna)  own | off | shuffled, concatenated before the fusion (absent when off)
  conv encoder  N_DOWN x (ConvBlock, Downsample by 2)          [B, W, L / 2**N_DOWN]
  transformer encoder on the coarse sequence (sinusoidal positions; in tokens mode the c_in
              tokens of available tracks and the c_out tokens of requested targets join it)
  conv decoder  N_DOWN x (Upsample by 2, ConvBlock)           [B, W, L]
  per-track readout  Linear(W, A * D_OUT) + a learned track embedding    [B, L, A, D_OUT]
  output covariates  concat | film with c_out; tokens: + Linear(W, D_OUT)(the c_out token after
              attention)
  NBHead(D_OUT, the same CountTransform): mu through the depth add-back at the target's own C'

Guardrail part 1 holds by construction: the only path from `x_counts` to anything downstream is
through `MaskHandler`, which replaces every hidden bin and every missing track by `torch.where`
before the first conv or attention layer; the window-level transforms (`local_ratio`,
`quantile`) read visible bins only. So neither convolution nor attention can carry hidden-bin
content. `tests/test_minicandi_rung3.py` checks it for every supported arm.

The trunk width W is fitted to `param_budget` (binary search over multiples of 8), so every arm
lands within 10% of the budget (matched parameters).

Config keys read: covariate_injection, mask_mode, dna, count_transform, qc_on_input,
average_input, scramble_covariates, scramble_all, two_step, context_bins, param_budget.
Tuning knobs (default-off; absent = the module constants, so every earlier run is unchanged):
`n_down` (N_DOWN; attention runs on context_bins / 2**n_down positions), `attn_layers`
(N_LAYERS), `attn_heads` (N_HEADS; must divide the fitted width) and `attn_dropout` (0.0; the
dropout of every transformer encoder layer). The width search refits W for each setting.
`control_input`, `predict_control`, `base_depth` act in the dataset; the model needs no change for
them. Scrambled covariates: `model_covariates`, the rule shared by all six rungs. `two_step = true`
raises NotApplicable, as in every rung.

Component arm keys (pinned interface E, chunk A3) reach every component through
`component_kwargs(cfg)`: `covariate_injection = adaln_zero`, `covariate_dropout` (passed to
`model_covariates`), `covariate_embedding`, `dna = precomputed` (the DnaTower reads
`batch["x_dna_emb"]`; needs `dna_embedding_dim`), `offset_eps`, `count_transform = soft_clip` and
`head = nb_shared_size`. `res_bp` (item A7; 1, 5, 25, 50, 125) reaches the DnaTower and the
count transform through `component_kwargs` as on rungs 2 and 4; `res_in_sub` / `res_out_sub` raise
NotApplicable. With every key at its default the model is today's, parameter for
parameter and output for output (tests/test_minicandi_rung_arms.py).

Trunk FiLM (t370 C1, `tools/minicandi/trunk_film.py`): `film_placement` (ends | encoder |
decoder | bottleneck | all; default ends = today's model, which builds nothing). The sites are the
`enc` ConvBlocks, the input of each transformer layer (gamma * seq + beta, the padding mask kept)
and the `dec` ConvBlocks; the condition is the pooled C and C' embedding of `cov_in` and
`cov_out`. `tokens` with a non-default placement, and the U-Net keys `skip_film` / `head_skip`,
raise NotApplicable.

Control branch (t370 C15): `control_branch` (shared | separate; default shared = today's model,
which builds nothing). `separate` gives track 35 (the ChIP control) its own copy of the NB head
(`ctl_head`, same shapes; this rung has no per-track output MLP, so there is none to copy), built
last and fed by the same head features; tracks 0 to 34 keep the shared head. The width is fitted
with the extra head counted.

DNA fusion (item A11, `tools/minicandi/dna_fusion.py`): `dna_fusion = film` (the DNA features
FiLM the track-fusion output, which then reads counts only), `late` (DNA joins after the conv
encoder through a learned patch pooling) or `xattn` (the coarse count tokens cross-attend to the
pooled DNA tokens before the self-attention). The width is the base width (the search reads the
config without `dna_fusion`). Absent = the early concatenation, bit for bit
(tests/test_ablation_dna_fusion.py).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from torch import nn

_TOOLS = Path(__file__).resolve().parents[2]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from minicandi import components as M  # noqa: E402
from minicandi.components import NotApplicable  # noqa: E402
from minicandi.contract import (A, CONTROL_INDEX, K_META, LOG2_DEPTH_COL, MiniCandi,  # noqa: E402
                                k_input, without_extras)
from minicandi.covariates import drop_field_indices, model_covariates, scramble_mode  # noqa: E402
from minicandi.dna_fusion import (DnaCrossAttn, DnaFiLM, DnaPatchDown, fusion_of,  # noqa: E402
                                  without_fusion)
from minicandi.trunk_film import (TrunkFiLM, check_unet_only, placement_of,  # noqa: E402
                                  pooled_condition, sites_of)

RUNG = 3
NAME = "conv_attn_ae"
DEFAULTS = {"covariate_injection": "film", "mask_mode": "token", "dna": "own",
            "count_transform": "offset", "qc_on_input": False, "average_input": False,
            "context_bins": 1024, "param_budget": 1.0e6}

D_TRACK = 16          # per-track input feature width
D_OUT = 16            # per-track output feature width
D_DNA = 32            # DNA tower output width
D_COV = 32            # covariate embedding width
N_DOWN = 4            # downsampling levels (factor 2 each): 1024 bins -> 64 latent positions
N_LAYERS = 2          # transformer encoder layers
N_HEADS = 4
FF_MULT = 2           # transformer feed-forward width = FF_MULT * W
KERNEL = 5
WIDTH_STEP = 8
WIDTH_RANGE = (16, 1024)
BUDGET_TOL = 0.10
CONTROL_BRANCHES = ("shared", "separate")   # t370 C15 `control_branch`; shared = today


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


def resolve_config(config: dict | None) -> dict:
    cfg = dict(DEFAULTS)
    cfg.update(config or {})
    return cfg


class Model(MiniCandi):
    """Rung 3. `Model(config)`; `forward(batch) -> {"n", "mu"}` each float32 [B, L, A] > 0."""

    def __init__(self, config: dict | None = None, *, width: int | None = None):
        super().__init__()
        cfg = resolve_config(config)
        check_rung(cfg)
        check_unet_only(cfg, NAME)
        film_sites = sites_of(placement_of(cfg))
        if max(M.res_subs_of(cfg)) > 1:               # item A7: the decoupled arms are rungs 2 and 4
            raise NotApplicable("res_in_sub / res_out_sub are built in rungs 2 and 4 only")
        # tuning knobs (round-2 tuning, default-off: absent = the module constants)
        n_down = int(cfg.get("n_down", N_DOWN))
        n_layers = int(cfg.get("attn_layers", N_LAYERS))
        n_heads = int(cfg.get("attn_heads", N_HEADS))
        attn_dropout = float(cfg.get("attn_dropout", 0.0))
        if n_down < 1 or n_layers < 1 or n_heads < 1 or not 0.0 <= attn_dropout < 1.0:
            raise ValueError(f"bad rung-3 knobs: n_down {n_down}, attn_layers {n_layers}, "
                             f"attn_heads {n_heads}, attn_dropout {attn_dropout}")
        L = int(cfg["context_bins"])
        if L % (2 ** n_down):
            raise ValueError(f"context_bins {L} must be a multiple of {2 ** n_down}")
        if width is None:
            width = fit_width(without_fusion(without_extras(cfg)))   # A11: base width
        self.config = cfg
        self.width = W = int(width)
        self.k_in = k_input(cfg)
        self.injection = cfg["covariate_injection"]
        self.average_input = bool(cfg.get("average_input"))
        self.scramble = scramble_mode(cfg)
        kw = M.component_kwargs(cfg)
        self.cov_dropout = kw["covariate_dropout"]
        _drop = drop_field_indices(cfg)   # A12: None unless covariate_drop_fields
        self.cov_drop = {} if _drop is None else {"drop": _drop}   # off: today's call

        self.transform = M.CountTransform(cfg["count_transform"], **kw["transform"])
        self.lift = nn.Linear(1, D_TRACK)
        self.mask = M.MaskHandler(cfg["mask_mode"], D_TRACK)
        if self.injection == "tokens":
            self.cov_in = M.CovariateInjector("tokens", W, self.k_in, D_COV, **kw["injector"])
            self.cov_out = M.CovariateInjector("tokens", W, K_META, D_COV, **kw["injector"])
            self.tok_read = nn.Linear(W, D_OUT)
        else:
            self.cov_in = M.CovariateInjector(self.injection, D_TRACK, self.k_in, D_COV,
                                              **kw["injector"])
            self.cov_out = M.CovariateInjector(self.injection, D_OUT, K_META, D_COV,
                                               **kw["injector"])
        if self.average_input:
            self.avg_lift = nn.Linear(1, D_TRACK)
        self.dna = M.DnaTower(cfg["dna"], D_DNA, **kw["dna"])
        self.fusion = fusion_of(cfg, NAME)                 # item A11; None = early concatenation
        d_fuse = A * D_TRACK + (D_DNA if cfg["dna"] != "off" and self.fusion is None else 0)
        self.fuse = nn.Linear(d_fuse, W)

        if W % n_heads:
            raise ValueError(f"width {W} is not a multiple of attn_heads {n_heads}")
        self.enc = nn.ModuleList(M.ConvBlock(W, W, KERNEL) for _ in range(n_down))
        self.down = nn.ModuleList(M.Downsample(W, W, 2) for _ in range(n_down))
        layer = nn.TransformerEncoderLayer(W, n_heads, FF_MULT * W, dropout=attn_dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.attn = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.up = nn.ModuleList(M.Upsample(W, W, 2) for _ in range(n_down))
        self.dec = nn.ModuleList(M.ConvBlock(W, W, KERNEL) for _ in range(n_down))

        self.readout = nn.Linear(W, A * D_OUT)
        self.track_emb = nn.Parameter(torch.zeros(A, D_OUT))
        nn.init.normal_(self.track_emb, std=0.02)
        self.head = M.NBHead(D_OUT, self.transform, **kw["head"])
        # DNA fusion arms (item A11), built last so the base modules keep their draws
        if self.fusion == "film":
            self.dna_film = DnaFiLM(D_DNA, W)
        elif self.fusion in ("late", "xattn"):
            self.dna_down = DnaPatchDown(D_DNA, W, 2 ** n_down)
            if self.fusion == "xattn":
                self.dna_xattn = DnaCrossAttn(W, n_heads)
        # trunk FiLM (t370 C1), built after the base modules; film_placement = ends (default)
        # builds nothing
        self.film_sites = film_sites
        if film_sites:
            ch = []
            if "encoder" in film_sites:
                ch += [W] * n_down
            if "bottleneck" in film_sites:
                ch += [W] * n_layers
            if "decoder" in film_sites:
                ch += [W] * n_down
            self.trunk_film = TrunkFiLM(2 * D_COV, ch)
        # extra inputs (chunk XI), constructed last; dna = off leaves no DNA features
        self.build_extras(cfg, D_DNA if cfg["dna"] != "off" else None, ("cov_in", "cov_out"))
        # control branch (t370 C15), constructed last; shared builds nothing
        self.control_branch = cfg.get("control_branch", "shared")
        if self.control_branch not in CONTROL_BRANCHES:
            raise ValueError(f"control_branch {self.control_branch!r} not in {CONTROL_BRANCHES}")
        if self.control_branch == "separate":
            self.ctl_head = M.NBHead(D_OUT, self.transform, **kw["head"])

    # -------------------------------------------------------------------------------- forward

    def forward(self, batch: dict) -> dict:
        self.begin_extras(batch)
        x_counts, x_avail, x_bin_mask = batch["x_counts"], batch["x_avail"], batch["x_bin_mask"]
        c_in, c_out, c_in_t, c_out_t = model_covariates(batch, self.scramble, self.training,
                                                         self.cov_dropout,
                                                         **self.cov_drop)
        B, L, Ac = x_counts.shape
        vis = M.MaskHandler.visible(x_avail, x_bin_mask)
        x = self.transform(x_counts, c_in_t[..., LOG2_DEPTH_COL], vis)
        h, _ = self.mask(self.lift(x[..., None]), x_avail, x_bin_mask)    # [B, L, A, D_TRACK]
        if self.injection != "tokens":
            h = self.cov_in(h, c_in[:, None])
        if self.average_input:
            h = h + self.avg_lift(torch.log1p(batch["avg"].float().clamp_min(0.0))[..., None])
        feats = [h.reshape(B, L, Ac * D_TRACK)]
        if self.dna.mode != "off":
            dna = self.add_extra_track(
                self.dna(batch["x_dna_emb" if self.dna.mode == "precomputed" else "x_dna"]),
                batch)
            if self.fusion is None:
                feats.append(dna)
            else:
                dna = dna.to(h.dtype).transpose(1, 2)                      # [B, D_DNA, L]
        z = self.fuse(torch.cat(feats, dim=-1)).transpose(1, 2)            # [B, W, L]
        if self.fusion == "film":
            z = self.dna_film(z, dna)

        films = iter(())
        if self.film_sites:
            films = iter(self.trunk_film(pooled_condition(
                self.cov_in.embed(c_in), x_avail, self.cov_out.embed(c_out), batch["y_avail"])))
        enc_f, mid_f, dec_f = (part in self.film_sites
                               for part in ("encoder", "bottleneck", "decoder"))
        for blk, dn in zip(self.enc, self.down):
            z = dn(blk(z, next(films) if enc_f else None))
        if self.fusion == "late":
            z = z + self.dna_down(dna)
        seq = z.transpose(1, 2)                                            # [B, L', W]
        if self.fusion == "xattn":
            seq = self.dna_xattn(seq, self.dna_down(dna).transpose(1, 2))
        Lc = seq.shape[1]
        seq = seq + M.sinusoidal_positions(Lc, self.width, seq.device).to(seq.dtype)
        pad = None
        if self.injection == "tokens":
            t_in = self.cov_in.tokens(c_in)                                # [B, A, W]
            t_out = self.cov_out.tokens(c_out)
            seq = torch.cat([seq, t_in, t_out], dim=1)
            pad = torch.cat([torch.zeros(B, Lc, dtype=torch.bool, device=seq.device),
                             ~x_avail, ~batch["y_avail"]], dim=1)          # True = not a key
        if mid_f:                     # FiLM on the input of each transformer layer
            for layer in self.attn.layers:
                gamma, beta = next(films)
                seq = gamma.transpose(1, 2) * seq + beta.transpose(1, 2)
                seq = layer(seq, src_key_padding_mask=pad)
            if self.attn.norm is not None:
                seq = self.attn.norm(seq)
        else:
            seq = self.attn(seq, src_key_padding_mask=pad)
        z = seq[:, :Lc].transpose(1, 2)
        for upm, blk in zip(self.up, self.dec):
            z = blk(upm(z), next(films) if dec_f else None)

        f = self.readout(z.transpose(1, 2)).reshape(B, L, Ac, D_OUT) + self.track_emb
        if self.injection == "tokens":
            f = f + self.tok_read(seq[:, Lc + Ac:])[:, None]               # c_out tokens
        else:
            f = self.cov_out(f, c_out[:, None])
        out = self.head(f, c_out_t)
        if self.control_branch == "separate":                              # t370 C15
            ctl, i = self.ctl_head(f, c_out_t), CONTROL_INDEX
            out = {**out, **{k: torch.cat([out[k][..., :i], ctl[k][..., i:]], dim=-1)
                             for k in ("n", "mu")}}
        return out


# ------------------------------------------------------------------------------- budget


def _count(cfg: dict, width: int) -> int:
    return Model(cfg, width=width).param_count()


def fit_width(config: dict | None) -> int:
    """The multiple of 8 whose parameter count is closest to `param_budget` (binary search; the
    count is increasing in the width). Raises ValueError when no width is within 10%."""
    cfg = resolve_config(config)
    budget = float(cfg["param_budget"])
    lo, hi = WIDTH_RANGE[0] // WIDTH_STEP, WIDTH_RANGE[1] // WIDTH_STEP
    while lo < hi:
        mid = (lo + hi) // 2
        if _count(cfg, mid * WIDTH_STEP) < budget:
            lo = mid + 1
        else:
            hi = mid
    cands = [w * WIDTH_STEP for w in (lo - 1, lo) if WIDTH_RANGE[0] <= w * WIDTH_STEP]
    best = min(cands, key=lambda w: abs(_count(cfg, w) - budget))
    n = _count(cfg, best)
    if abs(n - budget) > BUDGET_TOL * budget:
        raise ValueError(f"no width gives {budget:.0f} parameters within 10% (best {best}: {n})")
    return best


__all__ = ["DEFAULTS", "Model", "NAME", "RUNG", "TWO_STEP_MESSAGE", "fit_width", "resolve_config"]
