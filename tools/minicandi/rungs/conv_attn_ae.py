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
`control_input`, `predict_control`, `base_depth` act in the dataset; the model needs no change for
them. Scrambled covariates: `model_covariates`, the rule shared by all six rungs. `two_step = true`
raises NotApplicable, as in every rung.

Component arm keys (pinned interface E, chunk A3) reach every component through
`component_kwargs(cfg)`: `covariate_injection = adaln_zero`, `covariate_dropout` (passed to
`model_covariates`), `covariate_embedding`, `dna = precomputed` (the DnaTower reads
`batch["x_dna_emb"]`; needs `dna_embedding_dim`), `offset_eps`, `count_transform = soft_clip` and
`head = nb_shared_size`. With every key at its default the model is today's, parameter for
parameter and output for output (tests/test_minicandi_rung_arms.py).
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
from minicandi.contract import A, K_META, LOG2_DEPTH_COL, MiniCandi, k_input  # noqa: E402
from minicandi.covariates import model_covariates, scramble_mode  # noqa: E402

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
        L = int(cfg["context_bins"])
        if L % (2 ** N_DOWN):
            raise ValueError(f"context_bins {L} must be a multiple of {2 ** N_DOWN}")
        if width is None:
            width = fit_width(cfg)
        self.config = cfg
        self.width = W = int(width)
        self.k_in = k_input(cfg)
        self.injection = cfg["covariate_injection"]
        self.average_input = bool(cfg.get("average_input"))
        self.scramble = scramble_mode(cfg)
        kw = M.component_kwargs(cfg)
        self.cov_dropout = kw["covariate_dropout"]

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
        d_fuse = A * D_TRACK + (D_DNA if cfg["dna"] != "off" else 0)
        self.fuse = nn.Linear(d_fuse, W)

        self.enc = nn.ModuleList(M.ConvBlock(W, W, KERNEL) for _ in range(N_DOWN))
        self.down = nn.ModuleList(M.Downsample(W, W, 2) for _ in range(N_DOWN))
        layer = nn.TransformerEncoderLayer(W, N_HEADS, FF_MULT * W, dropout=0.0,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.attn = nn.TransformerEncoder(layer, N_LAYERS, enable_nested_tensor=False)
        self.up = nn.ModuleList(M.Upsample(W, W, 2) for _ in range(N_DOWN))
        self.dec = nn.ModuleList(M.ConvBlock(W, W, KERNEL) for _ in range(N_DOWN))

        self.readout = nn.Linear(W, A * D_OUT)
        self.track_emb = nn.Parameter(torch.zeros(A, D_OUT))
        nn.init.normal_(self.track_emb, std=0.02)
        self.head = M.NBHead(D_OUT, self.transform, **kw["head"])

    # -------------------------------------------------------------------------------- forward

    def forward(self, batch: dict) -> dict:
        x_counts, x_avail, x_bin_mask = batch["x_counts"], batch["x_avail"], batch["x_bin_mask"]
        c_in, c_out, c_in_t, c_out_t = model_covariates(batch, self.scramble, self.training,
                                                         self.cov_dropout)
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
            feats.append(self.dna(batch["x_dna_emb" if self.dna.mode == "precomputed"
                                        else "x_dna"]))
        z = self.fuse(torch.cat(feats, dim=-1)).transpose(1, 2)            # [B, W, L]

        for blk, dn in zip(self.enc, self.down):
            z = dn(blk(z))
        seq = z.transpose(1, 2)                                            # [B, L', W]
        Lc = seq.shape[1]
        seq = seq + M.sinusoidal_positions(Lc, self.width, seq.device).to(seq.dtype)
        pad = None
        if self.injection == "tokens":
            t_in = self.cov_in.tokens(c_in)                                # [B, A, W]
            t_out = self.cov_out.tokens(c_out)
            seq = torch.cat([seq, t_in, t_out], dim=1)
            pad = torch.cat([torch.zeros(B, Lc, dtype=torch.bool, device=seq.device),
                             ~x_avail, ~batch["y_avail"]], dim=1)          # True = not a key
        seq = self.attn(seq, src_key_padding_mask=pad)
        z = seq[:, :Lc].transpose(1, 2)
        for upm, blk in zip(self.up, self.dec):
            z = blk(upm(z))

        f = self.readout(z.transpose(1, 2)).reshape(B, L, Ac, D_OUT) + self.track_emb
        if self.injection == "tokens":
            f = f + self.tok_read(seq[:, Lc + Ac:])[:, None]               # c_out tokens
        else:
            f = self.cov_out(f, c_out[:, None])
        return self.head(f, c_out_t)


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
