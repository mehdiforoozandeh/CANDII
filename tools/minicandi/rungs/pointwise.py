"""Rung 1: the pointwise set model. Each bin is predicted from that bin alone.

Per bin, per track (channel-last `[B, L, A, d]`):
  member   x = CountTransform(x_counts, log2 depth of C, visible) -> Linear(1, d) + track embedding
           -> input covariate injection (film | concat with C; `tokens`: see below)
           -> phi (2-layer MLP) -> MaskHandler (drop: hidden and missing members become zeros and
           are left out of the pool; token: they become learned tokens and stay in it)
  pool     the mean over the pooled members (permutation-invariant), plus log(1 + member count)
  tokens   with `covariate_injection = tokens`, each AVAILABLE track's C is one extra set member
           (CovariateInjector.tokens -> phi), pooled with the count members
  trunk    MLP(cat(pool, log count, DNA vector)) -> z [B, L, d]; the DNA vector is the shared
           `DnaTower` at this bin (one-hot conv, kernel 15 bp, averaged over the bin's 25 bp)
  decode   per output track: z + output track embedding (+ Linear of log1p(avg) when
           `average_input`) -> output covariate injection with C' (film | concat; `tokens`: add
           the track's C' token) -> psi (2-layer residual MLP) -> NBHead (the depth add-back of
           the same CountTransform reads C' column 0, so a target is decoded at its own depth)

The width `d_model` is chosen so that the parameter count is the closest to `param_budget`
(default 1.0e6) among multiples of 8, unless the config gives `d_model`. The search builds
throwaway models under a forked RNG, so it does not move the caller's RNG.

Guardrail part 1 holds by construction: the only path from `x_counts` to the output goes through
MaskHandler, which replaces every hidden or missing member by `torch.where`, and the drop-mode pool
weights them by exactly 0. The window-level transforms (local_ratio, quantile) take `visible`.

Scrambled covariates (`scramble_covariates`, `scramble_all`): `model_covariates`, the rule shared
by all six rungs. `two_step = true` (decision D15) raises NotApplicable, as in every rung.

Component arm keys (pinned interface E, chunk A3) reach every component through
`component_kwargs(cfg)`: `covariate_injection = adaln_zero`, `covariate_dropout` (passed to
`model_covariates`), `covariate_embedding`, `dna = precomputed` (the DnaTower reads
`batch["x_dna_emb"]`; needs `dna_embedding_dim`), `offset_eps`, `count_transform = soft_clip` and
`head = nb_shared_size`. With every key at its default the model is today's, parameter for
parameter and output for output (tests/test_minicandi_rung_arms.py).
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
from minicandi.components import (CountTransform, CovariateInjector, DnaTower,  # noqa: E402
                                  MaskHandler, NBHead, NotApplicable, component_kwargs)
from minicandi.covariates import model_covariates, scramble_mode  # noqa: E402

RUNG = 1
NAME = "pointwise"

DEFAULT_PARAM_BUDGET = 1.0e6
D_DNA = 32


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


class _MLP(nn.Module):
    """Linear -> GELU -> Linear, residual when d_in == d_out."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.l1 = nn.Linear(d_in, d_hidden)
        self.l2 = nn.Linear(d_hidden, d_out)
        self.residual = d_in == d_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.l2(F.gelu(self.l1(x)))
        return x + h if self.residual else h


class Model(C.MiniCandi):
    """Rung 1 (see the module docstring). Built from the config dict alone."""

    def __init__(self, config: dict):
        super().__init__()
        cfg = dict(config)
        check_rung(cfg)
        self.scramble = scramble_mode(cfg)
        d = cfg.get("d_model")
        if d is None:
            d = choose_width(C.without_extras(cfg))
        self.d_model = d = int(d)
        self.injection = cfg.get("covariate_injection", "film")
        self.mask_mode = cfg.get("mask_mode", "drop")
        self.average_input = bool(cfg.get("average_input"))
        k_in = C.k_input(cfg)
        kw = component_kwargs(cfg)
        self.cov_dropout = kw["covariate_dropout"]

        self.transform = CountTransform(cfg.get("count_transform", "offset"), **kw["transform"])
        self.in_proj = nn.Linear(1, d)
        self.in_track = nn.Parameter(torch.zeros(C.A, d))
        nn.init.normal_(self.in_track, std=0.02)
        self.in_cov = CovariateInjector(self.injection, d, k_in, **kw["injector"])
        self.phi = _MLP(d, d, d)
        self.mask = MaskHandler(self.mask_mode, d)
        self.dna = DnaTower(cfg.get("dna", "own"), D_DNA, **kw["dna"])
        self.trunk = nn.Sequential(nn.Linear(d + 1 + D_DNA, d), nn.GELU(), _MLP(d, d, d))
        self.out_track = nn.Parameter(torch.zeros(C.A, d))
        nn.init.normal_(self.out_track, std=0.02)
        self.out_cov = CovariateInjector(self.injection, d, C.K_META, **kw["injector"])
        self.psi = _MLP(d, d, d)
        self.head = NBHead(d, self.transform, **kw["head"])
        if self.average_input:                    # additive, default off: constructed last
            self.avg_proj = nn.Linear(1, d)
        # extra inputs (chunk XI), constructed last; the DNA vector exists in every dna mode
        self.build_extras(cfg, D_DNA, ("in_cov", "out_cov"))

    def forward(self, batch: dict) -> dict:
        self.begin_extras(batch)
        x_avail, x_bin_mask = batch["x_avail"], batch["x_bin_mask"]
        c_in, c_out, c_in_t, c_out_t = model_covariates(batch, self.scramble, self.training,
                                                         self.cov_dropout)
        vis = MaskHandler.visible(x_avail, x_bin_mask)                     # [B, L, A]
        x = self.transform(batch["x_counts"], c_in_t[..., C.LOG2_DEPTH_COL], vis)
        h = self.in_proj(x.unsqueeze(-1)) + self.in_track                  # [B, L, A, d]
        h = self.in_cov(h, c_in[:, None])
        h, vis = self.mask(self.phi(h), x_avail, x_bin_mask)
        if self.mask_mode == "drop":
            w = vis.to(h.dtype)
        else:
            w = torch.ones_like(vis, dtype=h.dtype)
        total = (h * w[..., None]).sum(2)                                  # [B, L, d]
        count = w.sum(2)                                                   # [B, L]
        if self.injection == "tokens":
            tok = self.phi(self.in_cov.tokens(c_in))                       # [B, A, d]
            tw = x_avail.to(h.dtype)
            total = total + (tok * tw[..., None]).sum(1)[:, None, :]
            count = count + tw.sum(1)[:, None]
        pooled = total / count.clamp_min(1.0)[..., None]
        dna = self.dna(batch["x_dna_emb" if self.dna.mode == "precomputed" else "x_dna"])
        dna = self.add_extra_track(dna, batch)
        z = torch.cat([pooled, torch.log1p(count)[..., None], dna], -1)
        z = self.trunk(z)                                                  # [B, L, d]
        g = z[:, :, None, :] + self.out_track                              # [B, L, A, d]
        if self.average_input and "avg" in batch:
            g = g + self.avg_proj(torch.log1p(batch["avg"].float().clamp_min(0.0))[..., None])
        if self.injection == "tokens":
            g = g + self.out_cov.tokens(c_out)[:, None]
        else:
            g = self.out_cov(g, c_out[:, None])
        return self.head(self.psi(g), c_out_t)


def _count(cfg: dict, d: int) -> int:
    with torch.random.fork_rng(devices=[]):
        return Model(dict(cfg, d_model=d)).param_count()


def choose_width(cfg: dict, step: int = 8, lo: int = 8, hi: int = 2048) -> int:
    """The multiple of `step` whose parameter count is the closest to `param_budget`."""
    budget = float(cfg.get("param_budget") or DEFAULT_PARAM_BUDGET)
    a, b = lo // step, hi // step
    while a < b:                                   # smallest k with count(k * step) >= budget
        m = (a + b) // 2
        if _count(cfg, m * step) >= budget:
            b = m
        else:
            a = m + 1
    cands = {max(a - 1, lo // step) * step, a * step}
    return min(cands, key=lambda d: (abs(_count(cfg, d) - budget), d))


__all__ = ["Model", "NAME", "RUNG", "TWO_STEP_MESSAGE", "choose_width"]
