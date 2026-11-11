"""miniCANDI rung 6: Perceiver IO (Jaegle et al., ICLR 2022, arXiv:2107.14795).

Pipeline (B windows, L bins, A = 36 tracks, P = `patch_bins`, NP = L / P patches):

  1. Input elements. Each (track, patch) of the counts becomes one token: per-bin features
     [Linear(transformed count), visible flag] go through `MaskHandler`, are flattened over the
     patch's P bins and projected to `d_model`; a learned input-track embedding and a fixed
     sinusoidal patch-position code are added. Input covariates C enter by `covariate_injection`:
     `film` / `concat` modulate every track-patch token of the track; `tokens` (base) appends one
     covariate token per available track (it carries the same input-track embedding). DNA (`dna`
     own / shuffled) gives one token per patch (DnaTower features flattened over the patch);
     `dna = off` removes the DNA tokens and their projection. One learned null token is always
     present, so an item with no other element still has a key.
  2. Drop (base `mask_mode = drop`): only available, visible elements enter the encoder. A
     track-patch token is an element when its track is in `x_avail` and at least one of its bins is
     visible; hidden bins inside such a patch are exact zeros with visible flag 0 (MaskHandler,
     drop). Elements are gathered valid-first per item and padded to the batch's largest element
     count; the padding is masked out of attention (key padding). The `mask_token` arm
     (`mask_mode = token`) keeps every track-patch token, with learned hidden-bin and
     missing-track tokens from MaskHandler (token).
  3. Encoder: `n_latents` learned latent vectors cross-attend into the elements; then
     `n_self_layers` pre-norm self-attention blocks on the latents.
  4. Decoder: one output query per (track, patch), built from the patch position, a learned
     output-track embedding and the output covariates C' (by the same `covariate_injection` mode;
     in `tokens` mode the C' token is added to the query). Queries cross-attend into the latents;
     each query's output is unpatched by a Linear to P bins x `d_bin_out` features, then `NBHead`
     maps them to (n, mu) through the count transform's depth add-back at the target's own depth
     (c_out column 0).

Copy-leak guardrail part 1 holds by construction: a hidden bin's or a missing track's count reaches
nothing but a `torch.where` that discards it (MaskHandler), and the element set and its order depend
on the masks only, never on counts.

Config keys read (defaults in DEFAULTS): covariate_injection, mask_mode, dna, count_transform,
scramble_covariates, scramble_all, qc_on_input, average_input, two_step, and the architecture keys
d_model, n_heads, n_latents, n_self_layers, patch_bins, d_bin, d_dna, d_bin_out, mlp_ratio.
`n_latents` is the key of the agent-extra arms (PI ruling 14): `latents_small` = 32,
`latents_large` = 512 against the base 128; all three stay within 10% of the 1.0e6 budget.

Scrambled covariates: `model_covariates`, the rule shared by all six rungs.
`average_input` adds one always-present token per (track, patch) from log1p of the batch's `avg`
rate. `two_step = true` raises NotApplicable, as in every rung.
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

from minicandi.components import (CovariateInjector, CountTransform, DnaTower,  # noqa: E402
                                  MaskHandler, NBHead, NotApplicable, sinusoidal_positions)
from minicandi.contract import A, LOG2_DEPTH_COL, MiniCandi, k_input  # noqa: E402
from minicandi.covariates import scramble_covariates  # noqa: E402

RUNG = 6
NAME = "perceiver"

DEFAULTS = {
    "covariate_injection": "tokens", "mask_mode": "drop", "dna": "own",
    "count_transform": "offset", "qc_on_input": False, "average_input": False,
    "d_model": 128, "n_heads": 4, "n_latents": 128, "n_self_layers": 4, "patch_bins": 16,
    "d_bin": 4, "d_dna": 16, "d_bin_out": 16, "mlp_ratio": 2,
}


# ---------------------------------------------------------------------------------------------
# Rule shared by all six rungs, kept identical in each module (tests/test_minicandi_all_rungs.py
# proves it). Registration: the registry imports this module by name and builds `Model(config)`.
# ---------------------------------------------------------------------------------------------
TWO_STEP_MESSAGE = ("two_step (D15) is not built in any rung: two-step correction is a later "
                    "ablation on the winning design")
SCRAMBLE_SEED = 0


def check_rung(cfg: dict) -> None:
    """Refuse a config that names another rung (`rung` may be the number or the module name)."""
    r = cfg.get("rung", RUNG)
    if r not in (RUNG, str(RUNG), NAME):
        raise ValueError(f"{NAME} is rung {RUNG}; config has rung {r!r}")
    if cfg.get("two_step"):
        raise NotApplicable(TWO_STEP_MESSAGE)


def scramble_mode(cfg: dict) -> str:
    """`all` (scramble_all), `covariates` (scramble_covariates) or `none`."""
    if cfg.get("scramble_all"):
        return "all"
    return "covariates" if cfg.get("scramble_covariates") else "none"


def model_covariates(batch: dict, mode: str, training: bool) -> tuple:
    """(c_in, c_out) for the covariate encoders and (c_in_t, c_out_t) whose column 0 is the depth
    the count transform and the NB head use.

    `covariates`: every covariate column the encoders read (depth included) is permuted across the
    windows of the batch (`covariates.scramble_covariates`); the transform keeps each track's true
    depth, so the arm measures use of the covariates beyond the depth offset. `all`: the transform
    reads the same scrambled depth. Scrambling belongs to the model; the dataset does not apply
    it. Eval mode draws from a generator seeded with SCRAMBLE_SEED; training mode from the global
    torch RNG.
    """
    c_in, c_out = batch["c_in"], batch["c_out"]
    if mode == "none":
        return c_in, c_out, c_in, c_out
    gen = None if training else torch.Generator().manual_seed(SCRAMBLE_SEED)
    s_in = scramble_covariates(c_in.cpu(), batch["x_avail"].cpu(), gen).to(c_in.device)
    s_out = scramble_covariates(c_out.cpu(), batch["y_avail"].cpu(), gen).to(c_out.device)
    if mode == "all":
        return s_in, s_out, s_in, s_out
    return s_in, s_out, c_in, c_out


class Attention(nn.Module):
    """Multi-head attention with a boolean key mask [B, Tk] (True = attend)."""

    def __init__(self, d: int, n_heads: int):
        super().__init__()
        if d % n_heads:
            raise ValueError(f"d_model {d} not divisible by n_heads {n_heads}")
        self.h = n_heads
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.o = nn.Linear(d, d)

    def forward(self, xq: torch.Tensor, xkv: torch.Tensor,
                key_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, Tq, d = xq.shape
        Tk = xkv.shape[1]
        q = self.q(xq).view(B, Tq, self.h, -1).transpose(1, 2)
        k, v = self.kv(xkv).view(B, Tk, 2, self.h, -1).permute(2, 0, 3, 1, 4)
        mask = None if key_mask is None else key_mask[:, None, None, :]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o(out.transpose(1, 2).reshape(B, Tq, d))


class Block(nn.Module):
    """Pre-norm (cross- or self-) attention block plus an MLP, both residual."""

    def __init__(self, d: int, n_heads: int, mlp_ratio: int, cross: bool):
        super().__init__()
        self.nq = nn.LayerNorm(d)
        self.nkv = nn.LayerNorm(d) if cross else None
        self.attn = Attention(d, n_heads)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp_ratio * d), nn.GELU(),
                                 nn.Linear(mlp_ratio * d, d))

    def forward(self, x: torch.Tensor, kv: torch.Tensor | None = None,
                key_mask: torch.Tensor | None = None) -> torch.Tensor:
        q = self.nq(x)
        k = q if kv is None else self.nkv(kv)
        x = x + self.attn(q, k, key_mask)
        return x + self.mlp(self.n2(x))


def _gather_valid(tok: torch.Tensor, valid: torch.Tensor):
    """Put each item's valid tokens first (stable order) and cut to the largest valid count.

    tok [B, T, d], valid [B, T] -> (tok' [B, T', d], key_mask [B, T']). The order depends only on
    `valid`, so counts never change which element sits where.
    """
    order = torch.argsort((~valid).to(torch.int8), dim=1, stable=True)
    t_max = max(int(valid.sum(1).max()), 1)
    idx = order[:, :t_max]
    tok = torch.gather(tok, 1, idx[..., None].expand(-1, -1, tok.shape[-1]))
    keep = torch.gather(valid, 1, idx)
    return torch.where(keep[..., None], tok, torch.zeros_like(tok)), keep


class Model(MiniCandi):
    """Rung 6, Perceiver IO. `Model(config)`; `forward(batch) -> {"n", "mu"}` [B, L, A] > 0."""

    def __init__(self, config: dict | None = None):
        super().__init__()
        cfg = {**DEFAULTS, **(config or {})}
        check_rung(cfg)
        self.cfg = cfg
        d, P = int(cfg["d_model"]), int(cfg["patch_bins"])
        if d % 2:
            raise ValueError("d_model must be even (sinusoidal positions)")
        self.d, self.P = d, P
        self.injection = cfg["covariate_injection"]
        self.scramble = scramble_mode(cfg)
        self.average_input = bool(cfg["average_input"])
        d_bin, d_dna, d_out = int(cfg["d_bin"]), int(cfg["d_dna"]), int(cfg["d_bin_out"])

        # inputs
        self.transform = CountTransform(cfg["count_transform"])
        self.bin_in = nn.Linear(1, d_bin - 1)                  # + 1 visible-flag channel
        self.mask = MaskHandler(cfg["mask_mode"], d_bin - 1)
        self.patch_in = nn.Linear(P * d_bin, d)
        self.in_track = nn.Parameter(torch.randn(A, d) * 0.02)
        self.cin = CovariateInjector(self.injection, d, k_input(cfg))
        self.dna_mode = cfg["dna"]
        self.dna = DnaTower(self.dna_mode, d_dna)
        if self.dna_mode != "off":
            self.dna_patch = nn.Linear(P * d_dna, d)
            self.dna_type = nn.Parameter(torch.randn(d) * 0.02)
        if self.average_input:
            self.avg_patch = nn.Linear(P, d)
            self.avg_type = nn.Parameter(torch.randn(d) * 0.02)
        self.null = nn.Parameter(torch.randn(1, 1, d) * 0.02)

        # encoder and latents
        n_heads, mlp = int(cfg["n_heads"]), int(cfg["mlp_ratio"])
        self.latents = nn.Parameter(torch.randn(int(cfg["n_latents"]), d) * 0.02)
        self.encode = Block(d, n_heads, mlp, cross=True)
        self.process = nn.ModuleList(Block(d, n_heads, mlp, cross=False)
                                     for _ in range(int(cfg["n_self_layers"])))

        # decoder
        self.out_track = nn.Parameter(torch.randn(A, d) * 0.02)
        self.cout = CovariateInjector(self.injection, d, k_input(None))
        self.decode = Block(d, n_heads, mlp, cross=True)
        self.unpatch = nn.Linear(d, P * d_out)
        self.head = NBHead(d_out, self.transform)

    # -----------------------------------------------------------------------------------------

    def forward(self, batch: dict) -> dict:
        x_counts, x_avail, bin_mask = batch["x_counts"], batch["x_avail"], batch["x_bin_mask"]
        B, L, Ac = x_counts.shape
        P, d = self.P, self.d
        if L % P:
            raise ValueError(f"L = {L} is not a multiple of patch_bins = {P}")
        NP = L // P
        dev = x_counts.device
        c_in, c_out, c_in_t, c_out_t = model_covariates(batch, self.scramble, self.training)
        pos = sinusoidal_positions(NP, d, device=dev)                       # [NP, d]

        # (track, patch) count tokens
        vis = MaskHandler.visible(x_avail, bin_mask)                         # [B, L, A]
        x = self.transform(x_counts, c_in_t[..., LOG2_DEPTH_COL], vis)
        x = torch.where(vis, x, torch.zeros_like(x))       # no hidden value enters any op below
        h, vis = self.mask(self.bin_in(x[..., None]), x_avail, bin_mask)
        h = torch.cat([h, vis[..., None].to(h.dtype)], dim=-1)               # [B, L, A, d_bin]
        h = h.view(B, NP, P, Ac, -1).permute(0, 1, 3, 2, 4).reshape(B, NP, Ac, -1)
        tok = self.patch_in(h) + self.in_track + pos[None, :, None, :]       # [B, NP, A, d]
        if self.injection != "tokens":
            tok = self.cin(tok, c_in[:, None])
        if self.mask.mode == "drop":
            valid = vis.view(B, NP, P, Ac).any(2)                            # [B, NP, A]
        else:
            valid = torch.ones(B, NP, Ac, dtype=torch.bool, device=dev)
        parts = [tok.reshape(B, NP * Ac, d)]
        valids = [valid.reshape(B, NP * Ac)]
        if self.injection == "tokens":
            parts.append(self.cin.tokens(c_in) + self.in_track)
            valids.append(x_avail if self.mask.mode == "drop"
                          else torch.ones_like(x_avail))
        if self.dna_mode != "off":
            g = self.dna(batch["x_dna"]).reshape(B, NP, -1)                  # [B, NP, P*d_dna]
            parts.append(self.dna_patch(g) + self.dna_type + pos)
            valids.append(torch.ones(B, NP, dtype=torch.bool, device=dev))
        if self.average_input:
            if "avg" not in batch:
                raise ValueError("average_input is set but the batch has no `avg`")
            av = torch.log1p(batch["avg"].float().clamp_min(0))              # [B, L, A]
            av = av.view(B, NP, P, Ac).permute(0, 1, 3, 2).reshape(B, NP, Ac, P)
            av = self.avg_patch(av) + self.avg_type + self.in_track + pos[None, :, None, :]
            parts.append(av.reshape(B, NP * Ac, d))
            valids.append(torch.ones(B, NP * Ac, dtype=torch.bool, device=dev))
        parts.append(self.null.expand(B, 1, d))
        valids.append(torch.ones(B, 1, dtype=torch.bool, device=dev))
        elems, key_mask = _gather_valid(torch.cat(parts, 1), torch.cat(valids, 1))

        # encode, process
        z = self.encode(self.latents.expand(B, -1, -1), elems, key_mask)
        for blk in self.process:
            z = blk(z)

        # decode: one query per (track, patch)
        q = (pos[None, :, None, :] + self.out_track).expand(B, NP, Ac, d)
        if self.injection == "tokens":
            q = q + self.cout.tokens(c_out)[:, None]
        else:
            q = self.cout(q, c_out[:, None])
        o = self.decode(q.reshape(B, NP * Ac, d), z)                          # [B, NP*A, d]
        o = F.gelu(self.unpatch(o)).view(B, NP, Ac, P, -1)
        o = o.permute(0, 1, 3, 2, 4).reshape(B, L, Ac, -1)                   # [B, L, A, d_out]
        return self.head(o, c_out_t)


__all__ = ["DEFAULTS", "Model", "NAME", "RUNG", "TWO_STEP_MESSAGE", "model_covariates",
           "scramble_mode"]
