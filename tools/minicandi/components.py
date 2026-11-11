"""miniCANDI shared components: every rung and every per-rung ablation is built from these.

  CovariateInjector(mode)  concat | film | tokens   (config `covariate_injection`)
  DnaTower(mode)           own | off | shuffled     (config `dna`)
  MaskHandler(mode)        token | drop             (config `mask_mode`)
  CountTransform(name)     offset | log1p | asinh | anscombe | pearson | local_ratio | quantile |
                           raw                      (config `count_transform`), each with
                           `forward` (depth removed on the input side) and `add_back`
  NBHead                   features -> {"n", "mu"} through the transform's depth add-back
  ConvBlock, Downsample, Upsample, sinusoidal_positions   building blocks for rungs 2 to 6
  NotApplicable            raised by a rung constructor for an arm that does not apply to it

Layout convention: per-track features are channel-last, `[B, L, A, d]`, matching the batch's
`[B, L, A]`. Conv blocks take `[N, C, L]` as `nn.Conv1d` does.

Reused by import: `ladder.model.G` (the CAC testbed's g: zero-initialised last layer, bias = the
identity theta) generates the FiLM gamma and beta, so FiLM starts as the identity, as in t118 and
t146; `t146.model.onehot` turns DNA codes into one-hot; the clamps come from `ladder.model`.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

_TOOLS = Path(__file__).resolve().parents[1]
for _p in (_TOOLS / "t118", _TOOLS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ladder.model import G  # noqa: E402
from minicandi.contract import (A, LOG2_DEPTH_COL, MU_MAX, MU_MIN, N_MAX, N_MIN,  # noqa: E402
                                RES)
from minicandi.covariates import CovariateEmbedding  # noqa: E402
from t146.model import onehot  # noqa: E402

INJECTION_MODES = ("concat", "film", "tokens")
DNA_MODES = ("own", "off", "shuffled")
MASK_MODES = ("token", "drop")
TRANSFORMS = ("offset", "log1p", "asinh", "anscombe", "pearson", "local_ratio", "quantile", "raw")
#: transforms with no per-sample inverse; their `add_back` is the offset form (log2 rate)
INPUT_ONLY = ("local_ratio", "quantile")
#: reference depth D0 = 2**REF_LOG2_DEPTH reads; size factor s = 2**(log2_depth - REF_LOG2_DEPTH)
REF_LOG2_DEPTH = 24.0
EPS = 1.0
ASINH_COFACTOR = 1.0
PEARSON_CLIP = 30.0
LOCAL_WINDOWS = (40, 400)        # 1 kb and 10 kb in 25 bp bins


class NotApplicable(ValueError):
    """An ablation arm that a rung cannot express (e.g. `tokens` injection in rung 2)."""


def _check(mode: str, allowed: tuple, what: str) -> str:
    if mode not in allowed:
        raise ValueError(f"{what} {mode!r} not in {allowed}")
    return mode


# ---------------------------------------------------------------------------------------------
# covariates
# ---------------------------------------------------------------------------------------------


class CovariateInjector(nn.Module):
    """Condition features `h [..., d_model]` on raw covariates `c [..., K]`.

    concat  h' = Linear(cat(h, emb(c)))                       (adds 1 linear layer)
    film    gamma, beta = G(emb(c)); h' = gamma * h + beta     (identity at initialisation)
    tokens  `forward` returns h unchanged; `tokens(c)` gives one token per track,
            emb(c) + a learned track embedding, for the rung to append to its sequence

    `c` must broadcast against h's leading dims after embedding (e.g. c [B, 1, A, K] against
    h [B, L, A, d]). The embedding is computed once per call on the un-broadcast c.
    """

    def __init__(self, mode: str, d_model: int, k_in: int, d_cov: int = 32,
                 n_tracks: int = A):
        super().__init__()
        self.mode = _check(mode, INJECTION_MODES, "covariate_injection")
        self.d_model = d_model
        self.embed = CovariateEmbedding(k_in, d_cov)
        if mode == "concat":
            self.proj = nn.Linear(d_model + d_cov, d_model)
        elif mode == "film":
            init = torch.cat([torch.ones(d_model), torch.zeros(d_model)])
            self.g = G(d_cov, 2 * d_model, init)
        else:
            self.tok = nn.Linear(d_cov, d_model)
            self.track_emb = nn.Parameter(torch.zeros(n_tracks, d_model))
            nn.init.normal_(self.track_emb, std=0.02)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if self.mode == "tokens":
            return h
        e = self.embed(c)
        if self.mode == "concat":
            e = e.expand(*h.shape[:-1], e.shape[-1])
            return self.proj(torch.cat([h, e], dim=-1))
        theta = self.g(e)
        gamma, beta = theta[..., :self.d_model], theta[..., self.d_model:]
        return gamma * h + beta

    def tokens(self, c: torch.Tensor) -> torch.Tensor:
        """c [B, A, K] -> [B, A, d_model]."""
        if self.mode != "tokens":
            raise NotApplicable(f"tokens() called in {self.mode!r} mode")
        return self.tok(self.embed(c)) + self.track_emb


# ---------------------------------------------------------------------------------------------
# DNA
# ---------------------------------------------------------------------------------------------


class DnaTower(nn.Module):
    """x_dna uint8 [B, L*25] -> float [B, L, d_out].

    own       one-hot -> Conv1d(4, c, k) -> GELU -> AvgPool(25) -> Conv1d(c, d_out, 1)
              (the t146 `DNAEncoder` pattern, widened)
    off       zeros, no parameters (the `dna_off` arm)
    shuffled  the same tower, fed the window's 25 bp blocks in a random order: composition per
              bin is kept, alignment to the counts is broken. Deterministic in eval mode
              (generator seeded with `shuffle_seed`), fresh per call in training.
    """

    def __init__(self, mode: str, d_out: int, channels: int = 32, kernel: int = 15,
                 shuffle_seed: int = 0):
        super().__init__()
        self.mode = _check(mode, DNA_MODES, "dna")
        self.d_out = d_out
        self.shuffle_seed = shuffle_seed
        if mode != "off":
            self.conv1 = nn.Conv1d(4, channels, kernel, padding=kernel // 2)
            self.pool = nn.AvgPool1d(RES, RES)
            self.conv2 = nn.Conv1d(channels, d_out, 1)

    def forward(self, x_dna: torch.Tensor) -> torch.Tensor:
        B, n_bp = x_dna.shape
        L = n_bp // RES
        if self.mode == "off":
            return torch.zeros(B, L, self.d_out, device=x_dna.device)
        codes = x_dna
        if self.mode == "shuffled":
            gen = None if self.training else torch.Generator().manual_seed(self.shuffle_seed)
            perm = torch.randperm(L, generator=gen).to(x_dna.device)
            codes = x_dna.reshape(B, L, RES)[:, perm].reshape(B, n_bp)
        h = self.conv2(self.pool(F.gelu(self.conv1(onehot(codes)))))
        return h.transpose(1, 2)


# ---------------------------------------------------------------------------------------------
# masking
# ---------------------------------------------------------------------------------------------


class MaskHandler(nn.Module):
    """Hide masked bins and missing tracks from per-track features.

    forward(h [B, L, A, d], x_avail [B, A], x_bin_mask [B, L, A]) -> (h', visible [B, L, A])

    token  hidden bins get a learned `hidden_token`, missing tracks a learned `missing_token`
    drop   hidden and missing positions become exact zeros; `visible` tells the rung which
           positions to pool over or to give attention to

    Replacement is by `torch.where`, so no value at a hidden position reaches h' (guardrail
    part 1 holds by construction for everything downstream that reads only h').
    """

    def __init__(self, mode: str, d_model: int):
        super().__init__()
        self.mode = _check(mode, MASK_MODES, "mask_mode")
        if mode == "token":
            self.hidden_token = nn.Parameter(torch.zeros(d_model))
            self.missing_token = nn.Parameter(torch.zeros(d_model))
            nn.init.normal_(self.hidden_token, std=0.02)
            nn.init.normal_(self.missing_token, std=0.02)

    @staticmethod
    def visible(x_avail: torch.Tensor, x_bin_mask: torch.Tensor) -> torch.Tensor:
        return x_avail[:, None, :] & ~x_bin_mask

    def forward(self, h: torch.Tensor, x_avail: torch.Tensor, x_bin_mask: torch.Tensor):
        vis = self.visible(x_avail, x_bin_mask)
        if self.mode == "drop":
            return torch.where(vis[..., None], h, torch.zeros_like(h)), vis
        missing = ~x_avail[:, None, :].expand_as(vis)
        tok = torch.where(missing[..., None], self.missing_token.to(h.dtype),
                          self.hidden_token.to(h.dtype))
        return torch.where(vis[..., None], h, tok), vis


# ---------------------------------------------------------------------------------------------
# count transforms
# ---------------------------------------------------------------------------------------------


def _pos(z: torch.Tensor) -> torch.Tensor:
    """Smooth positive floor for inverses that can reach <= 0: ~z for z > 0.5, never < MU_MIN."""
    return F.softplus(z, beta=10.0) + MU_MIN


def _avg_rank_normal(x: torch.Tensor) -> torch.Tensor:
    """Per-window quantile normalisation along dim 1 of [B, L, A]: average ranks of ties, then
    the standard-normal quantile of (rank - 0.5) / L."""
    B, L, Ac = x.shape
    xt = x.transpose(1, 2).contiguous()                       # [B, A, L]
    srt, _ = torch.sort(xt, dim=-1)
    lo = torch.searchsorted(srt, xt, right=False).float()
    hi = torch.searchsorted(srt, xt, right=True).float()
    u = ((lo + hi) / 2.0) / L                                  # (avg 1-based rank - 0.5) / L
    u = u.clamp(0.5 / L, 1 - 0.5 / L)
    z = math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)
    return z.transpose(1, 2)


class CountTransform(nn.Module):
    """Input-side depth removal and output-side depth add-back (architecture ablations doc).

    forward(y int [B, L, A], log2_depth [B, A]) -> x float [B, L, A]
    add_back(eta float [B, L, A], log2_depth [B, A]) -> mu float [B, L, A] > 0

    s = 2**(log2_depth - 24) is the size factor against a 2**24-read reference.
      offset       x = log2((y + 1)/s)                    mu = s * 2**eta
      log1p        x = log(1 + y/s)                       mu = s * pos(e**eta - 1)
      asinh        x = asinh(y/(b s)), b = 1              mu = b s * pos(sinh eta)
      anscombe     x = 2 sqrt(y/s + 3/8)                  mu = s * pos((eta/2)**2 - 3/8)
      pearson      z = (y - s l_a)/sqrt(s l_a + (s l_a)**2/r_a), clipped at +-30
                                                          mu = pos(s l_a + eta sqrt(...))
      local_ratio  x = log2((y + 1)/(lambda_local + 1)), lambda_local = max of the window mean and
                   the 1 kb and 10 kb running means of the same track  (input only)
      quantile     per-window average-rank normal scores (input only)
      raw          x = y                                  mu = pos(eta) (depth via covariates)
    The input-only transforms have no per-sample inverse, so their add_back is the offset form.
    pos() is a smooth positive floor (softplus, beta 10, plus 1e-4). l_a and r_a (pearson) are
    per-track buffers, default 1 and 10; set them from training data with `set_pearson_stats`.
    """

    def __init__(self, name: str, n_tracks: int = A):
        super().__init__()
        self.name = _check(name, TRANSFORMS, "count_transform")
        self.register_buffer("lam", torch.ones(n_tracks))
        self.register_buffer("r", torch.full((n_tracks,), 10.0))

    def set_pearson_stats(self, lam: torch.Tensor, r: torch.Tensor) -> None:
        self.lam.copy_(torch.as_tensor(lam, dtype=self.lam.dtype))
        self.r.copy_(torch.as_tensor(r, dtype=self.r.dtype))

    @staticmethod
    def size_factor(log2_depth: torch.Tensor) -> torch.Tensor:
        return torch.exp2(log2_depth.float() - REF_LOG2_DEPTH)[:, None, :]   # [B, 1, A]

    def forward(self, y: torch.Tensor, log2_depth: torch.Tensor) -> torch.Tensor:
        y = y.float()
        s = self.size_factor(log2_depth)
        n = self.name
        if n == "offset":
            return torch.log2((y + EPS) / s)
        if n == "log1p":
            return torch.log1p(y / s)
        if n == "asinh":
            return torch.asinh(y / (ASINH_COFACTOR * s))
        if n == "anscombe":
            return 2.0 * torch.sqrt(y / s + 0.375)
        if n == "pearson":
            m = s * self.lam
            z = (y - m) / torch.sqrt(m + m * m / self.r)
            return z.clamp(-PEARSON_CLIP, PEARSON_CLIP)
        if n == "local_ratio":
            yt = y.transpose(1, 2)                                     # [B, A, L]
            lam = yt.mean(dim=-1, keepdim=True).expand_as(yt)
            for w in LOCAL_WINDOWS:
                if w < yt.shape[-1]:
                    run = F.avg_pool1d(yt, w, stride=1, padding=w // 2,
                                       count_include_pad=False)[..., :yt.shape[-1]]
                    lam = torch.maximum(lam, run)
            return torch.log2((yt + EPS) / (lam + EPS)).transpose(1, 2)
        if n == "quantile":
            return _avg_rank_normal(y)
        return y                                                      # raw

    def add_back(self, eta: torch.Tensor, log2_depth: torch.Tensor) -> torch.Tensor:
        s = self.size_factor(log2_depth)
        n = self.name
        if n in ("offset",) + INPUT_ONLY:
            mu = s * torch.exp2(eta.clamp(max=60.0))
        elif n == "log1p":
            mu = s * _pos(torch.expm1(eta.clamp(max=60.0)))
        elif n == "asinh":
            mu = ASINH_COFACTOR * s * _pos(torch.sinh(eta.clamp(-60.0, 60.0)))
        elif n == "anscombe":
            mu = s * _pos((eta / 2.0) ** 2 - 0.375)
        elif n == "pearson":
            m = s * self.lam
            mu = _pos(m + eta * torch.sqrt(m + m * m / self.r))
        else:                                                         # raw
            mu = _pos(eta)
        return mu.clamp(MU_MIN, MU_MAX)


class NBHead(nn.Module):
    """features [B, L, A, d] + C' -> {"n": [B, L, A], "mu": [B, L, A]}, both > 0.

    A shared Linear(d, 2) plus a per-track bias [A, 2]; the Linear is zero-initialised, so at
    initialisation eta = the track bias (0) and log n = log n0 for every bin. mu comes from the
    transform's depth add-back with log2_depth = c_out[..., 0] (the target's own C'), so an
    imputed target is decoded at its own depth.
    """

    def __init__(self, d_in: int, transform: "CountTransform | str" = "offset", n0: float = 2.0,
                 n_tracks: int = A):
        super().__init__()
        self.transform = transform if isinstance(transform, CountTransform) \
            else CountTransform(transform, n_tracks)
        self.lin = nn.Linear(d_in, 2)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)
        self.track_bias = nn.Parameter(torch.zeros(n_tracks, 2))
        self.log_n0 = math.log(n0)

    def forward(self, h: torch.Tensor, c_out: torch.Tensor) -> dict:
        out = self.lin(h) + self.track_bias                           # [B, L, A, 2]
        eta, log_n = out[..., 0], self.log_n0 + out[..., 1]
        n = log_n.clamp(math.log(N_MIN), math.log(N_MAX)).exp()
        mu = self.transform.add_back(eta, c_out[..., LOG2_DEPTH_COL])
        return {"n": n.float(), "mu": mu.float()}


# ---------------------------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------------------------


class ConvBlock(nn.Module):
    """[N, C_in, L] -> [N, C_out, L]: Conv1d (same padding) -> GroupNorm -> GELU, residual when
    C_in == C_out. An optional `film` (gamma, beta), each [N, C_out, 1], modulates after the norm.
    """

    def __init__(self, c_in: int, c_out: int, kernel: int = 5, dilation: int = 1,
                 groups: int = 8):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, kernel, dilation=dilation,
                              padding=(kernel - 1) // 2 * dilation)
        self.norm = nn.GroupNorm(math.gcd(groups, c_out), c_out)
        self.residual = c_in == c_out

    def forward(self, x: torch.Tensor, film: tuple | None = None) -> torch.Tensor:
        h = self.norm(self.conv(x))
        if film is not None:
            h = film[0] * h + film[1]
        h = F.gelu(h)
        return x + h if self.residual else h


class Downsample(nn.Module):
    """[N, C, L] -> [N, C_out, L / factor] by a strided conv."""

    def __init__(self, c_in: int, c_out: int, factor: int = 2):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, factor, stride=factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """[N, C, L] -> [N, C_out, L * factor] by a transposed conv."""

    def __init__(self, c_in: int, c_out: int, factor: int = 2):
        super().__init__()
        self.conv = nn.ConvTranspose1d(c_in, c_out, factor, stride=factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def sinusoidal_positions(L: int, d: int, device=None) -> torch.Tensor:
    """[L, d] fixed sinusoidal position code (d even)."""
    pos = torch.arange(L, device=device, dtype=torch.float32)[:, None]
    i = torch.arange(0, d, 2, device=device, dtype=torch.float32)
    ang = pos / torch.pow(10000.0, i / d)
    out = torch.zeros(L, d, device=device)
    out[:, 0::2] = torch.sin(ang)
    out[:, 1::2] = torch.cos(ang[:, : d // 2])
    return out
