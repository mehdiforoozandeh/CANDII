"""The CANDI decoder — grouped deconv trunk, per-assay per-layer FiLM, NB head.

The decoder mirrors the encoder's conv tower: every transposed convolution is grouped by assay, so
no channel of assay `a` ever reaches assay `b`, and conditioning is applied *while* each assay's
profile is forming rather than once at the end.

    z            [B, 96, d_model]
      input_proj                        -> [B, 96, A, lane]
      FiLM (0)
      deconv x2, groups=A, + FiLM (1)   -> [B, 192, A, lane]
      deconv x2, groups=A, + FiLM (2)   -> [B, 384, A, lane]
      deconv x2, groups=A, + FiLM (3)   -> [B, 768, A, lane]
      weight-shared head lane -> 1      -> eta, raw_n   [B, 768, A]

Two further heads read the same finished trunk features and are OFF unless `heads` names them:
`GaussianSignalHead` (arcsinh log-p-value, `mu`/`var`) and `PeakHead` (peak probability). They are
supervision, not prediction targets the kit scores — `candi.eval` is NB-only by ruling.

WHY FOUR FiLM TAPS AND NOT ONE AT THE END
-----------------------------------------
FiLM is a POSITION-CONSTANT affine: gamma and beta do not vary along the 768 positions. Applied
after the deconv it can raise, lower or rescale a finished profile but it cannot move a peak, narrow
a peak, or turn a sharp ATAC profile into a broad H3K36me3 domain. Assay-specific SHAPE forms INSIDE
the deconv, so the conditioning has to be there while it forms. Widening the lane does not fix that
— C=16 was never too few channels, the affine simply arrived too late.

Tap 0 (right after `input_proj`) is NOT redundant with tap 1: `input_proj` is a dense `Linear`, so
every output sees every input and the lanes are a labelling with no assay identity until FiLM gives
them one. The first grouped deconv would otherwise run on anonymous lanes.

WHY CONSTANT LANE WIDTH RATHER THAN AN 8 -> 4 -> 2 -> 1 MIRROR
--------------------------------------------------------------
The exact mirror shrinks conditioning capacity precisely as resolution grows: at 8->4->2->1 the last
FiLM tap modulates a single channel per assay, so target metadata collapses to one scalar gain at
full resolution, and the head degenerates to `Linear(1, 1)`. Once the trunk is grouped, width is
nearly free, so the lane is held constant and both FiLM and the head keep something to act on.
This is a reasoned choice, not a measured one — no run has yet varied it.

`LaneNorm`, `PerLaneFiLM` and `_init_grouped_weight` live here rather than in a separate module
because the decoder is now the only consumer of the per-lane primitives.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from candi._vendored import CLOZE, MISSING
from candi.encoder import MetadataEmbedding, init_film_proj

__all__ = ["DECONV_NORMS", "DECODER_FILM_TAPS", "DEFAULT_HEADS", "HEADS", "HEAD_SHARINGS",
           "SIGNAL_HEAD_INIT_STD", "SIGNAL_HEAD_VAR_BIAS", "SIGNAL_HEAD_VAR_EPS",
           "GaussianSignalHead", "LaneNorm", "NoFiLM", "PeakHead",
           "PerLaneFiLM", "PerLaneHead", "LaneDeconvBlock", "SymmetricDecoder", "parse_heads"]


# `deconv_norm`, not `lane_norm`: the old name collided with its own first value, so "lane_norm=lane"
# and "lane_norm=group" read as if only one of them were per-lane. Both are; what differs is which
# axes the statistic pools. The knob is named for WHERE it applies (the deconv trunk) and its values
# for WHAT they pool.
DECONV_NORMS = ("lane", "group")

# The decoder's FiLM taps, in the order they act. `pre_deconv` is the tap right after the input
# projection; `per_deconv` is one tap after each deconv block; `post_head` conditions the two head
# outputs themselves. See the module docstring for why the first two are not redundant.
DECODER_FILM_TAPS = ("pre_deconv", "per_deconv", "post_head")

HEAD_SHARINGS = ("shared", "per_assay")

# The output heads, in the order they are named everywhere. `count` is the shipped model and the only
# one `candi.eval` scores; `signal` and `peak` are additional supervision, off unless asked for.
HEADS = ("count", "signal", "peak")
DEFAULT_HEADS: Tuple[str, ...] = ("count",)

# The Gaussian head's parameterisation, COPIED from the production model rather than chosen here
# (`EpiDenoise/model.py::GaussianLayer`), so a run of this kit and a run of production put the same
# distribution on the same target and their likelihoods are comparable numbers.
#
#   mu  = softplus(W_mu  . lane)                 W_mu  ~ N(0, 1e-4),  bias 0.0
#   var = softplus(W_var . lane) + 1e-6          W_var ~ N(0, 1e-4),  bias 0.5
#
# WHY THE INIT IS THIS AND NOT XAVIER. At std=1e-4 the weights are effectively zero on the first
# steps, so every position predicts its BIAS: mu = softplus(0) = 0.693, var = softplus(0.5) = 0.974.
# That is a broad, near-unit Gaussian centred in the target's range, which is the well-conditioned
# start for an NLL whose `(y-mu)^2/var` term explodes when a small variance meets a wrong mean. Xavier
# would instead hand step 0 a per-position mean driven by an unlearned lane and a variance that can
# start near the 1e-6 floor, which is the same NLL with a 10^6 multiplier on its error term. The
# weights are not exactly zero because a head with identical outputs everywhere gives scipy's
# correlation metrics a constant input and no gradient direction to break the tie.
SIGNAL_HEAD_INIT_STD = 1e-4
SIGNAL_HEAD_VAR_BIAS = 0.5
# Added AFTER the softplus, never inside it. Softplus is positive but not bounded away from zero, and
# the NLL divides by the variance, so a saturated-negative logit is a division by ~0 rather than a
# large loss. This floor is what makes the head's own output a legal variance.
SIGNAL_HEAD_VAR_EPS = 1e-6


def parse_heads(heads) -> Tuple[str, ...]:
    """Normalise a comma string or iterable of head names into a validated tuple.

    `count` is REQUIRED, not merely defaulted. The NB count head is the frozen objective, it is the
    only head `candi.eval` scores, and every recorded number in this repo is one of its metrics — so a
    checkpoint trained without it could not be evaluated by anything in the kit and its run JSON would
    report metrics for a head that never existed. Refusing the set here is a one-line failure on the
    submit command; allowing it is a failure six GPU-hours later, in `evaluate`.
    """
    if isinstance(heads, str):
        names = [h.strip() for h in heads.split(",") if h.strip()]
    else:
        names = [str(h).strip() for h in heads]
    unknown = [h for h in names if h not in HEADS]
    if unknown:
        raise ValueError(f"unknown head(s) {unknown}; valid heads are {HEADS}")
    if "count" not in names:
        raise ValueError(
            f"heads {sorted(set(names))} omits 'count'. The NB count head is the objective the kit "
            "evaluates and the one every recorded metric describes; a model without it cannot be "
            "scored by candi.eval. Add 'count' — the signal and peak heads are auxiliary, not "
            "alternatives to it.")
    # de-duplicated, and ordered as HEADS so two spellings of the same set compare equal
    return tuple(h for h in HEADS if h in names)


class LaneNorm(nn.Module):
    """Per-assay normalisation on `[B, L, T, C]`. Neither mode moves information across lanes.

    TWO SEPARATE PROBLEMS LIVE IN THE WORD "LAYERNORM" HERE, AND ONLY ONE OF THEM IS THE LEAK.

    `mode="lane"` normalises the C channels of one lane AT EACH POSITION. It cannot leak — the lane's
    channels are the whole population — but it fixes each lane's energy at every position to 1, so a
    peak bin and a background bin survive only as a PATTERN across C, not as a difference in
    amplitude. Amplitude is the thing being predicted.

    `mode="group"` normalises the C channels of one lane ACROSS ALL L POSITIONS — the `nn.GroupNorm`
    statistic, at the granularity of one assay. It removes the lane's overall location and scale and
    leaves the profile along the genome intact. Depth is already carried by the decoder's log-link
    offset, so removing a lane's global scale costs nothing the model needs.

    Affine is per-CHANNEL in both modes, so each assay keeps an independent scale and shift; only the
    pooling of the statistic changes.

    WHY THIS IS NOT DECIDABLE FROM THEORY, and is therefore an arm rather than a fix: `ConvTower` is
    `gelu(norm(conv(x)) + rconv(x))`, and the residual branch is an UN-NORMALISED 1x1, so amplitude
    reaches the next layer under either mode. The argument above says `group` should be better; the
    residual says the gap may be small. Measure it.
    """

    def __init__(self, lane_width: int, mode: str = "lane", eps: float = 1e-5) -> None:
        super().__init__()
        if mode not in DECONV_NORMS:
            raise ValueError(f"deconv_norm must be one of {DECONV_NORMS}; got {mode!r}")
        self.C = int(lane_width)
        self.mode = str(mode)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.C))
        self.bias = nn.Parameter(torch.zeros(self.C))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.mode == "lane":
            dims = (-1,)                 # C only, at each position
        else:
            dims = (1, -1)               # C and L, per lane  (z is [B, L, T, C])
        mean = z.mean(dim=dims, keepdim=True)
        var = z.var(dim=dims, keepdim=True, unbiased=False)
        return (z - mean) * torch.rsqrt(var + self.eps) * self.weight + self.bias


def _init_grouped_weight(w: torch.Tensor, fan_in: int, fan_out: int) -> None:
    """Xavier-uniform a `[T, fan_in, fan_out]` stack as T independent `Linear(fan_in, fan_out)`."""
    bound = math.sqrt(6.0 / (fan_in + fan_out))
    nn.init.uniform_(w, -bound, bound)


class PerLaneFiLM(nn.Module):
    """adaLN-zero FiLM on `[B, L, T, C]`, one (gamma, beta) per track from that track's covariates.

    Shared projection weights, per-track inputs — the same construction `encoder.FiLMLayer` already
    uses in the conv tower. Zero-init so the arm starts as an exact no-op and steering grows.
    """

    def __init__(self, embed_dim: int, lane_width: int, init: str = "zero") -> None:
        super().__init__()
        self.C = int(lane_width)
        self.proj = nn.Linear(int(embed_dim), 2 * self.C)
        init_film_proj(self.proj, init)

    def forward(self, z: torch.Tensor, meta_embed: torch.Tensor) -> torch.Tensor:
        if meta_embed.shape[1] != z.shape[2]:
            raise ValueError(f"PerLaneFiLM track mismatch: z has {z.shape[2]} lanes, "
                             f"meta_embed has {meta_embed.shape[1]}")
        gamma, beta = self.proj(meta_embed).chunk(2, dim=-1)      # [B, T, C] each
        return z * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


class NoFiLM(nn.Module):
    """A tap that was switched off. Returns its input untouched and owns no parameters.

    Kept in the `film_layers` list rather than removed from it so tap N is always at index N: a tap
    set that drops `pre_deconv` must not silently renumber the taps after it, or `film/dec_*` in the
    gradient log would point at a different layer than it did in the run being compared to.
    """

    def forward(self, z: torch.Tensor, meta_embed: torch.Tensor) -> torch.Tensor:
        return z


class PerLaneHead(nn.Module):
    """`Linear(C,H) -> GELU -> Linear(H,1)`, but A INDEPENDENT copies — one per assay.

    The shipped head is weight-SHARED: one small MLP reads every assay's lane. That is defensible
    (the lane already carries assay identity, injected by four FiLM taps) and it is 1/A the
    parameters, but it is an assumption nobody has measured. This is the alternative, exposed so the
    question can be answered rather than argued: same arithmetic, same shapes, A separate weight
    stacks contracted with `einsum`.

    Returns `[B, L, A, 1]` so the caller's `.squeeze(-1)` is the same line for both head types.
    """

    def __init__(self, num_assays: int, lane: int, hidden: int) -> None:
        super().__init__()
        self.A, self.C, self.H = int(num_assays), int(lane), int(hidden)
        self.w1 = nn.Parameter(torch.empty(self.A, self.C, self.H))
        self.b1 = nn.Parameter(torch.zeros(self.A, self.H))
        self.w2 = nn.Parameter(torch.empty(self.A, self.H, 1))
        self.b2 = nn.Parameter(torch.zeros(self.A, 1))
        _init_grouped_weight(self.w1, self.C, self.H)
        _init_grouped_weight(self.w2, self.H, 1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        h = F.gelu(torch.einsum("blac,ach->blah", feat, self.w1) + self.b1)
        return torch.einsum("blah,aho->blao", h, self.w2) + self.b2


class GaussianSignalHead(nn.Module):
    """`(mu, var)` per position per assay, read off that assay's lane. Off unless `signal` is asked for.

    The target is the arcsinh log-p-value track, which the BAKE already transformed
    (`prep/reference_sample.py` loads the bigwig with `arcsinh=True`), so this head predicts the
    transformed quantity directly and nothing downstream may transform it again. That is the opposite
    of the counts rule in invariant 5 — counts arrive raw and the ENCODER transforms them — and the two
    are easy to conflate. The arcsinh of a `-log10 p` is non-negative, which is why a softplus mean is
    the right link here and would be wrong for a signal that can go negative.

    ONE `Linear` PER PARAMETER, NOT THE COUNT HEAD'S `Linear -> GELU -> Linear`. Production uses a
    single linear map and this head exists to be comparable to production; the count head's hidden
    layer and its `head_sharing` / `head_hidden` knobs are the subject of a measured question that has
    not been asked of the signal head. Giving it a different shape "for symmetry" would silently make
    the two heads a second uncontrolled difference. This is a reasoned choice, not a measured one.
    """

    def __init__(self, lane: int, init_std: float = SIGNAL_HEAD_INIT_STD,
                 var_bias: float = SIGNAL_HEAD_VAR_BIAS, var_eps: float = SIGNAL_HEAD_VAR_EPS) -> None:
        super().__init__()
        self.var_eps = float(var_eps)
        self.linear_mu = nn.Linear(int(lane), 1)
        self.linear_var = nn.Linear(int(lane), 1)
        # Re-init AFTER construction, which is the ordering that matters: `nn.Linear.__init__` has
        # already drawn from the global RNG by this point, so these calls change the weights without
        # rewinding the stream. That is why the whole head must be built last — see `SymmetricDecoder`.
        nn.init.normal_(self.linear_mu.weight, mean=0.0, std=float(init_std))
        nn.init.zeros_(self.linear_mu.bias)
        nn.init.normal_(self.linear_var.weight, mean=0.0, std=float(init_std))
        nn.init.constant_(self.linear_var.bias, float(var_bias))

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """`[B, L, A, lane] -> (mu, var)`, each `[B, L, A]`."""
        mu = F.softplus(self.linear_mu(feat)).squeeze(-1)
        var = F.softplus(self.linear_var(feat)).squeeze(-1) + self.var_eps
        return mu, var


class PeakHead(nn.Module):
    """`sigmoid(Linear(lane, 1))` — one peak PROBABILITY per position per assay. Off by default.

    Returns a probability rather than a logit because that is production's interface
    (`EpiDenoise/model.py::PeakLayer`) and because the peak output is consumed as a probability by the
    ENCODE precision/recall/AUROC evaluation. The cost is that the paired loss is a plain BCE on a
    number that can saturate, so `train.peak_bce_loss` clamps before taking the log — see the reason
    there. A logit head would be numerically better and would not be this head.

    NOTE for anyone diffing against production: `PeakLayer` carries a comment claiming "controlled
    initialization" and then applies none. This follows the CODE — default `nn.Linear` init — because
    that is what every production checkpoint was actually trained with.
    """

    def __init__(self, lane: int) -> None:
        super().__init__()
        self.linear_peak = nn.Linear(int(lane), 1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """`[B, L, A, lane] -> [B, L, A]` in (0, 1)."""
        return torch.sigmoid(self.linear_peak(feat)).squeeze(-1)


class LaneDeconvBlock(nn.Module):
    """Grouped transposed conv + 1x1 grouped residual + per-lane LayerNorm — mirror of `ConvTower`.

    `groups=num_assays`, so no channel of assay a ever reaches assay b. The per-lane LayerNorm
    normalises within a lane only, for the same reason.
    """

    def __init__(self, num_assays: int, lane: int, kernel_size: int = 3, upsample: int = 2,
                 deconv_norm: str = "lane") -> None:
        super().__init__()
        self.A = int(num_assays)
        self.lane = int(lane)
        ch = self.A * self.lane
        pad = (int(kernel_size) - 1) // 2
        self.deconv = nn.ConvTranspose1d(ch, ch, int(kernel_size), stride=int(upsample),
                                         padding=pad, output_padding=int(upsample) - 1,
                                         groups=self.A)
        self.rdeconv = nn.ConvTranspose1d(ch, ch, 1, stride=int(upsample),
                                          output_padding=int(upsample) - 1, groups=self.A)
        self.norm = LaneNorm(self.lane, deconv_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`[B, L, A, lane] -> [B, L*upsample, A, lane]`."""
        B, L, A, C = x.shape
        h = x.reshape(B, L, A * C).transpose(1, 2)          # [B, A*lane, L]
        y = self.deconv(h)
        Lo = y.shape[-1]
        # norm in lane space, before the residual add — mirrors ConvBlock(norm, apply_act=False)
        y = self.norm(y.transpose(1, 2).reshape(B, Lo, A, C))
        r = self.rdeconv(h).transpose(1, 2).reshape(B, Lo, A, C)
        return F.gelu(y + r)


class SymmetricDecoder(nn.Module):
    """Grouped deconv trunk + per-assay per-layer FiLM + depth-offset log-linked NB head.

    Accepts either a dense latent `[B, L2, d]` (`in_dense_dim` — what the shipped encoder emits) or
    a lane latent `[B, L2, T, C]` (`in_lane_width`). On the dense path `input_proj` is a cross-assay
    mixer and the `pre_deconv` tap is what re-establishes assay identity afterwards; on the lane path
    the projection is per-lane, so it never mixes and the tap is conditioning rather than repair.

    Head arithmetic is the recorded one, so the objective, the depth offset and the reference offset
    stay comparable to every run already on disk:
        log2_mu = (d - depth_center) + eta   [offset ON and depth not a sentinel]
        log2_mu = eta                        [otherwise]
        log2_mu += log_ref                   [when a reference is supplied]
        mu = 2^clamp;  n = softplus(raw_n) + eps;  p = n / (n + mu)

    `heads` names which distributions the decoder emits. `('count',)` — the default — is exactly the
    recorded model: the two optional heads are not constructed, own no parameters, and add no keys to
    the output dict. Naming `signal` or `peak` adds `GaussianSignalHead` / `PeakHead`, each reading the
    SAME finished trunk features the count head reads.
    """

    def __init__(self, *, num_assays: int, lane: int = 8, meta_embed_dim: int = 32,
                 in_dense_dim: Optional[int] = None, in_lane_width: Optional[int] = None,
                 n_deconv_layers: int = 3, upsample: int = 2, conv_kernel_size: int = 3,
                 use_offset: bool = True, depth_center: float = 25.1, mu_eps: float = 1e-6,
                 log2_mu_clamp: tuple = (-15.0, 30.0), use_layernorm: bool = True,
                 depth_row: int = 0, num_cells: int = 0, meta_gain: float = 1.0,
                 deconv_norm: str = "lane",
                 film_taps: tuple = ("pre_deconv", "per_deconv"), film_init: str = "zero",
                 head_sharing: str = "shared", head_hidden: int = 0,
                 heads: tuple = DEFAULT_HEADS) -> None:
        super().__init__()
        if (in_dense_dim is None) == (in_lane_width is None):
            raise ValueError("give exactly one of in_dense_dim (dense latent) or in_lane_width "
                             "(lane latent)")
        taps = tuple(film_taps)
        unknown = [t for t in taps if t not in DECODER_FILM_TAPS]
        if unknown:
            raise ValueError(f"unknown decoder film taps {unknown}; valid: {DECODER_FILM_TAPS}")
        if str(head_sharing) not in HEAD_SHARINGS:
            raise ValueError(f"head_sharing must be one of {HEAD_SHARINGS}; got {head_sharing!r}")
        # Validated before anything is built, like the tap set and the geometry: an unknown head name
        # must fail on the command line, not after the trunk is on the GPU.
        self.heads = parse_heads(heads)
        self.A = int(num_assays)
        self.lane = int(lane)
        self.use_offset = bool(use_offset)
        self.depth_center = float(depth_center)
        self.eps = float(mu_eps)
        self.clamp_lo, self.clamp_hi = float(log2_mu_clamp[0]), float(log2_mu_clamp[1])
        self.depth_row = int(depth_row)
        self.meta_gain = float(meta_gain)
        if not (self.meta_gain > 0.0) or not math.isfinite(self.meta_gain):
            raise ValueError(f"meta_gain must be finite and > 0; got {meta_gain!r}")

        self.dense_in = in_dense_dim is not None
        if self.dense_in:
            self.input_proj = nn.Linear(int(in_dense_dim), self.A * self.lane)
        else:
            self.w_in = nn.Parameter(torch.empty(self.A, int(in_lane_width), self.lane))
            self.b_in = nn.Parameter(torch.zeros(self.A, self.lane))
            _init_grouped_weight(self.w_in, int(in_lane_width), self.lane)

        self.blocks = nn.ModuleList([
            LaneDeconvBlock(self.A, self.lane, int(conv_kernel_size), int(upsample), deconv_norm)
            for _ in range(int(n_deconv_layers))
        ])
        # Index 0 is the `pre_deconv` tap (right after input_proj); indices 1..n are `per_deconv`,
        # one after each block. A disabled tap becomes a `NoFiLM` rather than disappearing, so the
        # indices never renumber — see `NoFiLM`.
        def _tap(kind: str) -> nn.Module:
            if kind in taps:
                return PerLaneFiLM(int(meta_embed_dim), self.lane, film_init)
            return NoFiLM()

        self.film_layers = nn.ModuleList(
            [_tap("pre_deconv")] + [_tap("per_deconv") for _ in range(int(n_deconv_layers))]
        )
        hidden = int(head_hidden) if int(head_hidden) > 0 else self.lane
        self.head_sharing = str(head_sharing)
        if self.head_sharing == "shared":
            self.head_eta = nn.Sequential(nn.Linear(self.lane, hidden), nn.GELU(),
                                          nn.Linear(hidden, 1))
            self.head_n = nn.Sequential(nn.Linear(self.lane, hidden), nn.GELU(),
                                        nn.Linear(hidden, 1))
        else:
            self.head_eta = PerLaneHead(self.A, self.lane, hidden)
            self.head_n = PerLaneHead(self.A, self.lane, hidden)
        self.meta_embedding = MetadataEmbedding(
            num_assays=self.A, embed_dim=int(meta_embed_dim),
            use_layernorm=bool(use_layernorm), num_cells=int(num_cells))
        # BUILT LAST, AND THAT IS THE WHOLE POINT. `post_head` is purely additive and off by default,
        # so switching it on should add a tap and change NOTHING else. Constructed anywhere earlier it
        # would consume RNG draws — `nn.Linear.__init__` samples before `init_film_proj` zeroes it —
        # and every module after it would land on different weights, so the arm would differ from its
        # control in a re-sampled trunk as well as in the tap. That is the failure the encoder's
        # `metadata_embedding` overwrite records; this is the same lesson applied forward.
        self.film_head = (PerLaneFiLM(int(meta_embed_dim), 2, film_init)
                          if "post_head" in taps else None)
        # AND THESE ARE BUILT AFTER IT, FOR THE SAME REASON, ONE STEP FURTHER ALONG. Each new
        # default-off module goes at the very END of the constructor, so switching it on appends RNG
        # draws instead of inserting them: everything above — trunk, count heads, meta embedder,
        # film_head — has already been sampled and cannot move. That is what lets
        # `tests/test_flags.py::test_adding_a_head_leaves_the_count_outputs_bit_identical` assert
        # ZERO difference in p/n/eta/log2_mu/mu with the signal and peak heads switched on, which is
        # the whole claim behind calling these heads additive. Insert anything between `film_head` and
        # here and that test is the one that fails.
        #
        # Both read `feat`, the trunk output, NOT the count head's outputs — `film_head` conditions
        # eta and raw_n and does not touch `feat`, so the two optional heads see the same tensor
        # whether or not the `post_head` tap is on.
        self.head_signal = GaussianSignalHead(self.lane) if "signal" in self.heads else None
        self.head_peak = PeakHead(self.lane) if "peak" in self.heads else None

    def forward(self, z: torch.Tensor, y_meta: torch.Tensor,
                log_ref: Optional[torch.Tensor] = None,
                memb: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # `memb` lets a caller supply the y_meta embedding instead of having this trunk compute it,
        # so a variant that conditions something else on the SAME vector has one path for the
        # covariate gradient to flow through rather than two. None = embed here, which is what the
        # shipped model does.
        if memb is None:
            memb = self.meta_embedding(y_meta.float())          # [B, A, E]
        if self.meta_gain != 1.0:
            memb = memb * self.meta_gain

        if self.dense_in:
            B, L, _ = z.shape
            feat = self.input_proj(z).view(B, L, self.A, self.lane)
        else:
            # lane latent may still carry the control track; the decoder predicts signal assays only,
            # so drop it by SLICING — never by a dense re-projection, which would re-mix the lanes.
            z = z[:, :, :self.A, :]
            feat = torch.einsum("bltc,tcd->bltd", z, self.w_in) + self.b_in

        feat = self.film_layers[0](feat, memb)
        for i, block in enumerate(self.blocks):
            feat = block(feat)
            feat = self.film_layers[i + 1](feat, memb)

        eta = self.head_eta(feat).squeeze(-1)                    # [B, L, A]
        raw_n = self.head_n(feat).squeeze(-1)
        if self.film_head is not None:
            # The two head outputs as one 2-channel lane, so the tap is the same `PerLaneFiLM` the
            # trunk uses rather than a second kind of conditioning with its own arithmetic.
            #
            # `.contiguous()` IS LOAD-BEARING AND IS NOT TIDYING. Slicing the stacked pair yields a
            # STRIDED view, and the arithmetic below — softplus, pow, the division for p — picks a
            # different kernel for a strided input than for a contiguous one. At zero init this tap
            # is algebraically an exact identity, so switching it on must change nothing; without the
            # copy it moved `p` by one ULP anyway, purely through memory layout. A tap that is
            # supposed to be a no-op at init and is not is exactly the thing the golden gate exists
            # to refuse, and this is where the test caught it.
            pair = self.film_head(torch.stack([eta, raw_n], dim=-1), memb)
            eta, raw_n = pair[..., 0].contiguous(), pair[..., 1].contiguous()

        depth = y_meta[:, self.depth_row, :]                     # [B, A]
        valid = (depth != MISSING) & (depth != CLOZE)
        if self.use_offset:
            d_off = (depth - self.depth_center).unsqueeze(1)     # [B, 1, A]
            log2_mu = torch.where(valid.unsqueeze(1), d_off + eta, eta)
        else:
            log2_mu = eta
        if log_ref is not None:
            log2_mu = log2_mu + log_ref
        log2_mu = log2_mu.clamp(self.clamp_lo, self.clamp_hi)
        mu = torch.pow(2.0, log2_mu).clamp_min(self.eps)
        n = F.softplus(raw_n) + self.eps
        p = (n / (n + mu)).clamp(self.eps, 1.0 - self.eps)
        out = dict(p=p, n=n, eta=eta, log2_mu=log2_mu, mu=mu)
        # The optional heads ADD keys; they never rename or replace one. `mu` is already the NB mean,
        # so the Gaussian mean cannot be called `mu` without silently changing what every existing
        # reader of this dict receives — hence `signal_mu`. A caller that does not build these heads
        # gets the five-key dict byte for byte, and `train.aux_head_losses` uses the ABSENCE of the
        # keys as its switch, the same way `forward_full` uses the absence of `log_ref`.
        if self.head_signal is not None:
            out["signal_mu"], out["signal_var"] = self.head_signal(feat)
        if self.head_peak is not None:
            out["peak_prob"] = self.head_peak(feat)
        return out
