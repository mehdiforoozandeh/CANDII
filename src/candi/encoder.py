"""CANDI v2 encoder — standalone, first-principles design.

VENDORED VERBATIM from EpiDenoise/sandbox/candi_v2/encoder.py:1-923 (imports
retargeted, unshipped branches raised; no body reformatted).

#############################################################################
# FROZEN CONSTRUCTION ORDER. V2Encoder.__init__ draws from the global torch  #
# RNG in a fixed sequence: metadata_embedding -> signal_tower -> mask        #
# injector -> dna_tower -> fusion -> transformer_blocks -> (film) ->         #
# output_norm. Historical q19 checkpoints and the --compat-q19 gate depend   #
# on that exact draw sequence. Do NOT reorder, insert, or delete module      #
# construction in this file.                                                 #
#############################################################################

Lineage: derived from EpiDenoise sandbox/jepa_model.py JEPAEncoder (E21–E26 promoted
defaults), copied here so v2 can diverge independently from JEPA experiments.

Architecture (default config):
  1. MetadataEmbedding: per-assay (depth, assay_id, readlen, runtype) → embed_dim
  2. SignalConvTower: grouped Conv1d + residual + MaxPool, FiLM at each layer
  3. MaskTokenInjector: per-assay learned vectors replace masked conv channels
  4. DNAConvTower: ungrouped Conv1d on one-hot DNA
  5. LinearFusion / GatedFusion: merge signal + DNA features
  6. x-transformers encoder with RoPE + optional per-layer FiLM

Shapes (default 8 assays, context_length=768, n_cnn_layers=3, pool=2):
  Input:  x_signal [B, 768, 9]   x_dna [B, 4, 19200]   x_meta [B, 4, 9]
  Output: z        [B, 96, d_model]
"""
from __future__ import annotations

from typing import List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from candi._vendored import exponential_linspace_int
from candi._vendored import CLOZE, MISSING
from candi.config import EncoderConfig

try:
    from x_transformers import Encoder as XEncoder
except ImportError as exc:
    raise ImportError(
        "x-transformers is required for sandbox.candi_v2. "
        "Install with: pip install x-transformers"
    ) from exc


# ---------------------------------------------------------------------------
# Lane views — making the encoder's per-track axis explicit
# ---------------------------------------------------------------------------
# Grouped `Conv1d` only accepts the flat channel-first layout `[B, T*C, L]`, so the track axis
# lives *inside* the channel ordering: group `t` owns channels `[t*C, (t+1)*C)`. That convention
# is load-bearing — `FiLMLayer`, `MaskTokenInjector` and every lane arm depend on it — and until
# now it was implicit in a bare `.view(bsz, channels)`. These two helpers name it.
#
# Both are pure reshapes: no copy, no arithmetic, no parameters. The tower's own shapes read as
#
#     input      [B, 768, 36, 1]      1 raw channel per track
#     block 0    [B, 384, 36, 2]
#     block 1    [B, 192, 36, 4]
#     block 2    [B,  96, 36, 8]      -> flattens to the [B, 96, 288] the fusion expects
#
# which is the same [B, L, T, C] shape the decoder carries end to end (`decoder.LaneDeconvBlock`).

def lanes_from_channels(x: torch.Tensor, num_tracks: int) -> torch.Tensor:
    """`[B, T*C, L] -> [B, T, C, L]`, channel-first. Pure view."""
    bsz, channels, seq = x.shape
    if channels % num_tracks != 0:
        raise ValueError(
            f"lanes_from_channels: channels ({channels}) not divisible by "
            f"num_tracks ({num_tracks}); the grouped-conv layout would misalign"
        )
    return x.view(bsz, num_tracks, channels // num_tracks, seq)


def channels_from_lanes(x: torch.Tensor) -> torch.Tensor:
    """`[B, T, C, L] -> [B, T*C, L]`, the inverse of `lanes_from_channels`."""
    bsz, num_tracks, lane_ch, seq = x.shape
    return x.reshape(bsz, num_tracks * lane_ch, seq)


# UNREACHABLE on the shipped path (output_rms_norm=False -> output_norm is nn.Identity).
# Kept verbatim: deleting it would be a refactor with checkpoint-compat risk for zero gain.
class RMSNormSeq(nn.Module):
    """RMS norm for sequence tensors [B, L, d_model], normalizing over the last dim."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


# ---------------------------------------------------------------------------
# Metadata embedding
# ---------------------------------------------------------------------------

class MetadataEmbedding(nn.Module):
    """Per-assay metadata encoder with distinct missing/cloze tokens.

    Each of the 4 covariate fields (depth, assay_id, read_length, run_type)
    is independently projected to `embed_dim`, then fused via a 2-layer MLP.
    Continuous fields (depth, read_length) use learned embeddings for the
    MISSING (-1) and CLOZE (-2) sentinel values.  Categorical fields
    (assay_id, run_type) have dedicated embedding entries for sentinels.

    Output shape: [B, num_assays+1, embed_dim]  (one vector per assay+control)
    """

    def __init__(
        self,
        num_assays: int,
        embed_dim: int,
        num_runtypes: int = 2,
        use_layernorm: bool = True,
        num_cells: int = 0,
    ) -> None:
        super().__init__()
        self.num_assays = int(num_assays)
        self.num_runtypes = int(num_runtypes)
        self.embed_dim = int(embed_dim)
        # num_cells > 0 turns on the 5th metadata row: per-sample cell-type identity, broadcast
        # identically across every assay column. 0 = the historical 4-row model, and the guard below
        # is what keeps that case bit-identical (see the construction-order note).
        self.num_cells = int(num_cells)
        self.n_rows = 5 if self.num_cells > 0 else 4

        # Continuous covariates: linear projection + sentinel embeddings
        self.depth_proj = nn.Linear(1, embed_dim)
        self.read_length_proj = nn.Linear(1, embed_dim)
        self.depth_missing_emb = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.depth_cloze_emb = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.readlen_missing_emb = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.readlen_cloze_emb = nn.Parameter(torch.randn(embed_dim) * 0.02)

        # Categorical covariates: embedding tables with sentinel entries
        # assay_id: [0..num_assays-1] + MISSING + CLOZE
        self.assay_embedding = nn.Embedding(num_assays + 3, embed_dim)
        # run_type: [0..num_runtypes-1] + MISSING + CLOZE
        self.runtype_embedding = nn.Embedding(num_runtypes + 2, embed_dim)

        # cell_id: [0..num_cells-1] + MISSING + CLOZE. CONSTRUCTION ORDER IS LOAD-BEARING — this
        # module is created only when num_cells > 0, so the num_cells == 0 path draws from the global
        # torch RNG in exactly the historical sequence and `--compat-q19` still holds. Do not hoist it
        # above assay_embedding/runtype_embedding, and do not create a zero-size table unconditionally.
        if self.num_cells > 0:
            self.cell_embedding = nn.Embedding(self.num_cells + 2, embed_dim)

        # Fuse n_rows × embed_dim → embed_dim
        fusion_layers: List[nn.Module] = [
            nn.Linear(self.n_rows * embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        ]
        if bool(use_layernorm):
            fusion_layers.append(nn.LayerNorm(embed_dim))
        self.fusion = nn.Sequential(*fusion_layers)

    def _embed_continuous(
        self,
        values: torch.Tensor,
        proj: nn.Linear,
        missing_emb: nn.Parameter,
        cloze_emb: nn.Parameter,
    ) -> torch.Tensor:
        """Project continuous values; substitute sentinel embeddings for MISSING/CLOZE."""
        missing_mask = values == MISSING
        cloze_mask = values == CLOZE
        emb = proj(values.unsqueeze(-1).float())
        if missing_mask.any():
            emb[missing_mask] = missing_emb.to(emb.dtype)
        if cloze_mask.any():
            emb[cloze_mask] = cloze_emb.to(emb.dtype)
        return emb

    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        """
        Args:
            metadata: [B, n_rows, num_assays+1] — rows are (depth, assay_id, readlen, runtype)
                and, when num_cells > 0, a 5th cell_id row constant across the assay axis
        Returns:
            [B, num_assays+1, embed_dim]
        """
        if metadata.shape[1] != self.n_rows:
            rows = "[log2_depth, assay_id, read_length, run_type]"
            if self.num_cells > 0:
                rows = rows[:-1] + ", cell_id]"
            raise ValueError(
                f"metadata must be {self.n_rows} rows {rows}; got {metadata.shape[1]}"
            )
        depth = metadata[:, 0, :]
        assay_id = metadata[:, 1, :]
        read_length = metadata[:, 2, :]
        runtype = metadata[:, 3, :]

        depth_emb = self._embed_continuous(
            depth, self.depth_proj, self.depth_missing_emb, self.depth_cloze_emb
        )
        readlen_emb = self._embed_continuous(
            read_length, self.read_length_proj,
            self.readlen_missing_emb, self.readlen_cloze_emb,
        )

        # Map sentinel values to dedicated embedding indices
        assay_id_long = assay_id.long()
        bad = assay_id_long[assay_id_long >= 0]
        if bad.numel() and int(bad.max()) > self.num_assays:
            raise ValueError(
                f"assay_id {int(bad.max())} exceeds table bound {self.num_assays} "
                "(control must use id == num_assays); higher ids silently alias onto "
                "the MISSING/CLOZE slots"
            )
        assay_id_long = torch.where(
            assay_id_long == MISSING,
            torch.full_like(assay_id_long, self.num_assays + 1),
            assay_id_long,
        )
        assay_id_long = torch.where(
            assay_id_long == CLOZE,
            torch.full_like(assay_id_long, self.num_assays + 2),
            assay_id_long,
        )
        assay_emb = self.assay_embedding(assay_id_long)

        runtype_long = runtype.long()
        bad = runtype_long[runtype_long >= 0]
        if bad.numel() and int(bad.max()) >= self.num_runtypes:
            raise ValueError(
                f"run_type {int(bad.max())} exceeds table bound {self.num_runtypes} "
                "(valid ids are 0..num_runtypes-1); higher ids silently alias onto "
                "the MISSING/CLOZE slots"
            )
        runtype_long = torch.where(
            runtype_long == MISSING,
            torch.full_like(runtype_long, self.num_runtypes),
            runtype_long,
        )
        runtype_long = torch.where(
            runtype_long == CLOZE,
            torch.full_like(runtype_long, self.num_runtypes + 1),
            runtype_long,
        )
        runtype_emb = self.runtype_embedding(runtype_long)

        parts = [depth_emb, assay_emb, readlen_emb, runtype_emb]

        if self.num_cells > 0:
            cell_long = metadata[:, 4, :].long()
            bad = cell_long[cell_long >= 0]
            if bad.numel() and int(bad.max()) >= self.num_cells:
                raise ValueError(
                    f"cell_id {int(bad.max())} exceeds table bound {self.num_cells} "
                    "(valid ids are 0..num_cells-1); higher ids silently alias onto "
                    "the MISSING/CLOZE slots"
                )
            cell_long = torch.where(
                cell_long == MISSING,
                torch.full_like(cell_long, self.num_cells),
                cell_long,
            )
            cell_long = torch.where(
                cell_long == CLOZE,
                torch.full_like(cell_long, self.num_cells + 1),
                cell_long,
            )
            parts.append(self.cell_embedding(cell_long))

        concat = torch.cat(parts, dim=-1)
        return self.fusion(concat)


# ---------------------------------------------------------------------------
# Conv building blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv1d → Norm → optional GELU.

    `norm="lane"` IS LOAD-BEARING FOR THE LANE ARMS. `norm="layer"` builds `LayerNorm(out_ch)` over
    the FULL channel dimension — all `num_tracks * k` of it — so although the convolution is grouped,
    the normalisation that follows it couples every track to every other track: each track's output
    is divided by a standard deviation computed across the whole panel at that position.

    A grouped conv tower with `norm="layer"` is therefore NOT per-assay, and never has been — this is
    true of candi_kit and of production candi_v2, both of which default to "layer". It is invisible in
    training and shows up only if you look for it: `tests/test_arms.py::
    test_lane_attention_is_the_only_mixer` disables the lane-axis attention and finds signal crossing
    between assays anyway.

    `norm="lane"` is the exact per-lane analogue: the SAME statistic (mean/variance over channels at
    one position), computed within a track instead of across the panel. The affine stays per-channel,
    so per-assay scale and shift survive; only the pooling of the statistic is restricted.
    `norm="group"` would also be per-track, but GroupNorm additionally pools over positions, which is
    a different normaliser, not a narrower one.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        norm: str,
        groups: int = 1,
        apply_act: bool = False,
    ) -> None:
        super().__init__()
        self.normtype = str(norm)
        self.apply_act = bool(apply_act)
        self.n_groups = int(groups)
        self.conv = nn.Conv1d(
            in_ch, out_ch,
            kernel_size=kernel_size, dilation=1, stride=1,
            groups=groups, padding="same",
        )
        if self.normtype == "batch":
            self.norm = nn.BatchNorm1d(out_ch)
        elif self.normtype == "layer":
            self.norm = nn.LayerNorm(out_ch)
        elif self.normtype == "group":
            self.norm = nn.GroupNorm(groups, out_ch)
        elif self.normtype == "instance":
            # Per-channel, per-sample, over POSITIONS. Unlike "lane" and "group" it pools no channels
            # at all, so it is the only option here that cannot mix assays even in principle — and
            # also the only one that removes each channel's own profile scale along the sequence.
            self.norm = nn.InstanceNorm1d(out_ch, affine=True)
        elif self.normtype == "lane":
            if out_ch % self.n_groups != 0:
                raise ValueError(f"conv_norm='lane' needs out_ch % groups == 0; "
                                 f"got out_ch={out_ch}, groups={self.n_groups}")
            self.lane_ch = out_ch // self.n_groups
            # statistics per lane; affine per channel, so assays keep independent scale/shift
            self.norm = nn.LayerNorm(self.lane_ch, elementwise_affine=False)
            self.lane_weight = nn.Parameter(torch.ones(out_ch))
            self.lane_bias = nn.Parameter(torch.zeros(out_ch))
        else:
            raise ValueError(f"Unsupported conv_norm={self.normtype}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.normtype == "layer":
            x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        elif self.normtype == "lane":
            bsz, ch, seq = x.shape
            h = x.permute(0, 2, 1).reshape(bsz, seq, self.n_groups, self.lane_ch)
            h = self.norm(h).reshape(bsz, seq, ch)
            x = (h * self.lane_weight + self.lane_bias).permute(0, 2, 1)
        else:
            x = self.norm(x)
        if self.apply_act:
            x = F.gelu(x)
        return x


class ConvTower(nn.Module):
    """Conv + residual skip (1×1 conv) + MaxPool."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        groups: int = 1,
        pool_size: int = 2,
        norm: str = "layer",
    ) -> None:
        super().__init__()
        self.conv1 = ConvBlock(in_ch, out_ch, kernel_size, norm=norm,
                               groups=groups, apply_act=False)
        self.rconv = nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=1, groups=groups)
        self.pool = nn.MaxPool1d(pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = F.gelu(y + self.rconv(x))
        return self.pool(y)


# ---------------------------------------------------------------------------
# FiLM conditioning layers
# ---------------------------------------------------------------------------

FILM_INITS = ("xavier", "zero", "normal")


def init_film_proj(proj: nn.Linear, init: str = "xavier") -> None:
    """Initialise a FiLM projection in place. Shared by both towers so the options cannot drift.

    THE CHOICE MOVES THE GLOBAL RNG STREAM, and that is not a detail: `xavier` draws twice (weight
    and bias) on top of `nn.Linear`'s own two draws, `zero` draws not at all. Every module built
    after a FiLM layer therefore lands on different weights under a different init, so only the
    per-tower defaults — `xavier` in the conv tower, `zero` in the decoder — reproduce a recorded run.

      xavier  Xavier-uniform weight + N(0, 0.1) bias. The encoder's historical pair: conditioning is
              live from step 0, which is what a tower that must separate 36 tracks immediately wants.
      zero    adaLN-zero. The layer starts as an exact identity and steering has to be learned, so a
              run cannot lose by being born mis-conditioned. The decoder's default.
      normal  N(0, 0.02) weight, zero bias — small but live. Offered because "live but quiet" is a
              real third position between the two above; no run has yet used it.
    """
    if init == "xavier":
        nn.init.xavier_uniform_(proj.weight)
        nn.init.normal_(proj.bias, mean=0.0, std=0.1)
    elif init == "zero":
        nn.init.zeros_(proj.weight)
        nn.init.zeros_(proj.bias)
    elif init == "normal":
        nn.init.normal_(proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(proj.bias)
    else:
        raise ValueError(f"film_init must be one of {FILM_INITS}; got {init!r}")


class FiLMLayer(nn.Module):
    """Per-assay FiLM in channel-first conv space.

    metadata_embed [B, A+1, embed_dim] → project to per-channel (scale, shift)
    and modulate conv activations: x ← x * (1 + scale) + shift.
    """

    def __init__(self, input_dim: int, output_dim: int, init: str = "xavier") -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        init_film_proj(self.proj, init)

    def forward(self, x: torch.Tensor, metadata_embed: torch.Tensor) -> torch.Tensor:
        # Same arithmetic as the flat `view(bsz, channels)` form this replaces — track `t` is
        # modulated by row `t` of the projection — but the lane axis is now named rather than
        # left to the reader to infer from the channel ordering.
        bsz, channels, seq = x.shape
        assays = metadata_embed.shape[1]
        if channels % assays != 0:
            raise ValueError(f"C % F != 0 for FiLMLayer. C={channels}, F={assays}")
        scale, shift = self.proj(metadata_embed).chunk(2, dim=-1)   # [B, T, C] each
        lane_ch = channels // assays
        if scale.shape[-1] != lane_ch:
            raise ValueError(
                f"FiLMLayer width mismatch: projection gives {scale.shape[-1]} params per track "
                f"but the activation has {lane_ch} channels per track (C={channels}, T={assays})"
            )
        lanes = lanes_from_channels(x, assays)                      # [B, T, C, L]
        lanes = lanes * (1.0 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        return channels_from_lanes(lanes)                           # [B, T*C, L]


# UNREACHABLE on the shipped path (film_mode='per_conv' -> only FiLMLayer is built).
class PerAssayFiLM(nn.Module):
    """FiLM in sequence-last (B, L, C) space, applied per-assay."""

    def __init__(self, emb_dim: int, d_per_assay: int, init: str = "xavier") -> None:
        super().__init__()
        self.d_per_assay = int(d_per_assay)
        self.proj = nn.Linear(int(emb_dim), 2 * int(d_per_assay))
        init_film_proj(self.proj, init)

    def forward(self, x: torch.Tensor, meta_embed: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, channels = x.shape
        assays = meta_embed.shape[1]
        d = self.d_per_assay
        if channels != assays * d:
            raise ValueError(
                f"PerAssayFiLM channel mismatch: C={channels}, expected {assays * d}"
            )
        x4 = x.view(bsz, seq_len, assays, d)
        scale, shift = self.proj(meta_embed).chunk(2, dim=-1)
        x4 = x4 * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x4.view(bsz, seq_len, channels)


# UNREACHABLE on the shipped path (film_mode='per_conv', not 'per_conv_and_transformer').
class TransformerFeatureFiLM(nn.Module):
    """FiLM on the fused d_model features, conditioned on pooled metadata."""

    def __init__(self, emb_dim: int, d_model: int, init: str = "xavier") -> None:
        super().__init__()
        self.proj = nn.Linear(int(emb_dim), 2 * int(d_model))
        init_film_proj(self.proj, init)

    def forward(self, x: torch.Tensor, pooled_meta: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(pooled_meta).chunk(2, dim=-1)
        return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ---------------------------------------------------------------------------
# Mask token injection
# ---------------------------------------------------------------------------

class MaskTokenInjector(nn.Module):
    """Replace conv features of masked assays with learned embeddings.

    After the signal conv tower, the output has shape [B, L2, num_tracks * d_per_assay]
    with grouped convolutions keeping each assay's channels independent.  For each
    assay flagged as CLOZE or MISSING in the availability vector, we replace that
    assay's d_per_assay-sized channel slice with its learned mask embedding.
    Available assays keep their real conv features.

    The full mask token is [num_tracks, d_per_assay] — effectively a d_model-sized
    vector partitioned by assay, where available assay slices are overwritten with
    real signal features.
    """

    def __init__(self, num_tracks: int, d_per_assay: int) -> None:
        super().__init__()
        self.num_tracks = int(num_tracks)
        self.d_per_assay = int(d_per_assay)
        self.mask_embedding = nn.Parameter(
            torch.randn(self.num_tracks, self.d_per_assay) * 0.02
        )

    def forward(self, x_conv: torch.Tensor, availability: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_conv:      [B, L2, num_tracks * d_per_assay] — grouped conv output
            availability: [B, num_tracks] — per-assay availability flags
        """
        bsz, seq_len, _ = x_conv.shape
        assays = availability.shape[1]
        if assays != self.num_tracks:
            raise ValueError(
                f"availability tracks ({assays}) != num_tracks ({self.num_tracks})"
            )
        x = x_conv.view(bsz, seq_len, assays, self.d_per_assay)
        replace = (availability == CLOZE) | (availability == MISSING)
        token = self.mask_embedding.view(1, 1, self.num_tracks, self.d_per_assay).to(x.dtype)
        x = torch.where(replace.unsqueeze(1).unsqueeze(-1), token, x)
        return x.view(bsz, seq_len, assays * self.d_per_assay)


# ---------------------------------------------------------------------------
# Signal and DNA conv towers
# ---------------------------------------------------------------------------

class SignalConvTower(nn.Module):
    """Grouped Conv1d tower for signal tracks with configurable FiLM.

    Uses grouped convolutions (groups=num_tracks) so each assay's channels
    are processed independently.  FiLM conditioning can be placed at various
    points depending on film_mode.
    """

    def __init__(
        self,
        num_tracks: int,
        n_layers: int,
        expansion_factor: int,
        kernel_size: int,
        pool_size: int,
        meta_embed_dim: int,
        conv_norm: str,
        film_mode: str,
        film_init: str = "xavier",
    ) -> None:
        super().__init__()
        self.num_tracks = int(num_tracks)
        self.film_mode = str(film_mode)

        # Channel schedule: each layer expands by expansion_factor (grouped)
        conv_channels = [
            self.num_tracks * (int(expansion_factor) ** l)
            for l in range(int(n_layers))
        ]
        out_channels_list: List[int] = []
        blocks: List[nn.Module] = []
        for i in range(int(n_layers)):
            out_ch = (
                conv_channels[i + 1]
                if i + 1 < int(n_layers)
                else int(expansion_factor) * conv_channels[i]
            )
            out_channels_list.append(out_ch)
            blocks.append(ConvTower(
                in_ch=conv_channels[i], out_ch=out_ch,
                kernel_size=int(kernel_size), groups=self.num_tracks,
                pool_size=int(pool_size), norm=str(conv_norm),
            ))
        self.blocks = nn.ModuleList(blocks)
        self.out_channels = out_channels_list[-1]
        self.out_per_assay = self.out_channels // self.num_tracks
        # plain python lists, not buffers — they hold no state and must not enter the state_dict
        self._out_channels_list = list(out_channels_list)
        self._pool_sizes = [int(pool_size)] * int(n_layers)

        # Build FiLM layers based on mode
        self.pre_film: Optional[FiLMLayer] = None
        self.post_film: Optional[PerAssayFiLM] = None
        self.per_conv_film_layers: Optional[nn.ModuleList] = None
        if self.film_mode == "off":
            pass                                    # no conditioning reaches the conv tower at all
        elif self.film_mode == "pre_conv":
            self.pre_film = FiLMLayer(int(meta_embed_dim), 2, film_init)
        elif self.film_mode == "post_conv":
            self.post_film = PerAssayFiLM(int(meta_embed_dim), self.out_per_assay, film_init)
        elif self.film_mode in ("per_conv", "per_conv_and_transformer"):
            self.per_conv_film_layers = nn.ModuleList([
                FiLMLayer(int(meta_embed_dim), 2 * (ch // self.num_tracks), film_init)
                for ch in out_channels_list
            ])
        else:
            raise ValueError(f"Unsupported film_mode={self.film_mode}")

    def lane_shapes(self, context_length: int, batch: str = "B") -> List[str]:
        """The `[B, L, T, C]` shape at the tower's input and after each block, as strings.

        Documentation that cannot drift: `tests/test_lane_view.py` asserts these against a real
        forward pass, so an edit to the channel schedule breaks the test rather than the comment.
        """
        seq, shapes = int(context_length), []
        shapes.append(f"[{batch}, {seq}, {self.num_tracks}, 1]")
        for ch, pool in zip(self._out_channels_list, self._pool_sizes):
            seq //= pool
            shapes.append(f"[{batch}, {seq}, {self.num_tracks}, {ch // self.num_tracks}]")
        return shapes

    def forward(self, x_signal: torch.Tensor, meta_embed: torch.Tensor,
                return_lanes: bool = False) -> torch.Tensor:
        """
        Args:
            x_signal:   [B, L, num_tracks] — raw signal (already transformed)
            meta_embed: [B, num_tracks, embed_dim]
            return_lanes: return the explicit `[B, L2, T, C]` lane view instead of the flat
                `[B, L2, T*C]`. The two are the same numbers in the same order — the flat form
                is what `MaskTokenInjector` and the fusion take, the lane form is what the
                decoder and the lane arms speak. Default False keeps every caller unchanged.
        Returns:
            [B, L2, out_channels] (or [B, L2, T, C] when `return_lanes`), L2 = L // pool^n_layers

        Per-track shapes, at the shipped 36-track / 768-bin panel:
            input    [B, 768, 36, 1]  ->  block 0  [B, 384, 36, 2]
                                          block 1  [B, 192, 36, 4]
                                          block 2  [B,  96, 36, 8]
        The convolution is grouped by track at every rung, so the C axis never mixes tracks.
        (`conv_norm` is a separate question — with the default "layer" the norm that follows
        each conv still pools its statistic across the whole panel. See `ConvBlock`.)
        """
        x = x_signal.permute(0, 2, 1).contiguous()  # [B, T*C, L] with C=1
        if self.pre_film is not None:
            x = self.pre_film(x, meta_embed)
        for i, block in enumerate(self.blocks):
            x = block(x)                            # C per track doubles, L halves
            if self.per_conv_film_layers is not None:
                x = self.per_conv_film_layers[i](x, meta_embed)
        x = x.permute(0, 2, 1).contiguous()  # [B, L2, T*C]
        if self.post_film is not None:
            x = self.post_film(x, meta_embed)
        if return_lanes:
            bsz, seq, channels = x.shape
            return x.view(bsz, seq, self.num_tracks, channels // self.num_tracks)
        return x


class DNAConvTower(nn.Module):
    """Standard (ungrouped) Conv1d tower for one-hot DNA sequence.

    Downsamples DNA length to match signal tower output length via alternating
    small (pool_size) and large (dna_pool_size) pooling steps.
    """

    def __init__(
        self,
        target_dim: int,
        n_cnn_layers: int,
        conv_kernel_size: int,
        pool_size: int,
        dna_pool_size: int,
        conv_norm: str,
        pool_order: str,
    ) -> None:
        super().__init__()
        channels = [4] + list(
            exponential_linspace_int(4, int(target_dim), int(n_cnn_layers) + 2)
        )
        blocks: List[nn.Module] = []
        total = int(n_cnn_layers) + 2
        for i in range(total):
            if str(pool_order) == "late":
                use_large_pool = i >= int(n_cnn_layers)
            else:
                use_large_pool = i < 2
            p = int(dna_pool_size) if use_large_pool else int(pool_size)
            blocks.append(ConvTower(
                in_ch=channels[i], out_ch=channels[i + 1],
                kernel_size=int(conv_kernel_size), groups=1,
                pool_size=p, norm=str(conv_norm),
            ))
        self.blocks = nn.ModuleList(blocks)
        self.out_channels = channels[-1]

    def forward(self, x_dna: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_dna: [B, 4, G] or [B, G, 4] — one-hot DNA
        Returns:
            [B, L2, out_channels]
        """
        if x_dna.ndim != 3:
            raise ValueError(f"x_dna must be rank-3, got shape={tuple(x_dna.shape)}")
        if x_dna.shape[1] == 4:
            x = x_dna.float()
        elif x_dna.shape[2] == 4:
            x = x_dna.permute(0, 2, 1).contiguous().float()
        else:
            raise ValueError(f"x_dna must have a dim=4, got {tuple(x_dna.shape)}")
        for block in self.blocks:
            x = block(x)
        return x.permute(0, 2, 1).contiguous()


# ---------------------------------------------------------------------------
# Fusion layers
# ---------------------------------------------------------------------------

class LinearFusion(nn.Module):
    """Concatenate signal + DNA features → linear project → GELU → optional norm."""

    def __init__(
        self,
        signal_dim: int,
        dna_dim: int,
        out_dim: int,
        dropout: float,
        fusion_norm: str = "layer",
        deep: bool = False,
    ) -> None:
        super().__init__()
        self.fusion_proj = nn.Linear(signal_dim + dna_dim, out_dim)
        self.gelu = nn.GELU()
        # deep=True adds one hidden Linear(out->out)+GELU before the norm
        self.hidden_projs = nn.ModuleList(
            [nn.Linear(out_dim, out_dim) for _ in range(1 if deep else 0)]
        )
        if fusion_norm == "layer":
            self.norm: nn.Module = nn.LayerNorm(out_dim)
        elif fusion_norm == "none":
            self.norm = nn.Identity()
        else:
            raise ValueError(f"Unsupported fusion_norm={fusion_norm}")
        self.dropout = nn.Dropout(dropout)

    def forward(self, signal: torch.Tensor, dna: torch.Tensor) -> torch.Tensor:
        if signal.shape[:2] != dna.shape[:2]:
            raise ValueError(
                f"Fusion sequence mismatch: signal={tuple(signal.shape)}, dna={tuple(dna.shape)}"
            )
        fused = self.gelu(self.fusion_proj(torch.cat([signal, dna], dim=-1)))
        for proj in self.hidden_projs:
            fused = self.gelu(proj(fused))
        return self.dropout(self.norm(fused))


# UNREACHABLE on the shipped path (fusion_mode='linear').
class GatedDNAFusion(nn.Module):
    """Gated fusion: signal * sigmoid(gate(dna)) + proj(dna)."""

    def __init__(
        self,
        signal_dim: int,
        dna_dim: int,
        dropout: float,
        fusion_norm: str = "layer",
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dna_dim, signal_dim)
        self.dna_proj = nn.Linear(dna_dim, signal_dim)
        self.gelu = nn.GELU()
        if fusion_norm == "layer":
            self.norm: nn.Module = nn.LayerNorm(signal_dim)
        elif fusion_norm == "none":
            self.norm = nn.Identity()
        else:
            raise ValueError(f"Unsupported fusion_norm={fusion_norm}")
        self.dropout = nn.Dropout(dropout)

    def forward(self, signal: torch.Tensor, dna: torch.Tensor) -> torch.Tensor:
        if signal.shape[:2] != dna.shape[:2]:
            raise ValueError(
                f"Fusion sequence mismatch: signal={tuple(signal.shape)}, dna={tuple(dna.shape)}"
            )
        gate = torch.sigmoid(self.gate_proj(dna))
        dna_contribution = self.dna_proj(dna)
        fused = signal * gate + dna_contribution
        return self.dropout(self.norm(self.gelu(fused)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_availability_from_meta(meta: torch.Tensor) -> torch.Tensor:
    """Derive per-assay availability [B, A+1] from metadata [B, n_rows, A+1].

    Only the FIRST FOUR rows vote. Availability is a per-assay fact and the four original covariates
    are per-assay; the 5th cell_id row is a per-sample fact that is deliberately NOT masked when an
    assay is clozed, so letting it vote would be wrong. Restricting the slice also makes a sentinel
    accidentally landing in row 4 a loud failure (the signal/meta availability equality check in
    `_prepare_signal` trips) rather than a silent blanking of the whole sample.
    """
    meta = meta[:, :4, :]
    has_cloze = (meta == CLOZE).any(dim=1)
    has_missing = (meta == MISSING).any(dim=1)
    avail = torch.ones_like(meta[:, 0, :], dtype=meta.dtype)
    avail = torch.where(has_missing, torch.full_like(avail, MISSING), avail)
    avail = torch.where(has_cloze, torch.full_like(avail, CLOZE), avail)
    return avail


def _infer_availability_from_signal(x_signal: torch.Tensor) -> torch.Tensor:
    """Derive per-assay availability [B, A+1] from signal [B, L, A+1]."""
    has_cloze = (x_signal == CLOZE).any(dim=1)
    has_missing = (x_signal == MISSING).any(dim=1)
    avail = torch.ones_like(x_signal[:, 0, :], dtype=x_signal.dtype)
    avail = torch.where(has_missing, torch.full_like(avail, MISSING), avail)
    avail = torch.where(has_cloze, torch.full_like(avail, CLOZE), avail)
    return avail


def _apply_signal_transform(x: torch.Tensor, mode: str) -> torch.Tensor:
    """Apply input signal compression, preserving sentinel values."""
    mode_l = str(mode).lower()
    if mode_l == "none":
        return x
    mask = (x != MISSING) & (x != CLOZE)
    if mode_l == "log1p":
        return torch.where(mask, torch.log1p(x), x)
    if mode_l == "arcsinh":
        return torch.where(mask, torch.asinh(x), x)
    raise ValueError(f"Unsupported signal_transform={mode}")


# UNREACHABLE on the shipped path (only DualAttentionEncoderBlock calls it).
def _get_divisible_heads(dim: int, preferred_heads: int) -> int:
    """Find largest head count <= preferred_heads that divides dim."""
    for h in range(preferred_heads, 0, -1):
        if dim % h == 0:
            return h
    return 1


# ---------------------------------------------------------------------------
# Dual attention (for transformer_type="dual")
#
# UNREACHABLE on the shipped path: both classes below are built only under
# transformer_type='dual'; the kit pins 'xtransformers'. Kept verbatim.
# ---------------------------------------------------------------------------

class RelativePositionBias(nn.Module):
    def __init__(self, num_heads: int, max_distance: int) -> None:
        super().__init__()
        self.max_distance = int(max_distance)
        self.relative_bias = nn.Parameter(
            torch.zeros(2 * self.max_distance - 1, num_heads)
        )
        nn.init.trunc_normal_(self.relative_bias, std=0.02)

    def forward(self, seq_len: int) -> torch.Tensor:
        pos = torch.arange(seq_len, device=self.relative_bias.device)
        rel_pos = pos[None, :] - pos[:, None] + self.max_distance - 1
        return self.relative_bias[rel_pos].permute(2, 0, 1).contiguous()


class DualAttentionEncoderBlock(nn.Module):
    """Sequence attention + channel attention + FFN with residual connections."""

    def __init__(
        self, d_model: int, num_heads: int, seq_length: int, dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.dropout = float(dropout)
        self.num_heads = _get_divisible_heads(self.d_model, int(num_heads))
        self.num_heads_chan = _get_divisible_heads(int(seq_length), int(num_heads))
        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.relative_bias = RelativePositionBias(
            self.num_heads, max(2, int(seq_length))
        )
        self.mha_channel = nn.MultiheadAttention(
            embed_dim=int(seq_length), num_heads=self.num_heads_chan,
            dropout=self.dropout, batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(2 * self.d_model, 2 * self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(2 * self.d_model, self.d_model),
            nn.Dropout(self.dropout),
        )
        self.norm_seq = nn.LayerNorm(self.d_model)
        self.norm_chan = nn.LayerNorm(self.d_model)
        self.norm_ffn = nn.LayerNorm(self.d_model)

    def _relative_multihead_attention(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        head_dim = self.d_model // self.num_heads
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(float(head_dim), device=x.device, dtype=x.dtype)
        )
        scores = scores + self.relative_bias(seq_len).unsqueeze(0)
        attn_weights = F.dropout(
            F.softmax(scores, dim=-1), p=self.dropout, training=self.training
        )
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.out_proj(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_attn = self._relative_multihead_attention(x)
        x_seq = self.norm_seq(x + seq_attn)
        x_trans = x.transpose(1, 2)
        chan_attn, _ = self.mha_channel(x_trans, x_trans, x_trans)
        x_chan = self.norm_chan(x + chan_attn.transpose(1, 2))
        ffn_out = self.ffn(torch.cat([x_seq, x_chan], dim=-1))
        return self.norm_ffn(x_seq + x_chan + ffn_out)


# ---------------------------------------------------------------------------
# Transformer normalisation options
# ---------------------------------------------------------------------------
# These values are passed straight through to `x_transformers.Encoder`, so they are only as stable
# as that package's keyword surface. Two defences, and the second is the one that earns its keep:
#
#   * the maps are EMPTY at the defaults, so the shipped call passes nothing extra and cannot be
#     affected by any of this;
#   * `tests/test_flags.py::test_every_flag_changes_the_model` builds a model at each non-default
#     value and asserts it differs from the default.
#
# The pinned x-transformers does assert on unrecognised keywords (`x_transformers.py:2198`), but only
# AFTER `groupby_prefix_and_trim` has routed everything starting `ff_`, `attn_` or `cross_attn_` into
# sub-dicts — and `attn_qk_norm` is one of those. So the assert is not a complete guard, and it is
# not a guarantee either: it is one line in a dependency nobody here controls. The test is.
#
# `resi_dual` IS ABSENT ON PURPOSE. It was in the agreed flag set, and the test above is exactly what
# found that the pinned version does not accept the keyword at all — offering the choice would have
# shipped a documented option that raises the moment anyone selects it.

TRANSFORMER_NORMS = ("layer", "rmsnorm", "simple_rmsnorm", "scalenorm")
TRANSFORMER_NORM_KW = {
    "layer": {},                                  # x-transformers' own default
    "rmsnorm": {"use_rmsnorm": True},
    "simple_rmsnorm": {"use_simple_rmsnorm": True},
    "scalenorm": {"use_scalenorm": True},
}

TRANSFORMER_PLACEMENTS = ("pre", "post", "sandwich")
TRANSFORMER_PLACEMENT_KW = {
    "pre": {},                                    # `pre_norm=True` is already in the base kwargs
    "post": {"pre_norm": False},
    "sandwich": {"sandwich_norm": True},          # norm before AND after each sub-layer
}


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class V2Encoder(nn.Module):
    """CANDI v2 encoder: signal conv → mask inject → DNA conv → fuse → transformer.

    This is a standalone module that takes raw signal, DNA, and metadata and
    produces a latent representation of shape [B, L2, d_model].
    """

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_tracks = int(cfg.num_assays) + 1  # +1 for control channel
        self.l1 = int(cfg.context_length)
        pool_factor = int(cfg.pool_size) ** int(cfg.n_cnn_layers)
        if self.l1 % pool_factor != 0:
            raise ValueError(
                f"context_length={self.l1} not divisible by "
                f"pool_size^n_cnn_layers={pool_factor}"
            )
        self.l2 = self.l1 // pool_factor
        self.resolution = int(cfg.dna_pool_size) ** 2
        self.required_dna_len = self.l1 * self.resolution
        self.missing_data_mode = str(cfg.missing_data_mode)
        self.film_mode = str(cfg.film_mode)
        self.transformer_type = str(cfg.transformer_type)
        # Checked here, before anything is built, so a typo fails on the submit line rather than
        # after the h5 is open — and with the valid set in the message, not a bare KeyError.
        if str(cfg.transformer_norm) not in TRANSFORMER_NORM_KW:
            raise ValueError(f"transformer_norm must be one of {TRANSFORMER_NORMS}; "
                             f"got {cfg.transformer_norm!r}")
        if str(cfg.transformer_norm_placement) not in TRANSFORMER_PLACEMENT_KW:
            raise ValueError(f"transformer_norm_placement must be one of {TRANSFORMER_PLACEMENTS}; "
                             f"got {cfg.transformer_norm_placement!r}")
        if str(cfg.film_init) not in FILM_INITS:
            raise ValueError(f"film_init must be one of {FILM_INITS}; got {cfg.film_init!r}")

        # -- Metadata embedding --
        self.metadata_embedding = MetadataEmbedding(
            num_assays=int(cfg.num_assays),
            embed_dim=int(cfg.metadata_embed_dim),
            use_layernorm=bool(cfg.meta_embed_layernorm),
            num_cells=int(cfg.num_cells),
        )

        # -- Signal conv tower --
        self.signal_tower = SignalConvTower(
            num_tracks=self.num_tracks,
            n_layers=int(cfg.n_cnn_layers),
            expansion_factor=int(cfg.expansion_factor),
            kernel_size=int(cfg.conv_kernel_size),
            pool_size=int(cfg.pool_size),
            meta_embed_dim=int(cfg.metadata_embed_dim),
            conv_norm=str(cfg.conv_norm),
            film_mode=self.film_mode,
            film_init=str(cfg.film_init),
        )

        # -- Missing data handling --
        self.mask_stem = None
        self.mask_injector: Optional[MaskTokenInjector] = None
        if self.missing_data_mode == "mask_stem":
            raise ValueError(
                "missing_data_mode='mask_stem' not shipped in candi "
                "(requires repo model.MaskStem); use 'mask_token'"
            )
        elif self.missing_data_mode == "mask_token":
            self.mask_injector = MaskTokenInjector(
                self.num_tracks, self.signal_tower.out_per_assay
            )
        else:
            raise ValueError(f"Unsupported missing_data_mode={self.missing_data_mode}")

        # -- DNA conv tower --
        signal_dim = self.signal_tower.out_channels
        self.d_model = int(cfg.d_model) if int(cfg.d_model) > 0 else signal_dim
        self.d_model_is_auto = int(cfg.d_model) == 0
        self.dna_tower = DNAConvTower(
            target_dim=signal_dim,
            n_cnn_layers=int(cfg.n_cnn_layers),
            conv_kernel_size=int(cfg.conv_kernel_size),
            pool_size=int(cfg.pool_size),
            dna_pool_size=int(cfg.dna_pool_size),
            conv_norm=str(cfg.conv_norm),
            pool_order=str(cfg.dna_pool_order),
        )

        # -- Fusion --
        fusion_mode = str(cfg.fusion_mode)
        fusion_norm = str(cfg.fusion_norm)
        if fusion_mode == "gated":
            if self.d_model != signal_dim:
                raise ValueError(
                    f"gated fusion requires d_model == signal_dim; "
                    f"got {self.d_model} vs {signal_dim}"
                )
            self.fusion: nn.Module = GatedDNAFusion(
                signal_dim=signal_dim, dna_dim=self.dna_tower.out_channels,
                dropout=float(cfg.dropout), fusion_norm=fusion_norm,
            )
        elif fusion_mode == "linear":
            self.fusion = LinearFusion(
                signal_dim=signal_dim, dna_dim=self.dna_tower.out_channels,
                out_dim=self.d_model, dropout=float(cfg.dropout),
                fusion_norm=fusion_norm, deep=bool(cfg.fusion_deep),
            )
        else:
            raise ValueError(f"Unsupported fusion_mode={fusion_mode}")

        # -- Transformer stack --
        if self.transformer_type == "dual":
            self.transformer_blocks = nn.ModuleList([
                DualAttentionEncoderBlock(
                    d_model=self.d_model, num_heads=int(cfg.nhead),
                    seq_length=self.l2, dropout=float(cfg.dropout),
                )
                for _ in range(int(cfg.n_transformer_layers))
            ])
        elif self.transformer_type == "xtransformers":
            # The kwargs are assembled rather than written inline so the DEFAULT call is character
            # for character the historical one: both `.update`s below are empty at the defaults, so
            # nothing new is passed and the RNG stream is untouched. A non-default value adds exactly
            # the one keyword it needs.
            xkw = dict(
                dim=self.d_model, depth=1, heads=int(cfg.nhead),
                rotary_pos_emb=True,
                attn_dropout=float(cfg.dropout),
                ff_dropout=float(cfg.dropout),
                ff_mult=4, pre_norm=True,
                attn_qk_norm=bool(cfg.attn_qk_norm),
            )
            xkw.update(TRANSFORMER_NORM_KW[str(cfg.transformer_norm)])
            xkw.update(TRANSFORMER_PLACEMENT_KW[str(cfg.transformer_norm_placement)])
            self.transformer_blocks = nn.ModuleList([
                XEncoder(**xkw) for _ in range(int(cfg.n_transformer_layers))
            ])
        elif self.transformer_type == "production_dual":
            # UNREACHABLE on the shipped path (transformer_type='xtransformers').
            raise ValueError(
                "transformer_type='production_dual' not shipped in candi"
            )
        else:
            raise ValueError(f"Unsupported transformer_type={self.transformer_type}")

        # -- Optional per-transformer-layer FiLM --
        self.transformer_film_layers: Optional[nn.ModuleList] = None
        if self.film_mode == "per_conv_and_transformer":
            self.transformer_film_layers = nn.ModuleList([
                TransformerFeatureFiLM(int(cfg.metadata_embed_dim), self.d_model,
                                       str(cfg.film_init))
                for _ in range(int(cfg.n_transformer_layers))
            ])

        # -- Optional RMSNorm on encoder output (before decoder) --
        self.output_norm: nn.Module = (
            RMSNormSeq(self.d_model) if bool(cfg.output_rms_norm) else nn.Identity()
        )

        # -- Stochastic depth: prob of dropping a transformer layer during training --
        self._transformer_layer_drop: float = float(cfg.transformer_layer_drop)
        if self._transformer_layer_drop != 0.0:
            raise ValueError(
                "transformer_layer_drop must be 0.0 in candi — it consumes global "
                "RNG inside forward and breaks step-for-step determinism"
            )

    def _prepare_signal(
        self, x_signal_t: torch.Tensor, x_meta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Zero out masked signal channels; return (prepared_signal, availability)."""
        availability_meta = _infer_availability_from_meta(x_meta)
        if self.missing_data_mode != "mask_token":
            return x_signal_t, availability_meta

        availability_signal = _infer_availability_from_signal(x_signal_t)
        if not torch.equal(availability_meta, availability_signal):
            raise ValueError(
                "Signal vs metadata assay availability mismatch. "
                "CANDI v2 requires assay-only masking (p_full_loci=0, p_chunks=0) "
                "when missing_data_mode=mask_token."
            )
        observed = (availability_meta != CLOZE) & (availability_meta != MISSING)
        x_zeroed = x_signal_t * observed.unsqueeze(1).to(x_signal_t.dtype)
        return x_zeroed, availability_meta

    def encode(
        self,
        x_signal: torch.Tensor,
        x_dna: torch.Tensor,
        x_meta: torch.Tensor,
        return_meta: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_signal: [B, L, A+1] — signal + control channel
            x_dna:    [B, 4, G]   — one-hot DNA
            x_meta:   [B, 4, A+1] — metadata
            return_meta: if True, also return metadata embeddings

        Returns:
            z: [B, L2, d_model]
            (z, meta_embed) if return_meta=True
        """
        x_signal_t = _apply_signal_transform(x_signal.float(), self.cfg.signal_transform)
        g = x_dna.shape[2] if x_dna.shape[1] == 4 else x_dna.shape[1]
        if g != x_signal.shape[1] * self.resolution:
            raise ValueError(
                f"DNA length {g} != context {x_signal.shape[1]} x "
                f"resolution {self.resolution}"
            )
        x_signal_t, availability = self._prepare_signal(x_signal_t, x_meta)

        meta_embed = self.metadata_embedding(x_meta.float())

        # Signal conv tower
        sig_input = x_signal_t
        if self.mask_stem is not None:
            raise ValueError(
                "missing_data_mode='mask_stem' not shipped in candi "
                "(requires repo model.MaskStem); use 'mask_token'"
            )
        sig = self.signal_tower(sig_input, meta_embed)

        # Mask token injection (after conv, before fusion)
        if self.mask_injector is not None:
            sig = self.mask_injector(sig, availability)

        # DNA conv tower
        dna = self.dna_tower(x_dna)

        # Fuse signal + DNA
        fused = self.fusion(sig, dna)

        # Transformer stack with optional per-layer FiLM and stochastic depth
        pooled_meta = meta_embed.mean(dim=1)
        for i, block in enumerate(self.transformer_blocks):
            if (
                self.training
                and self._transformer_layer_drop > 0.0
                and torch.rand(1).item() < self._transformer_layer_drop
            ):
                continue  # stochastic depth: skip FiLM + block together
            if self.transformer_film_layers is not None:
                fused = self.transformer_film_layers[i](fused, pooled_meta)
            fused = block(fused)

        fused = self.output_norm(fused)

        if return_meta:
            return fused, meta_embed
        return fused

    def forward(
        self,
        x_signal: torch.Tensor,
        x_dna: torch.Tensor,
        x_meta: torch.Tensor,
    ) -> torch.Tensor:
        return self.encode(x_signal, x_dna, x_meta, return_meta=False)
