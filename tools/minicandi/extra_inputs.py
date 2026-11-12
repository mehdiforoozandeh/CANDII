"""miniCANDI extra inputs (Amendment 3, chunk XM; pinned interface E): the modules that carry the
RNA-seq track, the cell's expression vector and a learned cell embedding into any rung.

  ExtraTrackStem(n_channels, d)   x [B, L, n], avail [B] -> [B, L, d]. A per-bin linear map; a
                                  sample whose `avail` is False gets one learned null row at
                                  every bin, so its output does not depend on its input values.
                                  The map and the null row start at 0, so a rung that adds the
                                  stem's output to its features starts as the rung without it.
  CellState(d_cov, n_genes, n_cells, dropout)
                                  batch -> (gamma, beta) [B, d_cov], the FiLM (feature-wise scale
                                  and shift) of the covariate embedding. A hidden code of width
                                  HIDDEN is the sum of (1) an MLP of `x_cell_vec` (a learned null
                                  code where `x_cell_vec_avail` is False) and (2) an embedding of
                                  `cell_id` (row 0 = unknown, itself learned). In training mode
                                  each part is sent to its null with probability `dropout`, per
                                  sample. `ladder.model.G` maps the code to (gamma, beta); its last
                                  layer is 0 with bias (1, 0), so gamma = 1 and beta = 0 for every
                                  input at initialisation.
  apply_cell_state(cov_emb [B, A, d_cov], gamma, beta) -> gamma * cov_emb + beta, per sample.
  extra_kwargs(cfg)               which of the modules a config asks for, as constructor kwargs.

Batch keys read (pinned interface E): `x_rna [B, L, 2]`, `x_rna_avail [B]` (the rung passes them
to the stem), `x_cell_vec [B, G]`, `x_cell_vec_avail [B]`, `cell_id [B]` (CellState). A missing
key is read as "not available" for every sample. Config keys: `rna_track`, `cell_vector`,
`cell_embedding`, `cell_state_dropout` (0.1); `n_genes` (default 2000, the width of
`expression.tsv`) and `n_cells` (the number of training cells; needed under `cell_embedding`).
A config without these keys asks for no module, so every rung stays as it was.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

_TOOLS = Path(__file__).resolve().parents[1]
for _p in (_TOOLS / "t118", _TOOLS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ladder.model import G  # noqa: E402

HIDDEN = 64
RNA_CHANNELS = 2          # plus and minus strand
N_GENES = 2000
CELL_STATE_DROPOUT = 0.1


def _avail(batch: dict, key: str, B: int, device) -> torch.Tensor:
    a = batch.get(key)
    if a is None:
        return torch.zeros(B, dtype=torch.bool, device=device)
    return a.to(device=device, dtype=torch.bool).reshape(B)


class ExtraTrackStem(nn.Module):
    """An input-only extra track (e.g. RNA-seq, 2 strands) to per-bin features `[B, L, d]`."""

    def __init__(self, n_channels: int = RNA_CHANNELS, d: int = 32):
        super().__init__()
        self.n_channels, self.d = int(n_channels), int(d)
        self.proj = nn.Linear(self.n_channels, self.d)
        self.null = nn.Parameter(torch.zeros(self.d))
        with torch.no_grad():
            self.proj.weight.zero_()
            self.proj.bias.zero_()

    def forward(self, x: torch.Tensor, avail: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.shape[-1] != self.n_channels:
            raise ValueError(f"ExtraTrackStem: x {tuple(x.shape)} must be [B, L, {self.n_channels}]")
        avail = avail.to(dtype=torch.bool).reshape(x.shape[0], 1, 1)
        # zero the unavailable inputs first, so a NaN there reaches neither output nor gradient
        h = self.proj(torch.where(avail, x, torch.zeros((), dtype=x.dtype, device=x.device)))
        return torch.where(avail, h, self.null.to(h.dtype))


class CellState(nn.Module):
    """Cell-state FiLM of the covariate embedding from an expression vector and/or a cell id."""

    def __init__(self, d_cov: int, n_genes: int = 0, n_cells: int = 0,
                 dropout: float = CELL_STATE_DROPOUT):
        super().__init__()
        self.d_cov, self.n_genes, self.n_cells = int(d_cov), int(n_genes), int(n_cells)
        self.dropout = float(dropout)
        if self.n_genes <= 0 and self.n_cells <= 0:
            raise ValueError("CellState needs n_genes > 0 (cell_vector) or n_cells > 0 "
                             "(cell_embedding)")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"cell_state_dropout {self.dropout} not in [0, 1)")
        if self.n_genes > 0:
            self.vec = nn.Sequential(nn.Linear(self.n_genes, HIDDEN), nn.ReLU(),
                                     nn.Linear(HIDDEN, HIDDEN))
            self.vec_null = nn.Parameter(torch.zeros(HIDDEN))
        if self.n_cells > 0:
            self.emb = nn.Embedding(self.n_cells + 1, HIDDEN)   # row 0 = unknown cell
        init = torch.cat([torch.ones(self.d_cov), torch.zeros(self.d_cov)])
        self.g = G(HIDDEN, 2 * self.d_cov, init)

    def _keep(self, a: torch.Tensor) -> torch.Tensor:
        if self.training and self.dropout > 0.0:
            a = a & (torch.rand(a.shape, device=a.device) >= self.dropout)
        return a

    @staticmethod
    def _size(batch: dict) -> tuple[int, torch.device]:
        for k in ("cell_id", "x_cell_vec", "x_cell_vec_avail", "x_avail", "x_counts"):
            if batch.get(k) is not None:
                return int(batch[k].shape[0]), batch[k].device
        raise ValueError("CellState: the batch carries none of cell_id, x_cell_vec, x_avail")

    def code(self, batch: dict) -> torch.Tensor:
        """The hidden code `[B, HIDDEN]` that G maps to (gamma, beta)."""
        B, dev = self._size(batch)
        z = torch.zeros(B, HIDDEN, device=dev)
        if self.n_genes > 0:
            a = self._keep(_avail(batch, "x_cell_vec_avail", B, dev))
            x = batch.get("x_cell_vec")
            if x is None:
                x = torch.zeros(B, self.n_genes, device=dev)
            if x.shape != (B, self.n_genes):
                raise ValueError(f"x_cell_vec {tuple(x.shape)} must be [{B}, {self.n_genes}]")
            x = torch.where(a[:, None], x.float(), torch.zeros((), device=dev))
            z = z + torch.where(a[:, None], self.vec(x), self.vec_null)
        if self.n_cells > 0:
            cid = batch.get("cell_id")
            cid = (torch.zeros(B, dtype=torch.long, device=dev) if cid is None
                   else cid.to(device=dev, dtype=torch.long).reshape(B))
            cid = torch.where((cid >= 0) & (cid <= self.n_cells), cid, torch.zeros_like(cid))
            cid = torch.where(self._keep(cid > 0), cid, torch.zeros_like(cid))
            z = z + self.emb(cid)
        return z

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        theta = self.g(self.code(batch))
        return theta[:, :self.d_cov], theta[:, self.d_cov:]


def apply_cell_state(cov_emb: torch.Tensor, gamma: torch.Tensor,
                     beta: torch.Tensor) -> torch.Tensor:
    """FiLM of the covariate embedding `[B, A, d_cov]` by per-sample `gamma, beta [B, d_cov]`."""
    if cov_emb.dim() != 3 or gamma.shape != (cov_emb.shape[0], cov_emb.shape[2]) \
            or beta.shape != gamma.shape:
        raise ValueError(f"apply_cell_state: cov_emb {tuple(cov_emb.shape)}, gamma "
                         f"{tuple(gamma.shape)}, beta {tuple(beta.shape)}")
    return gamma[:, None, :] * cov_emb + beta[:, None, :]


def _flag(cfg: dict, key: str) -> bool:
    v = cfg.get(key, False)
    if not isinstance(v, bool):
        raise ValueError(f"{key} must be a bool, got {v!r}")
    return v


def extra_kwargs(cfg: dict) -> dict:
    """The extra-input modules a config asks for:

    {"rna_track":  None or {"n_channels": 2}                         -> ExtraTrackStem(d=..., **)
     "cell_state": None or {"n_genes", "n_cells", "dropout"}         -> CellState(d_cov, **)}

    `n_genes` is 0 unless `cell_vector`; `n_cells` is 0 unless `cell_embedding`.
    """
    rna, vec, emb = _flag(cfg, "rna_track"), _flag(cfg, "cell_vector"), _flag(cfg, "cell_embedding")
    p = float(cfg.get("cell_state_dropout", CELL_STATE_DROPOUT))
    if not 0.0 <= p < 1.0:
        raise ValueError(f"cell_state_dropout {p} not in [0, 1)")
    cell = None
    if vec or emb:
        n_genes = int(cfg.get("n_genes", N_GENES)) if vec else 0
        n_cells = 0
        if emb:
            n_cells = cfg.get("n_cells")
            if not isinstance(n_cells, int) or isinstance(n_cells, bool) or n_cells < 1:
                raise ValueError(f"cell_embedding needs a positive int `n_cells`, got {n_cells!r}")
        if vec and n_genes < 1:
            raise ValueError(f"cell_vector needs n_genes >= 1, got {n_genes}")
        cell = {"n_genes": n_genes, "n_cells": n_cells, "dropout": p}
    return {"rna_track": {"n_channels": RNA_CHANNELS} if rna else None, "cell_state": cell}
