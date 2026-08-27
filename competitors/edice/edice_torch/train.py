"""The eDICE training loop -- shared by the Roadmap gate and, later, the EIC retrain.

Paper defaults (Methods; `scripts/train_roadmap.py` agrees): arcsinh target, MSE loss in arcsinh
space, Adam at 3e-4 with no schedule and no weight decay, 50 epochs, embed_dim 256, 4 heads, one
attention layer, decoder 2 x 2048 with dropout 0.3, 120 target tracks masked per bin, batch 256
bins. RIVALS_PLAN §6.3 forbids tuning any of them.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict, field
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from edice_torch.data import Batch, FixedTargetSampler, TrainSampler
from edice_torch.model import CellAssayCrossFactoriser

__all__ = ["Config", "build_model", "train", "predict"]


@dataclass
class Config:
    """Paper defaults. Anything changed here must be justified in the method README (§6.3)."""

    embed_dim: int = 256
    n_attn_heads: int = 4
    n_attn_layers: int = 1
    intermediate_fc_dim: int = 128
    decoder_layers: int = 2
    decoder_hidden: int = 2048
    decoder_dropout: float = 0.3
    transformer_dropout: float = 0.1
    intermediate_fc_dropout: float = 0.0
    embedding_dropout: float = 0.0
    lr: float = 3e-4
    epochs: int = 50
    batch_size: int = 256
    n_targets: int = 120
    seed: int = 211  # the reference's own default (scripts/train_roadmap.py:39)


def build_model(n_cells: int, n_assays: int, cfg: Config) -> CellAssayCrossFactoriser:
    return CellAssayCrossFactoriser(
        n_cells, n_assays, embed_dim=cfg.embed_dim, n_attn_layers=cfg.n_attn_layers,
        n_attn_heads=cfg.n_attn_heads, intermediate_fc_dim=cfg.intermediate_fc_dim,
        decoder_layers=cfg.decoder_layers, decoder_hidden=cfg.decoder_hidden,
        decoder_dropout=cfg.decoder_dropout, transformer_dropout=cfg.transformer_dropout,
        intermediate_fc_dropout=cfg.intermediate_fc_dropout,
        embedding_dropout=cfg.embedding_dropout)


def _to_device(batch: Batch, device: torch.device):
    supports, sc, sa, tc, ta, y = batch
    tensors = [
        torch.as_tensor(np.ascontiguousarray(supports), dtype=torch.float32, device=device),
        torch.as_tensor(np.ascontiguousarray(sc), dtype=torch.long, device=device),
        torch.as_tensor(np.ascontiguousarray(sa), dtype=torch.long, device=device),
        torch.as_tensor(np.ascontiguousarray(tc), dtype=torch.long, device=device),
        torch.as_tensor(np.ascontiguousarray(ta), dtype=torch.long, device=device),
    ]
    truth = (None if y is None else
             torch.as_tensor(np.ascontiguousarray(y), dtype=torch.float32, device=device))
    return tensors, truth


@torch.no_grad()
def predict(model: CellAssayCrossFactoriser, sampler: FixedTargetSampler,
            device: torch.device) -> np.ndarray:
    """(n_bins, n_targets) in ARCSINH space -- the caller decides whether to `sinh` back."""
    model.eval()
    out: List[np.ndarray] = []
    for batch in sampler:
        tensors, _ = _to_device(batch, device)
        out.append(model(*tensors).float().cpu().numpy())
    return np.concatenate(out, axis=0)


@torch.no_grad()
def _val_loss(model: CellAssayCrossFactoriser, sampler: FixedTargetSampler,
              device: torch.device) -> float:
    model.eval()
    total, count = 0.0, 0
    for batch in sampler:
        tensors, truth = _to_device(batch, device)
        pred = model(*tensors)
        total += float(((pred - truth) ** 2).sum())
        count += truth.numel()
    return total / max(count, 1)


def train(model: CellAssayCrossFactoriser, train_sampler: TrainSampler, cfg: Config,
          device: torch.device, val_sampler: Optional[FixedTargetSampler] = None,
          on_epoch: Optional[Callable[[int, Dict], None]] = None) -> List[Dict]:
    """MSE in arcsinh space, Adam, no schedule. Returns the per-epoch log."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history: List[Dict] = []
    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.time()
        total, count = 0.0, 0
        for batch in train_sampler:
            tensors, truth = _to_device(batch, device)
            pred = model(*tensors)
            loss = torch.nn.functional.mse_loss(pred, truth)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss) * truth.numel()
            count += truth.numel()
        row = {"epoch": epoch, "train_mse": total / max(count, 1),
               "seconds": round(time.time() - t0, 1)}
        if val_sampler is not None:
            row["val_mse"] = _val_loss(model, val_sampler, device)
        history.append(row)
        print(f"epoch {epoch:3d}  " + "  ".join(f"{k}={v}" for k, v in row.items() if k != "epoch"),
              flush=True)
        if on_epoch is not None:
            on_epoch(epoch, row)
    return history
