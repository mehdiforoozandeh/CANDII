"""Their two-stage, per-chromosome training schedule, in PyTorch. One job per chromosome.

Stage 1 (`Precamole`) is 3-way classification on terciles; stage 2 (`Guacamole`) is scalar
regression that starts from stage 1's trunk. Epoch counts, batch sizes and factor widths are
upstream's own, per chromosome, from `dataset3.UPSTREAM_HYPERPARAMS`.

**The sampler is the one deliberate departure.** Upstream builds a batch with a Python loop over
`batch_size` samples (`03_guacamole6_train.py:174-183`) — 10 000 to 21 000 iterations per step. On
CPU that cost 0.2 % of a 2.7 s step and did not matter; against a GPU step it would be the whole
run. `_batch` does the same thing as one fancy-index. Everything it computes is identical; only the
loop is gone.

**Contributor mode, and why the anchor uses `upstream`.** The PI's ruling is that anything we
report excludes the target from the average (`features.py`). The §7.4 anchor is not a reported
number — it is the validation step that asks whether the port reproduces their published rows, and
that question is only interpretable if the port is fed the features they fed theirs. So
`--contributor-mode upstream` is the anchor default and `loo` is what the later P1/P2 retrain on
our own EIC data uses. The mode is stamped into the checkpoint and into the manifest that
`emit.write_manifest` writes, so a run can never be mistaken for the other kind.

One approximation, stated once: stage 1's average-tercile input is always the **pooled** per-mark
average, in both modes. In `loo` mode the correct tercile would differ by one contributor out of
`k`, and computing it would mean a per-track rank over the whole chromosome. Stage 1 is pretraining
whose head is discarded — only the trunk survives into stage 2 — so this affects initialisation
alone. Stage 2, which produces every prediction, uses the mode-correct average and variance.

```bash
python -m lavawizard.train --cache <cache> --chrom chr21 --out <run> \
    --contributor-mode upstream
```
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn

from . import dataset3
from .model import Guacamole, Precamole
from .preprocess import CachedChrom

__all__ = ["Sampler", "train_chromosome", "main"]


class Sampler:
    """Upstream's generator, vectorised. Positions walk contiguously; tracks are drawn at random.

    Both halves are theirs: `genomic_25bp_idxs = arange(start, start+bs) % n_positions` with a
    cursor that advances every batch, and `idxs = np.random.randint(len(data), size=batch_size)`.
    Keeping the contiguous position walk matters — it is why an epoch is defined as
    `ceil(n_bins / batch_size)` steps and therefore covers the chromosome once.
    """

    def __init__(self, cache: CachedChrom, batch_size: int, mode: str, seed: int = 0):
        self.c = cache
        self.batch_size = int(batch_size)
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.cursor = 0
        self.steps_per_epoch = int(np.ceil(cache.n_bins / self.batch_size))

    def _batch(self) -> Tuple[np.ndarray, ...]:
        bs, n = self.batch_size, self.c.n_bins
        pos = (np.arange(self.cursor, self.cursor + bs) % n).astype(np.int64)
        self.cursor = (self.cursor + bs) % n
        tix = self.rng.integers(0, self.c.n_tracks, bs).astype(np.int64)
        x = self.c.values[tix, pos].astype(np.float32)
        avg, var = self.c.moments(tix, pos, x, self.mode)
        return tix, pos, x, avg, var

    def stage1(self, device) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        tix, pos, _, _, _ = self._batch()
        # Stage 1's own target and its average feature are BOTH terciles (`dict_3cat`,
        # `dict_avg3cat`). The average tercile is the pooled per-mark one — see the module docstring.
        y = self.c.tercile[tix, pos].astype(np.int64)
        m = self.c.mark_ix[tix]
        pooled = (self.c.sums[m, pos] / self.c.mark_count[m]).astype(np.float32)
        avg_ter = self._pooled_tercile(m, pooled)
        d = dict(
            celltype=torch.from_numpy(self.c.cell_ix[tix]).to(device),
            assay=torch.from_numpy(m).to(device),
            pos25=torch.from_numpy(pos).to(device),
            average_onehot=torch.from_numpy(
                np.eye(3, dtype=np.float32)[avg_ter]).to(device),
        )
        return d, torch.from_numpy(y).to(device)

    def _pooled_tercile(self, mark_ix: np.ndarray, value: np.ndarray) -> np.ndarray:
        """Tercile of the pooled per-mark average, against thresholds computed once per mark."""
        if not hasattr(self, "_thresholds"):
            th = np.zeros((len(self.c.marks), 2), dtype=np.float32)
            for j in range(len(self.c.marks)):
                a = self.c.sums[j] / max(int(self.c.mark_count[j]), 1)
                th[j] = np.quantile(a, [1 / 3, 2 / 3])
            self._thresholds = th
        lo = self._thresholds[mark_ix, 0]
        hi = self._thresholds[mark_ix, 1]
        return ((value > lo).astype(np.int64) + (value > hi).astype(np.int64))

    def stage2(self, device) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        tix, pos, x, avg, var = self._batch()
        d = dict(
            celltype=torch.from_numpy(self.c.cell_ix[tix]).to(device),
            assay=torch.from_numpy(self.c.mark_ix[tix]).to(device),
            pos25=torch.from_numpy(pos).to(device),
            average=torch.from_numpy(avg).to(device),
            variance=torch.from_numpy(var).to(device),
        )
        return d, torch.from_numpy(x).to(device)


def _run_stage(model: nn.Module, sampler: Sampler, *, epochs: int, lr: float, loss_fn,
               batch_fn, device, label: str, log_every: int = 50,
               max_steps: Optional[int] = None) -> Dict[str, float]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    spe = sampler.steps_per_epoch
    total = min(epochs * spe, max_steps) if max_steps else epochs * spe
    print(f"{label}: {epochs} epochs x {spe} steps = {epochs*spe} steps"
          + (f" (capped at {total})" if total != epochs * spe else ""), flush=True)
    t0 = time.time()
    step = 0
    last = float("nan")
    while step < total:
        d, y = batch_fn(device)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(**d), y)
        loss.backward()
        opt.step()
        last = float(loss.detach())
        step += 1
        if step % log_every == 0 or step == total:
            el = time.time() - t0
            print(f"  {label} {step}/{total}  loss {last:.6f}  "
                  f"{el/step*1000:.1f} ms/step  eta {(total-step)*el/step/60:.1f} min", flush=True)
    el = time.time() - t0
    return {"steps": step, "final_loss": last, "seconds": el, "ms_per_step": el / max(step, 1) * 1000}


def enable_tf32(on: bool = True) -> None:
    """TF32 tensor cores for the dense layers. **Training only — never for a parity run.**

    Measured on a `1g.10gb` MIG slice at chr21's batch of 10 000: a step is 136.4 ms in fp32 and
    57.4 ms with TF32. The split is `fwd 42.2 -> 16.1` and `fwd+bwd 124.8 -> 45.7`, with Adam a
    flat 11.7 ms either way, so the cost is the three 2048-wide matmuls — not the optimizer, and
    not the data (the in-RAM gather is 0.1 ms per batch).

    Safe here because this is a **retrain from scratch**: TF32 changes the arithmetic of a matmul,
    not the model, and nothing downstream compares these weights to anyone else's bit for bit.
    `parity_keras.py` turns it off explicitly, because that *is* a numerical comparison against
    Keras fp32 and TF32 would quietly widen the very tolerance it exists to measure.
    """
    torch.backends.cuda.matmul.allow_tf32 = bool(on)
    torch.backends.cudnn.allow_tf32 = bool(on)


def train_chromosome(cache_root: Path, chrom: str, out_dir: Path, *,
                     contributor_mode: str = "upstream", device: str = "cuda",
                     seed: int = 0, max_steps_per_stage: Optional[int] = None,
                     epoch_scale: float = 1.0, tf32: bool = True) -> Dict:
    """Both stages for one chromosome. Writes `guacamole_<chrom>.pt` and `train_<chrom>.json`."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)
    if dev.type == "cuda":
        enable_tf32(tf32)
    cache = CachedChrom(cache_root, chrom)
    sched = dataset3.schedule(chrom)
    factors = dataset3.factor_sizes(chrom)
    bs = sched["batch_size"]
    pre_ep = max(int(round(sched["pretrain_epochs"] * epoch_scale)), 1)
    tr_ep = max(int(round(sched["train_epochs"] * epoch_scale)), 1)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{chrom}: {cache.n_tracks} tracks, {cache.n_bins} bins, {len(cache.cells)} cells, "
          f"{len(cache.marks)} marks | bs={bs} pre={pre_ep} train={tr_ep} | {factors} | "
          f"mode={contributor_mode}", flush=True)

    common = dict(n_celltypes=len(cache.cells), n_assays=len(cache.marks),
                  n_positions=cache.n_bins, **factors)
    sampler = Sampler(cache, bs, contributor_mode, seed=seed)

    pre = Precamole(**common).to(dev)
    n_par = sum(p.numel() for p in pre.parameters())
    print(f"Precamole: {n_par:,} parameters", flush=True)
    s1 = _run_stage(pre, sampler, epochs=pre_ep, lr=5e-4,
                    loss_fn=nn.CrossEntropyLoss(), batch_fn=sampler.stage1, device=dev,
                    label="stage1", max_steps=max_steps_per_stage)

    gua = Guacamole(**common).to(dev).from_precamole(pre)
    del pre
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    n_par = sum(p.numel() for p in gua.parameters())
    print(f"Guacamole: {n_par:,} parameters", flush=True)
    s2 = _run_stage(gua, sampler, epochs=tr_ep, lr=1e-3,
                    loss_fn=nn.MSELoss(), batch_fn=sampler.stage2, device=dev,
                    label="stage2", max_steps=max_steps_per_stage)

    peak = (torch.cuda.max_memory_allocated(dev) / 2**30) if dev.type == "cuda" else 0.0
    reserved = (torch.cuda.max_memory_reserved(dev) / 2**30) if dev.type == "cuda" else 0.0
    ckpt = out_dir / f"guacamole_{chrom}.pt"
    torch.save({"state_dict": gua.state_dict(), "chrom": chrom, "config": common,
                "contributor_mode": contributor_mode, "seed": seed,
                "cells": cache.cells, "marks": cache.marks}, ckpt)

    record = {
        "chrom": chrom, "contributor_mode": contributor_mode, "seed": seed,
        "batch_size": bs, "pretrain_epochs": pre_ep, "train_epochs": tr_ep,
        "epoch_scale": epoch_scale, "factors": factors, "tf32": bool(tf32),
        "n_tracks": cache.n_tracks, "n_bins": cache.n_bins,
        "parameters": n_par, "stage1": s1, "stage2": s2,
        "gpu_peak_allocated_gib": round(peak, 3), "gpu_peak_reserved_gib": round(reserved, 3),
        "device": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
        "checkpoint": str(ckpt),
        "upstream": "github.com/ccchang0111/ENCODE_imputation_2019@d638b204",
    }
    (out_dir / f"train_{chrom}.json").write_text(json.dumps(record, indent=1) + "\n",
                                                 encoding="utf-8")
    print(f"\n{chrom}: peak GPU {peak:.2f} GiB allocated / {reserved:.2f} GiB reserved", flush=True)
    print(f"{chrom}: stage1 {s1['ms_per_step']:.1f} ms/step, stage2 {s2['ms_per_step']:.1f} ms/step",
          flush=True)
    return record


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--chrom", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--contributor-mode", default="upstream", choices=("upstream", "loo"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps-per-stage", type=int, default=None,
                   help="smoke runs: cap each stage, for throughput and memory measurement")
    p.add_argument("--epoch-scale", type=float, default=1.0,
                   help="scale upstream's epoch counts; 1.0 is their schedule")
    p.add_argument("--no-tf32", action="store_true",
                   help="force fp32 matmuls; 2.4x slower, and only needed for a numerical study")
    ns = p.parse_args(argv)
    train_chromosome(ns.cache, ns.chrom, ns.out, contributor_mode=ns.contributor_mode,
                     device=ns.device, seed=ns.seed,
                     max_steps_per_stage=ns.max_steps_per_stage, epoch_scale=ns.epoch_scale,
                     tf32=not ns.no_tf32)
    return 0


if __name__ == "__main__":              # run as `python -m lavawizard.train`
    raise SystemExit(main())
