#!/usr/bin/env python3
"""Train Avocado on one chromosome of our EIC store.  One GPU.

Adapted from `vendor/hpc_train.py` (Max's 005, md5 29ec22235acfbc9f5fcb4d4368f0e51b).  The model,
the objective, the optimiser, the two learning rates, the batch shape, the arcsinh target and the
deterministic 1-in-50 held-out entry mask are all the vendored code, untouched -- `Avocado` and
`holdout_mask` are IMPORTED from `vendor/avocado.py` rather than copied.  Three things changed:

1. **Where the columns come from.**  005 read a `bridge.csv` mapping challenge filenames to
   `C##`/`M##` codes.  We read `tracks.csv`, written by `bin_store.py` from the regime, whose
   cell/assay indices come from `index.py`.  That is also where RIVALS_PLAN.md 6.2 is enforced --
   the matrix has 267 columns, one per TRAINING-split track, and no V_ or B_ column exists to
   train on.

2. **Resume.**  005's `--max-hours` wrote the final checkpoint and stopped.  Ours writes a
   `.partial` carrying the optimiser state and the epoch counter, so a run that outgrows one job's
   walltime continues in the next instead of restarting.  Nothing about the optimisation changes:
   the per-epoch shuffle is drawn from a generator re-seeded to `seed + epoch`, so the sequence of
   batches a resumed run sees is the sequence an uninterrupted run would have seen.

3. **`--seed`.**  005 hard-coded `default_rng(0)` for the shuffle and torch's global seed for the
   init.  B2 wants one documented seed, so it is a flag that defaults to 0 -- the value 005 used.

Two modes, matching the paper's own two-stage scheme (see `vendor/avocado.py`):

  --mode shared    Fit everything.  Run once, on chr20.
  --mode genome    Load a `shared` run, freeze it, fit only this chromosome's genomic factors.
                   Run for all 23 chromosomes -- including chr20, refitted from a fresh init, so
                   no chromosome enjoys a training advantage the others lack.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "vendor"))
from avocado import Avocado, holdout_mask                  # noqa: E402  (vendored, unmodified)
from index import assay_index, cell_index, load_regime, read_tracks   # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", required=True)
    ap.add_argument("--chrom", required=True)
    ap.add_argument("--mode", required=True, choices=["shared", "genome"])
    ap.add_argument("--data-root", required=True, help="root of the binned matrices")
    ap.add_argument("--init", help="shared-mode checkpoint to start from (--mode genome)")
    ap.add_argument("--out", required=True, help="checkpoint path to write")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-positions", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="cell-type factors, assay factors and network -- parameters that receive "
                         "a gradient on every step")
    ap.add_argument("--genome-lr", type=float, default=1e-2,
                    help="genomic latent factors. Higher than --lr on purpose: a position is "
                         "visited once per epoch, not once per step")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-hours", type=float, default=11.0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--log", help="jsonl training log")
    ap.add_argument("--smoke-steps", type=int, default=0,
                    help="stop after N optimiser steps and write nothing. A throughput and "
                         "loss-is-falling probe, not a run.")
    args = ap.parse_args(argv)

    if os.path.exists(args.out):
        print(f"[train] {args.out} exists, skipping", flush=True)
        return 0

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)

    regime = load_regime(args.regime)
    cells, _ = cell_index(regime)
    assays, _ = assay_index(regime)
    rows = read_tracks(os.path.join(args.data_root, "tracks.csv"))

    t0 = time.time()
    Y = np.load(os.path.join(args.data_root, f"{args.chrom}.npy"))     # (n_bins, n_tracks)
    n_bins, n_tracks = Y.shape
    assert len(rows) == n_tracks, (len(rows), n_tracks)
    ci = np.array([r[3] for r in rows], dtype=np.int64)
    ai = np.array([r[4] for r in rows], dtype=np.int64)
    print(f"[train {args.chrom}/{args.mode}] {n_bins} bins x {n_tracks} tracks loaded in "
          f"{time.time() - t0:.0f}s ({Y.nbytes / 2**30:.1f} GiB); {len(cells)} cells, "
          f"{len(assays)} assays", flush=True)

    # arcsinh once, on CPU, in place: this is the training target.
    np.arcsinh(Y, out=Y)
    Yt = torch.from_numpy(Y)
    cell_idx = torch.from_numpy(ci).to(dev)
    assay_idx = torch.from_numpy(ai).to(dev)

    model = Avocado(len(cells), len(assays), n_bins).to(dev)
    if args.mode == "genome":
        ck = torch.load(args.init, map_location=dev, weights_only=False)
        # The checkpoint's genomic factors belong to chr20 and are a different length, so they are
        # dropped rather than loaded -- this chromosome's factors start from a fresh init.
        shared_state = {k: v for k, v in ck["model"].items()
                        if not k.startswith(("g25.", "g250.", "g5k."))}
        res = model.load_state_dict(shared_state, strict=False)
        assert not res.unexpected_keys, res.unexpected_keys
        print(f"[train] loaded {len(shared_state)} shared tensors from {args.init}; genomic "
              f"factors re-initialised for {args.chrom}", flush=True)
        model.replace_genome(n_bins, dev)
        model.freeze_shared()
        groups = [{"params": model.genome_parameters(), "lr": args.genome_lr}]
    else:
        groups = [{"params": model.genome_parameters(), "lr": args.genome_lr},
                  {"params": model.shared_parameters(), "lr": args.lr}]

    n_gen = sum(p.numel() for p in model.genome_parameters())
    n_par = sum(p.numel() for g in groups for p in g["params"])
    print(f"[train] optimising {n_par / 1e6:.1f}M parameters ({n_gen / 1e6:.1f}M genomic at lr "
          f"{args.genome_lr}, {(n_par - n_gen) / 1e6:.1f}M shared at lr {args.lr})", flush=True)
    opt = torch.optim.Adam(groups)

    # -- resume (adaptation 2) ------------------------------------------------------------------
    partial = args.out + ".partial"
    epoch0, step = 0, 0
    if args.smoke_steps == 0 and os.path.exists(partial):
        pk = torch.load(partial, map_location=dev, weights_only=False)
        model.load_state_dict(pk["model"])
        opt.load_state_dict(pk["opt"])
        epoch0, step = int(pk["epoch_done"]), int(pk["step"])
        print(f"[train] resumed from {partial} at epoch {epoch0}, step {step}", flush=True)

    B = args.batch_positions
    steps_per_epoch = n_bins // B
    if args.log:
        os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
    logf = open(args.log, "a") if args.log else None
    deadline = time.time() + args.max_hours * 3600
    stopped_early = False

    def evaluate(n_batches=20):
        """Mean squared error on held-out entries, on a fixed set of positions."""
        model.eval()
        g = np.random.default_rng(12345)
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for _ in range(n_batches):
                p = torch.from_numpy(g.integers(0, n_bins, size=B).astype(np.int64))
                y = Yt[p].to(dev, non_blocking=True)
                pos = p.to(dev, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
                    yh = model(pos, cell_idx, assay_idx)
                m = holdout_mask(pos, n_tracks, dev)
                d = (yh.float() - y)[m]
                tot += float((d * d).sum())
                cnt += int(m.sum())
        model.train()
        return tot / max(cnt, 1)

    print(f"[train] {steps_per_epoch} steps/epoch, {args.epochs} epochs, batch {B} positions x "
          f"{n_tracks} tracks", flush=True)
    t_start = time.time()
    for epoch in range(epoch0, args.epochs):
        # Re-seeded per epoch so a resumed run walks the batches an uninterrupted run would have.
        order = np.random.default_rng(args.seed + epoch).permutation(n_bins)
        run, runN = 0.0, 0
        for s in range(steps_per_epoch):
            p = torch.from_numpy(order[s * B:(s + 1) * B])
            y = Yt[p].to(dev, non_blocking=True)
            pos = p.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
                yh = model(pos, cell_idx, assay_idx)
            keep = ~holdout_mask(pos, n_tracks, dev)
            d = (yh.float() - y) * keep
            loss = (d * d).sum() / keep.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run += float(loss); runN += 1
            step += 1

            if step % args.eval_every == 0:
                ho = evaluate()
                el = time.time() - t_start
                rec = {"chrom": args.chrom, "mode": args.mode, "epoch": epoch, "step": step,
                       "train_mse": run / runN, "holdout_mse": ho, "elapsed_s": el,
                       "steps_per_s": step / el}
                print(f"  epoch {epoch} step {step}: train {run / runN:.5f} holdout {ho:.5f} "
                      f"({step / el:.2f} steps/s)", flush=True)
                if logf:
                    logf.write(json.dumps(rec) + "\n"); logf.flush()
                run, runN = 0.0, 0

            if args.smoke_steps and step >= args.smoke_steps:
                el = time.time() - t_start
                mem = (torch.cuda.max_memory_allocated() / 2**30) if dev.type == "cuda" else 0.0
                print(f"[smoke] {step} steps in {el:.0f}s = {step / el:.2f} steps/s; "
                      f"peak CUDA {mem:.2f} GiB; train_mse {loss.item():.5f}; "
                      f"holdout {evaluate():.5f}", flush=True)
                print("[smoke] writing nothing.", flush=True)
                return 0

            if time.time() > deadline:
                print(f"[train] wall-clock deadline reached at epoch {epoch} step {step}; "
                      f"stopping", flush=True)
                stopped_early = True
                break
        if stopped_early:
            break
        # a completed epoch is a resumable point
        tmpp = partial + ".tmp"
        os.makedirs(os.path.dirname(os.path.abspath(partial)), exist_ok=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch_done": epoch + 1, "step": step}, tmpp)
        os.replace(tmpp, partial)

    if stopped_early:
        # Not done: leave the .partial for the next job and write no final checkpoint, so a
        # downstream stage never picks up a half-trained model thinking it is finished.
        if logf:
            logf.write(json.dumps({"chrom": args.chrom, "mode": args.mode, "partial": True,
                                   "step": step}) + "\n")
            logf.close()
        print(f"[train] partial checkpoint at {partial}; re-submit to continue", flush=True)
        return 0

    ho = evaluate(n_batches=100)
    el = time.time() - t_start
    print(f"[train] final holdout MSE {ho:.5f} after {step} steps ({el / 3600:.2f} h)", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".tmp"
    torch.save({"model": model.state_dict(),
                "cells": cells, "assays": assays,
                "tracks": [(r[1], r[2]) for r in rows],
                "chrom": args.chrom, "mode": args.mode, "n_bins": n_bins, "steps": step,
                "epochs_requested": args.epochs, "holdout_mse": ho, "elapsed_s": el,
                "batch_positions": B, "lr": args.lr, "genome_lr": args.genome_lr,
                "seed": args.seed, "regime": os.path.abspath(args.regime)}, tmp)
    os.replace(tmp, args.out)
    if os.path.exists(partial):
        os.remove(partial)
    if logf:
        logf.write(json.dumps({"chrom": args.chrom, "mode": args.mode, "final": True,
                               "step": step, "holdout_mse": ho, "elapsed_s": el}) + "\n")
        logf.close()
    print(f"[train] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
