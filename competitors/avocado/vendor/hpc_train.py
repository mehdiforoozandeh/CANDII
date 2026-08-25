#!/usr/bin/env python3
"""Train Avocado on one chromosome of one processing.  Runs on Fir, one GPU.

Two modes, matching the paper's own two-stage scheme:

  --mode shared    Fit everything -- cell-type factors, assay factors, the
                   network, and this chromosome's genomic factors.  Run once per
                   dataset, on chr20 (the chromosome Avocado's authors used).

  --mode genome    Load the shared parameters from a `shared` run, freeze them,
                   and fit only this chromosome's genomic factors.  Run for all
                   23 scored chromosomes so that every chromosome is produced by
                   an identical procedure -- including chr20, whose factors are
                   re-fitted from scratch rather than inherited from the shared
                   run, so no chromosome enjoys a training advantage the others
                   lack.

Training target is arcsinh(signal); 2% of (position, track) entries are held out
by a deterministic rule and never enter the loss, purely to monitor fit.  No
blind-test track is read at any point -- the input matrix has 312 columns, the
train+validation experiments.
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from avocado import Avocado, holdout_mask     # noqa: E402


def load_index(bridge_path, tracks_path):
    """Map each of the 312 training columns to a (cell, assay) index pair.

    The index space covers every cell type and assay in the whole EIC panel, so
    the 12 blind-test cell types get embeddings -- learned entirely from their
    own train/validation experiments, of which each has between 1 and 11.
    """
    rows = list(csv.DictReader(open(bridge_path)))
    cells = sorted({r["cell_id"] for r in rows})
    assays = sorted({r["assay_id"] for r in rows})
    cell_ix = {c: i for i, c in enumerate(cells)}
    assay_ix = {a: i for i, a in enumerate(assays)}
    by_name = {r["filename"]: r for r in rows}

    names = [l.strip() for l in open(tracks_path) if l.strip()]
    ci = np.array([cell_ix[by_name[n]["cell_id"]] for n in names], dtype=np.int64)
    ai = np.array([assay_ix[by_name[n]["assay_id"]] for n in names], dtype=np.int64)
    return names, ci, ai, cells, assays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["d2", "d3"])
    ap.add_argument("--chrom", required=True)
    ap.add_argument("--mode", required=True, choices=["shared", "genome"])
    ap.add_argument("--data-root", required=True, help="root of the binned matrices")
    ap.add_argument("--bridge", required=True)
    ap.add_argument("--init", help="shared-mode checkpoint to start from (--mode genome)")
    ap.add_argument("--out", required=True, help="checkpoint path to write")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-positions", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="learning rate for the cell-type factors, assay factors "
                         "and network -- parameters that receive a gradient on "
                         "every step")
    ap.add_argument("--genome-lr", type=float, default=1e-2,
                    help="learning rate for the genomic latent factors.  Higher "
                         "than --lr on purpose: a genomic position is visited "
                         "once per epoch, not once per step, so with a common "
                         "learning rate its factors would move a few hundredths "
                         "over the whole run and never leave their initialisation")
    ap.add_argument("--max-hours", type=float, default=11.0)
    ap.add_argument("--eval-every", type=int, default=500, help="steps between monitor evals")
    ap.add_argument("--log", help="jsonl training log")
    args = ap.parse_args()

    if os.path.exists(args.out):
        print(f"[train] {args.out} exists, skipping", flush=True)
        return 0

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ddir = os.path.join(args.data_root, args.dataset)
    t0 = time.time()
    Y = np.load(os.path.join(ddir, f"{args.chrom}.npy"))          # (n_bins, n_tracks)
    n_bins, n_tracks = Y.shape
    names, ci, ai, cells, assays = load_index(args.bridge,
                                              os.path.join(ddir, "tracks.txt"))
    assert len(names) == n_tracks, (len(names), n_tracks)
    print(f"[train {args.dataset}/{args.chrom}/{args.mode}] {n_bins} bins x "
          f"{n_tracks} tracks loaded in {time.time()-t0:.0f}s "
          f"({Y.nbytes/2**30:.1f} GiB); {len(cells)} cells, {len(assays)} assays",
          flush=True)

    # arcsinh once, on CPU, in place: this is the training target.
    np.arcsinh(Y, out=Y)
    Yt = torch.from_numpy(Y)
    cell_idx = torch.from_numpy(ci).to(dev)
    assay_idx = torch.from_numpy(ai).to(dev)

    model = Avocado(len(cells), len(assays), n_bins).to(dev)
    if args.mode == "genome":
        ck = torch.load(args.init, map_location=dev, weights_only=False)
        # The checkpoint's genomic factors belong to chr20 and are a different
        # length, so they are dropped rather than loaded -- this chromosome's
        # factors start from a fresh initialisation.  (Dropping them is required,
        # not cosmetic: load_state_dict raises on a size mismatch even with
        # strict=False.)
        shared_state = {k: v for k, v in ck["model"].items()
                        if not k.startswith(("g25.", "g250.", "g5k."))}
        res = model.load_state_dict(shared_state, strict=False)
        assert not res.unexpected_keys, res.unexpected_keys
        print(f"[train] loaded {len(shared_state)} shared tensors from {args.init}; "
              f"genomic factors re-initialised for {args.chrom}", flush=True)
        model.replace_genome(n_bins, dev)
        model.freeze_shared()
        groups = [{"params": model.genome_parameters(), "lr": args.genome_lr}]
    else:
        groups = [{"params": model.genome_parameters(), "lr": args.genome_lr},
                  {"params": model.shared_parameters(), "lr": args.lr}]

    n_gen = sum(p.numel() for p in model.genome_parameters())
    n_par = sum(p.numel() for g in groups for p in g["params"])
    print(f"[train] optimising {n_par/1e6:.1f}M parameters "
          f"({n_gen/1e6:.1f}M genomic at lr {args.genome_lr}, "
          f"{(n_par-n_gen)/1e6:.1f}M shared at lr {args.lr})", flush=True)
    opt = torch.optim.Adam(groups)

    B = args.batch_positions
    steps_per_epoch = n_bins // B
    rng = np.random.default_rng(0)
    if args.log:
        os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
    logf = open(args.log, "a") if args.log else None
    deadline = time.time() + args.max_hours * 3600
    step = 0
    stopped_early = False

    def evaluate(n_batches=20):
        """Mean squared error on held-out entries, on a fixed set of positions."""
        model.eval()
        g = np.random.default_rng(12345)
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for _ in range(n_batches):
                p = torch.from_numpy(
                    g.integers(0, n_bins, size=B).astype(np.int64))
                y = Yt[p].to(dev, non_blocking=True)
                pos = p.to(dev, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=(dev.type == "cuda")):
                    yh = model(pos, cell_idx, assay_idx)
                m = holdout_mask(pos, n_tracks, dev)
                d = (yh.float() - y)[m]
                tot += float((d * d).sum())
                cnt += int(m.sum())
        model.train()
        return tot / max(cnt, 1)

    print(f"[train] {steps_per_epoch} steps/epoch, {args.epochs} epochs, "
          f"batch {B} positions x {n_tracks} tracks", flush=True)
    t_start = time.time()
    for epoch in range(args.epochs):
        order = rng.permutation(n_bins)
        run, runN = 0.0, 0
        for s in range(steps_per_epoch):
            p = torch.from_numpy(order[s * B:(s + 1) * B])
            y = Yt[p].to(dev, non_blocking=True)
            pos = p.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=(dev.type == "cuda")):
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
                rec = {"dataset": args.dataset, "chrom": args.chrom,
                       "mode": args.mode, "epoch": epoch, "step": step,
                       "train_mse": run / runN, "holdout_mse": ho,
                       "elapsed_s": el, "steps_per_s": step / el}
                print(f"  epoch {epoch} step {step}: train {run/runN:.5f} "
                      f"holdout {ho:.5f} ({step/el:.2f} steps/s)", flush=True)
                if logf:
                    logf.write(json.dumps(rec) + "\n"); logf.flush()
                run, runN = 0.0, 0

            if time.time() > deadline:
                print(f"[train] wall-clock deadline reached at epoch {epoch} "
                      f"step {step}; stopping", flush=True)
                stopped_early = True
                break
        if stopped_early:
            break

    ho = evaluate(n_batches=100)
    el = time.time() - t_start
    print(f"[train] final holdout MSE {ho:.5f} after {step} steps "
          f"({el/3600:.2f} h)", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp"
    torch.save({"model": model.state_dict(),
                "cells": cells, "assays": assays, "tracks": names,
                "dataset": args.dataset, "chrom": args.chrom, "mode": args.mode,
                "n_bins": n_bins, "steps": step, "epochs_requested": args.epochs,
                "stopped_early": stopped_early,
                "holdout_mse": ho, "elapsed_s": el,
                "batch_positions": B, "lr": args.lr,
                "genome_lr": args.genome_lr}, tmp)
    os.replace(tmp, args.out)
    if logf:
        logf.write(json.dumps({"dataset": args.dataset, "chrom": args.chrom,
                               "mode": args.mode, "final": True, "step": step,
                               "holdout_mse": ho, "elapsed_s": el,
                               "stopped_early": stopped_early}) + "\n")
        logf.close()
    print(f"[train] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
