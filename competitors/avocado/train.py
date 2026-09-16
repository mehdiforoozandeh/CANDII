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

4. **`--select-every`: checkpoint selection on the V_ panel** (BENCHMARK_DESIGN.md 5, PI ruling
   2026-08-31).  005 kept the last epoch, and the 1-in-50 entry mask it logs is a different number
   from the one every other method is ranked by.  The loop below writes this model's V_ predictions
   to a 4.1 root and scores them with `candi.bench.external.score_external` -- the same function,
   the same truth read and the same window walk that scores CANDI -- so "uniform selection" is a
   fact about which code ran, not a claim about two metrics being alike.  **B_ is never read:** the
   panel comes from a V_-only regime derived by `index.py::write_select_regime`.  The best weights
   are written THE MOMENT the metric improves, so a run killed by walltime still yields a selected
   checkpoint.

5. **`--positions`: a BED-restricted training scope** (D32, BENCHMARK_DESIGN.md 3.1).  Under
   `configs/regime.eic_pilot.json` the shared fit trains on the contained bins of the Pilot Regions,
   packed onto the compact axis `index.py::region_layout` defines and `bin_store.py --regions`
   writes.  The flag names that axis; without it every row of the matrix is a training position, as
   before.

Two modes, matching the paper's own two-stage scheme (see `vendor/avocado.py`):

  --mode shared    Fit everything.  Run once, on the regime's train scope (chr19, or the BED).
  --mode genome    Load a `shared` run, freeze it, fit only this chromosome's genomic factors.
                   Run once per eval chromosome (three -- BENCHMARK_DESIGN.md 12.2).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "vendor"))
from avocado import Avocado, holdout_mask                  # noqa: E402  (vendored, unmodified)
from index import (assay_index, cell_index, load_regime, read_layout,  # noqa: E402
                   read_tracks, region_slots, write_select_regime)
from predict import write_manifest, write_predictions      # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", required=True)
    ap.add_argument("--chrom", required=True,
                    help="the matrix stem: a chromosome, or `regions` with --positions")
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
    ap.add_argument("--positions", help="regions_layout.csv -- restrict training to the D32 BED "
                                        "scope this matrix was binned over")
    ap.add_argument("--select-every", type=int, default=0,
                    help="score the V_ panel every N EPOCHS and keep the best checkpoint "
                         "(BENCHMARK_DESIGN.md 5). 0 = no selection, which is 005's behaviour and "
                         "is what 5 forbids for a trainable method.")
    ap.add_argument("--select-metric", default="mse",
                    help="the key of `score_external`'s macro roll-up that selects. `crps` needs a "
                         "sigma-table; without one Avocado is point-only and has no crps at all.")
    ap.add_argument("--select-arm", default="pval", choices=["pval", "count"],
                    help="Avocado emits -log10 p and nothing else, so `pval` is the only arm it "
                         "populates (B1b forbids inventing a depth for a count arm).")
    ap.add_argument("--select-patience", type=int, default=0,
                    help="stop when the selection metric has not improved for MORE than this many "
                         "epochs. 0 = off, and a value BELOW --select-every can never fire, which "
                         "is why slurm/train.sh couples the two rather than defaulting them apart.")
    ap.add_argument("--select-sigma-table", help="6.1 sigma-table json, if one has been fitted; "
                                                 "without it the distributional keys are absent")
    ap.add_argument("--select-batch-positions", type=int, default=8192)
    args = ap.parse_args(argv)

    # A BED scope and a selection loop cannot coexist, and the reason is Avocado's own shape rather
    # than a missing feature. `score_external` scores a WHOLE chromosome -- `read_track_arrays`
    # refuses an array that is not `floor(chr_len/25)` long -- and a BED fit holds genomic factors
    # only for the contained bins, so it cannot produce one. Refused loudly here rather than
    # silently downgraded: a run that quietly stopped selecting would be scored as if it had.
    if args.select_every and args.positions:
        raise SystemExit(
            "--select-every with --positions: the shared fit under a `regions` regime has genomic "
            "factors only inside the BED, and candi.bench.external scores whole chromosomes. There "
            "is no V_ panel this stage can be scored on. Select in the genome stage, which does "
            "hold a whole chromosome, and run this one without --select-every.")

    if os.path.exists(args.out):
        print(f"[train] {args.out} exists, skipping", flush=True)
        return 0

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)

    regime = load_regime(args.regime)
    # The FULL regime's index space, never the derived selection regime's: the model is built over
    # it, and a V_ target's embedding is its T_ partner's whichever file declared the pair.
    cells, cix = cell_index(regime)
    assays, aix = assay_index(regime)
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

    # D32. `--positions` names the compact axis `bin_store.py --regions` packed: the slots that are
    # a 25 bp bin lying wholly inside a Pilot Region, and the alignment slots between them. Only the
    # real slots are trained on and only they are ever drawn for the held-out monitor, so an
    # alignment slot's genomic factors stay at their init and no gradient is taken outside the BED.
    train_pos = None
    if args.positions:
        spans, n_slots = read_layout(args.positions)
        if n_slots != n_bins:
            raise SystemExit(f"{args.positions} describes a {n_slots}-slot axis but "
                             f"{args.chrom}.npy has {n_bins} rows -- they were not built together")
        train_pos = region_slots(spans)
        print(f"[train] BED scope: {len(train_pos)} contained bins over {len(spans)} region(s) of "
              f"{n_slots} slots ({n_slots - len(train_pos)} alignment slots, never trained on)",
              flush=True)

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
    steps_per_epoch = (n_bins if train_pos is None else len(train_pos)) // B

    def _draw(g, size):
        """`size` training positions, uniformly. Inside the BED scope when there is one."""
        if train_pos is None:
            return g.integers(0, n_bins, size=size).astype(np.int64)
        return train_pos[g.integers(0, len(train_pos), size=size)]
    if args.log:
        os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
    logf = open(args.log, "a") if args.log else None
    deadline = time.time() + args.max_hours * 3600
    stopped_early = False

    def evaluate(n_batches=20):
        """Mean squared error on held-out entries, on a fixed set of positions.

        005's monitor, kept as a monitor. It is NOT what selects any more -- it is an entry mask
        over training columns, and BENCHMARK_DESIGN.md 5 asks every method to select on V_.
        """
        model.eval()
        g = np.random.default_rng(12345)
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for _ in range(n_batches):
                p = torch.from_numpy(_draw(g, B))
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

    # --- V_ checkpoint selection (adaptation 4) --------------------------------------------------
    best = {"value": float("inf"), "epoch": -1, "step": -1}
    best_path = args.out + ".best.pt"
    sel_root = Path(args.out + ".select_pred")
    sel_curve: list = []
    sel_source = sel_expected = sel_sigma = None
    if args.select_every:
        from candi.bench.external import _expected, read_sigma_table, score_external
        from candi.bench.harness import open_source
        sel_regime = write_select_regime(args.regime, args.out + ".select_regime.json")
        # OPENED ONCE, outside the check, for the reason `src/candi/train.py` gives of its monitor:
        # selection compares epoch 6 against epoch 12, and that is only a paired comparison if both
        # scored the same positions. `chroms=` pins the scope to the chromosome this fit holds
        # genomic factors for -- Avocado has no representation of any other position, so there is
        # nowhere else it could be scored.
        sel_source = open_source(store=args.out + ".select_regime.json", chroms=(args.chrom,))
        sel_expected = _expected(sel_source)
        sel_sigma = read_sigma_table(args.select_sigma_table)
        write_manifest(sel_root, version="005-port-selection",
                       notes=f"mid-training V_ panel, {args.mode} fit on {args.chrom}")
        print(f"[train] SELECTION every {args.select_every} epoch(s) on "
              f"{len(sel_regime['eval_pairs'])} V_ pairs / {len(sel_expected)} tracks over "
              f"{args.chrom}, by macro.{args.select_arm}.{args.select_metric} through "
              f"candi.bench.external -- the scorer that ranks the board. 0 B_ pairs are read.",
              flush=True)
        if sel_sigma is None and args.select_metric in ("crps", "pit_ks", "gaussian_nll"):
            raise SystemExit(f"--select-metric {args.select_metric} is a distributional key and "
                             f"Avocado emits a point; pass --select-sigma-table or select on `mse`")
    else:
        print("[train] NO CHECKPOINT SELECTION: the last epoch will be kept. "
              "BENCHMARK_DESIGN.md 5 asks every trainable method to select on V_ -- pass "
              "--select-every to comply.", flush=True)

    def _write_ckpt(path, step, ho, el):
        """The checkpoint `predict.py` reads. One writer, so `.best.pt` and the final file agree."""
        tmp = path + ".tmp"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({"model": model.state_dict(),
                    "cells": cells, "assays": assays,
                    "tracks": [(r[1], r[2]) for r in rows],
                    "chrom": args.chrom, "mode": args.mode, "n_bins": n_bins, "steps": step,
                    "epochs_requested": args.epochs, "holdout_mse": ho, "elapsed_s": el,
                    "batch_positions": B, "lr": args.lr, "genome_lr": args.genome_lr,
                    "seed": args.seed, "regime": os.path.abspath(args.regime),
                    "bed_scope": (None if train_pos is None else
                                  {"layout": os.path.abspath(args.positions),
                                   "trained_slots": int(len(train_pos))}),
                    "selection": (None if not args.select_every else
                                  {"metric": f"macro.{args.select_arm}.{args.select_metric}",
                                   "panel": "V_", "scored_by": "candi.bench.external",
                                   **{k: best[k] for k in ("value", "epoch", "step")}})}, tmp)
        os.replace(tmp, path)

    def select(epoch, step, el):
        """Score this model's V_ panel, keep the weights if it improved. Returns True to stop."""
        t = time.time()
        write_predictions(model, expected=sel_expected, cix=cix, aix=aix, chrom=args.chrom,
                          n_bins=n_bins, root=sel_root, device=dev,
                          batch_positions=args.select_batch_positions)
        res = score_external(sel_source, sel_root, seed=args.seed, sigma_table=sel_sigma,
                             sigma_table_path=args.select_sigma_table)
        macro = res["macro"].get(args.select_arm, {})
        if args.select_metric not in macro:
            raise SystemExit(f"score_external's macro.{args.select_arm} carries "
                             f"{sorted(macro)} and not `{args.select_metric}`")
        value = float(macro[args.select_metric])
        sel_curve.append({"epoch": epoch, "step": step, "value": value,
                          "n_tracks": macro.get("n_tracks"), "select_s": time.time() - t})
        # Written the MOMENT it improves, not at the end: a run killed by walltime must still leave
        # a selected checkpoint behind, and this one has already saved the project once.
        improved = np.isfinite(value) and value < best["value"]
        if improved:
            best.update(value=value, epoch=epoch, step=step)
            _write_ckpt(best_path, step, float("nan"), el)
        print(f"  [select] epoch {epoch}: {args.select_arm}.{args.select_metric} {value:.5f} over "
              f"{macro.get('n_tracks')} V_ tracks{' *BEST*' if improved else ''} "
              f"(best {best['value']:.5f} @ epoch {best['epoch']}, {time.time() - t:.0f}s)",
              flush=True)
        if logf:
            logf.write(json.dumps({"chrom": args.chrom, "mode": args.mode, "epoch": epoch,
                                   "step": step, "select_metric": args.select_metric,
                                   "select_value": value, "select_best": best["value"],
                                   "select_best_epoch": best["epoch"]}) + "\n")
            logf.flush()
        stall = epoch - best["epoch"]
        if args.select_patience and best["epoch"] >= 0 and stall > args.select_patience:
            print(f"[train] EARLY STOP at epoch {epoch}: no V_ improvement since epoch "
                  f"{best['epoch']} (patience {args.select_patience}). Nothing is lost -- "
                  f"{best_path} already holds those weights.", flush=True)
            return True
        return False

    print(f"[train] {steps_per_epoch} steps/epoch, {args.epochs} epochs, batch {B} positions x "
          f"{n_tracks} tracks", flush=True)
    t_start = time.time()
    for epoch in range(epoch0, args.epochs):
        # Re-seeded per epoch so a resumed run walks the batches an uninterrupted run would have.
        order = np.random.default_rng(args.seed + epoch).permutation(
            n_bins if train_pos is None else train_pos)
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

        # The deadline is tested BEFORE the check, not only inside the step loop: a selection pass
        # is tens of minutes, and starting one past the deadline would overrun the walltime by that
        # much and lose the epoch it was about to save.
        if (args.select_every and time.time() < deadline
                and ((epoch + 1) % args.select_every == 0 or epoch == args.epochs - 1)):
            if select(epoch, step, time.time() - t_start):
                break

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

    # `--out` IS the checkpoint this run ships, so it carries the SELECTED weights. The last epoch
    # is kept beside it under `.last.pt` rather than thrown away -- the drift after the minimum is
    # the shape README.md 3 reads -- but nothing downstream resolves that name, so no stage can
    # predict from the unselected model by accident.
    if best["epoch"] >= 0 and best["epoch"] != epoch:
        _write_ckpt(args.out + ".last.pt", step, ho, el)
        os.replace(best_path, args.out)
        print(f"[train] keeping the BEST checkpoint (epoch {best['epoch']}, "
              f"{args.select_arm}.{args.select_metric}={best['value']:.5f}), not the last; "
              f"the last epoch is at {args.out}.last.pt", flush=True)
    else:
        _write_ckpt(args.out, step, ho, el)
        if os.path.exists(best_path):
            os.remove(best_path)               # the last epoch won; it is the same weights twice
    if os.path.exists(partial):
        os.remove(partial)
    if logf:
        logf.write(json.dumps({"chrom": args.chrom, "mode": args.mode, "final": True,
                               "step": step, "holdout_mse": ho, "elapsed_s": el,
                               "select_curve": sel_curve, "selected_epoch": best["epoch"],
                               "selected_value": (None if best["epoch"] < 0
                                                  else best["value"])}) + "\n")
        logf.close()
    if sel_source is not None:
        sel_source.close()
    print(f"[train] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
