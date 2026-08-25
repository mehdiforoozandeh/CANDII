"""The validation gate (RIVALS_PLAN §7.3): reproduce eDICE's published Roadmap numbers.

The target is Supplementary Table 2 of Hawkins-Hooker et al. 2023 -- "Performance metrics for the
imputation of the 203 test tracks on chromosome 21":

    eDICE   GW Corr 0.735 +- 0.018      MSE Global 0.091 +- 0.005

Setup, from the paper's Methods and the reference README's own reproduction command: the PredictD
splits over 1032 Roadmap chr21 tracks (709 train / 102 val / 203 test), supports = train u val,
targets = the 203 test tracks, 50 epochs, arcsinh, Adam 3e-4, 120 targets per bin.

    # the packaged sample (10k bins) -- a smoke run, no published number to hit
    python run_roadmap.py --h5 sample_data/roadmap/SAMPLE_chr21_roadmap_train.h5 \
        --idmap sample_data/roadmap/idmap.json --splits sample_data/roadmap/predictd_splits.json \
        --epochs 20 --train-splits train --out out/sample

    # the gate: full chr21, doi:10.17617/3.VKEFB6
    python run_roadmap.py --h5 data/roadmap_tracks_shuffled.h5 --idmap .../idmap.json \
        --splits .../predictd_splits.json --epochs 50 --train-splits train val --out out/gate

Fg/Bg and the MACS classification columns of Supplementary Table 2 need per-track MACS2 peak calls
on Roadmap, which the deposit does not ship. The gate is therefore on the two genome-wide columns,
and the README says so rather than quietly reporting four numbers out of eight.
"""
from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from edice_torch.data import FixedTargetSampler, RoadmapH5, TrainSampler, read_idmap, read_splits
from edice_torch.metrics import gate_report
from edice_torch.train import Config, build_model, predict, train

#: Supplementary Table 2, eDICE row, genome-wide columns. Quoted, never recomputed.
PUBLISHED = {
    "source": "Hawkins-Hooker et al. 2023, Supplementary Table 2 (203 chr21 test tracks)",
    "gw_corr": {"mean": 0.735, "sem": 0.018},
    "mse_global": {"mean": 0.091, "sem": 0.005},
    "n_parameters": "~ 6M (Supplementary Table 1)",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--h5", required=True, help="Roadmap HDF5 (targets + track_names)")
    p.add_argument("--idmap", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--train-splits", nargs="+", default=["train"],
                   help="splits used as SUPPORTS and as training panel; the gate uses train val")
    p.add_argument("--target-split", default="test")
    p.add_argument("--epochs", type=int, default=Config.epochs)
    p.add_argument("--n-targets", type=int, default=Config.n_targets)
    p.add_argument("--batch-size", type=int, default=Config.batch_size)
    p.add_argument("--lr", type=float, default=Config.lr)
    p.add_argument("--embed-dim", type=int, default=Config.embed_dim)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--max-bins", type=int, default=None, help="debug: truncate the bin axis")
    p.add_argument("--val-bins", type=int, default=50_000,
                   help="bins in the per-epoch diagnostic pass; 0 for all. The SCORED prediction "
                        "always uses every bin.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    cfg = Config(embed_dim=args.embed_dim, lr=args.lr, epochs=args.epochs,
                 batch_size=args.batch_size, n_targets=args.n_targets, seed=args.seed)

    ds = RoadmapH5(args.h5)
    cell2id, assay2id = read_idmap(args.idmap)
    splits = read_splits(args.splits)

    known = set(ds.track_names)
    support_tracks = [t for s in args.train_splits for t in splits[s] if t in known]
    target_tracks = [t for t in splits[args.target_split] if t in known]
    print(f"{len(ds.track_names)} tracks in h5; {len(support_tracks)} support/train, "
          f"{len(target_tracks)} target; {ds.n_bins} bins", flush=True)

    # arcsinh once, up front -- the reference transforms inside the generator, same thing.
    support_raw = ds.load(support_tracks, n_bins=args.max_bins)
    target_raw = ds.load(target_tracks, n_bins=args.max_bins)
    support_x = np.arcsinh(support_raw)
    target_x = np.arcsinh(target_raw)

    s_cells, s_assays = ds.ids_for(support_tracks, cell2id, assay2id)
    t_cells, t_assays = ds.ids_for(target_tracks, cell2id, assay2id)

    model = build_model(len(cell2id), len(assay2id), cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,}", flush=True)

    train_sampler = TrainSampler(support_x, s_cells, s_assays, n_targets=cfg.n_targets,
                                 batch_size=cfg.batch_size, rng=rng)
    # The panel watched during training IS the target panel. eDICE's split is over TRACKS, not bins,
    # and the reference does the same (dataset_fit evaluates on a track split every epoch) -- so the
    # curve is diagnostic only. Nothing selects on it: RIVALS_PLAN §6.3 gives one seed and paper
    # defaults, and the scored model is the one at the final epoch.
    #
    # Watched on a random BIN SUBSET, because at the gate's scale a full pass costs about a third of
    # a training epoch and would add hours to the run for a curve nothing acts on. The SCORED
    # prediction below always uses every bin.
    watch = np.arange(support_x.shape[0])
    if args.val_bins and len(watch) > args.val_bins:
        watch = np.sort(rng.choice(len(watch), size=args.val_bins, replace=False))
    watch_sampler = FixedTargetSampler(support_x[watch], s_cells, s_assays, t_cells, t_assays,
                                       truth=target_x[watch], batch_size=cfg.batch_size)
    eval_sampler = FixedTargetSampler(support_x, s_cells, s_assays, t_cells, t_assays,
                                      truth=target_x, batch_size=cfg.batch_size)

    device = torch.device(args.device)
    history = train(model, train_sampler, cfg, device, val_sampler=watch_sampler)

    pred_x = predict(model, eval_sampler, device)
    report = gate_report(target_raw, np.sinh(pred_x), target_x, pred_x)

    result = {
        "published": PUBLISHED,
        "ours": report,
        "config": asdict(cfg),
        "n_parameters": n_params,
        "h5": str(Path(args.h5).resolve()),
        "n_bins_used": int(support_x.shape[0]),
        "n_watch_bins": int(len(watch)),
        "train_splits": args.train_splits,
        "target_split": args.target_split,
        "n_support_tracks": len(support_tracks),
        "n_target_tracks": len(target_tracks),
        "n_cells": len(cell2id),
        "n_assays": len(assay2id),
        "history": history,
        "torch": torch.__version__,
        "platform": platform.platform(),
    }
    (out / "gate.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(out / "test_preds.npz", pred_arcsinh=pred_x.astype(np.float32),
                        tracks=np.asarray(target_tracks))
    torch.save(model.state_dict(), out / "model.pt")

    raw = report["raw"]
    print("\n--- gate ---")
    print(f"published   GW Corr {PUBLISHED['gw_corr']['mean']:.3f} "
          f"+- {PUBLISHED['gw_corr']['sem']:.3f}   "
          f"MSE {PUBLISHED['mse_global']['mean']:.3f} +- {PUBLISHED['mse_global']['sem']:.3f}")
    print(f"ours (raw)  GW Corr {raw['pearson']['mean']:.3f} +- {raw['pearson']['sem']:.3f}   "
          f"MSE {raw['mse']['mean']:.3f} +- {raw['mse']['sem']:.3f}")
    print(f"wrote {out / 'gate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
