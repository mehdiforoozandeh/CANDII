"""eDICE on our EIC data: train on the 51 training cells, predict the declared pairs, emit §4.1.

    # 1. train
    python run_eic.py train --regime configs/regime.eic_val.json --out runs/eic --epochs 50
    # 2. predict -> a §4.1 prediction root (P1 = the regime's eval chroms; P2 = --chroms all)
    python run_eic.py predict --regime configs/regime.eic_val.json \
        --model runs/eic/model.pt --out preds/edice_p1
    # 3. score with OUR instrument, not eDICE's
    python -m candi.bench.external --store configs/regime.eic_val.json \
        --pred preds/edice_p1 --out scores/edice_p1.json --sigma-table preds/edice_sigma.json

Two things are pre-registered and appear in every eDICE row (see `eic_panel.py` and the README):
the 51-cell training restriction (§6.2), and the transductive substitution that gives a `V_X`
target its paired `T_X` cell's embedding (§7.3).

The prediction pass emits `signal_mu` only. eDICE predicts a point in `-log10 p`; it has no count
head, and B1b forbids inventing a read depth to manufacture one, so the count arm and the peak head
come out as ABSENT KEYS. `peak_score` is left out too, so `candi.bench.external` records the
coverage-ranking fallback itself (`has_peak_head=False`) rather than us pre-empting it. The forecast
distribution arrives later, from the σ-table `fit_sigma.py` writes (§6.1).
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import List

import numpy as np
import torch

from edice_torch.data import FixedTargetSampler, TrainSampler
from edice_torch.train import Config, build_model, predict as run_predict, train as run_train
from eic_panel import (Panel, TrackRequest, assert_no_eval_leakage, load_regime, read_slab,
                       requested_tracks, training_panel)

METHOD = "eDICE (PyTorch reimplementation)"
#: The pre-registered caveat, copied verbatim into every manifest so no table can lose it.
CAVEAT = ("transductive: a V_X/B_X target is queried with its paired T_X cell's learned embedding "
          "(RIVALS_PLAN.md §7.3). eDICE has no embedding for a cell it never trained on.")


#: eDICE's Roadmap masking: 120 targets out of 1032 tracks per bin.
PAPER_N_TARGETS, PAPER_N_TRACKS = 120, 1032


def masking_caveat(n_targets: int, n_tracks: int) -> str:
    """The second caveat every eDICE row carries — PI decision of 2026-08-25.

    Built from the run's ACTUAL numbers, so a manifest can never claim a masking rate the run did
    not use — and it describes whichever reading was run, because a provenance string that
    contradicts its own numbers is worse than no string at all.
    """
    rate = PAPER_N_TARGETS / PAPER_N_TRACKS
    head = f"masks {n_targets} of {n_tracks} tracks per bin ({n_targets / n_tracks:.1%})"
    if n_targets == PAPER_N_TARGETS:
        return (f"{head} — the paper's ABSOLUTE count ({PAPER_N_TARGETS} of {PAPER_N_TRACKS}), "
                f"whose rate was {rate:.1%} on Roadmap's larger panel.")
    if abs(n_targets / n_tracks - rate) < 0.01:
        return (f"{head} — the paper's {rate:.1%} RATE ({PAPER_N_TARGETS} of {PAPER_N_TRACKS}), not "
                f"its absolute {PAPER_N_TARGETS}, which would mask "
                f"{PAPER_N_TARGETS / n_tracks:.0%} of our smaller panel. Pre-registered departure, "
                f"PI 2026-08-25.")
    return (f"{head} — neither the paper's absolute {PAPER_N_TARGETS} nor its {rate:.1%} rate "
            f"({round(rate * n_tracks)} tracks here). NOT a pre-registered setting.")


def _open_store(regime: dict):
    from candi.store.reader import CorpusStore
    return CorpusStore(regime["store"])


def _cfg_from_args(args) -> Config:
    return Config(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                  n_targets=args.n_targets, embed_dim=args.embed_dim, seed=args.seed)


def _resolve_chroms(chroms, store):
    """`--chroms all` means every chromosome the store carries -- the P2 pass, as one word.

    P2 is "the same pairs and truth, over every chromosome the store carries" (§2), and spelling
    two dozen names on a command line is a way to leave one out by accident.
    """
    if not chroms:
        return None
    if len(chroms) == 1 and chroms[0] == "all":
        from candi.store import layout as L
        return L.sort_chroms(store.n_bins().keys())
    return list(chroms)


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def cmd_train(args) -> int:
    regime = load_regime(args.regime)
    store = _open_store(regime)
    panel = training_panel(store, regime)
    assert_no_eval_leakage(panel)

    chroms = args.train_chroms or list(regime["train_chroms"])
    print(f"[edice] panel: {panel.n_tracks} pval tracks over {len(panel.biosamples)} training "
          f"cells x {len(panel.assays)} assays; training bins from {chroms}", flush=True)

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    blocks: List[np.ndarray] = []
    for chrom in chroms:
        n = int(store[panel.biosamples[0]].n_bins(chrom))
        idx = np.arange(n)
        if args.max_train_bins and n > args.max_train_bins:
            # A bin subsample, not a truncation: eDICE has no positional structure, so a random
            # subset of bins is the same distribution as all of them, and a contiguous prefix is
            # not (it is one end of a chromosome).
            idx = np.sort(rng.choice(n, size=args.max_train_bins, replace=False))
        for lo in range(0, len(idx), args.slab):
            take = idx[lo:lo + args.slab]
            slab = read_slab(store, panel, chrom, int(take[0]), int(take[-1]) + 1)
            blocks.append(slab[take - take[0]])
        print(f"[edice]   {chrom}: {len(idx)} bins", flush=True)
    values = np.arcsinh(np.concatenate(blocks, axis=0))
    del blocks
    print(f"[edice] training matrix {values.shape} ({values.nbytes / 1e9:.1f} GB)", flush=True)

    cfg = _cfg_from_args(args)
    if cfg.n_targets >= panel.n_tracks:
        raise SystemExit(
            f"--n-targets {cfg.n_targets} >= panel {panel.n_tracks}: no supports would be left.")
    print(f"[edice] masking {cfg.n_targets}/{panel.n_tracks} tracks per bin "
          f"({cfg.n_targets / panel.n_tracks:.1%}); eDICE's Roadmap setting was "
          f"120/1032 = 11.6%", flush=True)

    model = build_model(len(panel.biosamples), len(panel.assays), cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[edice] parameters: {n_params:,}", flush=True)

    sampler = TrainSampler(values, panel.cell_ids, panel.assay_ids, n_targets=cfg.n_targets,
                           batch_size=cfg.batch_size, rng=rng)
    device = torch.device(args.device)
    history = run_train(model, sampler, cfg, device)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": asdict(cfg),
                "biosamples": panel.biosamples, "assays": panel.assays,
                "tracks": panel.tracks}, out / "model.pt")
    (out / "train.json").write_text(json.dumps({
        "method": METHOD, "caveat": CAVEAT,
        "masking_caveat": masking_caveat(cfg.n_targets, panel.n_tracks),
        "regime": str(Path(args.regime).resolve()),
        "config": asdict(cfg), "n_parameters": n_params, "train_chroms": chroms,
        "n_train_bins": int(values.shape[0]), "n_panel_tracks": panel.n_tracks,
        "n_cells": len(panel.biosamples), "n_assays": len(panel.assays),
        "history": history, "torch": torch.__version__, "platform": platform.platform(),
    }, indent=2))
    print(f"[edice] wrote {out / 'model.pt'}")
    return 0


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

def cmd_predict(args) -> int:
    from candi.bench.harness import open_source

    regime = load_regime(args.regime)
    store = _open_store(regime)
    panel = training_panel(store, regime)
    assert_no_eval_leakage(panel)

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    if ckpt["biosamples"] != panel.biosamples or ckpt["assays"] != panel.assays:
        raise SystemExit("the checkpoint's cell/assay vocabulary does not match this regime's")
    cfg = Config(**ckpt["config"])
    model = build_model(len(panel.biosamples), len(panel.assays), cfg)
    model.load_state_dict(ckpt["state_dict"])
    device = torch.device(args.device)
    model.to(device).eval()

    source = open_source(store=args.regime, chroms=_resolve_chroms(args.chroms, store))
    reqs = requested_tracks(source, panel)
    chroms = list(source.eval_chroms)
    print(f"[edice] {len(reqs)} declared tracks over {len(chroms)} chromosomes", flush=True)

    t_cells = np.asarray([r.cell_id for r in reqs], dtype=np.int64)
    t_assays = np.asarray([r.assay_id for r in reqs], dtype=np.int64)

    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    for r in reqs:
        (root / r.dirname).mkdir(exist_ok=True)

    for chrom in chroms:
        n = source.n_bins(chrom)
        t0 = time.time()
        buf = np.empty((n, len(reqs)), dtype=np.float32)
        for lo in range(0, n, args.slab):
            hi = min(lo + args.slab, n)
            supports = np.arcsinh(read_slab(store, panel, chrom, lo, hi))
            sampler = FixedTargetSampler(supports, panel.cell_ids, panel.assay_ids,
                                         t_cells, t_assays, batch_size=args.batch_size)
            buf[lo:hi] = np.sinh(run_predict(model, sampler, device))
        # eDICE is unconstrained in sign; `-log10 p` is not. Clipping at 0 is the same floor
        # Avocado's port applies (RIVALS_PLAN §7.1) and is recorded in the manifest.
        np.clip(buf, 0.0, None, out=buf)
        for j, r in enumerate(reqs):
            np.savez_compressed(root / r.dirname / f"{chrom}.npz",
                                signal_mu=np.ascontiguousarray(buf[:, j]))
        print(f"[edice]   {chrom}: {n} bins, {time.time() - t0:.0f}s", flush=True)
        del buf

    (root / "manifest.json").write_text(json.dumps({
        "method": METHOD,
        "version": "t52",
        "generated_by": "competitors/edice/run_eic.py predict",
        "date": date.today().isoformat(),
        "arms": ["pval"],
        "notes": (f"{CAVEAT} {masking_caveat(cfg.n_targets, panel.n_tracks)} Trained on the "
                  f"{len(panel.biosamples)} regime training cells' pval tracks only (§6.2); paper "
                  f"defaults otherwise, one seed (§6.3). signal_mu clipped at 0. No count head and "
                  f"no peak head."),
        "caveat": CAVEAT,
        "masking_caveat": masking_caveat(cfg.n_targets, panel.n_tracks),
        "model": str(Path(args.model).resolve()),
        "chroms": chroms,
        "n_tracks": len(reqs),
    }, indent=2))
    print(f"[edice] wrote {root / 'manifest.json'}")
    return 0


# ---------------------------------------------------------------------------
# dry-run: the panel and the track set, without touching a GPU
# ---------------------------------------------------------------------------

def cmd_panel(args) -> int:
    from candi.bench.harness import open_source

    regime = load_regime(args.regime)
    store = _open_store(regime)
    panel = training_panel(store, regime)
    assert_no_eval_leakage(panel)
    source = open_source(store=args.regime, chroms=_resolve_chroms(args.chroms, store))
    reqs = requested_tracks(source, panel)

    per_cell: dict = {}
    for b, _ in panel.tracks:
        per_cell[b] = per_cell.get(b, 0) + 1
    print(f"training panel: {panel.n_tracks} pval tracks, "
          f"{len(per_cell)} cells (min {min(per_cell.values())}, max {max(per_cell.values())} "
          f"assays per cell)")
    print(f"declared targets: {len(reqs)} tracks over {len(source.pairs('impute'))} pairs, "
          f"chroms {list(source.eval_chroms)}")
    print(f"bins on {source.eval_chroms[0]}: {source.n_bins(source.eval_chroms[0]):,}")
    for r in reqs[:3]:
        print(f"  e.g. {r.dirname}  (queried as cell_id={r.cell_id}, assay_id={r.assay_id})")
    overlap = {(b, a) for b, a in panel.tracks} & {(r.input_biosample, r.assay) for r in reqs}
    print(f"support/target overlap (must be 0): {len(overlap)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--regime", required=True)
        sp.add_argument("--slab", type=int, default=400_000, help="bins read per store slab")
        sp.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    tr = sub.add_parser("train")
    common(tr)
    tr.add_argument("--out", required=True)
    tr.add_argument("--epochs", type=int, default=Config.epochs)
    tr.add_argument("--lr", type=float, default=Config.lr)
    tr.add_argument("--batch-size", type=int, default=Config.batch_size)
    # NO DEFAULT, deliberately. "Paper defaults" (§6.3) is ambiguous here and the two readings are
    # far apart: eDICE masked 120 of Roadmap's 1032 tracks per bin (11.6%), and our EIC panel holds
    # 267 tracks, so the same ABSOLUTE 120 masks 45% while the same RATE is 31. That is a decision
    # about the training signal, not a knob, so the run refuses to pick one silently. See the
    # README's "The one open decision".
    tr.add_argument("--n-targets", type=int, required=True,
                    help="tracks masked per bin. 31 keeps eDICE's Roadmap RATE (11.6%% of 267); "
                         "120 keeps its absolute count and masks 45%%. Pre-register the choice.")
    tr.add_argument("--embed-dim", type=int, default=Config.embed_dim)
    tr.add_argument("--seed", type=int, default=Config.seed)
    tr.add_argument("--train-chroms", nargs="+", default=None,
                    help="default: the regime's train_chroms")
    tr.add_argument("--max-train-bins", type=int, default=None,
                    help="random bin subsample per chromosome, to bound host memory")
    tr.set_defaults(func=cmd_train)

    pr = sub.add_parser("predict")
    common(pr)
    pr.add_argument("--model", required=True)
    pr.add_argument("--out", required=True)
    pr.add_argument("--batch-size", type=int, default=4096)
    pr.add_argument("--chroms", nargs="+", default=None,
                    help="default: the regime's eval_chroms (P1). `all` = every chromosome the "
                         "store carries (P2).")
    pr.set_defaults(func=cmd_predict)

    pa = sub.add_parser("panel", help="print the panel and the declared track set; no compute")
    common(pa)
    pa.add_argument("--chroms", nargs="+", default=None)
    pa.set_defaults(func=cmd_panel)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
