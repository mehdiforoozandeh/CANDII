"""eDICE on our EIC data: train on the 51 training cells, predict the declared pairs, emit §4.1.

    # 1. train, selecting the checkpoint on V_ (plan/BENCHMARK_DESIGN.md §5)
    python run_eic.py train --regime configs/regime.eic_19.json --out runs/eic --epochs 50 \
        --n-targets 31 --eval-every 3 --early-stop-epochs 3
    # 2. predict -> a §4.1 prediction root, from the SELECTED checkpoint
    python run_eic.py predict --regime configs/regime.eic_19.json \
        --model runs/eic/model.selected.pt --out preds/edice_heldout
    # 3. score with OUR instrument, not eDICE's
    python -m candi.bench.external --store configs/regime.eic_19.json \
        --pred preds/edice_heldout --out scores/edice.json --sigma-table preds/edice_sigma.json

Two things are pre-registered and appear in every eDICE row (see `eic_panel.py` and the README):
the 51-cell training restriction (§6.2), and the transductive substitution that gives a `V_X`
target its paired `T_X` cell's embedding (§7.3).

**Checkpoint selection (§5).** `train` derives a `V_`-only regime, and every `--eval-every` epochs
it writes that panel's predictions to a §4.1 root and scores them with
`candi.bench.external.score_external` — the same instrument that produces a board row. eDICE's own
`edice_torch/metrics.py` is NOT used for this: it is cheaper, and using it would make the rule
non-uniform, which is the only thing §5 asks for. `B_` is never read; the selector asserts it.

**Training loci (D32).** With a `regions` BED declared, a bin trains only if it lies wholly inside
a region. See `eic_panel.train_bin_spans` for what containment means when the window is one bin.

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
from eic_panel import (Panel, TrackRequest, assert_no_eval_leakage, derive_v_only_regime,
                       load_regime, read_slab, requested_tracks, train_bin_spans, training_panel)

METHOD = "eDICE (PyTorch reimplementation)"
#: The pre-registered caveat, copied verbatim into every manifest so no table can lose it.
CAVEAT = ("transductive: a V_X/B_X target is queried with its paired T_X cell's learned embedding "
          "(RIVALS_PLAN.md §7.3). eDICE has no embedding for a cell it never trained on.")


#: eDICE's Roadmap masking: 120 targets out of 1032 tracks per bin.
PAPER_N_TARGETS, PAPER_N_TRACKS = 120, 1032

#: What `--select-metric` may name, and which direction wins. Every one is a `macro["pval"]` key
#: `candi.bench.external` computes for a POINT prediction — the E block, plus the two rank
#: correlations. The distributional keys (`crps`, `pit_ks`, `coverage_95`, `c_index`) are
#: deliberately absent: eDICE emits `signal_mu` and no `signal_sigma`, so `score_external` records
#: it as a point-only track and never computes them. CANDI's own loop selects on count-arm `crps`,
#: which eDICE structurally cannot produce (no count head; B1b forbids inventing a read depth), so
#: the uniformity §5 asks for holds here in the instrument, the panel and the cadence — NOT in the
#: key. That gap is real and is for the PI, not for this file to paper over.
SELECT_METRICS = {"mse": "min", "mse1obs": "min", "mse1imp": "min",
                  "gwcorr": "max", "gwspear": "max"}


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
# the §4.1 prediction root — written by the predict command AND by the selector
# ---------------------------------------------------------------------------

def write_pred_root(model, store, panel: Panel, reqs: List[TrackRequest], chroms: List[str],
                    root: Path, device, *, n_bins, slab: int, batch_size: int, notes: dict,
                    verbose: bool = True) -> Path:
    """Emit `<root>/<input>__<target>__<assay>/<chrom>.npz` + `manifest.json` (RIVALS_PLAN §4.1).

    One pass over the bins answers every request at once — a `FixedTargetSampler` carries the whole
    target panel — so the cost is set by the bin count, not by how many tracks are asked for.
    """
    t_cells = np.asarray([r.cell_id for r in reqs], dtype=np.int64)
    t_assays = np.asarray([r.assay_id for r in reqs], dtype=np.int64)

    root.mkdir(parents=True, exist_ok=True)
    for r in reqs:
        (root / r.dirname).mkdir(exist_ok=True)

    for chrom in chroms:
        n = int(n_bins(chrom))
        t0 = time.time()
        buf = np.empty((n, len(reqs)), dtype=np.float32)
        for lo in range(0, n, slab):
            hi = min(lo + slab, n)
            supports = np.arcsinh(read_slab(store, panel, chrom, lo, hi))
            sampler = FixedTargetSampler(supports, panel.cell_ids, panel.assay_ids,
                                         t_cells, t_assays, batch_size=batch_size)
            buf[lo:hi] = np.sinh(run_predict(model, sampler, device))
        # eDICE is unconstrained in sign; `-log10 p` is not. Clipping at 0 is the same floor
        # Avocado's port applies (RIVALS_PLAN §7.1) and is recorded in the manifest.
        np.clip(buf, 0.0, None, out=buf)
        for j, r in enumerate(reqs):
            np.savez_compressed(root / r.dirname / f"{chrom}.npz",
                                signal_mu=np.ascontiguousarray(buf[:, j]))
        if verbose:
            print(f"[edice]   {chrom}: {n} bins, {time.time() - t0:.0f}s", flush=True)
        del buf

    (root / "manifest.json").write_text(json.dumps({
        "method": METHOD,
        "version": "t52",
        "generated_by": "competitors/edice/run_eic.py",
        "date": date.today().isoformat(),
        "arms": ["pval"],
        "caveat": CAVEAT,
        "chroms": list(chroms),
        "n_tracks": len(reqs),
        **notes,
    }, indent=2))
    return root


# ---------------------------------------------------------------------------
# §5 checkpoint selection, on V_
# ---------------------------------------------------------------------------

class Selector:
    """Score the current weights on the `V_` panel through OUR instrument, and keep the best.

    §5 asks every trainable method to select its checkpoint by the same rule. The rule is not a
    number eDICE computes for itself — `edice_torch/metrics.py` holds eDICE's own per-track MSE and
    Pearson, which are its TRAINING diagnostics and would make the rule non-uniform if they
    selected. So a check here is exactly what a leaderboard row is: write a §4.1 prediction root
    and hand it to `candi.bench.external.score_external`. It is much more expensive than eDICE's
    own metric would be, and that is the price of the rule being the same one.

    Three properties are copied from `candi.train`'s `_keep_best`/`eval_hook`:

    * the eval source is opened ONCE, so every check scores the same positions and epoch 6 against
      epoch 12 is a paired comparison rather than merely a deterministic one;
    * the best weights are written THE MOMENT the metric improves, so a run killed by walltime
      still leaves a validly selected checkpoint on disk;
    * patience is counted in EPOCHS, so its resolution is `eval_every` and a patience below the
      cadence can never fire.

    `B_` is never opened: the source is built from `derive_v_only_regime`, and the constructor
    refuses a source that reaches a non-`V_` target.
    """

    def __init__(self, model, store, panel: Panel, source, best_path: Path, work: Path, *,
                 device, metric: str, eval_every: int, epochs: int, patience: int,
                 slab: int, batch_size: int, curve_path: Path) -> None:
        if metric not in SELECT_METRICS:
            raise SystemExit(f"--select-metric {metric} is not one of {sorted(SELECT_METRICS)}. "
                             f"See SELECT_METRICS for why the distributional keys are not there.")
        bad = sorted({p.target_biosample for p in source.pairs("impute")
                      if not p.target_biosample.startswith("V_")})
        if bad:
            raise AssertionError(
                f"the selection source reaches {bad}. §5, and the PI's ruling of 2026-08-31: B_ is "
                f"not merely kept out of selection, it is never READ.")
        self.model, self.store, self.panel, self.source = model, store, panel, source
        self.best_path, self.work, self.device = best_path, work, device
        self.metric, self.sign = metric, (1.0 if SELECT_METRICS[metric] == "min" else -1.0)
        self.eval_every, self.epochs, self.patience = eval_every, epochs, patience
        self.slab, self.batch_size, self.curve_path = slab, batch_size, curve_path
        self.reqs = requested_tracks(source, panel)
        self.chroms = list(source.eval_chroms)
        self.best = {"value": float("inf"), "epoch": -1}
        self.curve: List[dict] = []

    def announce(self) -> None:
        bins = sum(int(self.source.n_bins(c)) for c in self.chroms)
        print(f"[edice] selection: every {self.eval_every} epoch(s) on {len(self.reqs)} V_ tracks "
              f"x {bins:,} bins ({','.join(self.chroms)}), scored by candi.bench.external; keep "
              f"the best macro pval {self.metric} ({SELECT_METRICS[self.metric]}), patience "
              f"{self.patience} epoch(s).", flush=True)

    def __call__(self, epoch: int, row: dict) -> bool:
        """`edice_torch.train`'s `on_epoch`. True ends the run."""
        if (epoch + 1) % self.eval_every and epoch != self.epochs - 1:
            return False
        from candi.bench.external import score_external

        t0 = time.time()
        # `run_predict` calls `model.eval()` and never puts it back — harmless today, because the
        # loop calls `model.train()` at the top of the next epoch, but the restore lives here so
        # the hook cannot leave the model in a mode its caller did not choose.
        was_training = bool(self.model.training)
        try:
            write_pred_root(self.model, self.store, self.panel, self.reqs, self.chroms,
                            self.work, self.device, n_bins=self.source.n_bins, slab=self.slab,
                            batch_size=self.batch_size, verbose=False,
                            notes={"notes": f"V_ selection check at epoch {epoch}; not a board run."})
        finally:
            self.model.train(was_training)
        scores = score_external(self.source, self.work, seed=0)
        value = scores["macro"].get("pval", {}).get(self.metric, float("nan"))
        improved = bool(np.isfinite(value)) and self.sign * float(value) < self.best["value"]
        if improved:
            self.best.update(value=self.sign * float(value), epoch=epoch)
            # Written HERE, before anything else can fail — a walltime kill after this line still
            # leaves a selected checkpoint, and that property has already saved this project once.
            torch.save(self.model.state_dict(), self.best_path)
        self.curve.append({"epoch": epoch, "metric": self.metric, "value": float(value),
                           "n_tracks": scores["macro"]["pval"].get("n_tracks"),
                           "improved": improved, "seconds": round(time.time() - t0, 1)})
        self.curve_path.write_text(json.dumps(self.curve, indent=2))
        print(f"[edice] epoch {epoch:3d}  V_ {self.metric}={value:.6f}"
              f"{'  <- best' if improved else ''}  "
              f"(best epoch {self.best['epoch']}, {time.time() - t0:.0f}s)", flush=True)
        if self.patience and self.best["epoch"] >= 0 and (epoch - self.best["epoch"]) > self.patience:
            print(f"[edice] EARLY STOP at epoch {epoch}: no V_ {self.metric} improvement since "
                  f"epoch {self.best['epoch']} ({epoch - self.best['epoch']} epochs > patience "
                  f"{self.patience}). model.best.pt is already on disk and is what gets scored.",
                  flush=True)
            return True
        return False


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

    if regime.get("regions"):
        print(f"[edice] regions: {regime['regions']['bed']} policy={regime['regions'].get('policy', 'contain')} "
              f"— a bin trains only if it lies WHOLLY inside a region (D32)", flush=True)

    blocks: List[np.ndarray] = []
    n_scope = 0
    for chrom in chroms:
        n = int(store[panel.biosamples[0]].n_bins(chrom))
        # D32 — the whole chromosome without a `regions` key, its contained bins with one. Reading
        # region by region rather than by a scattered fancy index is what keeps host memory bounded:
        # the pilot regions are ≥500 kb but they are spread over a whole chromosome, so one slab of
        # 400,000 SCATTERED indices on chr1 would span ~10 M bins and need ~10 GB to materialise.
        spans = train_bin_spans(regime, args.regime, chrom, n, store.resolution)
        idx = (np.concatenate([np.arange(a, b) for a, b in spans]) if spans
               else np.empty(0, dtype=np.int64))
        n_scope += int(idx.size)
        if args.max_train_bins and idx.size > args.max_train_bins:
            # A bin subsample, not a truncation: eDICE has no positional structure, so a random
            # subset of bins is the same distribution as all of them, and a contiguous prefix is
            # not (it is one end of a chromosome).
            idx = np.sort(rng.choice(idx, size=args.max_train_bins, replace=False))
        keep = np.zeros(n, dtype=bool)
        keep[idx] = True
        for a, b in spans:
            for lo in range(a, b, args.slab):
                hi = min(lo + args.slab, b)
                sel = keep[lo:hi]
                if sel.any():
                    blocks.append(read_slab(store, panel, chrom, lo, hi)[sel])
        print(f"[edice]   {chrom}: {int(idx.size)} bins over {len(spans)} span(s)", flush=True)
    if not blocks:
        raise SystemExit(f"no training bin survives on {chroms}; check the regime's `regions` BED")
    values = np.arcsinh(np.concatenate(blocks, axis=0))
    del blocks
    print(f"[edice] training matrix {values.shape} ({values.nbytes / 1e9:.1f} GB); "
          f"declared scope {n_scope:,} bins", flush=True)

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
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    selector, v_regime_path = None, None
    if args.eval_every:
        v_regime_path = out / f"regime.{Path(args.regime).stem}.vsel.json"
        v_regime_path.write_text(json.dumps(derive_v_only_regime(regime, args.regime), indent=2))
        # OPENED ONCE, outside the hook, for `candi.train`'s reason: one source is what pins one
        # eval scope across every epoch, and selection only compares epoch 6 against epoch 12 if
        # both saw the same positions.
        from candi.bench.harness import open_source
        source = open_source(store=str(v_regime_path), chroms=args.eval_chroms or None)
        selector = Selector(model.to(device), store, panel, source, out / "model.best.pt",
                            out / "preds_select", device=device, metric=args.select_metric,
                            eval_every=args.eval_every, epochs=cfg.epochs,
                            patience=args.early_stop_epochs, slab=args.slab,
                            batch_size=args.eval_batch_size, curve_path=out / "select.json")
        selector.announce()
    else:
        print("[edice] --eval-every 0: NO checkpoint is selected, so this run does not satisfy "
              "plan/BENCHMARK_DESIGN.md §5. Deliberate only.", flush=True)

    history = run_train(model, sampler, cfg, device, on_epoch=selector)

    torch.save({"state_dict": model.state_dict(), "config": asdict(cfg),
                "biosamples": panel.biosamples, "assays": panel.assays,
                "tracks": panel.tracks}, out / "model.pt")      # the LAST epoch, always written
    if selector is not None and selector.best["epoch"] >= 0:
        # `model.best.pt` holds bare weights so it can be written cheaply mid-run; the scorable
        # checkpoint needs the vocabulary too, and it is written here from the SELECTED weights.
        torch.save({"state_dict": torch.load(out / "model.best.pt", map_location="cpu"),
                    "config": asdict(cfg), "biosamples": panel.biosamples,
                    "assays": panel.assays, "tracks": panel.tracks}, out / "model.selected.pt")
        print(f"[edice] selected epoch {selector.best['epoch']} on V_ {args.select_metric}; "
              f"score model.selected.pt, not model.pt", flush=True)
    (out / "train.json").write_text(json.dumps({
        "method": METHOD, "caveat": CAVEAT,
        "masking_caveat": masking_caveat(cfg.n_targets, panel.n_tracks),
        "regime": str(Path(args.regime).resolve()),
        "config": asdict(cfg), "n_parameters": n_params, "train_chroms": chroms,
        "n_train_bins": int(values.shape[0]), "n_train_scope_bins": n_scope,
        "regions": regime.get("regions"),
        "n_panel_tracks": panel.n_tracks,
        "n_cells": len(panel.biosamples), "n_assays": len(panel.assays),
        "selection": (None if selector is None else {
            "metric": args.select_metric, "panel": "V_", "eval_every": args.eval_every,
            "early_stop_epochs": args.early_stop_epochs,
            "scored_by": "candi.bench.external.score_external",
            "regime": str(v_regime_path), "chroms": selector.chroms,
            "n_tracks": len(selector.reqs), "best_epoch": selector.best["epoch"],
            "curve": selector.curve}),
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

    root = write_pred_root(
        model, store, panel, reqs, chroms, Path(args.out), device, n_bins=source.n_bins,
        slab=args.slab, batch_size=args.batch_size,
        notes={
            "generated_by": "competitors/edice/run_eic.py predict",
            "notes": (f"{CAVEAT} {masking_caveat(cfg.n_targets, panel.n_tracks)} Trained on the "
                      f"{len(panel.biosamples)} regime training cells' pval tracks only (§6.2); "
                      f"paper defaults otherwise, one seed (§6.3). signal_mu clipped at 0. No "
                      f"count head and no peak head."),
            "masking_caveat": masking_caveat(cfg.n_targets, panel.n_tracks),
            "model": str(Path(args.model).resolve()),
        })
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
    # §5's uniform rule. The cadence is a flag because a check costs a whole prediction pass over
    # the eval chromosomes plus a full `score_external` — see the launcher's sizing note — so the
    # patience trades directly against that, exactly as it does for CANDI.
    tr.add_argument("--eval-every", type=int, default=3,
                    help="epochs between V_ selection checks; 0 selects NOTHING and fails §5")
    tr.add_argument("--early-stop-epochs", type=int, default=3,
                    help="stop when the V_ metric has not improved for MORE than this many "
                         "epochs (0 = off). Resolution is --eval-every.")
    tr.add_argument("--select-metric", default="mse", choices=sorted(SELECT_METRICS),
                    help="the macro pval key selection reads off candi.bench.external")
    tr.add_argument("--eval-chroms", nargs="+", default=None,
                    help="default: the regime's eval_chroms — the scope CANDI selects on")
    tr.add_argument("--eval-batch-size", type=int, default=4096)
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
