"""Their two-stage, per-chromosome training schedule, in PyTorch. One job per chromosome.

Stage 1 (`Precamole`) is 3-way classification on terciles; stage 2 (`Guacamole`) is scalar
regression that starts from stage 1's trunk. Epoch counts, batch sizes and factor widths are
upstream's own, per chromosome, from `dataset3.UPSTREAM_HYPERPARAMS`.

**`--stage` is OURS, and it is the one place this file departs from upstream's method.** Upstream
fits one whole independent model per chromosome — cell factors, assay factors, dense network and
position tables together, all on that chromosome's own bins. `BENCHMARK_DESIGN.md` §2 Rule 2 says a
regime names the loci where a method's *transferable* parameters were fit, and the cell and assay
factors are transferable, so upstream's scheme fits them on the chromosomes it is then scored on.
Two stages fix that, and they are Avocado's own scheme (`competitors/avocado/train.py`, §3.2 and
§12.2), not Guacamole's:

    --stage full     upstream, unchanged. Everything, on one chromosome. The Dataset-3 anchor.
    --stage shared   everything, on the REGIME'S TRAINING SCOPE. Run once. Its product is the
                     transferable half; its position tables are thrown away.
    --stage genome   load a `shared` run's transferable half, FREEZE it, re-init the position
                     tables at this chromosome's size and fit only those. Once per eval chromosome.

So the run that predicts chr20 has never taken a gradient on chr20 in any parameter Rule 2 binds.
**This makes the board row our two-stage variant of Lavawizard and not the published one** — see
`README.md`; the 2019 submission itself stays on the board unmodified as one of the 23 entrants.

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

**Checkpoint selection is injected, not built here.** `BENCHMARK_DESIGN.md` §5 makes every
trainable method select its checkpoint on the `V_` panel by the same rule, and that rule is
`candi.bench.external.score_external` on a written prediction root. Scoring needs the store, the
store is `candi`'s, and this module must keep running on Fir with `candi` nowhere on the path — so
`train_chromosome` takes a `select_fn` and `store_eic.py` supplies it. Without one this file trains
exactly as it did before and selects nothing.

```bash
python -m lavawizard.train --cache <cache> --chrom chr21 --out <run> \
    --contributor-mode upstream                                    # anchor: no selection
python -m lavawizard.store_eic train --regime <regime> --stage shared \
    --cache <cache> --out <run>                                    # the transferable half, once
python -m lavawizard.store_eic train --regime <regime> --stage genome --chrom chr21 \
    --cache <cache> --out <run> --select-every 50                  # the board run: selects on V_
```
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

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
    `ceil(n_positions / batch_size)` steps and therefore covers the training scope once. That scope
    is the whole chromosome unless the cache carries a `regions` restriction (D32).
    """

    def __init__(self, cache: CachedChrom, batch_size: int, mode: str, seed: int = 0):
        self.c = cache
        self.batch_size = int(batch_size)
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.cursor = 0
        # D32 — under a `regions` regime the cache carries `train_bins`, the chromosome bins lying
        # WHOLLY inside a BED region. The walk moves to those bins and nothing else changes: the
        # indices stay absolute chromosome bins, so the embedding tables and every prediction are
        # still addressed the way they were, and the tiling is anchored at chromosome bin 0 rather
        # than re-anchored per region. Deriving `steps_per_epoch` from the walk's own length is what
        # keeps an epoch meaning "the training scope once" — a scope 40x smaller must buy 40x fewer
        # steps, not the same steps over the same bins 40 times.
        self.positions = (cache.train_bins if cache.train_bins is not None
                          else np.arange(cache.n_bins, dtype=np.int64))
        if self.positions.size == 0:
            raise ValueError("the training scope is empty: no bin of this chromosome lies inside "
                             "a region of the regime's BED")
        self.steps_per_epoch = int(np.ceil(self.positions.size / self.batch_size))
        # A mark carried by a single track has no leave-one-out average: removing the target
        # empties the pool. §5's answer is to skip and list, and skipping is right rather than
        # zeroing, because the head is `Dense(1)(x) + average` — a zero average is not a neutral
        # input, it teaches the trunk to emit the whole signal as a correction on those bins only.
        # Dataset 3 never hits this; our EIC has five such marks out of 35, none of them on a
        # declared eval track. `upstream` mode keeps every track, because there the target is in
        # its own average and the pool is never empty.
        thin = (cache.mark_count < 2) if mode == "loo" else np.zeros(len(cache.marks), bool)
        keep = ~thin[cache.mark_ix]
        self.eligible = np.flatnonzero(keep).astype(np.int64)
        self.skipped_marks = [m for j, m in enumerate(cache.marks) if thin[j]]
        if self.eligible.size == 0:
            raise ValueError("every track is on a single-contributor mark; nothing to train on")

    def _batch(self) -> Tuple[np.ndarray, ...]:
        bs, n = self.batch_size, self.positions.size
        pos = self.positions[np.arange(self.cursor, self.cursor + bs) % n].astype(np.int64)
        self.cursor = (self.cursor + bs) % n
        tix = self.eligible[self.rng.integers(0, self.eligible.size, bs)].astype(np.int64)
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
               max_steps: Optional[int] = None,
               eval_hook: Optional[Callable[[int, int], bool]] = None,
               eval_every: int = 0, params=None,
               set_train: Optional[Callable[[], None]] = None) -> Dict[str, float]:
    # `params` and `set_train` are how the genome stage keeps the transferable half out of the
    # optimiser and out of `train()` mode. Both default to the whole model, so every other caller
    # is the call it always was.
    opt = torch.optim.Adam(model.parameters() if params is None else params, lr=lr)
    set_train = set_train or model.train
    set_train()
    spe = sampler.steps_per_epoch
    total = min(epochs * spe, max_steps) if max_steps else epochs * spe
    print(f"{label}: {epochs} epochs x {spe} steps = {epochs*spe} steps"
          + (f" (capped at {total})" if total != epochs * spe else ""), flush=True)
    t0 = time.time()
    step = 0
    last = float("nan")
    stopped = False
    # Kept out of `ms_per_step`. A step rate that silently included the selection pass would make
    # the cadence look free and would make two runs at different cadences incomparable.
    eval_s = 0.0
    while step < total:
        d, y = batch_fn(device)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(**d), y)
        loss.backward()
        opt.step()
        last = float(loss.detach())
        step += 1
        # The hook fires on an EPOCH boundary and on the last step, mirroring `candi.train`'s
        # `(ep + 1) % eval_every == 0 or ep == epochs - 1`. Boundaries, not step counts, because the
        # position walk covers the training scope exactly once per epoch and a mid-epoch checkpoint
        # would have seen a prefix of the chromosome more often than the rest of it.
        if eval_hook and eval_every and (step % spe == 0 or step == total):
            ep = (step - 1) // spe
            if (ep + 1) % eval_every == 0 or step == total:
                te = time.time()
                stop = eval_hook(step, ep)
                eval_s += time.time() - te
                set_train()
                if stop:
                    stopped = True
                    break
        if step % log_every == 0 or step == total:
            el = time.time() - t0 - eval_s
            print(f"  {label} {step}/{total}  loss {last:.6f}  "
                  f"{el/step*1000:.1f} ms/step  eta {(total-step)*el/step/60:.1f} min", flush=True)
    el = time.time() - t0 - eval_s
    return {"steps": step, "final_loss": last, "seconds": el, "planned_steps": total,
            "early_stopped": stopped, "eval_seconds": round(eval_s, 1),
            "ms_per_step": el / max(step, 1) * 1000}


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
                     epoch_scale: float = 1.0, tf32: bool = True,
                     select_fn: Optional[Callable[["Guacamole", int, int], float]] = None,
                     select_every: int = 0, select_metric: str = "",
                     early_stop_epochs: int = 0, stage: str = "full",
                     init: Optional[Path] = None, hparams_chrom: str = "") -> Dict:
    """Both stages for one chromosome. Writes `guacamole_<chrom>.pt` and `train_<chrom>.json`.

    `stage` is `full` (upstream, and the default, so nothing that existed before this flag moved),
    `shared`, or `genome` — see the module docstring. `full` and `shared` run the same code and
    differ only in the scope the caller points them at, which is the honest way to say it: the
    transferable stage is upstream's own fit, moved to loci Rule 2 allows. `genome` needs `init`.

    `hparams_chrom` names the chromosome whose `dataset3.UPSTREAM_HYPERPARAMS` row to use when the
    cache stem is not itself one of the 23 — the packed multi-chromosome scope a `regions` regime's
    shared fit trains on. Empty means "use the stem", which is every other case.

    **`select_fn` is how BENCHMARK_DESIGN.md §5's uniform selection reaches this method.** It is
    injected rather than built here because scoring a `V_` panel means reading the store, and the
    store is `candi`'s — `store_eic.py` is the one module allowed to import it (see the README's
    split, pinned by `test_only_the_our_store_module_may_import_candi`). It takes
    `(model, epoch, step)` and returns the selection metric, lower better; `store_eic.selector`
    builds the one this method actually uses.

    Selection runs on **stage 2 only**. Stage 1's head is discarded by `from_precamole`, so there is
    no stage-1 checkpoint anyone could select and scoring one would be scoring a classifier against
    a regression panel.
    """
    if stage not in ("full", "shared", "genome"):
        raise ValueError(f"stage must be full, shared or genome, got {stage!r}")
    if (stage == "genome") != bool(init):
        raise ValueError("--stage genome needs an --init shared checkpoint, and no other stage "
                         "may take one: a `full` or `shared` fit that started from someone else's "
                         "transferable half was not fit on the loci its regime names.")
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)
    if dev.type == "cuda":
        enable_tf32(tf32)
    cache = CachedChrom(cache_root, chrom)
    hp = hparams_chrom or chrom
    sched = dataset3.schedule(hp)
    factors = dataset3.factor_sizes(hp)
    bs = sched["batch_size"]
    # The genome stage takes no stage 1. Stage 1 pretrains the TRUNK, the trunk is transferable, and
    # this stage's whole point is that the trunk arrives already fitted and frozen — re-running it
    # here would take a gradient on the eval chromosome in exactly the parameters Rule 2 binds.
    pre_ep = 0 if stage == "genome" else max(int(round(sched["pretrain_epochs"] * epoch_scale)), 1)
    tr_ep = max(int(round(sched["train_epochs"] * epoch_scale)), 1)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{chrom}: {cache.n_tracks} tracks, {cache.n_bins} bins, {len(cache.cells)} cells, "
          f"{len(cache.marks)} marks | stage={stage} bs={bs} pre={pre_ep} train={tr_ep} | "
          f"{factors} | mode={contributor_mode}"
          + (f" | hparams from {hp}" if hparams_chrom else ""), flush=True)

    common = dict(n_celltypes=len(cache.cells), n_assays=len(cache.marks),
                  n_positions=cache.n_bins, **factors)
    sampler = Sampler(cache, bs, contributor_mode, seed=seed)
    if sampler.skipped_marks:
        print(f"{chrom}: {len(sampler.skipped_marks)} mark(s) skipped for having no leave-one-out "
              f"pool: {sampler.skipped_marks}", flush=True)

    s1: Dict[str, float] = {"steps": 0, "final_loss": float("nan"), "seconds": 0.0,
                            "planned_steps": 0, "early_stopped": False, "eval_seconds": 0.0,
                            "ms_per_step": 0.0, "skipped": "the genome stage takes no stage 1"}
    # CONSTRUCTION ORDER IS FROZEN, and it is `Precamole` then `Guacamole` on every stage that
    # builds both. Both draw from torch's global RNG, so swapping them re-rolls every tensor in the
    # run — including the Guacamole-only `block3` and `y_pred`, which init AFTER stage 1 has already
    # advanced the stream. Building Guacamole first reads better and would silently make `--stage
    # full` a different run from the anchor it exists to reproduce.
    if stage == "genome":
        gua = Guacamole(**common).to(dev)
        # The checkpoint's position tables belong to the shared scope and are a different length, so
        # `load_transferable` never looks at them: this chromosome's tables were freshly built above
        # and stay at their init. The transferable half is loaded and then frozen, so every gradient
        # this stage takes lands on a position table.
        obj = torch.load(str(init), map_location=dev, weights_only=False)
        gua.load_transferable(obj["state_dict"]).freeze_transferable()
        n_tr = sum(p.numel() for p in gua.transferable_parameters())
        n_gen = sum(p.numel() for p in gua.genome_parameters())
        print(f"{chrom}: loaded {n_tr:,} transferable parameters from {Path(init).name} "
              f"(stage={obj.get('stage')}, scope={obj.get('chrom')}) and FROZE them; fitting "
              f"{n_gen:,} position-factor parameters on {chrom} alone", flush=True)
    else:
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

    ckpt = out_dir / f"guacamole_{chrom}.pt"
    best_path = out_dir / f"guacamole_{chrom}.best.pt"

    def _save(path: Path) -> None:
        torch.save({"state_dict": gua.state_dict(), "chrom": chrom, "config": common,
                    "contributor_mode": contributor_mode, "seed": seed,
                    "cells": cache.cells, "marks": cache.marks,
                    # STAMPED, so a checkpoint can never be mistaken for the other kind. A `shared`
                    # file's position tables are the training scope's and mean nothing anywhere
                    # else; a `genome` file is the only one that may be predicted from.
                    "stage": stage, "init": (str(init) if init else None),
                    "train_scope": cache.train_scope}, path)

    best = {"value": float("inf"), "epoch": -1, "step": -1}
    curve: list = []

    def eval_hook(step: int, ep: int) -> bool:
        t0 = time.time()
        value = float(select_fn(gua, ep, step))
        wall = time.time() - t0
        # WRITTEN THE MOMENT IT IMPROVES, before anything else happens, so a run killed by walltime
        # still leaves a checkpoint that was selected rather than merely the last one reached.
        improved = bool(np.isfinite(value)) and value < best["value"]
        if improved:
            best.update(value=value, epoch=ep, step=step)
            _save(best_path)
        curve.append({"epoch": ep, "step": step, "value": value, "wall_s": round(wall, 1),
                      "improved": improved})
        verdict = "BEST" if improved else f"(best {best['value']:.6f} @ epoch {best['epoch']})"
        print(f"  select epoch {ep} step {step}: {select_metric or 'metric'}={value:.6f} "
              f"{verdict}  [{wall/60:.1f} min]", flush=True)
        # Patience is counted in EPOCHS, as `candi.train --early-stop-epochs` counts it, so the two
        # methods are stopped by the same rule even though an epoch is a different amount of work.
        if early_stop_epochs and best["epoch"] >= 0 and (ep - best["epoch"]) > early_stop_epochs:
            print(f"  EARLY STOP at epoch {ep}: no improvement since epoch {best['epoch']} "
                  f"(> patience {early_stop_epochs}). {best_path.name} is what gets predicted.",
                  flush=True)
            return True
        return False

    # `lr` stays upstream's 1e-3 in both stages. Avocado gives its genomic factors a HIGHER rate
    # than its shared ones because a position is visited once per epoch there; Guacamole's sampler
    # walks positions contiguously and visits each exactly once per epoch too, so the same argument
    # would apply — but 1e-3 is the rate upstream fit these very tables at, and inventing a second
    # one would make the genome stage differ from upstream in a way nothing measured.
    def _set_train() -> None:
        gua.train()
        if stage == "genome":
            gua.freeze_transferable()          # `train()` un-does the BatchNorm eval; put it back
    s2 = _run_stage(gua, sampler, epochs=tr_ep, lr=1e-3,
                    loss_fn=nn.MSELoss(), batch_fn=sampler.stage2, device=dev,
                    label="stage2", max_steps=max_steps_per_stage,
                    eval_hook=(eval_hook if select_fn else None), eval_every=select_every,
                    params=(gua.genome_parameters() if stage == "genome" else None),
                    set_train=_set_train)

    peak = (torch.cuda.max_memory_allocated(dev) / 2**30) if dev.type == "cuda" else 0.0
    reserved = (torch.cuda.max_memory_reserved(dev) / 2**30) if dev.type == "cuda" else 0.0
    _save(ckpt)                                    # the LAST weights, always written

    record = {
        "chrom": chrom, "contributor_mode": contributor_mode, "seed": seed,
        # §2 Rule 2's answer for this run, in the run's own record: which stage it was, where the
        # transferable half came from, and how many parameters each half held.
        "stage": stage, "init": (str(init) if init else None),
        "hparams_chrom": hp,
        "transferable_parameters": sum(p.numel() for p in gua.transferable_parameters()),
        "genome_parameters": sum(p.numel() for p in gua.genome_parameters()),
        "transferable_frozen": (stage == "genome"),
        "batch_size": bs, "pretrain_epochs": pre_ep, "train_epochs": tr_ep,
        "epoch_scale": epoch_scale, "factors": factors, "tf32": bool(tf32),
        "n_tracks": cache.n_tracks, "n_bins": cache.n_bins,
        "n_tracks_sampled": int(sampler.eligible.size),
        # D32 — how many of this chromosome's bins were TRAINABLE, and where that scope came from.
        # Equal to `n_bins` on every regime without a `regions` key.
        "n_train_bins": int(sampler.positions.size),
        "train_scope": cache.train_scope,
        "skipped_marks": sampler.skipped_marks,
        "parameters": n_par, "stage1": s1, "stage2": s2,
        "selection": {
            "metric": select_metric or None,
            "every_epochs": int(select_every) if select_fn else 0,
            "early_stop_epochs": int(early_stop_epochs),
            "best": (dict(best) if best["epoch"] >= 0 else None),
            "checkpoint": (str(best_path) if best["epoch"] >= 0 else None),
            "curve": curve,
        },
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
    if select_fn:
        # SAY IT LOUDLY EITHER WAY. §5 asks for a checkpoint selected on V_, and a run that
        # produced only a last-epoch checkpoint does not satisfy it however well it trained.
        if best["epoch"] >= 0:
            print(f"{chrom}: SELECTED epoch {best['epoch']} on {select_metric}="
                  f"{best['value']:.6f} -> {best_path.name}; {len(curve)} check(s), "
                  f"{s2['eval_seconds']/60:.1f} min of selection against "
                  f"{s2['seconds']/60:.1f} min of stage-2 training", flush=True)
        else:
            print(f"{chrom}: NO CHECKPOINT WAS SELECTED — every check was non-finite. This run does "
                  f"NOT satisfy BENCHMARK_DESIGN.md §5; only the last-epoch weights exist.",
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
    p.add_argument("--stage", default="full", choices=("full", "shared", "genome"),
                   help="`full` is upstream and is the anchor's path; `shared` fits the "
                        "transferable half on the regime's training scope; `genome` freezes it and "
                        "fits this chromosome's position tables alone (BENCHMARK_DESIGN.md §2)")
    p.add_argument("--init", type=Path, default=None,
                   help="the `--stage shared` checkpoint a `--stage genome` run transfers from")
    p.add_argument("--hparams-chrom", default="",
                   help="borrow this chromosome's UPSTREAM_HYPERPARAMS row; needed when the cache "
                        "stem is a packed multi-chromosome scope rather than one of the 23")
    ns = p.parse_args(argv)
    train_chromosome(ns.cache, ns.chrom, ns.out, contributor_mode=ns.contributor_mode,
                     device=ns.device, seed=ns.seed,
                     max_steps_per_stage=ns.max_steps_per_stage, epoch_scale=ns.epoch_scale,
                     tf32=not ns.no_tf32, stage=ns.stage, init=ns.init,
                     hparams_chrom=ns.hparams_chrom)
    return 0


if __name__ == "__main__":              # run as `python -m lavawizard.train`
    raise SystemExit(main())
