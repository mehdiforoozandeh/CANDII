#!/usr/bin/env python3
"""t16 — measure the identity-copy leak the DSF ladder gives the model.

    python tools/dsf_leak.py --h5 <panel.h5> [--batches 20]
    python tools/dsf_leak.py --store <regime.json> [--batches 20] [--deterministic]

THE LEAK. Every batch draws a per-assay `x_dsf` and `y_dsf`. When they land on the same level the
model's INPUT column for that assay can be the same array as its TARGET column — a free identity
copy, worth exactly one `argmax` to learn and nothing to generalize from. It is not a bug in either
loader; it is what "sample x and y from the same ladder" means, and the two data paths differ in how
often it bites:

* the **bake** materializes the ladder, so `counts_dsf{d}` at equal `d` IS one stored array read
  twice — always identical, whatever the RNG is doing.
* the **store** generates the ladder by thinning DSF1 (D6), so equal `d` means two DRAWS. Whether
  they collide depends on the RNG regime (D22), which is what the `--deterministic` flag exposes.

WHAT THIS REPORTS, per (sample, assay) cell of every batch:

    avail      the assay is available for this biosample (`x_avail > 0`)
    equal_dsf  x_dsf == y_dsf on this cell
    identical  the x column and the y column are bit-identical
    leak       identical AND the cell is OBSERVED, i.e. not clozed by the masker

`leak` is the number that matters. A clozed column's input is the CLOZE sentinel, so there is
nothing to copy there; the leak lands entirely on the denoising half of the objective, which is the
half that is not scored (`EVAL.md`) and the half whose loss curve moves first.

This measures a RATE, not an effect. How much a leaked column changes what the model learns is a
training question and needs a run.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

import torch

from candi.batch import make_masker, prepare_masked_batch


def _dataset(args):
    if args.h5:
        from candi.dataset import CandiKitH5Dataset

        return CandiKitH5Dataset(args.h5, args.regime, train=True, batch_size=args.batch_size,
                                 dsf_sampling=args.dsf_sampling, seed=args.seed, shuffle=False,
                                 h5_cache_ram=False)
    from candi.store.dataset import StoreDataset

    return StoreDataset(args.store, train=True, batch_size=args.batch_size,
                        dsf_sampling=args.dsf_sampling, seed=args.seed, shuffle=False,
                        deterministic=args.deterministic)


def measure(ds, n_batches: int, *, p_full_assay: float = 1.0, mask_fraction: float = 0.2) -> dict:
    masker = make_masker(p_full_assay=p_full_assay, mask_fraction=mask_fraction)
    dev = torch.device("cpu")
    c = Counter()
    by_dsf: Counter = Counter()
    for bi, batch in enumerate(ds):
        if bi >= n_batches:
            break
        prep = prepare_masked_batch(batch, masker, dev, apply_mask=True)
        if prep is None:
            continue
        xd, yd = batch["x_dsf"], batch["y_dsf"]
        x_raw, y_raw = batch["x_data"], batch["y_data"]
        # `prep` is row-subset when a sample has no query, so read the maps off it and re-derive the
        # row mapping is not worth it: measure availability off the RAW batch and the cloze off the
        # masker's own map, which is what the model sees.
        obs = prep["observed_map"].any(dim=1)              # [B', F] — unmasked and supervised
        n_rows = min(x_raw.shape[0], obs.shape[0])
        for j in range(n_rows):
            for fi in range(x_raw.shape[2]):
                if float(batch["x_avail"][j, fi]) <= 0:
                    continue
                c["avail"] += 1
                eq = int(xd[j, fi]) == int(yd[j, fi])
                ident = bool(torch.equal(x_raw[j, :, fi], y_raw[j, :, fi]))
                c["equal_dsf"] += int(eq)
                c["identical"] += int(ident)
                if ident and bool(obs[j, fi]):
                    c["leak"] += 1
                if eq:
                    by_dsf[f"dsf{int(xd[j, fi])}"] += 1
                    if ident:
                        by_dsf[f"dsf{int(xd[j, fi])}_identical"] += 1
    n = max(1, c["avail"])
    return {
        "cells_available": c["avail"],
        "equal_dsf": c["equal_dsf"],
        "identical": c["identical"],
        "leak_observed": c["leak"],
        "rate_equal_dsf": c["equal_dsf"] / n,
        "rate_identical": c["identical"] / n,
        "rate_leak_observed": c["leak"] / n,
        "identical_given_equal_dsf": (c["identical"] / c["equal_dsf"]) if c["equal_dsf"] else None,
        "by_equal_dsf_level": dict(sorted(by_dsf.items())),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--h5", default=None)
    src.add_argument("--store", default=None, help="a regime file, as `train.py --store` takes")
    ap.add_argument("--regime", default="type1", help="h5 only: the MASKING regime")
    ap.add_argument("--dsf-sampling", dest="dsf_sampling", default="uniform",
                    choices=["off", "uniform", "x_eq_y", "upsample_only"])
    ap.add_argument("--deterministic", action="store_true",
                    help="store only: D22's counter-based eval RNG instead of the free-running one")
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None, help="write the report here as well as printing it")
    args = ap.parse_args()

    ds = _dataset(args)
    rep = measure(ds, args.batches)
    rep["source"] = {"h5": args.h5, "store": args.store, "dsf_sampling": args.dsf_sampling,
                     "deterministic": bool(args.deterministic), "batches": args.batches,
                     "batch_size": args.batch_size, "seed": args.seed}
    text = json.dumps(rep, indent=2)
    print(text)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    given = rep["identical_given_equal_dsf"]
    given_s = "n/a" if given is None else f"{given:.1%}"
    print(f"\nleak on {rep['rate_leak_observed']:.1%} of available columns "
          f"({rep['leak_observed']}/{rep['cells_available']}); "
          f"x_dsf == y_dsf on {rep['rate_equal_dsf']:.1%}, and {given_s} of those are "
          f"bit-identical.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
