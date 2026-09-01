#!/usr/bin/env python3
"""CANDI_STORE -> one `(n_bins, n_tracks)` float32 matrix per chromosome, the shape Avocado trains on.

Adapted from `vendor/hpc_bin_tracks.py` (Max's 005), which read bigWigs off the challenge's Synapse
drop.  We read the store instead, so there is no binning to do: `STORE.md`'s grid is already
`floor(chr_len / 25)` and `BiosampleStore.pval` returns decoded `-log10 p` on exactly that grid.
The only work left is assembling the columns in a fixed, recorded order.

`arcsinh` is NOT applied here -- `train.py` applies it in place, as the vendored trainer does, so
the matrix on disk stays in the same space the store hands out and can be diffed against it.

    python -m bin_store --regime configs/regime.eic_val.json --out /scratch/.../binned --chrom chr20

`--regions` writes the OTHER shape: one `regions.npy` over the contained bins of every train
chromosome, packed onto the compact axis `index.py::region_layout` defines, plus the
`regions_layout.csv` that says which absolute bin each slot is.  That is the shared fit's scope
under a D32 regime (`configs/regime.eic_pilot.json`) and it replaces the whole-chromosome matrix,
never supplements it: a bin outside the BED is not written, so it cannot be trained on.

    python -m bin_store --regime configs/regime.eic_pilot.json --out /scratch/.../binned --regions
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from index import (load_regime, region_layout, train_columns, write_layout,   # noqa: E402
                   write_tracks)

CHUNK_BINS = 1_000_000

#: The 5 kbp genomic-factor stride of `vendor/avocado.py`, which the compact axis aligns to. Read
#: off the vendored model rather than restated, so a change there cannot leave this packing stale.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
from avocado import G5K_STRIDE                                                # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regime", required=True)
    ap.add_argument("--out", required=True, help="directory for <chrom>.npy + tracks.csv")
    ap.add_argument("--chrom", help="one whole chromosome; required unless --regions")
    ap.add_argument("--regions", action="store_true",
                    help="write the D32 BED scope over every train chromosome instead of one "
                         "whole chromosome (regions.npy + regions_layout.csv)")
    ap.add_argument("--chunk-bins", type=int, default=CHUNK_BINS)
    a = ap.parse_args(argv)
    if bool(a.chrom) == bool(a.regions):
        ap.error("exactly one of --chrom and --regions")

    from candi.store.reader import CorpusStore

    regime = load_regime(a.regime)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = "regions" if a.regions else a.chrom
    dest = out / f"{stem}.npy"
    if dest.exists():
        print(f"[bin] {dest} exists, skipping", flush=True)
        return 0

    # `spans` is the read plan and the write plan at once: [(chrom, first_bin, end_bin, row0), ...].
    # A whole chromosome is the degenerate one-span case, so the copy loop below is written once.
    with CorpusStore(regime["store"]) as corpus:
        if a.regions:
            spans, n_rows = region_layout(a.regime, regime["train_chroms"],
                                          coarse_stride=G5K_STRIDE)
            write_layout(out / "regions_layout.csv", spans, n_rows)
            kept = sum(b - s for _, s, b, _ in spans)
            print(f"[bin regions] {len(spans)} contained region(s) over "
                  f"{len(regime['train_chroms'])} train chromosome(s): {kept} bins on a "
                  f"{n_rows}-slot axis ({n_rows - kept} alignment slots)", flush=True)
        else:
            spans, n_rows = [(a.chrom, 0, int(corpus.n_bins(a.chrom)), 0)], int(corpus.n_bins(a.chrom))

        cols = train_columns(regime, corpus)
        write_tracks(out / "tracks.csv", cols, regime)
        print(f"[bin {stem}] {n_rows} bins x {len(cols)} training tracks "
              f"({n_rows * len(cols) * 4 / 2**30:.1f} GiB)", flush=True)

        by_bios: dict[str, list[tuple[int, str]]] = {}
        for i, (b, assay) in enumerate(cols):
            by_bios.setdefault(b, []).append((i, assay))

        tmp = dest.with_suffix(".npy.tmp")
        Y = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32,
                                      shape=(n_rows, len(cols)))
        t0 = time.time()
        for k, (b, rows) in enumerate(sorted(by_bios.items())):
            idx = [i for i, _ in rows]
            names = [nm for _, nm in rows]
            bs = corpus[b]
            for chrom, first, end, row0 in spans:
                for s in range(first, end, a.chunk_bins):
                    e = min(s + a.chunk_bins, end)
                    Y[row0 + s - first:row0 + e - first, idx] = bs.pval(chrom, s, e, assays=names)
            print(f"  [{k + 1}/{len(by_bios)}] {b}: {len(rows)} track(s) "
                  f"({time.time() - t0:.0f}s)", flush=True)
        Y.flush()
        del Y

    os.replace(tmp, dest)
    print(f"[bin {stem}] wrote {dest} ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
