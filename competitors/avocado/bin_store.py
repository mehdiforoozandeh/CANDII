#!/usr/bin/env python3
"""CANDI_STORE -> one `(n_bins, n_tracks)` float32 matrix per chromosome, the shape Avocado trains on.

Adapted from `vendor/hpc_bin_tracks.py` (Max's 005), which read bigWigs off the challenge's Synapse
drop.  We read the store instead, so there is no binning to do: `STORE.md`'s grid is already
`floor(chr_len / 25)` and `BiosampleStore.pval` returns decoded `-log10 p` on exactly that grid.
The only work left is assembling the columns in a fixed, recorded order.

`arcsinh` is NOT applied here -- `train.py` applies it in place, as the vendored trainer does, so
the matrix on disk stays in the same space the store hands out and can be diffed against it.

    python -m bin_store --regime configs/regime.eic_val.json --out /scratch/.../binned --chrom chr20
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from index import load_regime, train_columns, write_tracks     # noqa: E402

CHUNK_BINS = 1_000_000


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regime", required=True)
    ap.add_argument("--out", required=True, help="directory for <chrom>.npy + tracks.csv")
    ap.add_argument("--chrom", required=True)
    ap.add_argument("--chunk-bins", type=int, default=CHUNK_BINS)
    a = ap.parse_args(argv)

    from candi.store.reader import CorpusStore

    regime = load_regime(a.regime)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"{a.chrom}.npy"
    if dest.exists():
        print(f"[bin] {dest} exists, skipping", flush=True)
        return 0

    with CorpusStore(regime["store"]) as corpus:
        cols = train_columns(regime, corpus)
        write_tracks(out / "tracks.csv", cols, regime)
        n_bins = int(corpus.n_bins(a.chrom))
        print(f"[bin {a.chrom}] {n_bins} bins x {len(cols)} training tracks "
              f"({n_bins * len(cols) * 4 / 2**30:.1f} GiB)", flush=True)

        by_bios: dict[str, list[tuple[int, str]]] = {}
        for i, (b, assay) in enumerate(cols):
            by_bios.setdefault(b, []).append((i, assay))

        tmp = dest.with_suffix(".npy.tmp")
        Y = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32,
                                      shape=(n_bins, len(cols)))
        t0 = time.time()
        for k, (b, rows) in enumerate(sorted(by_bios.items())):
            idx = [i for i, _ in rows]
            names = [nm for _, nm in rows]
            bs = corpus[b]
            for s in range(0, n_bins, a.chunk_bins):
                e = min(s + a.chunk_bins, n_bins)
                Y[s:e, idx] = bs.pval(a.chrom, s, e, assays=names)
            print(f"  [{k + 1}/{len(by_bios)}] {b}: {len(rows)} track(s) "
                  f"({time.time() - t0:.0f}s)", flush=True)
        Y.flush()
        del Y

    os.replace(tmp, dest)
    print(f"[bin {a.chrom}] wrote {dest} ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
