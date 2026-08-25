"""Bin one chromosome's challenge bigwigs once, cache the result, and never read a bigwig again.

Reading 312 bigwigs for a chromosome costs ~2.4 s each (measured, chr21), and the training loop
walks the same data for hundreds of epochs. So the bigwigs are read once into a cache and the
trainer memory-maps it.

The cache for a chromosome is a directory:

```
<cache>/<chrom>/
  tracks.npy     (n_tracks, n_bins) float32   arcsinh(-log10 p), upstream's binning
  tercile.npy    (n_tracks, n_bins) int8      stage-1 targets, 0/1/2
  sums.npy       (n_marks,   n_bins) float32  per-mark sum over the training pool
  sumsq.npy      (n_marks,   n_bins) float32  per-mark sum of squares
  index.json     track order, mark order, cell order, counts, provenance
```

`sums`/`sumsq` are what make the contributor switch cheap. On Dataset 3 a cell id *is* the cell
type — there are no `T_`/`V_` prefixes — so excluding the target's cell type is exactly excluding
the target's own track, and both moments follow by subtraction:

```
upstream:  avg = S/k                    var = S2/k - avg^2
loo:       avg = (S - x)/(k-1)          var = (S2 - x^2)/(k-1) - avg^2
```

No per-track average has to be stored, and the mode becomes a runtime flag rather than a
re-preprocessing.

**The tercile is upstream's, ties and all.** `vals2cat` ranks with `method='first'` and then
`pd.qcut(..., 3)`, so tied values — and a `-log10 p` track is mostly ties at the floor — are split
by position in the array rather than by value. That is arbitrary, and it is what they trained on,
so `_terciles` reproduces it with a stable argsort rather than value thresholds.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

__all__ = ["training_tracks", "cache_dir", "build_cache", "load_cache", "CachedChrom"]


def training_tracks(meta: Sequence[Dict[str, str]]) -> Tuple[List[Tuple[str, str]], List[str], List[str]]:
    """`(tracks, cells, marks)` — upstream's own selection, in upstream's own order.

    Two filters, both from `02_guacamole6_pretrain.py:49-65`:

    1. Drop every assay that appears **only** in the training split. Upstream drops them because
       such an assay is never a target, so its embedding would train on nothing that is scored.
    2. Train on `DataType != "B"` — training *and* validation tracks both.

    `cells` and `marks` are taken from the metadata **after** filter 1 but **before** filter 2, so
    they still include blind cells. That is deliberate upstream and load-bearing: a blind cell's
    embedding row is trained through the other assays that cell does have, which is the entire
    tensor-factorisation mechanism for imputing it.
    """
    by_assay: Dict[str, set] = {}
    for r in meta:
        by_assay.setdefault(r["Assay"], set()).add(r["DataType"])
    train_only = {a for a, splits in by_assay.items() if splits == {"T"}}
    kept = [r for r in meta if r["Assay"] not in train_only]

    cells, marks = [], []
    for r in kept:                                   # first-seen order, as pandas `.unique()`
        if r["Cell_ID"] not in cells:
            cells.append(r["Cell_ID"])
        if r["Mark_ID"] not in marks:
            marks.append(r["Mark_ID"])
    tracks = [(r["Cell_ID"], r["Mark_ID"]) for r in kept if r["DataType"] != "B"]
    return tracks, cells, marks


def _terciles(x: np.ndarray) -> np.ndarray:
    """`vals2cat`: rank with ties broken by position, then cut into three equal-count groups."""
    n = x.shape[0]
    order = np.argsort(x, kind="stable")             # 'stable' == pandas rank(method='first')
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n, dtype=np.int64)
    return (rank * 3 // n).astype(np.int8)


def cache_dir(root: Path | str, chrom: str) -> Path:
    return Path(root) / str(chrom)


def build_cache(data_dirs: Sequence[Path], meta_tsv: Path, chrom: str, out_root: Path,
                *, verbose: bool = True) -> Path:
    """Bin every training track for `chrom` and write the cache. Idempotent: returns early if done.

    `data_dirs` are searched in order for each `C##M##.bigwig`; pass
    `[training_data, validation_data]` because upstream trains on both.
    """
    from . import dataset3

    out = cache_dir(out_root, chrom)
    if (out / "index.json").exists():
        if verbose:
            print(f"{chrom}: cache already present at {out}")
        return out
    out.mkdir(parents=True, exist_ok=True)

    meta = dataset3.read_meta(meta_tsv)
    tracks, cells, marks = training_tracks(meta)

    def find(cell: str, mark: str) -> Path:
        for d in data_dirs:
            p = dataset3.track_path(d, cell, mark)
            if p.exists():
                return p
        raise FileNotFoundError(f"{cell}{mark}.bigwig not in any of {[str(d) for d in data_dirs]}")

    paths = [find(c, m) for c, m in tracks]
    reader = dataset3.Dataset3Reader(data_dirs[0], chrom, max_cached=1)
    n_bins, n_tracks, n_marks = reader.n_bins, len(tracks), len(marks)
    if verbose:
        print(f"{chrom}: {n_tracks} tracks x {n_bins} bins, {len(cells)} cells, {n_marks} marks "
              f"({n_tracks * n_bins * 4 / 1e9:.1f} GB)", flush=True)

    arr = np.lib.format.open_memmap(out / "tracks.npy", mode="w+",
                                    dtype=np.float32, shape=(n_tracks, n_bins))
    ter = np.lib.format.open_memmap(out / "tercile.npy", mode="w+",
                                    dtype=np.int8, shape=(n_tracks, n_bins))
    sums = np.zeros((n_marks, n_bins), dtype=np.float64)
    sumsq = np.zeros((n_marks, n_bins), dtype=np.float64)
    counts = np.zeros(n_marks, dtype=np.int64)
    mark_ix = {m: i for i, m in enumerate(marks)}

    import pyBigWig
    t0 = time.time()
    for i, ((cell, mark), path) in enumerate(zip(tracks, paths)):
        bw = pyBigWig.open(str(path))
        try:
            values = np.array(bw.values(chrom, 0, reader.chrom_length))
        finally:
            bw.close()
        binned = dataset3.bin_arcsinh(values, reader.chrom_length)
        arr[i] = binned
        ter[i] = _terciles(binned)
        j = mark_ix[mark]
        sums[j] += binned
        sumsq[j] += binned.astype(np.float64) ** 2
        counts[j] += 1
        if verbose and (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{n_tracks}  {el/60:.1f} min  "
                  f"(eta {el/(i+1)*(n_tracks-i-1)/60:.1f} min)", flush=True)
    arr.flush(); ter.flush()
    np.save(out / "sums.npy", sums.astype(np.float32))
    np.save(out / "sumsq.npy", sumsq.astype(np.float32))

    index = {
        "chrom": chrom, "n_bins": int(n_bins), "chrom_length": int(reader.chrom_length),
        "grid": "upstream_ceil",
        "tracks": [list(t) for t in tracks], "cells": cells, "marks": marks,
        "mark_counts": {m: int(counts[mark_ix[m]]) for m in marks},
        "data_dirs": [str(d) for d in data_dirs], "meta": str(meta_tsv),
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "upstream": "github.com/ccchang0111/ENCODE_imputation_2019@d638b204",
    }
    (out / "index.json").write_text(json.dumps(index, indent=1) + "\n", encoding="utf-8")
    if verbose:
        print(f"{chrom}: done in {(time.time()-t0)/60:.1f} min -> {out}", flush=True)
    return out


class CachedChrom:
    """A built cache, memory-mapped, with the moment arithmetic the two modes need."""

    def __init__(self, root: Path | str, chrom: str, *, mmap: bool = True):
        d = cache_dir(root, chrom)
        idx = json.loads((d / "index.json").read_text(encoding="utf-8"))
        self.dir = d
        self.chrom = idx["chrom"]
        self.n_bins = int(idx["n_bins"])
        self.tracks: List[Tuple[str, str]] = [tuple(t) for t in idx["tracks"]]
        self.cells: List[str] = idx["cells"]
        self.marks: List[str] = idx["marks"]
        mode = "r" if mmap else None
        self.values = np.load(d / "tracks.npy", mmap_mode=mode)
        self.tercile = np.load(d / "tercile.npy", mmap_mode=mode)
        self.sums = np.load(d / "sums.npy")
        self.sumsq = np.load(d / "sumsq.npy")
        self.cell_ix = np.array([self.cells.index(c) for c, _ in self.tracks], dtype=np.int64)
        self.mark_ix = np.array([self.marks.index(m) for _, m in self.tracks], dtype=np.int64)
        self.mark_count = np.array([idx["mark_counts"][m] for m in self.marks], dtype=np.int64)

    @property
    def n_tracks(self) -> int:
        return len(self.tracks)

    def moments(self, track_idx: np.ndarray, pos: np.ndarray, x: np.ndarray,
                mode: str) -> Tuple[np.ndarray, np.ndarray]:
        """Per-sample `(average, variance)` for the sampled `(track, bin)` pairs.

        `x` is `values[track_idx, pos]`, passed in because the caller has already gathered it.
        On Dataset 3 a cell id is the cell type, so `"loo"` is exactly "subtract the target's own
        track" — see the module docstring.
        """
        m = self.mark_ix[track_idx]
        s = self.sums[m, pos].astype(np.float64)
        s2 = self.sumsq[m, pos].astype(np.float64)
        k = self.mark_count[m].astype(np.float64)
        if mode == "loo":
            s = s - x
            s2 = s2 - x.astype(np.float64) ** 2
            k = k - 1.0
            if np.any(k < 1.0):
                raise ValueError("a mark with one contributor has no leave-one-out average; "
                                 "§5 says such a track is skipped and listed")
        elif mode != "upstream":
            raise ValueError(f"mode must be 'upstream' or 'loo', got {mode!r}")
        avg = s / k
        var = np.maximum(s2 / k - avg * avg, 0.0)
        return avg.astype(np.float32), var.astype(np.float32)


def load_cache(root: Path | str, chrom: str, **kw) -> CachedChrom:
    return CachedChrom(root, chrom, **kw)


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Bin one chromosome's challenge bigwigs into a cache.")
    p.add_argument("--training", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--meta", type=Path, required=True)
    p.add_argument("--chrom", required=True)
    p.add_argument("--cache", type=Path, required=True)
    ns = p.parse_args(argv)
    build_cache([ns.training, ns.validation], ns.meta, ns.chrom, ns.cache)
    return 0


if __name__ == "__main__":              # run as `python -m lavawizard.preprocess`
    raise SystemExit(main())
