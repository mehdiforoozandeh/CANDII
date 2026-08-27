"""CANDI_STORE -> the same per-chromosome cache `preprocess.py` builds, then the P1/P2 predictions.

This is the **our-EIC** side of the port. `dataset3.py` reads the challenge's own bigwigs and must
never import `candi`; this file reads our store and therefore must. Nothing else crosses: both
produce the identical cache layout, so `train.py` and `model.py` cannot tell which corpus they are
looking at, which is the point — the anchor and the deliverable are the same code on two corpora.

**No binning happens here.** `STORE.md`'s grid is already `floor(chr_len / 25)` and
`BiosampleStore.pval` returns decoded `-log10 p` on exactly that grid, so the only transform is
`arcsinh` on the binned value. Upstream applies `arcsinh` per *base* and then averages into a bin;
we cannot, and the difference is recorded in the README rather than papered over. Their grid also
*ceils* where the store *floors*, so a chromosome here is one bin shorter than the anchor's.

**§6.2 is enforced in `train_columns`, and it raises rather than filters.** A fairness rule that
silently drops a column is a fairness rule nobody can audit. The T_/V_ identification comes from
the regime's own `eval_pairs`, never from splitting a biosample name — `STORE.md` D16 says a
biosample id is opaque, so a regime declaring `[T_A, V_B]` is honoured exactly as written.

The shape of this module deliberately mirrors `competitors/avocado/index.py` and `bin_store.py`.
The two methods are separate deliverables under `RIVALS_PLAN.md` §3 and neither imports the other,
so the mirroring is by convention, not by shared code — one reading of §6.2 per method, in the same
place, checkable side by side.

```bash
python -m lavawizard.store_eic cache   --regime configs/regime.eic_val.json --chrom chr21 \
       --cache /scratch/.../eic_cache
python -m lavawizard.store_eic predict --regime configs/regime.eic_val.json --chrom chr21 \
       --cache /scratch/.../eic_cache --checkpoint runs/eic/guacamole_chr21.pt \
       --pred-root runs/eic/pred --clip
```
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from . import emit, preprocess

__all__ = ["FairnessError", "load_regime", "cell_index", "train_columns",
           "build_cache_from_store", "predict_chrom"]

CHUNK_BINS = 1_000_000


class FairnessError(RuntimeError):
    """A track that `RIVALS_PLAN.md` §6.2 forbids reached the training pool."""


def load_regime(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cell_index(regime: dict) -> Tuple[List[str], Dict[str, int]]:
    """`(cell names, biosample id -> cell index)`; a declared `[T_X, V_X]` pair shares one index.

    Guacamole factorises `(celltype, assay, position)`, so `T_X` and `V_X` must be one cell or a
    V_ target has no embedding to impute from. The embedding is still fitted from the T_ side
    alone — that is `train_columns`'s job, not this one — and the class name is the T_ id, so a
    checkpoint written here names cells in T_ terms only.
    """
    names = sorted(regime["biosamples"]["train"])
    ix = {c: i for i, c in enumerate(names)}
    for row in regime.get("eval_pairs", []):
        src, tgt = str(row[0]), str(row[1])
        if src not in ix:
            raise FairnessError(
                f"eval_pairs names `{src}` as an input biosample but biosamples.train does not; "
                f"there would be no fitted embedding for its cell.")
        ix[tgt] = ix[src]
    return names, ix


def train_columns(regime: dict, corpus) -> List[Tuple[str, str]]:
    """`[(biosample, assay), ...]` — every track this method is allowed to see. §6.2 lives here.

    Drawn from `biosamples.train` and from nowhere else. A train biosample that is also an eval
    **target** raises: that is the leak §6.2 forbids outright, and it must not be reachable by
    reading past a warning.
    """
    train = list(regime["biosamples"]["train"])
    targets = {str(row[1]) for row in regime.get("eval_pairs", [])}
    leaked = sorted(set(train) & targets)
    if leaked:
        raise FairnessError(
            f"biosamples.train contains {leaked}, which eval_pairs also names as imputation "
            f"TARGETS. Training on a target biosample's tracks is the leak §6.2 forbids.")
    declared = set(regime["assays"])
    cols: List[Tuple[str, str]] = []
    for b in sorted(train):
        have = set(corpus[b].assays()) & declared
        cols.extend((b, a) for a in regime["assays"] if a in have)
    if not cols:
        raise FairnessError("no training tracks: no declared train biosample carries a declared "
                            "assay.")
    return cols


def build_cache_from_store(regime_path: Path | str, chrom: str, out_root: Path | str,
                           *, chunk_bins: int = CHUNK_BINS, verbose: bool = True) -> Path:
    """Write `<cache>/<chrom>/` in exactly `preprocess.build_cache`'s layout, from our store.

    Idempotent: an existing `index.json` returns early, so a re-run of a partly finished array
    costs nothing.
    """
    from candi.store.reader import CorpusStore

    regime = load_regime(regime_path)
    out = preprocess.cache_dir(out_root, chrom)
    if (out / "index.json").exists():
        if verbose:
            print(f"{chrom}: cache already present at {out}", flush=True)
        return out
    out.mkdir(parents=True, exist_ok=True)

    cells, _ = cell_index(regime)
    marks: List[str] = list(regime["assays"])
    mark_ix = {m: i for i, m in enumerate(marks)}

    with CorpusStore(regime["store"]) as corpus:
        cols = train_columns(regime, corpus)
        n_bins = int(corpus.n_bins(chrom))
        n_tracks, n_marks = len(cols), len(marks)
        if verbose:
            print(f"{chrom}: {n_tracks} tracks x {n_bins} bins, {len(cells)} cells, {n_marks} "
                  f"marks ({n_tracks * n_bins * 4 / 2**30:.1f} GiB)", flush=True)

        arr = np.lib.format.open_memmap(out / "tracks.npy", mode="w+",
                                        dtype=np.float32, shape=(n_tracks, n_bins))
        ter = np.lib.format.open_memmap(out / "tercile.npy", mode="w+",
                                        dtype=np.int8, shape=(n_tracks, n_bins))
        sums = np.zeros((n_marks, n_bins), dtype=np.float64)
        sumsq = np.zeros((n_marks, n_bins), dtype=np.float64)
        counts = np.zeros(n_marks, dtype=np.int64)
        maxima = np.zeros(n_marks, dtype=np.float64)

        by_bios: Dict[str, List[Tuple[int, str]]] = {}
        for i, (b, assay) in enumerate(cols):
            by_bios.setdefault(b, []).append((i, assay))

        t0 = time.time()
        for k, (b, rows) in enumerate(sorted(by_bios.items())):
            names = [nm for _, nm in rows]
            bs = corpus[b]
            block = np.empty((len(rows), n_bins), dtype=np.float32)
            for s in range(0, n_bins, chunk_bins):
                e = min(s + chunk_bins, n_bins)
                block[:, s:e] = np.arcsinh(
                    np.asarray(bs.pval(chrom, s, e, assays=names), dtype=np.float32)).T
            for (i, assay), row in zip(rows, block):
                arr[i] = row
                ter[i] = preprocess._terciles(row)
                j = mark_ix[assay]
                sums[j] += row
                sumsq[j] += row.astype(np.float64) ** 2
                counts[j] += 1
                maxima[j] = max(maxima[j], float(row.max()))
            if verbose:
                print(f"  [{k+1}/{len(by_bios)}] {b}: {len(rows)} track(s) "
                      f"({time.time()-t0:.0f}s)", flush=True)
        arr.flush(); ter.flush()

    np.save(out / "sums.npy", sums.astype(np.float32))
    np.save(out / "sumsq.npy", sumsq.astype(np.float32))
    index = {
        "chrom": chrom, "n_bins": int(n_bins), "grid": "store_floor",
        "tracks": [list(t) for t in cols], "cells": cells, "marks": marks,
        "mark_counts": {m: int(counts[mark_ix[m]]) for m in marks},
        "mark_max": {m: float(maxima[mark_ix[m]]) for m in marks},
        "source": "CANDI_STORE", "store": str(regime["store"]),
        "regime": str(Path(regime_path).name),
        "signal": "arcsinh(pval) on the store's binned -log10 p",
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "upstream": "github.com/ccchang0111/ENCODE_imputation_2019@d638b204",
    }
    (out / "index.json").write_text(json.dumps(index, indent=1) + "\n", encoding="utf-8")
    if verbose:
        print(f"{chrom}: done in {(time.time()-t0)/60:.1f} min -> {out}", flush=True)
    return out


def _declared_tracks(regime: dict, corpus) -> List[Tuple[str, str, str]]:
    """`[(input, target, assay), ...]` — the §4.1 tracks a P1/P2 run must cover.

    An assay is declared for a pair when the **target** biosample carries it, because the target is
    where the truth comes from. Read from the store rather than assumed, so a pair with a partial
    panel produces exactly the tracks the scorer will look for.
    """
    declared = list(regime["assays"])
    out: List[Tuple[str, str, str]] = []
    for row in regime.get("eval_pairs", []):
        src, tgt = str(row[0]), str(row[1])
        have = set(corpus[tgt].assays())
        out.extend((src, tgt, a) for a in declared if a in have)
    return out


def predict_chrom(regime_path: Path | str, chrom: str, cache_root: Path | str,
                  checkpoint: Path | str, pred_root: Path | str, *,
                  clip: bool, device: str = "cpu", batch: int = 200_000,
                  verbose: bool = True) -> List[str]:
    """Every declared track for one chromosome, written as §4.1 npz. Returns the directory names.

    The contributor average is pooled over the mark's training tracks **minus the pair's input
    cell** — §6.2's exclusion, taken from the declared pair and not from a name suffix. On our
    store the target itself is never in the pool (a V_ track is not a training track), so this
    subtraction removes the one biosample that shares the target's cell.
    """
    import torch

    from .anchor import load_checkpoint
    from candi.store.reader import CorpusStore

    regime = load_regime(regime_path)
    cache = preprocess.CachedChrom(cache_root, chrom, mmap=True)
    if clip and cache.mark_max is None:
        raise ValueError(f"{chrom}: --clip needs `mark_max`, and this cache predates it. Rebuild "
                         f"the cache; a cap guessed from a cache that never measured one is worse "
                         f"than no cap.")
    model, meta = load_checkpoint(checkpoint, device=device)
    cells, cix = cell_index(regime)
    if meta["cells"] != cells or meta["marks"] != cache.marks:
        raise ValueError(
            f"{chrom}: the checkpoint's index space is not this regime's. An off-by-one in the "
            f"cell or mark order silently predicts the wrong track, so this refuses rather than "
            f"guesses. checkpoint {len(meta['cells'])} cells / {len(meta['marks'])} marks, "
            f"regime {len(cells)} / {len(cache.marks)}.")
    dev = torch.device(device)
    written: List[str] = []

    with CorpusStore(regime["store"]) as corpus:
        tracks = _declared_tracks(regime, corpus)
        n_bins = int(corpus.n_bins(chrom))
    if n_bins != cache.n_bins:
        raise ValueError(f"{chrom}: store says {n_bins} bins, cache says {cache.n_bins}")

    cell_of_track = np.array([cix[b] for b, _ in cache.tracks], dtype=np.int64)
    t0 = time.time()
    for i, (src, tgt, assay) in enumerate(tracks):
        ci, mi = cix[src], cache.marks.index(assay)
        rows = np.flatnonzero((cache.mark_ix == mi) & (cell_of_track == ci))
        k = int(cache.mark_count[mi]) - len(rows)
        if k < 1:
            print(f"  skip {src}->{tgt} {assay}: 0 contributors after §6.2 exclusion", flush=True)
            continue
        raw = np.empty(n_bins, dtype=np.float32)
        for s in range(0, n_bins, batch):
            e = min(s + batch, n_bins)
            drop = cache.values[rows, s:e].astype(np.float64) if len(rows) else 0.0
            ssum = cache.sums[mi, s:e].astype(np.float64) - (drop.sum(0) if len(rows) else 0.0)
            ssq = cache.sumsq[mi, s:e].astype(np.float64) - ((drop ** 2).sum(0) if len(rows) else 0.0)
            avg = (ssum / k).astype(np.float32)
            var = np.maximum(ssq / k - avg.astype(np.float64) ** 2, 0.0).astype(np.float32)
            with torch.no_grad():
                raw[s:e] = model(
                    celltype=torch.full((e - s,), ci, dtype=torch.long, device=dev),
                    assay=torch.full((e - s,), mi, dtype=torch.long, device=dev),
                    pos25=torch.arange(s, e, dtype=torch.long, device=dev),
                    average=torch.from_numpy(avg).to(dev),
                    variance=torch.from_numpy(var).to(dev),
                ).float().cpu().numpy()
        cap = float(np.sinh(cache.mark_max[mi])) if clip else None
        emit.write_track(pred_root, emit.Pair(src, tgt), assay, chrom, raw,
                         n_bins=n_bins, clip_max=cap)
        written.append(emit.track_dirname(emit.Pair(src, tgt), assay))
        if verbose:
            print(f"  [{i+1}/{len(tracks)}] {src}->{tgt} {assay}  k={k}  "
                  f"cap={'off' if cap is None else f'{cap:.1f}'}  ({time.time()-t0:.0f}s)",
                  flush=True)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("cache", help="build one chromosome's cache from the store")
    c.add_argument("--regime", required=True)
    c.add_argument("--chrom", required=True)
    c.add_argument("--cache", type=Path, required=True)
    c.add_argument("--chunk-bins", type=int, default=CHUNK_BINS)

    q = sub.add_parser("predict", help="write one chromosome's declared §4.1 tracks")
    q.add_argument("--regime", required=True)
    q.add_argument("--chrom", required=True)
    q.add_argument("--cache", type=Path, required=True)
    q.add_argument("--checkpoint", type=Path, required=True)
    q.add_argument("--pred-root", type=Path, required=True)
    q.add_argument("--device", default="cpu")
    q.add_argument("--clip", action="store_true",
                   help="cap signal_mu at the mark's training max on this chromosome (PI ruling "
                        "2026-08-26); off reproduces the faithful port")
    q.add_argument("--manifest", action="store_true",
                   help="also write manifest.json — pass on one task only, not all 23")

    ns = p.parse_args(argv)
    if ns.cmd == "cache":
        build_cache_from_store(ns.regime, ns.chrom, ns.cache, chunk_bins=ns.chunk_bins)
        return 0

    predict_chrom(ns.regime, ns.chrom, ns.cache, ns.checkpoint, ns.pred_root,
                  clip=ns.clip, device=ns.device)
    if ns.manifest:
        emit.write_manifest(
            ns.pred_root, version="0.1.0", generated_by="lavawizard.store_eic",
            contributor_mode="loo", weights=f"ported-retrain:{Path(ns.checkpoint).name}",
            clip=bool(ns.clip),
            notes=("Retrained on our EIC store, training-split biosamples only (§6.2). "
                   "Point-only pval arm: sigma comes from the §6.1 table, not from this root."))
    return 0


if __name__ == "__main__":              # run as `python -m lavawizard.store_eic`
    raise SystemExit(main())
