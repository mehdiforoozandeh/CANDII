"""Predict the blind tracks, and place the port against their own submission. The §7.4 anchor.

The anchor question is not "do our arrays look like theirs" but **"does the port, retrained on the
challenge's own training data, score like their submission scored"**. So this compares three
things on the challenge's own grid, in raw `-log10 p`:

```
ours   vs blind_truth
theirs vs blind_truth      <- their submitted bigwig, syn21976480
ours   vs theirs           <- how close the two methods are to each other
```

`theirs vs blind_truth` is computed here rather than quoted, so the comparison is like-for-like:
one binning, one masking, one measure, both methods. The published leaderboard number is a
different instrument (the 001 vendored scorer, §7.5, t54's work), and this file deliberately does
not pretend to be it — see the caveat `summarise` stamps into every record.

**Prediction inputs for a blind track.** A blind cell has no track for the target mark, so it
contributes nothing to that mark's average and the pooled per-mark moments are already
leave-one-out. That is exactly what `01_data_preproc.py` does: it looks the average up from a
*training* cell of the same mark. `predict_track` therefore always uses the pooled moments and
refuses a `loo` request, which would subtract a track that was never in the sum.

**Space.** The model works in `arcsinh(-log10 p)`; `emit.invert_arcsinh` brings a prediction back
(clip at zero, then `sinh`), and truth is binned with `transform="none"`. Comparing in the wrong
space is the easiest way to produce a number that looks fine and means nothing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from . import dataset3, emit
from .model import Guacamole
from .preprocess import CachedChrom

__all__ = ["load_checkpoint", "predict_track", "compare", "summarise", "main"]

#: Quoting rule, carried into every record this module writes.
CAVEAT = ("Dataset-3 anchor. Scored here with a like-for-like MSE/Pearson on the upstream 25 bp "
          "ceil grid in raw -log10 p, NOT with the 001 vendored EIC scorer (RIVALS_PLAN.md §7.5). "
          "Comparable to the 'theirs' column in the same table and to nothing else; never quote "
          "against a published leaderboard row, and never against a Dataset-2 (internal) number.")


def load_checkpoint(path: Path | str, device: str = "cpu") -> Tuple[Guacamole, Dict]:
    obj = torch.load(str(path), map_location=device, weights_only=False)
    model = Guacamole(**obj["config"]).to(device)
    model.load_state_dict(obj["state_dict"])
    model.eval()
    return model, obj


@torch.no_grad()
def predict_track(model: Guacamole, cache: CachedChrom, meta_obj: Dict, cell: str, mark: str,
                  *, device: str = "cpu", batch: int = 200_000) -> np.ndarray:
    """One blind track, whole chromosome, in `arcsinh(-log10 p)`.

    Uses the pooled per-mark average and variance — correct for a blind cell, which contributes
    nothing to the sum. See the module docstring.
    """
    cells, marks = meta_obj["cells"], meta_obj["marks"]
    if cell not in cells:
        raise ValueError(f"{cell} has no embedding row; the checkpoint knows {len(cells)} cells")
    if mark not in marks:
        raise ValueError(f"{mark} has no embedding row; the checkpoint knows {len(marks)} marks")
    ci, mi = cells.index(cell), marks.index(mark)
    k = int(cache.mark_count[mi])
    if k < 1:
        raise ValueError(f"{mark} has no contributors in the cache; nothing to average")

    n = cache.n_bins
    out = np.empty(n, dtype=np.float32)
    dev = torch.device(device)
    for s in range(0, n, batch):
        e = min(s + batch, n)
        pos = np.arange(s, e, dtype=np.int64)
        avg = (cache.sums[mi, s:e] / k).astype(np.float32)
        var = np.maximum(cache.sumsq[mi, s:e] / k - avg.astype(np.float64) ** 2, 0.0).astype(np.float32)
        out[s:e] = model(
            celltype=torch.full((e - s,), ci, dtype=torch.long, device=dev),
            assay=torch.full((e - s,), mi, dtype=torch.long, device=dev),
            pos25=torch.from_numpy(pos).to(dev),
            average=torch.from_numpy(avg).to(dev),
            variance=torch.from_numpy(var).to(dev),
        ).float().cpu().numpy()
    return out


def compare(ours: np.ndarray, theirs: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    """MSE and Pearson for the three pairings, on whatever bins all three cover."""
    n = min(len(ours), len(theirs), len(truth))
    o, t, y = (np.asarray(a[:n], dtype=np.float64) for a in (ours, theirs, truth))

    def pair(a, b) -> Tuple[float, float]:
        sa, sb = a.std(), b.std()
        r = float(np.corrcoef(a, b)[0, 1]) if sa > 0 and sb > 0 else float("nan")
        return float(np.mean((a - b) ** 2)), r

    mse_o, r_o = pair(o, y)
    mse_t, r_t = pair(t, y)
    mse_ot, r_ot = pair(o, t)
    return {
        "n_bins": int(n),
        "ours_vs_truth_mse": mse_o, "ours_vs_truth_pearson": r_o,
        "theirs_vs_truth_mse": mse_t, "theirs_vs_truth_pearson": r_t,
        "ours_vs_theirs_mse": mse_ot, "ours_vs_theirs_pearson": r_ot,
        "mse_ratio_ours_over_theirs": mse_o / mse_t if mse_t > 0 else float("nan"),
        "truth_mean": float(y.mean()), "truth_var": float(y.var()),
    }


def summarise(rows: Sequence[Dict]) -> Dict:
    """Per-track rows into one record, with the caveat attached where a reader cannot miss it."""
    ok = [r for r in rows if np.isfinite(r.get("ours_vs_truth_mse", np.nan))]
    def mean_of(key):
        return float(np.mean([r[key] for r in ok])) if ok else float("nan")
    return {
        "caveat": CAVEAT,
        "n_tracks": len(ok),
        "macro_ours_vs_truth_mse": mean_of("ours_vs_truth_mse"),
        "macro_theirs_vs_truth_mse": mean_of("theirs_vs_truth_mse"),
        "macro_ours_vs_truth_pearson": mean_of("ours_vs_truth_pearson"),
        "macro_theirs_vs_truth_pearson": mean_of("theirs_vs_truth_pearson"),
        "macro_ours_vs_theirs_pearson": mean_of("ours_vs_theirs_pearson"),
        "tracks": list(rows),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--chrom", required=True)
    p.add_argument("--their-tracks", type=Path, required=True, help="submitted_tracks/Guacamole")
    p.add_argument("--blind-truth", type=Path, required=True)
    p.add_argument("--meta", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--limit", type=int, default=0, help="0 = every blind track present")
    p.add_argument("--pred-root", type=Path, default=None,
                   help="also write a §4.1 prediction root here")
    ns = p.parse_args(argv)

    model, obj = load_checkpoint(ns.checkpoint, ns.device)
    # `mmap=True` on purpose: prediction touches only `sums`, `sumsq` and `mark_count`,
    # never `values` or `tercile`. Loading those into RAM would cost 15 GiB on chr1 for
    # arrays this path never reads — and a smaller memory ask schedules sooner.
    cache = CachedChrom(ns.cache, ns.chrom, mmap=True)
    meta = dataset3.read_meta(ns.meta)
    blind = [(r["Cell_ID"], r["Mark_ID"]) for r in meta if r["DataType"] == "B"]
    blind = [(c, m) for c, m in blind
             if m in obj["marks"] and c in obj["cells"]
             and dataset3.track_path(ns.their_tracks, c, m).exists()
             and dataset3.track_path(ns.blind_truth, c, m).exists()]
    if ns.limit:
        blind = blind[:ns.limit]
    print(f"{ns.chrom}: {len(blind)} blind tracks with both their submission and the truth",
          flush=True)

    rows: List[Dict] = []
    for i, (cell, mark) in enumerate(blind, 1):
        raw = predict_track(model, cache, obj, cell, mark, device=ns.device)
        ours = emit.invert_arcsinh(raw)
        theirs = dataset3.read_binned(dataset3.track_path(ns.their_tracks, cell, mark),
                                      ns.chrom, transform="none")
        truth = dataset3.read_binned(dataset3.track_path(ns.blind_truth, cell, mark),
                                     ns.chrom, transform="none")
        row = {"cell": cell, "mark": mark, **compare(ours, theirs, truth)}
        rows.append(row)
        print(f"  [{i:2d}/{len(blind)}] {cell}{mark}  ours {row['ours_vs_truth_mse']:.5f}  "
              f"theirs {row['theirs_vs_truth_mse']:.5f}  ratio "
              f"{row['mse_ratio_ours_over_theirs']:.3f}  r(ours,theirs) "
              f"{row['ours_vs_theirs_pearson']:.4f}", flush=True)
        if ns.pred_root:
            emit.write_track(ns.pred_root, emit.Pair(cell, cell), mark, ns.chrom, ours,
                             n_bins=len(ours), already_inverted=True)

    rec = summarise(rows)
    rec.update({"chrom": ns.chrom, "checkpoint": str(ns.checkpoint),
                "contributor_mode": obj.get("contributor_mode"),
                "their_tracks": str(ns.their_tracks), "blind_truth": str(ns.blind_truth)})
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    ns.out.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print(f"\nmacro MSE   ours {rec['macro_ours_vs_truth_mse']:.5f}   "
          f"theirs {rec['macro_theirs_vs_truth_mse']:.5f}")
    print(f"macro r     ours {rec['macro_ours_vs_truth_pearson']:.4f}   "
          f"theirs {rec['macro_theirs_vs_truth_pearson']:.4f}   "
          f"ours~theirs {rec['macro_ours_vs_theirs_pearson']:.4f}")
    print(f"-> {ns.out}")
    if ns.pred_root:
        emit.write_manifest(ns.pred_root, version="0.1.0",
                            generated_by="lavawizard.anchor",
                            contributor_mode=str(obj.get("contributor_mode")),
                            weights=f"ported-retrain:{Path(ns.checkpoint).name}",
                            notes=CAVEAT)
    return 0


if __name__ == "__main__":              # run as `python -m lavawizard.anchor`
    raise SystemExit(main())
