#!/usr/bin/env python3
"""Predict every declared track on one chromosome and write it in the RIVALS_PLAN.md 4.1 format.

Adapted from `vendor/hpc_predict.py` (Max's 005, md5 4806d6337764bca89326d136a8313683).  The
forward pass, the bf16 autocast, the `sinh` inversion and the clip at zero are the vendored code.
Two things changed:

1. **What gets predicted.**  005 predicted a fixed list of 51 challenge blind-test experiments.  We
   predict exactly the tracks our regime DECLARES, and we ask `candi.bench.external._expected` for
   that list rather than deriving it -- so the set this writes and the set the scorer demands are
   the same set by construction, and a track the regime does not declare cannot be emitted by
   accident.

2. **What gets written.**  005 wrote one `(51, n_bins)` `.npy` per chromosome and converted to
   bigWig.  We write `<pred_root>/<input>__<target>__<assay>/<chrom>.npz` holding `signal_mu`, on
   the store's absolute `floor(chr_len / 25)` grid.  `signal_mu` only: Avocado emits a point in
   `-log10 p` and nothing else.  It has no count head (B1b forbids inventing a depth), no variance
   (that is the sigma-table's job, 6.1) and no peak head (the scorer falls back to coverage
   ranking and labels the row).

The cell embedding a V_ target is predicted from is the T_ side's, because `index.py` gives the
declared pair one cell index and only the T_ side ever entered training.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "vendor"))
from avocado import Avocado                                     # noqa: E402  (vendored, unmodified)
from index import assay_index, cell_index, load_regime          # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", required=True)
    ap.add_argument("--chrom", help="required unless --write-manifest")
    ap.add_argument("--shared", help="the shared-mode checkpoint (chr19, or the BED scope)")
    ap.add_argument("--genome", help="this chromosome's genome-mode checkpoint")
    ap.add_argument("--out", required=True, help="prediction root (RIVALS_PLAN.md 4.1)")
    ap.add_argument("--batch-positions", type=int, default=8192)
    ap.add_argument("--compress", action="store_true",
                    help="np.savez_compressed instead of np.savez (smaller, slower to read)")
    ap.add_argument("--write-manifest", action="store_true",
                    help="write manifest.json and exit. Run ONCE before the array job: every task "
                         "writing it would race on the date field.")
    ap.add_argument("--version", default="005-port", help="manifest `version`")
    ap.add_argument("--notes", default="", help="manifest `notes`")
    a = ap.parse_args(argv)

    if a.write_manifest:
        write_manifest(Path(a.out), version=a.version, notes=a.notes)
        print(f"[pred] wrote {Path(a.out) / 'manifest.json'}", flush=True)
        return 0
    for req in ("chrom", "shared", "genome"):
        if not getattr(a, req):
            ap.error(f"--{req} is required unless --write-manifest")

    from candi.bench.external import _expected, track_dirname   # noqa: F401
    from candi.bench.harness import open_source

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True

    regime = load_regime(a.regime)
    cells, cix = cell_index(regime)
    assays, aix = assay_index(regime)

    source = open_source(store=a.regime, chroms=(a.chrom,))
    try:
        expected = _expected(source)
        n_bins = int(source.n_bins(a.chrom))
    finally:
        source.close()
    print(f"[pred {a.chrom}] {len(expected)} declared tracks x {n_bins} bins", flush=True)

    sh = torch.load(a.shared, map_location=dev, weights_only=False)
    gn = torch.load(a.genome, map_location=dev, weights_only=False)
    assert gn["cells"] == cells and gn["assays"] == assays, \
        "checkpoint index space differs from the regime's -- refusing to guess a column mapping"
    assert int(gn["n_bins"]) == n_bins, (gn["n_bins"], n_bins)

    model = Avocado(len(cells), len(assays), n_bins).to(dev)
    state = dict(sh["model"])
    # Genomic factors from this chromosome's run; everything else from the shared run.  They are
    # identical in the genome run, which froze them, but taking them from the shared checkpoint
    # makes the provenance explicit.
    for k, v in gn["model"].items():
        if k.startswith(("g25.", "g250.", "g5k.")):
            state[k] = v
    model.load_state_dict(state, strict=True)
    model.eval()

    t0 = time.time()
    out = write_predictions(model, expected=expected, cix=cix, aix=aix, chrom=a.chrom,
                            n_bins=n_bins, root=Path(a.out), device=dev,
                            batch_positions=a.batch_positions, compress=a.compress)
    print(f"[pred {a.chrom}] wrote {len(expected)} tracks in {time.time() - t0:.0f}s; "
          f"mean={out.mean():.4f} max={out.max():.1f}", flush=True)
    return 0


def write_predictions(model, *, expected, cix, aix, chrom, n_bins, root, device,
                      batch_positions=8192, compress=False):
    """Forward `model` over every bin of `chrom` and write the 4.1 root. Returns the predictions.

    Split out of `main` because the mid-training selection loop in `train.py` needs exactly this and
    a second copy of a `sinh`-and-clip would be a second chance to invert the target differently.
    The model comes in already assembled -- `main` merges a shared and a genome checkpoint, the
    selection loop hands over the model it is training.
    """
    order = sorted(expected)
    ci = torch.tensor([cix[expected[d][0].target_biosample] for d in order],
                      dtype=torch.int64, device=device)
    ai = torch.tensor([aix[expected[d][1]] for d in order], dtype=torch.int64, device=device)

    out = np.empty((len(order), n_bins), dtype=np.float32)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for s in range(0, n_bins, batch_positions):
            e = min(s + batch_positions, n_bins)
            pos = torch.arange(s, e, dtype=torch.int64, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                yh = model(pos, ci, ai)
            y = torch.sinh(yh.float()).clamp_(min=0.0)
            out[:, s:e] = y.T.cpu().numpy()
    model.train(was_training)

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    save = np.savez_compressed if compress else np.savez
    for i, d in enumerate(order):
        td = root / d
        td.mkdir(exist_ok=True)
        tmp = td / f"{chrom}.npz.tmp.npz"
        save(tmp, signal_mu=out[i])
        os.replace(tmp, td / f"{chrom}.npz")
    return out


def write_manifest(root: Path, *, version: str, notes: str) -> None:
    """`manifest.json` -- 4.1's provenance block, copied verbatim into every score file."""
    import datetime
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps({
        "method": "Avocado",
        "version": version,
        "generated_by": "competitors/avocado/predict.py (vendored model "
                        "vendor/avocado.py md5 0eca3ad67d1854ff05ece0a81466173c)",
        "date": datetime.date.today().isoformat(),
        "arms": ["pval"],
        "notes": notes,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
