#!/usr/bin/env python3
"""Predict all 51 blind-test experiments on one chromosome.  Runs on Fir, one GPU.

Takes the shared parameters (cell-type factors, assay factors, network) from the
chr20 `shared` run and this chromosome's genomic factors from its `genome` run,
and evaluates the network at every 25 bp bin for each of the 51 blind-test
(cell type, assay) pairs.

Every blind-test cell type has train/validation experiments of *other* assays,
which is where its cell-type embedding comes from; every blind-test assay has
training experiments in *other* cell types, which is where its assay embedding
comes from.  The pair itself was never observed -- that is the imputation task.

The model's output is on the arcsinh scale, so predictions are inverted with
sinh and clipped at zero (the signal is a -log10 p-value and cannot be negative;
without the clip a handful of bins go slightly below it).

Writes <out>/<dataset>/<chrom>.npy, shape (51, n_bins) float32, rows in the
order of the blind-experiment list.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from avocado import Avocado      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["d2", "d3"])
    ap.add_argument("--chrom", required=True)
    ap.add_argument("--shared", required=True, help="chr20 shared-mode checkpoint")
    ap.add_argument("--genome", required=True, help="this chromosome's genome-mode checkpoint")
    ap.add_argument("--bridge", required=True)
    ap.add_argument("--blind-list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-positions", type=int, default=8192)
    args = ap.parse_args()

    odir = os.path.join(args.out, args.dataset)
    os.makedirs(odir, exist_ok=True)
    opath = os.path.join(odir, f"{args.chrom}.npy")
    if os.path.exists(opath):
        print(f"[pred {args.dataset}/{args.chrom}] exists, skipping", flush=True)
        return 0

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True

    sh = torch.load(args.shared, map_location=dev, weights_only=False)
    gn = torch.load(args.genome, map_location=dev, weights_only=False)
    cells, assays = sh["cells"], sh["assays"]
    assert gn["cells"] == cells and gn["assays"] == assays
    n_bins = gn["n_bins"]

    model = Avocado(len(cells), len(assays), n_bins).to(dev)
    state = dict(sh["model"])
    # genomic factors come from this chromosome's run, everything else from the
    # shared run (they are identical in the genome run, which froze them, but
    # taking them from the shared checkpoint makes the provenance explicit)
    for k, v in gn["model"].items():
        if k.startswith(("g25.", "g250.", "g5k.")):
            state[k] = v
    model.load_state_dict(state, strict=True)
    model.eval()

    rows = {r["filename"]: r for r in csv.DictReader(open(args.bridge))}
    blind = [l.strip() for l in open(args.blind_list) if l.strip()]
    cell_ix = {c: i for i, c in enumerate(cells)}
    assay_ix = {a: i for i, a in enumerate(assays)}
    ci = torch.tensor([cell_ix[rows[b]["cell_id"]] for b in blind],
                      dtype=torch.int64, device=dev)
    ai = torch.tensor([assay_ix[rows[b]["assay_id"]] for b in blind],
                      dtype=torch.int64, device=dev)

    print(f"[pred {args.dataset}/{args.chrom}] {len(blind)} experiments x "
          f"{n_bins} bins", flush=True)
    out = np.empty((len(blind), n_bins), dtype=np.float32)
    B = args.batch_positions
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, n_bins, B):
            e = min(s + B, n_bins)
            pos = torch.arange(s, e, dtype=torch.int64, device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=(dev.type == "cuda")):
                yh = model(pos, ci, ai)
            y = torch.sinh(yh.float()).clamp_(min=0.0)
            out[:, s:e] = y.T.cpu().numpy()
    tmp = opath + ".tmp.npy"
    np.save(tmp, out)
    os.replace(tmp, opath)
    with open(os.path.join(odir, "experiments.txt"), "w") as fh:
        fh.write("\n".join(blind) + "\n")
    print(f"[pred {args.dataset}/{args.chrom}] wrote {opath} "
          f"({time.time()-t0:.0f}s); mean={out.mean():.4f} max={out.max():.1f}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
