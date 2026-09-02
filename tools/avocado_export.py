#!/usr/bin/env python3
"""CANDI_STORE -> the three files `rivals/avocado/hpc_train.py` reads.

Vendoring Avocado verbatim (`rivals/avocado/PROVENANCE.md`) means every difference between our
Avocado run and Maxwell's is in the data we hand it. This writes that data, and nothing else:

    <out>/<chrom>.npy   (n_bins, n_tracks) float32, raw signal at 25 bp -- the trainer arcsinhs it
    <out>/tracks.txt    one column name per line, in the matrix's column order
    <out>/bridge.csv    filename,cell_id,assay_id

**Rule 1 binds this file.** The matrix carries the regime's TRAINING columns only. A scored
`V_`/`B_` track appearing in it would put the answer in the exam paper, and no downstream check
would catch it -- Avocado would simply score well. So a target cell reaching the column set is a
hard error here, never a filter: silently dropping it would let a mis-declared regime export a
subtly different matrix and look like it worked.

**The one thing that needs explaining: `cell_id`.**

Avocado learns a factor per cell type, and a blind cell needs one or it cannot be predicted at all.
In this corpus the same biological cell type appears twice under different names -- its training
tracks under a `T_` biosample and its held-out tracks under the paired `V_`/`B_` one. Those two
must share a `cell_id`, or the blind cell gets no embedding.

That identity comes from the regime's DECLARED `eval_pairs` (D31), never from the names. D16 says
this store never parses a biosample name, and it applies here with force: `T_K562` and `V_K562`
look like an obvious pair to a reader and are one only because the pairing says so. A regime with
no declared pairing is refused rather than guessed at.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from candi.store.reader import CorpusStore                       # noqa: E402
from candi.store.regime import Regime                            # noqa: E402


def cell_ids(regime: Regime) -> dict:
    """`{biosample: cell_id}` — a paired input and target share one id, from `eval_pairs` alone."""
    if not regime.eval_pairs:
        raise SystemExit(
            "this regime declares no `eval_pairs`, so nothing here can say which training cell a "
            "blind cell is the same cell type as. Declare it with tools/declare_eval_pairs.py "
            "(D31); do not let this script guess from the names (D16).")
    out: dict = {}
    for prompt, target in regime.eval_pairs:
        cid = out.get(prompt) or out.get(target)
        if cid is None:
            cid = f"cell{len(set(out.values())):03d}"
        out[prompt] = cid
        out[target] = cid
    # A training cell no pair mentions is its own cell type. It still gets a factor -- it carries
    # training signal -- it simply has no blind counterpart to share one with.
    for b in sorted(regime.train_biosamples):
        if b not in out:
            out[b] = f"cell{len(set(out.values())):03d}"
    return out


def columns(store: CorpusStore, regime: Regime) -> list:
    """The `(biosample, assay)` columns of the matrix — training cells only, Rule 1."""
    targets = {t for _, t in regime.eval_pairs}
    cols = []
    for bios in regime.train_biosamples:
        if bios in targets:
            raise SystemExit(
                f"{bios} is BOTH a training biosample and an eval_pairs target. Exporting it "
                f"would put a scored track in Avocado's training matrix, which is the one thing "
                f"Rule 1 forbids and the one thing no downstream check can see.")
        have = set(store[bios].assays("pval"))
        for assay in regime.assays:
            if assay in have:
                cols.append((bios, assay))
    if not cols:
        raise SystemExit("no training columns: the regime's assays are in none of its train cells")
    return cols


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regime", required=True, help="a CANDI_STORE regime file (json)")
    ap.add_argument("--out", required=True, help="directory to write into")
    ap.add_argument("--chroms", required=True,
                    help="comma-separated. The SHARED fit takes the regime's training "
                         "chromosome; a `--mode genome` fit takes one chromosome it will predict.")
    a = ap.parse_args(argv)

    regime = Regime.from_file(a.regime)
    store = CorpusStore(regime.store)
    ids = cell_ids(regime)
    cols = columns(store, regime)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    names = [f"{b}|{assay}" for b, assay in cols]
    (out / "tracks.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    with (out / "bridge.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "cell_id", "assay_id"])
        for (bios, assay), name in zip(cols, names):
            w.writerow([name, ids[bios], assay])

    n_bins = store.n_bins()
    for chrom in (c.strip() for c in a.chroms.split(",") if c.strip()):
        if chrom not in n_bins:
            raise SystemExit(f"{chrom} is not in the store ({sorted(n_bins)})")
        nb = int(n_bins[chrom])
        Y = np.empty((nb, len(cols)), dtype=np.float32)
        for j, (bios, assay) in enumerate(cols):
            Y[:, j] = store[bios].pval(chrom, 0, nb, assays=[assay])[:, 0]
        np.save(out / f"{chrom}.npy", Y)
        print(f"[export] {chrom}: {nb} bins x {len(cols)} tracks -> {out / (chrom + '.npy')} "
              f"({Y.nbytes / 2**30:.2f} GiB)", flush=True)

    (out / "export.json").write_text(json.dumps({
        "tool": "tools/avocado_export.py",
        "regime": str(a.regime),
        "regime_sha256": regime.sha256,
        "store": regime.store,
        "layer": "pval",
        "n_tracks": len(cols),
        "n_cells": len(set(ids[b] for b, _ in cols)),
        "chroms": [c.strip() for c in a.chroms.split(",") if c.strip()],
        "rule_1": "training biosamples only; no eval_pairs target is a column",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"[export] {len(cols)} columns over "
          f"{len(set(ids[b] for b, _ in cols))} cell types; wrote tracks.txt, bridge.csv, "
          f"export.json", flush=True)
    return 0


if __name__ == "__main__":                                        # pragma: no cover
    raise SystemExit(main())
