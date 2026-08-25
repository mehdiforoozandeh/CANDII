#!/usr/bin/env python3
"""Fit the RIVALS_PLAN.md 6.1 sigma-table for a point-only prediction root.

    sigma^2(assay) = mean over V-pair tracks of (signal_mu - truth)^2,
                     pooled over every bin of the P1 eval chromosomes, in -log10 p space.

Avocado predicts a point, and B1a turns a point into a homoscedastic Gaussian so the pval arm can
carry a forecast distribution.  Two properties this file exists to keep honest:

* **It is fitted on the V panel and reused unchanged on the B panel.**  `fitted_on` records which,
  and the scorer copies that string into provenance.  A CRPS quoted from a B-pair run with a
  V-fitted table is leak-free; one refitted on B is not, and the string is the only thing that
  tells them apart afterwards.
* **V-pair CRPS for this method is IN-SAMPLE for sigma.**  Nothing here can fix that; every table
  showing it has to say so (6.1).

Truth comes from `candi.bench.external.stream_truth` -- the same read the scorer performs -- so the
residual sigma is fitted on is the residual the scorer will measure.

    python fit_sigma.py --regime configs/regime.eic_val.json --pred <root> --out sigma.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regime", required=True)
    ap.add_argument("--pred", required=True, help="prediction root (4.1)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chroms", default=None, help="comma-separated; default the regime's eval_chroms")
    ap.add_argument("--method", default="Avocado")
    ap.add_argument("--batch-windows", type=int, default=4)
    a = ap.parse_args(argv)

    from candi.bench.external import _expected, read_track_arrays
    from candi.bench.harness import open_source

    chroms = tuple(c.strip() for c in a.chroms.split(",")) if a.chroms else None
    source = open_source(store=a.regime, chroms=chroms)
    root = Path(a.pred)
    try:
        from candi.bench.external import stream_truth
        expected = _expected(source)
        use = list(source.eval_chroms)
        n_bins = {c: source.n_bins(c) for c in use}

        by_pair = defaultdict(list)
        for d, (pair, assay) in sorted(expected.items()):
            if (root / d).is_dir():
                by_pair[pair].append((assay, d))
        if not by_pair:
            raise SystemExit(f"{root} covers none of the {len(expected)} declared tracks")

        sse = defaultdict(float)
        n = defaultdict(int)
        for k, (pair, rows) in enumerate(by_pair.items()):
            cols = [source.assays.index(assay) for assay, _ in rows]
            truth = stream_truth(source, pair, cols, batch_windows=a.batch_windows)
            for (assay, d), col in zip(rows, cols):
                pred = read_track_arrays(root / d, use, n_bins)
                for c in use:
                    r = pred[c]["signal_mu"].astype(np.float64) - \
                        np.asarray(truth[col][c]["pval"], dtype=np.float64)
                    sse[assay] += float(np.dot(r, r))
                    n[assay] += r.size
            del truth
            print(f"[sigma] {k + 1}/{len(by_pair)} {pair}", flush=True)
    finally:
        source.close()

    sigma = {a_: float(np.sqrt(sse[a_] / n[a_])) for a_ in sorted(sse) if n[a_] > 0}
    obj = {"method": a.method,
           "fitted_on": f"{Path(a.regime).name} eval_pairs",
           "chroms": list(use),
           "n_bins_per_assay": {k: int(v) for k, v in sorted(n.items())},
           "sigma": sigma}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    for k, v in sorted(sigma.items()):
        print(f"  sigma[{k}] = {v:.4f}  (n={n[k]})")
    print(f"[sigma] wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
