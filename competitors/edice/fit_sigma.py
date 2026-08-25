"""The σ-table for a point-only rival (RIVALS_PLAN §6.1, B1a).

    sigma[assay]^2 = mean over V-pair tracks of (signal_mu - truth)^2,
                     pooled over every bin of the P1 eval chromosomes, in -log10 p space

    python fit_sigma.py --regime configs/regime.eic_val.json --pred preds/edice_p1 \
        --out preds/edice_sigma.json

Three properties this file must have, and does:

* It is fitted on **V pairs only**. The B-pair run reuses this table unchanged, which is what makes
  the B-pair CRPS leak-free -- and correspondingly makes the V-pair CRPS IN-SAMPLE for σ. The
  `fitted_on` string is how a reader tells which of the two they are looking at, so it is written
  from the regime, never typed by hand.
* One σ per assay, shared across pairs -- a homoscedastic Gaussian per method x assay. Not per
  track, not per bin: the point of B1a is to give a point-only rival the cheapest honest forecast
  distribution, not to fit it a variance model nobody else got.
* An assay with no residuals gets no entry. `candi.bench.external` refuses a non-positive σ, and an
  absent assay simply carries no gauss_suite -- absent keys, never NaN.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

METHOD = "eDICE (PyTorch reimplementation)"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--regime", required=True, help="the V-pair regime; B pairs never fit a table")
    p.add_argument("--pred", required=True, help="the §4.1 prediction root to fit on")
    p.add_argument("--out", required=True)
    p.add_argument("--chroms", nargs="+", default=None, help="default: the regime's eval_chroms")
    return p


def main(argv=None) -> int:
    from candi.store.reader import CorpusStore

    args = build_parser().parse_args(argv)
    regime = json.loads(Path(args.regime).read_text())
    store = CorpusStore(regime["store"])
    chroms = args.chroms or list(regime["eval_chroms"])
    root = Path(args.pred)

    eval_names = set(regime["biosamples"].get("eval", []))
    if any(n.startswith("B_") for n in eval_names):
        raise SystemExit(
            f"{args.regime} declares B_ biosamples. §6.1 fits σ on V-pair residuals and the B-pair "
            f"run reuses that table unchanged; fitting on B pairs is the leak the rule exists to "
            f"prevent.")

    sse: dict = {}
    n: dict = {}
    n_tracks = 0
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        parts = d.name.split("__")
        if len(parts) != 3:
            continue
        _input_bs, target_bs, assay = parts
        for chrom in chroms:
            npz = d / f"{chrom}.npz"
            if not npz.exists():
                raise SystemExit(f"{d.name}: no {chrom}.npz; the fit must cover the P1 chromosomes")
            with np.load(npz) as z:
                mu = np.asarray(z["signal_mu"], dtype=np.float64)
            truth = np.asarray(store[target_bs][assay].pval(chrom), dtype=np.float64)
            if truth.shape != mu.shape:
                raise SystemExit(
                    f"{d.name}/{chrom}: prediction {mu.shape} vs truth {truth.shape} -- the npz is "
                    f"not on the store's absolute 25 bp grid")
            r = mu - truth
            sse[assay] = sse.get(assay, 0.0) + float(np.dot(r, r))
            n[assay] = n.get(assay, 0) + r.size
        n_tracks += 1

    sigma = {a: float(np.sqrt(sse[a] / n[a])) for a in sorted(sse) if n[a] and sse[a] > 0.0}
    dropped = sorted(set(sse) - set(sigma))

    table = {
        "method": METHOD,
        "fitted_on": f"{Path(args.regime).name} eval_pairs, chroms {chroms}",
        "sigma": sigma,
        "n_tracks": n_tracks,
        "n_bins_per_assay": {a: int(n[a]) for a in sorted(sigma)},
        "dropped_assays": dropped,
        "note": ("§6.1 B1a. Homoscedastic Gaussian per method x assay, pooled over all bins of the "
                 "P1 eval chromosomes in -log10 p space. The B-pair run reuses this table "
                 "unchanged; V-pair CRPS is therefore IN-SAMPLE for σ and every table that shows "
                 "it must say so."),
    }
    Path(args.out).write_text(json.dumps(table, indent=2))
    print(f"fitted σ for {len(sigma)} assays over {n_tracks} tracks -> {args.out}")
    if dropped:
        print(f"no σ for {dropped} (no residuals); those assays carry no gauss_suite")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
