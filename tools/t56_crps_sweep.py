#!/usr/bin/env python3
"""t56 — measure the sampled NB CRPS against the closed form on REAL t49 prediction tracks.

The unit-level validation (`tests/test_crps_sampled.py`, and the (n, mu, y) grid behind it) shows
the estimator is unbiased per bin. That is necessary and not sufficient: a track number is a mean
over ~1.9 M bins AND an oracle argmin fitted on a 20 000-bin subsample, and the argmin is the part
that can go wrong quietly — a noisy objective overfits its own noise, which biases
`crps_oracle_scaled` DOWN and `scale_error` UP by an amount that depends on k and not on how many
bins the track has. Only a real panel settles it.

    python tools/t56_crps_sweep.py truth   --store configs/regime.eic_val.json --out DIR
    python tools/t56_crps_sweep.py sweep   --store ... --truth DIR --preds ROOT --out DIR \
                                           --shard i --nshards n
    python tools/t56_crps_sweep.py profile --store ... --truth DIR --preds ROOT --method avg

`truth` is split out because it is the only step that needs the store: every declared track's chr21
counts, written once, so the sweep tasks are pure numerics and their wall-clock is the estimator's
and not the reader's.

WHAT IS COMPUTED, AND WHY IT IS NOT `nb_suite`. Exactly the three keys `crps_approx` moves —
`crps`, `crps_oracle_scaled`, `crps_oracle_scaled_and_n` — plus the two derived from them, by
calling the same `distributional` functions `nb_suite` calls with the same arguments. Running the
whole suite would add ece, PIT, coverage and the C-index to every one of the 15 (k, seed) cells,
none of which sampling can touch, and the ratio being measured would stop being the estimator's.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

KS = (10, 25, 50, 100, 250)
SEEDS = (0, 1, 2)


def _expected_tracks(source):
    from candi.bench.external import _expected
    return sorted(_expected(source).items())


def cmd_truth(a) -> int:
    from candi.bench.external import stream_truth
    from candi.bench.harness import open_source

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    source = open_source(store=a.store, chroms=tuple(a.chroms.split(",")))
    try:
        rows = _expected_tracks(source)
        by_pair = {}
        for dirname, (pair, assay) in rows:
            by_pair.setdefault(pair, []).append((assay, dirname))
        for pair, items in by_pair.items():
            cols = [source.assays.index(assay) for assay, _ in items]
            t0 = time.time()
            truth = stream_truth(source, pair, cols, batch_windows=a.batch_windows)
            for (assay, dirname), col in zip(items, cols):
                y = np.concatenate([truth[col][c]["counts"] for c in source.eval_chroms])
                np.save(out / f"{dirname}.npy", y.astype(np.float32))
            print(f"[truth] {pair} {len(items)} track(s) in {time.time() - t0:.0f}s", flush=True)
            del truth
    finally:
        source.close()
    print(f"[truth] wrote {len(list(out.glob('*.npy')))} tracks -> {out}", flush=True)
    return 0


def _crps_family(n, mu, y, *, crps_approx=None, crps_seed=0, seed=0):
    """`nb_suite`'s CRPS keys, and only those, with the wall-clock they cost."""
    from candi.bench import distributional as D
    t0 = time.time()
    crps = float(np.mean(D.crps_eval(crps_approx, crps_seed)(n, D.p_from_mu(n, mu), y)))
    orc = D.oracle_scale(mu, n, y, seed=seed, crps_approx=crps_approx, crps_seed=crps_seed)
    return dict(crps=crps, **orc, scale_error=crps - orc["crps_oracle_scaled"],
                seconds=time.time() - t0)


def cmd_sweep(a) -> int:
    from candi.bench.harness import open_source

    source = open_source(store=a.store, chroms=tuple(a.chroms.split(",")))
    try:
        rows = _expected_tracks(source)
    finally:
        source.close()
    mine = rows[a.shard::a.nshards]
    methods = a.methods.split(",")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    chroms = a.chroms.split(",")
    results = []

    for dirname, (pair, assay) in mine:
        y = np.load(Path(a.truth) / f"{dirname}.npy").astype(np.float64)
        for method in methods:
            tdir = Path(a.preds) / method / dirname
            if not tdir.is_dir():
                print(f"[sweep] skip {method}/{dirname}: no such directory", flush=True)
                continue
            parts_mu, parts_n = [], []
            for c in chroms:
                with np.load(tdir / f"{c}.npz") as z:
                    if "mu" not in z:
                        parts_mu = None
                        break
                    parts_mu.append(np.asarray(z["mu"], dtype=np.float64))
                    parts_n.append(np.asarray(z["n"], dtype=np.float64))
            if parts_mu is None:
                print(f"[sweep] skip {method}/{dirname}: pval-only track", flush=True)
                continue
            mu, n = np.concatenate(parts_mu), np.concatenate(parts_n)
            assert mu.shape == y.shape, (mu.shape, y.shape, dirname)

            rec = dict(track=dirname, assay=assay, method=method, n_bins=int(y.size),
                       exact=_crps_family(n, mu, y), sampled=[])
            print(f"[sweep] {method} {dirname[:60]} exact {rec['exact']['seconds']:.1f}s "
                  f"crps={rec['exact']['crps']:.5f}", flush=True)
            for k in KS:
                for s in SEEDS:
                    got = _crps_family(n, mu, y, crps_approx=k, crps_seed=s)
                    got.update(k=k, crps_seed=s)
                    rec["sampled"].append(got)
                print(f"[sweep]     k={k:4d} {got['seconds']:6.1f}s "
                      f"crps={got['crps']:.5f} d={got['crps'] - rec['exact']['crps']:+.5f}",
                      flush=True)
            results.append(rec)
            json.dump(results, open(out / f"shard{a.shard:03d}.json", "w"), indent=1)
    print(f"[sweep] shard {a.shard}: {len(results)} track x method rows", flush=True)
    return 0


def cmd_profile(a) -> int:
    """Where the count arm's 2.6 h/track actually goes — the denominator of any saving claim."""
    from candi.bench import distributional as D
    from candi.bench.harness import open_source
    from candi.metrics import calibration_pit_curve, nb_crps, nb_crps_sampled

    source = open_source(store=a.store, chroms=tuple(a.chroms.split(",")))
    try:
        dirname = _expected_tracks(source)[a.track_index][0]
    finally:
        source.close()
    y = np.load(Path(a.truth) / f"{dirname}.npy").astype(np.float64)
    parts_mu, parts_n = [], []
    for c in a.chroms.split(","):
        with np.load(Path(a.preds) / a.method / dirname / f"{c}.npz") as z:
            parts_mu.append(np.asarray(z["mu"], dtype=np.float64))
            parts_n.append(np.asarray(z["n"], dtype=np.float64))
    mu, n = np.concatenate(parts_mu), np.concatenate(parts_n)
    p = D.p_from_mu(n, mu)

    def timed(name, fn):
        t0 = time.time()
        fn()
        dt = time.time() - t0
        print(f"[profile] {name:34s} {dt:8.2f}s", flush=True)
        return dt

    print(f"[profile] {a.method} {dirname} — {y.size} bins, "
          f"n at the 1e4 floor: {float((n >= 9999.0).mean()):.3f}", flush=True)
    rec = {"track": dirname, "method": a.method, "n_bins": int(y.size),
           "frac_n_at_floor": float((n >= 9999.0).mean())}
    rec["crps_exact"] = timed("nb_crps (once)", lambda: nb_crps(n, p, y))
    for k in (10, 25, 50, 100):
        rec[f"crps_sampled_k{k}"] = timed(f"nb_crps_sampled k={k}",
                                          lambda k=k: nb_crps_sampled(n, p, y, k=k, seed=0))
    rec["oracle_exact"] = timed("oracle_scale exact", lambda: D.oracle_scale(mu, n, y, seed=0))
    for k in (25, 50):
        rec[f"oracle_sampled_k{k}"] = timed(
            f"oracle_scale k={k}", lambda k=k: D.oracle_scale(mu, n, y, seed=0, crps_approx=k))
    rec["marginal_nb"] = timed("marginal_nb (exact both ways)", lambda: D.marginal_nb(y))
    rec["pit_ece"] = timed("calibration_pit_curve + ece", lambda: calibration_pit_curve(n, p, y))
    rec["coverage_95"] = timed("coverage_nb (nbinom.ppf x2)", lambda: D.coverage_nb(n, mu, y))
    rec["c_index"] = timed("c_index_nb (200k pairs)",
                           lambda: D.c_index_nb(n, mu, y, n_pairs=200_000, seed=0))
    Path(a.out).write_text(json.dumps(rec, indent=1))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("truth", "sweep", "profile"):
        q = sub.add_parser(name)
        q.add_argument("--store", required=True)
        q.add_argument("--chroms", default="chr21")
        q.add_argument("--out", required=True)
        if name == "truth":
            q.add_argument("--batch-windows", type=int, default=4)
        else:
            q.add_argument("--truth", required=True)
            q.add_argument("--preds", required=True)
        if name == "sweep":
            q.add_argument("--shard", type=int, default=0)
            q.add_argument("--nshards", type=int, default=1)
            q.add_argument("--methods", default="avg,knn5,knn1,marginal")
        if name == "profile":
            q.add_argument("--method", default="avg")
            q.add_argument("--track-index", type=int, default=0)
    a = p.parse_args()
    return {"truth": cmd_truth, "sweep": cmd_sweep, "profile": cmd_profile}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
