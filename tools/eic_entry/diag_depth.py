#!/usr/bin/env python3
"""Does the challenge truth look like fewer reads than the store holds? (crux task t390, diagnosis)

For one track, the experiment's own observed store counts go through the mean-mode conversion
(diag_scale.signal) as if only a fraction f of the reads were kept: counts * f and depth_reads * f
(so lambda_bg and the treatment-to-control ratio scale with it; this is read thinning, in
expectation). Per f, the error split against the truth (errdecomp.calibrate) on a chromosome subset.
The best f per track, fitted on TRAINING tracks, gives a depth rule that may be applied to
validation tracks; the validation tracks' own best f is reported only as a check.
-> <out>/depth/<track>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diag_scale as D  # noqa: E402

S, LP, E = D.S, D.LP, D.E


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cov", required=True)
    p.add_argument("--track", required=True)
    p.add_argument("--truth", required=True, help="folder of <track>.bigwig")
    p.add_argument("--out", required=True)
    p.add_argument("--chroms", default="chr9,chr21")
    p.add_argument("--fracs", default="1,0.7,0.5,0.35,0.25,0.18,0.125")
    p.add_argument("--eps", type=float, default=1e-6)
    a = p.parse_args(argv)
    import h5py

    cov = S.read_cov(a.cov, a.track)
    route, chroms = cov["route"], a.chroms.split(",")
    sizes = LP.chrom_sizes(S.SIZES)
    bl = E.load_blacklist_bins(E.asset_dir())
    truth = E.load_prediction(f"{a.truth}/{a.track}.bigwig", chroms, bl)
    obs, ctl = {}, {}
    with h5py.File(cov["observed_counts"], "r") as g:
        for ch in chroms:
            o = g[ch][:].astype(np.float64)
            n = LP.ngrid(sizes[ch])
            obs[ch] = np.zeros(n)
            obs[ch][: min(len(o), n)] = o[:n]
            if route == "chip":
                c = S._control(cov["control"], ch)[0]
                ctl[ch] = np.zeros(n)
                ctl[ch][: min(len(c), n)] = c[:n]
    res = []
    for f in [float(v) for v in a.fracs.split(",")]:
        c = dict(cov, depth_reads=float(cov["depth_reads"]) * f)
        tr = {}
        for ch in chroms:
            ob = obs[ch] * f + a.eps
            tr[ch] = E.apply_blacklist(D.signal(ob, ob, route, ctl.get(ch), c).astype(np.float32), bl.get(ch))
        P, T = E.align(tr, truth, chroms)
        x = np.concatenate([T[ch] for ch in chroms]).astype(np.float64)
        y = np.concatenate([P[ch] for ch in chroms]).astype(np.float64)
        t = D.terms(x, y)
        res.append(dict(frac=f, nmse=t["mse"], scale=t["scale"], slope_ratio=t["slope_ratio"], r=t["r"]))
    best = min(res, key=lambda r: r["nmse"])
    rep = dict(track=a.track, route=route, assay_class=cov["assay_class"], depth_reads=float(cov["depth_reads"]),
               read_length=float(cov["read_length"]), chroms=chroms, by_frac=res, best_frac=best["frac"],
               best_depth=best["frac"] * float(cov["depth_reads"]))
    dst = Path(a.out) / "depth" / f"{a.track}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(rep, indent=1))
    LP.log(f"depth {a.track}: best frac {best['frac']} (nmse {best['nmse']:.3f}; at 1: {res[0]['nmse']:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
