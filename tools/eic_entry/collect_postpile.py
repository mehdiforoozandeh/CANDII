#!/usr/bin/env python3
"""Collect the t390 post-prediction pileup probe (run_probe.sh pp) into one TSV.

    python collect_postpile.py <pulled ladder dir> <out.tsv>

Reads <dir>/qc/<exp>_{ppA,ppB,ppC,ppD,L6,L6f}_vs_{L5,L6f}.json (normalized MSE pairs from
`ladder_probe.py compare`), <exp>_pp_exact_vs_L6f_arrdiff.json (the conversion identity check) and
<dir>/postpile/<exp>/<V>/<exp>.npy.json (divisor, kappa, seconds). One row per (experiment, row, ref).
"""
import csv
import json
import sys
from pathlib import Path

EXPS = ["C19M22", "C40M17", "C12M02"]
ROWS = ["L6", "ppA", "ppD", "ppB", "ppC", "L6f"]


def main(src, out):
    src = Path(src)
    rows = []
    for e in EXPS:
        for r in ROWS:
            for ref in ("L5", "L6f"):
                if r == ref:
                    continue
                f = src / "qc" / f"{e}_{r}_vs_{ref}.json"
                if not f.exists():
                    continue
                j = json.load(open(f))
                meta = {}
                if r.startswith("pp"):
                    m = src / "postpile" / e / r[2:] / f"{e}.npy.json"
                    if m.exists():
                        meta = json.load(open(m))
                rows.append(dict(experiment=e, row=r, ref=ref, nmse=j["nmse"], pearson=j["pearson"],
                                 max_abs_diff=j["max_abs_diff"], nmse_asinh=j["nmse_asinh"],
                                 n_bins=j["n_bins"], divisor=meta.get("divisor", ""),
                                 kappa_trt=meta.get("kappa_trt", ""), read_length=meta.get("read_length", ""),
                                 route_seconds=meta.get("seconds", ""), note=""))
        f = src / "qc" / f"{e}_pp_exact_vs_L6f_arrdiff.json"
        if f.exists():
            j = json.load(open(f))
            rows.append(dict(experiment=e, row="exact_identity", ref="L6f", nmse="", pearson="",
                             max_abs_diff=j["max_abs_diff"], nmse_asinh="", n_bins=j["bins"], divisor="",
                             kappa_trt="", read_length="", route_seconds="",
                             note=f"bins_differing={j['bins_differing']} (raw grid, all bins)"))
    cols = list(rows[0].keys())
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, cols, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"wrote {out}: {len(rows)} rows")


if __name__ == "__main__":
    main(*sys.argv[1:3])
