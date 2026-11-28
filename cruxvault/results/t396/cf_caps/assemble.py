"""t396 it2 B5: assemble the controlled-pair cap tables from the pulled Fir outputs (local, cheap).

Reads fir/<tool>/<scenario>/{table,pairs,per_assay}.tsv (tool dedup | sepe; scenario `fitted` or
`rho<x>`, see fir_job.sh) and writes, in this folder:
  table.tsv          the `fitted` tables of both tools (the D25 rates), one column set
  per_assay.tsv      the `fitted` fits of both tools (rates, leave-one-track-out a and kappa)
  per_track.tsv      one row per pair, shape, law, space and scope of the `fitted` scenario, with
                     the per-pair gap closed vs depth scaling and two flags
  q_sensitivity.tsv  every table row of every scenario, keyed by q_scenario and rho
"""
from __future__ import annotations

import csv
import glob
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = ("dedup", "sepe")


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, cols, delimiter="\t", extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def gap(d_law, d_depth, d_floor):
    den = d_depth - d_floor
    return (d_depth - d_law) / den if den > 0 else float("nan")


def fmt(v):
    return f"{v:.6g}" if isinstance(v, float) else v


def main():
    fir = os.path.join(HERE, "fir")
    table, fits, pairs, sens = [], [], [], []
    for tool in TOOLS:
        base = os.path.join(fir, tool, "fitted")
        table += read(os.path.join(base, "table.tsv"))
        fits += read(os.path.join(base, "per_assay.tsv"))
        pairs += read(os.path.join(base, "pairs.tsv"))
        for d in sorted(glob.glob(os.path.join(fir, tool, "*", "table.tsv"))):
            scen = os.path.basename(os.path.dirname(d))
            rho = "fitted" if scen == "fitted" else scen[3:]
            sens += [dict(r, q_scenario=scen, rho=rho) for r in read(d)]
    tcols = list(read(os.path.join(fir, TOOLS[0], "fitted", "table.tsv"))[0].keys())
    write(os.path.join(HERE, "table.tsv"), table, tcols)
    write(os.path.join(HERE, "per_assay.tsv"), fits, list(fits[0].keys()))
    out = []
    for r in pairs:
        dl, dd, df = (float(r[k]) for k in ("D_law", "D_depth", "D_floor"))
        g = gap(dl, dd, df) if math.isfinite(df) else float("nan")
        out.append(dict(r, gap_closed_vs_depth_scaling=fmt(g),
                        law_below_depth_scaling="yes" if dl < dd else "no",
                        floor_above_depth_scaling="yes" if df >= dd else "no"))
    pcols = list(pairs[0].keys()) + ["gap_closed_vs_depth_scaling", "law_below_depth_scaling",
                                     "floor_above_depth_scaling"]
    write(os.path.join(HERE, "per_track.tsv"), out, pcols)
    write(os.path.join(HERE, "q_sensitivity.tsv"), sens, ["q_scenario", "rho"] + tcols)
    print(f"cf_caps: {len(table)} table rows, {len(out)} per-track rows, "
          f"{len(sens)} sensitivity rows", file=sys.stderr)


if __name__ == "__main__":
    main()
