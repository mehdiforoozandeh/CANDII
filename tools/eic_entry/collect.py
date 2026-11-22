#!/usr/bin/env python3
"""Collect the ladder probe's small outputs (pulled from Nibi /scratch/mforooz/eicx/ladder) into
the TSVs behind LADDER_PROBE.md, and print the markdown tables.

    python tools/eic_entry/collect.py <pulled dir> <out dir>

<pulled dir> holds scores/{qn,se}/L*.tsv, qc/*.json, placement_{qn,se}.tsv, sacct.tsv (from
`run_probe.sh sacct`), and rungs/L*/*.json (per-rung job reports).
Writes to <out dir>: scores_all23.tsv, placement.tsv, rung_compare.tsv, checks.tsv, resources.tsv.
"""
import csv
import glob
import json
import os
import sys

MEAS = ["mse", "gwcorr", "gwspear", "mseprom", "msegene", "mseenh", "msevar", "mse1obs", "mse1imp"]
RUNGS = ["L1", "L2", "L3", "L4", "L5", "L6", "L7"]


def rtsv(p):
    with open(p) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def wtsv(p, rows, keys):
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(keys)
        for r in rows:
            w.writerow([r.get(k, "") for k in keys])


def fmt(v):
    v = float(v)
    return f"{v:.4g}"


def main(src, out):
    os.makedirs(out, exist_ok=True)
    sc = []
    for t in ("qn", "se"):
        for r in RUNGS:
            p = f"{src}/scores/{t}/{r}.tsv"
            if os.path.exists(p):
                for x in rtsv(p):
                    if x["chrom_set"] == "all23":
                        sc.append(dict(truth=t, experiment=x["experiment"], rung=r, **{m: x[m] for m in MEAS}))
    wtsv(f"{out}/scores_all23.tsv", sc, ["truth", "experiment", "rung"] + MEAS)
    pl = []
    for t in ("qn", "se"):
        p = f"{src}/placement_{t}.tsv"
        if os.path.exists(p):
            pl += rtsv(p)
    if pl:
        wtsv(f"{out}/placement.tsv", pl, list(pl[0].keys()))
    cmp = []
    for p in sorted(glob.glob(f"{src}/qc/*_vs_*.json")):
        if "peakcheck" in p:
            continue
        j = json.load(open(p))
        cmp.append(dict(experiment=os.path.basename(p).split("_")[0], pair=j["pair"], nmse=j["nmse"],
                        nmse_asinh=j["nmse_asinh"], max_abs_diff=j["max_abs_diff"], pearson=j["pearson"],
                        n_bins=j["n_bins"]))
    wtsv(f"{out}/rung_compare.tsv", cmp, ["experiment", "pair", "nmse", "nmse_asinh", "max_abs_diff", "pearson", "n_bins"])
    chk = []
    for p in sorted(glob.glob(f"{src}/rungs/L2/*.npy.json")):
        j = json.load(open(p))
        chk.append(dict(experiment=os.path.basename(p)[:6], check="L2_vs_qn_truth",
                        value=f"per-base mismatched {j.get('mismatched_bases')} of {j.get('bases')}; "
                              f"max abs per base {j.get('max_abs_diff_per_base')}; "
                              f"max abs 25bp {j.get('max_abs_diff_25bp_vs_bin_bigwig')}"))
    for p in sorted(glob.glob(f"{src}/qc/*peakcheck*.json")):
        j = json.load(open(p))
        chk.append(dict(experiment=os.path.basename(p)[:6], check=os.path.basename(p)[7:-5],
                        value=f"jaccard {j['jaccard']:.4f}; bases ours {j['bases_a']} pipeline {j['bases_b']}"))
    for p in sorted(glob.glob(f"{src}/peaks/*.bed.json")):
        j = json.load(open(p))
        chk.append(dict(experiment=os.path.basename(p)[:6], check="callpeaks_" + os.path.basename(p)[7:-9],
                        value=f"{j['n_peaks']} peaks, {j['bases']} bases"))
    for p in sorted(glob.glob(f"{src}/starts/*.json")):
        j = json.load(open(p))
        chk.append(dict(experiment="", check="starts_" + os.path.basename(p)[:-5],
                        value=f"lines {j['n_lines']}; main {j['n_lines_main']}; mean read length {j['mean_read_length']:.2f}"))
    wtsv(f"{out}/checks.tsv", chk, ["experiment", "check", "value"])
    if os.path.exists(f"{src}/sacct.tsv"):
        rows = rtsv(f"{src}/sacct.tsv")
        wtsv(f"{out}/resources.tsv", rows, list(rows[0].keys()))
    # markdown
    for t in ("qn", "se"):
        for e in sorted({r["experiment"] for r in sc}):
            print(f"\n#### {e}, truth {t}\n")
            print("| rung | " + " | ".join(MEAS) + " |")
            print("|---" * (len(MEAS) + 1) + "|")
            for r in RUNGS:
                x = [s for s in sc if s["truth"] == t and s["experiment"] == e and s["rung"] == r]
                if not x:
                    continue
                rk = {p["measure"]: p["rank_among_entrants"] for p in pl
                      if p["truth"] == t and p["experiment"] == e and p["rung"] == r}
                cells = [f"{fmt(x[0][m])} ({rk.get(m, '?')})" for m in MEAS]
                print(f"| {r} | " + " | ".join(cells) + " |")
            ent = [p for p in pl if p["truth"] == t and p["experiment"] == e and p["rung"] == "L1"]
            if ent:
                b = {p["measure"]: p for p in ent}
                print("| best entrant | " + " | ".join(fmt(b[m]["best_entrant"]) for m in MEAS) + " |")
                print("| median entrant | " + " | ".join(fmt(b[m]["median_entrant"]) for m in MEAS) + " |")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
