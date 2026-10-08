"""Rebuild index.html from deck_src.html by embedding the numbers as DATA.

Inputs: snippet.json (a 10 kb real snippet, H3K27ac chr19:12,732,350-12,742,350, read from the
t112 products on Nibi on 2026-10-01) and cruxvault/results/row2/ (gitignored; present in the main
checkout): results_summary.tsv, checks_*.json, grid.md.

    python cruxvault/presentations/t118-grid/build_deck.py
"""
import collections, csv, json, pathlib

HERE = pathlib.Path(__file__).resolve().parent
R = HERE.parent.parent / "results" / "row2"

snip = json.load(open(HERE / "snippet.json"))
snip.pop("locus")

cells = {}
for r in csv.DictReader(open(R / "results_summary.tsv"), delimiter="\t"):
    if r["kind"] != "trained" or r["variant"] != "all" or r["metric"] not in ("crps_all", "crps_top1"):
        continue
    if r["model"] not in ("real", "xshuf", "nocov"):
        continue
    k = "|".join([r["rung"], r["g_version"], r["space"], r["model"], r["mark_class"], r["metric"]])
    cells[k] = [round(float(r["mean"]), 5), round(float(r["seed_wobble"]), 5)]

refs = {}
cur = sub = met = None
for l in open(R / "grid.md"):
    if l.startswith("## "):
        cur = l[3:].strip()
    elif l.startswith("### "):
        sub = l[4:].strip()
    elif l.startswith("**CRPS"):
        met = l.strip()
    elif l.startswith("| row 1") and "all bins" in (met or "") and "per track" in cur:
        c = l.split("|")
        if len(c) > 7:
            sp = "counts" if cur.endswith("counts") else "pval"
            refs[f"{sp}|{sub}"] = {"ns": float(c[6].split("(")[0]), "qm": float(c[7].split("(")[0])}

tally = collections.defaultdict(lambda: [0, 0])
depth = {}
for d in ["A", "B", "C", "D", "A2", "B2", "C2", "D2"]:
    for x in json.load(open(R / f"checks_{d}.json"))["checks"]:
        if x["check"] == "depthlaw":
            depth.setdefault(d, []).append([x["g_version"], x["mark_class"], round(x["value"], 3),
                                            round(x["seed_wobble"], 3), bool(x["met"])])
        else:
            k = f"{d}|{x['check']}|{x['space']}"
            tally[k][0] += bool(x["met"])
            tally[k][1] += 1

# exploding (pair, seed) records in -log10 p, summed over 3 classes x 2 versions of g (grid.md)
explode = {"row1": [0, 3, 1, 10], "row2": [13, 16, 24, 13], "twin": [0, 1, 1, 4]}

data = {"snip": snip, "cells": cells, "refs": refs, "tally": dict(tally), "depth": depth, "explode": explode}
src = open(HERE / "deck_src.html").read()
open(HERE / "index.html", "w").write(src.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
print(f"wrote {HERE / 'index.html'}: {len(cells)} cells")
