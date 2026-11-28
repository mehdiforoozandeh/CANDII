#!/usr/bin/env python3
"""t390 span_fix: before/after tables for SPAN_EXTRA 24 -> 25 (the store count rule).

    python collect_spanfix.py <src> <dst>

<src> holds the small files pulled from Nibi /scratch/mforooz/eicx/spanfix (made by
`run_entry.sh test` with ENTRY_ROOT=$SF and `run_probe.sh sf` / `sfid`):
  qc/*.json                       compare / arrdiff / peakcheck reports
  scores/placement_{qn,se}_<exp>.tsv
  entry/board_moesm2.tsv, entry/board_moesm4.tsv, entry/check_<exp>.tsv
  sacct.tsv
The "before" values are the committed tables of the earlier t390 folders (dst/..):
  ladder_probe/rung_compare.tsv, ladder_probe/placement.tsv, moesm2/placement.tsv,
  entry_pipeline/board_moesm2.tsv, entry_pipeline/board_moesm4.tsv.
Writes <dst>/{nmse,placement,combined,checks,resources}.tsv.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

EXPS = ("C19M22", "C40M17", "C12M02")
MEASURES = ("mse", "gwcorr", "gwspear", "mseprom", "msegene", "mseenh", "msevar", "mse1obs", "mse1imp")


def _rd(p):
    with open(p) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _w(p, head, rows):
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(head)
        w.writerows(rows)


def _g(x):
    return f"{float(x):.10g}"


def main(src, dst):
    src, dst = Path(src), Path(dst).resolve()
    old = dst.parent

    # 1. nMSE against the exact formula (L5), the pipeline track (L3), and L4 for the QN rung
    before = {(r["experiment"], r["pair"]): r for r in _rd(old / "ladder_probe/rung_compare.tsv")}
    rows = []
    for e in EXPS:
        for pair in ("L6_vs_L5", "L6_vs_L3", "L7_vs_L4", "L6_vs_L6old", "L7_vs_L7old"):
            a = json.loads((src / "qc" / f"{e}_{pair}.json").read_text())
            b = before.get((e, pair))
            rows.append([e, pair, _g(b["nmse"]) if b else "", _g(a["nmse"]),
                         _g(b["nmse_asinh"]) if b else "", _g(a["nmse_asinh"]),
                         _g(b["pearson"]) if b else "", _g(a["pearson"]), _g(a["max_abs_diff"]),
                         a["n_bins"]])
    _w(dst / "nmse.tsv", ["experiment", "pair", "nmse_before", "nmse_after", "nmse_asinh_before",
                          "nmse_asinh_after", "pearson_before", "pearson_after",
                          "max_abs_diff_after", "n_bins"], rows)

    # 2. per-measure value and rank among the 25 entrants: qn, se (L6, L7) and MOESM2 (L6 = raw)
    pb = {}
    for r in _rd(old / "ladder_probe/placement.tsv"):
        pb[(r["truth"], r["experiment"], r["rung"], r["measure"])] = r
    for r in _rd(old / "moesm2/placement.tsv"):
        pb[("moesm2", r["experiment"], r["rung"], r["measure"])] = r
    pa = {}
    for e in EXPS:
        for t in ("qn", "se"):
            for r in _rd(src / "scores" / f"placement_{t}_{e}.tsv"):
                pa[(t, e, r["rung"], r["measure"])] = r
    for r in _rd(src / "entry/board_moesm2.tsv"):
        if r["measure"] in MEASURES:
            pa[("moesm2", r["experiment"], "L6", r["measure"])] = r
    rows = []
    for t, rungs in (("qn", ("L6", "L7")), ("se", ("L6", "L7")), ("moesm2", ("L6",))):
        for e in EXPS:
            for rg in rungs:
                for m in MEASURES:
                    b, a = pb[(t, e, rg, m)], pa[(t, e, rg, m)]
                    rows.append([t, e, rg, m, _g(b["value"]), _g(a["value"]),
                                 b["rank_among_entrants"], a["rank_among_entrants"], a["n_entrants"]])
    _w(dst / "placement.tsv", ["truth", "experiment", "rung", "measure", "value_before", "value_after",
                               "rank_before", "rank_after", "n_entrants"], rows)

    # 3. the organizers' combined rank (entry added alone to the published field, boots 0-9)
    rows = []
    for board, rung in (("moesm2", "L6 (raw)"), ("moesm4", "L7 (qn)")):
        b = {r["experiment"]: r for r in _rd(old / f"entry_pipeline/board_{board}.tsv")
             if r["measure"] == "combined_rank"}
        a = {r["experiment"]: r for r in _rd(src / f"entry/board_{board}.tsv")
             if r["measure"] == "combined_rank"}
        for e in EXPS:
            rows.append([board, e, rung, b[e]["value"], b[e]["note"], a[e]["value"], a[e]["note"],
                         a[e]["n_entrants"]])
    _w(dst / "combined.tsv", ["board", "experiment", "rung", "combined_before", "range_before",
                              "combined_after", "range_after", "of"], rows)

    # 4. checks: identity (constant only), new vs old arrays and peaks, entry-pipeline check rows
    rows = []
    for p in sorted((src / "qc").glob("*_ident_*.json")) + sorted((src / "qc").glob("*_arrdiff.json")):
        a = json.loads(p.read_text())
        if "bins" not in a:                   # a peakcheck report (handled below)
            continue
        rows.append([p.name.split("_")[0], p.stem.split("_", 1)[1], a["bins"], a["bins_differing"],
                     _g(a["max_abs_diff"])])
    for p in sorted((src / "qc").glob("*_peakcheck_*.json")) + sorted((src / "qc").glob("*_ident_peaks_*.json")):
        a = json.loads(p.read_text())
        rows.append([p.name.split("_")[0], p.stem.split("_", 1)[1], a["bases_a"],
                     f"jaccard {a['jaccard']:.6f}", f"bases_b {a['bases_b']}"])
    for e in EXPS:
        for r in _rd(src / "entry" / f"check_{e}.tsv"):
            rows.append([e, "entry check vs OLD rungs: " + r["check"], r["n"], r["result"], r["detail"]])
    for p in sorted((src / "qc").glob("*_entrycheck_vs_newrungs.tsv")):
        for r in _rd(p):
            rows.append([r["experiment"], "entry check vs NEW rungs (ident/): " + r["check"], r["n"],
                         r["result"], r["detail"]])
    _w(dst / "checks.tsv", ["experiment", "check", "n", "result", "detail"], rows)

    # 5. resources (sacct, one row per job)
    rows = [r.split("|") for r in (src / "sacct.tsv").read_text().splitlines() if r.strip()]
    _w(dst / "resources.tsv", rows[0], rows[1:])
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
