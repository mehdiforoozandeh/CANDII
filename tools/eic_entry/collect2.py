#!/usr/bin/env python3
"""Collect probe 2's small outputs (pulled from Nibi /scratch/mforooz/eicx/ladder) into the TSVs
behind cruxvault/results/t390/ladder_probe2/PROBE2.md, and print the markdown tables.

    python tools/eic_entry/collect2.py <pulled dir> <probe-1 folder> <out dir>

<pulled dir> holds scores2/{qn,se}/<exp>_<rung>.tsv, scores2/placement_{qn,se}_<exp>.tsv, qc/*.json,
peaks/*.json and sacct2.tsv (`run_probe.sh sacct <ids>` output). <probe-1 folder> is
cruxvault/results/t390/ladder_probe (its scores_all23.tsv, placement.tsv, rung_compare.tsv give the
probe-1 rungs printed beside the new ones). Writes to <out dir>: scores_all23.tsv, placement.tsv,
rung_compare.tsv, checks.tsv, peaks.tsv, resources.tsv.
"""
import csv
import glob
import json
import os
import sys

MEAS = ["mse", "gwcorr", "gwspear", "mseprom", "msegene", "mseenh", "msevar", "mse1obs", "mse1imp"]
NEW = ["L7t", "L7o", "L6f", "L7f"]
OLD = ["L2", "L4", "L4o", "L5", "L6", "L7"]
EXPS = ["C19M22", "C40M17", "C12M02"]


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
    return f"{float(v):.4g}"


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else {}


def main(src, p1, out):
    os.makedirs(out, exist_ok=True)
    sc = []
    for t in ("qn", "se"):
        for p in sorted(glob.glob(f"{src}/scores2/{t}/*.tsv")):
            rung = os.path.basename(p)[:-4].split("_", 1)[1]
            for x in rtsv(p):
                if x["chrom_set"] == "all23":
                    sc.append(dict(truth=t, experiment=x["experiment"], rung=rung, **{m: x[m] for m in MEAS}))
    wtsv(f"{out}/scores_all23.tsv", sc, ["truth", "experiment", "rung"] + MEAS)
    pl = []
    for p in sorted(glob.glob(f"{src}/scores2/placement_*_*.tsv")):
        pl += rtsv(p)
    if pl:
        wtsv(f"{out}/placement.tsv", pl, list(pl[0].keys()))
    cmp = []
    for e in EXPS:
        for p in sorted(glob.glob(f"{src}/qc/{e}_L*_vs_L*.json")):
            j = json.load(open(p))
            if "pair" not in j or j["pair"].split("_vs_")[0] not in NEW:
                continue
            cmp.append(dict(experiment=e, pair=j["pair"], nmse=j["nmse"], nmse_asinh=j["nmse_asinh"],
                            max_abs_diff=j["max_abs_diff"], pearson=j["pearson"], n_bins=j["n_bins"]))
    wtsv(f"{out}/rung_compare.tsv", cmp, ["experiment", "pair", "nmse", "nmse_asinh", "max_abs_diff", "pearson", "n_bins"])
    chk, pk = [], []
    for e in EXPS:
        a = jl(f"{src}/qc/{e}_L6rerun_arrdiff.json")
        if a:
            chk.append(dict(experiment=e, check="L6_rerun_vs_probe1_L6",
                            value=f"max_abs_diff {a['max_abs_diff']}; bins differing {a['bins_differing']} of {a['bins']}"))
        a = jl(f"{src}/qc/{e}_peakcheck_L6p2_vs_L6.json")
        if a:
            chk.append(dict(experiment=e, check="L6_rerun_peaks_vs_probe1_L6_peaks", value=f"jaccard {a['jaccard']}"))
        c = jl(f"{src}/qc/{e}_container_overlap.json")
        if c:
            for k in ("L7o", "L7f"):
                chk.append(dict(experiment=e, check=f"overlap_port_vs_pipeline_script_{k}",
                                value=f"identical {c[k]['identical']}; n {c[k]['port_n']} vs {c[k]['container_n']}"))
            q = c["pipeline_inputs"]
            chk.append(dict(experiment=e, check="overlap_port_vs_pipeline_script_on_pipeline_bfilt_peaks",
                            value=f"identical {q['container_eq_port']}; n {q['port_n']} vs {q['container_n']}"))
            chk.append(dict(experiment=e, check="pipeline_script_on_bfilt_inputs_vs_pipeline_overlap_file",
                            value=f"identical {q['container_eq_pipeline_overlap']}; n {q['container_n']} vs {q['pipeline_overlap_n']}"))
        new = [s for s in sc if s["experiment"] == e and s["truth"] == "qn" and s["rung"] == "L7chk"]
        old = [r for r in rtsv(f"{p1}/scores_all23.tsv") if r["experiment"] == e and r["truth"] == "qn" and r["rung"] == "L7"]
        if new and old:
            d = max(abs(float(new[0][m]) - float(old[0][m])) / max(abs(float(old[0][m])), 1e-12) for m in MEAS)
            chk.append(dict(experiment=e, check="rescore_L7_qn_vs_probe1_scores_qn_L7",
                            value=f"max relative diff over 9 measures {d:.3g}"))
        for nm in ("port_vs_overlap", "L7o_vs_overlap", "L7f_vs_overlap", "L6f_vs_pipeline", "L6p2_vs_L6"):
            j = jl(f"{src}/qc/{e}_peakcheck_{nm}.json")
            if j:
                pk.append(dict(experiment=e, pair=nm, jaccard=j["jaccard"], bases_a=j["bases_a"], bases_b=j["bases_b"],
                               frac_a_in_b=j["frac_a_in_b"], frac_b_in_a=j["frac_b_in_a"]))
        for nm in ("L7o", "L7f", "port_on_pipeline"):
            j = jl(f"{src}/peaks/{e}_{nm}.bed.json")
            if j:
                pk.append(dict(experiment=e, pair=f"{nm}_counts", n_pr1=j["n_pr1"], n_pr2=j["n_pr2"],
                               n_pooled=j["n_pooled"], n_overlap=j["n_overlap"], bases_a=j["bases"]))
    wtsv(f"{out}/checks.tsv", chk, ["experiment", "check", "value"])
    wtsv(f"{out}/peaks.tsv", pk, ["experiment", "pair", "jaccard", "bases_a", "bases_b", "frac_a_in_b", "frac_b_in_a",
                                   "n_pr1", "n_pr2", "n_pooled", "n_overlap"])
    rows = []
    sp = f"{src}/sacct2.tsv"
    if os.path.exists(sp):
        lines = [l.rstrip("\n").split("|") for l in open(sp) if "|" in l]
        hdr = None
        mem = {}
        for f in lines:
            if f[0] == "JobID" and "JobName" in f:
                hdr = f
                continue
            if f[0] == "JobID":
                continue
            if hdr and len(f) == len(hdr):
                rows.append(dict(zip(hdr, f)))
            elif len(f) == 2 and f[0].endswith(".batch"):
                mem[f[0].split(".")[0]] = f[1]
        for r in rows:
            r["MaxRSS"] = mem.get(r["JobID"], "")
    if rows:
        wtsv(f"{out}/resources.tsv", rows, list(rows[0].keys()))
    # markdown
    old_sc = rtsv(f"{p1}/scores_all23.tsv")
    old_pl = rtsv(f"{p1}/placement.tsv")
    for t in ("qn", "se"):
        for e in EXPS:
            print(f"\n### {e}, truth {t}\n")
            print("| rung | " + " | ".join(MEAS) + " |")
            print("|---" * (len(MEAS) + 1) + "|")
            for r in OLD + NEW:
                src_sc, src_pl = (sc, pl) if r in NEW else (old_sc, old_pl)
                x = [s for s in src_sc if s["truth"] == t and s["experiment"] == e and s["rung"] == r]
                if not x:
                    continue
                rk = {p["measure"]: p["rank_among_entrants"] for p in src_pl
                      if p["truth"] == t and p["experiment"] == e and p["rung"] == r}
                lab = f"**{r}**" if r in NEW else r
                print(f"| {lab} | " + " | ".join(f"{fmt(x[0][m])} ({rk.get(m, '?')})" for m in MEAS) + " |")
            ent = {p["measure"]: p for p in pl if p["truth"] == t and p["experiment"] == e and p["rung"] == "L7t"}
            if ent:
                print("| best entrant | " + " | ".join(fmt(ent[m]["best_entrant"]) for m in MEAS) + " |")
                print("| median entrant | " + " | ".join(fmt(ent[m]["median_entrant"]) for m in MEAS) + " |")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
