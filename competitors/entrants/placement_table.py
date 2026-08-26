#!/usr/bin/env python3
"""Assemble the Dataset-3 placement table: 23 entrants + Average + Avocado_p0 + our methods.

The reporting invariants are plan §6.4 (decision B3) and they are enforced here rather than left to
whoever writes the markdown:

* **Per-assay rows first, macro second.** Most methods showed four orders of magnitude higher MSE on
  H3K4me3 than on H3K9me3, purely because the marks differ in dynamic range. A pooled number over
  mixed marks is close to meaningless, so this file will not emit one.
* **Broad and punctate never pooled without their separate medians beside the pool.** Broad is
  H3K27me3, H3K36me3, H3K9me3; everything else here is punctate.
* **DNase-seq excluded from every row.** In Dataset 2 it is read-depth normalized signal and in
  Dataset 3 a -log10 p-value -- a different physical quantity, not a pipeline version difference.
* **The msevar caveat travels with every table**, in the markdown and in the json, because the
  measure is absent rather than zero and an absent measure is easy to read as an oversight.

Aggregation, stated once: within an experiment, the ten bootstraps are averaged to one point
estimate and their spread is kept; within an assay, experiments are combined by **median**, which is
what 001 and 005 both report and what survives the heavy tail the blacklist gotcha leaves on weak
marks. Macro is the mean over assay medians, so each assay weighs equally rather than each
experiment -- the same reweighting argument the P-block rests on.

    python placement_table.py --scores <dir>/<method>/*.csv [...] --out-md table.md --out-json t.json
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

BROAD_MARKS = ["H3K27me3", "H3K36me3", "H3K9me3"]
EXCLUDED_ASSAYS = {"DNase-seq", "DNase"}

MEASURES = ["mse", "gwcorr", "gwspear", "mseprom", "msegene", "mseenh", "mse1obs", "mse1imp"]
# Higher is better for the two correlations; lower for every mse-type measure. Needed to rank.
HIGHER_IS_BETTER = {"gwcorr", "gwspear"}

CAVEATS = [
    "`msevar` is excluded from every row. Experiment 001 could not reconstruct the published "
    "variance vector (no candidate closer than median ratio 0.19 training-only, 0.61 pooled); "
    "eight of the nine challenge measures reproduce and are reported here.",
    "DNase-seq is excluded from every row (decision B3). In Dataset 2 it is `read-depth normalized "
    "signal` and in Dataset 3 a -log10 p-value -- a different physical quantity.",
    "`mseprom`, `msegene` and `mseenh` are evaluated at slightly displaced loci. The official "
    "scorer DELETES blacklisted bins rather than masking them, which shifts every downstream index; "
    "truth and prediction shift identically so the genome-wide measures stay paired, but the "
    "annotation windows read up to 7.35 kb off on 9 of 23 chromosomes. This is what the leaderboard "
    "did and is preserved deliberately.",
    "The P-block columns are ours, not the challenge's, and run on the INTACT grid so their "
    "promoter windows stay in register. Their truth is binarised at `signal >= 2` because the "
    "challenge distributed no peak calls -- they are not comparable to a store-path P-block row "
    "scored against MACS2 calls.",
    "These are Dataset-3 numbers and never enter an internal Dataset-2 table. Experiment 005 "
    "measured 12-66 % per-experiment error for rescaling between the two spaces; do not translate.",
    "The ten bootstraps are ten fixed chromosome subsets, not resamples of genomic positions.",
    "Some entrants are NOT independent. CUImpute1, CUWA and ICU submitted byte-identical tracks "
    "for all 26 broad-mark experiments and differ only on punctate marks, so on H3K27me3, "
    "H3K36me3 and H3K9me3 they are one submission under three names. Any per-assay table flags "
    "such a group beneath it; never count them separately when saying how many methods beat a "
    "given row.",
]

# A method that scored fewer experiments than expected gets this marker on the affected assay row.
# The D2 lesson is that a partial panel must never look like a complete one, and an assay median
# over 7 experiments sitting unmarked beside one over 8 is exactly that failure.
GAP_MARK = " (dagger)"


def load_method(paths: List[str]) -> Dict[str, dict]:
    """Per-experiment records from one method's CSVs, bootstraps collapsed to mean + spread."""
    by_exp: Dict[str, dict] = {}
    raw: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    meta: Dict[str, dict] = {}
    for path in paths:
        with open(path) as fh:
            for r in csv.DictReader(fh):
                exp = r["experiment"]
                assay = r.get("assay_name") or r.get("assay")
                if assay in EXCLUDED_ASSAYS:
                    continue
                meta[exp] = {"assay": assay, "cell": r.get("cell", ""),
                             "mark_class": r.get("mark_class")
                             or ("broad" if assay in BROAD_MARKS else "punctate")}
                for m in MEASURES:
                    if r.get(m) not in (None, "", "nan"):
                        raw[exp][m].append(float(r[m]))
    for exp, per_measure in raw.items():
        rec = dict(meta[exp])
        rec["n_bootstraps"] = max((len(v) for v in per_measure.values()), default=0)
        for m in MEASURES:
            v = np.array(per_measure.get(m, []), dtype=float)
            v = v[np.isfinite(v)]
            rec[m] = float(v.mean()) if v.size else float("nan")
            rec[f"{m}__bootstrap_sd"] = float(v.std(ddof=1)) if v.size > 1 else float("nan")
        by_exp[exp] = rec
    return by_exp


def load_pblock(paths: List[str]) -> Dict[str, dict]:
    """`macro_accuracy` per experiment from the per-experiment json score_entrant.py writes."""
    out: Dict[str, dict] = {}
    for path in paths:
        with open(path) as fh:
            d = json.load(fh)
        pb = d.get("pblock")
        if not pb or d.get("assay") in EXCLUDED_ASSAYS:
            continue
        out[d["experiment"]] = {
            "acc_obs": pb["acc_by_obs_strength"]["macro_accuracy"],
            "acc_imp": pb["acc_by_imp_strength"]["macro_accuracy"],
            "prom_corr": (pb.get("prom_corr_h3k4me3") or {}).get("mean_corr", float("nan")),
            "truth_binarisation": pb.get("truth_binarisation"),
        }
    return out


def per_assay(by_exp: Dict[str, dict], pblock: Dict[str, dict]) -> Dict[str, dict]:
    """Median over experiments, within each assay."""
    groups: Dict[str, List[str]] = defaultdict(list)
    for exp, rec in by_exp.items():
        groups[rec["assay"]].append(exp)

    out: Dict[str, dict] = {}
    for assay, exps in sorted(groups.items()):
        row = {"assay": assay, "n_experiments": len(exps),
               "mark_class": "broad" if assay in BROAD_MARKS else "punctate"}
        for m in MEASURES:
            v = np.array([by_exp[e][m] for e in exps], dtype=float)
            v = v[np.isfinite(v)]
            row[m] = float(np.median(v)) if v.size else float("nan")
        for key in ("acc_obs", "acc_imp", "prom_corr"):
            v = np.array([pblock[e][key] for e in exps if e in pblock], dtype=float)
            v = v[np.isfinite(v)]
            row[key] = float(np.median(v)) if v.size else float("nan")
        out[assay] = row
    return out


def macro(rows: Dict[str, dict], mark_class: Optional[str] = None) -> dict:
    """Mean over assay medians. Each assay weighs equally, not each experiment."""
    sel = [r for r in rows.values() if mark_class is None or r["mark_class"] == mark_class]
    out = {"n_assays": len(sel),
           "assays": sorted(r["assay"] for r in sel)}
    for m in MEASURES + ["acc_obs", "acc_imp"]:
        v = np.array([r[m] for r in sel], dtype=float)
        v = v[np.isfinite(v)]
        out[m] = float(np.mean(v)) if v.size else float("nan")
    return out


def load_expected(expect_path: str, bridge_path: str) -> Dict[str, str]:
    """Expected experiment -> assay, DNase already dropped.

    Without this the assembler can only report how many experiments a method HAPPENED to produce.
    With it, a method that is short can be named as short, and the specific `C##M##` said out loud.
    """
    bridge = {r["filename"]: r for r in csv.DictReader(open(bridge_path))}
    out: Dict[str, str] = {}
    for line in open(expect_path):
        exp = line.strip()
        if not exp:
            continue
        assay = bridge[exp]["assay_name"]
        if assay not in EXCLUDED_ASSAYS:
            out[exp] = assay
    return out


def coverage(by_exp: Dict[str, dict], expected: Dict[str, str]) -> dict:
    """What a method is missing against the expected panel, per assay."""
    missing = sorted(set(expected) - set(by_exp))
    per_assay: Dict[str, List[str]] = defaultdict(list)
    for e in missing:
        per_assay[expected[e]].append(e)
    return {"n_expected": len(expected), "n_scored": len(by_exp),
            "missing": missing, "missing_by_assay": dict(per_assay),
            "complete": not missing}


def duplicate_groups(methods: Dict[str, dict], assay: str) -> List[List[str]]:
    """Methods whose scores for one assay are bit-identical on every measure.

    Not a curiosity. `CUImpute1`, `CUWA` and `ICU` submitted **byte-identical tracks for all 26
    broad-mark experiments** and differ only on punctate marks, so on H3K27me3, H3K36me3 and
    H3K9me3 they are one submission wearing three names. Ranking them as three rows overstates how
    many distinct methods the field holds, and any claim of the form "N methods beat X on broad
    marks" triple-counts a single entry.

    Detection is from the SCORES rather than from track checksums on purpose: this file reads score
    CSVs and nothing else, and a group agreeing to the last bit of eight independently computed
    measures over eight-plus experiments is identical for a reason worth printing either way.
    """
    sig: Dict[tuple, List[str]] = defaultdict(list)
    for name, d in methods.items():
        r = d["per_assay"].get(assay)
        if not r:
            continue
        key = tuple(r[m] for m in MEASURES)
        if all(np.isfinite(v) for v in key):
            sig[key].append(name)
    return [sorted(v) for v in sig.values() if len(v) > 1]


def fmt(x: float) -> str:
    if not np.isfinite(x):
        return "--"
    return f"{x:.4f}" if abs(x) < 1000 else f"{x:.3e}"


def markdown(methods: Dict[str, dict]) -> str:
    """Per-assay blocks first, then the three macro rows. Never a single pooled number alone."""
    L: List[str] = ["# Dataset-3 placement table", "",
                    "Predictions scored directly against the challenge's own blind-test tracks "
                    "(Synapse `syn17083203`) with experiment 001's vendored scorer. "
                    "Aggregation: bootstraps averaged within an experiment, experiments combined "
                    "by median within an assay, assays averaged into macro.", ""]

    assays = sorted({a for m in methods.values() for a in m["per_assay"]},
                    key=lambda a: (a not in BROAD_MARKS, a))
    for assay in assays:
        klass = "broad" if assay in BROAD_MARKS else "punctate"
        L += [f"## {assay} ({klass})", "",
              "| method | n | " + " | ".join(MEASURES) + " | acc_obs | acc_imp |",
              "|---|---:|" + "---:|" * (len(MEASURES) + 2)]
        rank_key = "mse"
        have = [(name, d["per_assay"][assay]) for name, d in methods.items()
                if assay in d["per_assay"]]
        have.sort(key=lambda kv: (kv[1][rank_key] if np.isfinite(kv[1][rank_key]) else np.inf))
        gaps: List[str] = []
        for name, r in have:
            cov = methods[name].get("coverage") or {}
            short = (cov.get("missing_by_assay") or {}).get(assay, [])
            mark = GAP_MARK if short else ""
            if short:
                have_n, want_n = r["n_experiments"], r["n_experiments"] + len(short)
                gaps.append(f"**{name}** is missing {', '.join(short)} — its median here is over "
                            f"{have_n} experiment{'' if have_n == 1 else 's'}, not {want_n}.")
            L.append(f"| {name}{mark} | {r['n_experiments']} | "
                     + " | ".join(fmt(r[m]) for m in MEASURES)
                     + f" | {fmt(r['acc_obs'])} | {fmt(r['acc_imp'])} |")
        L.append("")
        for g in gaps:
            L += [f"(dagger) {g}", ""]
        for grp in duplicate_groups(methods, assay):
            L += [f"**Not independent:** {', '.join(grp)} scored bit-identically on all "
                  f"{len(MEASURES)} measures for {assay} — they submitted the same predictions. "
                  f"Count them once, not {len(grp)} times.", ""]

    for klass, title in (("broad", "Macro -- broad marks (H3K27me3, H3K36me3, H3K9me3)"),
                         ("punctate", "Macro -- punctate marks"),
                         (None, "Macro -- all non-DNase assays pooled")):
        L += [f"## {title}", "",
              "| method | n assays | " + " | ".join(MEASURES) + " | acc_obs | acc_imp |",
              "|---|---:|" + "---:|" * (len(MEASURES) + 2)]
        rows = [(name, macro(d["per_assay"], klass)) for name, d in methods.items()]
        rows = [(n, r) for n, r in rows if r["n_assays"]]
        rows.sort(key=lambda kv: (kv[1]["mse"] if np.isfinite(kv[1]["mse"]) else np.inf))
        for name, r in rows:
            L.append(f"| {name} | {r['n_assays']} | "
                     + " | ".join(fmt(r[m]) for m in MEASURES)
                     + f" | {fmt(r['acc_obs'])} | {fmt(r['acc_imp'])} |")
        L.append("")
        if klass is None:
            L += ["The pooled row is printed **after** the two class rows above and must never be "
                  "quoted without them: broad and punctate marks differ in dynamic range by orders "
                  "of magnitude, so the pool is dominated by whichever class has more assays.", ""]

    covs = {n: d["coverage"] for n, d in methods.items() if d.get("coverage")}
    if covs:
        L += ["## Coverage — which methods scored the full panel", "",
              "| method | scored | expected | missing |", "|---|---:|---:|---|"]
        for name in sorted(covs, key=lambda n: (covs[n]["complete"], n)):
            c = covs[name]
            L.append(f"| {name} | {c['n_scored']} | {c['n_expected']} | "
                     + (", ".join(c["missing"]) if c["missing"] else "—") + " |")
        L += ["", "A method missing an experiment has a smaller median in that assay's row above, "
                  "marked (dagger). It is listed here rather than silently averaged, because an "
                  "assay median over 7 experiments sitting unmarked beside one over 8 is exactly "
                  "the partial-panel failure the D2 lesson is about.", ""]

    L += ["## Caveats -- all of these travel with every number above", ""]
    L += [f"{i}. {c}" for i, c in enumerate(CAVEATS, 1)]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", nargs="+", required=True,
                    help="one or more '<name>=<glob>' pairs, e.g. Guacamole=work/Guacamole/*.csv")
    ap.add_argument("--pblock", nargs="*", default=[],
                    help="'<name>=<glob>' for the matching per-experiment jsons")
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--expect", help="blind_experiments.txt -- the panel every method should cover. "
                                     "Given this, a short method is NAMED as short rather than "
                                     "quietly contributing a smaller median.")
    ap.add_argument("--bridge", help="eic_bridge.csv, needed with --expect to map experiment->assay")
    args = ap.parse_args()

    expected = (load_expected(args.expect, args.bridge)
                if args.expect and args.bridge else None)
    if args.expect and not args.bridge:
        raise SystemExit("--expect needs --bridge to map each experiment to its assay")
    if expected:
        print(f"expected panel: {len(expected)} experiments (DNase excluded)")

    pb_globs = dict(p.split("=", 1) for p in args.pblock)
    methods: Dict[str, dict] = {}
    for spec in args.scores:
        name, pattern = spec.split("=", 1)
        paths = sorted(glob.glob(pattern))
        if not paths:
            print(f"WARNING: {name}: no csv matched {pattern}", file=sys.stderr)
            continue
        by_exp = load_method(paths)
        pblock = load_pblock(sorted(glob.glob(pb_globs[name]))) if name in pb_globs else {}
        cov = coverage(by_exp, expected) if expected else None
        methods[name] = {"per_exp": by_exp, "per_assay": per_assay(by_exp, pblock),
                         "n_experiments": len(by_exp), "n_csv": len(paths), "coverage": cov}
        note = ""
        if cov and cov["missing"]:
            note = f"  SHORT: missing {', '.join(cov['missing'])}"
        print(f"{name}: {len(by_exp)} experiments (DNase excluded), "
              f"{len(pblock)} p-blocks{note}")

    if not methods:
        raise SystemExit("no method produced any row")

    md = markdown(methods)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_md)), exist_ok=True)
    with open(args.out_md, "w") as fh:
        fh.write(md)

    payload = {
        "caveats": CAVEATS,
        "excluded_assays": sorted(EXCLUDED_ASSAYS),
        "broad_marks": BROAD_MARKS,
        "measures": MEASURES,
        "aggregation": ("bootstrap mean within experiment; median over experiments within assay; "
                        "mean over assay medians for macro"),
        "methods": {n: {"n_experiments": d["n_experiments"], "per_assay": d["per_assay"],
                        "coverage": d.get("coverage"),
                        "macro_broad": macro(d["per_assay"], "broad"),
                        "macro_punctate": macro(d["per_assay"], "punctate"),
                        "macro_all": macro(d["per_assay"], None)}
                    for n, d in methods.items()},
    }
    with open(args.out_json, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"-> {args.out_md}\n-> {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
