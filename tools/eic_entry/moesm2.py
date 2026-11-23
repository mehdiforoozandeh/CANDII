#!/usr/bin/env python3
"""MOESM2 board (original 2019 paired-end blind truth, no normalization) for the t390 ladder rungs.

The scoring itself is tools/onion_eval/eic_score.py (`verify-published --published <MOESM2 csv>`
and `score`, with `--truth eic_score.ORIGINAL_TRUTH`). This file holds the two pieces around it:

  varcheck  (Nibi) our msevar weights vs Max's sprint 04 pe weights, per assay, max relative diff
  collect   (Mac)  TSVs of cruxvault/results/t390/moesm2/ from the small files pulled from Nibi:
                   repro_ours.tsv      our rescored entrants vs MOESM2 and vs Max's cached pe scores
                   repro_max.tsv       Max's cached pe scores (25 entrants) vs MOESM2, per measure
                   board_check.tsv     3-experiment leaderboard, our rows in place of MOESM2's
                   scores_all23.tsv    rung scores, chromosome set all23
                   placement.tsv       rung rank among the 25 entrants per measure (all23)
                   placement_combined.tsv  the organizers' combined rank of each rung added alone
                                       to MOESM2's 25 entries, per experiment, over 10 bootstraps
                   resources.tsv       sacct wall time and peak memory

Max's cached pe scores (/project/def-maxwl/maxwl/epi_imputation/s04/scores/pe/<slug>/<exp>.csv)
come from his sprint 04, which reproduced MOESM2 on all 51 experiments (pe_vs_MOESM2).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from onion_eval import eic_score as E  # noqa: E402

EXPS = ["C19M22", "C40M17", "C12M02"]
RUNGS = ["L3", "L5", "L6", "L7"]
SLUG = {tid: re.sub(r"[^A-Za-z0-9_.-]", "_", name) for tid, name, _ in E.TEAMS}


def cmd_varcheck(a):
    out = {}
    for assay in a.assays.split(","):
        ours = np.load(Path(a.ours) / f"var_{assay}.npz")
        mx = np.load(Path(a.max) / f"var_{assay}.npz")
        worst = 0.0
        for c in E.CHROMS:
            x, y = ours[c], mx[c]
            if x.shape != y.shape:
                raise SystemExit(f"{assay} {c}: shape {x.shape} vs {y.shape}")
            worst = max(worst, float(np.max(np.abs(x - y) / np.maximum(np.abs(y), 1e-12))))
        mem_o = json.loads((Path(a.ours) / f"var_{assay}.json").read_text())["members"]
        mem_m = json.loads((Path(a.max) / f"var_{assay}.json").read_text())["members"]
        out[assay] = dict(max_rel_diff=worst, members_ours=mem_o, members_max=mem_m,
                          members_equal=sorted(mem_o) == sorted(mem_m))
        print(assay, out[assay], flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    return 0


def _w(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def _g(x):
    return f"{x:.10g}"


def _max_rows(src, tid, exp):
    """Max's cached pe rows for one entrant and experiment, keyed by chromosome set."""
    out = {}
    with open(Path(src) / "max_pe" / SLUG[tid] / f"{exp}.csv") as fh:
        for r in csv.DictReader(fh):
            out[r["chrom_set"]] = {m: float(r[m]) for m in E.MEASURES9}
    return out


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-12)


def cmd_collect(a):
    src, dst = Path(a.src), Path(a.dst)
    pub = [r for r in E.read_published(a.published) if f"{r['cell']}{r['assay']}" in EXPS]
    pmap = {(r["team_id"], r["cell"] + r["assay"], r["bootstrap_id"]): r for r in pub}

    # 1. Max's cached scores vs MOESM2 (every entrant, 3 experiments, 10 bootstraps)
    rows = []
    for m in E.MEASURES9:
        rel = []
        rank_miss = 0
        for exp in EXPS:
            for b in range(10):
                mine = {tid: _max_rows(src, tid, exp)[f"boot{b}"][m] for tid, _, _ in E.TEAMS}
                theirs = {tid: pmap[(tid, exp, b)][m] for tid, _, _ in E.TEAMS}
                rel += [_rel(mine[t], theirs[t]) for t in mine]
                sgn = -1 if m in E.HIGHER_IS_BETTER else 1
                ro = np.argsort(np.argsort([sgn * mine[t] for t in sorted(mine)], kind="stable"))
                rp = np.argsort(np.argsort([sgn * theirs[t] for t in sorted(theirs)], kind="stable"))
                rank_miss += int(np.sum(ro != rp))
        rel = np.array(rel)
        rows.append([m, len(rel), _g(rel.max()), _g(np.median(rel)), rank_miss])
    _w(dst / "repro_max.tsv", ["measure", "n_rows", "max_rel_diff", "median_rel_diff",
                               "rank_differences"], rows)

    # 2. our rescored entrants vs MOESM2 (boot rows) and vs Max's cache (all 11 sets)
    rows = []
    teams = sorted(int(p.name[5:]) for p in (src / "cache_scores").glob("team_*"))
    for m in E.MEASURES9:
        r_pub, r_max = [], []
        for tid in teams:
            for exp in EXPS:
                mx = _max_rows(src, tid, exp)
                for r in E.read_score_rows(src / "cache_scores" / f"team_{tid}" / f"{exp}.tsv"):
                    s = r["chrom_set"]
                    r_max.append(_rel(r[m], mx[s][m]))
                    if s.startswith("boot"):
                        r_pub.append(_rel(r[m], pmap[(tid, exp, int(s[4:]))][m]))
        rows.append([m, "+".join(E.TEAM_NAME[t] for t in teams), len(r_pub), _g(max(r_pub)),
                     _g(np.median(r_pub)), len(r_max), _g(max(r_max)), _g(np.median(r_max))])
    _w(dst / "repro_ours.tsv", ["measure", "entrants", "n_vs_moesm2", "max_rel_vs_moesm2",
                                "median_rel_vs_moesm2", "n_vs_max_cache", "max_rel_vs_max_cache",
                                "median_rel_vs_max_cache"], rows)

    # 3. the 3-experiment board check written by verify-published
    with open(src / "verify" / "verify_published.tsv") as fh:
        rd = list(csv.DictReader(fh, delimiter="\t"))
    _w(dst / "board_check.tsv", list(rd[0].keys()), [list(r.values()) for r in rd])

    # 4. rung scores and per-measure placement among the entrants (all23, Max's cached values)
    s_rows, p_rows, c_rows = [], [], []
    for rung in RUNGS:
        rr = {r["experiment"]: r for r in E.read_score_rows(src / "scores" / f"{rung}.tsv")
              if r["chrom_set"] == "all23"}
        boots = {}
        for r in E.read_score_rows(src / "scores" / f"{rung}.tsv"):
            if r["chrom_set"].startswith("boot"):
                boots[(r["experiment"], int(r["chrom_set"][4:]))] = r
        for exp in EXPS:
            r = rr[exp]
            s_rows.append([rung, exp] + [_g(r[m]) for m in E.MEASURES9])
            ent = [_max_rows(src, tid, exp)["all23"] for tid, _, _ in E.TEAMS]
            for m in E.MEASURES9:
                ev = np.array([x[m] for x in ent])
                hib = m in E.HIGHER_IS_BETTER
                better = int(np.sum(ev > r[m])) if hib else int(np.sum(ev < r[m]))
                p_rows.append([exp, rung, m, _g(r[m]), better + 1, len(ev),
                               _g(ev.max() if hib else ev.min()), _g(np.median(ev)),
                               _g(ev.min() if hib else ev.max())])
            # the organizers' combined rank within one experiment, rung added alone (26 entries)
            cr = []
            for b in range(10):
                field = [pmap[(tid, exp, b)] for tid, _, _ in E.TEAMS] + [boots[(exp, b)]]
                cr.append(float(E.combined_ranks(field)[-1]))
            c_rows.append([exp, rung, _g(np.mean(cr)), _g(min(cr)), _g(max(cr)), len(E.TEAMS) + 1])
    _w(dst / "scores_all23.tsv", ["rung", "experiment"] + E.MEASURES9, s_rows)
    _w(dst / "placement.tsv", ["experiment", "rung", "measure", "value", "rank_among_entrants",
                               "n_entrants", "best_entrant", "median_entrant", "worst_entrant"],
       p_rows)
    _w(dst / "placement_combined.tsv", ["experiment", "rung", "combined_rank_mean_over_boots",
                                        "combined_rank_min", "combined_rank_max", "of"], c_rows)

    # 5. checks (check-bins, varcheck) and resources
    chk = []
    for line in (src / "logs").glob("*.out"):
        for t in line.read_text().splitlines():
            if t.startswith("[check-bins]") and ("differing" in t or "DIFFERS" in t):
                chk.append(["check-bins", line.name, t])
            if t.startswith("max relative difference") or t.startswith("ranks identical") \
                    or t.startswith("rows complete") or "fast vs reference" in t:
                chk.append(["verify/score", line.name, t])
    vc = json.loads((src / "qc" / "varcheck.json").read_text())
    for k, v in sorted(vc.items()):
        chk.append(["varcheck", k, f"max_rel_diff {v['max_rel_diff']:.3e} members_equal "
                                   f"{v['members_equal']} members {','.join(v['members_ours'])}"])
    _w(dst / "checks.tsv", ["check", "where", "result"], sorted(chk))
    with open(src / "sacct.tsv") as fh:
        lines = [l.rstrip("\n").split("|") for l in fh if l.strip()]
    _w(dst / "resources.tsv", lines[0], lines[1:])
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("varcheck")
    p.add_argument("--ours", required=True); p.add_argument("--max", required=True)
    p.add_argument("--assays", default="M22,M17,M02"); p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_varcheck)
    p = sp.add_parser("collect")
    p.add_argument("--src", required=True); p.add_argument("--dst", required=True)
    p.add_argument("--published", required=True)
    p.set_defaults(fn=cmd_collect)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
