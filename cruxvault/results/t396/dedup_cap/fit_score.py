"""t396 C10c: fit, score and aggregate the dedup cap test from the sc_dedup statistics files.

A thin driver over `tools/t396/dedup_cap.py` (C10a). It splits the fit and the score into array
tasks, and it aggregates with decision D15: the exact shared-weight law (`dm_distinct_shared`) is
the PRIMARY start-space law; the independent-slot law (`dm_distinct`, K = 25 U), `occupancy` and
`start_cap` are secondary. For a 1-unit pair the shared-weight law is `dm_distinct` with K = 25.

  fit-group  --stats-dir D --rows R --tables T --task i --out-dir O   (one assay per task; the
             last task is the ("all", "all") point fit, no bootstrap)
  merge-fit  --out-dir O --out per_assay.tsv
  score      --stats-dir D --rows R --tables T --fit F --floor-dir FD --task i --group g --out-dir O
  aggregate  --pairs <pairs_*.tsv> --out table.tsv                   (local, cheap)
  totals     --totals <totals_*.tsv> --out totals.tsv                (local, cheap)
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
sys.path.insert(0, os.path.join(REPO, "tools"))
from t396 import caps  # noqa: E402
from t396 import dedup_cap as d  # noqa: E402

N_BOOT_FIT = d.N_BOOT_FIT          # 50, bootstrap over train experiments per fit group
N_BOOT_SCORE = d.N_BOOT_SCORE      # 1000, bootstrap over experiments per table row
N_REP = 20                         # simulate_floor draws per bin (C4 default)
PRIMARY = "dm_distinct_shared"     # D15
OTHER = d.COUNT_LAW                # count_saturation
CLASSES = {"narrow_histone": "narrow", "broad_histone": "broad", "accessibility_dnase": "DNase",
           "accessibility_atac": "ATAC", "control": "control"}
TABLE_COLUMNS = ["law", "space", "layer", "run_type", "assay_class", "in_eic", "scope",
                 "gap_closed_vs_depth_scaling", "gap_closed_vs_identity", "attainable_floor",
                 "bootstrap_spread", "bootstrap_lo", "bootstrap_hi", "n_pairs", "n_experiments",
                 "deviance_law", "deviance_depth_scaling", "deviance_identity", "explains",
                 "beats_other_law"]
PAIR_COLUMNS = ["pair", "experiment", "assay", "assay_class", "layer", "run_type", "in_eic",
                "units", "K", "fit_group", "law", "space", "scope", "D_law", "D_identity",
                "D_depth", "D_floor"]
TOTAL_COLUMNS = ["on_id", "experiment", "assay", "assay_class", "layer", "run_type", "in_eic",
                 "units", "reads_off", "reads_on", "reads_on_over_off", "starts_off", "starts_on",
                 "starts_on_over_off", "counts_off", "counts_on", "counts_on_over_off"]


def assay_classes(tables):
    out = {}
    for r in d.read_tsv(os.path.join(tables, "measurements.tsv")):
        out[r["measurement_id"]] = CLASSES.get(r.get("assay_class", ""), r.get("assay_class", ""))
    return out


def stats_paths(stats_dir, rows):
    return [os.path.join(stats_dir, r["on_id"] + ".npz") for r in rows]


# ------------------------------------------------------------------ fit
def _fit_set(sel):
    a = caps.fit_shape([x["start"] for x in sel], [x["K"] for x in sel])
    k = caps.fit_kappa([x["count"] for x in sel], [x["read_length"] for x in sel])
    sh = d.fit_shape_shared([x["start"] for x in sel], [x["shares"] for x in sel])
    return a, sh, k


def fit_one(its, n_boot, rng):
    """`dedup_cap.fit_groups` for one group: point fits and 2.5-97.5% bootstrap over experiments."""
    a, a_sh, kap = _fit_set(its)
    exps = sorted({x["experiment"] for x in its})
    boots = []
    for _ in range(n_boot if len(exps) > 1 else 0):
        pick = rng.choice(exps, len(exps), replace=True)
        boots.append(_fit_set([x for e in pick for x in its if x["experiment"] == e]))
    b = np.array(boots) if boots else np.full((0, 3), np.nan)
    lh = [d._boot_ci(b[:, j]) if b.size else (float("nan"),) * 2 for j in range(3)]
    return {"n_pairs": len(its), "n_experiments": len(exps), "a": a, "a_lo": lh[0][0],
            "a_hi": lh[0][1], "a_shared": a_sh, "a_shared_lo": lh[1][0], "a_shared_hi": lh[1][1],
            "kappa": kap, "kappa_lo": lh[2][0], "kappa_hi": lh[2][1]}


def cmd_fit_group(a):
    rows = [r for r in d.read_tsv(a.rows) if r["in_store"] == "1"]
    train = [r for r in rows if r["in_eic"] == "train"]
    assays = sorted({r["assay"] for r in train})
    by_on = {r["on_id"]: r for r in rows}
    rng = np.random.default_rng(1000 + a.task)
    out = []
    if a.task < len(assays):
        A = assays[a.task]
        items = d._items(stats_paths(a.stats_dir, [r for r in train if r["assay"] == A]), by_on)
        whole = fit_one(items, a.n_boot, rng)
        out.append({"assay": A, "run_type": "all", **whole})
        for rt in sorted({x["run_type"] for x in items}):
            sub = [x for x in items if x["run_type"] == rt]
            res = whole if len(sub) == len(items) else fit_one(sub, a.n_boot, rng)
            out.append({"assay": A, "run_type": rt, **res})
    elif a.task == len(assays):
        items = d._items(stats_paths(a.stats_dir, train), by_on)
        out.append({"assay": "all", "run_type": "all", **fit_one(items, 0, rng)})
    else:
        sys.exit(f"task {a.task} > {len(assays)} assays")
    d.write_tsv(os.path.join(a.out_dir, f"fit_{a.task:03d}.tsv"), out, d.FIT_COLUMNS)
    print(f"[fit_score] fit task {a.task}: {len(out)} rows")


def cmd_merge_fit(a):
    rows = []
    for p in sorted(glob.glob(os.path.join(a.out_dir, "fit_*.tsv"))):
        rows += d.read_tsv(p)
    rows.sort(key=lambda r: (r["assay"], r["run_type"]))
    d.write_tsv(a.out, rows, d.FIT_COLUMNS)
    print(f"[fit_score] merged {len(rows)} fit rows -> {a.out}")


# ------------------------------------------------------------------ score
def _total(h, col):
    s, t, w = caps._arrays(h)
    return float(((s if col == 0 else t) * w).sum())


def cmd_score(a):
    rows = [r for r in d.read_tsv(a.rows) if r["in_store"] == "1"]
    mine = rows[a.task * a.group:(a.task + 1) * a.group]
    if not mine:
        sys.exit(f"no pairs for task {a.task}")
    by_on = {r["on_id"]: r for r in rows}
    cls = assay_classes(a.tables)
    fit = d.read_tsv(a.fit)
    floors = d.load_floor_counts(sorted(glob.glob(os.path.join(a.floor_dir, "*.npz"))))
    pair_rows, totals = [], []
    for j, it in enumerate(d._items(stats_paths(a.stats_dir, mine), by_on)):
        r = by_on[it["id"]]
        p, src = d.lookup_fit(fit, it["assay"], it["run_type"])
        rng = np.random.default_rng(a.task * 10_000 + j)
        meta = {"pair": it["id"], "experiment": it["experiment"], "assay": it["assay"],
                "assay_class": cls.get(it["id"], ""), "layer": it["layer"],
                "run_type": it["run_type"], "in_eic": it["in_eic"], "units": r["units"],
                "K": it["K"], "fit_group": src}
        for pr in d.score_pair(it, p, floors.get(it["id"]), rng=rng, n_rep=a.n_rep):
            pair_rows.append({**meta, **pr})
        st = it["stats"]
        so, sn = _total(st["start"]["genome"], 0), _total(st["start"]["genome"], 1)
        co, cn = _total(st["count"]["genome"], 0), _total(st["count"]["genome"], 1)
        ro, rn = float(r["reads_off"]), float(r["reads_on"])
        totals.append({"on_id": it["id"], **{k: meta[k] for k in TOTAL_COLUMNS[1:8]},
                       "reads_off": ro, "reads_on": rn, "reads_on_over_off": rn / ro,
                       "starts_off": so, "starts_on": sn, "starts_on_over_off": sn / so,
                       "counts_off": co, "counts_on": cn, "counts_on_over_off": cn / co})
    d.write_tsv(os.path.join(a.out_dir, f"pairs_{a.task:03d}.tsv"), pair_rows, PAIR_COLUMNS)
    d.write_tsv(os.path.join(a.out_dir, f"totals_{a.task:03d}.tsv"), totals, TOTAL_COLUMNS)
    print(f"[fit_score] score task {a.task}: {len(mine)} pairs, {len(pair_rows)} rows")


# ------------------------------------------------------------------ aggregate (local)
NUM = ("D_law", "D_identity", "D_depth", "D_floor")
VIEWS = [  # (columns kept; the others are set to "all")
    ("layer", "run_type"),          # the C10a breakdown: native / uniform x SE / PE / mixed
    ("layer",),                     # native and uniform apart, run types pooled
    ("assay_class",),               # narrow, broad, DNase, ATAC, control; layers pooled
    ("layer", "assay_class"),       # assay class within a layer
    (),                             # everything pooled
]


def cmd_aggregate(a):
    pr = []
    for p in a.pairs:
        for r in d.read_tsv(p):
            for k in NUM:
                r[k] = float(r[k])
            pr.append(r)
    cols = ["layer", "run_type", "assay_class", "in_eic"]
    out, seen = [], set()
    for i, keep in enumerate(VIEWS):
        rows = [{**r, **{c: "all" for c in cols[:3] if c not in keep}} for r in pr]
        for row in d.aggregate(rows, cols, n_boot=a.n_boot, seed=i, primary=PRIMARY, other=OTHER):
            key = tuple(row[c] for c in cols) + (row["scope"], row["law"])
            if key not in seen:
                seen.add(key)
                out.append(row)
    d.write_tsv(a.out, out, TABLE_COLUMNS)
    print(f"[fit_score] aggregate: {len(pr)} pair rows -> {len(out)} table rows")


def cmd_totals(a):
    rows = []
    for p in a.totals:
        rows += d.read_tsv(p)
    rows.sort(key=lambda r: r["on_id"])
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, TOTAL_COLUMNS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fit-group")
    for k in ("--stats-dir", "--rows", "--tables", "--out-dir"):
        p.add_argument(k, required=True)
    p.add_argument("--task", type=int, required=True)
    p.add_argument("--n-boot", type=int, default=N_BOOT_FIT)
    p = sub.add_parser("merge-fit")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("score")
    for k in ("--stats-dir", "--rows", "--tables", "--fit", "--floor-dir", "--out-dir"):
        p.add_argument(k, required=True)
    p.add_argument("--task", type=int, required=True)
    p.add_argument("--group", type=int, required=True)
    p.add_argument("--n-rep", type=int, default=N_REP)
    p = sub.add_parser("aggregate")
    p.add_argument("--pairs", nargs="+", required=True)
    p.add_argument("--n-boot", type=int, default=N_BOOT_SCORE)
    p.add_argument("--out", required=True)
    p = sub.add_parser("totals")
    p.add_argument("--totals", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"fit-group": cmd_fit_group, "merge-fit": cmd_merge_fit, "score": cmd_score,
     "aggregate": cmd_aggregate, "totals": cmd_totals}[a.cmd](a)


if __name__ == "__main__":
    main()
