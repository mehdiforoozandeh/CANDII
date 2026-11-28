"""Cap test of the SE views at 25 bp: the PE parent thinned to one mate and capped per position
(t396 C10a; R44, R29, R30).

A pair is a `*.se` view (the target) and its PE or mixed `parent_measurement_id` (the base view,
the source), kept only when the parent is in the read-start store (21 of the 42 SE views; the
others' parents are in `missing.tsv`). How an SE view is made (`tools/onion_data/fastq_uniform.py`,
`fastq_dnase.py`, arm `se`): mate 1 of every read pair (plus the single-end runs of a mixed
experiment) is realigned as single-end by the experiment's own pipeline, which then deduplicates
each replicate by the single-end rule (5' position and strand; DNase: the 0x400 flag removed in the
tagAlign step). So per strand the law is `caps.thinned_dm_distinct(c_PE, q = 0.5, a, K)` with
`K = caps.K_of(units of the SE view)` under the replicate unit rule. On a mixed parent q = 0.5 is
wrong for the single-end replicates (their reads are all kept); mixed pairs are reported apart.

Subcommands
  rows        --tables <dir> --rs-store <dir> [--count-store <dir>] --out rows.tsv
  stats       --pair <parent id> <se id> --unit-rule UNIT_RULE.json --out <npz>
  fit         --stats <npz...> --rows rows.tsv [--dedup-fit dedup per_assay.tsv] --out per_assay.tsv
  floor-count --pair <parent id> <se id> --unit-rule UNIT_RULE.json --fit per_assay.tsv
              --rows rows.tsv --out <npz>
  score       --stats <npz...> --fit per_assay.tsv --rows rows.tsv [--floor-count <npz...>]
              --out <dir>

Fit: one pooled shape `a` over all assays and one pooled `kappa`, each fitted leave-one-experiment-
out (the row of experiment E is fitted on every other experiment and scores E); the row `all` is
fitted on every pair and also carries `q_fit`, the thinning that minimizes the deviance when `q` is
free (a check that the data say q = 0.5). The count law is `count_saturation(q c_PE, kappa, L)`:
the PE overlap counts thinned by the same q, then saturated (deviation: the brief names the law on
the PE overlap counts; without the q the law predicts twice the SE reads at low counts).
`(PE base) / (PE dedup-off)` in the top 1% of bins (by PE dedup-off starts) is the size of the PE
cap; it needs the parent's `*.nodedup` view in the store.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from t396 import caps  # noqa: E402
from t396 import dedup_cap as D  # noqa: E402

Q = 0.5
Q_GRID = tuple(np.round(np.arange(0.30, 0.701, 0.02), 4))
PRIMARY = "thinned_dm_distinct"
DEDUP_A_LAW = "thinned_dm_distinct_dedup_a"

ROW_COLUMNS = ["parent_id", "se_id", "experiment", "assay", "parent_run_type", "in_eic",
               "read_length", "read_length_parent", "reads_parent", "reads_se", "units",
               "unit_shares", "pipeline", "se_mate", "se_dedup_rule", "parent_nodedup_id"]
FIT_COLUMNS = ["experiment", "assay", "n_pairs_fit", "a", "kappa", "q_fit", "a_at_q_fit",
               "dedup_a"]
TABLE_COLUMNS = [c if c != "layer" else "parent_run_type" for c in D.TABLE_COLUMNS]

SE_MATE = "mate 1 of each pair, plus the single-end runs (mixed)"


def se_dedup_rule(pipeline: str) -> str:
    if "dnase" in (pipeline or "").lower():
        return "per replicate; reads flagged 0x400 dropped in the tagAlign step (dnase-seq-pipeline)"
    return "per replicate; the pipeline's single-end filter (MarkDuplicates by 5' position, strand)"


def sepe_rows(tables, rs_store, count_store=None):
    ms = D.read_tsv(os.path.join(tables, "measurements.tsv"))
    by_id = {r["measurement_id"]: r for r in ms}
    out = []
    for r in ms:
        if r.get("arm_knob") != "se" or r.get("in_eic") not in D.SPLITS:
            continue
        pid = r["parent_measurement_id"]
        if not D.in_store(rs_store, pid):
            continue
        par = by_id.get(pid, {})
        u = D.units_of(r["measurement_id"], rs_store, count_store)
        exp, unit, layer, _ = pid.split(".", 3)
        nd = f"{exp}.{unit}.{layer}.nodedup"
        out.append({
            "parent_id": pid, "se_id": r["measurement_id"], "experiment": r["experiment"],
            "assay": r["assay"], "parent_run_type": par.get("run_type", ""), "in_eic": r["in_eic"],
            "read_length": D.read_length_of(r["read_length"]),
            "read_length_parent": par.get("read_length", ""),
            "reads_parent": par.get("depth_reads", ""), "reads_se": r["depth_reads"],
            "units": u, "unit_shares": D.unit_shares(r["measurement_id"], u, by_id, arm_off="se"),
            "pipeline": r.get("pipeline", ""), "se_mate": SE_MATE,
            "se_dedup_rule": se_dedup_rule(r.get("pipeline", "")),
            "parent_nodedup_id": nd if D.in_store(rs_store, nd) else "",
        })
    out.sort(key=lambda x: (x["parent_run_type"], x["in_eic"], x["se_id"]))
    return out


def pe_cap_sums(base_h5, nodedup_h5):
    """[sum of PE base starts, sum of PE dedup-off starts] over the top 1% of strand-bins by the
    dedup-off starts (per strand, both strands added)."""
    tot = np.zeros(2)
    for col in (0, 1):
        chroms = D._chroms(base_h5, nodedup_h5)
        nd = {c: D._read(nodedup_h5, c, col) for c in chroms}
        m = D.top_masks(nd)
        for c in chroms:
            b = D._read(base_h5, c, col)
            tot += [b[m[c]].sum(), nd[c][m[c]].sum()]
    return tot


# ------------------------------------------------------------------ fits
def fit_kappa_q(hists, Ls, q=Q):
    def dev(lk):
        kappa = float(np.exp(lk))
        return sum(caps.deviance_of(h, lambda s, L=L: caps.count_saturation(q * s, kappa, L))
                   for h, L in zip(hists, Ls))
    return caps._fit(dev)


def fit_q_and_a(hists, Ks, q_grid=Q_GRID):
    """(q, a) minimizing the start-space deviance over a q grid, a fitted at each q."""
    best = None
    for q in q_grid:
        a = caps.fit_shape(hists, Ks, q=float(q))
        d = sum(caps.deviance_of(h, lambda s, k=k: caps.thinned_dm_distinct(s, float(q), a, k))
                for h, k in zip(hists, Ks))
        if best is None or d < best[0]:
            best = (d, float(q), a)
    return best[1], best[2]


def dedup_a_of(dedup_fit, assay):
    if not dedup_fit:
        return float("nan")
    idx = {(r["assay"], r["run_type"]): r for r in dedup_fit}
    for key in ((assay, "single-ended"), (assay, "all")):
        if key in idx:
            return float(idx[key]["a"])
    return float("nan")


def loeo_fit(items, dedup_fit=None, q_grid=Q_GRID):
    rows = []
    for e in sorted({x["experiment"] for x in items}):
        rest = [x for x in items if x["experiment"] != e]
        own = [x for x in items if x["experiment"] == e]
        a = caps.fit_shape([x["start"] for x in rest], [x["K"] for x in rest], q=Q)
        k = fit_kappa_q([x["count"] for x in rest], [x["read_length"] for x in rest])
        rows.append({"experiment": e, "assay": own[0]["assay"], "n_pairs_fit": len(rest), "a": a,
                     "kappa": k, "q_fit": float("nan"), "a_at_q_fit": float("nan"),
                     "dedup_a": dedup_a_of(dedup_fit, own[0]["assay"])})
    a = caps.fit_shape([x["start"] for x in items], [x["K"] for x in items], q=Q)
    k = fit_kappa_q([x["count"] for x in items], [x["read_length"] for x in items])
    qf, aq = fit_q_and_a([x["start"] for x in items], [x["K"] for x in items], q_grid)
    rows.append({"experiment": "all", "assay": "all", "n_pairs_fit": len(items), "a": a, "kappa": k,
                 "q_fit": qf, "a_at_q_fit": aq, "dedup_a": float("nan")})
    return rows


def _items(stats_paths, rows):
    by_se = {r["se_id"]: dict(r, on_id=r["se_id"], run_type=r["parent_run_type"], layer="")
             for r in rows}
    items = D._items(stats_paths, by_se)
    for it in items:
        it["parent_run_type"] = by_se[it["id"]]["parent_run_type"]
        it["shares"] = np.array([1.0])
    return items


# ------------------------------------------------------------------ CLI
def cmd_rows(a):
    rows = sepe_rows(a.tables, a.rs_store, a.count_store)
    D.write_tsv(a.out, rows, ROW_COLUMNS)
    print(f"[sepe_cap] rows: {len(rows)} pairs -> {a.out}")


def cmd_stats(a):
    rule = D.load_unit_rule(a.unit_rule)
    par, se = a.pair
    units = D.units_of(se, a.rs_store, a.count_store)
    tables = D.pair_stats(par, se, a.rs_store, a.count_store)
    exp, unit, layer, _ = par.split(".", 3)
    nd = f"{exp}.{unit}.{layer}.nodedup"
    if D.in_store(a.rs_store, nd):
        tables["extra_pe_cap_top1pct"] = pe_cap_sums(os.path.join(a.rs_store, par, "starts.h5"),
                                                     os.path.join(a.rs_store, nd, "starts.h5"))
    meta = {"source": par, "target": se, "units": units, "K": D.K_for(units, rule), "q": Q,
            "dedup_unit": rule.get("dedup_unit"), "parent_nodedup": nd if D.in_store(a.rs_store, nd)
            else "", "rs_store": a.rs_store, "count_store": a.count_store}
    D.save_stats(a.out, tables, meta)
    print(f"[sepe_cap] stats {par} -> {se}: K={meta['K']} -> {a.out}")


def cmd_fit(a):
    items = _items(a.stats, D.read_tsv(a.rows))
    dedup_fit = D.read_tsv(a.dedup_fit) if a.dedup_fit and os.path.isfile(a.dedup_fit) else None
    rows = loeo_fit(items, dedup_fit)
    D.write_tsv(a.out, rows, FIT_COLUMNS)
    print(f"[sepe_cap] fit: {len(items)} pairs, leave-one-experiment-out -> {a.out}")


def _fit_of(fit, experiment):
    r = {x["experiment"]: x for x in fit}.get(experiment) or {x["experiment"]: x for x in fit}["all"]
    return {k: float(r[k]) for k in ("a", "kappa", "dedup_a")}


def cmd_floor_count(a):
    rule = D.load_unit_rule(a.unit_rule)
    par, se = a.pair
    r = {x["se_id"]: x for x in D.read_tsv(a.rows)}[se]
    p = _fit_of(D.read_tsv(a.fit), r["experiment"])
    K = D.K_for(D.units_of(se, a.rs_store, a.count_store), rule)
    fl = D.simulate_count_floor(os.path.join(a.rs_store, par, "starts.h5"),
                                os.path.join(a.rs_store, se, "starts.h5"),
                                os.path.join(a.count_store, par, "counts.h5"), p["a"], K, q=Q,
                                n_rep=a.n_rep, rng=np.random.default_rng(a.seed))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, genome=fl["genome"], top1pct=fl["top1pct"],
             meta=np.array(json.dumps({"source": par, "target": se, "a": p["a"], "K": K, "q": Q})))
    print(f"[sepe_cap] floor-count {se}: {fl}")


def cmd_score(a):
    rows = D.read_tsv(a.rows)
    fit = D.read_tsv(a.fit)
    floors = D.load_floor_counts(a.floor_count)
    rng = np.random.default_rng(a.seed)
    pair_rows, caps_rows = [], []
    for it in _items(a.stats, rows):
        f = _fit_of(fit, it["experiment"])
        extra = {DEDUP_A_LAW: f["dedup_a"]} if np.isfinite(f["dedup_a"]) else {}
        p = {"a": f["a"], "kappa": f["kappa"], "a_shared": float("nan"), "extra": extra}
        for r in D.score_pair(it, p, floors.get(it["id"]), rng=rng, n_rep=a.n_rep,
                              start_laws=["dm_distinct"] + list(extra)):
            if r["law"] == "dm_distinct":
                r["law"] = PRIMARY
            r.update({"pair": it["id"], "experiment": it["experiment"], "assay": it["assay"],
                      "parent_run_type": it["parent_run_type"], "in_eic": it["in_eic"]})
            pair_rows.append(r)
        pc = it["stats"]["extra"].get("extra_pe_cap_top1pct")
        if pc is not None:
            caps_rows.append({"pair": it["id"], "parent_run_type": it["parent_run_type"],
                              "pe_base_top1pct": float(pc[0]), "pe_nodedup_top1pct": float(pc[1]),
                              "pe_base_over_nodedup_top1pct": float(pc[0] / pc[1]) if pc[1] else
                              float("nan")})
    os.makedirs(a.out, exist_ok=True)
    D.write_tsv(os.path.join(a.out, "pairs.tsv"), pair_rows,
                ["pair", "experiment", "assay", "parent_run_type", "in_eic", "law", "space",
                 "scope", "D_law", "D_identity", "D_depth", "D_floor"])
    D.write_tsv(os.path.join(a.out, "pe_cap.tsv"), caps_rows,
                ["pair", "parent_run_type", "pe_base_top1pct", "pe_nodedup_top1pct",
                 "pe_base_over_nodedup_top1pct"])
    table = D.aggregate(pair_rows, ["parent_run_type", "in_eic"], n_boot=a.n_boot, seed=a.seed,
                        primary=PRIMARY, other=D.COUNT_LAW)
    D.write_tsv(os.path.join(a.out, "table.tsv"), table, TABLE_COLUMNS)
    print(f"[sepe_cap] score: {len(pair_rows)} pair rows, {len(table)} table rows -> {a.out}")
    return table


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("rows")
    p.add_argument("--tables", default=D.TABLES)
    D.add_store_args(p)
    p.add_argument("--out", required=True)
    p = sub.add_parser("stats")
    p.add_argument("--pair", nargs=2, required=True, metavar=("PARENT", "SE"))
    p.add_argument("--unit-rule", required=True)
    D.add_store_args(p)
    p.add_argument("--out", required=True)
    p = sub.add_parser("fit")
    p.add_argument("--stats", nargs="+", required=True)
    p.add_argument("--rows", required=True)
    p.add_argument("--dedup-fit")
    p.add_argument("--out", required=True)
    p = sub.add_parser("floor-count")
    p.add_argument("--pair", nargs=2, required=True, metavar=("PARENT", "SE"))
    p.add_argument("--unit-rule", required=True)
    p.add_argument("--fit", required=True)
    p.add_argument("--rows", required=True)
    p.add_argument("--n-rep", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    D.add_store_args(p)
    p.add_argument("--out", required=True)
    p = sub.add_parser("score")
    p.add_argument("--stats", nargs="+", required=True)
    p.add_argument("--fit", required=True)
    p.add_argument("--rows", required=True)
    p.add_argument("--floor-count", nargs="*")
    p.add_argument("--n-boot", type=int, default=D.N_BOOT_SCORE)
    p.add_argument("--n-rep", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"rows": cmd_rows, "stats": cmd_stats, "fit": cmd_fit, "floor-count": cmd_floor_count,
            "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
