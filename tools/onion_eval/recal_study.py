"""Recalibration study of the per-assay NB dispersion factor (task t370, chunk recalstudy).

PI request: does one per-assay dispersion factor c (n' = n / c, mu unchanged;
`ensemble_store.recalibrate`) fitted on held-out regions carry over to other regions, to other
cells, and from observed-data pairs to imputation items? Per seed of the final model.

forward. Draws held-out calibration regions (`ensemble_store.calibration_regions`: candidate
blocks in no `regions` / `fit_regions` of the eval sets, and, with --exclude-regions-from, in no
region of an earlier forward), writes `<out>/calibration_evalset.json` as `ensemble_store.fit`
does (except that the pair jobs run on every calibration region: no `pair_regions`), and runs every
model over
  (a) the imputation items of `ensemble_store.FIT_REGIMES`, exactly as `ensemble_store.fit` does
      (`Scorer.ds.window` of the item's inputs and target), and
  (b) the correction pairs of `Scorer.jobs()` (source pr1 in, target pr2 scored, through
      `Scorer._pair_window`; `Scorer.jobs()` already drops the pairs with read_disjoint 0, and
      their count is recorded).
Per item it writes `<out>/items/<sha1 of key>.npz` with y [B] and n, mu [K, B] (float32; K runs,
B bins in region then window order) and `meta` (json: kind imputation|pair, key, cell, assay,
target, source, regime). `<out>/index.json` (written last) lists the items and the map from every
pair key of the Scorer (correction, replicate, same-view) to its target assay. A rerun skips a
stored item, and skips the whole command when index.json exists.

    python tools/onion_eval/recal_study.py forward --runs <seed0> <seed1> <seed2>
        --evalset <json> [--exclude-evalset <json> ...] --n-regions 32 --region-seed <int>
        [--exclude-regions-from <forward out dir>] --out <dir> [--max-bins 300000]
        [--infer-batch 8] [--device cpu]

analyze. Per seed k and per kind (imputation, pair), the pooled PIT deviation per assay
(`ensemble_store.pit_dev`, the quantity `ensemble_store.fit` minimizes) on the SCORE forward, with
bins chosen by `ensemble_store.fit_index` (finite in every run, n > 0, seeded subsample of at most
--max-bins), under the conditions
  none        c = 1;
  same_cells  c per assay fitted (`fit_factor`) on the FIT forward, all cells, same kind;
  cross_cell  2-fold cross-fit over cells: the distinct cells of the FIT forward's items of that
              kind, sorted, fold = position % 2; a SCORE item of a cell in fold f takes the factor
              fitted on the FIT items of the cells of the other fold. An assay with no fit data in
              that fold, or a cell not in the FIT forward, takes c = 1 and is counted;
  shipped     (imputation only) the same_cells factor of the pair kind applied to the imputation
              items (c = 1 and counted for an assay with no pair fit data).
Reports per seed, the mean and the range (max - min) over seeds, per assay and mean over assays,
with bins and items, every factor and its `at_bound`. Every condition leaves mu unchanged
(asserted). Writes <prefix>.json, <prefix>.md and <prefix>.pair_factors.json (factor sets seed<k>
and ensemble fitted on the FIT forward's pair items, with the pair -> target assay map; the input of
`ensemble_store.py combine --pair-factors`).

    python tools/onion_eval/recal_study.py analyze --fit <forward dir> --score <forward dir>
        --out <prefix> [--max-bins 300000]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from onion_eval import ensemble_store as ES  # noqa: E402

KINDS = ("imputation", "pair")
CONDITIONS = ("none", "same_cells", "cross_cell", "shipped")
INDEX = "index.json"


def _read_json(p) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _write_json(p: Path, doc) -> None:
    tmp = p.with_name("." + p.name + ".part")
    tmp.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, p)


# ------------------------------------------------------------------------------------ forward

def calibration_evalset(ev: dict, regions, meta: dict) -> dict:
    """The eval set of the calibration regions: `ev` with `regions` replaced, no strata, and no
    `pair_regions` (the pair jobs run on every calibration region)."""
    cal = dict(ev)
    cal.update({"regions": [list(x) for x in regions], "strata": [], "calibration_of": meta})
    cal.pop("pair_regions", None)
    cal.pop("pair_regions_p", None)
    return cal


def item_key(kind: str, job: dict) -> str:
    """The prediction-store key of a job: item/<target>/<regime> or pair/<pair key>."""
    if kind == "imputation":
        return f"item/{job['target']}/{job['regime']}"
    return "pair/" + (job.get("pair_id") or job["source"] + ">" + job["target"])


def item_meta(kind: str, job: dict, assay: str, cell: str) -> dict:
    return {"kind": kind, "key": item_key(kind, job), "cell": cell, "assay": assay,
            "target": job["target"], "source": job.get("source"), "regime": job.get("regime")}


def write_item(out: Path, meta: dict, y, n, mu) -> Path:
    """Write one item (y [B]; n, mu [K, B]) atomically to <out>/items/<sha1 of key>.npz."""
    from onion_eval import predict_store as P
    y = np.asarray(y, np.float32)
    n = np.asarray(n, np.float32)
    mu = np.asarray(mu, np.float32)
    if n.shape != mu.shape or n.ndim != 2 or n.shape[1] != y.size:
        raise ValueError(f"{meta['key']}: y {y.shape}, n {n.shape}, mu {mu.shape}")
    d = Path(out) / "items"
    d.mkdir(parents=True, exist_ok=True)
    f = d / P.key_file(meta["key"])
    tmp = d / ("." + f.name + ".part")
    with open(tmp, "wb") as fh:
        np.savez(fh, y=y, n=n, mu=mu, meta=np.array(json.dumps(meta, sort_keys=True)))
    os.replace(tmp, f)
    return f


def read_item(f: Path) -> dict:
    with np.load(f) as z:
        it = json.loads(str(z["meta"]))
        it.update(y=z["y"], n=z["n"], mu=z["mu"])
    return it


def prior_regions_doc(forward_dir) -> dict:
    """An earlier forward's calibration eval set, as an eval set to exclude the regions of."""
    p = Path(forward_dir) / "calibration_evalset.json"
    if not p.is_file():
        raise SystemExit(f"refused: {p} does not exist")
    return _read_json(p)


def forward(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="recal_study.py forward", description="model forwards of "
                                 "imputation items and correction pairs on calibration regions")
    ap.add_argument("--runs", nargs="+", required=True, help="seed run dirs; the first is seed0")
    ap.add_argument("--evalset", required=True)
    ap.add_argument("--exclude-evalset", nargs="*", default=[])
    ap.add_argument("--exclude-regions-from", default=None,
                    help="an earlier forward out dir whose calibration regions are avoided")
    ap.add_argument("--n-regions", type=int, default=32)
    ap.add_argument("--region-seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-bins", type=int, default=300000,
                    help="recorded only; analyze applies the subsample")
    ap.add_argument("--infer-batch", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)

    def log(*m):
        print(*m, flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / INDEX).is_file():
        log(f"forward: skip, {out / INDEX} exists")
        return 0
    from onion_eval import score as S
    from onion_eval import predict_store as P
    from minicandi.registry import load_run
    S.set_threads()
    S.tf32_off()
    t0 = time.time()
    ev = _read_json(a.evalset)
    exc = [_read_json(p) for p in a.exclude_evalset]
    if a.exclude_regions_from:
        exc.append(prior_regions_doc(a.exclude_regions_from))
    shas = {S.run_tables_sha256(r) for r in a.runs}
    if len(shas) != 1:
        raise SystemExit("refused: the runs were trained on different tables")
    models = [load_run(r) for r in a.runs]
    cfg0 = {k: v for k, v in models[0][1].items() if k != "seed"}
    for r, (_, cfg) in zip(a.runs[1:], models[1:]):
        d = sorted(k for k in set(cfg0) | set(cfg) if k != "seed" and cfg0.get(k) != cfg.get(k))
        if d:
            log(f"forward: WARNING: {r} config differs from the first run's in {d}")
    regions = ES.calibration_regions(ev, Path(ev["tables_dir"]), exc, a.n_regions, a.region_seed)
    meta = {"evalset": str(Path(a.evalset).resolve()), "evalset_sha256": S.sha256_file(a.evalset),
            "excluded": [str(Path(p).resolve()) for p in a.exclude_evalset],
            "excluded_regions_from": (str(Path(a.exclude_regions_from).resolve())
                                      if a.exclude_regions_from else None),
            "n_regions": a.n_regions, "region_seed": a.region_seed}
    cal_path = out / "calibration_evalset.json"
    _write_json(cal_path, calibration_evalset(ev, regions, meta))
    sc = S.Scorer(str(cal_path), main="model", model=models[0][0], config=models[0][1],
                  device=a.device, run_tables_sha256=shas.pop(), log=log, n_boot=0)
    preds = [S.ModelPredictor(m, a.device) for m, _ in models]
    imp, cor, rep, _ = sc.jobs()
    jobs = ([("imputation", j) for j in imp if j["regime"] in ES.FIT_REGIMES]
            + [("pair", j) for j in cor])
    pair_map = {S._pair_key(j): sc._row(j["target"])["assay"]
                for j in cor + rep + S.same_view_jobs(cor)}
    log(f"forward: {len(jobs)} items ({sum(k == 'pair' for k, _ in jobs)} correction pairs; "
        f"{sc.n_pairs_not_read_disjoint} pairs not read-disjoint left out), {len(regions)} "
        f"calibration regions, {len(preds)} models")
    tm = time.time()
    metas = []
    for ji, (kind, job) in enumerate(jobs):
        row = sc._row(job["target"])
        t = sc.C.TRACK_INDEX[row["assay"]]
        assay = job["assay"] if kind == "imputation" else row["assay"]
        m = item_meta(kind, job, assay, job.get("cell") or row["cell"])
        f = out / "items" / P.key_file(m["key"])
        metas.append(dict(m, file=f.name))
        if f.is_file():
            log(f"forward: item {ji + 1}/{len(jobs)} {m['key']} stored, skip")
            continue
        keep = set(range(len(sc.regions))) if kind == "imputation" else set(sc.pair_region_idx)
        ys, ns, mus = [], [[] for _ in preds], [[] for _ in preds]
        for ri, tiles in sc._tiles(sc.regions):
            if ri not in keep:
                continue
            sc._set_region(*sc.regions[ri])
            for b0 in range(0, len(tiles), a.infer_batch):
                tl = tiles[b0:b0 + a.infer_batch]
                if kind == "imputation":
                    wins = [sc.ds.window(c, s, job["inputs"], [job["target"]]) for c, s in tl]
                else:
                    wins = [sc._pair_window(job, c, s) for c, s in tl]
                batch = sc.C.collate(wins)
                ys.append(batch["y_counts"][:, :, t].numpy().reshape(-1).astype(np.float64))
                for k, pr in enumerate(preds):
                    p = pr(batch)
                    ns[k].append(p["n"][:, :, t].reshape(-1))
                    mus[k].append(p["mu"][:, :, t].reshape(-1))
        write_item(out, m, np.concatenate(ys), np.stack([np.concatenate(x) for x in ns]),
                   np.stack([np.concatenate(x) for x in mus]))
        log(f"forward: item {ji + 1}/{len(jobs)} {kind} {m['key']} ({time.time() - tm:.0f} s)")
    doc = {"runs": [str(Path(r).resolve()) for r in a.runs],
           "calibration_evalset": str(cal_path.resolve()), "calibration_of": meta,
           "regions": [list(x) for x in regions], "fit_regimes": list(ES.FIT_REGIMES),
           "n_pairs_not_read_disjoint": int(sc.n_pairs_not_read_disjoint),
           "infer_batch": a.infer_batch, "device": a.device, "max_bins": a.max_bins,
           "items": metas, "pair_target_assay": pair_map,
           "seconds": {"forward": round(time.time() - tm, 1), "total": round(time.time() - t0, 1)}}
    _write_json(out / INDEX, doc)
    log(f"forward: wrote {out / INDEX} ({time.time() - t0:.0f} s)")
    return 0


# ------------------------------------------------------------------------------------ analyze

def load_forward(d) -> tuple:
    """(index, items) of a forward out dir; each item is its meta plus y, n, mu."""
    d = Path(d)
    idx = _read_json(d / INDEX)
    return idx, [read_item(d / "items" / m["file"]) for m in idx["items"]]


def pool(items) -> tuple:
    """(y [B], n [K, B], mu [K, B], item position per bin) of items, concatenated (float64)."""
    y = np.concatenate([np.asarray(it["y"], np.float64) for it in items])
    n = np.concatenate([np.asarray(it["n"], np.float64) for it in items], axis=1)
    mu = np.concatenate([np.asarray(it["mu"], np.float64) for it in items], axis=1)
    pos = np.concatenate([np.full(np.asarray(it["y"]).size, i) for i, it in enumerate(items)])
    return y, n, mu, pos


def by_assay(items) -> dict:
    out = {}
    for it in items:
        out.setdefault(it["assay"], []).append(it)
    return dict(sorted(out.items()))


def apply_factor(n, mu, c):
    """(n / c, mu): c a positive scalar or one factor per bin; mu is returned unchanged."""
    c = np.asarray(c, np.float64)
    if not np.all(c > 0):
        raise ValueError("dispersion factors must be positive")
    return np.asarray(n, np.float64) / c, mu


def _short(r: dict) -> dict:
    return {k: r[k] for k in ("factor", "log2_factor", "at_bound", "pit_at_1", "pit_at_factor")}


def fit_per_seed(items, K: int, max_bins: int) -> dict:
    """{assay: [fit_factor result of seed k for k < K]} over the pooled items of each assay;
    an assay with no usable bin is left out."""
    out = {}
    for assay, its in by_assay(items).items():
        y, n, mu, _ = pool(its)
        _, idx = ES.fit_index(y, n, mu, max_bins)
        if idx.size:
            out[assay] = [ES.fit_factor(n[k, idx], mu[k, idx], y[idx]) for k in range(K)]
    return out


def cell_folds(items) -> dict:
    """{cell: fold}: the distinct cells sorted, fold = position % 2."""
    return {c: i % 2 for i, c in enumerate(sorted({it["cell"] for it in items}))}


def _summary(vals) -> dict:
    v = [x for x in vals if x is not None]
    if len(v) != len(vals) or not v:
        return {"mean": None, "range": None, "per_seed": list(vals)}
    return {"mean": float(np.mean(v)), "range": float(max(v) - min(v)), "per_seed": list(vals)}


def analyze_kind(fit_items, score_items, K: int, max_bins: int, shipped=None) -> dict:
    """The study of one kind (module docstring). `shipped`: {assay: [per-seed fit_factor result]}
    to apply as the `shipped` condition, or None."""
    same = fit_per_seed(fit_items, K, max_bins)
    folds = cell_folds(fit_items)
    fold_fit = {f: fit_per_seed([it for it in fit_items if folds[it["cell"]] == f], K, max_bins)
                for f in (0, 1)}
    conds = ["none", "same_cells", "cross_cell"] + (["shipped"] if shipped is not None else [])
    pit = {c: [{} for _ in range(K)] for c in conds}
    bins, fallback = {}, {c: [] for c in conds if c != "none"}
    mu_unchanged = True
    for assay, its in by_assay(score_items).items():
        y, n, mu, pos = pool(its)
        ok, idx = ES.fit_index(y, n, mu, max_bins)
        bins[assay] = {"items": len(its), "bins": int(y.size), "finite": int(ok.sum()),
                       "used": int(idx.size), "cells": sorted({it["cell"] for it in its})}
        if assay not in same:
            fallback["same_cells"].append({"assay": assay, "items": len(its)})
        if shipped is not None and assay not in shipped:
            fallback["shipped"].append({"assay": assay, "items": len(its)})
        # cross_cell: the fit fold of each item's cell (the other fold) and whether it has data
        src = []
        for it in its:
            f = folds.get(it["cell"])
            if f is None:
                src.append(None)
                fallback["cross_cell"].append({"assay": assay, "cell": it["cell"],
                                               "key": it["key"], "why": "cell not in the fit set"})
            elif assay not in fold_fit[1 - f]:
                src.append(None)
                fallback["cross_cell"].append({"assay": assay, "cell": it["cell"],
                                               "key": it["key"], "fit_fold": 1 - f,
                                               "why": "no fit data for this assay in the fold"})
            else:
                src.append(1 - f)
        for k in range(K):
            nk, mk, yk = n[k, idx], mu[k, idx], y[idx]
            if not idx.size:
                for c in conds:
                    pit[c][k][assay] = None
                continue
            cs = {"none": 1.0,
                  "same_cells": same[assay][k]["factor"] if assay in same else 1.0}
            item_c = np.array([1.0 if s is None else fold_fit[s][assay][k]["factor"] for s in src])
            cs["cross_cell"] = item_c[pos[idx]]
            if shipped is not None:
                cs["shipped"] = shipped[assay][k]["factor"] if assay in shipped else 1.0
            for c in conds:
                n2, m2 = apply_factor(nk, mk, cs[c])
                if not (m2 is mk and np.array_equal(m2, mu[k, idx])):
                    mu_unchanged = False
                assert mu_unchanged, f"{assay} {c}: mu changed"
                pit[c][k][assay] = ES.pit_dev(n2, m2, yk)
    assays = sorted(bins)
    res = {"conditions": {}, "bins": bins, "n_items": len(score_items),
           "n_bins_used": int(sum(b["used"] for b in bins.values())),
           "fit_n_items": len(fit_items), "folds": folds, "fallback_c1": fallback,
           "mu_unchanged": mu_unchanged,
           "factors": {"same_cells": {a: [_short(r) for r in v] for a, v in same.items()},
                       "cross_cell": {f"fitted_on_fold{f}": {a: [_short(r) for r in v]
                                                             for a, v in fold_fit[f].items()}
                                      for f in (0, 1)}}}
    if shipped is not None:
        res["factors"]["shipped"] = {a: [_short(r) for r in v] for a, v in shipped.items()}
    for c in conds:
        per_seed = []
        for k in range(K):
            vals = [pit[c][k][a] for a in assays]
            ok_vals = [v for v in vals if v is not None]
            per_seed.append({"per_assay": pit[c][k],
                             "mean_over_assays": (float(np.mean(ok_vals))
                                                  if ok_vals and len(ok_vals) == len(vals) else None)})
        res["conditions"][c] = {
            "per_seed": per_seed,
            "per_assay": {a: _summary([pit[c][k][a] for k in range(K)]) for a in assays},
            "mean_over_assays": _summary([s["mean_over_assays"] for s in per_seed])}
    return res


def pair_factors_doc(fit_items, K: int, max_bins: int, pair_map: dict) -> dict:
    """The pair factors file of `ensemble_store.combine --pair-factors`: seed<k> and ensemble sets
    fitted on the pooled pair items of each assay, and the pair key -> target assay map."""
    sets = {f"seed{k}": {} for k in range(K)}
    if K > 1:
        sets["ensemble"] = {}
    bins = {}
    for assay, its in by_assay([it for it in fit_items if it["kind"] == "pair"]).items():
        y, n, mu, _ = pool(its)
        _, idx = ES.fit_index(y, n, mu, max_bins)
        if not idx.size:
            continue
        res, bins[assay] = ES.assay_factor_sets(y, n, mu, max_bins)
        for v, r in res.items():
            sets[v][assay] = r
    return {"criterion": "least pooled PIT deviation per target assay over the correction pairs "
                         "of the fit forward; n' = n / factor",
            "applies_to": "pair/<pair key> prediction store requests, by pair_target_assay",
            "factor_sets": sets, "pair_target_assay": dict(sorted(pair_map.items())),
            "bins_per_assay": bins, "max_bins": max_bins, "subsample_seed": 0}


def analyze_items(fit_items, score_items, K: int, max_bins: int) -> dict:
    """{kind: analyze_kind result} with the `shipped` condition on the imputation kind."""
    fi = {k: [it for it in fit_items if it["kind"] == k] for k in KINDS}
    si = {k: [it for it in score_items if it["kind"] == k] for k in KINDS}
    out = {}
    pair_same = fit_per_seed(fi["pair"], K, max_bins) if fi["pair"] else None
    for k in KINDS:
        if not si[k]:
            continue
        out[k] = analyze_kind(fi[k], si[k], K, max_bins,
                              shipped=pair_same if k == "imputation" else None)
    return out


def _fmt(x, w=".4f"):
    return "n/a" if x is None else format(x, w)


def markdown(doc: dict) -> str:
    K = doc["n_seeds"]
    L = ["# Recalibration study: pooled PIT deviation per assay", "",
         f"Fit forward: `{doc['fit']}`. Score forward: `{doc['score']}`. Seeds: {K}. "
         f"Subsample: at most {doc['max_bins']} bins per assay (seed 0). "
         "Cells cross-fitted in 2 folds (sorted cells, position % 2). Lower is better.", ""]
    for kind, r in doc["kinds"].items():
        conds = list(r["conditions"])
        L += [f"## {kind}", "",
              f"Score items {r['n_items']}, bins used {r['n_bins_used']}; fit items "
              f"{r['fit_n_items']}. Mu unchanged under every condition: {r['mu_unchanged']}.", "",
              "| condition | mean over assays, mean over seeds | range over seeds | "
              + " | ".join(f"seed{k}" for k in range(K)) + " |",
              "|---|---|---|" + "---|" * K]
        for c in conds:
            m = r["conditions"][c]["mean_over_assays"]
            L.append(f"| {c} | {_fmt(m['mean'])} | {_fmt(m['range'])} | "
                     + " | ".join(_fmt(v) for v in m["per_seed"]) + " |")
        L += ["", "Per assay: mean over seeds (range over seeds).", "",
              "| assay | items | bins used | " + " | ".join(conds) + " | same_cells factor per seed |",
              "|---|---|---|" + "---|" * len(conds) + "---|"]
        for a, b in r["bins"].items():
            cells = [r["conditions"][c]["per_assay"][a] for c in conds]
            fac = r["factors"]["same_cells"].get(a)
            fs = ("c=1 (no fit data)" if fac is None else ", ".join(
                f"{x['factor']:.3g}{' (bound)' if x['at_bound'] else ''}" for x in fac))
            L.append(f"| {a} | {b['items']} | {b['used']} | "
                     + " | ".join(f"{_fmt(s['mean'])} ({_fmt(s['range'])})" for s in cells)
                     + f" | {fs} |")
        fb = {c: len(v) for c, v in r["fallback_c1"].items()}
        L += ["", "Fallbacks to c = 1 (count of entries): "
              + ", ".join(f"{c} {v}" for c, v in fb.items()) + ".", ""]
    return "\n".join(L) + "\n"


def analyze(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="recal_study.py analyze", description="pooled PIT deviation "
                                 "per assay under each recalibration condition")
    ap.add_argument("--fit", required=True, help="forward out dir the factors are fitted on")
    ap.add_argument("--score", required=True, help="forward out dir the conditions are scored on")
    ap.add_argument("--out", required=True, help="output prefix: <prefix>.json, <prefix>.md")
    ap.add_argument("--max-bins", type=int, default=300000)
    a = ap.parse_args(argv)
    t0 = time.time()
    fidx, fit_items = load_forward(a.fit)
    sidx, score_items = load_forward(a.score)
    K = len(fidx["runs"])
    if len(sidx["runs"]) != K:
        raise SystemExit(f"refused: {K} runs in the fit forward, {len(sidx['runs'])} in the score")
    if fidx["runs"] != sidx["runs"]:
        print("analyze: WARNING: the two forwards list different run dirs", flush=True)
    shared = sorted({tuple(r) for r in fidx["regions"]} & {tuple(r) for r in sidx["regions"]})
    if shared:
        print(f"analyze: WARNING: the forwards share {len(shared)} regions", flush=True)
    doc = {"fit": str(Path(a.fit).resolve()), "score": str(Path(a.score).resolve()),
           "runs": fidx["runs"], "n_seeds": K, "max_bins": a.max_bins, "subsample_seed": 0,
           "shared_regions": [list(r) for r in shared],
           "criterion": "pooled PIT deviation per assay (ensemble_store.pit_dev); n' = n / c",
           "kinds": analyze_items(fit_items, score_items, K, a.max_bins)}
    doc["seconds"] = round(time.time() - t0, 1)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_json(out.with_name(out.name + ".json"), doc)
    out.with_name(out.name + ".md").write_text(markdown(doc), encoding="utf-8")
    pmap = {**sidx.get("pair_target_assay", {}), **fidx.get("pair_target_assay", {})}
    pdoc = pair_factors_doc(fit_items, K, a.max_bins, pmap)
    pdoc["fit_forward"] = doc["fit"]
    _write_json(out.with_name(out.name + ".pair_factors.json"), pdoc)
    print(f"analyze: wrote {out}.json, .md, .pair_factors.json ({doc['seconds']} s)", flush=True)
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "forward":
        sys.exit(forward(sys.argv[2:]))
    if cmd == "analyze":
        sys.exit(analyze(sys.argv[2:]))
    print(__doc__)
    sys.exit(2)
