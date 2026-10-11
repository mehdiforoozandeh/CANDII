"""t125 part 1 — score one eta run from its checkpoint (trained, shuffle, swap, depth law) and the
law pass, with t118's records plus the MSE on log(1 + counts) of the predicted mean.

**Reuse.** The per-bin quantities, record layout, figdata and JSON writer are t118's
(`ladder.score`, imported, never edited); `_setup` fills its module dict `_CTX`, so its helpers read
the `eta.model.EtaPredictor`. CRPS keeps its oracle-scaled / scale-error split (trained/score/all,
as t118 and EVAL.md). The scorer here adds, per record:

  mse_log1p_<s>   mean over subset s (all / nonzero / top1, `baseline_rungs.subset_index`; top 1 %
                  by the true target) of (log1p(m) - log1p(y))^2, m = pred.mean(loc) (the NB mean,
                  clamped as in the loss). Shows whether a gain sits in the mean or the spread.
  crps_all_gt_20  crps_all > 20 (None when crps_all is None), so aggregation counts explosions.

**Depth law.** As t118: `depth_log2_scale_pred = log2(sum mu / sum X)` over the score bins on
depth-only pairs, against `depth_log2_ratio_true = log2(d'/d)` (manifest depths). The law pass
scores the depth->depth pairs only by default (`eta.pairs.law_pairs_depth`, PI decision);
`--law-scope all` scores every never-trained arm->arm pair.

**Row-2 nocov.** The averaged map reads only the source's x, but under the offset loc depends on
d' (the target's depth). Its jobs are grouped by source pid as t118 groups them, and the cache
(`PreOffsetCache`) holds loc BEFORE the offset per (x_pid, chromosome); each job adds its own
log d'. A post-offset prediction is never reused across targets. The result is bit-identical to
one `pred.predict` per job.

    python tools/t125/eta/score.py trained <manifest> <covariates.tsv> <data_dir> <blacklist.bed> <run_dir> [--workers N]
    python tools/t125/eta/score.py law     <manifest> <covariates.tsv> <data_dir> <blacklist.bed> <run_dir> [--workers N] [--law-scope depth|all]
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_TOOLS = Path(__file__).resolve().parents[2]
for _p in (_TOOLS / "t125", _TOOLS / "t118"):   # import convention: tools/t118 and tools/t125
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ladder import pairs as lp  # noqa: E402
from ladder import score as ls  # noqa: E402
from ladder.score import (SUBSETS, _eval_arrays, _is_depth_pair, _meta_profile,  # noqa: E402,F401
                          _pair_fields, _snip_plan, _snippets, _worker_init, g_version, pit_hist,
                          run_info, swap_ratio, write_json)
from eta import offset  # noqa: E402
from eta import pairs as ep  # noqa: E402

SPACE = "counts"
_CTX = ls._CTX                    # t118's module dict: its helpers and the forked workers read it
MSE_KEYS = tuple(f"mse_log1p_{s}" for s in SUBSETS)
EXPLODE_AT = 20.0
EXPLODE_KEY = "crps_all_gt_20"
CACHE_KEY = "eta_pre_offset"


# ---------------------------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------------------------


def metric_fields_eta(pred, loc: np.ndarray, disp: np.ndarray, y: np.ndarray,
                      split: bool = False) -> tuple[dict, dict]:
    """`ladder.score.metric_fields` plus mse_log1p_<subset> and the CRPS > 20 flag."""
    out, internals = ls.metric_fields(pred, loc, disp, y, split=split)
    sq = (np.log1p(internals["m"]) - np.log1p(internals["y64"])) ** 2
    for s, idx in internals["subsets"].items():
        out[f"mse_log1p_{s}"] = ls._finite(float(sq[idx].mean())) if idx.size else None
    c = out.get("crps_all")
    out[EXPLODE_KEY] = None if c is None else bool(c > EXPLODE_AT)
    return out, internals


# ---------------------------------------------------------------------------------------------
# row-2 nocov: loc cached before the offset
# ---------------------------------------------------------------------------------------------


class PreOffsetCache:
    """A stand-in for the predictor inside `ladder.score._eval_arrays` (it reads `corpus` and
    `predict`): loc before the offset per (x_pid, chromosome), plus each call's own log d'.
    Only for row-2 nocov, whose averaged map does not depend on the covariate pids."""

    def __init__(self, pred, cache: dict | None = None):
        if not (pred.per_bin and pred.model == "nocov"):
            raise ValueError(f"{pred.run_name}: the pre-offset cache is for row-2 nocov only")
        self.pred, self.corpus = pred, pred.corpus
        self.cache = {} if cache is None else cache

    def predict(self, x_src_pid: str, cov_src_pid: str, cov_tgt_pid: str, chrom: str):
        from eta import model as em   # torch; the predictor is already loaded when this runs
        pred = self.pred
        key = (x_src_pid, chrom)
        if key not in self.cache:
            X = pred.corpus.get(x_src_pid, SPACE, chrom)
            cond, avg = pred._cond(cov_src_pid, cov_tgt_pid)
            self.cache[key] = em.predict_chrom_bins_eta(pred.ladder, X, pred.log_d[x_src_pid], 0.0,
                                                        cond, pred.device, average=avg)
        loc_eta, disp = self.cache[key]
        return offset.offset_loc(loc_eta, pred.log_d[cov_tgt_pid]).astype(np.float32), disp


# ---------------------------------------------------------------------------------------------
# one (pair, eval) job
# ---------------------------------------------------------------------------------------------


def score_job_eta(job: dict) -> tuple[list, dict, list]:
    """`ladder.score.score_job` with `metric_fields_eta` and the pre-offset cache."""
    pred, blacklist, depth = _CTX["pred"], _CTX["blacklist"], _CTX["depth"]
    p, ev = job["pair"], job["eval"]
    chroms = ls._chroms_of(pred, job["x_pid"], SPACE, ev)
    keep_full = bool(job.get("meta") or job.get("snip"))
    cache = _CTX.get(CACHE_KEY)
    src = pred if cache is None else PreOffsetCache(pred, cache)
    _CTX["nocov_pred"] = None              # t118's post-offset cache is never used here
    arr = _eval_arrays(src, SPACE, job["x_pid"], job["y_pid"], job["cov_src"], job["cov_tgt"],
                       chroms, blacklist, keep_full)
    rec = {"kind": job["kind"], "eval": ev, **_pair_fields(p),
           "cov_src_pid": job["cov_src"], "cov_tgt_pid": job["cov_tgt"],
           "n_blacklisted": arr["n_blacklisted"]}
    fields, internals = metric_fields_eta(pred, arr["loc"], arr["disp"], arr["y"],
                                          split=bool(job.get("split")))
    internals["mean_full"] = lambda a: np.asarray(pred.mean(a), np.float64)
    rec.update(fields)
    if job.get("describe"):
        rec["describe"] = pred.describe(job["cov_src"], job["cov_tgt"])
    if job.get("pit"):
        rec["pit_hist"] = pit_hist(SPACE, internals["m"], internals["spread"], internals["y64"],
                                   arr["loc"])
    if job["kind"] == "swap":
        rec[ls.SWAP_KEY] = swap_ratio(internals["m"], arr["x"])
    out = [rec]
    if job.get("depth"):
        sx = float(arr["x"].astype(np.float64).sum())
        sm = float(internals["m"].sum())
        dr = {**rec, "kind": "depthlaw", "depth_source": job["kind"],
              "depth_log2_ratio_true": math.log2(depth[p["target_pid"]] / depth[p["source_pid"]]),
              "depth_log2_scale_pred": math.log2(sm / sx) if sx > 0 and sm > 0 else None}
        for k in ("describe", "pit_hist", "crps_oracle_scaled_all", "scale_error_all", "c_star_all"):
            dr.pop(k, None)
        out.append(dr)
    figs, locs = {}, []
    if job.get("meta"):
        figs[f"meta__{p['source_pid']}__{p['target_pid']}"] = _meta_profile(arr, internals)
    for arm, arm_index in job.get("snip") or ():
        f, lo = _snippets(pred, arr, internals, arm, arm_index, pred.seed, p)
        figs.update(f)
        locs.extend(lo)
    return out, figs, locs


# ---------------------------------------------------------------------------------------------
# job running (t118's, with the eta job and the pre-offset cache)
# ---------------------------------------------------------------------------------------------


def _run_jobs(jobs: list[dict], workers: int) -> list:
    if _CTX.get("by_source"):
        return _run_by_source(jobs, workers)
    if workers <= 1 or len(jobs) <= 1:
        return [score_job_eta(j) for j in jobs]
    ctx = mp.get_context("fork")
    with ctx.Pool(min(workers, len(jobs)), initializer=_worker_init) as pool:
        return pool.map(score_job_eta, jobs, chunksize=1)


def _score_source(items: list) -> list:
    """One source pid's jobs [(index, job)] -> [(index, result)]; loc before the offset is
    predicted once per (x_pid, chromosome)."""
    _CTX[CACHE_KEY] = {}
    try:
        return [(i, score_job_eta(job)) for i, job in items]
    finally:
        _CTX[CACHE_KEY] = None


def _run_by_source(jobs: list[dict], workers: int) -> list:
    groups: dict = {}
    for i, j in enumerate(jobs):
        groups.setdefault(j["x_pid"], []).append((i, j))
    tasks = sorted(groups.values(), key=len, reverse=True)
    if workers <= 1 or len(tasks) <= 1:
        done = [_score_source(t) for t in tasks]
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(min(workers, len(tasks)), initializer=_worker_init) as pool:
            done = pool.map(_score_source, tasks, chunksize=1)
    out = [None] * len(jobs)
    for group in done:
        for i, r in group:
            out[i] = r
    return out


def _setup(pred, rows, blacklist):
    """t118's `_setup` (pred, blacklist, manifest depths, by_source = row-2 nocov), no cache yet."""
    ls._setup(pred, rows, blacklist)
    _CTX["nocov_pred"] = None
    _CTX[CACHE_KEY] = None


def _config(blacklist) -> dict:
    cfg = ls._config(blacklist)
    cfg["mse_log1p"] = ("mean over the subset of (log1p(pred.mean(loc)) - log1p(y))^2; "
                        "subsets as crps")
    cfg["offset"] = "eta_in = log1p(X) - log(d/1e6); loc = log d' + eta_out (d' = cov_tgt's depth)"
    cfg[EXPLODE_KEY] = f"crps_all > {EXPLODE_AT:g}"
    return cfg


def _t125(law_scope) -> dict:
    return {"law_scope": law_scope, "metrics_added": list(MSE_KEYS), "explode_key": EXPLODE_KEY,
            "explode_at": EXPLODE_AT}


def _n_explode(records) -> int:
    return sum(bool(r.get(EXPLODE_KEY)) for r in records)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------------------------
# run level
# ---------------------------------------------------------------------------------------------


def score_trained(pred, rows: list[dict], blacklist, run_dir, workers: int = 1) -> Path:
    """trained (score + val), shuffle, swap and depth-law records -> scores.json, figdata.npz."""
    t0 = time.time()
    run_dir = Path(run_dir)
    _setup(pred, rows, blacklist)
    real = pred.model == "real"
    train = lp.train_pairs(rows, pred.g_id)
    snip = _snip_plan(train) if real else {}
    jobs, skipped = [], []
    for p in train:
        key = (p["source_pid"], p["target_pid"])
        for ev in ("score", "val"):
            sc = ev == "score"
            jobs.append({"kind": "trained", "eval": ev, "pair": p, "x_pid": p["source_pid"],
                         "y_pid": p["target_pid"], "cov_src": p["source_pid"],
                         "cov_tgt": p["target_pid"], "split": sc, "pit": sc, "describe": True,
                         "meta": real and sc, "snip": snip.get(key) if sc else None,
                         "depth": sc and _is_depth_pair(p, "trained")})
    for p in train:
        try:
            wrong = lp.shuffle_target(rows, p, SPACE, pred.seed)
        except ValueError as e:
            skipped.append({"kind": "shuffle", "source_pid": p["source_pid"],
                            "target_pid": p["target_pid"], "reason": str(e)})
            continue
        jobs.append({"kind": "shuffle", "eval": "score", "pair": p, "x_pid": p["source_pid"],
                     "y_pid": p["target_pid"], "cov_src": p["source_pid"], "cov_tgt": wrong})
    by_pid = {r["pid"]: r for r in lp.usable_products(rows)}
    for pid in lp.fit_pids(rows, pred.g_id):
        r = by_pid[pid]
        p = lp._pair(r, r, "swap")
        jobs.append({"kind": "swap", "eval": "score", "pair": p, "x_pid": pid, "y_pid": pid,
                     "cov_src": pid, "cov_tgt": pid})
    t1 = time.time()
    results = _run_jobs(jobs, workers)
    records, figs, snippets = [], {}, []
    for recs, f, lo in results:
        records.extend(recs)
        figs.update(f)
        snippets.extend(lo)
    t2 = time.time()
    np.savez(run_dir / "figdata.npz.tmp.npz", **figs)
    os.replace(run_dir / "figdata.npz.tmp.npz", run_dir / "figdata.npz")
    out = {"created_utc": _now(), "run": run_info(pred, run_dir), "config": _config(blacklist),
           "records": records, "snippets": snippets, "skipped": skipped,
           "n_records_crps_nonfinite": sum(r.get("n_crps_nonfinite", 0) > 0 for r in records),
           "n_records_crps_gt_20": _n_explode(records), "t125": _t125(None),
           "timing": {"setup": round(t1 - t0, 3), "score": round(t2 - t1, 3),
                      "total": round(time.time() - t0, 3), "n_jobs": len(jobs),
                      "workers": int(workers)}}
    write_json(out, run_dir / "scores.json")
    (run_dir / "SCORE_DONE").write_text(out["created_utc"] + "\n")
    return run_dir / "scores.json"


def score_law(pred, rows: list[dict], blacklist, run_dir, workers: int = 1,
              law_scope: str = "depth") -> Path:
    """The never-trained law pairs of the run's g (`law_scope` "depth": depth->depth only, the
    default; "all": every arm->arm pair), plus depth-law records on the depth->depth pairs."""
    if law_scope not in ep.LAW_SCOPES:
        raise ValueError(f"law_scope {law_scope!r} not in {ep.LAW_SCOPES}")
    t0 = time.time()
    run_dir = Path(run_dir)
    _setup(pred, rows, blacklist)
    jobs = [{"kind": "law", "eval": "score", "pair": p, "x_pid": p["source_pid"],
             "y_pid": p["target_pid"], "cov_src": p["source_pid"], "cov_tgt": p["target_pid"],
             "depth": _is_depth_pair(p, "law")}
            for p in ep.law_pairs(rows, pred.g_id, scope=law_scope)]
    t1 = time.time()
    records = [r for recs, _, _ in _run_jobs(jobs, workers) for r in recs]
    t2 = time.time()
    out = {"created_utc": _now(), "run": run_info(pred, run_dir), "config": _config(blacklist),
           "records": records,
           "n_records_crps_nonfinite": sum(r.get("n_crps_nonfinite", 0) > 0 for r in records),
           "n_records_crps_gt_20": _n_explode(records), "t125": _t125(law_scope),
           "timing": {"setup": round(t1 - t0, 3), "score": round(t2 - t1, 3),
                      "total": round(time.time() - t0, 3), "n_jobs": len(jobs),
                      "workers": int(workers)}}
    write_json(out, run_dir / "law.json")
    (run_dir / "LAW_DONE").write_text(out["created_utc"] + "\n")
    return run_dir / "law.json"


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("trained", "law"):
        a = sub.add_parser(name)
        for pos in ("manifest", "covariates", "data_dir", "blacklist", "run_dir"):
            a.add_argument(pos, type=Path)
        a.add_argument("--workers", type=int, default=ls.default_workers())
        if name == "law":
            a.add_argument("--law-scope", choices=ep.LAW_SCOPES, default="depth")
    args = ap.parse_args(argv)
    done = args.run_dir / ("SCORE_DONE" if args.cmd == "trained" else "LAW_DONE")
    if done.exists():
        print(f"{done} exists; nothing to do")
        return 0
    from eta import model as em  # the checkpoint loader (torch); imported only here
    t0 = time.time()
    pred = em.load_run(args.run_dir, args.data_dir, args.covariates, args.manifest, device="cpu")
    rows = ep.read_manifest(args.manifest)
    if args.cmd == "trained":
        path = score_trained(pred, rows, args.blacklist, args.run_dir, workers=args.workers)
    else:
        path = score_law(pred, rows, args.blacklist, args.run_dir, workers=args.workers,
                         law_scope=args.law_scope)
    print(f"{args.cmd} {args.run_dir.name} -> {path} ({time.time() - t0:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
