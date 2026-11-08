"""t150 — the depth-only baseline beside the learned correction in the final CAC comparison.

No network and no covariate except the recorded depth: the prediction for a pair (source a,
target b) is the source's counts rescaled by the ratio of recorded depths,

    mean m = max(x_a * d_b / d_a, PRED_FLOOR)        d = the manifest's `depth`, as score.py reads it

and, as the reference that reads nothing at all, the identity map m = max(x_a, PRED_FLOOR)
(t118's noSolution). Each rule is an NB(mean m, size n) with ONE size n per (pair, rule), the
maximum-likelihood size with the mean held fixed, fitted on the training chromosomes (every main
chromosome except chr19, chr21 and chr22) on a systematic 1-in-`FIT_STRIDE` subsample of the
non-blacklisted bins. A one-value rule has no spread of its own; fitting n gives it the best
single dispersion it could have, so the comparison with the learned correction is not lost to a
spread the rule never chose.

Scored exactly as `score.py` scores a trained run: the same pairs (`kind` trained, heldout and
law, route direct), chr19 + chr21 with blacklisted bins removed, `ladder.score.crps_given`, the
`split_all` oracle-scale split, and the per-mappability-decile fields. The shuffle, chain, mix and
ceiling records do not apply: the rules read no covariate other than depth and have no base leg.

    python tools/t146/depth_baseline.py run <manifest> <cache_dir> <blacklist.bed> <out_dir>
        --features DIR [--workers N] [--law-stride K]

Writes <out_dir>/scores.json (one record per pair and rule; field `rule`) and SCORE_DONE.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

_TOOLS = Path(__file__).resolve().parents[1]
for _p in (_TOOLS / "t118", _TOOLS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import baseline_rungs as br  # noqa: E402
from ladder import data, pairs as lp  # noqa: E402
from ladder import score as lscore  # noqa: E402
from ladder.model import N_MAX, N_MIN  # noqa: E402
from t146 import tasks as T  # noqa: E402
from t146.features import Features  # noqa: E402

RULES = ("depth_only", "identity")
FIT_STRIDE = 100   # ~1.1 M bins per fit; 10 timed the smoke out (job 26494971)
TRAIN_CHROMS = tuple(c for c in lp.MAIN_CHROMS
                     if c not in lp.SCORE_CHROMS + lp.VAL_CHROMS and c not in lp.DROP_CHROMS)

_CTX: dict = {}


def nb_loglik(n: float, m: np.ndarray, y: np.ndarray) -> float:
    """Sum over bins of log NB(y | mean m, size n)."""
    return float(np.sum(gammaln(y + n) - gammaln(n) - gammaln(y + 1)
                        + n * np.log(n / (n + m)) + y * np.log(m / (n + m))))


def fit_size(m: np.ndarray, y: np.ndarray) -> float:
    """The ML NB size with the mean fixed, searched on log n over [N_MIN, N_MAX]."""
    r = minimize_scalar(lambda t: -nb_loglik(math.exp(t), m, y),
                        bounds=(math.log(N_MIN), math.log(N_MAX)), method="bounded",
                        options={"xatol": 1e-4})
    return float(math.exp(r.x))


def rule_mean(rule: str, x: np.ndarray, ratio: float) -> np.ndarray:
    m = np.asarray(x, np.float64) * (ratio if rule == "depth_only" else 1.0)
    return np.maximum(m, br.PRED_FLOOR)


def score_job(job: dict) -> list[dict]:
    corpus, blacklist, feats, depth = (_CTX[k] for k in ("corpus", "blacklist", "feats", "depth"))
    p = job["pair"]
    src, tgt = p["source_pid"], p["target_pid"]
    t0 = time.time()
    ratio = depth[tgt] / depth[src]
    # training-chromosome subsample for the size fit
    fx, fy = [], []
    for c in TRAIN_CHROMS:
        if c not in corpus.chroms(src, "counts"):
            continue
        y = corpus.get(tgt, "counts", c)
        x = corpus.get(src, "counts", c)
        keep = ~data.read_blacklist_flags(blacklist, c, y.size)
        idx = np.flatnonzero(keep)[::FIT_STRIDE]
        fx.append(np.asarray(x[idx], np.float64))
        fy.append(np.asarray(y[idx], np.float64))
    fx, fy = np.concatenate(fx), np.concatenate(fy)
    # the scored bins, as score.py
    xs, ys, ds = [], [], []
    n_black = 0
    for c in lp.SCORE_CHROMS:
        y = corpus.get(tgt, "counts", c)
        x = corpus.get(src, "counts", c)
        bad = data.read_blacklist_flags(blacklist, c, y.size)
        keep = ~bad
        n_black += int(bad.sum())
        xs.append(np.asarray(x, np.float64)[keep])
        ys.append(np.asarray(y, np.float64)[keep])
        ds.append(np.asarray(feats.dec_full(c)[:y.size])[keep])
    x, y, dec = np.concatenate(xs), np.concatenate(ys), np.concatenate(ds)
    recs = []
    for rule in RULES:
        n_fit = fit_size(rule_mean(rule, fx, ratio), fy)
        m = rule_mean(rule, x, ratio)
        n = np.full(m.shape, n_fit)
        crps = lscore.crps_given("counts", m, n, y)
        rec = {"kind": job["kind"], "route": "direct", "rule": rule, "eval": "score",
               **{k: p[k] for k in lp.PAIR_KEYS}, "x_pid": src, "n_blacklisted": n_black,
               "n_all": int(y.size), "nb_size": n_fit, "n_fit_bins": int(fy.size),
               "depth_ratio": ratio,
               "n_crps_nonfinite": int(np.count_nonzero(~np.isfinite(crps))),
               "crps_all": lscore._finite(float(crps.mean()))}
        rec.update(lscore.split_all("counts", m, n, y, crps))
        for k in range(10):
            idx = np.flatnonzero(dec == k)
            rec[f"n_dec{k}"] = int(idx.size)
            rec[f"crps_dec{k}"] = lscore._finite(float(crps[idx].mean())) if idx.size else None
            if k in (0, _CTX["dec_top"]) and idx.size:
                sp = lscore.split_all("counts", m[idx], n[idx], y[idx], crps[idx])
                sfx = "dec0" if k == 0 else "dectop"
                rec.update({f"{kk.replace('_all', '')}_{sfx}": v for kk, v in sp.items()})
        rec["dectop_k"] = _CTX["dec_top"]
        rec["crps_dectop"] = rec.get(f"crps_dec{_CTX['dec_top']}")
        if job["depth"]:
            sx, sm = float(x.sum()), float(m.sum())
            rec["depth_log2_ratio_true"] = math.log2(ratio)
            rec["depth_log2_scale_pred"] = math.log2(sm / sx) if sx > 0 and sm > 0 else None
        recs.append(rec)
    print(f"pair {job['kind']} {src}->{tgt}: {time.time() - t0:.1f} s", flush=True)
    return recs


def build_jobs(rows, law_stride: int = 1) -> list[dict]:
    """score.py's direct pairs: trained, heldout and law (every `law_stride`-th by index)."""
    def depth_pair(p, kind):
        arms = {p["arm_src"], p["arm_tgt"]}
        return arms == ({"base", "depth"} if kind != "law" else {"depth"})

    jobs = [{"kind": "trained", "pair": p, "depth": depth_pair(p, "trained")}
            for p in T.train_pairs(rows)]
    jobs += [{"kind": "heldout", "pair": p, "depth": depth_pair(p, "heldout")}
             for p in lp.heldout_pairs(rows, T.G, T.HOLDOUT)]
    jobs += [{"kind": "law", "pair": p, "law_index": li, "depth": depth_pair(p, "law")}
             for li, p in enumerate(T.law_pairs(rows)) if li % law_stride == 0]
    return jobs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("run")
    for pos in ("manifest", "data_dir", "blacklist", "out_dir"):
        a.add_argument(pos)
    a.add_argument("--features", required=True)
    a.add_argument("--workers", type=int, default=lscore.default_workers())
    a.add_argument("--law-stride", type=int, default=1)
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    if (out / "SCORE_DONE").exists():
        print(f"{out / 'SCORE_DONE'} exists; nothing to do")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = lp.read_manifest(args.manifest)
    feats = Features(args.features)
    st = json.loads((feats.root / "strata.json").read_text())["strata"]
    _CTX.update(corpus=data.Corpus(args.data_dir), blacklist=str(args.blacklist), feats=feats,
                depth={r["pid"]: float(r["depth"]) for r in rows},
                dec_top=max(int(k) for k, v in st["decile_min_max_n"].items() if v))
    jobs = build_jobs(rows, args.law_stride)
    if args.workers <= 1:
        res = [score_job(j) for j in jobs]
    else:
        with mp.get_context("fork").Pool(args.workers) as pool:
            res = pool.map(score_job, jobs, chunksize=1)
    recs = [r for rs in res for r in rs]
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lscore.write_json({
        "created_utc": stamp,
        "run": {"run_name": "depth_baseline", "rules": list(RULES)},
        "config": {**lscore._config(args.blacklist), "law_stride": args.law_stride,
                   "fit_stride": FIT_STRIDE, "train_chroms": list(TRAIN_CHROMS),
                   "pred_floor": br.PRED_FLOOR, "features_dir": str(feats.root)},
        "records": recs,
        "n_records_crps_nonfinite": sum(r["n_crps_nonfinite"] > 0 for r in recs),
        "timing": {"score": round(time.time() - t0, 3), "n_jobs": len(jobs),
                   "workers": int(args.workers)}}, out / "scores.json")
    (out / "SCORE_DONE").write_text(stamp + "\n")
    print(f"depth baseline: {len(jobs)} pairs x {len(RULES)} rules -> {out / 'scores.json'} "
          f"({time.time() - t0:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
