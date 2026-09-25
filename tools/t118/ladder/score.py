"""t118 ladder — score one trained run from its checkpoint: trained pairs, shuffle, swap, depth law; and
the never-trained arm->arm law test in a separate pass.

**Distributions** (the `base` parameterisation, per bin): counts NB with mean mu = `pred.mean(loc)`
(exp(loc) clamped to the loss's [1e-4, 1e6]) and size n = exp(disp) clipped to the loss's
[1e-3, 1e4]; CRPS = `candi.metrics.nb_crps(n, p_from_mu(n, mu), y)`; pval log-normal with log
median loc and sigma = exp(disp) clipped to [1e-3, 1e3], CRPS = `baseline_rungs.lognormal_crps`.
Bins whose CRPS is still not finite are counted per record (`n_crps_nonfinite`) and per file
(`n_records_crps_nonfinite`), never dropped silently. Targets are never floored.
ONE per-bin CRPS array per (pair, eval) serves all three subsets (`baseline_rungs.subset_index`:
all / nonzero / top1, top-1 % ties broken by genomic order). Spearman is on pred.mean(loc) (the NB mean
in counts, the log-normal median in pval). The CRPS split (`baseline_rungs.scale_split`, the
spread parameter held per bin) is computed only for kind `trained`, eval `score`, subset `all`.

**Evaluation sets.** score = chr19 + chr21, val = chr22, blacklist bins (`data.read_blacklist_flags`)
removed; chromosomes concatenated in that order.

**Record kinds.**
  trained   every training pair, eval score and val; covariates (C_src, C_tgt).
  shuffle   every training pair, eval score; C' = the covariates of `pairs.shuffle_target(rows,
            pair, space, run seed)` (a same-track product whose target differs in this space).
  swap      one per fit pid p, eval score: X = p, target = p, covariates (C_p, C_p); carries
            `swap_median_abs_log_ratio` = median over bins with X > 0 of |log(pred.mean(loc) / X)|.
  depthlaw  counts only: a copy of the trained/score (or law) record of every pair whose two
            products differ only in depth (base<->depth trained, depth->depth law), with
            `depth_log2_ratio_true = log2(depth_tgt / depth_src)` (manifest depths),
            `depth_log2_scale_pred = log2(sum mu / sum X)` over the score bins, and
            `depth_source` in {trained, law}.
  law       (law pass) every never-trained arm->arm pair of `pairs.law_pairs(rows, g)`, eval score.

`pit_hist` (trained/score, every model): int[20] histogram of the randomised PIT — NB
u = F(y-1) + v (F(y) - F(y-1)), v ~ U(0, 1) from default_rng(0); log-normal Phi((log y - loc)/sigma),
0 at y = 0.

`figdata.npz` (real model only; written empty for the twins): `meta__<src>__<tgt>` float32[3, 161]
= rows X, X', mu, the mean over the 500 highest-X' score bins of the +-80-bin window (zero-padded
past a chromosome end); `snip__<arm>__<i>` float32[5, 400] = rows X, X', mu, q05, q95 over the 400
bins centred on bin i of the rule: 0 the top-1 % bin with the largest |X' - X|, 1 a random top-1 %
bin, 2 a random non-top-1 % bin (rng `default_rng([seed, 7120, arm_index])`), on the base->arm pair
of the arm's first level in sorted order. `arm_index` is the arm's position among the run's sorted
arm names; under the across-track g the pair is the first track's (sorted) that has the arm.

    python tools/t118/ladder/score.py trained <manifest> <covariates.tsv> <data_dir> <blacklist.bed> <run_dir> [--workers N]
    python tools/t118/ladder/score.py law     <manifest> <covariates.tsv> <data_dir> <blacklist.bed> <run_dir> [--workers N]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import nbinom, norm

_T118 = Path(__file__).resolve().parents[1]
if str(_T118) not in sys.path:          # script mode: make `baseline_rungs` and `ladder` importable
    sys.path.insert(0, str(_T118))

import baseline_rungs as br  # noqa: E402
from candi.bench.distributional import p_from_mu  # noqa: E402
from candi.metrics import nb_crps, spearman  # noqa: E402
from ladder import data, pairs  # noqa: E402
from ladder.model import N_MAX, N_MIN, SIGMA_MAX, SIGMA_MIN  # noqa: E402  (the loss's bounds)

SUBSETS = br.SUBSETS                     # ("all", "nonzero", "top1")
METRICS = tuple(f"{m}_{s}" for m in ("crps", "spearman") for s in SUBSETS)
PIT_BINS = 20
META_TOP = 500
META_HALF = 80
SNIP_LEN = 400
SNIP_RULES = ("top1_max_abs_diff", "top1_random", "background_random")
SWAP_KEY = "swap_median_abs_log_ratio"

#: module-level state the forked pool workers read (set before the pool is created)
_CTX: dict = {}


# ---------------------------------------------------------------------------------------------
# per-bin quantities
# ---------------------------------------------------------------------------------------------


def crps_given(space: str, m: np.ndarray, spread: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-bin CRPS. counts: NB(mean m, size spread); pval: log-normal(median m, sigma spread)."""
    if space == "counts":
        return nb_crps(spread, p_from_mu(spread, m), y)
    return br.lognormal_crps(m, spread, y)


def _finite(v):
    return float(v) if v is not None and np.isfinite(v) else None


def split_all(space: str, m: np.ndarray, spread: np.ndarray, y: np.ndarray,
              crps_bins: np.ndarray) -> dict:
    """`baseline_rungs.scale_split` with the spread held PER BIN.

    `scale_split` calls `crps_fn(mm, yy)` on the full input and on its seeded `fit_budget`
    subsample; the closure hands the matching spread slice by recomputing the same subsample
    (`default_rng(SCALE_SEED).choice(N, SCALE_FIT_BUDGET)`), and returns the already-computed
    per-bin CRPS for the unscaled call so the full array is not evaluated twice.
    """
    n_bins = m.size
    sel = (None if n_bins <= br.SCALE_FIT_BUDGET else
           np.random.default_rng(br.SCALE_SEED).choice(n_bins, br.SCALE_FIT_BUDGET, replace=False))

    def fn(mm, yy):
        if mm is m:
            return crps_bins
        s = spread if (sel is None or mm.shape[0] == n_bins) else spread[sel]
        return crps_given(space, mm, s, yy)

    r = br.scale_split(m, y, fn)
    return {"crps_oracle_scaled_all": r["crps_oracle_scaled"], "scale_error_all": r["scale_error"],
            "c_star_all": r["c_star"]}


def spread_of(space: str, disp: np.ndarray) -> np.ndarray:
    """NB size n (counts) / sigma (pval) = exp(disp), clipped to the training loss's bounds."""
    lo, hi = (N_MIN, N_MAX) if space == "counts" else (SIGMA_MIN, SIGMA_MAX)
    return np.exp(np.clip(disp, math.log(lo), math.log(hi)))


def metric_fields(pred, loc: np.ndarray, disp: np.ndarray, y: np.ndarray,
                  split: bool = False) -> tuple[dict, dict]:
    """(fields, internals): n_* / crps_* / spearman_* per subset (+ the split); internals hold the
    per-bin arrays and subset indices for reuse. The mean is the Predictor's own (`pred.mean`,
    clamped as in the loss); the spread is clipped as in the loss (`spread_of`).
    `n_crps_nonfinite` counts the bins whose CRPS is still not finite (their subset means are then
    None, and the count says why)."""
    space = pred.space
    m = np.asarray(pred.mean(loc), np.float64)
    spread = spread_of(space, disp)
    y64 = y.astype(np.float64)
    crps = crps_given(space, m, spread, y64)
    sub = br.subset_index(y)
    out: dict = {"n_crps_nonfinite": int(np.count_nonzero(~np.isfinite(crps)))}
    for s, idx in sub.items():
        out[f"n_{s}"] = int(idx.size)
        if idx.size == 0:
            out[f"crps_{s}"] = None
            out[f"spearman_{s}"] = None
            continue
        out[f"crps_{s}"] = _finite(float(crps[idx].mean()))
        out[f"spearman_{s}"] = _finite(spearman(m[idx], y64[idx]))
    if split:
        out.update(split_all(space, m, spread, y64, crps) if y.size else
                   {"crps_oracle_scaled_all": None, "scale_error_all": None, "c_star_all": None})
    return out, {"m": m, "spread": spread, "y64": y64, "subsets": sub}


def pit_hist(space: str, m: np.ndarray, spread: np.ndarray, y64: np.ndarray, loc: np.ndarray) -> list:
    """int[20]: randomised PIT (NB, v ~ U(0,1) seeded 0) / continuous PIT (log-normal, 0 at y = 0)."""
    if space == "counts":
        p = p_from_mu(spread, m)
        f_hi = nbinom.cdf(y64, spread, p)
        f_lo = nbinom.cdf(y64 - 1.0, spread, p)
        v = np.random.default_rng(0).random(y64.size)
        u = f_lo + v * (f_hi - f_lo)
    else:
        with np.errstate(divide="ignore"):
            z = (np.log(y64) - loc) / spread
        u = np.where(y64 > 0, norm.cdf(z), 0.0)
    h, _ = np.histogram(np.clip(u, 0.0, 1.0), bins=PIT_BINS, range=(0.0, 1.0))
    return [int(v) for v in h]


def swap_ratio(m: np.ndarray, x: np.ndarray):
    """median over bins with X > 0 of |log(m / X)|."""
    pos = x > 0
    if not pos.any():
        return None
    return float(np.median(np.abs(np.log(m[pos] / x[pos].astype(np.float64)))))


# ---------------------------------------------------------------------------------------------
# one (pair, eval) job
# ---------------------------------------------------------------------------------------------


def _chroms_of(pred, pid: str, space: str, ev: str) -> list[str]:
    want = pairs.SCORE_CHROMS if ev == "score" else pairs.VAL_CHROMS
    have = set(pred.corpus.chroms(pid, space))
    missing = [c for c in want if c not in have]
    if missing:
        raise ValueError(f"{pid}/{space}: no {missing}")
    return list(want)


def _eval_arrays(pred, space, x_pid, y_pid, cov_src, cov_tgt, chroms, blacklist, keep_full):
    xs, ys, locs, disps, cid, pos = [], [], [], [], [], []
    full, n_black = {}, 0
    for ci, c in enumerate(chroms):
        x = pred.corpus.get(x_pid, space, c)
        y = pred.corpus.get(y_pid, space, c)
        if x.shape != y.shape:
            raise ValueError(f"{c}: source {x.shape} and target {y.shape} differ in length")
        loc, disp = pred.predict(x_pid, cov_src, cov_tgt, c)
        loc = np.asarray(loc, np.float64)
        disp = np.asarray(disp, np.float64)
        if loc.shape != x.shape or disp.shape != x.shape:
            raise ValueError(f"{c}: prediction {loc.shape}/{disp.shape} != chromosome {x.shape}")
        bad = data.read_blacklist_flags(blacklist, c, x.size)
        keep = ~bad
        n_black += int(bad.sum())
        xs.append(x[keep])
        ys.append(y[keep])
        locs.append(loc[keep])
        disps.append(disp[keep])
        idx = np.flatnonzero(keep)
        cid.append(np.full(idx.size, ci, np.int64))
        pos.append(idx)
        if keep_full:
            full[c] = (x, y, loc, disp)
    cat = np.concatenate
    return {"x": cat(xs), "y": cat(ys), "loc": cat(locs), "disp": cat(disps), "cid": cat(cid),
            "pos": cat(pos), "chroms": chroms, "n_blacklisted": n_black, "full": full}


def _window(a: np.ndarray, start: int, length: int) -> np.ndarray:
    out = np.zeros(length, np.float64)
    s, e = max(start, 0), min(start + length, a.shape[0])
    if e > s:
        out[s - start:e - start] = a[s:e]
    return out


def _meta_profile(arr: dict, internals: dict) -> np.ndarray:
    y = arr["y"]
    k = min(META_TOP, y.size)
    top = np.argsort(-y.astype(np.float64), kind="stable")[:k]
    acc = np.zeros((3, 2 * META_HALF + 1), np.float64)
    for i in top:
        c = arr["chroms"][arr["cid"][i]]
        x, yy, loc, _ = arr["full"][c]
        st = int(arr["pos"][i]) - META_HALF
        acc[0] += _window(x, st, acc.shape[1])
        acc[1] += _window(yy, st, acc.shape[1])
        acc[2] += _window(internals["mean_full"](loc), st, acc.shape[1])
    return (acc / max(k, 1)).astype(np.float32)


def _snippets(pred, arr: dict, internals: dict, arm: str, arm_index: int, seed: int,
              pair: dict) -> tuple[dict, list]:
    y = arr["y"]
    if y.size == 0:
        return {}, []
    top = internals["subsets"]["top1"]
    rest = np.setdiff1d(np.arange(y.size), top, assume_unique=True)
    rng = np.random.default_rng([int(seed), 7120, int(arm_index)])
    diff = np.abs(y.astype(np.float64)[top] - arr["x"].astype(np.float64)[top])
    picks = [int(top[int(np.argmax(diff))]), int(top[int(rng.integers(top.size))]),
             int(rest[int(rng.integers(rest.size))]) if rest.size else None]
    figs, locs = {}, []
    for i, b in enumerate(picks):
        if b is None:
            continue
        c = arr["chroms"][arr["cid"][b]]
        x, yy, loc, disp = arr["full"][c]
        st = int(arr["pos"][b]) - SNIP_LEN // 2
        s, e = max(st, 0), min(st + SNIP_LEN, x.shape[0])
        q = np.zeros((2, SNIP_LEN), np.float64)
        for j, qq in enumerate((0.05, 0.95)):
            q[j, s - st:e - st] = np.asarray(pred.quantiles(loc[s:e], disp[s:e], qq), np.float64)
        figs[f"snip__{arm}__{i}"] = np.stack([
            _window(x, st, SNIP_LEN), _window(yy, st, SNIP_LEN),
            _window(internals["mean_full"](loc), st, SNIP_LEN), q[0], q[1]]).astype(np.float32)
        locs.append({"arm": arm, "i": i, "rule": SNIP_RULES[i], "chrom": c, "start_bin": st,
                     "source_pid": pair["source_pid"], "target_pid": pair["target_pid"]})
    return figs, locs


def _pair_fields(p: dict) -> dict:
    return {k: p[k] for k in pairs.PAIR_KEYS}


def _is_depth_pair(p: dict, kind: str) -> bool:
    arms = {p["arm_src"], p["arm_tgt"]}
    return arms == {"base", "depth"} if kind == "trained" else arms == {"depth"}


def score_job(job: dict) -> tuple[list, dict, list]:
    """One (pair, eval, covariate pair) -> (records, figdata arrays, snippet loci)."""
    pred, blacklist, depth = _CTX["pred"], _CTX["blacklist"], _CTX["depth"]
    space = pred.space
    p = job["pair"]
    ev = job["eval"]
    chroms = _chroms_of(pred, job["x_pid"], space, ev)
    keep_full = bool(job.get("meta") or job.get("snip"))
    arr = _eval_arrays(pred, space, job["x_pid"], job["y_pid"], job["cov_src"], job["cov_tgt"],
                       chroms, blacklist, keep_full)
    rec = {"kind": job["kind"], "eval": ev, **_pair_fields(p),
           "cov_src_pid": job["cov_src"], "cov_tgt_pid": job["cov_tgt"],
           "n_blacklisted": arr["n_blacklisted"]}
    fields, internals = metric_fields(pred, arr["loc"], arr["disp"], arr["y"],
                                      split=bool(job.get("split")))
    internals["mean_full"] = lambda a: np.asarray(pred.mean(a), np.float64)
    rec.update(fields)
    if job.get("describe"):
        rec["describe"] = pred.describe(job["cov_src"], job["cov_tgt"])
    if job.get("pit"):
        rec["pit_hist"] = pit_hist(space, internals["m"], internals["spread"], internals["y64"],
                                   arr["loc"])
    if job["kind"] == "swap":
        rec[SWAP_KEY] = swap_ratio(internals["m"], arr["x"])
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
# run level
# ---------------------------------------------------------------------------------------------


def g_version(g: str) -> str:
    return "across" if g == "all" else "per_track"


def _train_log(run_dir: Path) -> dict:
    path = Path(run_dir) / "train_log.tsv"
    if not path.is_file():
        return {}
    steps, best = [], (None, None)
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            try:
                st = int(float(r.get("step", "")))
            except ValueError:
                continue
            steps.append(st)
            try:
                v = float(r.get("val_nll", ""))
            except ValueError:
                continue
            if np.isfinite(v) and (best[1] is None or v < best[1]):
                best = (st, v)
    return {"steps": max(steps) if steps else None, "best_step": best[0], "val_nll": best[1]}


def run_info(pred, run_dir) -> dict:
    run_dir = Path(run_dir)
    log = _train_log(run_dir)
    info = {"run_name": run_dir.name, "rung": pred.rung, "g": pred.g_id,
            "g_version": g_version(pred.g_id), "space": pred.space, "model": pred.model,
            "seed": int(pred.seed), "steps": log.get("steps"), "best_step": log.get("best_step"),
            "val_nll": log.get("val_nll")}
    for k in ("steps", "best_step", "val_nll"):
        v = getattr(pred, k, None)
        if v is not None:
            info[k] = v
    return info


def _run_jobs(jobs: list[dict], workers: int) -> list:
    if workers <= 1 or len(jobs) <= 1:
        return [score_job(j) for j in jobs]
    ctx = mp.get_context("fork")
    with ctx.Pool(min(workers, len(jobs)), initializer=_worker_init) as pool:
        return pool.map(score_job, jobs, chunksize=1)


def _worker_init():
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.set_num_threads(1)


def _setup(pred, rows, blacklist):
    _CTX.clear()
    _CTX.update(pred=pred, blacklist=str(blacklist),
                depth={r["pid"]: float(r["depth"]) for r in rows})


def _config(blacklist) -> dict:
    return {"score_chroms": list(pairs.SCORE_CHROMS), "val_chroms": list(pairs.VAL_CHROMS),
            "blacklist": str(blacklist),
            "blacklist_sha256": hashlib.sha256(Path(blacklist).read_bytes()).hexdigest(),
            "counts": "NB(mean pred.mean(loc), size exp(disp) clipped to [N_MIN, N_MAX]); "
                      "nb_crps(n, p_from_mu(n, mu), y)",
            "pval": "log-normal(log median loc, sigma exp(disp) clipped to [SIGMA_MIN, "
                    "SIGMA_MAX]); baseline_rungs.lognormal_crps",
            "subsets": "baseline_rungs.subset_index", "spearman_on": "pred.mean(loc)",
            "split": "baseline_rungs.scale_split, spread held per bin; trained/score/all only"}


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return _finite(float(o))
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def _clean(v):
    """NaN / inf -> None, recursively (the JSON stays strict)."""
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


def write_json(obj, path: Path, indent=1) -> None:
    """Strict JSON (NaN -> null), tmp + replace; `indent=None` writes compact JSON."""
    tmp = Path(str(path) + ".tmp")
    sep = None if indent is not None else (",", ":")
    tmp.write_text(json.dumps(_clean(obj), indent=indent, separators=sep,
                              default=_json_default) + "\n", "utf-8")
    os.replace(tmp, path)


def _snip_plan(train: list[dict]) -> dict:
    """(source_pid, target_pid) -> [(arm, arm_index)]: the base->arm pair of each arm's first
    level in sorted order (first track in sorted order under the across-track g)."""
    b2a = [p for p in train if p["direction"] == "base_to_arm"]
    arms = sorted({p["arm_tgt"] for p in b2a})
    plan: dict = {}
    for ai, arm in enumerate(arms):
        p = min((q for q in b2a if q["arm_tgt"] == arm), key=lambda q: (q["track"], q["target_pid"]))
        plan.setdefault((p["source_pid"], p["target_pid"]), []).append((arm, ai))
    return plan


def score_trained(pred, rows: list[dict], blacklist, run_dir, workers: int = 1) -> Path:
    """trained (score + val), shuffle, swap and depth-law records -> scores.json, figdata.npz."""
    t0 = time.time()
    run_dir = Path(run_dir)
    _setup(pred, rows, blacklist)
    space, real = pred.space, pred.model == "real"
    train = pairs.train_pairs(rows, pred.g_id)
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
                         "depth": sc and space == "counts" and _is_depth_pair(p, "trained")})
    for p in train:
        try:
            wrong = pairs.shuffle_target(rows, p, space, pred.seed)
        except ValueError as e:
            skipped.append({"kind": "shuffle", "source_pid": p["source_pid"],
                            "target_pid": p["target_pid"], "reason": str(e)})
            continue
        jobs.append({"kind": "shuffle", "eval": "score", "pair": p, "x_pid": p["source_pid"],
                     "y_pid": p["target_pid"], "cov_src": p["source_pid"], "cov_tgt": wrong})
    by_pid = {r["pid"]: r for r in pairs.usable_products(rows)}
    for pid in pairs.fit_pids(rows, pred.g_id):
        r = by_pid[pid]
        p = pairs._pair(r, r, "swap")
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
    out = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "run": run_info(pred, run_dir), "config": _config(blacklist), "records": records,
           "snippets": snippets, "skipped": skipped,
           "n_records_crps_nonfinite": sum(r.get("n_crps_nonfinite", 0) > 0 for r in records),
           "timing": {"setup": round(t1 - t0, 3), "score": round(t2 - t1, 3),
                      "total": round(time.time() - t0, 3), "n_jobs": len(jobs),
                      "workers": int(workers)}}
    write_json(out, run_dir / "scores.json")
    (run_dir / "SCORE_DONE").write_text(out["created_utc"] + "\n")
    return run_dir / "scores.json"


def score_law(pred, rows: list[dict], blacklist, run_dir, workers: int = 1) -> Path:
    """Every never-trained arm->arm pair of the run's g (+ depth->depth depth-law records)."""
    t0 = time.time()
    run_dir = Path(run_dir)
    _setup(pred, rows, blacklist)
    jobs = [{"kind": "law", "eval": "score", "pair": p, "x_pid": p["source_pid"],
             "y_pid": p["target_pid"], "cov_src": p["source_pid"], "cov_tgt": p["target_pid"],
             "depth": pred.space == "counts" and _is_depth_pair(p, "law")}
            for p in pairs.law_pairs(rows, pred.g_id)]
    t1 = time.time()
    records = [r for recs, _, _ in _run_jobs(jobs, workers) for r in recs]
    t2 = time.time()
    out = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "run": run_info(pred, run_dir), "config": _config(blacklist), "records": records,
           "n_records_crps_nonfinite": sum(r.get("n_crps_nonfinite", 0) > 0 for r in records),
           "timing": {"setup": round(t1 - t0, 3), "score": round(t2 - t1, 3),
                      "total": round(time.time() - t0, 3), "n_jobs": len(jobs),
                      "workers": int(workers)}}
    write_json(out, run_dir / "law.json")
    (run_dir / "LAW_DONE").write_text(out["created_utc"] + "\n")
    return run_dir / "law.json"


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def default_workers() -> int:
    """The CPUs this process may run on (the SLURM allocation), else the machine's count."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:          # macOS has no sched_getaffinity
        return os.cpu_count() or 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("trained", "law"):
        a = sub.add_parser(name)
        for pos in ("manifest", "covariates", "data_dir", "blacklist", "run_dir"):
            a.add_argument(pos, type=Path)
        a.add_argument("--workers", type=int, default=default_workers())
    args = ap.parse_args(argv)
    done = args.run_dir / ("SCORE_DONE" if args.cmd == "trained" else "LAW_DONE")
    if done.exists():
        print(f"{done} exists; nothing to do")
        return 0
    from ladder import model  # the checkpoint loader (t118-C1); imported only here
    t0 = time.time()
    pred = model.load_run(args.run_dir, args.data_dir, args.covariates, args.manifest, device="cpu")
    rows = pairs.read_manifest(args.manifest)
    fn = score_trained if args.cmd == "trained" else score_law
    path = fn(pred, rows, args.blacklist, args.run_dir, workers=args.workers)
    print(f"{args.cmd} {args.run_dir.name} -> {path} ({time.time() - t0:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
