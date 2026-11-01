"""t146 — score one trained run: direct, through the base (h30) and by mappability decile (h31).

    python tools/t146/score.py run <manifest> <covariates.tsv> <cache_dir> <blacklist.bed> <run_dir>
        --features DIR [--workers N] [--law-stride K] [--mix-stride K] [--mix-draws 16]

**One estimator for every route.** Per (pair, route) the prediction over chr19 + chr21 is an NB
(mean m, size n) per bin; blacklist bins are removed (`ladder.data.read_blacklist_flags`); CRPS =
`ladder.score.crps_given("counts", m, n, y)` = `candi.metrics.nb_crps`, the t118/t128 estimator;
the split = `ladder.score.split_all` (oracle scale c* of the mean, spread held per bin):
crps_all = crps_oracle_scaled_all + scale_error_all. The SAME bins, the SAME pairs and the SAME
function serve every route.

**Routes** (field `route`):
  direct      g(C_src, C_tgt) on the source's counts.
  chain       (h30; arms `tasks.CHAIN_ARMS`) leg 1 = g(C_a, C_base) on x_a; leg 2 = g(C_base, C_b)
              on the leg-1 NB mean (float64, not rounded), as `tools/t128/viabase.py`.
  mix         (h30 second reading) leg 1's NB is sampled `mix_draws` times (rng default_rng([run
              seed, 7148, law pair index, chromosome index])); each draw is fed to leg 2; the 16
              leg-2 NBs form an equal-weight mixture, scored as ONE NB with the mixture's mean and
              variance (moment matching: n = m^2 / (var - m), N_MAX if var <= m) so that the same
              closed-form NB CRPS applies. This is an approximation of the mixture's CRPS.
  ceiling     (h30) per target b: leg 2 fed the REAL base, g(C_base, C_b) on x_base; one record per
              distinct target of the scored law pairs; the tables join it to every pair (a, b).

**Record kinds**: trained (every correction pair), heldout (base<->depth 7.5M), shuffle (every
correction pair, C' = `ladder.pairs.shuffle_target`, the scrambled-covariate reading), law (every
never-trained arm->arm pair, `tasks.law_pairs`, every `law_stride`-th by index for a smoke).
Depth pairs carry depth_log2_ratio_true / depth_log2_scale_pred (log2 sum m / sum x).

**Deciles** (h31): every record carries crps_dec<k>, n_dec<k> for the fixed strata k = 0..9
(`features.py`, dec__<chrom>.npy) and the split of dec0 and dec9 (c* refitted inside the stratum).
Bins of blacklisted or tail 1 kb blocks (decile -1) count in `all` only.

Writes <run_dir>/scores.json and SCORE_DONE.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

_TOOLS = Path(__file__).resolve().parents[1]
for _p in (_TOOLS / "t118", _TOOLS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ladder import data, encoding, pairs as lp  # noqa: E402
from ladder import score as lscore  # noqa: E402
from ladder.model import N_MAX, N_MIN  # noqa: E402
from t146 import tasks as T  # noqa: E402
from t146.features import Features  # noqa: E402
from t146.model import T146Model, mean_of, predict_chrom, size_of  # noqa: E402
from t146.train import full_inputs  # noqa: E402

DEC_SPLIT = (0, 9)
_RUN: dict = {}


class Run:
    """A trained t146 run, ready to predict whole chromosomes."""

    def __init__(self, run_dir, data_dir, features_dir=None, device="cpu"):
        self.run_dir = Path(run_dir)
        ck = torch.load(self.run_dir / "ckpt.pt", map_location="cpu", weights_only=True)
        self.cfg = ck["config"]
        self.step, self.val_nll = int(ck["step"]), float(ck["val_nll"])
        aux = self.cfg["arm_spec"]["aux"]
        self.model = T146Model(inputs=tuple(self.cfg["inputs"]),
                               aux_head=aux in ("xassay", "denoise"))
        self.model.load_state_dict(ck["state"])
        self.model.eval()
        self.dev = torch.device(device)
        self.model.to(self.dev)
        self.enc = encoding.Encoder.from_state(ck["encoder"])
        self.corpus = data.Corpus(data_dir)
        fd = features_dir or self.cfg.get("features_dir")
        self.feats = Features(fd) if fd else None
        self.seed = int(self.cfg["seed"])
        self._ex: dict = {}

    def theta(self, cs, ct):
        cov = torch.from_numpy(self.enc.encode_pair(cs, ct)).reshape(1, -1).to(self.dev)
        with torch.no_grad():
            return self.model.theta(cov)[0]

    def extras(self, chrom, n):
        if chrom not in self._ex:
            f = self.model.form
            self._ex[chrom] = (full_inputs(self.feats, f.scalar, f.has_dna, chrom, n)
                               if (f.scalar or f.has_dna) else (None, None))
        return self._ex[chrom]

    def predict(self, X, cs, ct, chrom):
        ex, dna = self.extras(chrom, len(X))
        return predict_chrom(self.model, X, self.theta(cs, ct), self.dev, extra_full=ex,
                             dna_full=dna)


def route_nb(run: Run, route: str, job: dict, chrom: str, ci: int):
    """(m, n) float64 per bin of one chromosome for one job and route."""
    src, tgt = job["x_pid"], job["y_pid"]
    X = run.corpus.get(src, "counts", chrom)
    if route == "direct":
        loc, disp = run.predict(X, job["cov_src"], job["cov_tgt"], chrom)
        return mean_of(loc), size_of(disp)
    base = job["base"]
    loc1, disp1 = run.predict(X, src, base, chrom)
    if route == "chain":
        loc, disp = run.predict(mean_of(loc1), base, tgt, chrom)
        return mean_of(loc), size_of(disp)
    if route == "mix":
        mu1, n1 = mean_of(loc1), size_of(disp1)
        rng = np.random.default_rng([run.seed, 7148, int(job["law_index"]), ci])
        s1 = np.zeros(X.shape[0])
        s2 = np.zeros(X.shape[0])
        K = int(job["mix_draws"])
        for _ in range(K):
            draw = rng.negative_binomial(n1, n1 / (n1 + mu1)).astype(np.float64)
            loc, disp = run.predict(draw, base, tgt, chrom)
            mu, nn = mean_of(loc), size_of(disp)
            s1 += mu
            s2 += mu + mu * mu / nn + mu * mu
        m = s1 / K
        var = s2 / K - m * m
        with np.errstate(divide="ignore", invalid="ignore"):
            n = np.where(var > m, m * m / (var - m), N_MAX)
        return m, np.clip(n, N_MIN, N_MAX)
    raise ValueError(route)


def score_job(job: dict) -> dict:
    run, blacklist = _RUN["run"], _RUN["blacklist"]
    ms, ns, ys, xs, ds = [], [], [], [], []
    n_black = 0
    for ci, c in enumerate(lp.SCORE_CHROMS):
        y = run.corpus.get(job["y_pid"], "counts", c)
        x = run.corpus.get(job["x_pid"], "counts", c)
        if job["route"] == "ceiling":
            loc, disp = run.predict(x, job["cov_src"], job["cov_tgt"], c)
            m, n = mean_of(loc), size_of(disp)
        else:
            m, n = route_nb(run, job["route"], job, c, ci)
        bad = data.read_blacklist_flags(blacklist, c, y.size)
        keep = ~bad
        n_black += int(bad.sum())
        ms.append(m[keep])
        ns.append(n[keep])
        ys.append(np.asarray(y, np.float64)[keep])
        xs.append(np.asarray(x, np.float64)[keep])
        if run.feats is not None:
            ds.append(np.asarray(run.feats.dec_full(c)[:y.size])[keep])
    m, n, y, x = (np.concatenate(a) for a in (ms, ns, ys, xs))
    crps = lscore.crps_given("counts", m, n, y)
    p = job["pair"]
    rec = {"kind": job["kind"], "route": job["route"], "eval": "score",
           **{k: p[k] for k in lp.PAIR_KEYS}, "cov_src_pid": job["cov_src"],
           "cov_tgt_pid": job["cov_tgt"], "x_pid": job["x_pid"], "n_blacklisted": n_black,
           "n_all": int(y.size), "n_crps_nonfinite": int(np.count_nonzero(~np.isfinite(crps))),
           "crps_all": lscore._finite(float(crps.mean()))}
    rec.update(lscore.split_all("counts", m, n, y, crps))
    if ds:
        dec = np.concatenate(ds)
        for k in range(10):
            idx = np.flatnonzero(dec == k)
            rec[f"n_dec{k}"] = int(idx.size)
            rec[f"crps_dec{k}"] = lscore._finite(float(crps[idx].mean())) if idx.size else None
            if k in DEC_SPLIT and idx.size:
                sp = lscore.split_all("counts", m[idx], n[idx], y[idx], crps[idx])
                rec.update({f"{kk.replace('_all', '')}_dec{k}": v for kk, v in sp.items()})
    if job.get("depth"):
        sx, sm = float(x.sum()), float(m.sum())
        rec["depth_log2_ratio_true"] = math.log2(_RUN["depth"][p["target_pid"]]
                                                 / _RUN["depth"][p["source_pid"]])
        rec["depth_log2_scale_pred"] = math.log2(sm / sx) if sx > 0 and sm > 0 else None
    return rec


def _worker_init():
    torch.set_num_threads(1)


def build_jobs(run: Run, rows, law_stride=1, mix_stride=1, mix_draws=16) -> list[dict]:
    tp = T.train_pairs(rows)
    jobs = []

    def j(kind, route, p, x, cs, ct, **kw):
        jobs.append({"kind": kind, "route": route, "pair": p, "x_pid": x, "y_pid": p["target_pid"],
                     "cov_src": cs, "cov_tgt": ct, **kw})

    def depth_pair(p, kind):
        arms = {p["arm_src"], p["arm_tgt"]}
        return arms == ({"base", "depth"} if kind != "law" else {"depth"})

    for p in tp:
        j("trained", "direct", p, p["source_pid"], p["source_pid"], p["target_pid"],
          depth=depth_pair(p, "trained"))
    for p in lp.heldout_pairs(rows, T.G, T.HOLDOUT):
        j("heldout", "direct", p, p["source_pid"], p["source_pid"], p["target_pid"],
          depth=depth_pair(p, "heldout"))
    for p in tp:
        try:
            wrong = lp.shuffle_target(rows, p, "counts", run.seed)
        except ValueError:
            continue
        j("shuffle", "direct", p, p["source_pid"], p["source_pid"], wrong)
    chain = run.cfg["arm"] in T.CHAIN_ARMS
    targets = {}
    for li, p in enumerate(T.law_pairs(rows)):
        if li % law_stride:
            continue
        base = T.base_pid_of(rows, p["source_pid"])
        kw = {"law_index": li, "base": base, "depth": depth_pair(p, "law")}
        j("law", "direct", p, p["source_pid"], p["source_pid"], p["target_pid"], **kw)
        if chain:
            j("law", "chain", p, p["source_pid"], p["source_pid"], p["target_pid"], **kw)
            if li % (law_stride * mix_stride) == 0:
                j("law", "mix", p, p["source_pid"], p["source_pid"], p["target_pid"],
                  mix_draws=mix_draws, **kw)
            targets.setdefault(p["target_pid"], (base, p))
    for tgt, (base, p) in sorted(targets.items()):
        q = lp._pair({r["pid"]: r for r in lp.usable_products(rows)}[base],
                     {r["pid"]: r for r in lp.usable_products(rows)}[tgt], "base_to_arm")
        j("ceiling", "ceiling", q, base, base, tgt)
    return jobs


def score_run(run: Run, rows, blacklist, workers=1, law_stride=1, mix_stride=1,
              mix_draws=16) -> Path:
    t0 = time.time()
    jobs = build_jobs(run, rows, law_stride, mix_stride, mix_draws)
    # heaviest first so the pool tail is short
    order = sorted(range(len(jobs)), key=lambda i: {"mix": 0, "chain": 1}.get(jobs[i]["route"], 2))
    _RUN.clear()
    _RUN.update(run=run, blacklist=str(blacklist),
                depth={r["pid"]: float(r["depth"]) for r in rows})
    recs = [None] * len(jobs)
    if workers <= 1:
        for i in order:
            recs[i] = score_job(jobs[i])
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(workers, initializer=_worker_init) as pool:
            for i, r in zip(order, pool.imap(score_job, [jobs[i] for i in order], chunksize=1)):
                recs[i] = r
    t1 = time.time()
    out = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "run": {"run_name": run.cfg["run_name"], "arm": run.cfg["arm"],
                   "weight": run.cfg["weight"], "seed": run.seed, "best_step": run.step,
                   "val_nll": run.val_nll, "inputs": run.cfg["inputs"]},
           "config": {**lscore._config(blacklist), "law_stride": law_stride,
                      "mix_stride": mix_stride, "mix_draws": mix_draws,
                      "features_dir": None if run.feats is None else str(run.feats.root)},
           "records": recs,
           "n_records_crps_nonfinite": sum(r["n_crps_nonfinite"] > 0 for r in recs),
           "timing": {"score": round(t1 - t0, 3), "n_jobs": len(jobs), "workers": int(workers)}}
    lscore.write_json(out, run.run_dir / "scores.json")
    (run.run_dir / "SCORE_DONE").write_text(out["created_utc"] + "\n")
    return run.run_dir / "scores.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("run")
    for pos in ("manifest", "covariates", "data_dir", "blacklist", "run_dir"):
        a.add_argument(pos)
    a.add_argument("--features", default=None)
    a.add_argument("--workers", type=int, default=lscore.default_workers())
    a.add_argument("--law-stride", type=int, default=1)
    a.add_argument("--mix-stride", type=int, default=1)
    a.add_argument("--mix-draws", type=int, default=16)
    args = ap.parse_args(argv)
    done = Path(args.run_dir) / "SCORE_DONE"
    if done.exists():
        print(f"{done} exists; nothing to do")
        return 0
    if not (Path(args.run_dir) / "TRAIN_DONE").is_file():
        print(f"no TRAIN_DONE in {args.run_dir}", file=sys.stderr)
        return 3
    t0 = time.time()
    run = Run(args.run_dir, args.data_dir, args.features)
    rows = lp.read_manifest(args.manifest)
    path = score_run(run, rows, args.blacklist, args.workers, args.law_stride, args.mix_stride,
                     args.mix_draws)
    print(f"scored {run.cfg['run_name']} -> {path} ({time.time() - t0:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
