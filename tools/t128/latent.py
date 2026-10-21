"""t128 — the latent pass of one trained DS / DSBOX run: readings 4-6 of plan/T126_ENCDEC_D2.md.

Z is deterministic, Z = mu_z = `form.encode(x, theta_e)[0]`, with theta_e = g(c, C_clean) of the
product's own covariates (`theta_e_of`). Evaluation bins: chr19 + chr21 (`pairs.SCORE_CHROMS`),
blacklist bins removed (`data.read_blacklist_flags`), chromosomes concatenated in that order, as
`score._eval_arrays`. x = log1p(X), the encoder's own input (`model.transform_x`).

**Record kinds** (`latent.json["records"]`).
  latent  one per usable product of the run's g tracks (7.5M included): median |Z - x|,
          Pearson corr(Z, x), mean Z, mean x, median sigma_z (reading 4, collapse).
  agree   one per unordered pair (a < b by pid) of those products within a track: median
          |Z_a - Z_b| against median |x_a - x_b|, and corr(Z_a, Z_b) (reading 5).
  ident   one per trained, held-out and never-trained (law) pair: the distance of the cross map
          from the identity, median over bins with X > 0 of |log(mu / X)| with mu the NB mean of
          `pred.predict(source, source, target)` and X the SOURCE counts (`score.swap_ratio`).
          Reading 6 plots the swap check (scores.json, kind swap) against it.

Z of each product is computed once per track job and shared by its `agree` records.

    python tools/t128/latent.py run <manifest> <covariates.tsv> <data_dir> <blacklist.bed> <run_dir> [--workers N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

_T118 = Path(__file__).resolve().parents[1] / "t118"
if str(_T118) not in sys.path:          # import convention: tools/t118 holds `ladder`
    sys.path.insert(0, str(_T118))

from ladder import data, model, pairs, score  # noqa: E402
from ladder.fforms import form_ds  # noqa: E402

RUNGS = ("DS", "DSBOX")
CHUNK = model.CHUNK
#: the encoder's receptive field each side: ONE f (old D), not the full map's 2 * 30
HALO = form_ds.CTX1
IDENT_KEY = "identity_median_abs_log_ratio"

#: module-level state the forked pool workers read (set before the pool is created)
_CTX: dict = {}


# ---------------------------------------------------------------------------------------------
# Z over a chromosome
# ---------------------------------------------------------------------------------------------


@torch.no_grad()
def encode_chrom(ladder, X: np.ndarray, theta_e: torch.Tensor, device,
                 chunk: int = CHUNK) -> tuple[np.ndarray, np.ndarray]:
    """(mu_z, log_sigma_z) float32[n] over a whole chromosome X (counts) of one product.

    Mirrors `model.predict_chrom`: chunks of `chunk` bins, each with a zero-padded halo of
    `HALO` (30) bins, outside the chromosome X = 0; batches of `model.BATCH_CHUNKS` chunks.
    """
    if ladder.rung not in RUNGS:
        raise ValueError(f"rung {ladder.rung!r} has no encoder (want one of {RUNGS})")
    h = HALO
    n = int(np.asarray(X).shape[0])
    chunk = max(1, min(int(chunk), n))
    n_chunks = max(1, -(-n // chunk))
    Xp = np.zeros(n_chunks * chunk + 2 * h, dtype=np.float64)
    Xp[h:h + n] = X
    xp = model.transform_x(Xp, ladder.space)
    theta = theta_e.reshape(1, -1).to(device=device, dtype=torch.float32)
    mus, lss = [], []
    for b0 in range(0, n_chunks, model.BATCH_CHUNKS):
        idx = range(b0, min(b0 + model.BATCH_CHUNKS, n_chunks))
        xb = np.stack([xp[i * chunk: i * chunk + chunk + 2 * h] for i in idx])
        xt = torch.from_numpy(xb).to(device)
        mu, ls = ladder.form.encode(xt, theta.expand(xt.shape[0], -1))
        mus.append(mu.reshape(-1).float().cpu().numpy())
        lss.append(ls.reshape(-1).float().cpu().numpy())
    return (np.concatenate(mus)[:n].astype(np.float32),
            np.concatenate(lss)[:n].astype(np.float32))


def theta_e_of(pred, pid: str) -> torch.Tensor:
    """theta_e [256] = g(c, C_clean) of pid: the first half of theta(c, c). The nocov twin's
    theta is one mean over its training pairs, so this is that mean's first half."""
    return pred._theta_t(pid, pid)[:form_ds.FILM]


# ---------------------------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------------------------


def _pearson(a: np.ndarray, b: np.ndarray):
    if a.size < 2:
        return None
    a = a.astype(np.float64) - a.mean(dtype=np.float64)
    b = b.astype(np.float64) - b.mean(dtype=np.float64)
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else None


def _median(v: np.ndarray):
    return float(np.median(v)) if v.size else None


def _z_of(pred, pid: str, blacklist: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(Z, log sigma_z, x) on the evaluation bins of pid, float32."""
    chroms = score._chroms_of(pred, pid, pred.space, "score")
    theta_e = theta_e_of(pred, pid)
    zs, lss, xs = [], [], []
    for c in chroms:
        X = pred.corpus.get(pid, pred.space, c)
        mu, ls = encode_chrom(pred.ladder, X, theta_e, pred.device)
        keep = ~data.read_blacklist_flags(blacklist, c, X.size)
        zs.append(mu[keep])
        lss.append(ls[keep])
        xs.append(model.transform_x(X, pred.space)[keep])
    return np.concatenate(zs), np.concatenate(lss), np.concatenate(xs)


def _track_job(track: str) -> list[dict]:
    """latent records of every product of the track, then its agree records."""
    pred, blacklist, prods = _CTX["pred"], _CTX["blacklist"], _CTX["products"][track]
    zs, xs, out = {}, {}, []
    for r in prods:
        z, ls, x = _z_of(pred, r["pid"], blacklist)
        zs[r["pid"]], xs[r["pid"]] = z, x
        d = z.astype(np.float64) - x.astype(np.float64)
        out.append({"kind": "latent", "pid": r["pid"], "track": track, "arm": r["arm"],
                    "level": r["level"], "n_bins": int(z.size),
                    "median_abs_z_minus_x": _median(np.abs(d)), "corr_z_x": _pearson(z, x),
                    "mean_z": float(z.mean(dtype=np.float64)) if z.size else None,
                    "mean_x": float(x.mean(dtype=np.float64)) if x.size else None,
                    "median_sigma_z": _median(np.exp(ls.astype(np.float64)))})
    for i, a in enumerate(prods):
        for b in prods[i + 1:]:
            za, zb, xa, xb = zs[a["pid"]], zs[b["pid"]], xs[a["pid"]], xs[b["pid"]]
            if za.shape != zb.shape:
                raise ValueError(f"{a['pid']} / {b['pid']}: {za.size} and {zb.size} bins")
            out.append({"kind": "agree", "track": track, "pid_a": a["pid"], "pid_b": b["pid"],
                        "arm_a": a["arm"], "level_a": a["level"],
                        "arm_b": b["arm"], "level_b": b["level"], "n_bins": int(za.size),
                        "median_abs_z_diff": _median(np.abs(za.astype(np.float64) - zb)),
                        "median_abs_x_diff": _median(np.abs(xa.astype(np.float64) - xb)),
                        "corr_z_a_b": _pearson(za, zb)})
    return out


def _ident_job(group: str, p: dict) -> list[dict]:
    pred, blacklist = _CTX["pred"], _CTX["blacklist"]
    src, tgt = p["source_pid"], p["target_pid"]
    chroms = score._chroms_of(pred, src, pred.space, "score")
    arr = score._eval_arrays(pred, pred.space, src, tgt, src, tgt, chroms, blacklist, False)
    m = np.asarray(pred.mean(arr["loc"]), np.float64)
    return [{"kind": "ident", "group": group, **score._pair_fields(p),
             "n_bins": int(arr["x"].size), "n_blacklisted": arr["n_blacklisted"],
             IDENT_KEY: score.swap_ratio(m, arr["x"])}]


def _job(job: tuple) -> list[dict]:
    if job[0] == "track":
        return _track_job(job[1])
    return _ident_job(job[1], job[2])


def _run_jobs(jobs: list[tuple], workers: int) -> list[list[dict]]:
    """As `score._run_jobs`: fork pool, module-level state, one torch thread per worker."""
    if workers <= 1 or len(jobs) <= 1:
        return [_job(j) for j in jobs]
    ctx = mp.get_context("fork")
    with ctx.Pool(min(workers, len(jobs)), initializer=score._worker_init) as pool:
        return pool.map(_job, jobs, chunksize=1)


# ---------------------------------------------------------------------------------------------
# run level
# ---------------------------------------------------------------------------------------------


def _config(blacklist) -> dict:
    return {"chroms": list(pairs.SCORE_CHROMS), "blacklist": str(blacklist),
            "blacklist_md5": hashlib.md5(Path(blacklist).read_bytes()).hexdigest(),
            "z_rule": "deterministic Z = mu_z = form.encode(log1p(X), theta_e)[0], theta_e = "
                      "g(c, C_clean) of the product's own c (first half of theta(c, c))",
            "x_rule": "x = log1p(X) (model.transform_x)",
            "bins_rule": "score chroms concatenated, blacklist bins removed "
                         "(data.read_blacklist_flags)",
            "latent": "median |Z - x|, Pearson corr(Z, x), mean Z, mean x, median exp(log sigma_z)",
            "agree": "unordered same-track product pairs (a < b by pid): median |Z_a - Z_b|, "
                     "median |x_a - x_b|, Pearson corr(Z_a, Z_b)",
            "ident": "median over bins with X > 0 of |log(mu / X)|, mu = pred.mean(loc) of "
                     "pred.predict(source, source, target), X = source counts (score.swap_ratio)",
            "halo": f"{HALO} zero-padded bins per chunk side (one f's receptive field)"}


def run_latent(pred, rows: list[dict], blacklist, run_dir, workers: int = 1) -> Path:
    """latent, agree and ident records -> `<run_dir>/latent.json`, then `LATENT_DONE`."""
    if pred.rung not in RUNGS:
        raise ValueError(f"rung {pred.rung!r}: the latent pass is for {RUNGS} only")
    t0 = time.time()
    run_dir = Path(run_dir)
    g = pred.g_id
    holdout = [tuple(h) for h in (pred.config or {}).get("holdout", ())]
    tracks = pairs._tracks_of(rows, g)
    prods = {t: [r for r in pairs.usable_products(rows) if r["track"] == t] for t in tracks}
    _CTX.clear()
    _CTX.update(pred=pred, blacklist=str(blacklist), products=prods)
    groups = (("trained", pairs.train_pairs(rows, g, holdout)),
              ("heldout", pairs.heldout_pairs(rows, g, holdout)),
              ("law", pairs.law_pairs(rows, g)))
    jobs = [("track", t) for t in tracks] + [("ident", grp, p) for grp, ps in groups for p in ps]
    t1 = time.time()
    results = _run_jobs(jobs, workers)
    t2 = time.time()
    flat = [r for recs in results for r in recs]
    order = {"latent": 0, "agree": 1, "ident": 2}
    records = [r for _, r in sorted(enumerate(flat), key=lambda ir: (order[ir[1]["kind"]], ir[0]))]
    out = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "run": score.run_info(pred, run_dir), "config": _config(blacklist),
           "records": records,
           "timing": {"setup": round(t1 - t0, 3), "latent": round(t2 - t1, 3),
                      "total": round(time.time() - t0, 3), "n_jobs": len(jobs),
                      "workers": int(workers)}}
    score.write_json(out, run_dir / "latent.json")
    (run_dir / "LATENT_DONE").write_text(out["created_utc"] + "\n")
    _CTX.clear()
    return run_dir / "latent.json"


def _rung_of(run_dir: Path) -> str:
    cfg = run_dir / "config.json"
    if cfg.is_file():
        return json.loads(cfg.read_text())["rung"]
    return torch.load(run_dir / "ckpt.pt", map_location="cpu", weights_only=True)["config"]["rung"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("run")
    for pos in ("manifest", "covariates", "data_dir", "blacklist", "run_dir"):
        a.add_argument(pos, type=Path)
    a.add_argument("--workers", type=int, default=score.default_workers())
    args = ap.parse_args(argv)
    done = args.run_dir / "LATENT_DONE"
    if done.exists():
        print(f"{done} exists; nothing to do")
        return 0
    rung = _rung_of(args.run_dir)
    if rung not in RUNGS:
        print(f"{args.run_dir.name}: rung {rung!r} has no latent (want one of {RUNGS})",
              file=sys.stderr)
        return 2
    t0 = time.time()
    pred = model.load_run(args.run_dir, args.data_dir, args.covariates, args.manifest, device="cpu")
    rows = pairs.read_manifest(args.manifest)
    path = run_latent(pred, rows, args.blacklist, args.run_dir, workers=args.workers)
    print(f"latent {args.run_dir.name} -> {path} ({time.time() - t0:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
