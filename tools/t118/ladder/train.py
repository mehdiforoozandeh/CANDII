"""t118 ladder — train one (rung, g, space, model, seed) run, and build one product's cache.

    python tools/t118/ladder/train.py train <manifest.tsv> <covariates.tsv> <data_dir> <runs_dir>
        <rung> <g> <space> <model> <seed> [--max-steps N] [--eval-every N] [--device auto|cpu|cuda]
    python tools/t118/ladder/train.py train-index <manifest.tsv> <covariates.tsv> <data_dir>
        <runs_dir> <index> [...]                        (the index-th row of `pairs.tasks`)
    python tools/t118/ladder/train.py cache <manifest.tsv> <products_dir> <cache_dir> <index>
                                                        (`data.build_cache`, all products by pid)

**Data.** Pairs = `pairs.train_pairs(rows, g)`. Training chromosomes = every chromosome except
chr19, chr21 (score) and chr22 (validation); chrY, chrM dropped by the corpus. Each step draws
`batch` windows: pair uniform over the training pairs, chromosome proportional to its length,
start uniform; x = the source's window plus the form's context halo (zero-padded off the
chromosome), y = the target's window. `<data_dir>` is a products dir or the memory-mapped cache
(`data.Corpus`); nothing loads whole products.

**Models.** real = the pair's own (C, C'); nocov = at every step each sample takes the (C, C') of a
uniformly random training pair (its own rng, so the window stream is the same for all three
models); ids = pair i takes the (C, C') of pair perm[i], perm = `pairs.ids_permutation(n, seed)`,
fixed for the run. Same architecture, data, steps and seed for all three. The nocov twin is
validated (and predicted, `model.Predictor`) with one theta: g's mean theta over the training pairs.

**Schedule** (defaults pinned in DEFAULTS): Adam lr 1e-3 with a 100-step linear warmup, grad-clip
1.0, float32, 4000 steps; validation = mean over the trained pairs of the mean NLL over the whole of
chr22, at step 0 and every 250 steps (and at the last step); early stopping after 4 evaluations
without improvement; the best state is restored and saved.

**Knots** (rungs B, C; computed for every rung, A and D ignore them): `base.knot_stats` of x over a
4 M-bin sample from `default_rng(0)`: blocks of 1024 contiguous bins, each at a uniform
(fit pid, chromosome proportional to length, start) over the training chromosomes.

**Run dir** `<runs_dir>/<run_name>/`: config.json, ckpt.pt, train_log.tsv (`step loss val_nll
seconds`), timing.json, TRAIN_DONE (written last; if present the CLI exits 0 without work).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import resource
import subprocess
import sys
import time
import warnings
from pathlib import Path

_T118 = Path(__file__).resolve().parents[1]
if str(_T118) not in sys.path:          # script mode: make `ladder` importable
    sys.path.insert(0, str(_T118))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ladder import base, data, encoding, model, pairs  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
DEFAULTS = {"window": 2048, "batch": 32, "lr": 1e-3, "warmup": 100, "grad_clip": 1.0,
            "dtype": "float32", "max_steps": 4000, "eval_every": 250, "patience": 4,
            "knot_sample": 4_000_000, "knot_block": 1024, "n0": 5.0, "sigma0": 0.5}


def run_name(rung: str, g: str, space: str, model_name: str, seed: int) -> str:
    return f"{rung}_{g}_{space}_{model_name}_s{int(seed)}"


def git_sha() -> str:
    """`<repo>/GIT_SHA` (the Nibi kit) if present, else `git rev-parse HEAD`, else "unknown"."""
    f = REPO / "GIT_SHA"
    if f.is_file():
        return f.read_text().strip()
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 1024.0     # bytes on macOS, KiB on Linux


def training_chroms(corpus, pid: str, space: str) -> tuple[list[str], np.ndarray]:
    held = set(pairs.SCORE_CHROMS) | set(pairs.VAL_CHROMS)
    chroms = [c for c in corpus.chroms(pid, space) if c not in held]
    if not chroms:
        raise ValueError(f"{pid}/{space}: no training chromosomes")
    return chroms, np.array([corpus.n_bins(pid, space, c) for c in chroms], dtype=np.int64)


def knot_sample(corpus, fit_pids, space, chroms, lens, n: int, block: int) -> np.ndarray:
    """x over `n` bins: blocks of `block` bins at uniform (pid, chrom ∝ length, start); rng 0."""
    rng = np.random.default_rng(0)
    n_blocks = max(1, -(-int(n) // int(block)))
    pid_i = rng.integers(len(fit_pids), size=n_blocks)
    chrom_i = rng.choice(len(chroms), size=n_blocks, p=lens / lens.sum())
    u = rng.random(n_blocks)
    out = []
    for p, c, uu in zip(pid_i, chrom_i, u):
        start = int(uu * (max(0, int(lens[c]) - block) + 1))
        out.append(corpus.window(fit_pids[p], space, chroms[c], start, start + block))
    return model.transform_x(np.concatenate(out)[:n], space).astype(np.float64)


class Sampler:
    """Random training windows. `draw(B)` -> (pair_idx, cov_idx, x [B, W+2c], y [B, W], mask)."""

    def __init__(self, corpus, space, train_pairs, chroms, lens, window, context, seed,
                 model_name, perm):
        self.corpus, self.space, self.pairs = corpus, space, train_pairs
        self.chroms, self.lens = chroms, lens
        self.probs = lens / lens.sum()
        self.window, self.context = int(window), int(context)
        self.model, self.perm = model_name, perm
        self.rng = np.random.default_rng(int(seed))             # pairs, chromosomes, positions
        self.cov_rng = np.random.default_rng([int(seed), 7121])  # the nocov re-draw only

    def cov_index(self, pair_idx: np.ndarray) -> np.ndarray:
        if self.model == "real":
            return pair_idx.copy()
        if self.model == "ids":
            return self.perm[pair_idx]
        if self.model == "nocov":
            return self.cov_rng.integers(len(self.pairs), size=pair_idx.shape[0])
        raise ValueError(f"model {self.model!r} not in {pairs.MODELS}")

    def draw(self, batch: int):
        W, c = self.window, self.context
        pi = self.rng.integers(len(self.pairs), size=batch)
        ci = self.rng.choice(len(self.chroms), size=batch, p=self.probs)
        u = self.rng.random(batch)
        xs = np.empty((batch, W + 2 * c), dtype=np.float32)
        ys = np.empty((batch, W), dtype=np.float32)
        mask = np.zeros((batch, W), dtype=bool)
        for b in range(batch):
            p, chrom, n = self.pairs[pi[b]], self.chroms[ci[b]], int(self.lens[ci[b]])
            s = int(u[b] * (max(0, n - W) + 1))
            xs[b] = model.transform_x(
                self.corpus.window(p["source_pid"], self.space, chrom, s - c, s + W + c),
                self.space)
            ys[b] = self.corpus.window(p["target_pid"], self.space, chrom, s, s + W)
            mask[b, :max(0, min(W, n - s))] = True
        return pi, self.cov_index(pi), xs, ys, mask


def validate(ladder, corpus, train_pairs, cov_val: torch.Tensor, space, device,
             average_theta: bool = False) -> float:
    """Mean over the trained pairs of the mean per-bin NLL over the whole of chr22.

    `average_theta` (the nocov twin): every pair uses one theta, the mean over `cov_val`, exactly
    as the twin's Predictor does, so early stopping selects the model that is scored.
    """
    chrom = pairs.VAL_CHROMS[0]
    was_training = ladder.training
    ladder.eval()
    per_pair = []
    with torch.no_grad():
        theta_avg = model.nocov_theta(ladder, cov_val) if average_theta else None
        for i, p in enumerate(train_pairs):
            X = corpus.get(p["source_pid"], space, chrom)
            Y = corpus.get(p["target_pid"], space, chrom)
            theta = theta_avg if average_theta else ladder.theta(cov_val[i:i + 1])[0]
            loc, disp = model.predict_chrom(ladder, X, theta, device)
            y = torch.from_numpy(np.array(Y, dtype=np.float32)).to(device)
            per_pair.append(float(ladder.nll(torch.from_numpy(loc).to(device),
                                             torch.from_numpy(disp).to(device), y)))
    ladder.train(was_training)
    return float(np.mean(per_pair))


def run_training(manifest, covariates, data_dir, runs_dir, rung, g, space, model_name, seed,
                 max_steps=None, eval_every=None, device="auto", cov_rows=None, **overrides):
    """Train one run into `<runs_dir>/<run_name>/`; returns the run dir. Skips if TRAIN_DONE.

    `overrides` may replace any DEFAULTS entry (tests only; the CLI exposes max_steps, eval_every
    and device). `cov_rows` replaces the rows read from `covariates` (tests only).
    """
    t0 = time.time()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if space not in pairs.SPACES:
        raise ValueError(f"space {space!r} not in {pairs.SPACES}")
    if model_name not in pairs.MODELS:
        raise ValueError(f"model {model_name!r} not in {pairs.MODELS}")
    seed = int(seed)
    cfg = dict(DEFAULTS)
    unknown = set(overrides) - set(cfg)
    if unknown:
        raise KeyError(f"unknown training settings: {sorted(unknown)}")
    cfg.update(overrides)
    if max_steps is not None:
        cfg["max_steps"] = int(max_steps)
    if eval_every is not None:
        cfg["eval_every"] = int(eval_every)
    name = run_name(rung, g, space, model_name, seed)
    run_dir = Path(runs_dir) / name
    if (run_dir / "TRAIN_DONE").is_file():
        print(f"{name}: TRAIN_DONE present, nothing to do")
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    dev = model.pick_device(device)

    rows = pairs.read_manifest(manifest)
    if cov_rows is None:
        cov_rows = encoding.read_covariates(covariates)
    tp = pairs.train_pairs(rows, g)
    fit = pairs.fit_pids(rows, g)
    if len(tp) < 2:
        raise ValueError(f"g {g!r}: {len(tp)} training pairs")
    corpus = data.Corpus(data_dir)
    if not corpus.is_cache:
        warnings.warn(f"{data_dir} is a products dir of npz files, not the memory-mapped cache: "
                      f"every window decompresses a chromosome — slow, meant for synthetic data "
                      f"only; build the cache with `train.py cache` for real runs", stacklevel=2)
    enc = encoding.Encoder.fit(cov_rows, fit)
    perm = pairs.ids_permutation(len(tp), seed)
    cov_all = np.stack([enc.encode_pair(p["source_pid"], p["target_pid"]) for p in tp])
    cov_val_idx = perm if model_name == "ids" else np.arange(len(tp))
    cov_all_t = torch.from_numpy(cov_all).to(dev)
    cov_val = cov_all_t[torch.from_numpy(cov_val_idx).to(dev)]

    chroms, lens = training_chroms(corpus, tp[0]["source_pid"], space)
    xs = knot_sample(corpus, fit, space, chroms, lens, cfg["knot_sample"], cfg["knot_block"])
    stats = {"knots_x": base.knot_stats(xs), "n0": float(cfg["n0"]), "sigma0": float(cfg["sigma0"])}
    del xs

    torch.manual_seed(seed)
    np.random.seed(seed)
    ladder = model.Ladder(rung, space, stats).to(dev)
    sampler = Sampler(corpus, space, tp, chroms, lens, cfg["window"], ladder.context, seed,
                      model_name, perm)
    opt = torch.optim.Adam(ladder.parameters(), lr=cfg["lr"])
    warm = max(1, int(cfg["warmup"]))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))

    config = {
        "run_name": name, "rung": rung, "g": g, "space": space, "model": model_name, "seed": seed,
        **cfg, "device": str(dev), "n_cov": encoding.N_COV, "cov_fields": list(encoding.FIELDS),
        "context": ladder.context, "n_theta": ladder.n_theta,
        "train_chroms": chroms, "val_chroms": list(pairs.VAL_CHROMS),
        "score_chroms": list(pairs.SCORE_CHROMS),
        "sampling": "pair uniform, chromosome proportional to length, start uniform",
        "nocov_rng": "default_rng([seed, 7121]) per sample per step",
        "knots": "base.knot_stats on knot_sample bins in knot_block blocks, default_rng(0)",
        "git_sha": git_sha(), "manifest": str(manifest), "manifest_md5": model.md5_path(manifest),
        "covariates": str(covariates),
        "covariates_md5": model.md5_path(covariates) if covariates else None,
        "data_dir": str(data_dir), "fit_pids": fit, "train_pairs": tp,
        "ids_perm": [int(v) for v in perm],
        "knots_x": [float(v) for v in stats["knots_x"]],
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=1) + "\n")

    t_setup = time.time() - t0
    t1 = time.time()
    log = []
    val = validate(ladder, corpus, tp, cov_val, space, dev, model_name == "nocov")
    log.append((0, float("nan"), val, time.time() - t1))
    best = {"val": val, "step": 0, "g": copy.deepcopy(ladder.g.state_dict()),
            "form": copy.deepcopy(ladder.form.state_dict())}
    bad, losses, step = 0, [], 0
    ladder.train()
    for step in range(1, cfg["max_steps"] + 1):
        _, ci, xb, yb, mb = sampler.draw(cfg["batch"])
        cov = cov_all_t[torch.from_numpy(ci).to(dev)]
        loc, disp = ladder(cov, torch.from_numpy(xb).to(dev))
        loss = ladder.nll(loc, disp, torch.from_numpy(yb).to(dev), torch.from_numpy(mb).to(dev))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{name}: non-finite loss at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ladder.parameters(), cfg["grad_clip"])
        opt.step()
        sched.step()
        losses.append(float(loss))
        if step % cfg["eval_every"] == 0 or step == cfg["max_steps"]:
            val = validate(ladder, corpus, tp, cov_val, space, dev, model_name == "nocov")
            log.append((step, float(np.mean(losses)), val, time.time() - t1))
            losses = []
            if val < best["val"]:
                best = {"val": val, "step": step, "g": copy.deepcopy(ladder.g.state_dict()),
                        "form": copy.deepcopy(ladder.form.state_dict())}
                bad = 0
            else:
                bad += 1
                if bad >= cfg["patience"]:
                    break
    t_train = time.time() - t1

    ckpt = {"g": {k: v.cpu() for k, v in best["g"].items()},
            "form": {k: v.cpu() for k, v in best["form"].items()},
            "stats": {"knots_x": [float(v) for v in stats["knots_x"]], "n0": stats["n0"],
                      "sigma0": stats["sigma0"]},
            "encoder": enc.state(), "config": config, "step": int(best["step"]),
            "val_nll": float(best["val"])}
    tmp = run_dir / "ckpt.pt.tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, run_dir / "ckpt.pt")
    with open(run_dir / "train_log.tsv", "w", encoding="utf-8") as fh:
        fh.write("step\tloss\tval_nll\tseconds\n")
        for s, lo, v, sec in log:
            fh.write(f"{s}\t{lo!r}\t{v!r}\t{sec:.3f}\n")
    timing = {"setup": t_setup, "train": t_train, "total": time.time() - t0,
              "peak_rss_mb": peak_rss_mb(), "steps_run": step, "best_step": int(best["step"]),
              "val_nll": float(best["val"])}
    (run_dir / "timing.json").write_text(json.dumps(timing, indent=1) + "\n")
    (run_dir / "TRAIN_DONE").write_text(f"{name} best_step {best['step']} val_nll {best['val']!r}\n")
    print(f"{name}: best_step {best['step']} val_nll {best['val']:.6g} "
          f"steps {step} train {t_train:.1f}s total {timing['total']:.1f}s")
    return run_dir


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def knobs(p):
        p.add_argument("--max-steps", type=int, default=None)
        p.add_argument("--eval-every", type=int, default=None)
        p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")

    p = sub.add_parser("train")
    for a in ("manifest", "covariates", "data_dir", "runs_dir", "rung", "g", "space", "model"):
        p.add_argument(a)
    p.add_argument("seed", type=int)
    knobs(p)
    p = sub.add_parser("train-index")
    for a in ("manifest", "covariates", "data_dir", "runs_dir"):
        p.add_argument(a)
    p.add_argument("index", type=int)
    knobs(p)
    p = sub.add_parser("cache")
    for a in ("manifest", "products_dir", "cache_dir"):
        p.add_argument(a)
    p.add_argument("index", type=int)
    args = ap.parse_args(argv)

    if args.cmd == "cache":
        rows = sorted(pairs.read_manifest(args.manifest), key=lambda r: r["pid"])
        if not 0 <= args.index < len(rows):
            raise IndexError(f"cache index {args.index} outside 0..{len(rows) - 1}")
        row = rows[args.index]
        status = data.build_cache(args.products_dir, args.cache_dir, row["pid"], row)
        print(f"{row['pid']}: " + " ".join(f"{s}={v}" for s, v in status.items()))
        return 0
    if args.cmd == "train-index":
        table = pairs.tasks(pairs.read_manifest(args.manifest))
        if not 0 <= args.index < len(table):
            raise IndexError(f"task index {args.index} outside 0..{len(table) - 1}")
        t = table[args.index]
        rung, g, space, model_name, seed = t["rung"], t["g"], t["space"], t["model"], t["seed"]
    else:
        rung, g, space, model_name, seed = args.rung, args.g, args.space, args.model, args.seed
    run_training(args.manifest, args.covariates, args.data_dir, args.runs_dir, rung, g, space,
                 model_name, seed, max_steps=args.max_steps, eval_every=args.eval_every,
                 device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
