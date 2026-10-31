"""t125 part 1 — train one (rung, g, model, seed) count run with the depth offset.

    python tools/t125/eta/train.py train <manifest.tsv> <covariates.tsv> <data_dir> <runs_dir>
        <rung> <g> <model> <seed> [--max-steps N] [--eval-every N] [--device auto|cpu|cuda]
    python tools/t125/eta/train.py train-index <manifest.tsv> <covariates.tsv> <data_dir>
        <runs_dir> <index> [...]                      (the index-th row of `eta.pairs.tasks()`)

Everything is t118's (`ladder/train.py`: pairs, chromosome split, schedule, sampler rng streams,
early stopping, run dir) except the depth offset (plan/ETA_OFFSET_AND_H_DESIGN.md "Part 1"):

- f reads x = eta_in = log1p(X) - log d_src (`offset.transform_eta`, d in millions of reads,
  from the manifest column `depth`); the target NB has log mu' = log d' + eta_out, d' = the
  depth of the sample's TARGET pid. The nocov twin keeps the offset (PI 2026-10-11): its
  covariate pair is redrawn, its d' is still its own pair's target (the offset belongs to the
  counts, the covariates to g).
- Knots (spread curve of A/B, levels of row 2): `base.knot_stats` over t118's knot draw (rng 0,
  blocks of `knot_block`), each block transformed with its own pid's log d.
- Validation: mean over the trained pairs of the mean NB NLL over chr22 with the offset. The nocov
  twin's prediction depends on the source only up to the offset, so it is computed once per source
  pid with log d' = 0 (loc before the offset) and each pair adds its own log d' — never a
  per-source cache after the offset.
- Models: real and nocov (no ids twin). config.json = t118's keys plus `log_depth` (pid -> log d,
  every manifest row), `t125`, `form_module`, and `row: 2` for per-bin rungs.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import warnings
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[2]
for _p in (_TOOLS / "t125", _TOOLS / "t118"):   # import convention: tools/t118 and tools/t125
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ladder import base, data, encoding  # noqa: E402
from ladder import model as lmodel  # noqa: E402
from ladder import pairs as lpairs  # noqa: E402
from ladder import train as ltrain  # noqa: E402
from eta import model as em  # noqa: E402
from eta import offset  # noqa: E402
from eta import pairs as ep  # noqa: E402

SPACE = ep.SPACE
DEFAULTS = dict(ltrain.DEFAULTS)
T125_CONFIG = {"offset": "log1p(X) - log(d/1e6)", "d_unit": offset.D_UNIT}


def run_name(rung: str, g: str, model_name: str, seed: int) -> str:
    return f"{rung}_{g}_{SPACE}_{model_name}_s{int(seed)}"


def task_of_index(manifest, index: int) -> dict:
    """The `index`-th row of `eta.pairs.tasks()`; IndexError outside 0..431."""
    table = ep.tasks(ep.read_manifest(manifest))
    if not 0 <= int(index) < len(table):
        raise IndexError(f"task index {index} outside 0..{len(table) - 1}")
    return table[int(index)]


def knot_sample_eta(corpus, fit_pids, chroms, lens, n: int, block: int,
                    log_d: dict) -> np.ndarray:
    """eta_in over `n` bins: t118's knot draw (rng 0; pid, then chrom ∝ length, then start, in
    blocks of `block`), each block transformed with its own pid's log d; float64."""
    rng = np.random.default_rng(0)
    n_blocks = max(1, -(-int(n) // int(block)))
    pid_i = rng.integers(len(fit_pids), size=n_blocks)
    chrom_i = rng.choice(len(chroms), size=n_blocks, p=lens / lens.sum())
    u = rng.random(n_blocks)
    out = []
    for p, c, uu in zip(pid_i, chrom_i, u):
        start = int(uu * (max(0, int(lens[c]) - block) + 1))
        pid = fit_pids[p]
        out.append(offset.transform_eta(
            corpus.window(pid, SPACE, chroms[c], start, start + block), log_d[pid]))
    return np.concatenate(out)[:n].astype(np.float64)


class EtaSampler(ltrain.Sampler):
    """t118's sampler (same rng streams) on eta. `draw(B)` -> (pair_idx, cov_idx, x [B, W+2c],
    y [B, W], mask, ld_tgt float32[B]); ld_tgt = log d of each sample's own pair's target."""

    def __init__(self, corpus, train_pairs, chroms, lens, window, context, seed, model_name,
                 log_d: dict):
        if model_name not in ep.MODELS:
            raise ValueError(f"model {model_name!r} not in {ep.MODELS}")
        super().__init__(corpus, SPACE, train_pairs, chroms, lens, window, context, seed,
                         model_name, None)
        self.log_d = log_d
        self.ld_src = np.array([log_d[p["source_pid"]] for p in train_pairs], dtype=np.float64)
        self.ld_tgt = np.array([log_d[p["target_pid"]] for p in train_pairs], dtype=np.float32)

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
            xs[b] = offset.transform_eta(
                self.corpus.window(p["source_pid"], SPACE, chrom, s - c, s + W + c),
                self.ld_src[pi[b]])
            ys[b] = self.corpus.window(p["target_pid"], SPACE, chrom, s, s + W)
            mask[b, :max(0, min(W, n - s))] = True
        return pi, self.cov_index(pi), xs, ys, mask, self.ld_tgt[pi].copy()


def validate_eta(ladder, corpus, train_pairs, cov_val: torch.Tensor, log_d: dict, device,
                 average_theta: bool = False, level_cache: dict | None = None) -> float:
    """Mean over the trained pairs of the mean NB NLL over the whole of chr22, with the offset.

    real: each pair's own (C, C') (row 1: its theta, `predict_chrom_eta`; row 2:
    `predict_chrom_bins_eta`). nocov (`average_theta`): row 1 one theta, `nocov_theta(cov_val)`;
    row 2 the averaged map over `cov_val`. The nocov prediction reads only the source, so it is
    made once per source pid with log d' = 0 (loc BEFORE the offset; row 2 with `level_cache`, a
    dict the caller keeps for the run, keyed (source, chrom)) and every pair adds its own target's
    log d'. Bit-identical to the per-pair loop; the mean is over the pairs in their own order.
    """
    chrom = lpairs.VAL_CHROMS[0]
    was_training = ladder.training
    ladder.eval()
    per_pair = [None] * len(train_pairs)
    with torch.no_grad():
        if average_theta:
            theta_avg = None if ladder.per_bin else lmodel.nocov_theta(ladder, cov_val)
            by_src: dict = {}
            for i, p in enumerate(train_pairs):
                by_src.setdefault(p["source_pid"], []).append(i)
            for src, idx in by_src.items():
                X = corpus.get(src, SPACE, chrom)
                if ladder.per_bin:
                    lc = None if level_cache is None else level_cache.setdefault((src, chrom), {})
                    loc0, disp = em.predict_chrom_bins_eta(ladder, X, log_d[src], 0.0, cov_val,
                                                           device, average=True, level_cache=lc)
                else:
                    loc0, disp = em.predict_chrom_eta(ladder, X, log_d[src], 0.0, theta_avg,
                                                      device)
                disp_t = torch.from_numpy(disp).to(device)
                for i in idx:
                    tgt = train_pairs[i]["target_pid"]
                    loc = offset.offset_loc(loc0, log_d[tgt]).astype(np.float32)
                    y = np.array(corpus.get(tgt, SPACE, chrom), dtype=np.float32)
                    per_pair[i] = float(ladder.nll(torch.from_numpy(loc).to(device), disp_t,
                                                   torch.from_numpy(y).to(device)))
        else:
            for i, p in enumerate(train_pairs):
                src, tgt = p["source_pid"], p["target_pid"]
                X = corpus.get(src, SPACE, chrom)
                if ladder.per_bin:
                    loc, disp = em.predict_chrom_bins_eta(ladder, X, log_d[src], log_d[tgt],
                                                          cov_val[i:i + 1], device)
                else:
                    loc, disp = em.predict_chrom_eta(ladder, X, log_d[src], log_d[tgt],
                                                     ladder.theta(cov_val[i:i + 1])[0], device)
                y = np.array(corpus.get(tgt, SPACE, chrom), dtype=np.float32)
                per_pair[i] = float(ladder.nll(torch.from_numpy(loc).to(device),
                                               torch.from_numpy(disp).to(device),
                                               torch.from_numpy(y).to(device)))
    ladder.train(was_training)
    return float(np.mean(per_pair))


def run_training(manifest, covariates, data_dir, runs_dir, rung, g, model_name, seed,
                 max_steps=None, eval_every=None, device="auto", cov_rows=None, **overrides):
    """Train one run into `<runs_dir>/<run_name>/`; returns the run dir. Skips if TRAIN_DONE.

    `overrides` may replace any DEFAULTS entry (tests only; the CLI exposes max_steps, eval_every
    and device). `cov_rows` replaces the rows read from `covariates` (tests only).
    """
    t0 = time.time()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if rung not in ep.RUNGS:
        raise ValueError(f"rung {rung!r} not in {ep.RUNGS}")
    if model_name not in ep.MODELS:
        raise ValueError(f"model {model_name!r} not in {ep.MODELS}")
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
    name = run_name(rung, g, model_name, seed)
    run_dir = Path(runs_dir) / name
    if (run_dir / "TRAIN_DONE").is_file():
        print(f"{name}: TRAIN_DONE present, nothing to do")
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    dev = lmodel.pick_device(device)

    rows = ep.read_manifest(manifest)
    if cov_rows is None:
        cov_rows = encoding.read_covariates(covariates)
    log_d = offset.depth_table(rows)
    tp = ep.train_pairs(rows, g)
    fit = ep.fit_pids(rows, g)
    if len(tp) < 2:
        raise ValueError(f"g {g!r}: {len(tp)} training pairs")
    corpus = data.Corpus(data_dir)
    if not corpus.is_cache:
        warnings.warn(f"{data_dir} is a products dir of npz files, not the memory-mapped cache: "
                      f"every window decompresses a chromosome — slow, meant for synthetic data "
                      f"only; build the cache with `ladder/train.py cache` for real runs",
                      stacklevel=2)
    enc = encoding.Encoder.fit(cov_rows, fit)
    perm = lpairs.ids_permutation(len(tp), seed)       # recorded as t118 does; no ids twin here
    cov_all = np.stack([enc.encode_pair(p["source_pid"], p["target_pid"]) for p in tp])
    cov_all_t = torch.from_numpy(cov_all).to(dev)
    cov_val = cov_all_t

    chroms, lens = ltrain.training_chroms(corpus, tp[0]["source_pid"], SPACE)
    xs = knot_sample_eta(corpus, fit, chroms, lens, cfg["knot_sample"], cfg["knot_block"], log_d)
    stats = {"knots_x": base.knot_stats(xs), "n0": float(cfg["n0"]), "sigma0": float(cfg["sigma0"])}
    del xs

    torch.manual_seed(seed)
    np.random.seed(seed)
    ladder = em.EtaLadder(rung, stats).to(dev)
    sampler = EtaSampler(corpus, tp, chroms, lens, cfg["window"], ladder.context, seed,
                         model_name, log_d)
    opt = torch.optim.Adam(ladder.parameters(), lr=cfg["lr"])
    warm = max(1, int(cfg["warmup"]))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))

    config = {
        "run_name": name, "rung": rung, "g": g, "space": SPACE, "model": model_name, "seed": seed,
        **({"row": 2} if ladder.per_bin else {}), **cfg, "device": str(dev),
        "n_cov": encoding.N_COV, "cov_fields": list(encoding.FIELDS),
        "context": ladder.context, "n_theta": ladder.n_theta,
        "train_chroms": chroms, "val_chroms": list(lpairs.VAL_CHROMS),
        "score_chroms": list(lpairs.SCORE_CHROMS),
        "sampling": "pair uniform, chromosome proportional to length, start uniform",
        "nocov_rng": "default_rng([seed, 7121]) per sample per step",
        "knots": "base.knot_stats on eta_in of knot_sample bins in knot_block blocks, "
                 "default_rng(0), each block with its own pid's log d",
        "git_sha": ltrain.git_sha(), "manifest": str(manifest),
        "manifest_md5": lmodel.md5_path(manifest),
        "covariates": str(covariates),
        "covariates_md5": lmodel.md5_path(covariates) if covariates else None,
        "data_dir": str(data_dir), "fit_pids": fit, "train_pairs": tp,
        "ids_perm": [int(v) for v in perm],
        "knots_x": [float(v) for v in stats["knots_x"]],
        "form_module": ep.FORM_MODULE[rung],
        "log_depth": {k: float(v) for k, v in log_d.items()},
        "t125": dict(T125_CONFIG),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=1) + "\n")

    t_setup = time.time() - t0
    t1 = time.time()
    log = []
    nocov = model_name == "nocov"
    level_cache = {} if (ladder.per_bin and nocov) else None
    val = validate_eta(ladder, corpus, tp, cov_val, log_d, dev, nocov, level_cache=level_cache)
    log.append((0, float("nan"), val, time.time() - t1))
    best = {"val": val, "step": 0, "g": copy.deepcopy(ladder.g.state_dict()),
            "form": copy.deepcopy(ladder.form.state_dict())}
    bad, losses, step = 0, [], 0
    ladder.train()
    for step in range(1, cfg["max_steps"] + 1):
        _, ci, xb, yb, mb, ldb = sampler.draw(cfg["batch"])
        cov = cov_all_t[torch.from_numpy(ci).to(dev)]
        loc, disp = ladder(cov, torch.from_numpy(xb).to(dev), torch.from_numpy(ldb).to(dev))
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
            val = validate_eta(ladder, corpus, tp, cov_val, log_d, dev, nocov,
                               level_cache=level_cache)
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
              "peak_rss_mb": ltrain.peak_rss_mb(), "steps_run": step,
              "best_step": int(best["step"]), "val_nll": float(best["val"])}
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
    for a in ("manifest", "covariates", "data_dir", "runs_dir", "rung", "g", "model"):
        p.add_argument(a)
    p.add_argument("seed", type=int)
    knobs(p)
    p = sub.add_parser("train-index")
    for a in ("manifest", "covariates", "data_dir", "runs_dir"):
        p.add_argument(a)
    p.add_argument("index", type=int)
    knobs(p)
    args = ap.parse_args(argv)

    if args.cmd == "train-index":
        t = task_of_index(args.manifest, args.index)
        rung, g, model_name, seed = t["rung"], t["g"], t["model"], t["seed"]
    else:
        rung, g, model_name, seed = args.rung, args.g, args.model, args.seed
    run_training(args.manifest, args.covariates, args.data_dir, args.runs_dir, rung, g,
                 model_name, seed, max_steps=args.max_steps, eval_every=args.eval_every,
                 device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
