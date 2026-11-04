"""t146 — train one run of the task table (`tasks.tasks()`): one shared design-D model over all tracks.

    python tools/t146/train.py train-index <manifest> <covariates.tsv> <cache_dir> <runs_dir> <index>
        [--features DIR] [--max-steps N] [--eval-every N] [--device auto|cpu|cuda]

**Correction** (every arm): exactly old shared D's stream — pairs = base<->arm of all 7 tracks with
depth 7.5M held out (`tasks.train_pairs`), per step `batch` windows drawn as `ladder.train.Sampler`
draws them (pair uniform, chromosome proportional to length, start uniform; rng default_rng(seed),
same draw order), x = log1p(source window) with the 30-bin halo, y = target counts; NB NLL.
Training chromosomes = all but chr19, chr21 (score) and chr22 (validation). Adam lr 1e-3, 100-step
linear warmup, grad-clip 1.0 on the TOTAL loss, batch 32, window 2048.

**Auxiliary batches** (one per step, the same batch size, their own rngs
default_rng([seed, 7146, k]), so the correction stream is the alone arm's stream):
  xassay   pair uniform over `tasks.xassay_pairs`; x = source (a base) window, y = the other
           assay's product; g gets (C_src, C_tgt); the aux head.
  denoise  product uniform over the fit pids; each input bin masked (x = 0, mask channel = 1) with
           probability MASK_RATE; y = the same product's counts; NLL over the masked in-chromosome
           output bins only; g gets (C_p, C_p); the aux head. Correction batches carry mask 0.
  consist  pair uniform over `tasks.consist_pairs`; x_a with a 60-bin halo; chained: leg 1 =
           model(C_a, C_base) on x_a, x_mid = log1p(NB mean, clamped) (0 off the chromosome), leg 2
           = model(C_base, C_a') on x_mid; direct = model(C_a, C_a') on x_a; L_cons = mean over the
           in-chromosome bins of (loc_d - loc_c)^2 + (disp_d - disp_c)^2, unclamped, with gradient
           through BOTH routes (h30's v2 checks that the chained route did not get worse).

**Loss normalisation** (`step0_losses`): L0_corr and L0_aux = the mean loss of the initial model
over NORM_BATCHES batches drawn from default_rng([seed, 7147, 0 | 1]) (not the training stream).
total = corr_factor * L_corr + w * L_aux / L0_aux, corr_factor = 1 / L0_corr for matched, xassay and
denoise, 1 for alone, consist and the h31 arms. consist: total = L_corr + W_CONSIST * L_cons.

**Validation / early stopping**: mean over the correction pairs of the mean NB NLL over the whole
of chr22 (as `ladder.train.validate`), at step 0, every `eval_every` steps and the last step; the
correction loss ONLY, for every arm; patience 4; the best state is saved.

Run dir: config.json, leakage_audit.json, ckpt.pt, train_log.tsv, timing.json, TRAIN_DONE (last).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_TOOLS = Path(__file__).resolve().parents[1]
for _p in (_TOOLS / "t118", _TOOLS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ladder import data, encoding, pairs as lp  # noqa: E402
from ladder.model import MU_MAX, MU_MIN, N_MAX, N_MIN, pick_device, transform_x  # noqa: E402
from ladder.train import git_sha, peak_rss_mb, training_chroms  # noqa: E402
from t146 import tasks as T  # noqa: E402
from t146.features import Features, shifted_window  # noqa: E402
from t146.model import CONTEXT, T146Model, predict_chrom  # noqa: E402

DEFAULTS = {"window": 2048, "batch": 32, "lr": 1e-3, "warmup": 100, "grad_clip": 1.0,
            "max_steps": T.MAX_STEPS, "eval_every": 250, "patience": 4, "norm_batches": 8}


# ---------------------------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------------------------


def full_inputs(feats, scalar, has_dna, chrom: str, n: int):
    """Whole-chromosome extras: ([k, n] float32 or None, uint8[n*25] or None); mask = 0."""
    ex = None
    if scalar:
        rows = []
        for kind in scalar:
            if kind == "mask":
                rows.append(np.zeros(n, np.float32))
            elif kind == "map":
                rows.append(np.asarray(feats.map_full(chrom)[:n], np.float32))
            elif kind == "mapshift":
                rows.append(shifted_window(np.asarray(feats.map_full(chrom)), 0, n))
        ex = np.stack(rows)
    dna = np.asarray(feats._arr("dna", chrom)) if has_dna else None
    return ex, dna


class Windows:
    """Window drawing over the training chromosomes, plus the extras for any (chrom, start)."""

    def __init__(self, corpus, chroms, lens, window, feats, scalar, has_dna):
        self.corpus, self.chroms, self.lens = corpus, chroms, lens
        self.probs = lens / lens.sum()
        self.W = int(window)
        self.feats, self.scalar, self.has_dna = feats, tuple(scalar), has_dna

    def positions(self, rng, B):
        ci = rng.choice(len(self.chroms), size=B, p=self.probs)
        u = rng.random(B)
        st = [int(uu * (max(0, int(self.lens[c]) - self.W) + 1)) for c, uu in zip(ci, u)]
        return ci, st

    def x(self, pid, ci, st, halo):
        return np.stack([transform_x(self.corpus.window(pid[b], "counts", self.chroms[c], s - halo,
                                                        s + self.W + halo), "counts")
                         for b, (c, s) in enumerate(zip(ci, st))])

    def y(self, pid, ci, st):
        ys = np.stack([self.corpus.window(pid[b], "counts", self.chroms[c], s, s + self.W)
                       for b, (c, s) in enumerate(zip(ci, st))]).astype(np.float32)
        return ys

    def inchrom(self, ci, st, halo):
        """bool [B, W + 2 halo]: the window position lies on the chromosome."""
        out = np.zeros((len(ci), self.W + 2 * halo), dtype=bool)
        for b, (c, s) in enumerate(zip(ci, st)):
            n = int(self.lens[c])
            lo, hi = max(0, halo - s), min(self.W + 2 * halo, n - s + halo)
            out[b, lo:max(lo, hi)] = True
        return out

    def extras(self, ci, st, halo, mask=None):
        if not self.scalar and not self.has_dna:
            return None, None
        ex = None
        if self.scalar:
            rows = []
            for kind in self.scalar:
                if kind == "mask":
                    rows.append(mask if mask is not None
                                else np.zeros((len(ci), self.W + 2 * halo), np.float32))
                else:
                    rows.append(np.stack([self.feats.window(kind, self.chroms[c], s - halo,
                                                            s + self.W + halo)
                                          for c, s in zip(ci, st)]))
            ex = np.stack(rows, 1).astype(np.float32)
        dna = None
        if self.has_dna:
            dna = np.stack([self.feats.window("dna", self.chroms[c], s - halo, s + self.W + halo)
                            for c, s in zip(ci, st)])
        return ex, dna


def _t(a, dev):
    return None if a is None else torch.from_numpy(np.ascontiguousarray(a)).to(dev)


# ---------------------------------------------------------------------------------------------
# the losses of one batch
# ---------------------------------------------------------------------------------------------


class Batches:
    def __init__(self, model, wins: Windows, enc, rows, tp, dev, batch, mode="real", seed=0):
        self.m, self.w, self.enc, self.dev, self.B = model, wins, enc, dev, int(batch)
        if mode not in T.MODELS:
            raise ValueError(f"model {mode!r} not in {T.MODELS}")
        self.mode, self.seed = mode, int(seed)
        self.rows, self.tp = rows, tp
        self.cov_tp = torch.from_numpy(np.stack([enc.encode_pair(p["source_pid"], p["target_pid"])
                                                 for p in tp])).to(dev)
        self.xa = T.xassay_pairs(rows)
        self.cov_xa = torch.from_numpy(np.stack([enc.encode_pair(p["source_pid"], p["target_pid"])
                                                 for p in self.xa])).to(dev)
        self.fit = T.fit_pids(rows)
        self.cp = T.consist_pairs(rows)
        self.base = {r["track"]: r["pid"] for r in lp.usable_products(rows) if r["arm"] == "base"}
        self.track = {r["pid"]: r["track"] for r in lp.usable_products(rows)}
        # t118 twins: ids = a fixed derangement within each pair set; nocov = a random correction
        # pair's covariates per sample, from its own rng (the window stream stays the real one's)
        self.perm_tp = lp.ids_permutation(len(tp), seed)
        self.perm_xa = lp.ids_permutation(len(self.xa), seed)
        self.perm_fit = lp.ids_permutation(len(self.fit), seed)
        self.reset_cov_rng()

    def reset_cov_rng(self):
        self.cov_rng = np.random.default_rng([self.seed, 7121])

    def cov_index(self, pi):
        """Which correction pair's (C, C') each sample is fed."""
        if self.mode == "ids":
            return self.perm_tp[pi]
        if self.mode == "nocov":
            return self.cov_rng.integers(len(self.tp), size=len(pi))
        return pi

    def cov_val(self):
        """[n_pairs, 40] the covariates validation feeds each correction pair (ids: the scrambled
        labels it trained on; nocov: every row is used through the mean theta, see validate)."""
        return self.cov_tp[torch.from_numpy(self.perm_tp).to(self.dev)] if self.mode == "ids" \
            else self.cov_tp

    def corr(self, rng):
        """The correction batch, drawn in `ladder.train.Sampler.draw`'s order."""
        pi = rng.integers(len(self.tp), size=self.B)
        ci, st = self.w.positions(rng, self.B)
        c = CONTEXT
        x = self.w.x([self.tp[i]["source_pid"] for i in pi], ci, st, c)
        y = self.w.y([self.tp[i]["target_pid"] for i in pi], ci, st)
        m = self.w.inchrom(ci, st, 0)
        ex, dna = self.w.extras(ci, st, c)
        ci_ = self.cov_index(pi)
        loc, disp = self.m(self.cov_tp[torch.from_numpy(ci_).to(self.dev)], _t(x, self.dev),
                           _t(ex, self.dev), _t(dna, self.dev))
        return self.m.nll(loc, disp, _t(y, self.dev), _t(m, self.dev))

    def xassay(self, rng):
        pi = rng.integers(len(self.xa), size=self.B)
        ci, st = self.w.positions(rng, self.B)
        c = CONTEXT
        x = self.w.x([self.xa[i]["source_pid"] for i in pi], ci, st, c)
        y = self.w.y([self.xa[i]["target_pid"] for i in pi], ci, st)
        m = self.w.inchrom(ci, st, 0)
        ex, dna = self.w.extras(ci, st, c)
        cx = self.perm_xa[pi] if self.mode == "ids" else pi
        loc, disp = self.m(self.cov_xa[torch.from_numpy(cx).to(self.dev)], _t(x, self.dev),
                           _t(ex, self.dev), _t(dna, self.dev), head="aux")
        return self.m.nll(loc, disp, _t(y, self.dev), _t(m, self.dev))

    def denoise(self, rng):
        pi = rng.integers(len(self.fit), size=self.B)
        pids = [self.fit[i] for i in pi]
        ci, st = self.w.positions(rng, self.B)
        c = CONTEXT
        x = self.w.x(pids, ci, st, c)
        inc = self.w.inchrom(ci, st, c)
        mask = (rng.random(x.shape) < T.MASK_RATE) & inc
        x = np.where(mask, 0.0, x).astype(np.float32)
        y = self.w.y(pids, ci, st)
        lm = mask[:, c:c + self.w.W]
        ex, dna = self.w.extras(ci, st, c, mask=mask.astype(np.float32))
        cp = ([self.fit[self.perm_fit[i]] for i in pi] if self.mode == "ids" else pids)
        cov = torch.from_numpy(np.stack([self.enc.encode_pair(p, p) for p in cp])).to(self.dev)
        loc, disp = self.m(cov, _t(x, self.dev), _t(ex, self.dev), _t(dna, self.dev), head="aux")
        return self.m.nll(loc, disp, _t(y, self.dev), _t(lm, self.dev))

    def consist(self, rng):
        if self.m.inputs:
            raise ValueError("consist is defined for the no-extra-input model only")
        pi = rng.integers(len(self.cp), size=self.B)
        prs = [self.cp[i] for i in pi]
        ci, st = self.w.positions(rng, self.B)
        c = CONTEXT
        xa = self.w.x([p["source_pid"] for p in prs], ci, st, 2 * c)
        inc1 = self.w.inchrom(ci, st, c)                     # leg-1 output positions
        inc = self.w.inchrom(ci, st, 0)
        enc = self.enc
        bases = [self.base[self.track[p["source_pid"]]] for p in prs]
        cov_ab = torch.from_numpy(np.stack([enc.encode_pair(p["source_pid"], b)
                                            for p, b in zip(prs, bases)])).to(self.dev)
        cov_bt = torch.from_numpy(np.stack([enc.encode_pair(b, p["target_pid"])
                                            for p, b in zip(prs, bases)])).to(self.dev)
        cov_d = torch.from_numpy(np.stack([enc.encode_pair(p["source_pid"], p["target_pid"])
                                           for p in prs])).to(self.dev)
        xt = _t(xa, self.dev)
        lo_mu, hi_mu = math.log(MU_MIN), math.log(MU_MAX)
        # both routes carry gradient (an agreement loss, not a bootstrapped target): the
        # stop-gradient version collapsed in the smoke (job 25715856_27) to loc below the NLL's
        # clamp everywhere, where every gradient is zero. Neither route is clamped in the
        # squared error, so a route outside the clamp is still pulled back.
        loc1, _ = self.m(cov_ab, xt)
        xm = torch.log1p(loc1.clamp(lo_mu, hi_mu).exp()) * _t(inc1, self.dev).to(loc1.dtype)
        loc_c, disp_c = self.m(cov_bt, xm)
        loc_d, disp_d = self.m(cov_d, xt[:, c:xt.shape[1] - c])
        se = (loc_d - loc_c) ** 2 + (disp_d - disp_c) ** 2
        mm = _t(inc, self.dev).to(se.dtype)
        return (se * mm).sum() / mm.sum().clamp(min=1.0)


def step0_losses(bt: Batches, aux, seed, n_batches) -> dict:
    """L0 of the correction and the aux objective: the initial model's mean loss over
    `n_batches` batches each from default_rng([seed, 7147, 0 | 1])."""
    out = {}
    with torch.no_grad():
        r = np.random.default_rng([int(seed), 7147, 0])
        out["L0_corr"] = float(np.mean([float(bt.corr(r)) for _ in range(n_batches)]))
        if aux in ("xassay", "denoise", "consist"):
            r = np.random.default_rng([int(seed), 7147, 1])
            fn = getattr(bt, aux)
            out["L0_aux"] = float(np.mean([float(fn(r)) for _ in range(n_batches)]))
    return out


# ---------------------------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------------------------


@torch.no_grad()
def validate(model, corpus, feats, tp, cov_tp, dev, chrom=lp.VAL_CHROMS[0], cache=None,
             average_theta=False) -> float:
    """`average_theta` (the nocov twin): one theta for every pair, g's mean over `cov_tp`, exactly
    as it is scored."""
    was = model.training
    model.eval()
    per = []
    sc, has_dna = model.form.scalar, model.form.has_dna
    for i, p in enumerate(tp):
        X = corpus.get(p["source_pid"], "counts", chrom)
        Y = corpus.get(p["target_pid"], "counts", chrom)
        if cache is not None and "ex" not in cache:
            cache["ex"] = full_inputs(feats, sc, has_dna, chrom, X.shape[0])
        ex, dna = cache["ex"] if cache is not None else full_inputs(feats, sc, has_dna, chrom,
                                                                     X.shape[0])
        theta = (model.theta(cov_tp).mean(0) if average_theta
                 else model.theta(cov_tp[i:i + 1])[0])
        loc, disp = predict_chrom(model, X, theta, dev, extra_full=ex, dna_full=dna)
        y = torch.from_numpy(np.asarray(Y, np.float32)).to(dev)
        per.append(float(model.nll(torch.from_numpy(loc).to(dev), torch.from_numpy(disp).to(dev), y)))
    model.train(was)
    return float(np.mean(per))


# ---------------------------------------------------------------------------------------------
# one run
# ---------------------------------------------------------------------------------------------


def run_training(manifest, covariates, data_dir, runs_dir, task: dict, features_dir=None,
                 max_steps=None, eval_every=None, device="auto", **overrides) -> Path:
    t0 = time.time()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    arm = T.ARMS[task["arm"]]
    seed, w = int(task["seed"]), task["weight"]
    cfg = dict(DEFAULTS)
    unknown = set(overrides) - set(cfg)
    if unknown:
        raise KeyError(f"unknown settings {sorted(unknown)}")
    cfg.update(overrides)
    if max_steps is not None:
        cfg["max_steps"] = int(max_steps)
    if eval_every is not None:
        cfg["eval_every"] = int(eval_every)
    name = task["run_name"]
    run_dir = Path(runs_dir) / name
    if (run_dir / "TRAIN_DONE").is_file():
        print(f"{name}: TRAIN_DONE present, nothing to do")
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    dev = pick_device(device)

    rows = lp.read_manifest(manifest)
    tp = T.train_pairs(rows)
    fit = T.fit_pids(rows)
    enc = encoding.Encoder.fit(encoding.read_covariates(covariates), fit)
    inputs = arm["inputs"]
    needs_feats = any(i in ("map", "mapshift", "dna") for i in inputs)
    feats = Features(features_dir) if needs_feats else None
    if needs_feats and not (Path(features_dir) / "FEATURES_DONE").is_file():
        raise FileNotFoundError(f"no FEATURES_DONE under {features_dir}")
    corpus = data.Corpus(data_dir)
    chroms, lens = training_chroms(corpus, tp[0]["source_pid"], "counts")

    aux = arm["aux"]
    trained = [dict(p) for p in tp]
    if aux == "xassay":
        trained += T.xassay_pairs(rows)
    if aux == "denoise":
        trained += [{"source_pid": p, "target_pid": p, "direction": "denoise"} for p in fit]
    if aux == "consist":
        trained += [dict(p, direction="consist") for p in T.consist_pairs(rows)]
    audit = T.leakage_audit(rows, enc, trained, T.law_pairs(rows))
    (run_dir / "leakage_audit.json").write_text(json.dumps(audit, indent=1) + "\n")

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = T146Model(inputs=inputs, aux_head=aux in ("xassay", "denoise")).to(dev)
    wins = Windows(corpus, chroms, lens, cfg["window"], feats, model.form.scalar,
                   model.form.has_dna)
    mode = arm["model"]
    bt = Batches(model, wins, enc, rows, tp, dev, cfg["batch"], mode=mode, seed=seed)
    norm = step0_losses(bt, aux, seed, cfg["norm_batches"])
    bt.reset_cov_rng()                                      # step 0 must not move the nocov stream
    avg = mode == "nocov"
    corr_factor = 1.0 / norm["L0_corr"] if arm["corr_scale"] == "step0" else 1.0
    aux_factor = 0.0
    if aux in ("xassay", "denoise"):
        aux_factor = float(w) / norm["L0_aux"]
    elif aux == "consist":
        aux_factor = float(T.W_CONSIST)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    warm = max(1, int(cfg["warmup"]))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))
    config = {"run_name": name, "task": task, "arm": task["arm"], "arm_spec": arm,
              "seed": seed, "weight": w, "rung": T.RUNG, "g": T.G, "space": T.SPACE,
              "model": mode, "ids_perm": [int(v) for v in bt.perm_tp], "inputs": list(inputs), "holdout": [list(h) for h in T.HOLDOUT],
              **cfg, "device": str(dev), "context": CONTEXT, "n_theta": model.form.n_theta,
              "n_params": int(sum(p.numel() for p in model.parameters())),
              "train_chroms": chroms, "val_chroms": list(lp.VAL_CHROMS),
              "score_chroms": list(lp.SCORE_CHROMS), "fit_pids": fit, "train_pairs": tp,
              "n_xassay_pairs": len(bt.xa), "n_consist_pairs": len(bt.cp),
              "mask_rate": T.MASK_RATE if aux == "denoise" else None,
              "norm": norm, "corr_factor": corr_factor, "aux_factor": aux_factor,
              "features_dir": str(features_dir) if needs_feats else None,
              "git_sha": git_sha(), "manifest": str(manifest), "covariates": str(covariates),
              "data_dir": str(data_dir)}
    (run_dir / "config.json").write_text(json.dumps(config, indent=1) + "\n")

    rng = np.random.default_rng(seed)                       # the correction stream (old D's)
    arng = np.random.default_rng([seed, 7146, 0])           # the aux stream
    t_setup = time.time() - t0
    t1 = time.time()
    vcache = {}
    t_val = time.time()
    val = validate(model, corpus, feats, tp, bt.cov_val(), dev, cache=vcache,
                           average_theta=avg)
    t_val = time.time() - t_val
    log = [(0, float("nan"), float("nan"), float("nan"), val, time.time() - t1)]
    print(f"{name}: step 0 val {val:.6f} ({t_val:.1f} s) norm {norm}", flush=True)
    best = {"val": val, "step": 0, "state": copy.deepcopy(model.state_dict())}
    bad, step, acc = 0, 0, []
    model.train()
    for step in range(1, cfg["max_steps"] + 1):
        lc = bt.corr(rng)
        total = corr_factor * lc
        la = torch.zeros((), device=dev)
        if aux is not None:
            la = getattr(bt, aux)(arng)
            total = total + aux_factor * la
        if not torch.isfinite(total):
            raise FloatingPointError(f"{name}: non-finite loss at step {step}")
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        sched.step()
        acc.append((float(total), float(lc), float(la)))
        if step % cfg["eval_every"] == 0 or step == cfg["max_steps"]:
            val = validate(model, corpus, feats, tp, bt.cov_val(), dev, cache=vcache,
                           average_theta=avg)
            a = np.mean(acc, 0)
            log.append((step, float(a[0]), float(a[1]), float(a[2]), val, time.time() - t1))
            acc = []
            print(f"{name}: step {step} loss {a[0]:.5f} corr {a[1]:.5f} aux {a[2]:.5f} "
                  f"val {val:.6f} t {time.time() - t1:.0f}s", flush=True)
            if val < best["val"]:
                best = {"val": val, "step": step, "state": copy.deepcopy(model.state_dict())}
                bad = 0
            else:
                bad += 1
                if bad >= cfg["patience"]:
                    break
    t_train = time.time() - t1
    ckpt = {"state": {k: v.cpu() for k, v in best["state"].items()}, "config": config,
            "encoder": enc.state(), "step": int(best["step"]), "val_nll": float(best["val"])}
    torch.save(ckpt, run_dir / "ckpt.pt.tmp")
    os.replace(run_dir / "ckpt.pt.tmp", run_dir / "ckpt.pt")
    with open(run_dir / "train_log.tsv", "w", encoding="utf-8") as fh:
        fh.write("step\tloss\tloss_corr\tloss_aux\tval_nll\tseconds\n")
        for s, lo, lc_, la_, v, sec in log:
            fh.write(f"{s}\t{lo!r}\t{lc_!r}\t{la_!r}\t{v!r}\t{sec:.3f}\n")
    timing = {"setup": t_setup, "train": t_train, "total": time.time() - t0,
              "val_seconds_step0": t_val, "peak_rss_mb": peak_rss_mb(), "steps_run": step,
              "best_step": int(best["step"]), "val_nll": float(best["val"]),
              "s_per_step": t_train / max(step, 1)}
    (run_dir / "timing.json").write_text(json.dumps(timing, indent=1) + "\n")
    (run_dir / "TRAIN_DONE").write_text(f"{name} best_step {best['step']} val_nll {best['val']!r}\n")
    print(f"{name}: best_step {best['step']} val_nll {best['val']:.6g} steps {step} "
          f"train {t_train:.1f}s", flush=True)
    return run_dir


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train-index")
    for a in ("manifest", "covariates", "data_dir", "runs_dir"):
        p.add_argument(a)
    p.add_argument("index", type=int)
    p.add_argument("--features", default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = ap.parse_args(argv)
    table = T.tasks()
    if not 0 <= args.index < len(table):
        raise IndexError(f"task index {args.index} outside 0..{len(table) - 1}")
    run_training(args.manifest, args.covariates, args.data_dir, args.runs_dir, table[args.index],
                 features_dir=args.features, max_steps=args.max_steps,
                 eval_every=args.eval_every, device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
