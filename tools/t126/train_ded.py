"""t126 rung D — train one encoder-decoder D run (form DED) with the four-term loss.

`run_training_ded(...)` is `ladder.train.run_training` (counts only, depth 7.5M held out) with the
training step replaced (plan/T126_ENCDEC_A.md "Rung D"). For a sampled base<->arm pair
(X, c) -> (X', c'), with Z = enc(X, c) and Z' = enc(X', c'):

    loss = NLL(X' | dec(Z, c')) + NLL(X | dec(Z', c)) + NLL(X | dec(Z, c)) + NLL(X' | dec(Z', c'))
           + lam * mse(Z, Z')

Each NLL is the mean NB NLL over the window's in-chromosome bins (`Ladder.nll` with the mask).
mse(Z, Z') is the mean over the 32 channels and the latent positions the decoder reads
(the window plus the decoder's 24-bin halo each side: positions ENC_CONTEXT .. W + 2*30 -
ENC_CONTEXT, where Z sees no padding of the window). The no-covariates twin draws (c, c') of a
uniformly random training pair per sample, as t118's twin, and uses them in all four terms.

Everything else is t118 row 1's: sampler rng streams (the window stream equals `Sampler.draw`'s),
Adam lr 1e-3 with 100-step warmup, grad clip 1.0, 4000 steps, validation (mean over the trained
pairs of the cross-prediction NLL X -> X' on chr22, every 250 steps; the twin with its mean theta),
patience 4, best state saved, the run dir layout, and `ckpt.pt` readable by `ladder.model.load_run`
(rung "DED"), so `ladder/score.py trained` / `law` score it unchanged.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
import warnings
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[1]
for _p in (_TOOLS / "t118", _TOOLS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ladder import base, data, encoding, model, pairs  # noqa: E402
from ladder import train as ltrain  # noqa: E402
from ladder.fforms.form_ded import CONTEXT, ENC_CONTEXT  # noqa: E402

RUNG = "DED"
SPACE = "counts"


class PairSampler(ltrain.Sampler):
    """`ladder.train.Sampler` that also returns the target window with its halo and the raw source
    centre. Same rng calls in the same order as `Sampler.draw`, so the window stream is t118's.
    `draw(B)` -> (pair_idx, cov_idx, x_src [B, W+2c], x_tgt [B, W+2c], y_src [B, W], y_tgt [B, W],
    mask [B, W])."""

    def draw(self, batch: int):
        W, c = self.window, self.context
        pi = self.rng.integers(len(self.pairs), size=batch)
        ci = self.rng.choice(len(self.chroms), size=batch, p=self.probs)
        u = self.rng.random(batch)
        xs = np.empty((batch, W + 2 * c), dtype=np.float32)
        xt = np.empty((batch, W + 2 * c), dtype=np.float32)
        ys = np.empty((batch, W), dtype=np.float32)
        yt = np.empty((batch, W), dtype=np.float32)
        mask = np.zeros((batch, W), dtype=bool)
        for b in range(batch):
            p, chrom, n = self.pairs[pi[b]], self.chroms[ci[b]], int(self.lens[ci[b]])
            s = int(u[b] * (max(0, n - W) + 1))
            src = self.corpus.window(p["source_pid"], self.space, chrom, s - c, s + W + c)
            tgt = self.corpus.window(p["target_pid"], self.space, chrom, s - c, s + W + c)
            xs[b] = model.transform_x(src, self.space)
            xt[b] = model.transform_x(tgt, self.space)
            ys[b] = src[c:c + W]
            yt[b] = tgt[c:c + W]
            mask[b, :max(0, min(W, n - s))] = True
        return pi, self.cov_index(pi), xs, xt, ys, yt, mask


def four_term_loss(ladder: model.Ladder, cov: torch.Tensor, xs, xt, ys, yt, mask, lam: float):
    """(total, terms dict) of the four-term loss for a batch; `cov` [B, 40] = (c, c')."""
    form, g = ladder.form, ladder.g
    n = g.n_cov
    c, ct = cov[:, :n], cov[:, n:]
    z = form.encode(xs, g.g_enc(c))
    zt = form.encode(xt, g.g_enc(ct))
    dec_c, dec_t = g.g_dec(c), g.g_dec(ct)
    terms = {
        "cross_st": ladder.nll(*form.decode(z, dec_t), yt, mask),
        "cross_ts": ladder.nll(*form.decode(zt, dec_c), ys, mask),
        "self_s": ladder.nll(*form.decode(z, dec_c), ys, mask),
        "self_t": ladder.nll(*form.decode(zt, dec_t), yt, mask),
    }
    total = terms["cross_st"] + terms["cross_ts"] + terms["self_s"] + terms["self_t"]
    lo, hi = ENC_CONTEXT, z.shape[2] - ENC_CONTEXT
    mse = ((z[:, :, lo:hi] - zt[:, :, lo:hi]) ** 2).mean()
    terms["mse_z"] = mse
    if lam:
        total = total + float(lam) * mse
    return total, terms


def run_training_ded(manifest, covariates, data_dir, runs_dir, arm, g, model_name, seed, lam,
                     holdout=(), max_steps=None, eval_every=None, device="auto", cov_rows=None,
                     **overrides):
    """Train one DED run into `<runs_dir>/<arm>_<g>_counts_<model>_s<seed>/`; skips if TRAIN_DONE."""
    t0 = time.time()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if model_name not in ("real", "nocov"):
        raise ValueError(f"model {model_name!r} not in (real, nocov)")
    seed = int(seed)
    cfg = dict(ltrain.DEFAULTS)
    unknown = set(overrides) - set(cfg)
    if unknown:
        raise KeyError(f"unknown training settings: {sorted(unknown)}")
    cfg.update(overrides)
    if max_steps is not None:
        cfg["max_steps"] = int(max_steps)
    if eval_every is not None:
        cfg["eval_every"] = int(eval_every)
    name = f"{arm}_{g}_{SPACE}_{model_name}_s{seed}"
    run_dir = Path(runs_dir) / name
    if (run_dir / "TRAIN_DONE").is_file():
        print(f"{name}: TRAIN_DONE present, nothing to do")
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    dev = model.pick_device(device)

    rows = pairs.read_manifest(manifest)
    if cov_rows is None:
        cov_rows = encoding.read_covariates(covariates)
    holdout = [[str(a), str(lv)] for a, lv in holdout]
    tp = pairs.train_pairs(rows, g, holdout)
    fit = pairs.fit_pids(rows, g, holdout)
    if len(tp) < 2:
        raise ValueError(f"g {g!r}: {len(tp)} training pairs")
    corpus = data.Corpus(data_dir)
    if not corpus.is_cache:
        warnings.warn(f"{data_dir} is a products dir of npz files, not the memory-mapped cache: "
                      f"meant for synthetic data only", stacklevel=2)
    enc = encoding.Encoder.fit(cov_rows, fit)
    perm = pairs.ids_permutation(len(tp), seed)
    cov_all = np.stack([enc.encode_pair(p["source_pid"], p["target_pid"]) for p in tp])
    cov_all_t = torch.from_numpy(cov_all).to(dev)
    cov_val = cov_all_t

    chroms, lens = ltrain.training_chroms(corpus, tp[0]["source_pid"], SPACE)
    xs = ltrain.knot_sample(corpus, fit, SPACE, chroms, lens, cfg["knot_sample"], cfg["knot_block"])
    stats = {"knots_x": base.knot_stats(xs), "n0": float(cfg["n0"]), "sigma0": float(cfg["sigma0"])}
    del xs

    torch.manual_seed(seed)
    np.random.seed(seed)
    ladder = model.Ladder(RUNG, SPACE, stats).to(dev)
    assert ladder.context == CONTEXT
    sampler = PairSampler(corpus, SPACE, tp, chroms, lens, cfg["window"], ladder.context, seed,
                          model_name, perm)
    opt = torch.optim.Adam(ladder.parameters(), lr=cfg["lr"])
    warm = max(1, int(cfg["warmup"]))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))

    config = {
        "run_name": name, "rung": RUNG, "arm": arm, "lam": float(lam), "g": g, "space": SPACE,
        "model": model_name, "seed": seed, **cfg, "device": str(dev), "n_cov": encoding.N_COV,
        "cov_fields": list(encoding.FIELDS), "context": ladder.context, "n_theta": ladder.n_theta,
        "train_chroms": chroms, "val_chroms": list(pairs.VAL_CHROMS),
        "score_chroms": list(pairs.SCORE_CHROMS),
        "sampling": "pair uniform, chromosome proportional to length, start uniform",
        "nocov_rng": "default_rng([seed, 7121]) per sample per step",
        "knots": "base.knot_stats on knot_sample bins in knot_block blocks, default_rng(0)",
        "loss": "NLL(X'|dec(Z,c')) + NLL(X|dec(Z',c)) + NLL(X|dec(Z,c)) + NLL(X'|dec(Z',c')) "
                "+ lam*mse(Z,Z')",
        "git_sha": ltrain.git_sha(), "manifest": str(manifest),
        "manifest_md5": model.md5_path(manifest), "covariates": str(covariates),
        "covariates_md5": model.md5_path(covariates) if covariates else None,
        "data_dir": str(data_dir), "fit_pids": fit, "train_pairs": tp,
        "ids_perm": [int(v) for v in perm], "knots_x": [float(v) for v in stats["knots_x"]],
    }
    if holdout:
        config["holdout"] = holdout
        config["heldout_pairs"] = pairs.heldout_pairs(rows, g, holdout)
    (run_dir / "config.json").write_text(json.dumps(config, indent=1) + "\n")

    t_setup = time.time() - t0
    t1 = time.time()
    log = []
    nocov = model_name == "nocov"
    val = ltrain.validate(ladder, corpus, tp, cov_val, SPACE, dev, nocov)
    log.append((0, float("nan"), float("nan"), val, time.time() - t1))
    best = {"val": val, "step": 0, "g": copy.deepcopy(ladder.g.state_dict()),
            "form": copy.deepcopy(ladder.form.state_dict())}
    bad, losses, mses, step = 0, [], [], 0
    ladder.train()
    for step in range(1, cfg["max_steps"] + 1):
        _, ci, xb, xtb, ysb, ytb, mb = sampler.draw(cfg["batch"])
        cov = cov_all_t[torch.from_numpy(ci).to(dev)]
        t = lambda a: torch.from_numpy(a).to(dev)  # noqa: E731
        loss, terms = four_term_loss(ladder, cov, t(xb), t(xtb), t(ysb), t(ytb), t(mb), lam)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{name}: non-finite loss at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ladder.parameters(), cfg["grad_clip"])
        opt.step()
        sched.step()
        losses.append(float(loss))
        mses.append(float(terms["mse_z"]))
        if step % cfg["eval_every"] == 0 or step == cfg["max_steps"]:
            val = ltrain.validate(ladder, corpus, tp, cov_val, SPACE, dev, nocov)
            log.append((step, float(np.mean(losses)), float(np.mean(mses)), val, time.time() - t1))
            losses, mses = [], []
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
        fh.write("step\tloss\tmse_z\tval_nll\tseconds\n")
        for s, lo, ms, v, sec in log:
            fh.write(f"{s}\t{lo!r}\t{ms!r}\t{v!r}\t{sec:.3f}\n")
    timing = {"setup": t_setup, "train": t_train, "total": time.time() - t0,
              "peak_rss_mb": ltrain.peak_rss_mb(), "steps_run": step,
              "best_step": int(best["step"]), "val_nll": float(best["val"])}
    (run_dir / "timing.json").write_text(json.dumps(timing, indent=1) + "\n")
    (run_dir / "TRAIN_DONE").write_text(f"{name} best_step {best['step']} val_nll {best['val']!r}\n")
    print(f"{name}: best_step {best['step']} val_nll {best['val']:.6g} "
          f"steps {step} train {t_train:.1f}s total {timing['total']:.1f}s")
    return run_dir
