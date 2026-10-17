"""t125 part 2 — train one learned h on the true X' or on one f's predicted NB.

    python tools/t125/hmap/train.py train <manifest> <covariates.tsv> <data_dir> <ctl_cache>
        <runs_dir> <variant> <rung> <noise 0|1> <source true|fnb> <seed> [--fnb-run RUN_DIR]
        [--fnb-noise-loss expected|mixture] [--max-steps N] [--eval-every N]
        [--device auto|cpu|cuda] [--control-rl-tsv PATH]
    python tools/t125/hmap/train.py train-index <manifest> <covariates.tsv> <data_dir> <ctl_cache>
        <runs_dir> <table true|fnb> <index> [--eta-runs DIR] [--fnb-seeds 1|3] [...]

(variant, rung, noise) is one of `hmap.tasks.H_LEARNED` (12). The run dir is
`<runs_dir>/<h_name>/` with `h_name = tasks.h_name(variant, rung, noise, "true" | <f run>, seed)`;
an existing TRAIN_DONE makes the call a no-op. `train-index` takes the index-th row of
`tasks.tasks_true()` or `tasks.tasks_fnb(eta.pairs.f_runs_seed0(), seeds)`; an fnb row
`fnb:<run_name>` resolves to `<--eta-runs>/<run_name>`.

Settings (PI 2026-10-11: "the same training settings as the t118 g"): `DEFAULTS =
ladder.train.DEFAULTS` — Adam lr 1e-3, linear warmup 100 steps, grad-norm clip 1, at most 4000
steps of `batch` = 32 windows of `window` = 2048 bins, validation every 250 steps on chr22,
early stopping after 4 evaluations without a gain, the best state saved. MLP h = 2 x 64
(`hmap.model`). One shared h over every assay (the assay one-hot is an input).

Sources (`hmap.data`): true = every usable product (`ladder.pairs.usable_products`, 128 on t112),
its true counts; fnb = the f's `config["train_pairs"]`, f's NB of the target (`eta.model.load_run`).

Losses (`hmap.model`; y = the true P; the loss is the mean over the batch's bins on the
chromosome):
- push, true:   (mu - log1p P)^2
- noise, true:  Gaussian NLL of log1p P
- push, fnb:    sum_k NB(k) (mu_k - log1p P)^2
- noise, fnb:   sum_k NB(k) NLL_k ("expected", PI decision (iii)); `--fnb-noise-loss mixture`
                gives -log sum_k NB(k) N(log1p P; mu_k, sigma_k)
The f-NB sums run over the exact NB support of `hmap.pushforward` (tail 1e-6, K_MAX 4096), in
dense chunks sorted by support size (`hmap.data.fnb_chunks`); each chunk's loss is weighted by its
share of the batch's bins and back-propagated on its own, so the gradient equals the whole batch's.
Rows whose support exceeds K_MAX are counted: `timing.json` holds `n_capped_train` (every training
draw), `n_capped_val` (one validation pass; f is fixed, so every pass sees the same rows) and
`n_capped` (their sum), with `n_k_mean` / `n_k_max` of the training draws. A nonzero count goes to
the PI.

Validation every `eval_every` steps: the mean over products (true) or training pairs (fnb) of the
mean loss over the whole of chr22, computed in `VAL_CHUNK` = 65 536-bin chunks (window features
over a halo, so the chunking does not move the value). f is frozen, so for fnb its chr22 (loc,
disp) per pair is predicted once before training (`f_val_pass`, time in `timing.json`
`f_pass_seconds`) and reused by every pass; the NB support is still enumerated per pass.
`timing.json` `val_seconds` lists the wall time of each validation pass. Training reads only the
training chromosomes and chr22, never chr19 or chr21 (the score chromosomes).

Standardiser (`hmap.model.Standardiser`): fitted once, before training, on the first
`N_STD_WINDOWS` = 64 windows a fresh source of the same seed draws, and frozen. The true source
fits it on the true features; the fnb source on the features at k = f's predicted mean (log1p of
the mean; the fragment window over the means). The formula column is never scaled.

Run dir: `config.json`, `ckpt.pt` (keys h, std, config, step, val_loss), `train_log.tsv`
(`step loss val_loss seconds`), `timing.json`, `TRAIN_DONE` last.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[2]
for _p in (_TOOLS / "t125", _TOOLS / "t118"):   # import convention: tools/t118 and tools/t125
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ladder import model as lmodel  # noqa: E402
from ladder import pairs as lpairs  # noqa: E402
from ladder import train as ltrain  # noqa: E402
from eta import model as em  # noqa: E402
from eta import pairs as ep  # noqa: E402
from hmap import data as hd  # noqa: E402
from hmap import features, products, pushforward, tasks  # noqa: E402
from hmap import model as hm  # noqa: E402

DEFAULTS = dict(ltrain.DEFAULTS)
SOURCES = ("true", "fnb")
TABLES = ("true", "fnb")
VAL_CHUNK = 65536
N_STD_WINDOWS = 64


def is_formula(variant: str) -> bool:
    return variant.startswith("formula")


def feature_names(variant: str, rung: int) -> list[str]:
    return list(features.FEATURES[rung]) + ([features.FEATURE_FORMULA] if is_formula(variant)
                                            else [])


def task_of_index(table: str, index: int, eta_runs=None, fnb_seeds: int = 1) -> dict:
    """The `index`-th row of the true or fnb table, with `source` "true" | "fnb" and `fnb_run`
    (`<eta_runs>/<run_name>` for fnb, else None)."""
    if table not in TABLES:
        raise ValueError(f"table {table!r} not in {TABLES}")
    if table == "fnb" and eta_runs is None:
        raise ValueError("fnb rows need --eta-runs to resolve fnb:<run_name>")
    rows = (tasks.tasks_true() if table == "true"
            else tasks.tasks_fnb(ep.f_runs_seed0(), seeds=int(fnb_seeds)))
    index = int(index)
    if not 0 <= index < len(rows):
        raise IndexError(f"{table} index {index} outside 0..{len(rows) - 1}")
    r = rows[index]
    out = {k: r[k] for k in ("variant", "rung", "noise", "seed", "h_name")}
    if table == "true":
        out.update(source="true", fnb_run=None)
    else:
        f_run = r["source"].split(":", 1)[1]
        out.update(source="fnb", fnb_run=str(Path(eta_runs) / f_run))
    return out


# ---------------------------------------------------------------------------------------------
# inputs and losses
# ---------------------------------------------------------------------------------------------


def true_input(feat_r5: np.ndarray, fcol: np.ndarray | None, rung: int, formula: bool):
    """h's input [N, F] from the rung-5 matrix: rung 1 = its first 9 columns; formula last."""
    x = feat_r5[:, :len(features.FEATURES_R1)] if rung == 1 else feat_r5
    if formula:
        if fcol is None:
            raise ValueError("a formula variant needs the formula column")
        x = np.concatenate([x, fcol[:, None]], axis=1)
    return np.ascontiguousarray(x, dtype=np.float32)


def _true_loss(h, std, x: np.ndarray, y: np.ndarray, device):
    xt = std.apply(torch.from_numpy(x).to(device))
    yt = torch.from_numpy(y).to(device)
    mu, ls = h(xt)
    return hm.loss_noise_true(mu, ls, yt) if h.noise else hm.loss_push_true(mu, yt)


def true_batch_loss(h, std, tb: hd.TrueBatch, rung: int, formula: bool, device,
                    backward: bool = True) -> float:
    """Mean loss over the batch's bins on the chromosome; back-propagated when `backward`."""
    x = true_input(tb.feat_r5, tb.formula_col, rung, formula)[tb.mask]
    y = tb.y[tb.mask]
    with torch.set_grad_enabled(backward):
        loss = _true_loss(h, std, x, y, device)
        if backward:
            loss.backward()
    return float(loss)


def fnb_batch_loss(h, std, fb: hd.FnbBatch, rung: int, formula: bool, noise: bool, rule: str,
                   device, elems: int = hd.FNB_ELEMS, k_max: int = pushforward.K_MAX,
                   backward: bool = True) -> tuple[float, dict]:
    """(mean loss over the batch's bins, stats) with the exact NB support in chunks sorted by
    support size; each chunk's loss x (its rows / N) is back-propagated on its own when
    `backward`, so the accumulated gradient is the whole batch's. stats: n_bins, n_capped,
    n_k_sum, n_k_max."""
    N = int(fb.y.shape[0])
    total = 0.0
    st = {"n_bins": N, "n_capped": 0, "n_k_sum": 0, "n_k_max": 0}
    if N == 0:
        return float("nan"), st
    with torch.set_grad_enabled(backward):
        for ch in hd.fnb_chunks(fb, rung, formula, elems=elems, k_max=k_max):
            r = ch.idx.shape[0]
            xt = std.apply(torch.from_numpy(ch.feat).to(device))
            wt = torch.from_numpy(ch.w).to(device=device, dtype=torch.float32)
            yt = torch.from_numpy(ch.y).to(device)
            mu, ls = h(xt)
            loss = (hm.loss_noise_fnb(mu, ls, wt, yt, rule) if noise
                    else hm.loss_push_fnb(mu, wt, yt))
            part = loss * (r / N)
            if backward:
                part.backward()
            total += float(part)
            st["n_capped"] += ch.n_capped
            st["n_k_sum"] += int(ch.n_k.sum())
            st["n_k_max"] = max(st["n_k_max"], int(ch.n_k.max()))
    return total, st


# ---------------------------------------------------------------------------------------------
# validation on chr22
# ---------------------------------------------------------------------------------------------


def validate_true(h, std, corpus: hd.HCorpus, infos: dict, pids, rung: int, formula: bool,
                  device, val_chunk: int = VAL_CHUNK) -> float:
    """Mean over `pids` of the mean loss over the whole of chr22 (true counts)."""
    chrom = lpairs.VAL_CHROMS[0]
    was = h.training
    h.eval()
    per = []
    with torch.no_grad():
        for pid in pids:
            info = infos[pid]
            n = corpus.n_bins(pid, chrom)
            acc = 0.0
            for s in range(0, n, int(val_chunk)):
                f5, fcol, y = hd.true_bins(corpus, info, chrom, s, s + int(val_chunk), formula)
                acc += float(_true_loss(h, std, true_input(f5, fcol, rung, formula), y, device)) \
                    * y.shape[0]
            per.append(acc / n)
    h.train(was)
    return float(np.mean(per))


def f_val_pass(pred, train_pairs) -> dict:
    """f's chr22 prediction per training pair, once: {(source, target): (loc, disp)} float32[n].

    f is frozen while h trains, so every validation pass reuses this; `validate_fnb` with and
    without it gives the same value bit for bit. Kept as f's raw float32 (loc, disp), 8 bytes a
    bin a pair (about 16 MB per pair on chr22; 242 pairs of g "all" about 3.9 GB)."""
    chrom = lpairs.VAL_CHROMS[0]
    out = {}
    for p in train_pairs:
        key = (p["source_pid"], p["target_pid"])
        if key not in out:
            out[key] = pred.predict(key[0], key[0], key[1], chrom)
    return out


def validate_fnb(h, std, corpus: hd.HCorpus, infos: dict, pred, train_pairs, rung: int,
                 formula: bool, rule: str, device, val_chunk: int = VAL_CHUNK,
                 elems: int = hd.FNB_ELEMS, k_max: int = pushforward.K_MAX,
                 f_val: dict | None = None) -> tuple[float, dict]:
    """(mean over the training pairs of the mean f-NB loss over the whole of chr22, stats with
    n_bins and n_capped of the pass). `f_val` (`f_val_pass`) holds f's chr22 prediction per pair;
    without it f predicts here."""
    chrom = lpairs.VAL_CHROMS[0]
    was = h.training
    h.eval()
    per, st = [], {"n_bins": 0, "n_capped": 0}
    for p in train_pairs:
        src, tgt = p["source_pid"], p["target_pid"]
        info = infos[tgt]
        loc, disp = (f_val[(src, tgt)] if f_val is not None
                     else pred.predict(src, src, tgt, chrom))
        mu, nn = hd.nb_params(pred, loc, disp)
        n = corpus.n_bins(tgt, chrom)
        if mu.shape[0] != n:
            raise ValueError(f"{src}->{tgt} {chrom}: f predicts {mu.shape[0]} bins, target has {n}")
        acc = 0.0
        for s in range(0, n, int(val_chunk)):
            e = min(s + int(val_chunk), n)
            a, b = hd._span(n, s, e, hd.HALO)
            fb = hd.concat_fnb([hd.fnb_bins(corpus, info, chrom, a, b, s, e, mu[a:b], nn[a:b])],
                               [info], [0])
            loss, cst = fnb_batch_loss(h, std, fb, rung, formula, h.noise, rule, device,
                                       elems=elems, k_max=k_max, backward=False)
            acc += loss * (e - s)
            st["n_capped"] += cst["n_capped"]
        st["n_bins"] += n
        per.append(acc / n)
    h.train(was)
    return float(np.mean(per)), st


# ---------------------------------------------------------------------------------------------
# standardiser
# ---------------------------------------------------------------------------------------------


def fit_standardiser_true(src: hd.TrueSource, rung: int, formula: bool) -> hm.Standardiser:
    tb = src.draw(N_STD_WINDOWS)
    fcol = np.zeros(tb.y.shape[0], dtype=np.float32) if formula else None   # never scaled
    x = true_input(tb.feat_r5, fcol, rung, formula)[tb.mask]
    return hm.Standardiser().fit(x, hm.continuous_cols(rung))


def fit_standardiser_fnb(src: hd.FnbSource, rung: int, formula: bool) -> hm.Standardiser:
    """Features at k = f's predicted mean: log1p(mu) and the fixed columns (whose fragment window
    is already the window mean of f's means, the chain's value at k = mu)."""
    fb = src.draw(N_STD_WINDOWS)
    fixed = fb.feat_fixed[:, :hd.N_FIXED_R1] if rung == 1 else fb.feat_fixed
    cols = [np.log1p(fb.mu)[:, None].astype(np.float32), fixed]
    if formula:
        cols.append(np.zeros((fb.y.shape[0], 1), dtype=np.float32))           # never scaled
    return hm.Standardiser().fit(np.concatenate(cols, axis=1), hm.continuous_cols(rung))


# ---------------------------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------------------------


def run_training(manifest, covariates, data_dir, ctl_cache, runs_dir, variant, rung, noise,
                 source, seed, fnb_run=None, fnb_noise_loss="expected", max_steps=None,
                 eval_every=None, device="auto", control_rl_tsv=products.CONTROL_RL_TSV,
                 val_chunk=VAL_CHUNK, elems=hd.FNB_ELEMS, k_max=pushforward.K_MAX,
                 **overrides) -> Path:
    """Train one h into `<runs_dir>/<h_name>/`; returns the run dir. Skips if TRAIN_DONE.

    `overrides` may replace any DEFAULTS entry (tests only; the CLI exposes max_steps, eval_every
    and device). `val_chunk`, `elems`, `k_max` are for tests.
    """
    t0 = time.time()
    rung, noise, seed = int(rung), int(noise), int(seed)
    if source not in SOURCES:
        raise ValueError(f"source {source!r} not in {SOURCES}")
    if (variant, rung, noise) not in tasks.H_LEARNED:
        raise ValueError(f"(variant, rung, noise) = {(variant, rung, noise)} not in "
                         f"hmap.tasks.H_LEARNED")
    if fnb_noise_loss not in hm.NOISE_RULES:
        raise ValueError(f"fnb_noise_loss {fnb_noise_loss!r} not in {hm.NOISE_RULES}")
    if (source == "fnb") != (fnb_run is not None):
        raise ValueError("source fnb needs --fnb-run, and only fnb takes it")
    cfg = dict(DEFAULTS)
    unknown = set(overrides) - set(cfg)
    if unknown:
        raise KeyError(f"unknown training settings: {sorted(unknown)}")
    cfg.update(overrides)
    if max_steps is not None:
        cfg["max_steps"] = int(max_steps)
    if eval_every is not None:
        cfg["eval_every"] = int(eval_every)
    tag = "true" if source == "true" else Path(fnb_run).name
    name = tasks.h_name(variant, rung, noise, tag, seed)
    run_dir = Path(runs_dir) / name
    if (run_dir / "TRAIN_DONE").is_file():
        print(f"{name}: TRAIN_DONE present, nothing to do")
        return run_dir
    dev = lmodel.pick_device(device)
    formula = is_formula(variant)

    rows = lpairs.read_manifest(manifest)
    infos = products.load(manifest, covariates, control_rl_tsv)
    corpus = hd.HCorpus(data_dir, ctl_cache)
    if source == "true":
        pids = [r["pid"] for r in lpairs.usable_products(rows)]
        first, pred, tp = pids[0], None, None
    else:
        pred = em.load_run(fnb_run, data_dir, covariates, manifest, device=device)
        if pred.config.get("run_name") != tag:
            raise ValueError(f"{fnb_run}: the f run's config names {pred.config.get('run_name')!r}")
        tp = list(pred.config["train_pairs"])
        first, pids = tp[0]["source_pid"], None
    chroms, lens = ltrain.training_chroms(corpus.data, first, hd.SPACE_COUNTS)

    def make_source():
        if source == "true":
            return hd.TrueSource(corpus, infos, pids, chroms, lens, cfg["window"], seed,
                                 with_formula=formula)
        return hd.FnbSource(corpus, infos, pred, tp, chroms, lens, cfg["window"], seed)

    std = (fit_standardiser_true(make_source(), rung, formula) if source == "true"
           else fit_standardiser_fnb(make_source(), rung, formula))
    src = make_source()

    torch.manual_seed(seed)
    np.random.seed(seed)
    h = hm.HModel(variant, rung, bool(noise), hm.N_IN[rung]).to(dev)
    opt = torch.optim.Adam(h.parameters(), lr=cfg["lr"])
    warm = max(1, int(cfg["warmup"]))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))

    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "h_name": name, "variant": variant, "rung": rung, "noise": noise,
        "source": "true" if source == "true" else f"fnb:{tag}", "seed": seed,
        "fnb_run": None if source == "true" else str(fnb_run),
        "fnb_noise_loss": None if source == "true" else fnb_noise_loss,
        "features": feature_names(variant, rung), "n_in": hm.N_IN[rung],
        "DEFAULTS": cfg, "device": str(dev),
        "train_chroms": chroms, "val_chroms": list(lpairs.VAL_CHROMS),
        "sampling": ("pid uniform" if source == "true" else "training pair uniform")
                    + ", chromosome proportional to length, start uniform, default_rng(seed)",
        "standardiser": f"fitted on the first {N_STD_WINDOWS} windows of a fresh source of the "
                        f"same seed, frozen" + ("" if source == "true" else
                                                "; features at k = f's predicted mean"),
        "losses": {"true": "push: (mu - log1p P)^2; noise: Gaussian NLL of log1p P",
                   "fnb": "push: sum_k NB(k) (mu_k - log1p P)^2; noise: expected = sum_k NB(k) "
                          "NLL_k, mixture = -log sum_k NB(k) N(log1p P; mu_k, sigma_k)"},
        "nb_support": {"tail": pushforward.TAIL, "k_max": int(k_max), "chunk_elems": int(elems)},
        "halo": hd.HALO, "val_chunk": int(val_chunk),
        "git_sha": ltrain.git_sha(), "manifest": str(manifest),
        "manifest_md5": lmodel.md5_path(manifest), "covariates": str(covariates),
        "covariates_md5": lmodel.md5_path(covariates),
        "control_rl_tsv": str(control_rl_tsv), "control_rl_md5": lmodel.md5_path(control_rl_tsv),
        "data_dir": str(data_dir), "ctl_cache": str(ctl_cache),
    }
    if source == "true":
        config["pids"] = pids
    else:
        config["train_pairs"] = tp
    (run_dir / "config.json").write_text(json.dumps(config, indent=1) + "\n")

    f_val, f_pass_seconds = None, None
    if source == "fnb":
        tf = time.time()
        f_val = f_val_pass(pred, tp)
        f_pass_seconds = time.time() - tf
    val_seconds = []

    def validate():
        tv = time.time()
        if source == "true":
            out = validate_true(h, std, corpus, infos, pids, rung, formula, dev, val_chunk), None
        else:
            out = validate_fnb(h, std, corpus, infos, pred, tp, rung, formula, fnb_noise_loss,
                               dev, val_chunk, elems, k_max, f_val=f_val)
        val_seconds.append(time.time() - tv)
        return out

    t_setup = time.time() - t0
    t1 = time.time()
    log = []
    val, vst = validate()
    log.append((0, float("nan"), val, time.time() - t1))
    best = {"val": val, "step": 0, "h": copy.deepcopy(h.state_dict())}
    n_capped_val = 0 if vst is None else int(vst["n_capped"])
    tr = {"n_bins": 0, "n_capped": 0, "n_k_sum": 0, "n_k_max": 0}
    bad, losses, step = 0, [], 0
    h.train()
    for step in range(1, cfg["max_steps"] + 1):
        opt.zero_grad(set_to_none=True)
        if source == "true":
            loss = true_batch_loss(h, std, src.draw(cfg["batch"]), rung, formula, dev)
        else:
            loss, st = fnb_batch_loss(h, std, src.draw(cfg["batch"]), rung, formula, bool(noise),
                                      fnb_noise_loss, dev, elems=elems, k_max=k_max)
            for key in ("n_bins", "n_capped", "n_k_sum"):
                tr[key] += st[key]
            tr["n_k_max"] = max(tr["n_k_max"], st["n_k_max"])
        if not np.isfinite(loss):
            raise FloatingPointError(f"{name}: non-finite loss at step {step}")
        torch.nn.utils.clip_grad_norm_(h.parameters(), cfg["grad_clip"])
        opt.step()
        sched.step()
        losses.append(loss)
        if step % cfg["eval_every"] == 0 or step == cfg["max_steps"]:
            val, _ = validate()
            log.append((step, float(np.mean(losses)), val, time.time() - t1))
            losses = []
            if val < best["val"]:
                best = {"val": val, "step": step, "h": copy.deepcopy(h.state_dict())}
                bad = 0
            else:
                bad += 1
                if bad >= cfg["patience"]:
                    break
    t_train = time.time() - t1

    h.load_state_dict(best["h"])
    tmp = run_dir / "ckpt.pt.tmp"
    hm.save_ckpt(tmp, h, std, config, best["step"], best["val"])
    os.replace(tmp, run_dir / "ckpt.pt")
    with open(run_dir / "train_log.tsv", "w", encoding="utf-8") as fh:
        fh.write("step\tloss\tval_loss\tseconds\n")
        for s, lo, v, sec in log:
            fh.write(f"{s}\t{lo!r}\t{v!r}\t{sec:.3f}\n")
    timing = {"setup": t_setup, "train": t_train, "total": time.time() - t0,
              "peak_rss_mb": ltrain.peak_rss_mb(), "steps_run": step,
              "best_step": int(best["step"]), "val_loss": float(best["val"]),
              "val_seconds": val_seconds}
    if source == "fnb":
        timing["f_pass_seconds"] = f_pass_seconds
        timing.update(n_k_mean=tr["n_k_sum"] / max(1, tr["n_bins"]), n_k_max=tr["n_k_max"],
                      n_bins_train=tr["n_bins"], n_capped_train=int(tr["n_capped"]),
                      n_capped_val=n_capped_val, n_capped=int(tr["n_capped"]) + n_capped_val)
    (run_dir / "timing.json").write_text(json.dumps(timing, indent=1) + "\n")
    capped = f" n_capped {timing['n_capped']}" if source == "fnb" else ""
    (run_dir / "TRAIN_DONE").write_text(
        f"{name} best_step {best['step']} val_loss {best['val']!r}{capped}\n")
    print(f"{name}: best_step {best['step']} val_loss {best['val']:.6g} steps {step} "
          f"train {t_train:.1f}s total {timing['total']:.1f}s{capped}")
    return run_dir


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def knobs(p):
        p.add_argument("--fnb-noise-loss", choices=hm.NOISE_RULES, default="expected")
        p.add_argument("--max-steps", type=int, default=None)
        p.add_argument("--eval-every", type=int, default=None)
        p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
        p.add_argument("--control-rl-tsv", default=str(products.CONTROL_RL_TSV))

    common = ("manifest", "covariates", "data_dir", "ctl_cache", "runs_dir")
    p = sub.add_parser("train")
    for a in common + ("variant",):
        p.add_argument(a)
    p.add_argument("rung", type=int, choices=(1, 5))
    p.add_argument("noise", type=int, choices=(0, 1))
    p.add_argument("source", choices=SOURCES)
    p.add_argument("seed", type=int)
    p.add_argument("--fnb-run", default=None)
    knobs(p)
    p = sub.add_parser("train-index")
    for a in common:
        p.add_argument(a)
    p.add_argument("table", choices=TABLES)
    p.add_argument("index", type=int)
    p.add_argument("--eta-runs", default=None)
    p.add_argument("--fnb-seeds", type=int, choices=(1, 3), default=1)
    knobs(p)
    args = ap.parse_args(argv)

    if args.cmd == "train-index":
        t = task_of_index(args.table, args.index, args.eta_runs, args.fnb_seeds)
        variant, rung, noise, source, seed, fnb_run = (t["variant"], t["rung"], t["noise"],
                                                       t["source"], t["seed"], t["fnb_run"])
    else:
        variant, rung, noise, source, seed, fnb_run = (args.variant, args.rung, args.noise,
                                                       args.source, args.seed, args.fnb_run)
    run_training(args.manifest, args.covariates, args.data_dir, args.ctl_cache, args.runs_dir,
                 variant, rung, noise, source, seed, fnb_run=fnb_run,
                 fnb_noise_loss=args.fnb_noise_loss, max_steps=args.max_steps,
                 eval_every=args.eval_every, device=args.device,
                 control_rl_tsv=args.control_rl_tsv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
