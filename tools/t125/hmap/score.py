"""t125 part 2 — score an h alone (on the true X') and in the chain (on f's predicted NB).

**h versions** (`HSpec`): `"formula:bin"`, `"formula:frag"` (the caller's tail, nothing learned;
P2b) or a learned h run dir (P2e: `ckpt.pt` with P2d's keys, `TRAIN_DONE`). `load_h(spec)` gives a
callable: for a batch of bins with an enumerated count axis k [n, K] it returns mu [n, K] =
predicted log(1 + P), and log_sigma [n, K] for own-noise h (None for push only).

**Inputs per bin** (P2a's features, the PI's chain rule): the bin's count k (enumerated), the
product's control, the assay; at rung 5 the fragment-window treatment mean = (k + the neighbours'
treatment) / bins kept (`features.frag_rest`, `features.window_count`), every other treatment
window and the formula's no-control λ window read the treatment TRACK: the true counts alone, f's
predicted NB mean in the chain. The formula variants add log1p of the fragment-window formula's P
at k as the last column. Alone, k is the true count (a point support, K = 1).

**Scores** (design note "Scoring"; PI decisions 2026-10-11, plan/T125_STATUS.md):
  push only   the pushed set {P_hat(mu(k)), NB(k)}, P_hat = expm1(mu) floored at 0, from
              `pushforward.nb_support` (tail 1e-6, k_max 4096): `crps_*` = its exact CRPS on P
              (sorted one-pass pair sum), `crps_log_*` = its exact CRPS on log(1 + P) (log1p is
              monotone: same sort), the mean = sum_k NB(k) P_hat(k).
  own noise   the mixture sum_k NB(k) N(mu_k, sigma_k^2) on log(1 + P): `crps_log_*` = its exact
              CRPS (`pushforward.crps_gauss_mixture`); the native number is the point CRPS of
              the P-scale mean sum_k NB(k) (exp(mu_k + sigma_k^2 / 2) - 1) (decision (iv) (a)),
              written as `crps_mean_point_*` so it never shares a key with push's proper CRPS.
  both        `mse_log1p_*` = (log1p(mean_P) - log1p(y))^2 and `mse_*` = (mean_P - y)^2.
Subsets all / nonzero / top1 of the true P (`baseline_rungs.subset_index`, top 1 % by y), on
chr19 + chr21 with blacklist bins removed (`ladder.data.read_blacklist_flags`); per record the
subset means, the non-finite bin counts (`n_crps_nonfinite` or `n_crps_mean_point_nonfinite`, and
`n_crps_log_nonfinite`; a subset mean over a non-finite bin is None, as t118), and the explosion
flag `crps_all_gt_20` (own noise: `crps_mean_point_all_gt_20`) that aggregation counts per
(pair, seed). The full key list is in `score_context`.

**Splits** (the t118 rule: trained/score/all; `baseline_rungs.scale_split`'s search): c* on a
2^c grid (-6..6 step 0.25, then +-0.25 step 0.01) on the seeded 20 000-bin subsample, the CRPS
at c* on every bin.
  native, push       the whole pushed set scaled, {2^c P_hat(k)} (as t118 p-space scales the
                     log-normal median with sigma held, i.e. the whole law); the pair term scales
                     by 2^c.
  native, own noise  `scale_split(mean_P, y, |m - y|)` of its point CRPS (`crps_mean_point_*`).
  log scale, both    (PI 2026-10-11) log1p(P) shifted by c ln2, i.e. 1 + P scaled by 2^c: push =
                     {log1p P_hat(k) + c ln2}, own noise = every mu_k + c ln2 with sigma held;
                     the pair term is unchanged, so only the first term is re-evaluated
                     (`crps_log_oracle_scaled_all`, `crps_log_scale_error_all`,
                     `crps_log_c_star_all`).

**Memory.** Bins are sorted by their NB support size and processed in chunks of at most
`CHUNK_BINS` bins and `ELEM_BUDGET` bin x k elements (one capped row is 4097 wide; it no longer
widens 65 536 others). Rows whose support exceeds `K_MAX` are capped by `nb_support`; their count
is written per record (`n_capped`) and per file (`n_capped_total`). The NB support, the fixed
features and the formula λ are built once per (pair, chromosome) and shared by every h.

**Speed (t125-P2h, no number changes).** On a CUDA device `score_context` runs
`score_context_fast`: chunks of up to FAST_ELEM_BUDGET bins x k (the same sort by support size),
each chunk's NB support computed on its distinct (mu, n) rows, the formula P from scipy on the
distinct (λ, floor(pileup)) pairs (kept per pair across chunks), and every per-(bin, k) step
(features, h, sort, sums, the mixture pair term) in torch on the device, float64 except the h
forward (float32, as before). Only summation order, the chunk padding of the NB weights and the
device's libm differ from the numpy reference, which stays the CPU path.

**Merged mixture** (chain only, `--mix-merge G`, PI 2026-10-16/17): the own-noise h's crps_log
pair sum over at most G moment-matched components per bin (`pushforward.merge_mixture_t`); every
such record carries `crps_mix_merged`, `crps_mix_groups`, `n_mix_merged_bins`. The chain measures
the merge's own error and writes `<out_dir>/<f_run>/merge_error.json`: {f_run_name, mix_groups,
mix_check_bins, check (the rule in words), h: {h_name: {variant, rung, noise, source, kind
("<variant>_r<rung>_noise"), n_pairs, n_bins, support_median, support_max, rel_err_median,
rel_err_q99, rel_err_max, signed_rel_err_mean, crps_log_exact_mean, crps_log_merged_mean,
avg_rel_err}}}, over the MIX_CHECK_BINS widest merged bins of every pair.

    python tools/t125/hmap/score.py alone <manifest> <covariates.tsv> <data_dir> <ctl_cache> <blacklist.bed> <h_spec> <out_dir> [--workers N]
    python tools/t125/hmap/score.py chain <manifest> <covariates.tsv> <data_dir> <ctl_cache> <blacklist.bed> <f_run_dir> <out_dir> --h <h_spec> [--h ...] [--workers N]
      both: [--control-rl-tsv PATH] (default: products.CONTROL_RL_TSV) [--device auto|cpu|cuda] (default auto, as hmap/train.py)
      chain: [--mix-merge G] own-noise h: mixture CRPS over <= G merged components per bin (0 = exact, default)

alone → `<out_dir>/<h_name>/scores.json` + `ALONE_DONE`, one record per usable product.
chain → `<out_dir>/<f_run_name>/<h_name>/scores.json` per h, one record per training pair of f;
`<out_dir>/<f_run_name>/CHAIN_DONE` (the h names, one per line) once every listed h has its
scores.json. An h whose scores.json exists is not scored again. Each pair's records are kept in
`<out_dir>/<f_run_name>.partial/pair_<i>.json` as soon as it is scored (t125-P2h), so a task
stopped by its walltime resumes where it stopped (a kept file counts only if it holds every h of
the call). With a CUDA device the jobs run in one process (no fork). Workers are forked (Linux;
on macOS a forked torch forward can segfault, so run learned h with --workers 1 there); a dead
worker fails the call, never hangs it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.special import ndtr

_TOOLS = Path(__file__).resolve().parents[2]
for _p in (_TOOLS / "t118", _TOOLS / "t125"):      # import convention (plan): t118 and t125
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402

import baseline_rungs as br  # noqa: E402
from ladder import data as ldata  # noqa: E402
from ladder import model as lmodel  # noqa: E402
from ladder import pairs as lpairs  # noqa: E402
from ladder import score as lscore  # noqa: E402
from hmap import data as hd  # noqa: E402
from hmap import features, formula, products  # noqa: E402
from hmap import model as hmodel  # noqa: E402
from hmap import pushforward as pf  # noqa: E402

SCORE_CHROMS = lpairs.SCORE_CHROMS
SUBSETS = br.SUBSETS
FORMULA_SPECS = {"formula:bin": "bin", "formula:frag": "frag"}
CHUNK_BINS = 65_536
ELEM_BUDGET = 1 << 20          # bins x k per chunk (a rung-5 feature block is ~92 MB at float32)
#: the fast (device) path's chunks: 2^22 bins x k (float64 temporaries 32 MB each, the rung-5
#: features 370 MB, the h's 2 x 64 hidden layers 2 GB: inside a 10 GB MIG slice)
FAST_ELEM_BUDGET = 1 << 22
#: the merged mixture's own check, per (pair, own-noise h): this many widest-support merged bins
MIX_CHECK_BINS = 64
MERGE_ERROR_DOC = ("per pair, the MIX_CHECK_BINS bins of widest support among those with more "
                   "than mix_groups components (chr19 + chr21, blacklist removed): crps_log of the "
                   "exact mixture and of the merged one (the value in the records); relative "
                   "error per bin = (merged - exact) / |exact|; avg_rel_err = |mean merged - mean "
                   "exact| / |mean exact| over those bins")
K_MAX = pf.K_MAX
TAIL = pf.TAIL
CRPS_EXPLODE = 20.0
N_FIX_R1 = len(features.FEATURES_R1) - 1                       # 8: the fixed rung-1 columns
KFRAG_FIX = features.FEATURES_R5.index("log1p_k_frag") - 1     # 8: its index among the fixed
N_FIX_R5 = len(features.FEATURES_R5) - 1                       # 20

# hooks the tests replace (module attributes, read at call time)
nb_support = pf.nb_support
support_size = pf.support_size

#: state the forked workers read (set before the pool is created)
_CTX: dict = {}


# ---------------------------------------------------------------------------------------------
# h versions
# ---------------------------------------------------------------------------------------------


class FormulaH:
    """The caller's tail at k: mu = log1p(P(k)), push only."""

    kind, noise, rung, formula = "formula", False, None, False

    def __init__(self, window: str):
        self.window = window
        self.spec = f"formula:{window}"
        self.name = f"h_formula_{window}"

    def info(self) -> dict:
        return {"h_name": self.name, "spec": self.spec, "kind": self.kind, "variant": "formula",
                "window": self.window, "rung": None, "noise": 0, "source": None, "seed": None}

    def __call__(self, batch):
        return np.log1p(batch.get(f"P_{self.window}")), None


class LearnedH:
    """A trained h (P2d model, P2e run dir); features assembled per (bin, k)."""

    kind = "learned"

    def __init__(self, run_dir, device="cpu"):
        run_dir = Path(run_dir)
        if not (run_dir / "TRAIN_DONE").is_file():
            raise FileNotFoundError(f"{run_dir}: no TRAIN_DONE")
        self.device = lmodel.pick_device(device) if isinstance(device, str) else device
        h, std, cfg, step, val = hmodel.load_ckpt(run_dir / "ckpt.pt", device=self.device)
        self.h = h.eval()
        self.std = std
        jcfg = {}
        if (run_dir / "config.json").is_file():
            jcfg = json.loads((run_dir / "config.json").read_text())
        self.cfg = {**jcfg, **cfg}
        self.name = self.cfg.get("h_name") or run_dir.name
        self.spec = str(run_dir)
        self.noise, self.rung, self.formula = bool(h.noise), int(h.rung), bool(h.formula)
        self.step, self.val_loss = step, val

    def info(self) -> dict:
        c = self.cfg
        return {"h_name": self.name, "spec": self.spec, "kind": self.kind,
                "variant": self.h.variant, "rung": self.rung, "noise": int(self.noise),
                "source": c.get("source"), "seed": c.get("seed"), "fnb_run": c.get("fnb_run"),
                "step": self.step, "val_loss": self.val_loss}

    def standardise(self, feat: np.ndarray) -> np.ndarray:
        """P2d's standardiser; fitted on all n_feat columns or on the n_in columns only (the
        formula column, last, is never scaled either way)."""
        width = self.std.mean.shape[0]
        if width == feat.shape[-1]:
            return self.std.apply(feat)
        if self.formula and width == self.h.n_in:
            return np.concatenate([self.std.apply(feat[..., :width]), feat[..., width:]], axis=-1)
        raise ValueError(f"{self.name}: standardiser width {width}, features {feat.shape[-1]}")

    def features(self, batch) -> np.ndarray:
        fx = batch.get("fixed")
        n, K = batch.k.shape
        parts = [batch.get("lk")[..., None],
                 np.broadcast_to(fx[:, None, :N_FIX_R1], (n, K, N_FIX_R1))]
        if self.rung == 5:
            parts += [batch.get("lkfrag")[..., None],
                      np.broadcast_to(fx[:, None, KFRAG_FIX + 1:N_FIX_R5],
                                      (n, K, N_FIX_R5 - KFRAG_FIX - 1))]
        if self.formula:
            parts.append(np.log1p(batch.get("P_frag")).astype(np.float32)[..., None])
        return np.concatenate(parts, axis=-1).astype(np.float32, copy=False)

    @torch.no_grad()
    def __call__(self, batch):
        feat = self.standardise(self.features(batch))
        mu, ls = self.h(torch.from_numpy(np.ascontiguousarray(feat)).to(self.device))
        mu = mu.double().cpu().numpy()
        return mu, (None if ls is None else ls.double().cpu().numpy())


def load_h(spec, device="cpu"):
    """`"formula:bin" | "formula:frag" | <h_run_dir>` -> a FormulaH or LearnedH."""
    s = str(spec)
    if s.startswith("formula:"):
        if s not in FORMULA_SPECS:
            raise ValueError(f"unknown formula h {s!r}; one of {tuple(FORMULA_SPECS)}")
        return FormulaH(FORMULA_SPECS[s])
    return LearnedH(s, device)


# ---------------------------------------------------------------------------------------------
# per-bin context and batches
# ---------------------------------------------------------------------------------------------


def f_nb(pred, src: str, tgt: str, chrom: str):
    """f's NB on one chromosome: (mu, n) float64, as t118 scores counts (`pred.mean`, the size
    exp(disp) clipped to the loss's bounds)."""
    loc, disp = pred.predict(src, src, tgt, chrom)
    return (np.asarray(pred.mean(loc), np.float64),
            lscore.spread_of("counts", np.asarray(disp, np.float64)))


def build_context(info, per_chrom: list[dict], blacklist, need_fixed: bool) -> dict:
    """Concatenate the kept bins of the score chromosomes.

    per_chrom: {chrom, treat [n] (true counts, or f's mean), ctl [n] | None, y [n] (true P),
    and either kpt [n] (a point support) or mu, n_size [n] (the NB)}. Window terms are built on
    the whole chromosome before the blacklist bins are dropped.
    """
    w_frag = features.frag_bins(info.fraglen)
    cols: dict = {k: [] for k in ("y", "lam", "frest_f", "frest_k", "fcnt", "fixed", "kpt", "mu",
                                  "n_size")}
    n_black = 0
    for d in per_chrom:
        treat = np.asarray(d["treat"], np.float64)
        n = treat.shape[0]
        if np.asarray(d["y"]).shape != (n,):
            raise ValueError(f"{info.pid}/{d['chrom']}: P {np.shape(d['y'])} != ({n},)")
        keep = ~ldata.read_blacklist_flags(blacklist, d["chrom"], n)
        n_black += int(n - keep.sum())
        cols["y"].append(np.asarray(d["y"], np.float64)[keep])
        cols["lam"].append(formula.lambda_track(info, d["ctl"], treat)[keep])
        cols["frest_f"].append(formula.frag_rest(treat, info)[keep])
        cols["frest_k"].append(features.frag_rest(treat, w_frag)[keep])
        cols["fcnt"].append(features.window_count(n, w_frag)[keep].astype(np.float64))
        if need_fixed:
            cols["fixed"].append(features.fixed_features(info, treat, d["ctl"], 5)[keep])
        if "kpt" in d:
            cols["kpt"].append(np.asarray(d["kpt"]).astype(np.int64)[keep])
        else:
            cols["mu"].append(np.asarray(d["mu"], np.float64)[keep])
            cols["n_size"].append(np.asarray(d["n_size"], np.float64)[keep])
    ctx = {k: np.concatenate(v) for k, v in cols.items() if v}
    ctx["info"] = info
    ctx["mode"] = "point" if "kpt" in ctx else "nb"
    ctx["n_blacklisted"] = n_black
    return ctx


def support_sizes(ctx) -> np.ndarray:
    """int64[N]: support points per bin, capped at K_MAX + 1 (a capped row's width)."""
    if ctx["mode"] == "point":
        return np.ones(ctx["y"].shape[0], np.int64)
    return np.minimum(support_size(ctx["mu"], ctx["n_size"], tail=TAIL, k_max=K_MAX), K_MAX + 1)


def support_rows(ctx, rows):
    if ctx["mode"] == "point":
        k = ctx["kpt"][rows][:, None]
        return k, np.ones(k.shape), np.ones(k.shape, bool)
    return nb_support(ctx["mu"][rows], ctx["n_size"][rows], tail=TAIL, k_max=K_MAX)


def chunk_rows(sizes: np.ndarray, idx: np.ndarray, budget: int | None = None):
    """Yield the bins of `idx` sorted by support size, in chunks of at most CHUNK_BINS bins and
    at most ELEM_BUDGET (or `budget`) bins x (the chunk's widest support), or one bin."""
    budget = ELEM_BUDGET if budget is None else int(budget)
    order = idx[np.argsort(sizes[idx], kind="stable")]
    w = sizes[order]
    s, n = 0, order.size
    while s < n:
        L = min(CHUNK_BINS, n - s)
        over = np.flatnonzero(np.arange(1, L + 1) * w[s:s + L] > budget)
        e = s + max(1, int(over[0]) if over.size else L)
        yield order[s:e]
        s = e


class Batch:
    """The per-(bin, k) inputs of one chunk, built once and shared by every h."""

    def __init__(self, ctx, rows, k):
        self.ctx, self.rows, self.k = ctx, rows, k
        self._c: dict = {}

    def get(self, name):
        if name not in self._c:
            c, r, k = self.ctx, self.rows, self.k
            if name == "lk":
                v = features.k_column(k)
            elif name == "lkfrag":
                v = np.log1p((k + c["frest_k"][r, None]) / c["fcnt"][r, None]).astype(np.float32)
            elif name == "fixed":
                v = c["fixed"][r]
            elif name == "P_bin":
                v = formula.P_enumerated("bin", c["info"], k, c["lam"][r])
            elif name == "P_frag":
                v = formula.P_enumerated("frag", c["info"], k, c["lam"][r], c["frest_f"][r])
            else:
                raise KeyError(name)
            self._c[name] = v
        return self._c[name]


# ---------------------------------------------------------------------------------------------
# scores of one context
# ---------------------------------------------------------------------------------------------


def p_hat(mu: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        return np.maximum(np.expm1(mu), 0.0)


def sorted_set(v, w):
    """(v sorted per row, its normalised weights, coef = C_below - C_above) of {v, w}; padding
    (w = 0) is set to v = 0, weight 0. sum_ij w_i w_j |v_i - v_j| = 2 sum_i w_i v_i coef_i."""
    pos = w > 0
    v = np.where(pos, v, 0.0)
    w = np.where(pos, w, 0.0)
    w = w / w.sum(axis=1, keepdims=True)
    order = np.argsort(v, axis=1, kind="stable")
    vs = np.take_along_axis(v, order, axis=1)
    ws = np.take_along_axis(w, order, axis=1)
    coef = 2.0 * (np.cumsum(ws, axis=1) - ws) + ws - 1.0
    return vs, ws, coef


def _set_crps(vs, ws, coef, y, a=1.0):
    """CRPS of {a v, w} against y (rows already sorted by `sorted_set`; a > 0 keeps the order)."""
    with np.errstate(invalid="ignore"):
        return (ws * np.abs(a * vs - y[:, None])).sum(axis=1) - a * (ws * vs * coef).sum(axis=1)


LN2 = math.log(2.0)


def _grid(fit) -> float:
    """c* by `baseline_rungs.scale_split`'s search: 2^c grid -6..6 step 0.25, then +-0.25 step
    0.01 around the best."""
    c = float(min(np.arange(-6.0, 6.001, 0.25), key=fit))
    return float(min(np.arange(c - 0.25, c + 0.2501, 0.01), key=fit))


def _flat(vs, ws, y):
    """The positive-weight entries of sorted rows: (v, w, the row's y) per entry."""
    pos = ws > 0
    return vs[pos], ws[pos], np.broadcast_to(y[:, None], vs.shape)[pos]


def _mix_e1(mu, sg, w, t, shift: float = 0.0) -> np.ndarray:
    """sum_k w_k A(t - (mu_k + shift), sigma_k^2): the first term of the mixture CRPS with every
    mu shifted (the pair term does not move under a common shift). A as `pushforward`."""
    pos = w > 0
    mu = np.where(pos, mu, 0.0)
    sg = np.where(pos, sg, 1.0)
    w = np.where(pos, w, 0.0)
    w = w / w.sum(axis=1, keepdims=True)
    m = t[:, None] - (mu + shift)
    z = m / sg
    A = m * (2.0 * ndtr(z) - 1.0) + 2.0 * sg * np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return (w * A).sum(axis=1)


class _Fit:
    """The c-search of one h on the seeded subsample, for the two splits that move a whole set:
    native push ({2^c P_hat(k)}: the pair term scales by 2^c) and log scale (log1p(P) + c ln2:
    the pair term is unchanged; push = the pushed set {log1p P_hat(k)}, own noise = the mixture
    with every mu_k + c ln2, sigma held)."""

    def __init__(self, noise: bool):
        self.noise = noise
        self.nat, self.log = [], []      # flattened entries per chunk
        self.nat_pair = 0.0              # sum over bins of the half pair term (native)
        self.log_pair = 0.0              # the same on the log scale
        self.n = 0

    def add_push(self, vs, ws, coef, y, ly):
        self.n += vs.shape[0]
        self.nat.append(_flat(vs, ws, y))
        self.nat_pair += float((ws * vs * coef).sum())
        lv = np.log1p(vs)
        self.log.append(_flat(lv, ws, ly))
        self.log_pair += float((ws * lv * coef).sum())

    def add_noise(self, mu, sg, w, ly, crps_log):
        self.n += mu.shape[0]
        pos = w > 0
        wn = np.where(pos, w, 0.0) / np.where(pos, w, 0.0).sum(axis=1, keepdims=True)
        self.log.append((mu[pos], sg[pos], wn[pos], np.broadcast_to(ly[:, None], mu.shape)[pos]))
        self.log_pair += float((_mix_e1(mu, sg, w, ly) - crps_log).sum())

    def c_native(self) -> float:
        v, w, y = (np.concatenate(x) for x in zip(*self.nat))

        def fit(c):
            a = 2.0 ** c
            with np.errstate(invalid="ignore"):
                return float((w @ np.abs(a * v - y) - a * self.nat_pair) / self.n)
        return _grid(fit)

    def c_log(self) -> float:
        if not self.noise:
            v, w, y = (np.concatenate(x) for x in zip(*self.log))

            def fit(c):
                with np.errstate(invalid="ignore"):
                    return float((w @ np.abs(v + c * LN2 - y) - self.log_pair) / self.n)
            return _grid(fit)
        mu, sg, w, t = (np.concatenate(x) for x in zip(*self.log))

        def fit(c):
            m = t - (mu + c * LN2)
            z = m / sg
            with np.errstate(invalid="ignore", over="ignore"):
                A = m * (2.0 * ndtr(z) - 1.0) + 2.0 * sg * np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
                return float((w @ A - self.log_pair) / self.n)
        return _grid(fit)


def _fmean(a: np.ndarray, idx: np.ndarray):
    if idx.size == 0:
        return None
    v = float(a[idx].mean())
    return v if math.isfinite(v) else None


def _split(crps_all, cs, c, prefix: str) -> dict:
    return {f"{prefix}_oracle_scaled_all": cs,
            f"{prefix}_scale_error_all": (None if cs is None or crps_all is None
                                          else crps_all - cs),
            f"{prefix}_c_star_all": c}


def score_context(hs: list, ctx: dict, device=None, fast: bool | None = None,
                  mix_groups: int | None = None) -> tuple[dict, dict]:
    """({h_name: metric fields}, shared fields {n_k_mean, n_k_max, n_capped}) of one context.

    `fast` (default: True exactly when `device` is a CUDA device) runs `score_context_fast` on
    `device`; otherwise the numpy reference below runs (the CPU path, unchanged by t125-P2h).
    `mix_groups` (the chain's merged mixture, PI 2026-10-16; None = exact) needs the fast path
    and is used for the own-noise h only; shared then holds `n_mix_merged_bins`.

    Keys. Push only: `crps_*` (the proper CRPS of the pushed set on P) with its split
    `crps_oracle_scaled_all`, `scale_error_all`, `c_star_all`, the flag `crps_all_gt_20` and
    `n_crps_nonfinite`. Own noise: the native number is the point CRPS of the mean, so it is
    written under `crps_mean_point_*` (split `crps_mean_point_oracle_scaled_all`,
    `crps_mean_point_scale_error_all`, `crps_mean_point_c_star_all`, flag
    `crps_mean_point_all_gt_20`, `n_crps_mean_point_nonfinite`) and never under the push keys.
    Both: `crps_log_*` (proper, on log1p P) with `crps_log_oracle_scaled_all`,
    `crps_log_scale_error_all`, `crps_log_c_star_all`, `n_crps_log_nonfinite`; `mse_log1p_*`,
    `mse_*`."""
    if fast is None:
        fast = (device is not None and torch.device(device).type == "cuda") or bool(mix_groups)
    if mix_groups and not fast:
        raise ValueError("the merged mixture (mix_groups) runs on the fast path only")
    if fast:
        return score_context_fast(hs, ctx, "cpu" if device is None else device, mix_groups)
    y = ctx["y"]
    N = y.shape[0]
    ly = np.log1p(y)
    sizes = support_sizes(ctx)
    shared = {"n_capped": int(np.count_nonzero(sizes > K_MAX))}
    # phase A: c* of every split that moves a whole set, on the seeded subsample (as
    # scale_split draws it)
    c_nat, c_log = {}, {}
    if N:
        sel = (np.arange(N) if N <= br.SCALE_FIT_BUDGET else
               np.random.default_rng(br.SCALE_SEED).choice(N, br.SCALE_FIT_BUDGET, replace=False))
        fits = {h.name: _Fit(h.noise) for h in hs}
        for rows in chunk_rows(sizes, sel):
            k, w, _ = support_rows(ctx, rows)
            b = Batch(ctx, rows, k)
            for h in hs:
                mu, ls = h(b)
                if h.noise:
                    with np.errstate(over="ignore"):
                        sg = np.exp(ls)
                    fits[h.name].add_noise(mu, sg, w, ly[rows],
                                           pf.crps_gauss_mixture(mu, sg, w, ly[rows]))
                else:
                    fits[h.name].add_push(*sorted_set(p_hat(mu), w), y[rows], ly[rows])
        for h in hs:
            c_log[h.name] = fits[h.name].c_log()
            if not h.noise:
                c_nat[h.name] = fits[h.name].c_native()
    # phase B: every bin
    arr = {h.name: {key: np.full(N, np.nan)
                    for key in ("crps", "crps_log", "mean", "crps_c", "crps_log_c")}
           for h in hs}
    n_k = np.zeros(N, np.int64)
    for rows in chunk_rows(sizes, np.arange(N)):
        k, w, valid = support_rows(ctx, rows)
        n_k[rows] = valid.sum(axis=1)
        b = Batch(ctx, rows, k)
        yy, lyy = y[rows], ly[rows]
        for h in hs:
            mu, ls = h(b)
            a = arr[h.name]
            s = c_log[h.name] * LN2
            if h.noise:
                with np.errstate(over="ignore"):
                    sg = np.exp(ls)
                m = pf.mean_gauss_mixture_logp(mu, sg, w)
                cl = pf.crps_gauss_mixture(mu, sg, w, lyy)
                a["crps_log"][rows] = cl
                a["crps_log_c"][rows] = cl + _mix_e1(mu, sg, w, lyy, s) - _mix_e1(mu, sg, w, lyy)
                a["crps"][rows] = np.abs(m - yy)
            else:
                vs, ws, coef = sorted_set(p_hat(mu), w)
                m = (ws * vs).sum(axis=1)
                lv = np.log1p(vs)
                a["crps"][rows] = _set_crps(vs, ws, coef, yy)
                a["crps_c"][rows] = _set_crps(vs, ws, coef, yy, 2.0 ** c_nat[h.name])
                a["crps_log"][rows] = _set_crps(lv, ws, coef, lyy)
                a["crps_log_c"][rows] = _set_crps(lv + s, ws, coef, lyy)
            a["mean"][rows] = m
    return _records(hs, arr, c_nat, c_log, n_k, y, ly, shared)


def _records(hs, arr: dict, c_nat: dict, c_log: dict, n_k, y, ly, shared: dict,
             mix_groups: int | None = None):
    """The per-h record fields from the per-bin arrays (shared by both paths)."""
    N = y.shape[0]
    if mix_groups:
        shared["n_mix_merged_bins"] = int(np.count_nonzero(n_k > int(mix_groups)))
    shared["n_k_mean"] = float(n_k.mean()) if N else None
    shared["n_k_max"] = int(n_k.max()) if N else None
    sub = br.subset_index(y)
    out = {}
    for h in hs:
        a = arr[h.name]
        nat = "crps_mean_point" if h.noise else "crps"
        with np.errstate(invalid="ignore", divide="ignore"):
            se_log = (np.log1p(a["mean"]) - ly) ** 2
            se = (a["mean"] - y) ** 2
        f: dict = {f"n_{nat}_nonfinite": int(np.count_nonzero(~np.isfinite(a["crps"]))),
                   "n_crps_log_nonfinite": int(np.count_nonzero(~np.isfinite(a["crps_log"])))}
        for s, idx in sub.items():
            f[f"n_{s}"] = int(idx.size)
            f[f"{nat}_{s}"] = _fmean(a["crps"], idx)
            f[f"crps_log_{s}"] = _fmean(a["crps_log"], idx)
            f[f"mse_log1p_{s}"] = _fmean(se_log, idx)
            f[f"mse_{s}"] = _fmean(se, idx)
        if not N:
            f.update(_split(None, None, None, "crps_log"))
            f.update(_split(None, None, None, nat) if h.noise else
                     {"crps_oracle_scaled_all": None, "scale_error_all": None, "c_star_all": None})
        else:
            f.update(_split(f["crps_log_all"], _fmean(a["crps_log_c"], sub["all"]),
                            c_log[h.name], "crps_log"))
            if h.noise:
                r = br.scale_split(a["mean"], y, lambda mm, yy: np.abs(mm - yy))
                f.update(_split(f[f"{nat}_all"], r["crps_oracle_scaled"], r["c_star"], nat))
            else:
                cs = _fmean(a["crps_c"], sub["all"])
                f.update(crps_oracle_scaled_all=cs,
                         scale_error_all=(None if cs is None or f["crps_all"] is None
                                          else f["crps_all"] - cs),
                         c_star_all=c_nat[h.name])
        f[f"{nat}_all_gt_20"] = f[f"{nat}_all"] is not None and f[f"{nat}_all"] > CRPS_EXPLODE
        out[h.name] = f
    return out, shared


# ---------------------------------------------------------------------------------------------
# the fast path on a torch device (t125-P2h): the same numbers, built once per chunk on the device
# ---------------------------------------------------------------------------------------------


def _support_rows_unique(ctx, rows):
    """(k_u int64[u, K], w_u float64[u, K], valid_u bool[u, K], inv int64[r]): `support_rows` of
    the chunk's distinct rows (the same K: the widest row is among them), and the map back."""
    if ctx["mode"] == "point":
        ku, inv = np.unique(ctx["kpt"][rows], return_inverse=True)
        k = ku[:, None]
        return k, np.ones(k.shape), np.ones(k.shape, bool), inv.reshape(-1)
    mu_u, n_u, inv = pf.unique_pairs(ctx["mu"][rows], ctx["n_size"][rows])
    k, w, valid = nb_support(mu_u, n_u, tail=TAIL, k_max=K_MAX)
    return k, w, valid, inv


class BatchT:
    """`Batch` on a torch device: the per-(bin, k) inputs of one chunk, shared by every h, bit for
    bit `Batch`'s on every device: log1p on the host (`hd.log1p_f32`), the formula P from
    `hd.formula_log1p_t` (scipy on the distinct pairs)."""

    def __init__(self, ctx, rows, k_u, inv, device, fcache: dict | None = None):
        self.ctx, self.rows, self.k_u, self.device, self.fcache = ctx, rows, k_u, device, fcache
        self.inv = inv
        self.inv_t = torch.from_numpy(inv).to(device)
        self.shape = (int(rows.shape[0]), int(k_u.shape[1]))
        self._c: dict = {}

    def _np(self, a) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(a)).to(self.device)

    def get(self, name):
        if name not in self._c:
            c, r = self.ctx, self.rows
            if name == "kf":
                v = self._np(self.k_u.astype(np.float64))[self.inv_t]
            elif name == "lk":
                v = self._np(features.k_column(self.k_u))[self.inv_t]
            elif name == "lkfrag":
                v = hd.log1p_kfrag_f32(self.k_u[:, 0][self.inv], self.shape[1], c["frest_k"][r],
                                       c["fcnt"][r], self.device)
            elif name == "fixed":
                v = self._np(c["fixed"][r])
            elif name == "lP_bin":
                v = hd.formula_log1p_t("bin", c["info"], self.get("kf"), c["lam"][r],
                                       cache=self.fcache)
            elif name == "lP_frag":
                v = hd.formula_log1p_t("frag", c["info"], self.get("kf"), c["lam"][r],
                                       self._np(c["frest_f"][r]), cache=self.fcache)
            else:
                raise KeyError(name)
            self._c[name] = v
        return self._c[name]


def _formula_t(h: FormulaH, bt: BatchT):
    return bt.get(f"lP_{h.window}"), None


@torch.no_grad()
def _learned_t(h: LearnedH, bt: BatchT):
    """`LearnedH.__call__` on the device: the same float32 features, the standardiser in float64
    then float32 per value (as `Standardiser.apply` on numpy), the same forward. A column that is
    the same for every k of a bin is standardised once per bin and then repeated (the same value:
    the standardiser works value by value)."""
    fx = bt.get("fixed")
    n, K = bt.shape
    scale, shift = h.std._coeffs()
    width = scale.shape[0]
    F = h.h.n_feat
    if not (width == F or (h.formula and width == h.h.n_in)):
        raise ValueError(f"{h.name}: standardiser width {width}, features {F}")
    sc, sh = bt._np(scale), bt._np(shift)

    def apply(x, cols):
        return (x.to(torch.float64) * sc[cols] + sh[cols]).to(torch.float32)

    if h.rung == 5:
        fixed_cols = list(range(1, 1 + N_FIX_R1)) + list(range(2 + N_FIX_R1, 1 + N_FIX_R5))
        fixed = torch.cat([fx[:, :N_FIX_R1], fx[:, KFRAG_FIX + 1:N_FIX_R5]], dim=1)
    else:
        fixed_cols = list(range(1, 1 + N_FIX_R1))
        fixed = fx[:, :N_FIX_R1]
    feat = torch.empty((n, K, F), dtype=torch.float32, device=bt.device)
    feat[:, :, fixed_cols] = apply(fixed, fixed_cols)[:, None, :]
    feat[:, :, 0] = apply(bt.get("lk"), 0)
    if h.rung == 5:
        feat[:, :, 1 + N_FIX_R1] = apply(bt.get("lkfrag"), 1 + N_FIX_R1)
    if h.formula:
        lp = bt.get("lP_frag").to(torch.float32)
        feat[:, :, F - 1] = apply(lp, F - 1) if width == F else lp
    mu, ls = h.h(feat.to(h.device))
    return mu.to(device=bt.device, dtype=torch.float64), (
        None if ls is None else ls.to(device=bt.device, dtype=torch.float64))


def _call_t(h, bt: BatchT):
    """h on the chunk. A rung-1 learned h sees only (k, the control bin, the assay), so bins with
    the same NB row and control bin get the same output row: h runs once per such row and the
    rows are copied out (`_row_key_t`)."""
    if h.kind == "formula":
        return _formula_t(h, bt)
    key = _row_key_t(h, bt)
    if key is None:
        return _learned_t(h, bt)
    uk, uinv = torch.unique(key, return_inverse=True)
    n = key.shape[0]
    rep = torch.full((uk.numel(),), n, dtype=torch.int64, device=key.device)
    rep.scatter_reduce_(0, uinv, torch.arange(n, device=key.device), reduce="amin")
    rep_np = rep.cpu().numpy()
    sub = BatchT(bt.ctx, bt.rows[rep_np], bt.k_u, bt.inv_t[rep].cpu().numpy(), bt.device,
                 bt.fcache)
    mu, ls = _learned_t(h, sub)
    return mu[uinv], (None if ls is None else ls[uinv])


def _row_key_t(h, bt: BatchT) -> torch.Tensor | None:
    """For a rung-1 learned h (inputs: k, the control bin, the assay), int64 [r] equal exactly for
    bins whose (mu, sigma) rows over k are the same: the NB row and the control bin's bits."""
    if h.kind != "learned" or h.rung != 1 or h.formula:
        return None
    ctl = bt.get("fixed")[:, 0].contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    return bt.inv_t * (1 << 32) + ctl


def _p_hat_t(mu: torch.Tensor) -> torch.Tensor:
    return torch.clamp_min(torch.expm1(mu), 0.0)


def _sorted_set_t(v: torch.Tensor, w: torch.Tensor):
    """`sorted_set` in torch. Only the rows whose positive-weight values are not already in order
    are sorted (stable): the pair-sum identity needs the weighted entries in order, and the
    zero-weight ones add nothing wherever they sit (a monotone h, e.g. every formula, sorts no
    row; the sums then run in another order)."""
    pos = w > 0
    zero = v.new_zeros(())
    v = torch.where(pos, v, zero)
    w = torch.where(pos, w, zero)
    w = w / w.sum(dim=1, keepdim=True)
    run = torch.cummax(torch.where(pos, v, torch.full_like(v, -math.inf)), dim=1).values
    prev = torch.cat([torch.full_like(run[:, :1], -math.inf), run[:, :-1]], dim=1)
    bad = torch.nonzero(~(((~pos) | (v >= prev)).all(dim=1))).squeeze(1)
    vs, ws = v, w
    if bad.numel():
        vs, ws = v.clone(), w.clone()
        vb, order = torch.sort(v[bad], dim=1, stable=True)
        vs[bad] = vb
        ws[bad] = torch.gather(w[bad], 1, order)
    coef = 2.0 * (torch.cumsum(ws, dim=1) - ws) + ws - 1.0
    return vs, ws, coef


def _set_crps_t(vs, ws, coef, y, a: float = 1.0):
    return (ws * torch.abs(a * vs - y[:, None])).sum(dim=1) - a * (ws * vs * coef).sum(dim=1)


def _mix_e1_t(mu, sg, w, t, shift: float = 0.0):
    pos = w > 0
    zero, one = mu.new_zeros(()), mu.new_ones(())
    mu = torch.where(pos, mu, zero)
    sg = torch.where(pos, sg, one)
    w = torch.where(pos, w, zero)
    w = w / w.sum(dim=1, keepdim=True)
    m = t[:, None] - (mu + shift)
    z = m / sg
    A = (m * (2.0 * torch.special.ndtr(z) - 1.0)
         + 2.0 * sg * torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi))
    return (w * A).sum(dim=1)


def _flat_t(vs, ws, y):
    pos = ws > 0
    return vs[pos], ws[pos], y[:, None].expand_as(vs)[pos]


class _FitT(_Fit):
    """`_Fit` on device tensors (the grid search itself is `_grid`, unchanged)."""

    def add_push(self, vs, ws, coef, y, ly):
        self.n += vs.shape[0]
        self.nat.append(_flat_t(vs, ws, y))
        self.nat_pair += float((ws * vs * coef).sum())
        lv = torch.log1p(vs)
        self.log.append(_flat_t(lv, ws, ly))
        self.log_pair += float((ws * lv * coef).sum())

    def add_noise(self, mu, sg, w, ly, crps_log):
        self.n += mu.shape[0]
        pos = w > 0
        wz = torch.where(pos, w, w.new_zeros(()))
        wn = wz / wz.sum(dim=1, keepdim=True)
        self.log.append((mu[pos], sg[pos], wn[pos], ly[:, None].expand_as(mu)[pos]))
        self.log_pair += float((_mix_e1_t(mu, sg, w, ly) - crps_log).sum())

    def c_native(self) -> float:
        v, w, y = (torch.cat(x) for x in zip(*self.nat))

        def fits(cs):
            a = torch.tensor([float(2.0 ** c) for c in cs], dtype=torch.float64, device=v.device)
            tot = _block_dot(w, lambda s_: torch.abs(a[None, :] * v[s_, None] - y[s_, None]))
            return [float((t - float(ai) * self.nat_pair) / self.n)
                    for t, ai in zip(tot.tolist(), a.tolist())]
        return _grid_many(fits)

    def c_log(self) -> float:
        if not self.noise:
            v, w, y = (torch.cat(x) for x in zip(*self.log))

            def fits(cs):
                sh = torch.tensor([float(c) * LN2 for c in cs], dtype=torch.float64,
                                  device=v.device)
                tot = _block_dot(w, lambda s_: torch.abs(v[s_, None] + sh[None, :] - y[s_, None]))
                return [float((t - self.log_pair) / self.n) for t in tot.tolist()]
            return _grid_many(fits)
        mu, sg, w, t = (torch.cat(x) for x in zip(*self.log))

        def fits(cs):
            sh = torch.tensor([float(c) * LN2 for c in cs], dtype=torch.float64, device=mu.device)

            def A(s_):
                m = t[s_, None] - (mu[s_, None] + sh[None, :])
                z = m / sg[s_, None]
                return (m * (2.0 * torch.special.ndtr(z) - 1.0)
                        + 2.0 * sg[s_, None] * torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi))
            tot = _block_dot(w, A, block=1 << 18)
            return [float((x - self.log_pair) / self.n) for x in tot.tolist()]
        return _grid_many(fits)


def _block_dot(w: torch.Tensor, cols, block: int = 1 << 19) -> torch.Tensor:
    """sum_e w_e M[e, g] for every grid point g, M built by `cols(slice)` one block of entries at
    a time (one pass over the entries for the whole grid instead of one per grid point)."""
    tot = None
    for s0 in range(0, w.shape[0], block):
        s_ = slice(s0, s0 + block)
        part = w[s_] @ cols(s_)
        tot = part if tot is None else tot + part
    return tot


def _grid_many(fits) -> float:
    """`_grid` with every grid point of a stage evaluated in one call: the same 2^c grid (-6..6 step
    0.25, then +-0.25 step 0.01 around the best) and the same first-minimum rule."""
    g1 = np.arange(-6.0, 6.001, 0.25)
    v1 = fits(g1)
    c = float(g1[int(np.argmin(v1))])
    g2 = np.arange(c - 0.25, c + 0.2501, 0.01)
    v2 = fits(g2)
    return float(g2[int(np.argmin(v2))])


def score_context_fast(hs: list, ctx: dict, device, mix_groups: int | None = None
                       ) -> tuple[dict, dict]:
    """`score_context` with every per-(bin, k) step on `device` in torch float64 (the h forward in
    float32, as the reference). The same chunks, NB supports (on each chunk's distinct rows),
    features and formula values; only the summation order and the device's libm differ."""
    dev = torch.device(device)
    y = ctx["y"]
    N = y.shape[0]
    ly = np.log1p(y)
    yt, lyt = torch.from_numpy(y).to(dev), torch.from_numpy(ly).to(dev)
    sizes = (np.ones(N, np.int64) if ctx["mode"] == "point" else
             np.minimum(pf.support_size_unique(ctx["mu"], ctx["n_size"], tail=TAIL, k_max=K_MAX,
                                               fn=support_size), K_MAX + 1))
    shared = {"n_capped": int(np.count_nonzero(sizes > K_MAX))}
    fcache: dict = {}

    def chunk(rows):
        k_u, w_u, valid_u, inv = _support_rows_unique(ctx, rows)
        bt = BatchT(ctx, rows, k_u, inv, dev, fcache)
        return bt, torch.from_numpy(w_u).to(dev)[bt.inv_t], valid_u.sum(axis=1)[inv]

    c_nat, c_log = {}, {}
    if N:
        sel = (np.arange(N) if N <= br.SCALE_FIT_BUDGET else
               np.random.default_rng(br.SCALE_SEED).choice(N, br.SCALE_FIT_BUDGET, replace=False))
        # one h at a time: a fit keeps its subsample's (bin, k) entries on the device (up to
        # 20 000 x 4097), so the fits of 50 h would not fit in a 10 GB slice together
        for h in hs:
            fit = _FitT(h.noise)
            for rows in chunk_rows(sizes, sel, FAST_ELEM_BUDGET):
                bt, w, _ = chunk(rows)
                rt = torch.from_numpy(rows).to(dev)
                mu, ls = _call_t(h, bt)
                if h.noise:
                    sg = torch.exp(ls)
                    fit.add_noise(mu, sg, w, lyt[rt], pf.crps_gauss_mixture_t(
                        mu, sg, w, lyt[rt], row_key=_row_key_t(h, bt), max_groups=mix_groups))
                else:
                    fit.add_push(*_sorted_set_t(_p_hat_t(mu), w), yt[rt], lyt[rt])
                del bt, w, mu, ls
            c_log[h.name] = fit.c_log()
            if not h.noise:
                c_nat[h.name] = fit.c_native()
            del fit
    arr = {h.name: {key: np.full(N, np.nan)
                    for key in ("crps", "crps_log", "mean", "crps_c", "crps_log_c")}
           for h in hs}
    n_k = np.zeros(N, np.int64)
    for rows in chunk_rows(sizes, np.arange(N), FAST_ELEM_BUDGET):
        bt, w, nk = chunk(rows)
        n_k[rows] = nk
        rt = torch.from_numpy(rows).to(dev)
        yy, lyy = yt[rt], lyt[rt]
        for h in hs:
            mu, ls = _call_t(h, bt)
            a = arr[h.name]
            s = c_log[h.name] * LN2
            if h.noise:
                sg = torch.exp(ls)
                m = pf.mean_gauss_mixture_logp_t(mu, sg, w)
                cl = pf.crps_gauss_mixture_t(mu, sg, w, lyy, row_key=_row_key_t(h, bt),
                                             max_groups=mix_groups)
                a["crps_log"][rows] = cl.cpu().numpy()
                a["crps_log_c"][rows] = (cl + _mix_e1_t(mu, sg, w, lyy, s)
                                         - _mix_e1_t(mu, sg, w, lyy)).cpu().numpy()
                a["crps"][rows] = torch.abs(m - yy).cpu().numpy()
            else:
                vs, ws, coef = _sorted_set_t(_p_hat_t(mu), w)
                m = (ws * vs).sum(dim=1)
                lv = torch.log1p(vs)
                a["crps"][rows] = _set_crps_t(vs, ws, coef, yy).cpu().numpy()
                a["crps_c"][rows] = _set_crps_t(vs, ws, coef, yy,
                                                float(2.0 ** c_nat[h.name])).cpu().numpy()
                a["crps_log"][rows] = _set_crps_t(lv, ws, coef, lyy).cpu().numpy()
                a["crps_log_c"][rows] = _set_crps_t(lv + s, ws, coef, lyy).cpu().numpy()
            a["mean"][rows] = m.cpu().numpy()
    if mix_groups and N and any(h.noise for h in hs):
        shared["mix_check"] = _mix_check(hs, ctx, sizes, arr, lyt, dev, mix_groups)
    return _records(hs, arr, c_nat, c_log, n_k, y, ly, shared, mix_groups)


def _mix_check(hs, ctx, sizes, arr, lyt, dev, mix_groups: int) -> dict:
    """The merge's own error on this context: the MIX_CHECK_BINS widest-support bins with more than
    mix_groups components, their crps_log exact and merged ({h_name: {support, exact, merged}}
    for the own-noise h; the merged values are the ones written to the records)."""
    big = np.flatnonzero(sizes > int(mix_groups))
    if big.size == 0:
        return {h.name: {"support": [], "exact": [], "merged": []} for h in hs if h.noise}
    pick = big[np.argsort(sizes[big], kind="stable")[::-1][:MIX_CHECK_BINS]]
    pick = np.sort(pick)
    out = {}
    for rows in chunk_rows(sizes, pick, FAST_ELEM_BUDGET):
        k_u, w_u, valid_u, inv = _support_rows_unique(ctx, rows)
        bt = BatchT(ctx, rows, k_u, inv, dev, {})
        w = torch.from_numpy(w_u).to(dev)[bt.inv_t]
        rt = torch.from_numpy(rows).to(dev)
        for h in hs:
            if not h.noise:
                continue
            mu, ls = _call_t(h, bt)
            ex = pf.crps_gauss_mixture_t(mu, torch.exp(ls), w, lyt[rt]).cpu().numpy()
            d = out.setdefault(h.name, {"support": [], "exact": [], "merged": []})
            d["support"] += [int(v) for v in sizes[rows]]
            d["exact"] += [float(v) for v in ex]
            d["merged"] += [float(v) for v in arr[h.name]["crps_log"][rows]]
    return out


def merge_error_summary(h, checks: list) -> dict:
    """One own-noise h's entry of merge_error.json from its per-pair `_mix_check` lists."""
    sup = np.array([v for c in checks for v in c["support"]], dtype=np.int64)
    ex = np.array([v for c in checks for v in c["exact"]], dtype=np.float64)
    mg = np.array([v for c in checks for v in c["merged"]], dtype=np.float64)
    info = h.info()
    base = {"variant": info["variant"], "rung": info["rung"], "noise": 1,
            "source": info.get("source"), "kind": f"{info['variant']}_r{info['rung']}_noise",
            "n_pairs": len(checks), "n_bins": int(ex.size)}
    if ex.size == 0:
        return {**base, "support_median": None, "support_max": None, "rel_err_median": None,
                "rel_err_q99": None, "rel_err_max": None, "signed_rel_err_mean": None,
                "crps_log_exact_mean": None, "crps_log_merged_mean": None, "avg_rel_err": None}
    with np.errstate(divide="ignore", invalid="ignore"):
        signed = (mg - ex) / np.abs(ex)
    rel = np.abs(signed)
    return {**base, "support_median": float(np.median(sup)), "support_max": int(sup.max()),
            "rel_err_median": float(np.median(rel)), "rel_err_q99": float(np.quantile(rel, 0.99)),
            "rel_err_max": float(rel.max()), "signed_rel_err_mean": float(signed.mean()),
            "crps_log_exact_mean": float(ex.mean()), "crps_log_merged_mean": float(mg.mean()),
            "avg_rel_err": float(abs(mg.mean() - ex.mean()) / abs(ex.mean()))}


# ---------------------------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------------------------


def _need_fixed(hs) -> bool:
    return any(h.kind == "learned" for h in hs)


def _ctl(pid: str, chrom: str):
    info = _CTX["infos"][pid]
    if not info.has_control:
        return None
    return _CTX["ctl_corpus"].get(pid, "counts", chrom).astype(np.float64)


def alone_job(pid: str) -> dict:
    hs, info, corpus = _CTX["hs"], _CTX["infos"][pid], _CTX["corpus"]
    per = []
    for c in SCORE_CHROMS:
        x = corpus.get(pid, "counts", c)
        per.append({"chrom": c, "treat": x, "ctl": _ctl(pid, c), "kpt": x,
                    "y": corpus.get(pid, "pval", c)})
    ctx = build_context(info, per, _CTX["blacklist"], _need_fixed(hs))
    fields, _ = score_context(hs, ctx, device=_CTX.get("device"))
    base = {"pid": pid, "track": info.track, "assay": info.assay, "mark_class": info.mark_class,
            "arm": info.arm, "level": info.level, "has_control": int(info.has_control)}
    return {name: {"h_name": name, **base, **f} for name, f in fields.items()}


def chain_job(pair: dict) -> dict:
    hs, pred, corpus = _CTX["hs"], _CTX["pred"], _CTX["corpus"]
    src, tgt = pair["source_pid"], pair["target_pid"]
    info = _CTX["infos"][tgt]
    per = []
    for c in SCORE_CHROMS:
        mu, n_size = f_nb(pred, src, tgt, c)
        per.append({"chrom": c, "treat": mu, "ctl": _ctl(tgt, c), "mu": mu, "n_size": n_size,
                    "y": corpus.get(tgt, "pval", c)})
    ctx = build_context(info, per, _CTX["blacklist"], _need_fixed(hs))
    G = _CTX.get("mix_groups")
    fields, shared = score_context(hs, ctx, device=_CTX.get("device"), mix_groups=G)
    base = {k: pair[k] for k in lpairs.PAIR_KEYS}
    out = {name: {**base, "f_run_name": _CTX["f_run_name"], "h_name": name,
                  "n_k_mean": shared["n_k_mean"], "n_k_max": shared["n_k_max"],
                  "n_capped": shared["n_capped"], **f}
           for name, f in fields.items()}
    if G:
        # the own-noise h's crps_log_* come from the merged mixture (PI 2026-10-16)
        for h in hs:
            if h.noise:
                out[h.name].update(crps_mix_merged=True, crps_mix_groups=int(G),
                                   n_mix_merged_bins=shared["n_mix_merged_bins"])
        if "mix_check" in shared:
            out["_mix_check"] = shared["mix_check"]
    return out


def _partial_path(i: int) -> Path:
    return Path(_CTX["partial_dir"]) / f"pair_{i:04d}.json"


def chain_pair(i: int) -> dict:
    """`chain_job` of training pair i, kept in `<out_dir>/<f_run>.partial/pair_<i>.json` once done,
    so a task that is stopped (walltime) resumes at the first pair without one. A kept file is
    used only when it holds every h of this call, for the same pair and the same mix_groups."""
    pair = _CTX["pred"].train_pairs[i]
    names = [h.name for h in _CTX["hs"]]
    path = _partial_path(i)
    key = {k: pair[k] for k in ("source_pid", "target_pid")}
    if path.is_file():
        try:
            kept = json.loads(path.read_text())
        except ValueError:
            kept = None
        if (kept and kept.get("pair") == key and kept.get("mix_groups") == _CTX.get("mix_groups")
                and set(names) <= set(kept["records"])):
            got = {n: kept["records"][n] for n in names}
            mc = kept["records"].get("_mix_check")
            if mc is not None:
                got["_mix_check"] = {n: v for n, v in mc.items() if n in names}
            return got
    t = time.time()
    out = chain_job(pair)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"pair": key, "mix_groups": _CTX.get("mix_groups"),
                               "seconds": round(time.time() - t, 3), "records": out}))
    tmp.replace(path)
    print(f"pair {i + 1}/{len(_CTX['pred'].train_pairs)} {key['source_pid']} -> "
          f"{key['target_pid']}: {time.time() - t:.0f}s", flush=True)
    return out


def _map(fn, items: list, workers: int, device) -> list:
    if workers <= 1 or len(items) <= 1 or getattr(device, "type", str(device)) == "cuda":
        return [fn(i) for i in items]
    # a worker that dies (e.g. a segfault) raises BrokenProcessPool instead of hanging the job
    with ProcessPoolExecutor(min(workers, len(items)), mp_context=mp.get_context("fork"),
                             initializer=lscore._worker_init) as pool:
        return list(pool.map(fn, items))


def _setup(manifest, covariates, data_dir, ctl_cache, blacklist, hs, control_rl_tsv, **extra):
    _CTX.clear()
    ctl_cache = Path(ctl_cache)
    _CTX.update(hs=hs, infos=products.load(manifest, covariates, control_rl_tsv),
                corpus=ldata.Corpus(data_dir),
                ctl_corpus=ldata.Corpus(ctl_cache) if ctl_cache.is_dir() else None,
                blacklist=str(blacklist), **extra)


def _config(blacklist) -> dict:
    return {"score_chroms": list(SCORE_CHROMS), "blacklist": str(blacklist),
            "blacklist_sha256": hashlib.sha256(Path(blacklist).read_bytes()).hexdigest(),
            "subsets": "baseline_rungs.subset_index on the true P (top1 = top 1 % by y)",
            "push": "pushed set {expm1(mu(k))_+, NB(k)}: crps exact on P, crps_log exact on "
                    "log1p(P), mean = sum_k NB(k) P_hat(k)",
            "own_noise": "mixture sum_k NB(k) N(mu_k, sigma_k^2) on log1p(P): crps_log exact "
                         "(Grimit 2006); crps = |mean_P - y|, mean_P = sum_k NB(k) "
                         "(exp(mu_k + sigma_k^2/2) - 1)",
            "mse": "mse_log1p = (log1p(mean_P) - log1p(y))^2; mse = (mean_P - y)^2",
            "split": "baseline_rungs.scale_split grid; push: the whole set scaled by 2^c; "
                     "own noise: scale_split(mean_P, y, |m - y|); all bins",
            "f_nb": "mu = pred.mean(loc), n = exp(disp) clipped to [N_MIN, N_MAX]",
            "tail": TAIL, "k_max": K_MAX, "chunk_bins": CHUNK_BINS, "elem_budget": ELEM_BUDGET,
            "crps_explode": CRPS_EXPLODE}


def write_h_scores(d, obj) -> None:
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    lscore.write_json(obj, d / "scores.json")


def _summary(records: list) -> dict:
    """File-level counts; the native keys follow the h's kind (crps / crps_mean_point)."""
    nat = "crps_mean_point" if records and "crps_mean_point_all" in records[0] else "crps"
    return {f"n_records_{nat}_nonfinite": sum(r[f"n_{nat}_nonfinite"] > 0 for r in records),
            "n_records_crps_log_nonfinite": sum(r["n_crps_log_nonfinite"] > 0 for r in records),
            f"n_records_{nat}_gt_20": sum(bool(r[f"{nat}_all_gt_20"]) for r in records)}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def score_alone(manifest, covariates, data_dir, ctl_cache, blacklist, h_spec, out_dir,
                workers: int = 1, control_rl_tsv=products.CONTROL_RL_TSV, device="auto") -> Path:
    """One h on the true X' of every usable product -> `<out_dir>/<h_name>/scores.json`."""
    t0 = time.time()
    h = load_h(h_spec, device)
    d = Path(out_dir) / h.name
    if (d / "ALONE_DONE").is_file():
        return d / "scores.json"
    _setup(manifest, covariates, data_dir, ctl_cache, blacklist, [h], control_rl_tsv,
           device=getattr(h, "device", None))
    pids = [r["pid"] for r in lpairs.usable_products(lpairs.read_manifest(manifest))]
    t1 = time.time()
    records = [r[h.name] for r in _map(alone_job, pids, workers, getattr(h, "device", "cpu"))]
    out = {"created_utc": _now(), "mode": "alone", "h": h.info(), "config": _config(blacklist),
           "records": records, **_summary(records),
           "timing": {"setup": round(t1 - t0, 3), "score": round(time.time() - t1, 3),
                      "n_jobs": len(pids), "workers": int(workers)}}
    write_h_scores(d, out)
    (d / "ALONE_DONE").write_text(out["created_utc"] + "\n")
    return d / "scores.json"


def score_chain(manifest, covariates, data_dir, ctl_cache, blacklist, f_run_dir, out_dir,
                h_specs: list, workers: int = 1, control_rl_tsv=products.CONTROL_RL_TSV,
                device="auto", mix_groups: int | None = None) -> Path:
    """Every listed h on f's NB over f's training pairs -> `<out_dir>/<f_run>/<h_name>/scores.json`;
    CHAIN_DONE once every listed h is scored. Returns `<out_dir>/<f_run>`. `mix_groups` (> 0):
    the own-noise h's mixture CRPS over at most that many merged components per bin (each such
    record carries crps_mix_merged, crps_mix_groups, n_mix_merged_bins); None or 0: exact."""
    mix_groups = int(mix_groups) if mix_groups else None
    t0 = time.time()
    f_run_dir = Path(f_run_dir)
    hs = [load_h(s, device) for s in h_specs]          # every spec checked before any work
    names = [h.name for h in hs]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate h names in {names}")
    fd = Path(out_dir) / f_run_dir.name
    todo = [h for h in hs if not (fd / h.name / "scores.json").is_file()]
    if todo:
        from eta import model as emodel                # chunk t125-P1c's loader
        pred = emodel.load_run(f_run_dir, data_dir, covariates, manifest, device=device)
        if pred.run_name != f_run_dir.name:
            raise ValueError(f"{f_run_dir}: run_name {pred.run_name!r} != its directory name")
        _setup(manifest, covariates, data_dir, ctl_cache, blacklist, todo, control_rl_tsv,
               pred=pred, f_run_name=pred.run_name, device=pred.device, mix_groups=mix_groups)
        part = Path(out_dir) / f"{f_run_dir.name}.partial"     # beside the f run's h dirs
        part.mkdir(parents=True, exist_ok=True)
        _CTX["partial_dir"] = str(part)
        t1 = time.time()
        results = _map(chain_pair, list(range(len(pred.train_pairs))), workers, pred.device)
        t2 = time.time()
        checks = [r.pop("_mix_check", None) for r in results]
        f_info = lscore.run_info(pred, f_run_dir)
        for h in todo:
            records = [r[h.name] for r in results]
            out = {"created_utc": _now(), "mode": "chain", "f_run": f_info, "h": h.info(),
                   "config": {**_config(blacklist), "mix_groups": mix_groups},
                   "records": records, **_summary(records),
                   "n_capped_total": sum(r["n_capped"] for r in records),
                   "timing": {"setup": round(t1 - t0, 3), "score_all_h": round(t2 - t1, 3),
                              "n_h_scored_together": len(todo), "n_jobs": len(records),
                              "workers": int(workers)}}
            write_h_scores(fd / h.name, out)
        noisy = [h for h in todo if h.noise]
        if mix_groups and noisy:
            path = fd / "merge_error.json"
            me = json.loads(path.read_text()) if path.is_file() else {"h": {}}
            if me.get("mix_groups") not in (None, mix_groups):
                me = {"h": {}}
            me.update(f_run_name=f_run_dir.name, mix_groups=mix_groups,
                      mix_check_bins=MIX_CHECK_BINS, check=MERGE_ERROR_DOC)
            for h in noisy:
                me["h"][h.name] = merge_error_summary(
                    h, [c[h.name] for c in checks if c is not None and h.name in c])
            lscore.write_json(me, path)
    if all((fd / n / "scores.json").is_file() for n in names):
        (fd / "CHAIN_DONE").write_text("\n".join(names) + "\n")
    return fd


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("alone", "chain"):
        a = sub.add_parser(name)
        for pos in ("manifest", "covariates", "data_dir", "ctl_cache", "blacklist"):
            a.add_argument(pos, type=Path)
        a.add_argument("h_spec" if name == "alone" else "f_run_dir")
        a.add_argument("out_dir", type=Path)
        if name == "chain":
            a.add_argument("--h", dest="h_specs", action="append", required=True)
            a.add_argument("--mix-merge", dest="mix_groups", type=int, default=0,
                           help="own-noise h: merge the mixture to at most this many components "
                                "per bin (0 = exact, the default)")
        a.add_argument("--workers", type=int, default=lscore.default_workers())
        a.add_argument("--control-rl-tsv", type=Path, default=products.CONTROL_RL_TSV)
        a.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = ap.parse_args(argv)
    t0 = time.time()
    common = (args.manifest, args.covariates, args.data_dir, args.ctl_cache, args.blacklist)
    if args.cmd == "alone":
        h_name = load_h(args.h_spec, "cpu").name
        if (args.out_dir / h_name / "ALONE_DONE").is_file():
            print(f"{args.out_dir / h_name / 'ALONE_DONE'} exists; nothing to do")
            return 0
        path = score_alone(*common, args.h_spec, args.out_dir, workers=args.workers,
                           control_rl_tsv=args.control_rl_tsv, device=args.device)
    else:
        fd = args.out_dir / Path(args.f_run_dir).name
        if (fd / "CHAIN_DONE").is_file():
            names = [load_h(s, "cpu").name for s in args.h_specs]
            if all((fd / n / "scores.json").is_file() for n in names):
                print(f"{fd / 'CHAIN_DONE'} exists and every h is scored; nothing to do")
                return 0
        path = score_chain(*common, args.f_run_dir, args.out_dir, args.h_specs,
                           workers=args.workers, control_rl_tsv=args.control_rl_tsv,
                           device=args.device, mix_groups=args.mix_groups or None)
    print(f"{args.cmd} -> {path} ({time.time() - t0:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
