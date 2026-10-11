"""t125 part 2 — the two training sources of h (design note "Part 2", "Training source").

**HCorpus** reads the treatment counts and P (`ladder.data.Corpus(data_dir)`, spaces counts / pval)
and the control counts (`ladder.data.Corpus(ctl_cache_dir)`, space "counts", built by
`hmap.controls`; None for a product without a control).

**TrueSource** (h on the true X'): windows of W bins, pid uniform over `pids`, chromosome
proportional to its length, start uniform (`default_rng(seed)`; the draw order is pid, chromosome,
start, as `ladder.train.Sampler`). Per bin: P2a's rung-5 features of the TRUE counts and control
(`log1p_k` + `fixed_features`, 21 columns; rung 1 is the first 9), log1p of P2b's fragment-window
formula (`formula_col`), y = the product's true P, mask = the bin lies on the chromosome.

**FnbSource** (h on f's predicted NB, the repair source): windows drawn pair uniform over the f's
training pairs, chromosome and start as above. f's (loc, disp) on the window comes from
`pred.predict_window(source, source, target, ...)`; the NB has mean `pred.mean(loc)` and size
exp(disp) clipped to [N_MIN, N_MAX] (t118's law). Every feature is the TARGET's (its control, its
assay, depths, ...) and y is the target's true P. The treatment-derived windows use f's predicted
mean (P2a's chain rule): the no-control λ window (`formula.lambda_track` on the mean) and the
fragment window, whose value for an enumerated count k is (k + the neighbours' predicted means) /
the bins the window keeps (`features.k_frag_chain`).

**Halo.** Every window feature (fragment, 1 kb, 10 kb) is computed on the window plus `HALO` bins on
each side, clipped to the chromosome, then sliced: inside the clip the windows see exactly what the
whole-chromosome computation sees, and at a chromosome end the clip IS the end, so the truncation
matches too. HALO = 400 >= the 10 kb window's half (200).

**Chunks by support size** (`fnb_chunks`). The f-NB losses enumerate k over the exact NB support of
P2c (`pushforward.nb_support`, tail 1e-6, capped at K_MAX = 4096). A dense [rows, K] block takes K =
the widest row, so one capped row would make a whole batch 4097 wide; instead the bins are sorted by
`support_size` and cut into chunks with rows x K <= `elems` (one row at least), so padding stays
small and memory bounded. Each chunk counts its capped rows (support > k_max); the training loop
writes the count into the run.

The f-NB batch is returned per bin (`FnbBatch`: the NB's mean and size, not a dense (k, w)), with
only the bins that lie on the chromosome; `fnb_chunks` expands it.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterator, NamedTuple

import numpy as np

from ladder import data as ldata
from ladder import model as lmodel

from hmap import features, formula, pushforward
from hmap.products import ProductInfo

SPACE_COUNTS, SPACE_PVAL = "counts", "pval"
#: bins added on each side of a window before the window features are computed (>= 10 kb / 2)
HALO = features.W_10K
#: dense f-NB chunk budget: rows x K entries (a chunk of 2^18 entries x 22 float32 = 23 MB)
FNB_ELEMS = 1 << 18
N_FIXED_R1 = len(features.FEATURES_R1) - 1                    # 8: fixed columns of rung 1
K_FRAG_COL = features.FEATURES_R5.index("log1p_k_frag")       # 9, in the full rung-5 matrix
_LOG_N_MIN, _LOG_N_MAX = math.log(lmodel.N_MIN), math.log(lmodel.N_MAX)


class HCorpus:
    """Treatment counts and P of the products, and the control counts of the control cache."""

    def __init__(self, data_dir, ctl_cache_dir):
        self.data = ldata.Corpus(data_dir)
        self.ctl_dir = Path(ctl_cache_dir) if ctl_cache_dir is not None else None
        self._ctl = None
        if self.ctl_dir is not None and self.ctl_dir.is_dir() and any(
                self.ctl_dir.glob("*__counts.json")):
            self._ctl = ldata.Corpus(self.ctl_dir)

    def counts(self, pid: str, chrom: str) -> np.ndarray:
        return self.data.get(pid, SPACE_COUNTS, chrom)

    def pval(self, pid: str, chrom: str) -> np.ndarray:
        return self.data.get(pid, SPACE_PVAL, chrom)

    def has_control(self, pid: str) -> bool:
        return self._ctl is not None and (self.ctl_dir / f"{pid}__{SPACE_COUNTS}.json").is_file()

    def control(self, pid: str, chrom: str) -> np.ndarray | None:
        """The control counts of `pid` on `chrom` (uint32), None when the product has none."""
        if not self.has_control(pid):
            return None
        c = self._ctl.get(pid, SPACE_COUNTS, chrom)
        n = self.n_bins(pid, chrom)
        if c.shape[0] != n:
            raise ValueError(f"{pid}/{chrom}: control has {c.shape[0]} bins, treatment {n}")
        return c

    def chroms(self, pid: str) -> list[str]:
        return self.data.chroms(pid, SPACE_COUNTS)

    def n_bins(self, pid: str, chrom: str) -> int:
        return self.data.n_bins(pid, SPACE_COUNTS, chrom)


def _span(n: int, start: int, stop: int, halo: int) -> tuple[int, int]:
    return max(0, start - halo), min(n, stop + halo)


def _ctl_slice(corpus: HCorpus, info: ProductInfo, chrom: str, a: int, b: int):
    c = corpus.control(info.pid, chrom)
    if bool(info.has_control) != (c is not None):
        raise ValueError(f"{info.pid}: has_control = {info.has_control} but the control cache "
                         f"{'has no' if c is None else 'has an'} entry")
    return None if c is None else np.asarray(c[a:b])


# ---------------------------------------------------------------------------------------------
# true X'
# ---------------------------------------------------------------------------------------------


def true_bins(corpus: HCorpus, info: ProductInfo, chrom: str, start: int, stop: int,
              with_formula: bool = True, halo: int = HALO):
    """Bins start..min(stop, n) of one product: (feat_r5 float32[m, 21], formula_col float32[m] |
    None, y float32[m]) from the TRUE counts and control (window features over a halo)."""
    n = corpus.n_bins(info.pid, chrom)
    stop = min(int(stop), n)
    start = int(start)
    a, b = _span(n, start, stop, halo)
    ct = np.asarray(corpus.counts(info.pid, chrom)[a:b])
    cc = _ctl_slice(corpus, info, chrom, a, b)
    i0, i1 = start - a, stop - a
    fixed = features.fixed_features(info, ct, cc, 5)[i0:i1]
    feat = np.concatenate([features.k_column(ct[i0:i1])[:, None], fixed], axis=1)
    fcol = None
    if with_formula:
        fcol = np.log1p(formula.P("frag", info, ct, cc)[i0:i1]).astype(np.float32)
    y = np.array(corpus.pval(info.pid, chrom)[start:stop], dtype=np.float32)
    return feat.astype(np.float32, copy=False), fcol, y


class TrueBatch(NamedTuple):
    feat_r1: np.ndarray          # float32[B*W, 9]   (a view of feat_r5's first 9 columns)
    feat_r5: np.ndarray          # float32[B*W, 21]
    formula_col: np.ndarray | None   # float32[B*W]: log1p of the fragment-window formula
    y: np.ndarray                # float32[B*W]: the true P (0 off the chromosome)
    mask: np.ndarray             # bool[B*W]: the bin lies on the chromosome


class TrueSource:
    """Windows of the true X' (see the module docstring)."""

    def __init__(self, corpus: HCorpus, infos: dict, pids, chroms, lens, window: int, seed: int,
                 with_formula: bool = True, halo: int = HALO):
        if not pids:
            raise ValueError("TrueSource needs at least one pid")
        self.corpus, self.infos, self.pids = corpus, infos, list(pids)
        self.chroms = list(chroms)
        self.lens = np.asarray(lens, dtype=np.int64)
        self.probs = self.lens / self.lens.sum()
        self.window, self.with_formula, self.halo = int(window), bool(with_formula), int(halo)
        self.rng = np.random.default_rng(int(seed))

    def draw(self, batch: int) -> TrueBatch:
        W = self.window
        pi = self.rng.integers(len(self.pids), size=batch)
        ci = self.rng.choice(len(self.chroms), size=batch, p=self.probs)
        u = self.rng.random(batch)
        F = len(features.FEATURES_R5)
        feat = np.zeros((batch * W, F), dtype=np.float32)
        fcol = np.zeros(batch * W, dtype=np.float32) if self.with_formula else None
        y = np.zeros(batch * W, dtype=np.float32)
        mask = np.zeros(batch * W, dtype=bool)
        for b in range(batch):
            chrom, n = self.chroms[ci[b]], int(self.lens[ci[b]])
            s = int(u[b] * (max(0, n - W) + 1))
            info = self.infos[self.pids[pi[b]]]
            f5, fc, yy = true_bins(self.corpus, info, chrom, s, s + W, self.with_formula,
                                   self.halo)
            m = yy.shape[0]
            o = b * W
            feat[o:o + m] = f5
            y[o:o + m] = yy
            mask[o:o + m] = True
            if fcol is not None:
                fcol[o:o + m] = fc
        return TrueBatch(feat[:, :len(features.FEATURES_R1)], feat, fcol, y, mask)


# ---------------------------------------------------------------------------------------------
# f's NB
# ---------------------------------------------------------------------------------------------


def nb_params(pred, loc, disp) -> tuple[np.ndarray, np.ndarray]:
    """(mu, n_size) float64 of f's NB: `pred.mean(loc)` and exp(disp) clipped to [N_MIN, N_MAX]."""
    mu = np.asarray(pred.mean(loc), dtype=np.float64)
    n = np.exp(np.clip(np.asarray(disp, dtype=np.float64), _LOG_N_MIN, _LOG_N_MAX))
    return mu, n


def fnb_bins(corpus: HCorpus, info: ProductInfo, chrom: str, a: int, b: int, start: int,
             stop: int, mu_seg: np.ndarray, n_seg: np.ndarray) -> dict:
    """Per-bin f-NB arrays for bins start..stop of the TARGET `info`, from f's NB over the clipped
    segment [a, b) that holds them with their halo (`mu_seg`, `n_seg` [b - a])."""
    mu_seg = np.asarray(mu_seg, dtype=np.float64)
    if mu_seg.shape != (b - a,):
        raise ValueError(f"mu_seg {mu_seg.shape} != ({b - a},)")
    cc = _ctl_slice(corpus, info, chrom, a, b)
    i0, i1 = start - a, stop - a
    wf = features.frag_bins(info.fraglen)
    rest = features.frag_rest(mu_seg, wf)
    return {
        "feat_fixed": features.fixed_features(info, mu_seg, cc, 5, treat_mean=mu_seg)[i0:i1],
        "mu": mu_seg[i0:i1],
        "n_size": np.asarray(n_seg, dtype=np.float64)[i0:i1],
        "frag_rest": rest[i0:i1],
        "frag_cnt": features.window_count(b - a, wf)[i0:i1].astype(np.float64),
        "lam": formula.lambda_track(info, cc, mu_seg)[i0:i1],
        "y": np.array(corpus.pval(info.pid, chrom)[start:stop], dtype=np.float32),
    }


class FnbBatch(NamedTuple):
    """Per bin (only bins on the chromosome): N rows."""
    feat_fixed: np.ndarray       # float32[N, 20]: rung-5 fixed features (rung 1 = first 8)
    mu: np.ndarray               # float64[N]: f's NB mean
    n_size: np.ndarray           # float64[N]: f's NB size
    frag_rest: np.ndarray        # float64[N]: neighbours' predicted-mean sum in the fragment window
    frag_cnt: np.ndarray         # float64[N]: bins the fragment window keeps
    lam: np.ndarray              # float64[N]: the formula's local λ (no control: f's mean window)
    y: np.ndarray                # float32[N]: the target's true P
    info_idx: np.ndarray         # int64[N]: index into `infos`
    infos: list                  # the target ProductInfos of the batch


def concat_fnb(parts: list[dict], info_list: list, info_of_part: list[int]) -> FnbBatch:
    keys = ("feat_fixed", "mu", "n_size", "frag_rest", "frag_cnt", "lam", "y")
    cat = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    idx = np.concatenate([np.full(p["y"].shape[0], g, dtype=np.int64)
                          for p, g in zip(parts, info_of_part)])
    return FnbBatch(cat["feat_fixed"].astype(np.float32, copy=False), cat["mu"], cat["n_size"],
                    cat["frag_rest"], cat["frag_cnt"], cat["lam"], cat["y"], idx, info_list)


class FnbSource:
    """Windows of f's predicted NB over the f's training pairs (see the module docstring)."""

    def __init__(self, corpus: HCorpus, infos: dict, pred, train_pairs, chroms, lens,
                 window: int, seed: int, halo: int = HALO):
        if not train_pairs:
            raise ValueError("FnbSource needs at least one training pair")
        self.corpus, self.infos, self.pred = corpus, infos, pred
        self.pairs = list(train_pairs)
        self.chroms = list(chroms)
        self.lens = np.asarray(lens, dtype=np.int64)
        self.probs = self.lens / self.lens.sum()
        self.window, self.halo = int(window), int(halo)
        self.rng = np.random.default_rng(int(seed))

    def draw(self, batch: int) -> FnbBatch:
        W = self.window
        pi = self.rng.integers(len(self.pairs), size=batch)
        ci = self.rng.choice(len(self.chroms), size=batch, p=self.probs)
        u = self.rng.random(batch)
        parts, info_list, info_of_part, slot = [], [], [], {}
        for b in range(batch):
            pair = self.pairs[pi[b]]
            chrom, n = self.chroms[ci[b]], int(self.lens[ci[b]])
            s = int(u[b] * (max(0, n - W) + 1))
            e = min(s + W, n)
            info = self.infos[pair["target_pid"]]
            a, bb = _span(n, s, e, self.halo)
            src = pair["source_pid"]
            loc, disp = self.pred.predict_window(src, src, pair["target_pid"], chrom, a, bb)
            mu, nn = nb_params(self.pred, loc, disp)
            parts.append(fnb_bins(self.corpus, info, chrom, a, bb, s, e, mu, nn))
            if info.pid not in slot:
                slot[info.pid] = len(info_list)
                info_list.append(info)
            info_of_part.append(slot[info.pid])
        return concat_fnb(parts, info_list, info_of_part)


# ---------------------------------------------------------------------------------------------
# dense chunks over the NB support
# ---------------------------------------------------------------------------------------------


class FnbChunk(NamedTuple):
    idx: np.ndarray              # int64[r]: rows of the FnbBatch
    feat: np.ndarray             # float32[r, K, F]: h's input per enumerated k
    w: np.ndarray                # float64[r, K]: NB weights over the kept support (0 = padding)
    y: np.ndarray                # float32[r]
    n_capped: int                # rows of this chunk whose support exceeds k_max
    n_k: np.ndarray              # int64[r]: support points kept per row


def chunk_plan(size: np.ndarray, elems: int, k_max: int) -> list[np.ndarray]:
    """Rows sorted by support size, cut so that rows x (widest row, capped at k_max + 1) <= elems;
    a chunk holds one row at least."""
    eff = np.minimum(np.asarray(size, dtype=np.int64), int(k_max) + 1)
    order = np.argsort(eff, kind="stable")
    es = eff[order]
    out, i, N = [], 0, order.shape[0]
    while i < N:
        prod = (np.arange(1, N - i + 1, dtype=np.int64)) * es[i:]
        j = i + max(1, int(np.searchsorted(prod, int(elems), side="right")))
        out.append(order[i:j])
        i = j
    return out


def chunk_features(fb: FnbBatch, idx: np.ndarray, k: np.ndarray, rung: int,
                   with_formula: bool) -> np.ndarray:
    """float32[r, K, F]: log1p k, the fixed columns (rung 1: 8, rung 5: 20, with log1p_k_frag
    replaced per k by log1p((k + frag_rest) / frag_cnt)), and for the formula variants log1p of
    `formula.P_enumerated("frag", ...)` last."""
    r, K = k.shape
    fixed = fb.feat_fixed[idx]
    if rung == 1:
        fixed = fixed[:, :N_FIXED_R1]
    elif rung != 5:
        raise ValueError(f"rung {rung!r} not in (1, 5)")
    nf = fixed.shape[1]
    F = 1 + nf + (1 if with_formula else 0)
    feat = np.empty((r, K, F), dtype=np.float32)
    kf = k.astype(np.float64)
    feat[:, :, 0] = np.log1p(kf)
    feat[:, :, 1:1 + nf] = fixed[:, None, :]
    rest = fb.frag_rest[idx]
    if rung == 5:
        feat[:, :, K_FRAG_COL] = np.log1p((kf + rest[:, None]) / fb.frag_cnt[idx][:, None])
    if with_formula:
        lam = fb.lam[idx]
        g_of = fb.info_idx[idx]
        col = np.empty((r, K), dtype=np.float64)
        for g in np.unique(g_of):
            sel = g_of == g
            col[sel] = formula.P_enumerated("frag", fb.infos[g], kf[sel], lam[sel], rest[sel])
        feat[:, :, -1] = np.log1p(col)
    return feat


def fnb_chunks(fb: FnbBatch, rung: int, with_formula: bool, elems: int = FNB_ELEMS,
               k_max: int = pushforward.K_MAX, tail: float = pushforward.TAIL,
               size: np.ndarray | None = None) -> Iterator[FnbChunk]:
    """The batch as dense chunks over the exact NB support, sorted by support size."""
    if size is None:
        size = pushforward.support_size(fb.mu, fb.n_size, tail=tail, k_max=k_max)
    for idx in chunk_plan(size, elems, k_max):
        k, w, valid = pushforward.nb_support(fb.mu[idx], fb.n_size[idx], tail=tail, k_max=k_max)
        yield FnbChunk(idx, chunk_features(fb, idx, k, rung, with_formula), w, fb.y[idx],
                       int((size[idx] > k_max).sum()), valid.sum(axis=1))
