"""t118 — the baseline rungs for X' = f(X | C, C'): noSolution (identity), QuantileMatching, oracle.

A *pair* is one (track, arm product, direction): `base_to_arm` predicts the arm product from the
track's base product, `arm_to_base` the reverse. Every non-base row of the t112 MANIFEST.tsv gives
two pairs. Counts (`counts25.npz`) and -log10 p (`pval25.npz`) are two separate spaces; each pair
is fitted and scored in both.

**Rungs.** All three emit one value per bin.
  noSolution        the prediction of X' is X itself.
  QuantileMatching  a monotone value-axis map g, fitted once per (pair, space) on the training
                    chromosomes by matching empirical quantiles. Source bins are sorted; a distinct
                    source value v that occupies the rank block [a, b) maps to the MEAN of the
                    sorted target values at ranks [a, b) (the tie-block mean — most bins are 0, so
                    ties are the rule). Values not seen in training are linearly interpolated
                    between the neighbouring fitted source values and clamped at both ends (flat
                    above the training maximum).
  oracle            per PRODUCT, not per pair: pseudoreplicate pr1 of the product predicts pr2, and
                    pr2 predicts pr1 (identity prediction, two directions). Each pseudoreplicate
                    holds HALF the product's reads, so the oracle is a half-depth repeat. A pair is
                    joined to the oracle of its TARGET product in `aggregate` (base_to_arm -> the
                    arm's oracle, arm_to_base -> the base's), the pair's oracle metric being the
                    mean of the two directions.

**Distribution of a one-value rung** (PI rulings 2026-09-24). The prediction is first floored at
PRED_FLOOR = 1e-3 (targets are never floored).
  counts  Poisson with mean = the floored prediction. `poisson_crps` is the closed form.
  p       log-normal with median = the floored prediction and ONE sigma per (pair or oracle
          direction, rung), the ML estimate on the training chromosomes of the log residual
          r = log(target) - log(prediction) with the median fixed: sigma^2 = mean(r^2). Bins whose
          target is exactly 0 have log(target) = -inf, zero likelihood under every log-normal,
          and are left out of the sigma fit; their count is recorded (`fit_n_zero_target`).
          `lognormal_crps` is the closed form (Baran & Lerch 2015).

**Scoring, each rung two ways.**
  point   CRPS of a point forecast = |prediction - X'| on the UNfloored prediction (= MAE).
  spread  the Poisson / log-normal CRPS above, with its split (`scale_split`): crps, the best
          single multiplier c* = 2^c_star on the predicted mean (counts) or median (p), and
          crps_oracle_scaled / scale_error = crps - crps_oracle_scaled. The c search is the grid
          of `candi.bench.distributional.oracle_scale` (c in [-6, 6] step 0.25, then +-0.25 step
          0.01, on a 20 000-bin subsample, seed 0; both CRPS on the full input). `nb_suite` is not
          used: `p_from_mu` clips p at 1 - 1e-9, so an NB cannot be pushed to its Poisson limit.
  Spearman on the unfloored prediction.

**Chromosomes.** Fit on every chromosome in the npz except chr19, chr21, chr22 (chrY, chrM dropped
everywhere; the t112 npz never holds them). chr22 is validation: these rungs do not use it, it is
only scored and reported under `eval = "val"`. chr19 + chr21 are the scored set (`eval =
"score"`). ENCODE blacklist bins (any bp of overlap, `candi.store.genome.blacklist_bin_flags`) are
removed from BOTH evaluation sets and kept in training.

**Subsets of the evaluated bins**, pooled over the chromosomes of the set:
  all       every non-blacklisted bin
  nonzero   bins where the real X' > 0
  top1      the ceil(1 %) bins with the largest real X'; ties at the cut broken by genomic order
            (stable sort on -X', chromosomes in npz order), so the subset is exactly 1 %.

**Input layout** (t112, `tools/t112/bin25.py::write_npz`, checked by `tools/t112/checks.py::
check_structure`): `<products>/<pid>/counts25.npz` and `pval25.npz`, one npz per product, one array
per main chromosome (chr1..chr22, chrX) keyed by the chromosome name; counts uint32, p float32
(finite, >= 0), each of length floor(chrom_len / 25). Pseudoreplicates have the same layout at
`<pseudoreps>/<pid>/pr1/` and `<pseudoreps>/<pid>/pr2/`.

Memory is bounded by streaming: training is one chromosome at a time and keeps only histograms
(QuantileMatching) and running sums (sigma), plus a second chromosome-at-a-time pass for the p
QuantileMatching residuals. Only the evaluation chromosomes (chr19 + chr21, ~4.2 M bins; chr22
~0.8 M) are held whole.

    python tools/t118/baseline_rungs.py pairs --manifest MANIFEST.tsv
    python tools/t118/baseline_rungs.py products --manifest MANIFEST.tsv
    python tools/t118/baseline_rungs.py run --manifest MANIFEST.tsv --products DIR \
        --blacklist hg38-blacklist.v2.bed --out DIR --index I
    python tools/t118/baseline_rungs.py oracle --manifest MANIFEST.tsv --pseudoreps DIR \
        --blacklist hg38-blacklist.v2.bed --out DIR --index I
    python tools/t118/baseline_rungs.py aggregate --manifest MANIFEST.tsv --out DIR
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.special import ive
from scipy.stats import norm, poisson

from candi.metrics import spearman
from candi.store.genome import blacklist_bin_flags, read_blacklist

RES = 25
SCORE_CHROMS = ("chr19", "chr21")
VAL_CHROMS = ("chr22",)
DROP_CHROMS = frozenset({"chrY", "chrM"})
#: floor on every one-value rung's prediction before its distribution is formed (PI ruling
#: 2026-09-24): a log-normal median of 0 has log -inf, and a Poisson of mean 0 is a point mass.
PRED_FLOOR = 1e-3
TOP_FRACTION = 0.01
SUBSETS = ("all", "nonzero", "top1")
RUNGS = ("noSolution", "QuantileMatching")
SPACES = {"counts": ("counts25.npz", np.uint32), "pval": ("pval25.npz", np.float32)}
DIRECTIONS = ("base_to_arm", "arm_to_base")
ORACLE_DIRECTIONS = ("pr1_to_pr2", "pr2_to_pr1")
MARK_CLASS = {"DNase-seq": "DNase",
              "H3K27ac": "narrow", "H3K4me3": "narrow", "H3K4me1": "narrow",
              "H3K27me3": "broad", "H3K36me3": "broad", "H3K9me3": "broad"}
SPLIT_KEYS = ("crps_oracle_scaled", "scale_error", "c_star")
#: `candi.bench.distributional.oracle_scale`'s search, reused for the Poisson and log-normal split
SCALE_FIT_BUDGET = 20_000
SCALE_SEED = 0


# ---------------------------------------------------------------------------------------------
# manifest -> pairs, products
# ---------------------------------------------------------------------------------------------


def read_manifest(path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def list_pairs(rows: list[dict]) -> list[dict]:
    """Every (arm product, direction) of every track, sorted by (track, pid, direction).

    The list index is the SLURM array index, so the order is a pure function of the manifest.
    """
    by_track: dict = {}
    for r in rows:
        by_track.setdefault(r["track"], []).append(r)
    pairs = []
    for track in sorted(by_track):
        base = [r for r in by_track[track] if r["arm"] == "base"]
        if len(base) != 1:
            raise ValueError(f"track {track}: {len(base)} base rows, need exactly 1")
        base = base[0]
        for r in sorted((r for r in by_track[track] if r["arm"] != "base"), key=lambda r: r["pid"]):
            if r["assay"] != base["assay"]:
                raise ValueError(f"{r['pid']}: assay {r['assay']} != base assay {base['assay']}")
            for d in DIRECTIONS:
                src, tgt = (base, r) if d == "base_to_arm" else (r, base)
                pairs.append({
                    "index": len(pairs), "track": track, "cell": r["cell"], "assay": r["assay"],
                    "mark_class": MARK_CLASS[r["assay"]], "arm": r["arm"], "level": r["level"],
                    "knob": r["knob"], "arm_pid": r["pid"], "direction": d,
                    "source_pid": src["pid"], "target_pid": tgt["pid"],
                })
    return pairs


def list_products(rows: list[dict]) -> list[dict]:
    """Every product of the manifest sorted by pid; the list index is the oracle array index."""
    pids = [r["pid"] for r in rows]
    if len(set(pids)) != len(pids):
        raise ValueError("duplicate pid in the manifest")
    return [{"index": i, **{k: r[k] for k in ("pid", "track", "assay", "arm", "level")}}
            for i, r in enumerate(sorted(rows, key=lambda r: r["pid"]))]


def pair_name(p: dict) -> str:
    return f"{p['arm_pid']}__{p['direction']}"


# ---------------------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------------------


class Track:
    """One product's npz for one space, opened lazily: `get(chrom)` decompresses one member."""

    def __init__(self, product_dir, space: str):
        fname, self.dtype = SPACES[space]
        self.path = Path(product_dir) / fname
        self.space = space
        self._z = np.load(self.path, allow_pickle=False)
        self.chroms = [c for c in self._z.files if c not in DROP_CHROMS]

    def get(self, chrom: str) -> np.ndarray:
        a = self._z[chrom]
        if a.ndim != 1:
            raise ValueError(f"{self.path}:{chrom}: ndim {a.ndim} != 1")
        if a.dtype != self.dtype:
            raise ValueError(f"{self.path}:{chrom}: dtype {a.dtype} != {np.dtype(self.dtype)}")
        if self.space == "pval" and a.size and not (np.isfinite(a).all() and a.min() >= 0):
            raise ValueError(f"{self.path}:{chrom}: non-finite or negative -log10 p")
        return a

    def close(self):
        self._z.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def split_chroms(chroms) -> tuple[list, list, list]:
    """(train, val, score) from the chromosomes both tracks hold, in their npz order."""
    chroms = [c for c in chroms if c not in DROP_CHROMS]
    missing = [c for c in SCORE_CHROMS + VAL_CHROMS if c not in chroms]
    if missing:
        raise ValueError(f"npz lacks the held-out chromosomes {missing}")
    held = set(SCORE_CHROMS) | set(VAL_CHROMS)
    return [c for c in chroms if c not in held], list(VAL_CHROMS), list(SCORE_CHROMS)


def paired(src: Track, tgt: Track, chrom: str) -> tuple[np.ndarray, np.ndarray]:
    x, y = src.get(chrom), tgt.get(chrom)
    if x.shape != y.shape:
        raise ValueError(f"{chrom}: source {x.shape} and target {y.shape} differ in length")
    return x, y


def load_eval(src: Track, tgt: Track, chroms, blacklist: dict) -> dict:
    """The evaluation bins of `chroms`, blacklist removed, concatenated in the given order."""
    xs, ys, n_total, n_black = [], [], 0, 0
    for c in chroms:
        x, y = paired(src, tgt, c)
        bad = blacklist_bin_flags(blacklist.get(c), x.size, RES)
        n_total += x.size
        n_black += int(bad.sum())
        xs.append(x[~bad])
        ys.append(y[~bad])
    return {"x": np.concatenate(xs), "y": np.concatenate(ys),
            "n_bins_total": n_total, "n_bins_blacklisted": n_black}


# ---------------------------------------------------------------------------------------------
# QuantileMatching
# ---------------------------------------------------------------------------------------------


def merge_hist(parts) -> tuple[np.ndarray, np.ndarray]:
    """Merge [(sorted distinct values, counts), ...] into one sorted (values, int64 counts)."""
    vals = np.concatenate([p[0] for p in parts])
    cnts = np.concatenate([p[1] for p in parts]).astype(np.int64)
    u, inv = np.unique(vals, return_inverse=True)
    out = np.zeros(u.size, dtype=np.int64)
    np.add.at(out, inv, cnts)
    return u, out


def qm_fit(sv, sc, tv, tc) -> tuple[np.ndarray, np.ndarray]:
    """Tie-block-mean quantile map from source (values, counts) to target (values, counts).

    Both histograms describe the same N bins. Sorted source rank block of value `sv[i]` is
    `[a_i, b_i)`; its image is the mean of the sorted target values at ranks `[a_i, b_i)`, read
    from the cumulative target sum S(k) = sum of the k smallest target values.
    Returns the knots `(sv, g(sv))`, float64.
    """
    sv, tv = np.asarray(sv, np.float64), np.asarray(tv, np.float64)
    sc, tc = np.asarray(sc, np.int64), np.asarray(tc, np.int64)
    if sc.sum() != tc.sum():
        raise ValueError(f"source holds {sc.sum()} bins, target {tc.sum()}")
    b = np.cumsum(sc)
    a = b - sc
    ctc = np.cumsum(tc)
    cts = np.cumsum(tv * tc)

    def S(k):
        j = np.searchsorted(ctc, k, side="left")
        prev_c = np.where(j > 0, ctc[np.maximum(j - 1, 0)], 0)
        prev_s = np.where(j > 0, cts[np.maximum(j - 1, 0)], 0.0)
        return prev_s + (k - prev_c) * tv[np.minimum(j, tv.size - 1)]

    return sv, (S(b) - S(a)) / sc


def qm_apply(knots, x) -> np.ndarray:
    kx, ky = knots
    return np.interp(np.asarray(x, np.float64), kx, ky)


# ---------------------------------------------------------------------------------------------
# the two predictive distributions of a one-value rung, and their CRPS
# ---------------------------------------------------------------------------------------------


def poisson_crps(lam, y) -> np.ndarray:
    """Closed-form CRPS of Poisson(lam) against integer y >= 0, element-wise.

    CRPS = E|X - y| - 1/2 E|X - X'| with
      E|X - y|  = (lam - y) + 2 [y F(y; lam) - lam F(y - 1; lam)]
      E|X - X'| = 2 lam e^{-2 lam} [I0(2 lam) + I1(2 lam)]   (exponentially scaled Bessel `ive`)
    The Poisson limit `candi.metrics.nb_crps` uses above its hyp2f1 range, at r = 1.
    """
    lam = np.asarray(lam, np.float64)
    y = np.asarray(y, np.float64)
    exy = (lam - y) + 2.0 * (y * poisson.cdf(y, lam) - lam * poisson.cdf(y - 1.0, lam))
    gmd = 2.0 * lam * (ive(0, 2.0 * lam) + ive(1, 2.0 * lam))
    return np.maximum(exy - 0.5 * gmd, 0.0)


def lognormal_crps(median, sigma, y) -> np.ndarray:
    """Closed-form CRPS of LogNormal(mu = log median, sigma) against y >= 0 (Baran & Lerch 2015):

        CRPS = y [2 Phi(z) - 1] - 2 e^{mu + sigma^2/2} [Phi(z - sigma) + Phi(sigma / sqrt 2) - 1],
        z = (log y - mu) / sigma.

    At y = 0, z = -inf and the expression is 2 e^{mu + sigma^2/2} [1 - Phi(sigma / sqrt 2)] (the
    mean minus half the Gini mean difference). sigma is floored at 1e-12 as `gauss_crps` floors
    it, so sigma -> 0 gives the point-forecast limit |y - median|.
    """
    mu = np.log(np.asarray(median, np.float64))
    sigma = np.maximum(np.asarray(sigma, np.float64), 1e-12)
    y = np.asarray(y, np.float64)
    with np.errstate(divide="ignore"):
        z = (np.log(y) - mu) / sigma                      # -inf where y == 0
    mean = np.exp(mu + 0.5 * sigma * sigma)
    return (y * (2.0 * norm.cdf(z) - 1.0)
            - 2.0 * mean * (norm.cdf(z - sigma) + norm.cdf(sigma / math.sqrt(2.0)) - 1.0))


def log_residual_sums(pred, y) -> tuple[float, int, int]:
    """(sum of r^2, bins used, bins left out) for r = log(y) - log(max(pred, PRED_FLOOR)), y > 0.

    The ML sigma of a log-normal with its median fixed at the prediction is sqrt(sum r^2 / used);
    a target of exactly 0 has zero likelihood under every log-normal and is left out.
    """
    y = np.asarray(y, np.float64)
    pos = y > 0
    r = np.log(y[pos]) - np.log(np.maximum(np.asarray(pred, np.float64)[pos], PRED_FLOOR))
    return float(np.dot(r, r)), int(pos.sum()), int(y.size - pos.sum())


def sigma_from_sums(ss: float, n_pos: int, n_zero: int) -> dict:
    return {"sigma": math.sqrt(ss / n_pos) if n_pos else float("nan"),
            "n_pos_target": n_pos, "n_zero_target": n_zero}


def scale_split(m, y, crps_fn, *, fit_budget: int = SCALE_FIT_BUDGET,
                seed: int = SCALE_SEED) -> dict:
    """crps, and c* = argmin_c mean crps_fn(m 2^c, y): the same search as `oracle_scale`.

    `m` is the predicted mean (Poisson) or median (log-normal); the spread parameter is held.
    The grid runs on a `fit_budget` subsample; both reported CRPS are on every bin.
    """
    m = np.asarray(m, np.float64)
    y = np.asarray(y, np.float64)
    crps = float(np.mean(crps_fn(m, y)))
    sel = (np.arange(m.size) if m.size <= fit_budget
           else np.random.default_rng(seed).choice(m.size, fit_budget, replace=False))
    mf, yf = m[sel], y[sel]

    def fit(c: float) -> float:
        return float(np.mean(crps_fn(mf * 2.0 ** c, yf)))

    c = float(min(np.arange(-6.0, 6.001, 0.25), key=fit))
    c = float(min(np.arange(c - 0.25, c + 0.2501, 0.01), key=fit))
    cs = float(np.mean(crps_fn(m * 2.0 ** c, y)))
    return {"crps": crps, "crps_oracle_scaled": cs, "scale_error": crps - cs, "c_star": c}


# ---------------------------------------------------------------------------------------------
# fit one pair in one space (training chromosomes only)
# ---------------------------------------------------------------------------------------------


def fit_space(src: Track, tgt: Track, train_chroms, space: str) -> dict:
    n_train, identical = 0, True
    sh, th = [], []
    ss_id, pos_id, zero_id = 0.0, 0, 0
    for c in train_chroms:
        x, y = paired(src, tgt, c)
        n_train += x.size
        identical = identical and bool(np.array_equal(x, y))
        if space == "counts":
            for arr, acc in ((x, sh), (y, th)):
                h = np.bincount(arr)
                nz = np.flatnonzero(h)
                acc.append((nz, h[nz]))
        else:
            sh.append(np.unique(x, return_counts=True))
            th.append(np.unique(y, return_counts=True))
            s, p, z = log_residual_sums(x, y)
            ss_id, pos_id, zero_id = ss_id + s, pos_id + p, zero_id + z
        del x, y
    knots = qm_fit(*merge_hist(sh), *merge_hist(th))
    if space == "counts":
        spread = {r: {} for r in RUNGS}                       # Poisson: nothing to fit
    else:
        ss_qm, pos_qm, zero_qm = 0.0, 0, 0                    # second pass: needs the fitted g
        for c in train_chroms:
            x, y = paired(src, tgt, c)
            s, p, z = log_residual_sums(qm_apply(knots, x), y)
            ss_qm, pos_qm, zero_qm = ss_qm + s, pos_qm + p, zero_qm + z
            del x, y
        spread = {"noSolution": sigma_from_sums(ss_id, pos_id, zero_id),
                  "QuantileMatching": sigma_from_sums(ss_qm, pos_qm, zero_qm)}
    return {"n_train_bins": n_train, "source_equals_target_train": identical,
            "qm_n_knots": int(knots[0].size), "spread": spread, "knots": knots}


# ---------------------------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------------------------


def subset_index(y: np.ndarray) -> dict:
    """Index arrays of the three subsets over the evaluated bins (see module docstring)."""
    k = int(math.ceil(TOP_FRACTION * y.size))
    order = np.argsort(-y.astype(np.float64), kind="stable")
    return {"all": np.arange(y.size), "nonzero": np.flatnonzero(y > 0), "top1": np.sort(order[:k])}


def score_rung(pred: np.ndarray, y: np.ndarray, space: str, spread: dict, subsets: dict) -> dict:
    """Point CRPS, spread CRPS with its split, Spearman, per subset (see module docstring)."""
    pred = np.asarray(pred, np.float64)
    y64 = y.astype(np.float64)
    m = np.maximum(pred, PRED_FLOOR)
    if space == "counts":
        def crps_fn(mm, yy):
            return poisson_crps(mm, yy)
    else:
        sigma = spread["sigma"]

        def crps_fn(mm, yy):
            return lognormal_crps(mm, sigma, yy)
    out: dict = {}
    abs_err = np.abs(pred - y64)
    for s, idx in subsets.items():
        out[f"n_{s}"] = int(idx.size)
        if idx.size == 0:
            for key in ("point_crps", "spread_crps", "spearman") + SPLIT_KEYS:
                out[f"{key}_{s}"] = None
            continue
        out[f"point_crps_{s}"] = float(abs_err[idx].mean())
        out[f"spearman_{s}"] = spearman(pred[idx], y64[idx])
        split = scale_split(m[idx], y64[idx], crps_fn)
        out[f"spread_crps_{s}"] = split["crps"]
        for key in SPLIT_KEYS:
            out[f"{key}_{s}"] = split[key]
    out["frac_bins_at_floor"] = float(np.mean(pred < PRED_FLOOR))
    return out


def _config(blacklist_path) -> dict:
    return {"score_chroms": list(SCORE_CHROMS), "val_chroms": list(VAL_CHROMS),
            "dropped_chroms": sorted(DROP_CHROMS), "pred_floor": PRED_FLOOR,
            "top_fraction": TOP_FRACTION,
            "counts_distribution": "Poisson, mean = max(prediction, pred_floor)",
            "pval_distribution": "log-normal, median = max(prediction, pred_floor), one sigma "
                                 "per (pair or oracle direction, rung): ML of log(target) - "
                                 "log(median) on the training chromosomes, target > 0 bins",
            "scale_split": "c* = 2^c_star on the mean/median; oracle_scale grid, "
                           f"fit_budget {SCALE_FIT_BUDGET}, seed {SCALE_SEED}",
            "qm_ties": "tie-block mean of the matched target quantiles",
            "qm_unseen": "linear interpolation between fitted source values, clamped",
            "blacklist": str(blacklist_path),
            "blacklist_sha256": hashlib.sha256(Path(blacklist_path).read_bytes()).hexdigest()}


def _write_json(out: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def run_pair(pair: dict, products, blacklist_path, out_dir, git_sha: str = "") -> Path:
    t0 = time.time()
    blacklist = read_blacklist(blacklist_path)
    out = {"pair": pair, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "git_sha": git_sha, "config": _config(blacklist_path), "spaces": {}, "records": []}
    for space in SPACES:
        with Track(Path(products) / pair["source_pid"], space) as src, \
                Track(Path(products) / pair["target_pid"], space) as tgt:
            if src.chroms != tgt.chroms:
                raise ValueError(f"{space}: source chroms {src.chroms} != target {tgt.chroms}")
            train, val, score = split_chroms(src.chroms)
            fit = fit_space(src, tgt, train, space)
            knots = fit.pop("knots")
            sp = {"train_chroms": train, **fit, "evals": {}}
            for ev, chroms in (("score", score), ("val", val)):
                d = load_eval(src, tgt, chroms, blacklist)
                sub = subset_index(d["y"])
                sp["evals"][ev] = {"chroms": chroms, "n_bins_total": d["n_bins_total"],
                                   "n_bins_blacklisted": d["n_bins_blacklisted"],
                                   "source_equals_target": bool(np.array_equal(d["x"], d["y"]))}
                preds = {"noSolution": d["x"].astype(np.float64),
                         "QuantileMatching": qm_apply(knots, d["x"])}
                for rung, pred in preds.items():
                    rec = {"space": space, "rung": rung, "eval": ev,
                           "source_equals_target": sp["evals"][ev]["source_equals_target"],
                           **{f"fit_{k}": v for k, v in fit["spread"][rung].items()},
                           **score_rung(pred, d["y"], space, fit["spread"][rung], sub)}
                    out["records"].append(rec)
                del d, sub, preds
            out["spaces"][space] = sp
    out["seconds"] = round(time.time() - t0, 1)
    return _write_json(out, Path(out_dir) / f"{pair_name(pair)}.json")


def run_oracle(product: dict, pseudoreps, blacklist_path, out_dir, git_sha: str = "") -> Path:
    """pr1 -> pr2 and pr2 -> pr1 of one product, both spaces, as the identity rung."""
    t0 = time.time()
    blacklist = read_blacklist(blacklist_path)
    root = Path(pseudoreps) / product["pid"]
    out = {"product": product, "rung": "oracle",
           "depth": "half: each pseudoreplicate holds half of the product's reads",
           "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "git_sha": git_sha, "config": _config(blacklist_path),
           "pseudoreps": {h: str(root / h) for h in ("pr1", "pr2")}, "spaces": {}, "records": []}
    for space in SPACES:
        sp = {}
        for direction in ORACLE_DIRECTIONS:
            a, b = ("pr1", "pr2") if direction == "pr1_to_pr2" else ("pr2", "pr1")
            with Track(root / a, space) as src, Track(root / b, space) as tgt:
                if src.chroms != tgt.chroms:
                    raise ValueError(f"{space}: {a} chroms {src.chroms} != {b} {tgt.chroms}")
                train, val, score = split_chroms(src.chroms)
                n_train, identical, ss, n_pos, n_zero = 0, True, 0.0, 0, 0
                for c in train:
                    x, y = paired(src, tgt, c)
                    n_train += x.size
                    identical = identical and bool(np.array_equal(x, y))
                    if space == "pval":
                        s, p, z = log_residual_sums(x, y)
                        ss, n_pos, n_zero = ss + s, n_pos + p, n_zero + z
                    del x, y
                spread = sigma_from_sums(ss, n_pos, n_zero) if space == "pval" else {}
                dd = {"train_chroms": train, "n_train_bins": n_train,
                      "source_equals_target_train": identical, "spread": spread, "evals": {}}
                for ev, chroms in (("score", score), ("val", val)):
                    d = load_eval(src, tgt, chroms, blacklist)
                    sub = subset_index(d["y"])
                    same = bool(np.array_equal(d["x"], d["y"]))
                    dd["evals"][ev] = {"chroms": chroms, "n_bins_total": d["n_bins_total"],
                                       "n_bins_blacklisted": d["n_bins_blacklisted"],
                                       "source_equals_target": same}
                    out["records"].append({
                        "space": space, "rung": "oracle", "direction": direction, "eval": ev,
                        "source_equals_target": same,
                        **{f"fit_{k}": v for k, v in spread.items()},
                        **score_rung(d["x"].astype(np.float64), d["y"], space, spread, sub)})
                    del d, sub
                sp[direction] = dd
        out["spaces"][space] = sp
    out["seconds"] = round(time.time() - t0, 1)
    return _write_json(out, Path(out_dir) / "oracle" / f"{product['pid']}.json")


# ---------------------------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------------------------

#: the rung tables; p drops `nonzero` (the p tracks have next to no exact zeros), the TSV keeps it
TABLE_METRICS = {
    "counts": ("spread_crps_all", "crps_oracle_scaled_all", "scale_error_all",
               "spread_crps_nonzero", "spread_crps_top1", "point_crps_all",
               "spearman_all", "spearman_top1"),
    "pval": ("spread_crps_all", "crps_oracle_scaled_all", "scale_error_all",
             "spread_crps_top1", "point_crps_all", "spearman_all", "spearman_top1"),
}
#: the gap tables: D_noSolution, D_QM, D_oracle and the fraction of noSolution -> oracle QM closes
GAP_METRICS = ("spread_crps_all", "spread_crps_top1", "point_crps_all", "spearman_all")
TABLE_RUNGS = RUNGS + ("oracle",)


def _mean(vals):
    v = [x for x in vals if x is not None and not isinstance(x, bool) and np.isfinite(x)]
    return float(np.mean(v)) if v else None


def _oracle_mean(recs: list[dict]) -> dict:
    """The mean over the two oracle directions of every numeric key (bools dropped)."""
    keys = [k for k in recs[0] if all(k in r for r in recs)]
    out = {}
    for k in keys:
        vals = [r[k] for r in recs]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            out[k] = float(np.mean(vals))
        elif all(v is None for v in vals):
            out[k] = None
    return out


def gap_fraction(d_none, d_qm, d_oracle, higher_is_better: bool = False):
    """(D_noSolution - D_QM) / (D_noSolution - D_oracle); None where noSolution is already at or
    past the oracle (D_noSolution <= D_oracle, or >= for a higher-is-better metric)."""
    if d_none is None or d_qm is None or d_oracle is None:
        return None
    if (d_none >= d_oracle) if higher_is_better else (d_none <= d_oracle):
        return None
    return (d_none - d_qm) / (d_none - d_oracle)


def aggregate(manifest, out_dir, tsv_path=None, md_path=None) -> dict:
    """One TSV row per (pair, space, rung, eval); a markdown summary of the scored set.

    The oracle rows of a pair are its target product's oracle: `oracle` (mean of the two
    directions, the rung the tables show) and `oracle_pr1_to_pr2` / `oracle_pr2_to_pr1`.
    Per mark class: mean over the class's tracks of each track's mean over its pairs (both
    directions pooled). Per track and per arm: mean over pairs.
    """
    out_dir = Path(out_dir)
    pairs = list_pairs(read_manifest(manifest))
    oracle, missing_oracle = {}, []
    for pid in sorted({p["target_pid"] for p in pairs}):
        f = out_dir / "oracle" / f"{pid}.json"
        if f.is_file():
            oracle[pid] = json.loads(f.read_text("utf-8"))["records"]
        else:
            missing_oracle.append(pid)
    rows, missing = [], []
    for p in pairs:
        f = out_dir / f"{pair_name(p)}.json"
        if not f.is_file():
            missing.append(pair_name(p))
            continue
        res = json.loads(f.read_text("utf-8"))
        meta = {k: p[k] for k in ("index", "track", "assay", "mark_class", "arm", "level",
                                  "arm_pid", "direction", "source_pid", "target_pid")}
        for rec in res["records"]:
            rows.append({**meta, **rec})
        for space in SPACES:
            for ev in ("score", "val"):
                pair_same = next(r["source_equals_target"] for r in res["records"]
                                 if r["space"] == space and r["eval"] == ev)
                orc = [r for r in oracle.get(p["target_pid"], [])
                       if r["space"] == space and r["eval"] == ev]
                if len(orc) != len(ORACLE_DIRECTIONS):
                    continue
                base = {**meta, "space": space, "eval": ev, "source_equals_target": pair_same}
                for r in orc:
                    rows.append({**base, **{k: v for k, v in r.items() if k not in base},
                                 "rung": f"oracle_{r['direction']}",
                                 "oracle_source_equals_target": r["source_equals_target"]})
                rows.append({**base, **_oracle_mean(orc), "rung": "oracle",
                             "direction": p["direction"],
                             "oracle_source_equals_target":
                                 any(r["source_equals_target"] for r in orc)})
    cols = list(dict.fromkeys(k for r in rows for k in r))
    tsv_path = Path(tsv_path or out_dir / "rungs_v2.tsv")
    with tsv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", restval="")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})

    # PI ruling 2026-09-23: count pairs whose counts are bit-identical to base (the p-only arms)
    # and, in both spaces, the DNase MAPQ arms (a t112 defect: their reads equal the DNase base's)
    # stay in the TSV but are left out of the markdown tables.
    scored = [r for r in rows if r["eval"] == "score" and r["rung"] in TABLE_RUNGS
              and not (r["space"] == "counts" and r.get("source_equals_target"))
              and not (r["track"] == "C12M02" and r["arm"] == "mapq")]
    n_orc = len(oracle)
    lines = [f"# t118 baseline rungs v2 — scored on {' + '.join(SCORE_CHROMS)}, blacklist removed",
             "",
             f"{len(pairs) - len(missing)} of {len(pairs)} pairs present; oracle for {n_orc} of "
             f"{n_orc + len(missing_oracle)} target products. Every rung emits one value per bin, "
             f"floored at {PRED_FLOOR} for its distribution: counts are Poisson with that mean; "
             "-log10 p is log-normal with that median and one sigma per (pair, rung), fitted by "
             "ML on the training chromosomes. `spread_crps` = CRPS of that distribution; "
             "`crps_oracle_scaled` = the same after the best single multiplier on the mean/median "
             "(c*), `scale_error` = the difference; `point_crps` = CRPS of the point forecast = "
             "MAE. The **oracle** is one pseudoreplicate of the target product predicting the "
             "other, mean of both directions — each half holds HALF the reads, so it is a "
             "half-depth repeat and a noisier bar than a full-depth one. Mark class = mean over "
             "tracks of each track's mean over pairs. Count pairs identical to base (p-only arms) "
             "are left out of these tables and kept in the TSV, as are both spaces of the DNase "
             "MAPQ arms (identical to the DNase base, a t112 defect). `nonzero` is dropped from "
             "the p tables (kept in the TSV).", ""]
    if missing:
        lines += [f"Missing pairs: {', '.join(missing)}", ""]
    if missing_oracle:
        lines += [f"Missing oracle products: {', '.join(missing_oracle)}", ""]

    def group_value(sel, m, macro_over_tracks):
        if macro_over_tracks:
            return _mean([_mean([r.get(m) for r in sel if r["track"] == t])
                          for t in sorted({r["track"] for r in sel})])
        return _mean([r.get(m) for r in sel])

    def fmt(v):
        return "—" if v is None else f"{v:.4f}"

    def gap_table(title, key_fn, macro_over_tracks):
        lines.extend([f"## {title}", "",
                      "Fraction closed = (D_noSolution − D_QM) / (D_noSolution − D_oracle): the "
                      "share of the noSolution → oracle interval that QuantileMatching closes. "
                      "“—” where noSolution is already at or past the oracle (for Spearman, "
                      "higher is better, so where ρ_noSolution ≥ ρ_oracle).", "",
                      "| group | space | metric | D_noSolution | D_QM | D_oracle | fraction closed |",
                      "|---|---|---|---|---|---|---|"])
        n_dash, n_all = 0, 0
        for space in SPACES:
            for g in sorted({key_fn(r) for r in scored if r["space"] == space}):
                sel = {rung: [r for r in scored if key_fn(r) == g and r["space"] == space
                              and r["rung"] == rung] for rung in TABLE_RUNGS}
                for m in GAP_METRICS:
                    d = {rung: group_value(sel[rung], m, macro_over_tracks) for rung in TABLE_RUNGS}
                    f = gap_fraction(d["noSolution"], d["QuantileMatching"], d["oracle"],
                                     higher_is_better=m.startswith("spearman"))
                    n_all += 1
                    n_dash += f is None
                    lines.append(f"| {g} | {space} | {m} | {fmt(d['noSolution'])} | "
                                 f"{fmt(d['QuantileMatching'])} | {fmt(d['oracle'])} | {fmt(f)} |")
        lines.extend(["", f"{n_dash} of {n_all} rows are “—”.", ""])

    def table(title, key_fn, macro_over_tracks):
        for space in SPACES:
            metrics = TABLE_METRICS[space]
            lines.extend([f"## {title} — {space}", "",
                          "| group | rung | " + " | ".join(metrics) + " |",
                          "|---|---|" + "---|" * len(metrics)])
            for g in sorted({key_fn(r) for r in scored if r["space"] == space}):
                for rung in TABLE_RUNGS:
                    sel = [r for r in scored if key_fn(r) == g and r["space"] == space
                           and r["rung"] == rung]
                    if not sel:
                        continue
                    vals = [fmt(group_value(sel, m, macro_over_tracks)) for m in metrics]
                    lines.append(f"| {g} | {rung} | " + " | ".join(vals) + " |")
            lines.append("")

    gap_table("QuantileMatching's share of the noSolution → oracle interval, by mark class",
              lambda r: r["mark_class"], True)
    gap_table("QuantileMatching's share of the noSolution → oracle interval, by track",
              lambda r: f"{r['track']} {r['assay']}", False)
    table("By mark class", lambda r: r["mark_class"], True)
    table("By track", lambda r: f"{r['track']} {r['assay']}", False)
    table("By arm", lambda r: r["arm"], False)
    md_path = Path(md_path or out_dir / "rungs_v2.md")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"n_rows": len(rows), "missing": missing, "missing_oracle": missing_oracle,
            "tsv": str(tsv_path), "md": str(md_path)}


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def _git_sha() -> str:
    if os.environ.get("T118_GIT_SHA"):
        return os.environ["T118_GIT_SHA"]
    try:
        return subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse",
                               "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("pairs", help="print the pair table; its row index is the array index")
    a.add_argument("--manifest", required=True, type=Path)
    b = sub.add_parser("products", help="print the product table (oracle array index)")
    b.add_argument("--manifest", required=True, type=Path)
    r = sub.add_parser("run", help="fit and score one pair")
    r.add_argument("--manifest", required=True, type=Path)
    r.add_argument("--products", required=True, type=Path)
    r.add_argument("--blacklist", required=True, type=Path)
    r.add_argument("--out", required=True, type=Path)
    r.add_argument("--index", required=True, type=int)
    r.add_argument("--overwrite", action="store_true")
    o = sub.add_parser("oracle", help="score one product's pseudoreplicate oracle")
    o.add_argument("--manifest", required=True, type=Path)
    o.add_argument("--pseudoreps", required=True, type=Path)
    o.add_argument("--blacklist", required=True, type=Path)
    o.add_argument("--out", required=True, type=Path)
    o.add_argument("--index", required=True, type=int)
    o.add_argument("--overwrite", action="store_true")
    g = sub.add_parser("aggregate", help="per-pair and oracle JSONs -> TSV + markdown")
    g.add_argument("--manifest", required=True, type=Path)
    g.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    rows = read_manifest(args.manifest)
    pairs = list_pairs(rows)
    if args.cmd == "pairs":
        cols = ("index", "track", "assay", "mark_class", "arm", "level", "direction",
                "source_pid", "target_pid")
        print("\t".join(cols))
        for p in pairs:
            print("\t".join(str(p[c]) for c in cols))
        return 0
    if args.cmd == "products":
        prods = list_products(rows)
        print("\t".join(prods[0]))
        for p in prods:
            print("\t".join(str(v) for v in p.values()))
        return 0
    if args.cmd in ("run", "oracle"):
        items = pairs if args.cmd == "run" else list_products(rows)
        if not 0 <= args.index < len(items):
            print(f"index {args.index} outside 0..{len(items) - 1}", file=sys.stderr)
            return 2
        it = items[args.index]
        name = pair_name(it) if args.cmd == "run" else it["pid"]
        dest = (args.out / f"{name}.json" if args.cmd == "run"
                else args.out / "oracle" / f"{name}.json")
        if dest.exists() and not args.overwrite:
            print(f"{dest} exists; pass --overwrite to redo")
            return 0
        if args.cmd == "run":
            path = run_pair(it, args.products, args.blacklist, args.out, _git_sha())
        else:
            path = run_oracle(it, args.pseudoreps, args.blacklist, args.out, _git_sha())
        print(f"{name} -> {path}")
        return 0
    s = aggregate(args.manifest, args.out)
    print(f"{s['n_rows']} rows -> {s['tsv']}; summary {s['md']}; missing {len(s['missing'])} "
          f"pairs, {len(s['missing_oracle'])} oracle products")
    return 0 if not (s["missing"] or s["missing_oracle"]) else 1


if __name__ == "__main__":
    sys.exit(main())
