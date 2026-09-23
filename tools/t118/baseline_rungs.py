"""t118 — the two baseline rungs for X' = f(X | C, C'): noSolution (identity) and QuantileMatching.

A *pair* is one (track, arm product, direction): `base_to_arm` predicts the arm product from the
track's base product, `arm_to_base` the reverse. Every non-base row of the t112 MANIFEST.tsv gives
two pairs. Counts (`counts25.npz`) and -log10 p (`pval25.npz`) are two separate spaces; each pair
is fitted and scored in both.

**Rungs.**
  noSolution        the prediction of X' is X itself.
  QuantileMatching  a monotone value-axis map g, fitted once per (pair, space) on the training
                    chromosomes by matching empirical quantiles. Source bins are sorted; a distinct
                    source value v that occupies the rank block [a, b) maps to the MEAN of the
                    sorted target values at ranks [a, b) (the tie-block mean — most bins are 0, so
                    ties are the rule). Values not seen in training are linearly interpolated
                    between the neighbouring fitted source values and clamped at both ends.

**Scoring, each rung two ways.**
  point   CRPS of a point forecast = |prediction - X'|.
  spread  counts: NB with mean = max(prediction, NB_MEAN_FLOOR) and one dispersion n per
          (pair, rung), fitted by maximum likelihood on the training chromosomes;
          p: Gaussian with mean = prediction and one sigma per (pair, rung), the ML estimate
          sqrt(mean squared training residual). NB CRPS and the CRPS split come from
          `candi.bench.distributional.nb_suite` (which calls `candi.metrics.nb_crps`); Gaussian
          CRPS is `candi.bench.distributional.gauss_crps`. Nothing here re-derives a CRPS.

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
(finite, >= 0), each of length floor(chrom_len / 25).

Memory is bounded by streaming: training is one chromosome at a time and keeps only histograms
(counts: value histograms and the joint (X, X') histogram; p: distinct values with counts), plus a
second chromosome-at-a-time pass for the p QuantileMatching residuals. Only the evaluation
chromosomes (chr19 + chr21, ~4.2 M bins; chr22 ~0.8 M) are held whole.

    python tools/t118/baseline_rungs.py pairs --manifest MANIFEST.tsv
    python tools/t118/baseline_rungs.py run --manifest MANIFEST.tsv --products DIR \
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
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

from candi.bench.distributional import gauss_crps, nb_suite, p_from_mu
from candi.metrics import spearman
from candi.store.genome import blacklist_bin_flags, read_blacklist

RES = 25
SCORE_CHROMS = ("chr19", "chr21")
VAL_CHROMS = ("chr22",)
DROP_CHROMS = frozenset({"chrY", "chrM"})
#: floor on the NB mean. The identity rung predicts 0 wherever X = 0, and an NB with mean 0 is a
#: point mass that gives any X' > 0 zero likelihood. 1e-3 is the floor `marginal_nb` already uses.
NB_MEAN_FLOOR = 1e-3
#: search interval of the NB dispersion n (size). The upper end is reached when X' equals the
#: prediction (the Poisson limit is still wider than a zero residual); nb_crps scores n > 1e4 in
#: its sd-standardised Poisson limit, so a fit at the bound is still scoreable. The cap is 1e6
#: and not higher because `p_from_mu` clips p at 1 - P_EPS (1e-9): above n = 1e6 that clip, not
#: NB_MEAN_FLOOR, would set the smallest mean (n * 1e-9), silently raising the floor.
DISPERSION_BOUNDS = (1e-4, 1e6)
TOP_FRACTION = 0.01
SUBSETS = ("all", "nonzero", "top1")
RUNGS = ("noSolution", "QuantileMatching")
SPACES = {"counts": ("counts25.npz", np.uint32), "pval": ("pval25.npz", np.float32)}
DIRECTIONS = ("base_to_arm", "arm_to_base")
MARK_CLASS = {"DNase-seq": "DNase",
              "H3K27ac": "narrow", "H3K4me3": "narrow", "H3K4me1": "narrow",
              "H3K27me3": "broad", "H3K36me3": "broad", "H3K9me3": "broad"}
SPLIT_KEYS = ("crps", "crps_oracle_scaled", "scale_error", "crps_oracle_scaled_and_n",
              "c_star", "n_star_log2", "coverage_95", "ece", "marg_crps", "beats_marginal")


# ---------------------------------------------------------------------------------------------
# manifest -> pairs
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
# spreads
# ---------------------------------------------------------------------------------------------


def nb_loglik(n: float, mu, y, w) -> float:
    """Weighted NB log-likelihood, mean parameterisation, p = n / (n + mu) as `p_from_mu`."""
    p = p_from_mu(np.full(mu.shape, n), mu)
    return float(np.dot(w, gammaln(y + n) - gammaln(n) - gammaln(y + 1.0)
                        + n * np.log(p) + y * np.log1p(-p)))


def nb_fit_dispersion(mu, y, w) -> dict:
    """ML dispersion n of NB(mean = mu) for observations y with weights w; log n bounded."""
    mu = np.maximum(np.asarray(mu, np.float64), NB_MEAN_FLOOR)
    y, w = np.asarray(y, np.float64), np.asarray(w, np.float64)
    lo, hi = np.log(DISPERSION_BOUNDS[0]), np.log(DISPERSION_BOUNDS[1])
    res = minimize_scalar(lambda t: -nb_loglik(float(np.exp(t)), mu, y, w), bounds=(lo, hi),
                          method="bounded", options={"xatol": 1e-6})
    n = float(np.exp(res.x))
    return {"n": n, "at_bound": bool(min(res.x - lo, hi - res.x) < 1e-2),
            "mean_loglik": float(-res.fun / w.sum())}


# ---------------------------------------------------------------------------------------------
# fit one pair in one space (training chromosomes only)
# ---------------------------------------------------------------------------------------------


def fit_space(src: Track, tgt: Track, train_chroms, space: str) -> dict:
    n_train, identical = 0, True
    if space == "counts":
        sh, th, joint = [], [], []
        for c in train_chroms:
            x, y = paired(src, tgt, c)
            n_train += x.size
            identical = identical and bool(np.array_equal(x, y))
            for arr, acc in ((x, sh), (y, th)):
                h = np.bincount(arr)
                nz = np.flatnonzero(h)
                acc.append((nz, h[nz]))
            key = (x.astype(np.uint64) << np.uint64(32)) | y.astype(np.uint64)
            joint.append(np.unique(key, return_counts=True))
            del x, y, key
        knots = qm_fit(*merge_hist(sh), *merge_hist(th))
        keys, w = merge_hist(joint)
        xu = (keys >> np.uint64(32)).astype(np.float64)
        yu = (keys & np.uint64(0xFFFFFFFF)).astype(np.float64)
        means = {"noSolution": xu, "QuantileMatching": qm_apply(knots, xu)}
        spread = {r: nb_fit_dispersion(m, yu, w) for r, m in means.items()}
        extra = {"n_joint_pairs": int(keys.size)}
    else:
        sh, th, ss_id = [], [], 0.0
        for c in train_chroms:
            x, y = paired(src, tgt, c)
            n_train += x.size
            identical = identical and bool(np.array_equal(x, y))
            sh.append(np.unique(x, return_counts=True))
            th.append(np.unique(y, return_counts=True))
            ss_id += float(np.sum((y.astype(np.float64) - x) ** 2))
            del x, y
        knots = qm_fit(*merge_hist(sh), *merge_hist(th))
        ss_qm = 0.0                                   # second pass: residuals need the fitted g
        for c in train_chroms:
            x, y = paired(src, tgt, c)
            ss_qm += float(np.sum((y.astype(np.float64) - qm_apply(knots, x)) ** 2))
            del x, y
        spread = {"noSolution": {"sigma": math.sqrt(ss_id / n_train)},
                  "QuantileMatching": {"sigma": math.sqrt(ss_qm / n_train)}}
        extra = {}
    return {"n_train_bins": n_train, "source_equals_target_train": identical,
            "qm_n_knots": int(knots[0].size), "spread": spread, "knots": knots, **extra}


# ---------------------------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------------------------


def subset_index(y: np.ndarray) -> dict:
    """Index arrays of the three subsets over the evaluated bins (see module docstring)."""
    k = int(math.ceil(TOP_FRACTION * y.size))
    order = np.argsort(-y.astype(np.float64), kind="stable")
    return {"all": np.arange(y.size), "nonzero": np.flatnonzero(y > 0), "top1": np.sort(order[:k])}


def score_rung(pred: np.ndarray, y: np.ndarray, space: str, spread: dict, subsets: dict) -> dict:
    y64 = y.astype(np.float64)
    out: dict = {}
    abs_err = np.abs(pred - y64)
    if space == "pval":
        per_bin = gauss_crps(pred, np.full(pred.shape, spread["sigma"]), y64)
    for s, idx in subsets.items():
        m = idx.size
        out[f"n_{s}"] = int(m)
        if m == 0:
            for key in ("point_crps", "spread_crps", "spearman"):
                out[f"{key}_{s}"] = None
            continue
        out[f"point_crps_{s}"] = float(abs_err[idx].mean())
        out[f"spearman_{s}"] = spearman(pred[idx], y64[idx])
        if space == "pval":
            out[f"spread_crps_{s}"] = float(per_bin[idx].mean())
        else:
            mu = np.maximum(pred[idx], NB_MEAN_FLOOR)
            suite = nb_suite(np.full(m, spread["n"]), mu, y64[idx], with_marginal=True)
            out[f"spread_crps_{s}"] = suite["crps"]
            for key in SPLIT_KEYS:
                v = suite.get(key)
                out[f"{key}_{s}"] = v if v is None or isinstance(v, bool) else float(v)
    if space == "counts":
        out["frac_bins_at_nb_floor"] = float(np.mean(pred < NB_MEAN_FLOOR))
    return out


def run_pair(pair: dict, products, blacklist_path, out_dir, git_sha: str = "") -> Path:
    t0 = time.time()
    blacklist = read_blacklist(blacklist_path)
    out = {"pair": pair, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "git_sha": git_sha,
           "config": {"score_chroms": list(SCORE_CHROMS), "val_chroms": list(VAL_CHROMS),
                      "dropped_chroms": sorted(DROP_CHROMS), "nb_mean_floor": NB_MEAN_FLOOR,
                      "dispersion_bounds": list(DISPERSION_BOUNDS), "top_fraction": TOP_FRACTION,
                      "qm_ties": "tie-block mean of the matched target quantiles",
                      "qm_unseen": "linear interpolation between fitted source values, clamped",
                      "blacklist": str(blacklist_path),
                      "blacklist_sha256": hashlib.sha256(Path(blacklist_path).read_bytes())
                      .hexdigest()},
           "spaces": {}, "records": []}
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
                           **{f"fit_{k}": v for k, v in fit["spread"][rung].items()},
                           **score_rung(pred, d["y"], space, fit["spread"][rung], sub)}
                    out["records"].append(rec)
                del d, sub, preds
            out["spaces"][space] = sp
    out["seconds"] = round(time.time() - t0, 1)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{pair_name(pair)}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


# ---------------------------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------------------------

TABLE_METRICS = ("spread_crps_all", "spread_crps_nonzero", "spread_crps_top1", "point_crps_all",
                 "spearman_all", "spearman_top1")


def _mean(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else None


def aggregate(manifest, out_dir, tsv_path=None, md_path=None) -> dict:
    """One TSV row per (pair, space, rung, eval); a markdown summary of the scored set.

    Per mark class: mean over the class's tracks of each track's mean over its pairs (both
    directions pooled). Per track and per arm: mean over pairs.
    """
    out_dir = Path(out_dir)
    pairs = list_pairs(read_manifest(manifest))
    rows, missing = [], []
    for p in pairs:
        f = out_dir / f"{pair_name(p)}.json"
        if not f.is_file():
            missing.append(pair_name(p))
            continue
        res = json.loads(f.read_text("utf-8"))
        for rec in res["records"]:
            rows.append({**{k: p[k] for k in ("index", "track", "assay", "mark_class", "arm",
                                              "level", "arm_pid", "direction", "source_pid",
                                              "target_pid")}, **rec})
    cols = list(dict.fromkeys(k for r in rows for k in r))
    tsv_path = Path(tsv_path or out_dir / "baseline_rungs.tsv")
    with tsv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", restval="")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})

    scored = [r for r in rows if r["eval"] == "score"]
    lines = [f"# t118 baseline rungs — scored on {' + '.join(SCORE_CHROMS)}, blacklist removed",
             "",
             f"{len(pairs) - len(missing)} of {len(pairs)} pairs present. Counts: NB CRPS "
             f"(mean floored at {NB_MEAN_FLOOR}); p: Gaussian CRPS in -log10 p. `point` = CRPS of "
             "the point forecast = MAE. QuantileMatching ties: tie-block mean. Mark class = mean "
             "over tracks of each track's mean over pairs.", ""]
    if missing:
        lines += [f"Missing: {', '.join(missing)}", ""]

    def table(title, key_fn, macro_over_tracks):
        lines.extend([f"## {title}", "",
                      "| group | space | rung | " + " | ".join(TABLE_METRICS) + " |",
                      "|---|---|---|" + "---|" * len(TABLE_METRICS)])
        groups = sorted({(key_fn(r), r["space"], r["rung"]) for r in scored})
        for g, space, rung in groups:
            sel = [r for r in scored if key_fn(r) == g and r["space"] == space and r["rung"] == rung]
            vals = []
            for m in TABLE_METRICS:
                if macro_over_tracks:
                    per_track = [_mean([r.get(m) for r in sel if r["track"] == t])
                                 for t in sorted({r["track"] for r in sel})]
                    v = _mean(per_track)
                else:
                    v = _mean([r.get(m) for r in sel])
                vals.append("—" if v is None else f"{v:.4f}")
            lines.append(f"| {g} | {space} | {rung} | " + " | ".join(vals) + " |")
        lines.append("")

    table("By mark class", lambda r: r["mark_class"], True)
    table("By track", lambda r: f"{r['track']} {r['assay']}", False)
    table("By arm", lambda r: r["arm"], False)
    md_path = Path(md_path or out_dir / "baseline_rungs.md")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"n_rows": len(rows), "missing": missing, "tsv": str(tsv_path), "md": str(md_path)}


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
    r = sub.add_parser("run", help="fit and score one pair")
    r.add_argument("--manifest", required=True, type=Path)
    r.add_argument("--products", required=True, type=Path)
    r.add_argument("--blacklist", required=True, type=Path)
    r.add_argument("--out", required=True, type=Path)
    r.add_argument("--index", required=True, type=int)
    r.add_argument("--overwrite", action="store_true")
    g = sub.add_parser("aggregate", help="per-pair JSONs -> TSV + markdown")
    g.add_argument("--manifest", required=True, type=Path)
    g.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    pairs = list_pairs(read_manifest(args.manifest))
    if args.cmd == "pairs":
        cols = ("index", "track", "assay", "mark_class", "arm", "level", "direction",
                "source_pid", "target_pid")
        print("\t".join(cols))
        for p in pairs:
            print("\t".join(str(p[c]) for c in cols))
        return 0
    if args.cmd == "run":
        if not 0 <= args.index < len(pairs):
            print(f"index {args.index} outside 0..{len(pairs) - 1}", file=sys.stderr)
            return 2
        p = pairs[args.index]
        dest = args.out / f"{pair_name(p)}.json"
        if dest.exists() and not args.overwrite:
            print(f"{dest} exists; pass --overwrite to redo")
            return 0
        path = run_pair(p, args.products, args.blacklist, args.out, _git_sha())
        print(f"{pair_name(p)} -> {path}")
        return 0
    s = aggregate(args.manifest, args.out)
    print(f"{s['n_rows']} rows -> {s['tsv']}; summary {s['md']}; missing {len(s['missing'])}")
    return 0 if not s["missing"] else 1


if __name__ == "__main__":
    sys.exit(main())
