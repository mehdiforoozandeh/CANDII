"""t138: where does the split-half disagreement concentrate? (exploratory; readings, no verdicts)

    python map.py run <bins1k_dir> <features.npz> <products_manifest> <out_dir>

For each product of the manifest and each depth in (15, 375) -- the untouched halves a15/b15 and the
ratio-0.25 halves a375/b375 of G1's bins1k layout -- per 1 kb bin over the feature chromosomes:

  d      = |log2((a + 0.5) / (b + 0.5))|            (bins with a + b == 0 dropped; no blacklist)
  level  = log2(a + b + 1)
  d_hat  = the NULL: the median of d in 40 quantile bins of level, linearly interpolated in level
  r      = d - d_hat                                  (the residual the features are read against)

Signal level is fitted FIRST; every feature reading is on r, never on d alone.

Readings per (pid, depth): Spearman(r, gc|map|rep) on a fixed-seed 500 000-bin subsample; mean r and
mean d per context class (0 other, 1 domain, 2 peak) and per feature decile (rank deciles over the
kept bins, ties averaged); R2_level = 1 - var(d - d_hat)/var(d) over the kept bins; R2_level_feat =
OLS R^2 of d on [1, d_hat, gc, map, rep, ctx==1, ctx==2] on the subsample; R2_level_ols = OLS R^2
of d on [1, d_hat] on the same rows; delta_R2 = R2_level_feat - R2_level_ols (both OLS, so a
level-only world reads ~0; the brief's R2_level here read 0.18 on pure Poisson halves);
delta_R2_shuffle = the same with the feature rows permuted jointly (seed 0); enrichment = the
top-1 % r bins' share in a class / the class's share of all kept bins, for each context class, the
bottom mappability decile and the top repeat decile.

Outputs: map_products.tsv, map_classes.tsv, map_context.tsv, map_deciles.tsv, map_check.txt,
config.json. Needs numpy + scipy only (Nibi scipy-stack).
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

_T118 = Path(__file__).resolve().parents[1] / "t118"
if str(_T118) not in sys.path:
    sys.path.insert(0, str(_T118))
from knob_effects import MARK_CLASS  # noqa: E402

DEPTHS = {"15": ("a15", "b15", 15_000_000), "375": ("a375", "b375", 3_750_000)}
N_LEVEL_BINS = 40
SUBSAMPLE = 500_000
SEED = 0
TOP_FRACTION = 0.01
SHUFFLE_PASS_FRACTION = 0.9
CTX_LABEL = {0: "other", 1: "domain", 2: "peak"}
ALL_FEATURES = ("gc", "map", "rep")


def read_manifest(path: Path) -> list:
    with open(path) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    seen, out = set(), []
    for r in rows:
        if r["pid"] not in seen:
            seen.add(r["pid"])
            out.append(r)
    return sorted(out, key=lambda r: r["pid"])


def load_features(path: Path):
    z = np.load(path, allow_pickle=False)
    chroms = sorted({k[3:] for k in z.files if k.startswith("gc_")})
    feats = [f for f in ALL_FEATURES if all(f"{f}_{c}" in z.files for c in chroms)]
    cat = {f: np.concatenate([z[f"{f}_{c}"].astype(np.float64) for c in chroms]) for f in feats}
    c0 = "_" + chroms[0]
    tracks = sorted({k[4:-len(c0)] for k in z.files if k.startswith("ctx_") and k.endswith(c0)})
    ctx = {t: np.concatenate([z[f"ctx_{t}_{c}"] for c in chroms]).astype(np.int64)
           for t in tracks}
    lengths = {c: int(z[f"gc_{c}"].size) for c in chroms}
    z.close()
    return chroms, lengths, feats, cat, ctx


def level_null(d: np.ndarray, level: np.ndarray, n_bins: int = N_LEVEL_BINS) -> np.ndarray:
    """d_hat(level): the median of d in quantile bins of level, interpolated in level."""
    edges = np.unique(np.quantile(level, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 2:
        return np.full(d.shape, np.median(d))
    idx = np.clip(np.searchsorted(edges, level, side="right") - 1, 0, edges.size - 2)
    centers, meds = [], []
    for k in range(edges.size - 1):
        m = idx == k
        if m.any():
            centers.append(np.median(level[m]))
            meds.append(np.median(d[m]))
    centers, meds = np.asarray(centers), np.asarray(meds)
    o = np.argsort(centers, kind="stable")
    return np.interp(level, centers[o], meds[o])


def deciles(x: np.ndarray) -> np.ndarray:
    """Rank decile 0..9 over the finite values (ties averaged); -1 where x is NaN."""
    out = np.full(x.shape, -1, dtype=np.int64)
    ok = np.isfinite(x)
    n = int(ok.sum())
    if n:
        rk = rankdata(x[ok], method="average")
        out[ok] = np.minimum(((rk - 1) * 10 // n).astype(np.int64), 9)
    return out


def r2_ols(y: np.ndarray, X: np.ndarray) -> float:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    tot = y - y.mean()
    return float(1.0 - (res @ res) / (tot @ tot)) if tot @ tot > 0 else float("nan")


def design(dh, F: dict, ctx, feats) -> np.ndarray:
    cols = [np.ones(dh.size), dh] + [F[f] for f in feats]
    cols += [(ctx == 1).astype(float), (ctx == 2).astype(float)]
    return np.column_stack(cols)


def _nan(x):
    return float(x) if np.isfinite(x) else float("nan")


def readings(a: np.ndarray, b: np.ndarray, F: dict, ctx: np.ndarray, feats) -> dict:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    keep = (a + b) > 0
    a, b, ctx = a[keep], b[keep], ctx[keep]
    Fk = {f: F[f][keep] for f in feats}
    d = np.abs(np.log2((a + 0.5) / (b + 0.5)))
    level = np.log2(a + b + 1.0)
    dh = level_null(d, level)
    r = d - dh
    n = d.size
    out = {"n_bins": n, "mean_d": float(d.mean()), "mean_r": float(r.mean())}
    vd = d.var()
    out["R2_level"] = float(1.0 - r.var() / vd) if vd > 0 else float("nan")

    rng = np.random.default_rng(SEED)
    sub = np.sort(rng.choice(n, size=SUBSAMPLE, replace=False)) if n > SUBSAMPLE else np.arange(n)
    for f in ALL_FEATURES:
        if f in feats:
            x, y = r[sub], Fk[f][sub]
            ok = np.isfinite(y)
            rho = spearmanr(x[ok], y[ok]).correlation if ok.sum() > 2 else np.nan
            out[f"spearman_r_{f}"] = _nan(rho)
        else:
            out[f"spearman_r_{f}"] = float("nan")

    # OLS on the subsample rows with every feature finite
    ok = np.ones(sub.size, dtype=bool)
    for f in feats:
        ok &= np.isfinite(Fk[f][sub])
    rows = sub[ok]
    Fs = {f: Fk[f][rows] for f in feats}
    X = design(dh[rows], Fs, ctx[rows], feats)
    # delta_R2 compares like with like: both terms are OLS fits on the same rows, so a level-only
    # world reads ~0. (1 - var(r)/var(d)) is not an OLS fit of d_hat, and the gap between the
    # median null and the conditional mean would otherwise show up as a "feature" gain.
    out["R2_level_ols"] = r2_ols(d[rows], X[:, :2])
    out["R2_level_feat"] = r2_ols(d[rows], X)
    out["delta_R2"] = out["R2_level_feat"] - out["R2_level_ols"]
    perm = np.random.default_rng(SEED).permutation(rows.size)
    Xs = design(dh[rows], {f: Fs[f][perm] for f in feats}, ctx[rows][perm], feats)
    out["R2_level_feat_shuffle"] = r2_ols(d[rows], Xs)
    out["delta_R2_shuffle"] = out["R2_level_feat_shuffle"] - out["R2_level_ols"]

    n_top = max(1, int(np.ceil(TOP_FRACTION * n)))
    top = np.zeros(n, dtype=bool)
    top[np.argsort(-r, kind="stable")[:n_top]] = True

    def enrich(mask):
        share_all = mask.mean()
        return float(mask[top].mean() / share_all) if share_all > 0 else float("nan")

    for k in (0, 1, 2):
        m = ctx == k
        out[f"share_ctx{k}"] = float(m.mean())
        out[f"mean_d_ctx{k}"] = float(d[m].mean()) if m.any() else float("nan")
        out[f"mean_r_ctx{k}"] = float(r[m].mean()) if m.any() else float("nan")
        out[f"enr_ctx{k}"] = enrich(m)
    dec = {f: deciles(Fk[f]) for f in feats}
    out["enr_map_bottom_decile"] = enrich(dec["map"] == 0) if "map" in feats else float("nan")
    out["enr_rep_top_decile"] = enrich(dec["rep"] == 9) if "rep" in feats else float("nan")
    for f in ALL_FEATURES:
        for k in range(10):
            m = dec[f] == k if f in feats else np.zeros(n, dtype=bool)
            out[f"mean_r_{f}_dec{k}"] = float(r[m].mean()) if m.any() else float("nan")
            out[f"mean_d_{f}_dec{k}"] = float(d[m].mean()) if m.any() else float("nan")
    return out


def _fmt(v):
    if isinstance(v, float):
        return "nan" if not np.isfinite(v) else f"{v:.6g}"
    return str(v)


def write_tsv(path: Path, rows: list, cols: list):
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(_fmt(r[c]) for c in cols) + "\n")


def _agg(vals):
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    if not v.size:
        return float("nan"), float("nan"), float("nan")
    return float(np.median(v)), float(v.min()), float(v.max())


def git_sha() -> str:
    if os.environ.get("GIT_SHA"):
        return os.environ["GIT_SHA"]
    try:
        return subprocess.run(["git", "-C", str(Path(__file__).parent), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


META = ["pid", "track", "assay", "mark_class", "arm", "level", "depth", "reads_per_half"]


def run(bins1k_dir: Path, features: Path, manifest: Path, out_dir: Path) -> dict:
    bins1k_dir, features, manifest, out_dir = map(Path, (bins1k_dir, features, manifest, out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    chroms, lengths, feats, F, ctx_all = load_features(features)
    products = read_manifest(manifest)
    rows = []
    for p in products:
        pid, track = p["pid"], p["track"]
        if track not in ctx_all:
            raise SystemExit(f"{pid}: no ctx_{track}_* in {features}")
        with np.load(bins1k_dir / f"{pid}.npz", allow_pickle=False) as z:
            for c in chroms:
                if z[f"a15_{c}"].size != lengths[c]:
                    raise SystemExit(f"{pid} {c}: bins1k {z[f'a15_{c}'].size} != features "
                                     f"{lengths[c]}")
            halves = {k: np.concatenate([z[f"{k}_{c}"] for c in chroms])
                      for dk in DEPTHS.values() for k in dk[:2]}
        for depth, (ka, kb, reads) in DEPTHS.items():
            rd = readings(halves[ka], halves[kb], F, ctx_all[track], feats)
            rows.append({"pid": pid, "track": track, "assay": p["assay"],
                         "mark_class": MARK_CLASS[p["assay"]], "arm": p["arm"],
                         "level": p["level"], "depth": depth, "reads_per_half": reads, **rd})
        print(f"{pid} done", flush=True)

    reading_cols = [k for k in rows[0] if k not in META] if rows else []
    write_tsv(out_dir / "map_products.tsv", rows, META + reading_cols)

    classes = sorted({r["mark_class"] for r in rows})
    cls_rows, ctx_rows, dec_rows = [], [], []
    for mc in classes:
        for depth in DEPTHS:
            sel = [r for r in rows if r["mark_class"] == mc and r["depth"] == depth]
            for k in reading_cols:
                med, lo, hi = _agg([r[k] for r in sel])
                cls_rows.append({"mark_class": mc, "depth": depth, "reading": k, "median": med,
                                 "min": lo, "max": hi, "n_products": len(sel)})
            for k in (0, 1, 2):
                ctx_rows.append({
                    "mark_class": mc, "depth": depth, "context": k, "context_label": CTX_LABEL[k],
                    "mean_d": _agg([r[f"mean_d_ctx{k}"] for r in sel])[0],
                    "mean_r": _agg([r[f"mean_r_ctx{k}"] for r in sel])[0],
                    "share_bins": _agg([r[f"share_ctx{k}"] for r in sel])[0],
                    "enrichment": _agg([r[f"enr_ctx{k}"] for r in sel])[0],
                    "n_products": len(sel)})
            for f in feats:
                for k in range(10):
                    dec_rows.append({"mark_class": mc, "depth": depth, "feature": f,
                                     "decile": k,
                                     "mean_r": _agg([r[f"mean_r_{f}_dec{k}"] for r in sel])[0],
                                     "n_products": len(sel)})
    write_tsv(out_dir / "map_classes.tsv", cls_rows,
              ["mark_class", "depth", "reading", "median", "min", "max", "n_products"])
    write_tsv(out_dir / "map_context.tsv", ctx_rows,
              ["mark_class", "depth", "context", "context_label", "mean_d", "mean_r",
               "share_bins", "enrichment", "n_products"])
    write_tsv(out_dir / "map_deciles.tsv", dec_rows,
              ["mark_class", "depth", "feature", "decile", "mean_r", "n_products"])

    expected = 2 * len(products)
    lines = [f"{'PASS' if len(rows) == expected else 'FAIL'} rows {len(rows)} expected {expected}"]
    neg = [r for r in rows if not (r["R2_level"] >= 0)]
    lines.append("PASS R2_level>=0" if not neg else
                 "FAIL R2_level<0 " + " ".join(f"{r['pid']}:{r['depth']}" for r in neg))
    beats = sum(1 for r in rows if r["delta_R2"] > r["delta_R2_shuffle"])
    frac = beats / len(rows) if rows else 0.0
    lines.append(f"{'PASS' if frac >= SHUFFLE_PASS_FRACTION else 'FAIL'} shuffle-control "
                 f"{beats}/{len(rows)}")
    (out_dir / "map_check.txt").write_text("\n".join(lines) + "\n")

    config = {
        "inputs": {"bins1k_dir": str(bins1k_dir), "features": str(features),
                   "manifest": str(manifest)},
        "features": list(feats) + ["ctx"],
        "mappability_missing": "map" not in feats,
        "n_products": len(products),
        "depths": {k: {"halves": list(v[:2]), "reads_per_half": v[2]} for k, v in DEPTHS.items()},
        "chroms": chroms,
        "formulas": {
            "d": "|log2((a + 0.5) / (b + 0.5))| per 1 kb bin; bins with a + b == 0 dropped",
            "level": "log2(a + b + 1)",
            "d_hat": f"median of d in {N_LEVEL_BINS} quantile bins of level (unique edges), "
                     "interpolated linearly at the bins' median level",
            "r": "d - d_hat",
            "R2_level": "1 - var(d - d_hat) / var(d), all kept bins",
            "R2_level_ols": "OLS R^2 of d on [1, d_hat] on the subsample rows with finite "
                            "features",
            "R2_level_feat": "OLS R^2 of d on [1, d_hat, features, ctx==1, ctx==2] on the "
                             "same rows",
            "delta_R2": "R2_level_feat - R2_level_ols (both OLS on the same rows)",
            "delta_R2_shuffle": "the same with feature + ctx rows permuted jointly (seed 0)",
            "deciles": "rank deciles over kept bins, ties averaged, NaN excluded",
            "enrichment": "share of the top-1 % r bins in the class / the class's share of kept "
                          "bins",
        },
        "subsample": SUBSAMPLE, "seed": SEED, "top_fraction": TOP_FRACTION,
        "shuffle_pass_fraction": SHUFFLE_PASS_FRACTION,
        "git_sha": git_sha(),
        "note": "exploratory readings; no verdicts",
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=1))
    return config


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    if len(a) != 5 or a[0] != "run":
        raise SystemExit("usage: map.py run <bins1k_dir> <features.npz> <products_manifest> "
                         "<out_dir>")
    run(*map(Path, a[1:]))


if __name__ == "__main__":
    main()
