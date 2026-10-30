"""t141 (q11): where along the genome each knob acts -- the one-warp residual against the t138 features.

Exploratory (PI, 2026-10-29); readings only, no verdicts, no thresholds.

    python map_knobs.py run <arms1k_dir> <features.npz> <products_manifest> <t138_out_dir> <out_dir>

Input: `arms.py`'s `<arms1k_dir>/<pid>.npz` per arm product (a `<pid>.DROPPED` marker for the DNase
mapq arms). For every arm product, space it was scored in, and series (`arm`; for the depth arm
also `null_s0`, `null_s1`, the binomial-thinning null, counts only), per 1 kb bin over the feature
chromosomes (the t138 grid; no blacklist, as t138):

  d      = |r|, the size of the one-warp residual (counts: |log2((arm+0.5)/(warp(base)+0.5))|;
           pval: |arm - warp(base)|, a difference on the -log10 p scale)
  level  = counts: log2(arm + warp(base) + 1); pval: arm + warp(base). Only its ranks matter (the
           null bins it by quantile). Bins with arm + warp(base) == 0 are dropped (t138's a + b == 0).
  d_hat  = the NULL, fitted FIRST: t138 `map.level_null` (mean of d in 40 quantile bins of level)
  r      = d - d_hat; every feature reading is on r

The readings are t138 `map.readings` with d and level supplied directly (`readings` below; on counts
with a = arm, b = warp(base) it returns exactly what `map.readings(a, b)` returns -- a test pins
this), plus: median_d (the residual size), and delta_R2 per feature: delta_R2_<f> = OLS R^2 of d on
[1, d_hat, f] (ctx: [1, d_hat, ctx==1, ctx==2]) minus R2_level_ols, and its shuffle control
delta_R2_<f>_shuffle (that feature's rows permuted, seed 0). A residual that is zero to 1e-9
everywhere (an arm bit-identical to its base, or an exact monotone transform) has undefined R^2:
the R^2 readings are NaN, every delta_R2 is 0 and `residual_zero` is 1.

The floor: the t138 pseudoreplicate map at matched depth, joined by mark class and depth: the arm's
MANIFEST depth equals the t138 reads_per_half (15M <-> "15", 3.75M <-> "375"), else NA; counts
only (t138 maps counts halves); NA everywhere when `<t138_out_dir>` lacks the table.

Outputs (atomic, .tmp then rename): knob_products.tsv (pid x space x series), knob_classes.tsv
(arm x level x mark_class x space x series x reading: median, min, max over tracks; floor columns),
knob_context.tsv (... x context 0 other / 1 domain / 2 peak), knob_deciles.tsv (... x feature x
decile), config.json. Needs numpy + scipy only (Nibi scipy-stack).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve()
for _d in ("t118", "t138"):
    if str(HERE.parents[1] / _d) not in sys.path:
        sys.path.insert(0, str(HERE.parents[1] / _d))
import map as M  # noqa: E402  (tools/t138/map.py)
from knob_effects import MARK_CLASS  # noqa: E402

CONTROL_P_ONLY = ("ctlid", "ctldepth", "ratio")
ZERO_TOL = 1e-9
FLOOR_DEPTHS = {15_000_000: "15", 3_750_000: "375"}
FLOOR_PRODUCT_READINGS = ("mean_d", "R2_level", "delta_R2", "delta_R2_shuffle")
GROUP = ("arm", "level", "mark_class", "space", "series")
META = ["pid", "track", "assay", "mark_class", "arm", "level", "knob", "depth", "space", "series",
        "identical_to_base"]


def spaces_for(mark_class: str, arm: str) -> tuple:
    """Same rule as arms.spaces_for (not imported: arms.py needs the candi venv)."""
    if mark_class == "DNase" and arm == "mapq":
        return ()
    if arm in CONTROL_P_ONLY:
        return ("pval",)
    return ("counts", "pval")


def _r2_delta(d, X0, extra, perm):
    base = M.r2_ols(d, X0)
    real = M.r2_ols(d, np.column_stack([X0, extra]))
    shuf = M.r2_ols(d, np.column_stack([X0, extra[perm]]))
    return real - base, shuf - base


def readings(d: np.ndarray, level: np.ndarray, keep: np.ndarray, F: dict, ctx: np.ndarray,
             feats) -> dict:
    """t138 map.readings on a supplied d and level (same formulas, same seed), plus extras."""
    d, level, ctx = d[keep].astype(np.float64), level[keep].astype(np.float64), ctx[keep]
    Fk = {f: F[f][keep] for f in feats}
    dh = M.level_null(d, level)
    r = d - dh
    n = d.size
    zero = bool(n == 0 or np.max(np.abs(d)) <= ZERO_TOL)
    out = {"n_bins": n, "median_d": float(np.median(d)) if n else float("nan"),
           "mean_d": float(d.mean()), "mean_r": float(r.mean()), "residual_zero": int(zero)}
    vd = d.var()
    out["R2_level"] = float(1.0 - r.var() / vd) if vd > 0 and not zero else float("nan")

    rng = np.random.default_rng(M.SEED)
    sub = (np.sort(rng.choice(n, size=M.SUBSAMPLE, replace=False)) if n > M.SUBSAMPLE
           else np.arange(n))
    for f in M.ALL_FEATURES:
        if f in feats and not zero:
            x, y = r[sub], Fk[f][sub]
            ok = np.isfinite(y)
            rho = spearmanr(x[ok], y[ok]).correlation if ok.sum() > 2 else np.nan
            out[f"spearman_r_{f}"] = M._nan(rho)
        else:
            out[f"spearman_r_{f}"] = float("nan")

    ok = np.ones(sub.size, dtype=bool)
    for f in feats:
        ok &= np.isfinite(Fk[f][sub])
    rows = sub[ok]
    Fs = {f: Fk[f][rows] for f in feats}
    X = M.design(dh[rows], Fs, ctx[rows], feats)
    perm = np.random.default_rng(M.SEED).permutation(rows.size)
    per_feat = list(feats) + ["ctx"]
    if zero:
        for k in ("R2_level_ols", "R2_level_feat", "R2_level_feat_shuffle"):
            out[k] = float("nan")
        out["delta_R2"] = out["delta_R2_shuffle"] = 0.0
        for f in M.ALL_FEATURES + ("ctx",):
            out[f"delta_R2_{f}"] = 0.0 if f in per_feat else float("nan")
            out[f"delta_R2_{f}_shuffle"] = 0.0 if f in per_feat else float("nan")
    else:
        out["R2_level_ols"] = M.r2_ols(d[rows], X[:, :2])
        out["R2_level_feat"] = M.r2_ols(d[rows], X)
        out["delta_R2"] = out["R2_level_feat"] - out["R2_level_ols"]
        Xs = M.design(dh[rows], {f: Fs[f][perm] for f in feats}, ctx[rows][perm], feats)
        out["R2_level_feat_shuffle"] = M.r2_ols(d[rows], Xs)
        out["delta_R2_shuffle"] = out["R2_level_feat_shuffle"] - out["R2_level_ols"]
        X0 = X[:, :2]
        for f in M.ALL_FEATURES + ("ctx",):
            if f not in per_feat:
                out[f"delta_R2_{f}"] = out[f"delta_R2_{f}_shuffle"] = float("nan")
                continue
            extra = (np.column_stack([(ctx[rows] == 1), (ctx[rows] == 2)]).astype(float)
                     if f == "ctx" else Fs[f][:, None])
            out[f"delta_R2_{f}"], out[f"delta_R2_{f}_shuffle"] = _r2_delta(d[rows], X0, extra,
                                                                            perm)

    n_top = max(1, int(np.ceil(M.TOP_FRACTION * n)))
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
    dec = {f: M.deciles(Fk[f]) for f in feats}
    out["enr_map_bottom_decile"] = enrich(dec["map"] == 0) if "map" in feats else float("nan")
    out["enr_rep_top_decile"] = enrich(dec["rep"] == 9) if "rep" in feats else float("nan")
    for f in M.ALL_FEATURES:
        for k in range(10):
            m = dec[f] == k if f in feats else np.zeros(n, dtype=bool)
            out[f"mean_r_{f}_dec{k}"] = float(r[m].mean()) if m.any() else float("nan")
            out[f"mean_d_{f}_dec{k}"] = float(d[m].mean()) if m.any() else float("nan")
    return out


def series_inputs(z, chroms, space: str, series: str):
    """(d, level, keep) of one series of an arms1k npz, concatenated over chroms."""
    cat = lambda key: np.concatenate([z[f"{key}_{c}"].astype(np.float64) for c in chroms])  # noqa
    if series == "arm":
        tgt, w, r = cat(f"{space}_arm"), cat(f"{space}_warp"), cat(f"{space}_r")
    else:
        s = series[-1]
        tgt, w, r = cat(f"counts_null{s}"), cat(f"counts_nullwarp{s}"), cat(f"counts_nullr{s}")
    level = np.log2(tgt + w + 1.0) if space == "counts" else tgt + w
    return np.abs(r), level, (tgt + w) > 0


def read_floor(t138_out: Path) -> dict:
    """The t138 tables, keyed for the join; {} entries when a table is absent."""
    out = {"classes": {}, "context": {}, "deciles": {}}
    p = Path(t138_out)
    if (p / "map_classes.tsv").exists():
        for r in _tsv(p / "map_classes.tsv"):
            out["classes"][(r["mark_class"], r["depth"], r["reading"])] = float(r["median"])
    if (p / "map_context.tsv").exists():
        for r in _tsv(p / "map_context.tsv"):
            out["context"][(r["mark_class"], r["depth"], int(r["context"]))] = r
    if (p / "map_deciles.tsv").exists():
        for r in _tsv(p / "map_deciles.tsv"):
            out["deciles"][(r["mark_class"], r["depth"], r["feature"], int(r["decile"]))] = \
                float(r["mean_r"])
    return out


def _tsv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def floor_depth(depth: str, space: str) -> str:
    if space != "counts":
        return "NA"
    try:
        return FLOOR_DEPTHS.get(int(float(depth)), "NA")
    except ValueError:
        return "NA"


def _fget(d: dict, key, field=None):
    v = d.get(key)
    if v is None:
        return float("nan")
    return float(v[field]) if field else float(v)


def write_tsv_atomic(path: Path, rows: list, cols: list):
    tmp = path.with_name(path.name + ".tmp")
    M.write_tsv(tmp, rows, cols)
    tmp.replace(path)


def run(arms1k_dir, features, manifest, t138_out, out_dir) -> dict:
    arms1k_dir, features, manifest, t138_out, out_dir = map(
        Path, (arms1k_dir, features, manifest, t138_out, out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    chroms, lengths, feats, F, ctx_all = M.load_features(features)
    products = [r for r in M.read_manifest(manifest) if r["arm"] != "base"]
    missing, dropped, rows = [], [], []
    for p in products:
        pid = p["pid"]
        mc = MARK_CLASS[p["assay"]]
        if not spaces_for(mc, p["arm"]):
            dropped.append(pid)
            continue
        if not (arms1k_dir / f"{pid}.npz").exists():
            missing.append(pid)
    if missing:
        raise SystemExit(f"{len(missing)} arm npz missing under {arms1k_dir}: {missing}")

    floor = read_floor(t138_out)
    for p in products:
        pid = p["pid"]
        if pid in dropped:
            continue
        track = p["track"]
        if track not in ctx_all:
            raise SystemExit(f"{pid}: no ctx_{track}_* in {features}")
        with np.load(arms1k_dir / f"{pid}.npz", allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            for c in chroms:
                k = f"{meta['spaces'][0]}_arm_{c}"
                if k not in z.files or z[k].size != lengths[c]:
                    raise SystemExit(f"{pid} {c}: arms1k length differs from features {lengths[c]}")
            for space in meta["spaces"]:
                series = ["arm"] + (["null_s0", "null_s1"]
                                    if space == "counts" and "null" in meta else [])
                for s in series:
                    d, level, keep = series_inputs(z, chroms, space, s)
                    rd = readings(d, level, keep, F, ctx_all[track], feats)
                    fd = floor_depth(meta["depth"], space)
                    row = {"pid": pid, "track": track, "assay": p["assay"], "mark_class":
                           MARK_CLASS[p["assay"]], "arm": p["arm"], "level": p["level"],
                           "knob": p.get("knob", ""), "depth": meta["depth"], "space": space,
                           "series": s, "identical_to_base": int(meta["identical_to_base"][space]),
                           **rd, "floor_depth": fd}
                    for k in FLOOR_PRODUCT_READINGS:
                        row[f"floor_{k}"] = _fget(floor["classes"], (row["mark_class"], fd, k))
                    rows.append(row)
        print(f"{pid} done", flush=True)

    floor_cols = ["floor_depth"] + [f"floor_{k}" for k in FLOOR_PRODUCT_READINGS]
    reading_cols = [k for k in rows[0] if k not in META and k not in floor_cols] if rows else []
    write_tsv_atomic(out_dir / "knob_products.tsv", rows, META + reading_cols + floor_cols)

    groups: dict = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in GROUP), []).append(r)
    cls_rows, ctx_rows, dec_rows = [], [], []
    for key, sel in sorted(groups.items()):
        g = dict(zip(GROUP, key))
        fd = sel[0]["floor_depth"]
        common = {**g, "depth": sel[0]["depth"], "n_products": len(sel), "floor_depth": fd}
        for k in reading_cols:
            med, lo, hi = M._agg([r[k] for r in sel])
            cls_rows.append({**common, "reading": k, "median": med, "min": lo, "max": hi,
                             "floor_median": _fget(floor["classes"], (g["mark_class"], fd, k))})
        for c in (0, 1, 2):
            fk = (g["mark_class"], fd, c)
            ctx_rows.append({
                **common, "context": c, "context_label": M.CTX_LABEL[c],
                "mean_d": M._agg([r[f"mean_d_ctx{c}"] for r in sel])[0],
                "mean_r": M._agg([r[f"mean_r_ctx{c}"] for r in sel])[0],
                "share_bins": M._agg([r[f"share_ctx{c}"] for r in sel])[0],
                "enrichment": M._agg([r[f"enr_ctx{c}"] for r in sel])[0],
                "floor_mean_d": _fget(floor["context"], fk, "mean_d"),
                "floor_mean_r": _fget(floor["context"], fk, "mean_r"),
                "floor_enrichment": _fget(floor["context"], fk, "enrichment")})
        for f in feats:
            for k in range(10):
                dec_rows.append({
                    **common, "feature": f, "decile": k,
                    "mean_r": M._agg([r[f"mean_r_{f}_dec{k}"] for r in sel])[0],
                    "mean_d": M._agg([r[f"mean_d_{f}_dec{k}"] for r in sel])[0],
                    "floor_mean_r": _fget(floor["deciles"], (g["mark_class"], fd, f, k))})
    head = list(GROUP) + ["depth", "n_products", "floor_depth"]
    write_tsv_atomic(out_dir / "knob_classes.tsv", cls_rows,
                     head + ["reading", "median", "min", "max", "floor_median"])
    write_tsv_atomic(out_dir / "knob_context.tsv", ctx_rows,
                     head + ["context", "context_label", "mean_d", "mean_r", "share_bins",
                             "enrichment", "floor_mean_d", "floor_mean_r", "floor_enrichment"])
    write_tsv_atomic(out_dir / "knob_deciles.tsv", dec_rows,
                     head + ["feature", "decile", "mean_r", "mean_d", "floor_mean_r"])

    config = {
        "inputs": {"arms1k_dir": str(arms1k_dir), "features": str(features),
                   "manifest": str(manifest), "t138_out": str(t138_out)},
        "floor_tables_present": {k: bool(v) for k, v in floor.items()},
        "features": list(feats) + ["ctx"], "mappability_missing": "map" not in feats,
        "chroms": chroms, "n_arm_products": len(products), "dropped": dropped,
        "n_rows": len(rows),
        "space_rules": {"pval_only": list(CONTROL_P_ONLY), "dropped": "DNase mapq",
                        "both": "every other knob, extsize included",
                        "null": "depth arm, counts only (null_s0, null_s1)"},
        "formulas": {
            "d": "|r_warp|; counts r_warp = log2((arm+0.5)/(warp(base)+0.5)), pval r_warp = "
                 "arm - warp(base) (difference on the -log10 p scale)",
            "level": "counts log2(arm + warp(base) + 1); pval arm + warp(base); bins with "
                     "arm + warp(base) == 0 dropped",
            "d_hat": "t138 map.level_null: mean of d in 40 quantile bins of level (fitted first)",
            "r": "d - d_hat", "median_d": "median of d over kept bins (the residual size)",
            "delta_R2": "t138: OLS R^2 of d on [1, d_hat, features, ctx] - on [1, d_hat]",
            "delta_R2_<f>": "OLS R^2 of d on [1, d_hat, f] (ctx: two one-hot columns) - on "
                            "[1, d_hat]; _shuffle: the f rows permuted (seed 0)",
            "residual_zero": "max d <= 1e-9: R^2 NaN, every delta_R2 0",
            "floor": "t138 map at reads_per_half == the arm's MANIFEST depth (15M, 3.75M), class "
                     "median; counts only; NA otherwise",
        },
        "subsample": M.SUBSAMPLE, "seed": M.SEED, "top_fraction": M.TOP_FRACTION,
        "git_sha": M.git_sha(), "note": "exploratory readings; no verdicts, no thresholds",
    }
    tmp = out_dir / "config.json.tmp"
    tmp.write_text(json.dumps(config, indent=1))
    tmp.replace(out_dir / "config.json")
    return config


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    if len(a) != 6 or a[0] != "run":
        raise SystemExit("usage: map_knobs.py run <arms1k_dir> <features.npz> <products_manifest> "
                         "<t138_out_dir> <out_dir>")
    run(*a[1:])


if __name__ == "__main__":
    main()
