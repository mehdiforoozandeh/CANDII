"""t137 — the resampling floor: how far apart two halves of ONE library are, as a function of depth.

Exploratory (PI, 2026-10-29). Each t117 pseudoreplicate pair (pr1, pr2: 15M reads each, the two
halves of one 30M t112 product) is two measurements of the same biology that differ only by read
sampling. Thinning both halves by binomial draws on the 25 bp counts walks them down a depth
ladder; at each rung the distance between the halves is the floor a processing knob must clear
before its effect can be told apart from sampling noise.

The depth ladder (`RATIOS`): thin ratio 1, 0.5, 0.25, 0.125 = 15M, 7.5M, 3.75M, 1.875M reads per
half. Seeds 0 and 1 per ratio < 1; ratio 1 is the untouched halves (seed 0, no draw). The rng of
one configuration is `default_rng([seed, pid_index, depth_index])`; pr1 is thinned first (every
chromosome of the npz but chrM / chrY, in npz order), then pr2, from the same rng object.

Metrics, on chr19 + chr21 with blacklisted 25 bp bins removed (`knob_effects.load`'s scored set):
  log2fc_top1     log2(mean target / mean source) over the product's OWN top-1 % 25 bp bins (the
                  full 30M product's peaks: one set for every rung of that product). Sign flips
                  between the two direction rows.
  shape_1k        1 - Spearman(a_1k, b_1k) on 1 kb sums (40 x 25 bp; a 1 kb bin is dropped if any
                  of its 25 bp bins is blacklisted).
  divergence_bin  mean |log2((a + 0.5) / (b + 0.5))| over the kept 25 bp bins.
  spearman_fine   Spearman(a, b) on the kept 25 bp bins.
  CRPS            per direction, the source half predicts the target half as the identity with a
                  Poisson spread (mean = max(source, 1e-3)): the rungs_v2 oracle rule. `all` bins
                  and `top1` (top 1 % of bins ranked by the TARGET, `baseline_rungs.subset_index`),
                  with the oracle_scaled / scale_error split of `baseline_rungs.scale_split`.

    python floor.py product <pseudoreps_dir> <products_dir> <blacklist.bed> <out_dir> <pid_index>
    python floor.py aggregate <out_dir> <rungs_v2_dir> [--expect 130]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "t118"))
from knob_effects import (CHROMS, COARSE, MARK_CLASS, RES, bad_bins, load,  # noqa: E402
                          read_blacklist, top_idx)
import baseline_rungs as br  # noqa: E402  (imports candi: run under the kit venv)

RATIOS = (1.0, 0.5, 0.25, 0.125)
SEEDS = (0, 1)
READS_PER_HALF = 15_000_000
TARGET_READS = 30_000_000
DROP = ("chrM", "chrY")
DIRECTIONS = ("pr1_to_pr2", "pr2_to_pr1")
BINS1K_RATIO, BINS1K_SEED = 0.25, 0
PSEUDO = 0.5
METRICS_CURVE = ("abs_log2fc_top1", "shape_1k", "divergence_bin", "crps_all", "crps_top1")
DEPTH_LABEL = {1.0: "15M", 0.5: "7.5M", 0.25: "3.75M", 0.125: "1.875M"}
COLUMNS = ("pid track assay mark_class arm level thin_ratio reads_per_half seed direction "
           "log2fc_top1 abs_log2fc_top1 shape_1k divergence_bin spearman_fine crps_all crps_top1 "
           "crps_oracle_scaled_all scale_error_all c_star_all n_all n_top1 n_coarse").split()
FORMULAS = {
    "thin": "rng.binomial(counts.astype(int64), ratio) per 25 bp bin; ratio 1 = no draw",
    "rng": "np.random.default_rng([seed, pid_index, depth_index])",
    "log2fc_top1": "log2(mean(target[top]) / mean(source[top])), top = top_idx(product fine)",
    "shape_1k": "1 - spearman(a_1k, b_1k)",
    "divergence_bin": "mean |log2((a + 0.5) / (b + 0.5))| over kept 25 bp bins",
    "crps": "mean poisson_crps(max(source, 1e-3), target); split by baseline_rungs.scale_split",
    "curve": "log(metric) = alpha + beta * log2(reads_per_half), least squares over 4 ratios",
    "value_30M_extrap": "exp(alpha + beta * log2(30e6))",
}


def thin(counts: np.ndarray, ratio: float, rng) -> np.ndarray:
    """Binomial thinning of 25 bp counts; ratio 1 returns the input unchanged (no draw)."""
    if ratio == 1.0:
        return counts
    return rng.binomial(counts.astype(np.int64), ratio).astype(np.uint32)


def rng_for(seed: int, pid_index: int, depth_index: int):
    return np.random.default_rng([seed, pid_index, depth_index])


def read_tsv(path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def sorted_pids(pseudoreps_dir) -> list[str]:
    return sorted({r["pid"] for r in read_tsv(Path(pseudoreps_dir) / "MANIFEST.tsv")})


def sums_1k(x: np.ndarray) -> np.ndarray:
    m = (x.size // COARSE) * COARSE
    return x[:m].reshape(-1, COARSE).sum(1, dtype=np.uint64).astype(np.uint32)


def scored(arrs: dict, masks: dict) -> tuple[np.ndarray, np.ndarray]:
    """(kept 25 bp bins, kept 1 kb sums) on CHROMS, as `knob_effects.load` builds them."""
    fine, coarse = [], []
    for c in CHROMS:
        x = arrs[c].astype(np.float64)
        bad = masks[c]
        fine.append(x[~bad])
        m = (x.size // COARSE) * COARSE
        coarse.append(x[:m].reshape(-1, COARSE).sum(1)[~bad[:m].reshape(-1, COARSE).any(1)])
    return np.concatenate(fine), np.concatenate(coarse)


def crps_fields(src: np.ndarray, tgt: np.ndarray) -> dict:
    sub = br.subset_index(tgt)
    m = np.maximum(src, br.PRED_FLOOR)
    out = {}
    for s in ("all", "top1"):
        idx = sub[s]
        sp = br.scale_split(m[idx], tgt[idx], br.poisson_crps)
        out[f"crps_{s}"] = sp["crps"]
        out[f"n_{s}"] = int(idx.size)
        if s == "all":
            out["crps_oracle_scaled_all"] = sp["crps_oracle_scaled"]
            out["scale_error_all"] = sp["scale_error"]
            out["c_star_all"] = sp["c_star"]
    return out


def product(pseudoreps_dir, products_dir, blacklist_path, out_dir, pid_index: int) -> int:
    pseudoreps_dir, products_dir, out_dir = Path(pseudoreps_dir), Path(products_dir), Path(out_dir)
    pid = sorted_pids(pseudoreps_dir)[pid_index]
    meta = next((r for r in read_tsv(products_dir / "MANIFEST.tsv") if r["pid"] == pid), None)
    if meta is None:
        print(f"{pid}: not in {products_dir}/MANIFEST.tsv", file=sys.stderr)
        return 2
    blacklist = read_blacklist(blacklist_path)

    halves = {}
    for h in ("pr1", "pr2"):
        with np.load(pseudoreps_dir / pid / h / "counts25.npz", allow_pickle=False) as z:
            halves[h] = {c: z[c] for c in z.files if c not in DROP}
    chroms = list(halves["pr1"])
    if chroms != list(halves["pr2"]):
        print(f"{pid}: pr1 chroms {chroms} != pr2 {list(halves['pr2'])}", file=sys.stderr)
        return 2
    with np.load(products_dir / pid / "counts25.npz", allow_pickle=False) as z:
        prod19 = z["chr19"].astype(np.int64)
    s19 = halves["pr1"]["chr19"].astype(np.int64) + halves["pr2"]["chr19"].astype(np.int64)
    if s19.shape != prod19.shape or not np.array_equal(s19, prod19):
        print(f"{pid}: pr1 + pr2 != product on chr19", file=sys.stderr)
        return 3
    del prod19, s19
    top = top_idx(load(products_dir / pid, "counts", blacklist)["fine"])
    masks = {c: bad_bins(blacklist.get(c), halves["pr1"][c].size) for c in CHROMS}

    rows, bins1k = [], {}
    for c in chroms:
        bins1k[f"a15_{c}"] = sums_1k(halves["pr1"][c])
        bins1k[f"b15_{c}"] = sums_1k(halves["pr2"][c])
    for depth_index, ratio in enumerate(RATIOS):
        for seed in ((0,) if ratio == 1.0 else SEEDS):
            rng = rng_for(seed, pid_index, depth_index)
            keep_all = ratio == BINS1K_RATIO and seed == BINS1K_SEED
            thinned = {}
            for h in ("pr1", "pr2"):
                thinned[h] = {}
                for c in chroms:              # genome-wide, npz order: the draw is reproducible
                    t = thin(halves[h][c], ratio, rng)
                    if c in CHROMS:
                        thinned[h][c] = t
                    if keep_all:
                        bins1k[f"{'a' if h == 'pr1' else 'b'}375_{c}"] = sums_1k(t)
            a, a1k = scored(thinned["pr1"], masks)
            b, b1k = scored(thinned["pr2"], masks)
            del thinned
            common = {
                "pid": pid, "track": meta["track"], "assay": meta["assay"],
                "mark_class": MARK_CLASS[meta["assay"]], "arm": meta["arm"],
                "level": meta["level"], "thin_ratio": ratio,
                "reads_per_half": int(READS_PER_HALF * ratio), "seed": seed,
                "shape_1k": float(1 - spearmanr(a1k, b1k).statistic),
                "divergence_bin": float(np.mean(np.abs(np.log2((a + PSEUDO) / (b + PSEUDO))))),
                "spearman_fine": float(spearmanr(a, b).statistic), "n_coarse": int(a1k.size)}
            with np.errstate(divide="ignore", invalid="ignore"):
                lfc = float(np.log2(b[top].mean() / a[top].mean()))   # pr2 over pr1
            for direction in DIRECTIONS:
                src, tgt = (a, b) if direction == "pr1_to_pr2" else (b, a)
                sgn = 1.0 if direction == "pr1_to_pr2" else -1.0
                row = {**common, "direction": direction, "log2fc_top1": sgn * lfc + 0.0,
                       "abs_log2fc_top1": abs(lfc), **crps_fields(src, tgt)}
                rows.append({k: row[k] for k in COLUMNS})
            print(f"{pid} ratio={ratio} seed={seed} shape={common['shape_1k']:.4f} "
                  f"div={common['divergence_bin']:.4f} crps={rows[-2]['crps_all']:.4f}", flush=True)

    (out_dir / "pairs").mkdir(parents=True, exist_ok=True)
    (out_dir / "bins1k").mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "pairs" / f"{pid}.tsv.tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(out_dir / "pairs" / f"{pid}.tsv")
    tmpz = out_dir / "bins1k" / f"{pid}.tmp.npz"
    np.savez_compressed(tmpz, **bins1k)
    tmpz.replace(out_dir / "bins1k" / f"{pid}.npz")
    (out_dir / "pairs" / f"{pid}.DONE").write_text(f"{len(rows)} rows\n")
    print(f"done {pid}: {len(rows)} rows, {len(bins1k)} 1 kb arrays", flush=True)
    return 0


# ---------------------------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------------------------


def _f(v) -> float:
    return float(v) if v not in ("", "None", "nan", None) else float("nan")


def fit_curve(values: dict) -> dict:
    """values {ratio: metric}; least squares of log(metric) on log2(reads_per_half)."""
    x = np.array([math.log2(READS_PER_HALF * r) for r in RATIOS])
    y = np.array([values[r] for r in RATIOS], dtype=np.float64)
    nan = float("nan")
    if not (np.isfinite(y).all() and (y > 0).all()):
        return {"alpha": nan, "beta": nan, "r2": nan, "value_30M_extrap": nan}
    ly = np.log(y)
    beta, alpha = np.polyfit(x, ly, 1)
    resid = ly - (alpha + beta * x)
    ss = float(((ly - ly.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss if ss > 0 else nan
    return {"alpha": float(alpha), "beta": float(beta), "r2": r2,
            "value_30M_extrap": float(math.exp(alpha + beta * math.log2(TARGET_READS)))}


def _oracle_json(rungs_dir: Path, pid: str):
    for p in (rungs_dir / f"{pid}.json", rungs_dir / "oracle" / f"{pid}.json"):
        if p.is_file():
            return json.loads(p.read_text())
    return None


def _oracle_rec(doc, space: str, direction: str):
    for r in (doc or {}).get("records", []):
        if (r.get("space") == space and r.get("direction") == direction
                and r.get("rung", "oracle") == "oracle" and r.get("eval", "score") == "score"):
            return r
    return None


def _crps(rec, subset: str):
    v = rec.get(f"spread_crps_{subset}", rec.get(f"crps_{subset}"))
    return float("nan") if v is None else float(v)


def _git_sha() -> str:
    if os.environ.get("GIT_SHA"):
        return os.environ["GIT_SHA"]
    try:
        return subprocess.run(["git", "-C", str(HERE.parent), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _write(path: Path, rows: list[dict], cols) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cols), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def aggregate(out_dir, rungs_dir, expect: int = 130) -> int:
    out_dir, rungs_dir = Path(out_dir), Path(rungs_dir)
    done = sorted(p.name[:-len(".DONE")] for p in (out_dir / "pairs").glob("*.DONE"))
    if len(done) != expect:
        print(f"{len(done)} .DONE under {out_dir}/pairs, need {expect}", file=sys.stderr)
        return 2
    rows = []
    for pid in done:
        rows += read_tsv(out_dir / "pairs" / f"{pid}.tsv")
    rows.sort(key=lambda r: (r["pid"], -float(r["thin_ratio"]), int(r["seed"]), r["direction"]))
    _write(out_dir / "floor_pairs.tsv", rows, COLUMNS)

    # per product and metric: direction-mean per (ratio, seed), then seed-mean per ratio
    curve = []
    for pid in done:
        pr = [r for r in rows if r["pid"] == pid]
        mclass = pr[0]["mark_class"]
        for m in METRICS_CURVE:
            by = {}
            for r in pr:
                by.setdefault((float(r["thin_ratio"]), int(r["seed"])), []).append(_f(r[m]))
            dm = {k: float(np.mean(v)) for k, v in by.items()}
            val = {ra: float(np.mean([v for (r2, _), v in dm.items() if r2 == ra])) for ra in RATIOS}
            row = {"pid": pid, "mark_class": mclass, "metric": m, **fit_curve(val)}
            for ra in RATIOS:
                row[f"value_{DEPTH_LABEL[ra]}"] = val[ra]
            for ra in RATIOS[1:]:
                row[f"seed_spread_{DEPTH_LABEL[ra]}"] = abs(dm.get((ra, 0), float("nan"))
                                                            - dm.get((ra, 1), float("nan")))
            curve.append(row)
    curve_cols = (["pid", "mark_class", "metric", "alpha", "beta", "r2"]
                  + [f"value_{DEPTH_LABEL[r]}" for r in RATIOS] + ["value_30M_extrap"]
                  + [f"seed_spread_{DEPTH_LABEL[r]}" for r in RATIOS[1:]])
    _write(out_dir / "floor_curve.tsv", curve, curve_cols)

    # per mark class (and `all`) x metric: medians over the class's products
    at30 = []
    for mclass in sorted({r["mark_class"] for r in curve}) + ["all"]:
        for m in METRICS_CURVE:
            cr = [r for r in curve if r["metric"] == m and mclass in ("all", r["mark_class"])]
            if not cr:
                continue
            ext = np.array([r["value_30M_extrap"] for r in cr], dtype=np.float64)
            row = {"mark_class": mclass, "metric": m, "n_products": len(cr),
                   "value_30M_extrap": float(np.nanmedian(ext)) if np.isfinite(ext).any() else "nan",
                   "value_30M_extrap_min": float(np.nanmin(ext)) if np.isfinite(ext).any() else "nan",
                   "value_30M_extrap_max": float(np.nanmax(ext)) if np.isfinite(ext).any() else "nan"}
            for ra in RATIOS:
                row[f"value_{DEPTH_LABEL[ra]}"] = float(np.median([r[f"value_{DEPTH_LABEL[ra]}"]
                                                                   for r in cr]))
            for ra in RATIOS[1:]:
                k = f"seed_spread_{DEPTH_LABEL[ra]}"
                row[k] = float(np.median([r[k] for r in cr]))
            row["beta"] = float(np.nanmedian([r["beta"] for r in cr]))
            at30.append(row)
    at30_cols = (["mark_class", "metric", "n_products", "value_30M_extrap", "value_30M_extrap_min",
                  "value_30M_extrap_max"] + [f"value_{DEPTH_LABEL[r]}" for r in RATIOS]
                 + [f"seed_spread_{DEPTH_LABEL[r]}" for r in RATIOS[1:]] + ["beta"])
    _write(out_dir / "floor_at_30M.tsv", at30, at30_cols)

    # p space: read, not recomputed (no thinning in -log10 p without MACS2 reruns)
    p15, check = [], []
    mclass_of = {r["pid"]: r["mark_class"] for r in rows}
    for pid in done:
        doc = _oracle_json(rungs_dir, pid)
        for d in DIRECTIONS:
            rp = _oracle_rec(doc, "pval", d)
            p15.append({"pid": pid, "mark_class": mclass_of[pid], "direction": d,
                        "reads_per_half": READS_PER_HALF,
                        "crps_all": _crps(rp, "all") if rp else "nan",
                        "crps_top1": _crps(rp, "top1") if rp else "nan",
                        "crps_oracle_scaled_all": (rp or {}).get("crps_oracle_scaled_all", "nan"),
                        "scale_error_all": (rp or {}).get("scale_error_all", "nan")})
            rc = _oracle_rec(doc, "counts", d)
            mine = next(r for r in rows if r["pid"] == pid and r["direction"] == d
                        and float(r["thin_ratio"]) == 1.0)
            if rc is None:
                check.append(f"FAIL {pid} {d} no counts oracle record in {rungs_dir}")
                continue
            d1 = abs(_f(mine["crps_all"]) - _crps(rc, "all"))
            d2 = abs(_f(mine["crps_oracle_scaled_all"]) - float(rc["crps_oracle_scaled_all"]))
            ok = d1 <= 1e-6 and d2 <= 1e-6
            check.append(f"{'PASS' if ok else 'FAIL'} {pid} {d} "
                         f"crps_all_absdiff={d1:.3g} crps_oracle_scaled_all_absdiff={d2:.3g}")
    _write(out_dir / "floor_p15M.tsv", p15, ["pid", "mark_class", "direction", "reads_per_half",
                                             "crps_all", "crps_top1", "crps_oracle_scaled_all",
                                             "scale_error_all"])
    n_want = expect * 7 * len(DIRECTIONS)
    check.append(f"{'PASS' if len(rows) == n_want else 'FAIL'} rows {len(rows)}")
    (out_dir / "floor_check.txt").write_text("\n".join(check) + "\n")
    (out_dir / "config.json").write_text(json.dumps(
        {"ratios": RATIOS, "seeds": SEEDS, "reads_per_half": READS_PER_HALF,
         "target_reads": TARGET_READS, "drop": DROP, "chroms": CHROMS, "res": RES,
         "coarse_bins": COARSE, "bins1k": {"ratio": BINS1K_RATIO, "seed": BINS1K_SEED},
         "metrics_curve": METRICS_CURVE, "formulas": FORMULAS, "rungs_v2": str(rungs_dir),
         "n_products": len(done), "git_sha": _git_sha()}, indent=1))
    n_fail = sum(line.startswith("FAIL") for line in check)
    print(f"aggregate: {len(rows)} rows, {len(curve)} curve rows, {n_fail} FAIL", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("product", help="score one product's halves over the depth ladder")
    for name in ("pseudoreps_dir", "products_dir", "blacklist", "out_dir"):
        p.add_argument(name)
    p.add_argument("pid_index", type=int)
    g = sub.add_parser("aggregate", help="concatenate, fit the depth curve, check vs rungs_v2")
    g.add_argument("out_dir")
    g.add_argument("rungs_dir")
    g.add_argument("--expect", type=int, default=130, help="number of products (.DONE files)")
    a = ap.parse_args(argv)
    if a.cmd == "product":
        return product(a.pseudoreps_dir, a.products_dir, a.blacklist, a.out_dir, a.pid_index)
    return aggregate(a.out_dir, a.rungs_dir, a.expect)


if __name__ == "__main__":
    sys.exit(main())
