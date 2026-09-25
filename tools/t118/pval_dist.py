"""Is -log10 p (MACS2 ppois, 25 bp bins) closer to Gaussian or log-normal?

Per base product, on chr1 (a training chromosome):
  1. marginal: zeros, quantiles, skewness of x and of log(x) on x > 0
  2. repeat noise: pr2 given pr1 (two independent halves of the same reads), in 40 quantile bins
     of pr1: slope b of log SD(pr2) on log mean(pr2)  (b ~ 0 additive/Gaussian, b ~ 1 multiplicative/log-normal)
  3. per bin, mean log-likelihood of pr2 under a fitted Normal vs a fitted log-normal (x > 0 only),
     plus the fraction of the Normal's mass below 0.
"""
import json, sys
import numpy as np

ROOT = "/project/def-maxwl/mforooz/t112_cf"
PIDS = ["C07M20__base__base", "C07M29__base__base", "C12M02__base__base", "C19M16__base__base",
        "C19M22__base__base", "C40M17__base__base", "C40M18__base__base"]
CHROM = "chr1"


def skew(v):
    v = v - v.mean()
    return float((v ** 3).mean() / (v ** 2).mean() ** 1.5)


def load(path):
    with np.load(path) as z:
        return z[CHROM].astype(np.float64)


out = {}
for pid in PIDS:
    x = load(f"{ROOT}/products/{pid}/pval25.npz")
    a = load(f"{ROOT}/pseudoreps/{pid}/pr1/pval25.npz")
    b = load(f"{ROOT}/pseudoreps/{pid}/pr2/pval25.npz")
    pos = x[x > 0]
    rec = {"n": int(x.size), "frac_zero": float((x == 0).mean()), "min_pos": float(pos.min()),
           "q": {k: float(np.quantile(x, q)) for k, q in
                 (("q10", .1), ("q50", .5), ("q90", .9), ("q99", .99), ("q999", .999))},
           "skew_x": skew(x), "skew_logx": skew(np.log(pos))}
    # repeat noise
    edges = np.unique(np.quantile(a, np.linspace(0, 1, 41)))
    idx = np.clip(np.searchsorted(edges, a, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for k in range(len(edges) - 1):
        y = b[idx == k]
        if y.size < 500:
            continue
        m, s = y.mean(), y.std()
        yp = y[y > 0]
        ll_n = float(np.mean(-0.5 * np.log(2 * np.pi * s ** 2) - (yp - m) ** 2 / (2 * s ** 2)))
        ly = np.log(yp)
        mu, sd = ly.mean(), ly.std()
        ll_ln = float(np.mean(-ly - 0.5 * np.log(2 * np.pi * sd ** 2) - (ly - mu) ** 2 / (2 * sd ** 2)))
        from math import erf, sqrt
        rows.append({"pr1_lo": float(edges[k]), "n": int(y.size), "mean": float(m), "sd": float(s),
                     "frac_pos": float(yp.size / y.size), "ll_normal": ll_n, "ll_lognormal": ll_ln,
                     "normal_mass_below0": 0.5 * (1 + erf((0 - m) / (s * sqrt(2))))})
    mm = np.array([r["mean"] for r in rows]); ss = np.array([r["sd"] for r in rows])
    ok = (mm > 0) & (ss > 0)
    rec["mv_slope"] = float(np.polyfit(np.log(mm[ok]), np.log(ss[ok]), 1)[0])
    rec["frac_bins_lognormal_wins"] = float(np.mean([r["ll_lognormal"] > r["ll_normal"] for r in rows]))
    rec["bins"] = rows
    out[pid] = rec
    print(pid, f"zero={rec['frac_zero']:.3f} skew x={rec['skew_x']:.2f} logx={rec['skew_logx']:.2f} "
          f"mv_slope={rec['mv_slope']:.2f} lognormal_wins={rec['frac_bins_lognormal_wins']:.2f}", flush=True)

json.dump(out, open(sys.argv[1], "w"), indent=1)
