"""Simulation checks of the deviance metrics (task t371): do they behave as they should?

    python tools/onion_eval/deviance_sim.py --out <dir> [--bins 200000] [--reps 5]
    -> <dir>/sim_depth.tsv, sim_floor.tsv, sim_calibration.tsv, sim_correction.tsv, SIM.md

A known rate track lam (background plus peaks, about one read per 25 bp bin at depth 1) is the
truth. Reads are drawn at 1/4 to 4 times that depth, and predictions with one planted mistake each
are scored. The design follows sprint 10 of Max's research repo (`notes/counts.md`, section 4.1).

Pre-registered checks (written before the first run; never loosened after):
  C1 floor      the Poisson plug-in floor (isotonic rate) is within 0.02 of the true floor, in
                normalized units (floor / D0), at every depth
  C2 depth      the floor-corrected error (D - F) / (1 - F) of every mistake moves by at most 0.03
                or 20% (whichever is larger) from 1/4 to 4 times depth
  C3 ranking    the true rate scores best; within each mistake family, a larger mistake scores worse
  C4 calibration with NB reads, R is within 0.03 of 1 when the predictive size equals the true
                size, above 1 when it is 4 times too large (overconfident), below 1 when 4 times
                too small (underconfident)
  C5 correction depth-only scaling scores a gap of exactly 0; the true rate scores within 0.03 of 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from onion_eval import deviance as DV  # noqa: E402

DEPTHS = (0.25, 0.5, 1.0, 2.0, 4.0)


def truth_rate(bins: int, rng) -> np.ndarray:
    """A rate track with mean 1: a smooth log-normal background plus sparse peaks."""
    z = rng.normal(size=bins)
    k = np.exp(-0.5 * (np.arange(-60, 61) / 20.0) ** 2)
    bg = np.exp(0.8 * np.convolve(z, k / np.sqrt((k ** 2).sum()), mode="same"))
    peaks = np.zeros(bins)
    centres = rng.choice(bins, size=bins // 400, replace=False)
    peaks[centres] = rng.gamma(2.0, 15.0, size=centres.size)
    kp = np.exp(-0.5 * (np.arange(-12, 13) / 4.0) ** 2)
    lam = 0.3 * bg + np.convolve(peaks, kp / kp.sum(), mode="same")
    return lam / lam.mean()


def _blur(v, w):
    k = np.ones(w) / w
    return np.convolve(v, k, mode="same")


def mistakes(lam, rng):
    """{family: [(label, severity, prediction)]}; each prediction keeps the true total."""
    tot = lam.sum()
    norm = lambda v: v * tot / v.sum()
    peak = lam > np.quantile(lam, 0.99)
    out = {
        "truth": [("truth", 0, lam)],
        "scale": [(f"x{s}", i, lam * s) for i, s in enumerate((1.25, 1.5, 2.0, 4.0), 1)],
        "scale_down": [(f"x{s}", i, lam * s) for i, s in enumerate((0.8, 0.67, 0.5, 0.25), 1)],
        "enrichment_low": [(f"pow{p}", i, norm(lam ** p)) for i, p in enumerate((0.9, 0.7, 0.5), 1)],
        "enrichment_high": [(f"pow{p}", i, norm(lam ** p)) for i, p in enumerate((1.1, 1.4, 1.8), 1)],
        "blur": [(f"blur{w}", i, norm(_blur(lam, w))) for i, w in enumerate((8, 40, 200), 1)],
        "noise": [(f"noise{s}", i, norm(lam * np.exp(s * rng.normal(size=lam.size) - s * s / 2)))
                  for i, s in enumerate((0.25, 0.5, 1.0), 1)],
    }
    drop = peak & (rng.random(lam.size) < 0.5)
    v = lam.copy()
    v[drop] = np.median(lam)
    out["peaks_removed"] = [("half_peaks", 1, norm(v))]
    return out


def _norm(y, mu):
    t = DV.target_terms(y, mu)
    return t["D_model"] / t["D0"], t["floor"] / t["D0"], t


def sim_depth(bins, reps, seed=0):
    """Rows: rep, depth, family, label, severity, D (normalized), plug-in floor, true floor,
    floor-corrected error, and log1p squared error / Var for contrast."""
    rows = []
    for rep in range(reps):
        rng = np.random.default_rng(seed + rep)
        lam = truth_rate(bins, rng)
        preds = mistakes(lam, rng)
        for d in DEPTHS:
            y = rng.poisson(lam * d).astype(np.float64)
            D0 = DV.deviance(y, np.full_like(y, y.mean()))
            true_floor = float(DV.poisson_expected_deviance(lam * d).sum()) / D0
            ly = np.log1p(y)
            for fam, items in preds.items():
                for label, sev, p in items:
                    mu = p * d
                    Dn, Fn, _ = _norm(y, mu)
                    # log1p unit for contrast: E[log1p(Y)] under Poisson(mu) is not used here;
                    # plain log1p(mu) shows how a transform unit drifts with depth
                    mse = float(np.mean((ly - np.log1p(mu)) ** 2) / np.var(ly))
                    rows.append({"rep": rep, "depth": d, "family": fam, "label": label,
                                 "severity": sev, "D": Dn, "floor_plugin": Fn,
                                 "floor_true": true_floor,
                                 "corrected": (Dn - true_floor) / (1 - true_floor),
                                 "log1p_mse_over_var": mse})
    return rows


def sim_calibration(bins, reps, seed=100):
    rows = []
    for rep in range(reps):
        rng = np.random.default_rng(seed + rep)
        lam = truth_rate(bins, rng)
        for n_true in (0.5, 2.0, 10.0):
            for d in (0.5, 2.0):
                mu = lam * d
                y = rng.negative_binomial(n_true, n_true / (n_true + mu)).astype(np.float64)
                for label, f in (("true_size", 1.0), ("overconfident_x4", 4.0),
                                 ("underconfident_div4", 0.25)):
                    n = np.full_like(mu, n_true * f)
                    t = DV.target_terms(y, mu, n)
                    rows.append({"rep": rep, "n_true": n_true, "depth": d, "case": label,
                                 "R": t["D_model"] / t["E_nb"]})
    return rows


def sim_correction(bins, reps, seed=200):
    """A pair: source pr1 at depth d_s, target pr2 at depth d_t. Depth-only scaling predicts
    pr1 * d_t / d_s; the true rate predicts lam * d_t; a smoothed source is a partial correction."""
    rows = []
    for rep in range(reps):
        rng = np.random.default_rng(seed + rep)
        lam = truth_rate(bins, rng)
        for ds, dt in ((0.25, 1.0), (0.5, 2.0), (1.0, 4.0), (1.0, 1.0)):
            x = rng.poisson(lam * ds).astype(np.float64)
            y = rng.poisson(lam * dt).astype(np.float64)
            dep = np.maximum(x * dt / ds, DV.MU_MIN)
            sm = np.maximum(_blur(x, 9) * dt / ds, DV.MU_MIN)
            for label, mu in (("depth_only", dep), ("smoothed_source", sm), ("truth", lam * dt)):
                t = DV.pair_terms(y, mu, dep)
                rows.append({"rep": rep, "source_depth": ds, "target_depth": dt, "predictor": label,
                             "gap": DV.gap([t], "D_depth")})
    return rows


def checks(depth_rows, cal_rows, cor_rows) -> dict:
    import pandas as pd
    dr, cr, kr = pd.DataFrame(depth_rows), pd.DataFrame(cal_rows), pd.DataFrame(cor_rows)
    res = {}
    fl = dr.groupby(["rep", "depth"])[["floor_plugin", "floor_true"]].first()
    # C1 uses the plug-in floor of the true rate's own isotonic fit, i.e. the truth row
    tr = dr[dr.family == "truth"]
    res["C1_floor_max_abs_error"] = float((tr.floor_plugin - tr.floor_true).abs().max())
    res["C1_pass"] = res["C1_floor_max_abs_error"] <= 0.02
    g = dr[dr.family != "truth"].groupby(["family", "label", "depth"]).corrected.mean().unstack()
    span = g.max(axis=1) - g.min(axis=1)
    tol = np.maximum(0.03, 0.2 * g.abs().mean(axis=1))
    res["C2_worst_drift"] = float(span.max())
    res["C2_worst_label"] = str(span.idxmax())
    res["C2_fail_labels"] = [str(i) for i in span.index[span > tol]]
    res["C2_pass"] = not res["C2_fail_labels"]
    # C2 paired (added after the first small run; a smoke check, not the pre-registered C2): the
    # floor is the realized deviance of the true rate on the same draw, which removes the sampling
    # noise every mistake shares at one depth
    tru = dr[dr.family == "truth"].set_index(["rep", "depth"]).D
    F = tru.reindex(pd.MultiIndex.from_frame(dr[["rep", "depth"]])).values
    dp = dr.assign(paired=(dr.D.values - F) / (1 - F))
    gp = dp[dp.family != "truth"].groupby(["family", "label", "depth"]).paired.mean().unstack()
    sp = gp.max(axis=1) - gp.min(axis=1)
    tp = np.maximum(0.03, 0.2 * gp.abs().mean(axis=1))
    res["C2_paired_worst_drift"] = float(sp.max())
    res["C2_paired_fail_labels"] = [str(i) for i in sp.index[sp > tp]]
    res["C2_paired_pass"] = not res["C2_paired_fail_labels"]
    gl = dr[dr.family != "truth"].groupby(["family", "label", "depth"]).log1p_mse_over_var.mean().unstack()
    res["contrast_log1p_worst_drift"] = float((gl.max(axis=1) - gl.min(axis=1)).max())
    m = dr.groupby(["depth", "family", "severity"]).D.mean()
    ok = True
    for d in DEPTHS:
        best = m.loc[d, "truth", 0]
        ok &= bool((m.loc[d].drop("truth", level=0) > best).all())
        for fam in dr.family.unique():
            if fam == "truth":
                continue
            s = m.loc[d, fam].sort_index()
            ok &= bool((np.diff(s.values) > 0).all())
    res["C3_pass"] = ok
    c = cr.groupby("case").R.agg(["min", "max", "mean"])
    res["C4_true_size_max_abs"] = float((cr[cr.case == "true_size"].R - 1).abs().max())
    res["C4_pass"] = (res["C4_true_size_max_abs"] <= 0.03
                      and bool((cr[cr.case == "overconfident_x4"].R > 1).all())
                      and bool((cr[cr.case == "underconfident_div4"].R < 1).all()))
    res["C4_summary"] = {k: {kk: round(float(vv), 3) for kk, vv in v.items()} for k, v in c.iterrows()}
    dep = kr[kr.predictor == "depth_only"].gap
    tru = kr[kr.predictor == "truth"].gap
    res["C5_depth_only_max_abs"] = float(dep.abs().max())
    res["C5_truth_min"] = float(tru.min())
    res["C5_truth_max"] = float(tru.max())
    res["C5_pass"] = res["C5_depth_only_max_abs"] < 1e-12 and float((tru - 1).abs().max()) <= 0.03
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--bins", type=int, default=200000)
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args(argv)
    import json
    import pandas as pd
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dr, cr, kr = sim_depth(a.bins, a.reps), sim_calibration(a.bins, a.reps), sim_correction(a.bins, a.reps)
    for name, rows in (("sim_depth", dr), ("sim_calibration", cr), ("sim_correction", kr)):
        pd.DataFrame(rows).to_csv(out / f"{name}.tsv", sep="\t", index=False, float_format="%.6g")
    res = checks(dr, cr, kr)
    (out / "checks.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))
    return 0 if all(v for k, v in res.items() if k.endswith("_pass")) else 1


if __name__ == "__main__":
    sys.exit(main())
