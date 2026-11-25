#!/usr/bin/env python3
"""Where does a validation track's scale error come from? (crux task t390, diagnosis only)

For one validation track, on a chromosome subset (default chr1, chr9, chr21; 25 bp, blacklist
deleted, the scorer's grid), the exact error split of `tools/eic_decomp/errdecomp.calibrate`
(MSE / Var(truth) = offset + scale + nonlinear + residual) of these tracks against the truth:

  pred        the scored track: predicted mean `mu` through the mean-mode conversion (must equal
              <out>/raw/<track>/<track>.npy; checked)
  observed    the experiment's own observed store counts through the same conversion: the
              conversion's floor, i.e. what a perfect count prediction would score
  pred_depth  pred with `mu` scaled by k = sum(observed) / sum(mu) on the subset (oracle depth;
              diagnosis only, it reads the observed counts)
  pred_lamobs DNase-seq only: predicted pileup scored against the lambda of the observed counts
              (ChIP-seq lambda comes from the observed control and the depth covariate, so it is
              the same for pred and observed)

plus pred against observed (in units of Var(observed)), the count-space split of `mu` against the
observed counts, and where the squared error sits (top 1 % of truth bins vs the rest).
-> <out>/diag/<track>.json. Reads only files already on disk; writes nothing else.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "eic_decomp"))
import score_entry as S  # noqa: E402
import errdecomp as ED  # noqa: E402

LP, E = S.LP, S.E
from onion_eval import signal_from_counts as SFC  # noqa: E402


def signal(mu, lam_mu, route, ctl, c):
    """Mean-mode signal of `mu`, with lambda computed from `lam_mu` (DNase) or the control (ChIP)."""
    mu = np.asarray(mu, np.float64)
    rl = float(c["read_length"])
    if route == "dnase":
        d = float(SFC.ATAC_SMOOTH_WIN)
        lam = SFC.macs2_lambda(lambda_bg=d * float(c["depth_reads"]) / SFC.MACS2_GSIZE_HS,
                               treat_pileup=SFC.pileup_from_counts(lam_mu, d, rl))
    else:
        d = float(c["frag_len"])
        lam = SFC.macs2_lambda(lambda_bg=d * float(c["depth_reads"]) / SFC.MACS2_GSIZE_HS,
                               control_pileup=SFC.pileup_from_counts(SFC.box_spread(ctl, d), d,
                                                                     float(c["control_read_length"])),
                               ratio_treat2control=float(c["depth_reads"]) / float(c["control_depth_reads"]))
    return SFC._score_pileup(SFC.box_spread(mu, d), None, d / (rl + SFC.SPAN_EXTRA), lam, "plug_in", "box_d")


def terms(x, y):
    t, _ = ED.calibrate(x, y)
    return {k: (float(v) if not isinstance(v, (bool, np.bool_)) else bool(v)) for k, v in t.items()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--pred", required=True, help="predict.py folder (counts/<track>.h5)")
    p.add_argument("--cov", required=True)
    p.add_argument("--track", required=True)
    p.add_argument("--out", required=True, help="the scoring run's folder (raw/<track>/<track>.npy)")
    p.add_argument("--chroms", default="chr1,chr9,chr21")
    p.add_argument("--eps", type=float, default=1e-6)
    a = p.parse_args(argv)
    import h5py

    cov = S.read_cov(a.cov, a.track)
    route, chroms = cov["route"], a.chroms.split(",")
    sizes = LP.chrom_sizes(S.SIZES)
    bl = E.load_blacklist_bins(E.asset_dir())
    truth = E.load_prediction(f"{S.VAL_TRUTH}/{a.track}.bigwig", chroms, bl)
    scored = E.load_prediction(str(Path(a.out) / "raw" / a.track / f"{a.track}.npy"), chroms, bl)
    raw = {k: {} for k in ("pred", "observed", "pred_depth", "pred_lamobs")}
    mus, obss = {}, {}
    with h5py.File(Path(a.pred) / "counts" / f"{a.track}.h5", "r") as f, h5py.File(cov["observed_counts"], "r") as g:
        for ch in chroms:
            mus[ch] = f["mu"][ch][:].astype(np.float64) + a.eps
            o = g[ch][:].astype(np.float64)
            obss[ch] = np.zeros_like(mus[ch])
            obss[ch][: min(len(o), len(mus[ch]))] = o[: len(mus[ch])]
    k = sum(o.sum() for o in obss.values()) / sum(m.sum() for m in mus.values())
    for ch in chroms:
        ctl = S._control(cov["control"], ch)[0] if route == "chip" else None
        mu, ob = mus[ch], obss[ch] + a.eps
        tr = {"pred": signal(mu, mu, route, ctl, cov), "observed": signal(ob, ob, route, ctl, cov),
              "pred_depth": signal(mu * k, mu * k, route, ctl, cov)}
        if route == "dnase":
            tr["pred_lamobs"] = signal(mu, ob, route, ctl, cov)
        for name, sig in tr.items():
            y = np.zeros(LP.ngrid(sizes[ch]), np.float32)
            y[: sig.shape[0]] = sig[: y.shape[0]]
            raw[name][ch] = E.apply_blacklist(y, bl.get(ch))
    cat = lambda d: np.concatenate([d[c] for c in chroms]).astype(np.float64)  # noqa: E731
    X = {}
    for name in raw:
        if raw[name]:
            P, T = E.align(raw[name], truth, chroms)
            X[name] = cat(P)
    P, T = E.align(scored, truth, chroms)
    x, ysc = cat(T), cat(P)
    rep = dict(track=a.track, route=route, assay_class=cov["assay_class"], chroms=chroms, n_bins=int(x.size),
               depth_reads=float(cov["depth_reads"]), k_depth=float(k),
               sum_mu=float(sum(m.sum() for m in mus.values())), sum_obs=float(sum(o.sum() for o in obss.values())),
               pred_matches_scored=float(np.max(np.abs(X["pred"] - ysc))),
               mean_truth=float(x.mean()), var_truth=float(x.var()))
    rep["vs_truth"] = {name: terms(x, y) for name, y in X.items()}
    rep["pred_vs_observed"] = terms(X["observed"], X["pred"])
    mu_all, ob_all = cat(mus), cat(obss)
    rep["counts_mu_vs_observed"] = terms(ob_all, mu_all)
    top = x >= np.quantile(x, 0.99)
    for name in ("pred", "observed"):
        e = X[name] - x
        rep[f"{name}_top1pct"] = dict(mse_share=float((e[top] ** 2).sum() / (e ** 2).sum()),
                                      mean_err_top=float(e[top].mean()), mean_err_rest=float(e[~top].mean()),
                                      mean_pred_top=float(X[name][top].mean()), mean_truth_top=float(x[top].mean()))
    dst = Path(a.out) / "diag" / f"{a.track}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(rep, indent=1))
    LP.log(f"diag {a.track}: k {k:.3f}, nmse pred {rep['vs_truth']['pred']['mse']:.3f}, "
           f"observed {rep['vs_truth']['observed']['mse']:.3f}, pred_depth {rep['vs_truth']['pred_depth']['mse']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
