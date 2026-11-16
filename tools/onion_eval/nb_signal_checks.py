"""E2 (close-out): two checks of the route from counts to signal p-value and peaks.

    python tools/onion_eval/nb_signal_checks.py formula --evalset <json> --out <dir>
        [--fraglen-table <tsv>] [--dnase-route macs2_no_control|hotspot2_density] [--targets a,b]
    -> <out>/formula_check.tsv (one row per validation imputation target) and formula_check.json

    python tools/onion_eval/nb_signal_checks.py lambda --evalset <json> --dump [name=]<regime dir>
        [--dump ...] --out <dir> [--fraglen-table <tsv>]
    -> <out>/lambda_check.tsv (one row per dump and ChIP-seq target) and lambda_check.json

    python tools/onion_eval/nb_signal_checks.py table --run <name>=<scores.json> [--run ...] --out <dir>
    -> <out>/signal_space_macro.tsv, signal_space_per_target.tsv, signal_space_reasons.json

`formula`: the formula of `signal_from_counts` applied to the OBSERVED counts of each target, against
the pipeline's own p-value track (`signal.h5`, the truth of the signal space). Observed counts in,
pipeline p-value out: a disagreement here is an error of the formula (binned read-span counts in
place of base-pair fragment pileups, region-edge truncation of the local windows, fragment length),
never an error of a model. The formula is run in `plug_in` mode: on observed counts the NB has no
spread, and the expectation mode equals the plug-in in the limit of an infinite NB size. A zero
count is passed as 1e-9 (the formula needs mu > 0); floor(1e-9 * scale) = 0, so the score is the
zero count's score. Columns: Pearson of the raw -log10 p and of its asinh, Spearman (Max's 2,000,000
bin subsample), the median and 99th percentile of |formula - pipeline| in -log10 p, the mean
difference, the least-squares slope of the pipeline on the formula, and the peak agreement at
-log10 p >= 2 (`ag2_peaks.derived_peaks`), per 25 bp bin and per 500 bp tile of the AG2 scorer
(precision and recall of the formula's calls against the pipeline's, F1, Jaccard). The rows, the
fragment length and the lambda source come from the scorer's own code (`Scorer._sig_row`,
`Scorer._signal_files`, `signal_from_counts.signal_from_nb`), so the check runs the scorer's route.

`lambda`: the MACS2 local lambda of each ChIP-seq target from the model's predicted control
(`<mid>.ctl_pred.npy`, dumped by `score.py --dump-signal ... --lambda-source predicted_control`)
against the lambda from the cell's observed control, both by the rule of `signal_track` (the
predicted control is asked for at the treatment's C', so its depth ratio is 1; the observed control
has r = N_treat / N_control). Per target: the median of lambda_pred / lambda_obs, the fraction of
bins within 10% (|ratio - 1| <= 0.1), the Pearson of log lambda, and the fraction of bins where each
lambda sits at its genome background floor lambda_bg. In regime `accessibility_only` the model sees
no control (the control is hidden); in `leave_one_assay_out` the control is one of its inputs.

`table`: headline 1 (specific variance recovered above noise) and r_spec per run, in count space
(`counts_log1p`) beside the signal space (`signal_arcsinh`, its plug-in, and the predicted-control
variant), macro per regime, and per target with the lambda source of each signal variant.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from candi.store.pval_from_counts import MACS2_GSIZE_HS  # noqa: E402
from onion_eval import ag2_peaks as PK  # noqa: E402
from onion_eval import signal_from_counts as SG  # noqa: E402
from onion_eval.ag2_score import corr, spearman  # noqa: E402

ZERO_COUNT = 1e-9
WITHIN = 0.10

FORMULA_COLS = ["target", "cell", "assay", "assay_class", "route", "lambda_source", "fraglen_source",
                "frag_len", "n_bins", "frac_zero_counts", "pearson_raw", "pearson_asinh", "spearman",
                "median_abs_diff", "p99_abs_diff", "mean_diff", "slope", "bin_precision",
                "bin_recall", "bin_f1", "bin_jaccard", "tile_precision", "tile_recall", "tile_f1",
                "tile_jaccard", "reason"]
LAMBDA_COLS = ["target", "cell", "assay", "regime", "n_bins", "median_ratio", "frac_within_10pct",
               "pearson_log_lambda", "frac_bg_obs", "frac_bg_pred", "within_10pct_median", "reason"]


def _write_tsv(path: Path, cols, rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            w.writerow([("" if r.get(c) is None else
                         (f"{r[c]:.6g}" if isinstance(r.get(c), float) else r[c])) for c in cols])


def _prf(called, truth) -> dict:
    called, truth = np.asarray(called, bool), np.asarray(truth, bool)
    tp = float(np.sum(called & truth))
    nc, nt, nu = float(called.sum()), float(truth.sum()), float(np.sum(called | truth))
    p = tp / nc if nc else float("nan")
    r = tp / nt if nt else float("nan")
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else float("nan")
    return {"precision": p, "recall": r, "f1": f1, "jaccard": tp / nu if nu else float("nan")}


def agreement(formula, pipeline, rb: int) -> dict:
    """Agreement of a formula -log10 p track with the pipeline's, both [n_regions * rb]."""
    x, y = np.asarray(formula, np.float64), np.asarray(pipeline, np.float64)
    d = x - y
    out = {"n_bins": int(x.size), "pearson_raw": corr(x, y),
           "pearson_asinh": corr(np.arcsinh(x), np.arcsinh(y)), "spearman": spearman(x, y),
           "median_abs_diff": float(np.median(np.abs(d))),
           "p99_abs_diff": float(np.quantile(np.abs(d), 0.99)), "mean_diff": float(d.mean())}
    vx = float(np.var(x))
    out["slope"] = float(np.cov(x, y, bias=True)[0, 1] / vx) if vx > 0 else float("nan")
    bx, by = PK.derived_peaks(x), PK.derived_peaks(y)
    out.update({f"bin_{k}": v for k, v in _prf(bx, by).items()})
    tx, ty = PK.tile_any(bx, region_bins=rb), PK.tile_any(by, region_bins=rb)
    out.update({f"tile_{k}": v for k, v in _prf(tx, ty).items()})
    return out


def chip_lambda(ctl_counts, *, depth, frag_len, ctl_depth, ctl_read_length,
                gsize: float = MACS2_GSIZE_HS) -> np.ndarray:
    """The ChIP-seq lambda of `signal_track` for one stretch (treatment pileup units)."""
    d = float(frag_len)
    return SG.macs2_lambda(lambda_bg=d * float(depth) / gsize,
                           control_pileup=SG.pileup_from_counts(ctl_counts, d, float(ctl_read_length)),
                           ratio_treat2control=float(depth) / float(ctl_depth))


def lambda_compare(lam_pred, lam_obs, lam_bg_pred, lam_bg_obs) -> dict:
    a, b = np.asarray(lam_pred, np.float64), np.asarray(lam_obs, np.float64)
    ratio = a / b
    med = float(np.median(ratio))
    return {"n_bins": int(a.size), "median_ratio": med,
            "frac_within_10pct": float(np.mean(np.abs(ratio - 1.0) <= WITHIN)),
            "pearson_log_lambda": corr(np.log(a), np.log(b)),
            "frac_bg_obs": float(np.mean(b <= lam_bg_obs * (1 + 1e-12))),
            "frac_bg_pred": float(np.mean(a <= lam_bg_pred * (1 + 1e-12))),
            "within_10pct_median": int(abs(med - 1.0) <= WITHIN)}


def _scorer(evalset, fraglen_table, dnase_route):
    from onion_eval.score import Scorer
    return Scorer(evalset, main="marginal_nb", fraglen_table=fraglen_table, dnase_route=dnase_route,
                  log=lambda *a, **k: None)


def _targets(sc, only):
    seen, out = set(), []
    for t in sc.ev["imputation_targets"]:
        if t.get("split", "val") != sc.split or t["target"] in seen:
            continue
        if only and t["target"] not in only:
            continue
        seen.add(t["target"])
        out.append(t)
    return out


def cmd_formula(a) -> int:
    import h5py
    sc = _scorer(a.evalset, a.fraglen_table, a.dnase_route)
    rb, regions = sc.rb, sc.regions
    rows = []
    for t in _targets(sc, set(a.targets.split(",")) if a.targets else None):
        mid = t["target"]
        row, fl_src, _ = sc._sig_row(sc._row(mid))
        rec = {"target": mid, "cell": t["cell"], "assay": t["assay"],
               "assay_class": row.get("assay_class"), "fraglen_source": fl_src,
               "frag_len": row.get("frag_len")}
        try:
            rec["route"] = SG.route_of(row["assay_class"] or row["assay"])
        except ValueError as e:
            rows.append(dict(rec, reason=str(e)))
            continue
        files, why = sc._signal_files(row)
        if files is None:
            rows.append(dict(rec, reason=why))
            continue
        y = np.concatenate([sc.ds.read(mid, c, s, rb).astype(np.float64) for c, s in regions])
        with h5py.File(files["signal"], "r") as f:
            truth = np.concatenate([np.asarray(f[c][s:s + rb], np.float64) for c, s in regions])
        mu = np.where(y > 0, y, ZERO_COUNT)
        try:
            sig, src = SG.signal_from_nb(np.ones_like(mu), mu, row, ds=sc.ds, regions=regions, rb=rb,
                                         dnase_route=sc.dnase_route, mode="plug_in")
        except (ValueError, KeyError) as e:
            rows.append(dict(rec, reason=f"no formula signal: {e}"))
            continue
        rec.update(lambda_source=src, frac_zero_counts=float(np.mean(y <= 0)), reason="")
        rec.update(agreement(sig, truth, rb))
        rows.append(rec)
        print(f"formula {mid} {t['assay']}: pearson {rec['pearson_raw']:.4f} "
              f"tile_f1 {rec['tile_f1']:.4f}", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    _write_tsv(out / "formula_check.tsv", FORMULA_COLS, rows)
    ok = [r for r in rows if not r.get("reason")]
    summ = {"evalset": str(sc.evalset_path), "fraglen_table": a.fraglen_table,
            "dnase_route": sc.dnase_route, "n_targets": len(rows), "n_checked": len(ok),
            "by_route": {}}
    for route in sorted({r["route"] for r in ok}):
        rr = [r for r in ok if r["route"] == route]
        summ["by_route"][route] = {
            "n": len(rr),
            **{f"median_{k}": float(np.nanmedian([r[k] for r in rr]))
               for k in ("pearson_raw", "pearson_asinh", "spearman", "median_abs_diff",
                         "p99_abs_diff", "slope", "bin_f1", "tile_f1", "tile_precision",
                         "tile_recall")},
            "min_pearson_raw": float(np.nanmin([r["pearson_raw"] for r in rr]))}
    (out / "formula_check.json").write_text(json.dumps(summ, indent=1, sort_keys=True) + "\n")
    print(json.dumps(summ, indent=1, sort_keys=True))
    return 0


def cmd_lambda(a) -> int:
    sc = _scorer(a.evalset, a.fraglen_table, None)
    rows, summ = [], []
    for name, d in (x.split("=", 1) if "=" in x else (x, x) for x in a.dump):
        r, s = _lambda_one(sc, Path(d))
        rows += [dict(x, dump=name) for x in r]
        summ.append(dict(s, dump=name))
        print(json.dumps(summ[-1], sort_keys=True), flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    _write_tsv(out / "lambda_check.tsv", ["dump"] + LAMBDA_COLS, rows)
    (out / "lambda_check.json").write_text(json.dumps(summ, indent=1, sort_keys=True) + "\n")
    return 0


def _lambda_one(sc, d: Path):
    rb, regions = sc.rb, sc.regions
    meta = json.loads((d / "tracks.json").read_text(encoding="utf-8"))
    regime = meta["regime"]
    rows = []
    for mid in meta["targets"]:
        row, _, _ = sc._sig_row(sc._row(mid))
        rec = {"target": mid, "cell": row.get("cell"), "assay": row.get("assay"), "regime": regime}
        try:
            if SG.route_of(row["assay_class"] or row["assay"]) != "chip":
                continue
        except ValueError:
            continue
        p = d / f"{mid}.ctl_pred.npy"
        ctl_id = (row.get("control_measurement_id") or "").strip()
        if not p.is_file():
            rows.append(dict(rec, reason="no ctl_pred.npy"))
            continue
        if not ctl_id or ctl_id not in sc.ds.rows:
            rows.append(dict(rec, reason="no observed control"))
            continue
        crow = sc.ds.rows[ctl_id]
        if not (row.get("frag_len") or "").strip():
            rows.append(dict(rec, reason="no frag_len"))
            continue
        pred = np.load(p).astype(np.float64)
        lp, lo = [], []
        for i, (c, s) in enumerate(regions):
            sl = slice(i * rb, (i + 1) * rb)
            obs = sc.ds.read(ctl_id, c, s, rb).astype(np.float64)
            lo.append(chip_lambda(obs, depth=row["depth_reads"], frag_len=row["frag_len"],
                                  ctl_depth=crow["depth_reads"],
                                  ctl_read_length=crow["read_length"]))
            lp.append(chip_lambda(pred[sl], depth=row["depth_reads"], frag_len=row["frag_len"],
                                  ctl_depth=row["depth_reads"], ctl_read_length=row["read_length"]))
        bg = float(row["frag_len"]) * float(row["depth_reads"]) / MACS2_GSIZE_HS
        rec.update(lambda_compare(np.concatenate(lp), np.concatenate(lo), bg, bg), reason="")
        rows.append(rec)
    ok = [r for r in rows if not r.get("reason")]
    summ = {"path": str(d.resolve()), "method": meta.get("method"), "regime": regime,
            "n_chip_targets": len(rows), "n_compared": len(ok),
            "n_cells_compared": len({r["cell"] for r in ok}),
            "n_within_10pct_median": int(sum(r["within_10pct_median"] for r in ok)),
            "median_of_median_ratio": float(np.median([r["median_ratio"] for r in ok])) if ok else None,
            "median_frac_within_10pct": float(np.median([r["frac_within_10pct"] for r in ok])) if ok else None}
    return rows, summ


#: (scores.json block, space key) pairs of the signal-space table; headline 1 and r_spec
TABLE_KEYS = [(blk, sp) for blk in ("specific_variance_recovered_above_noise", "r_spec")
              for sp in ("counts_log1p", "signal_arcsinh", "signal_arcsinh_plug_in",
                         "signal_arcsinh_predicted_control")]


def _macro(scores: dict, regime: str, blk: str, sp: str):
    src = scores if regime == "leave_one_assay_out" else scores.get(regime, {})
    v = src.get(blk, {}).get(sp)
    if not isinstance(v, dict):
        return None, "absent"
    return v.get("macro"), (v.get("reason") or "")


def cmd_table(a) -> int:
    """Signal-space metric 1 beside count space, per run (`--run name=<scores.json>`, repeated):
    the macro of headline 1 and r_spec in each space and regime; and one row per (run, target) with
    the headline-1 `recovered` value per space and the target's lambda source."""
    runs, per = [], []
    for spec in a.run:
        name, path = spec.split("=", 1)
        s = json.loads(Path(path).read_text(encoding="utf-8"))
        for regime in ("leave_one_assay_out", "accessibility_only"):
            row = {"run": name, "regime": regime, "scores": path}
            for blk, sp in TABLE_KEYS:
                m, why = _macro(s, regime, blk, sp)
                row[f"{blk}.{sp}"] = m
                if m is None:
                    row[f"{blk}.{sp}.reason"] = why[:200]
            runs.append(row)
            src = s if regime == "leave_one_assay_out" else s.get(regime, {})
            h = src.get("specific_variance_recovered_above_noise", {})
            for mid, rec in (h.get("counts_log1p", {}).get("per_target") or {}).items():
                r = {"run": name, "regime": regime, "target": mid, "assay": rec.get("assay"),
                     "cell": rec.get("cell"), "counts_log1p": rec.get("recovered")}
                for sp in ("signal_arcsinh", "signal_arcsinh_plug_in",
                           "signal_arcsinh_predicted_control"):
                    t = (h.get(sp, {}).get("per_target") or {}).get(mid, {})
                    r[sp] = t.get("recovered")
                    r[f"lambda_source.{sp}"] = t.get("lambda_source")
                per.append(r)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cols = ["run", "regime"] + [f"{b}.{s}" for b, s in TABLE_KEYS] + ["scores"]
    _write_tsv(out / "signal_space_macro.tsv", cols, runs)
    pcols = ["run", "regime", "target", "assay", "cell", "counts_log1p", "signal_arcsinh",
             "signal_arcsinh_plug_in", "signal_arcsinh_predicted_control",
             "lambda_source.signal_arcsinh", "lambda_source.signal_arcsinh_plug_in",
             "lambda_source.signal_arcsinh_predicted_control"]
    _write_tsv(out / "signal_space_per_target.tsv", pcols, per)
    (out / "signal_space_reasons.json").write_text(json.dumps(
        [{k: v for k, v in r.items() if k.endswith(".reason") or k in ("run", "regime")} for r in runs],
        indent=1) + "\n")
    print(f"{len(runs)} macro rows, {len(per)} per-target rows -> {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("formula")
    f.add_argument("--evalset", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--fraglen-table", default=None)
    f.add_argument("--dnase-route", default=None, choices=SG.DNASE_ROUTES)
    f.add_argument("--targets", default=None, help="comma list of target ids (default: all)")
    lm = sub.add_parser("lambda")
    lm.add_argument("--evalset", required=True)
    lm.add_argument("--dump", required=True, action="append",
                    help="[name=]<regime dir of a --dump-signal output>; repeat for several")
    lm.add_argument("--out", required=True)
    lm.add_argument("--fraglen-table", default=None)
    tb = sub.add_parser("table")
    tb.add_argument("--run", required=True, action="append", help="name=<scores.json>; repeat")
    tb.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"formula": cmd_formula, "lambda": cmd_lambda, "table": cmd_table}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
