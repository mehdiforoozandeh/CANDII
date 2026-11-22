"""t371: compare the deviance metrics with the current headline terms on the same predictions.

    python tools/onion_eval/deviance_compare.py --runs <runs_rcp> --arms "<run> ..." \
        [--seeds "0 1"] [--sub p16_dev] [--ref-sub p16_sv] --out <dir>
    -> <dir>/per_seed.tsv, per_arm.tsv, rank_agreement.tsv, seed_stability.tsv, kit_check.tsv

Kit check: in <sub>/scores.json every key except `deviance` and the volatile ones must equal
<ref-sub>/scores.json (same model, same eval set, same options; only the switch differs).

Old terms come from `select_rule.py table` (imputation_raw, correction_v3_raw = the oracle-floor
gap on the 18 no-duplicate pairs, correction_oracle_all, uncertainty_raw) and from scores.json
(count CRPS skill, r_spec). New terms come from scores.json["deviance"].

Agreement: Spearman correlation of arm means between each old and new term of the same goal.
Seed stability: for each metric, the share of its variance that lies between arms,
var(arm means) / (var(arm means) + mean within-arm variance / n_seeds) -- higher means the
metric separates arms better than seeds separate runs.
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

TOOLS = Path(__file__).resolve().parents[1]
VOLATILE = {"wall_seconds", "timing_seconds", "torch_threads", "pred_store", "deviance",
            "scorer_settings", "refs"}

GOALS = {
    "imputation": {"old": ["imputation_raw", "count_crps_skill", "r_spec"],
                   "new": ["dev_imputation_skill", "dev_imputation_r2"]},
    "correction": {"old": ["correction_v3_raw", "correction_oracle_all", "correction_crps_model_neg"],
                   "new": ["dev_correction_gap_closed", "dev_correction_model_neg"]},
    "calibration": {"old": ["uncertainty_raw"],
                    "new": ["dev_calibration_error_neg", "nb_logscore_gain_vs_marginal"]},
}


def _macro(blk, key):
    v = blk.get(key, {})
    v = v.get("macro") if isinstance(v, dict) else None
    return float(v) if v is not None else np.nan


def _table(runs, prefix, sub):
    out = subprocess.run([sys.executable, str(TOOLS / "onion_eval/select_rule.py"), "table",
                          "--runs", runs, "--prefix", prefix, "--sub", sub],
                         capture_output=True, text=True, check=True).stdout
    return pd.read_csv(io.StringIO(out), sep="\t")


def _strip(d):
    return {k: v for k, v in d.items() if k not in VOLATILE}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--seeds", default="0 1")
    ap.add_argument("--sub", default="p16_dev")
    ap.add_argument("--ref-sub", default="p16_sv")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = Path(a.runs)
    rows, kit = [], []
    for arm in a.arms.split():
        tab = _table(a.runs, arm, a.sub)
        tab = tab[tab.arm == arm]
        for s in a.seeds.split():
            f = runs / arm / f"seed{s}" / a.sub / "scores.json"
            if not f.is_file():
                continue
            d = json.loads(f.read_text())
            ref = runs / arm / f"seed{s}" / a.ref_sub / "scores.json"
            if ref.is_file():
                r = json.loads(ref.read_text())
                diff = sorted(k for k in set(_strip(d)) | set(_strip(r))
                              if _strip(d).get(k) != _strip(r).get(k))
                kit.append({"arm": arm, "seed": int(s), "equal": not diff,
                            "differing_keys": ",".join(diff)})
            t = tab[tab.seed == int(s)]
            dv = d.get("deviance", {})
            row = {"arm": arm, "seed": int(s),
                   "imputation_raw": float(t.imputation_raw.iloc[0]) if len(t) else np.nan,
                   "correction_v3_raw": float(t.correction_gap_closed_oracle_no_nodedup_side.iloc[0])
                   if len(t) else np.nan,
                   "correction_oracle_all": float(t.correction_gap_closed_oracle_all.iloc[0])
                   if len(t) else np.nan,
                   "uncertainty_raw": float(t.uncertainty_raw.iloc[0]) if len(t) else np.nan,
                   "count_crps_skill": _macro(d, "count_crps_skill_vs_position_average"),
                   "r_spec": _macro(d.get("r_spec", {}), "counts_log1p"),
                   "correction_crps_model_neg": -_macro(d, "correction_crps_model")}
            for k in ("dev_imputation_skill", "dev_imputation_r2", "dev_imputation_model",
                      "dev_imputation_position_average", "dev_imputation_floor",
                      "dev_imputation_scale_error", "dev_imputation_monotone_error",
                      "dev_correction_gap_closed", "dev_correction_gap_closed_vs_pseudoreplicate",
                      "dev_correction_model", "dev_correction_depth_only", "dev_correction_floor",
                      "dev_correction_pseudoreplicate", "dev_calibration_ratio",
                      "dev_calibration_error", "nb_logscore_gain_vs_marginal"):
                row[k] = _macro(dv, k)
            row["dev_calibration_error_neg"] = -row["dev_calibration_error"]
            row["dev_correction_model_neg"] = -row["dev_correction_model"]
            rows.append(row)
    ps = pd.DataFrame(rows)
    ps.to_csv(out / "per_seed.tsv", sep="\t", index=False, float_format="%.5f")
    pd.DataFrame(kit).to_csv(out / "kit_check.tsv", sep="\t", index=False)
    num = [c for c in ps.columns if c not in ("arm", "seed")]
    pa = ps.groupby("arm")[num].agg(["mean", lambda x: x.max() - x.min()])
    pa.columns = [f"{c}.{'mean' if s == 'mean' else 'range'}" for c, s in pa.columns]
    pa.to_csv(out / "per_arm.tsv", sep="\t", float_format="%.5f")
    means = ps.groupby("arm")[num].mean()
    agree = []
    for g, d in GOALS.items():
        for o in d["old"]:
            for n in d["new"]:
                ok = means[[o, n]].dropna()
                rho, p = stats.spearmanr(ok[o], ok[n]) if len(ok) > 2 else (np.nan, np.nan)
                agree.append({"goal": g, "old": o, "new": n, "n_arms": len(ok),
                              "spearman": rho, "p": p})
    pd.DataFrame(agree).to_csv(out / "rank_agreement.tsv", sep="\t", index=False, float_format="%.4f")
    stab = []
    for c in num:
        g = ps.groupby("arm")[c]
        m, v, n = g.mean(), g.var(ddof=1), g.count()
        if (n >= 2).sum() < 3:
            continue
        between = m.var(ddof=1)
        within = float((v / n).mean())
        stab.append({"metric": c, "between_arm_var": between, "within_var_of_mean": within,
                     "between_share": between / (between + within) if between + within > 0 else np.nan})
    pd.DataFrame(stab).to_csv(out / "seed_stability.tsv", sep="\t", index=False, float_format="%.5g")
    print(pd.DataFrame(agree).round(3).to_string())
    print(pd.DataFrame(stab).round(3).to_string())
    print("kit check:", sum(k["equal"] for k in kit), "of", len(kit), "equal")
    return 0


if __name__ == "__main__":
    sys.exit(main())
