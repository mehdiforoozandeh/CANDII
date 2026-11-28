"""Aggregate <out>/diag/*.json into DIAG_TABLE.tsv and DIAG_BY_ASSAY.tsv (median per assay)."""
import csv, json, glob, statistics as st, sys
O = sys.argv[1]
meta = {r["cell_type_id"] + r["mark_id"]: r["mark/assay"] for r in csv.DictReader(
    open("/project/def-maxwl/mforooz/DATA_EIC_SYNAPSE/validation_data/metadata_validation_data.tsv"), delimiter="\t")}
rows = []
for p in sorted(glob.glob(f"{O}/diag/*.json")):
    d = json.load(open(p)); v = d["vs_truth"]
    r = dict(track=d["track"], assay=meta[d["track"]], route=d["route"],
             nmse_pred=v["pred"]["mse"], scale_pred=v["pred"]["scale"], offset_pred=v["pred"]["offset"],
             nonlin_pred=v["pred"]["nonlinear"], resid_pred=v["pred"]["residual"], r_pred=v["pred"]["r"],
             slope_pred=v["pred"]["slope_ratio"],
             nmse_obs=v["observed"]["mse"], scale_obs=v["observed"]["scale"], slope_obs=v["observed"]["slope_ratio"],
             k_depth=d["k_depth"], nmse_pred_depth=v["pred_depth"]["mse"], slope_pred_depth=v["pred_depth"]["slope_ratio"],
             nmse_pred_lamobs=v.get("pred_lamobs", {}).get("mse", float("nan")),
             slope_counts=d["counts_mu_vs_observed"]["slope_ratio"], r_counts=d["counts_mu_vs_observed"]["r"],
             slope_pred_vs_obs=d["pred_vs_observed"]["slope_ratio"],
             top1_share=d["pred_top1pct"]["mse_share"], top1_pred=d["pred_top1pct"]["mean_pred_top"],
             top1_obs=d["observed_top1pct"]["mean_pred_top"], top1_truth=d["pred_top1pct"]["mean_truth_top"],
             match=d["pred_matches_scored"])
    rows.append(r)
keys = list(rows[0])
with open(f"{O}/DIAG_TABLE.tsv", "w") as fh:
    w = csv.writer(fh, delimiter="\t"); w.writerow(keys)
    for r in rows: w.writerow([r[k] if isinstance(r[k], str) else f"{r[k]:.4g}" for k in keys])
num = [k for k in keys if k not in ("track", "assay", "route")]
by = {}
for r in rows: by.setdefault(r["assay"], []).append(r)
with open(f"{O}/DIAG_BY_ASSAY.tsv", "w") as fh:
    w = csv.writer(fh, delimiter="\t"); w.writerow(["assay", "n"] + num)
    for a, rs in sorted(by.items()):
        w.writerow([a, len(rs)] + [f"{st.median([x[k] for x in rs]):.4g}" for k in num])
print(open(f"{O}/DIAG_BY_ASSAY.tsv").read())
