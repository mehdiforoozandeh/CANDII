"""t392 investigation: tables for smooth_skill.py outputs, and the calibration ratio after n -> n / c.

    python smooth_summary.py skill --records <records.jsonl ...> --resid <resid.jsonl ...>
        --scores <per_assay_scores.json> --out <dir>
    python smooth_summary.py calib --pred-store <seed store> --resid <resid.jsonl ...> --out <jsonl>
        [--kit DIR] --nb-table <npy>

skill: per assay and W, the deviance imputation skill `deviance.gap(recs, "D_ref")` (score.py's
per-assay value), next to the scorer's seed-0 per-assay values of both models.
calib: per leave-one-assay-out target of the stored seed predictions (all 256 regions), D0,
D_model, E_nb and E_nb with n / c, c = the measured `c_model` of the target when measured, else the
mean over the measured targets of its assay, else the mean over all measured targets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

EVALSET = "/project/def-maxwl/mforooz/ONION/eval/eic_v1_2/evalset_p16.json"
TABLE = "/project/def-maxwl/mforooz/ONION/tables/eic_v1_2/measurements.tsv"
SCORABLE = ("H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me2", "H3K4me3", "H3K79me2",
            "H3K9ac", "H3K9me3")


def _read(paths):
    return [json.loads(l) for p in paths for l in open(p) if l.strip()]


def _deviance(kit=None):
    root = Path(kit) / "tools" if kit else Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "onion_eval"))
    import deviance as dev
    return dev


def cmd_skill(a) -> int:
    import pandas as pd
    dev = _deviance()
    recs = _read(a.records)
    res = _read(a.resid)
    sc = json.loads(Path(a.scores).read_text())
    rows = []
    for assay in sorted({r["assay"] for r in recs}):
        row = {"assay": assay,
               "scorer_rs_seed0_256": sc["rs0"]["dev_imputation_skill"].get(assay),
               "scorer_ref_seed0_256": sc["rcp0"]["dev_imputation_skill"].get(assay),
               "scorer_ref_mean3_256": float(np.mean([sc[f"rcp{k}"]["dev_imputation_skill"][assay]
                                                      for k in range(3)]))}
        st = [r["dev"] for r in recs if r["assay"] == assay and r.get("source") == "store"]
        row["n_targets"] = len(st)
        row["repro_store_256"] = dev.gap(st, "D_ref") if st else None
        for W in (1, 9, 41):
            d = [r["dev"] for r in recs if r["assay"] == assay and r.get("source") != "store"
                 and r["W"] == W]
            row[f"skill_W{W}_16"] = dev.gap(d, "D_ref") if d else None
            row[f"calratio_W{W}_16"] = dev.calibration_ratio(d) if d else None
        rz = [r for r in res if r["assay"] == assay]
        row["lag_below_0.5"] = [r["lag_below_0.5"] for r in rz]
        row["lag_below_0.1"] = [r["lag_below_0.1"] for r in rz]
        row["c_model"] = [round(r["c_model"], 2) for r in rz]
        row["c_linkformula"] = [round(r["c"], 2) for r in rz]
        row["acf_od_lag1_10_100_400"] = [[round(r["acf_od"][k], 3) for k in (1, 10, 100, 400)] for r in rz]
        row["sum_mu_over_sum_y"] = [round(r["sum_mu"] / r["sum_y"], 3) for r in rz]
        row["res2_over_mu_plus_od"] = [round(r["sum_res2"] / (r["sum_mu"] + r["sum_mu2n"]), 3) for r in rz]
        rows.append(row)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).set_index("assay")
    df.to_csv(out / "table_skill.tsv", sep="\t", float_format="%.4f")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(df.T.to_string())
    return 0


def cmd_calib(a) -> int:
    import h5py
    import pandas as pd
    dev = _deviance(a.kit)
    T = np.load(a.nb_table)
    dev._nb_table = lambda: T
    res = _read(a.resid)
    c_t = {r["target"]: r["c_model"] for r in res}
    by_a = {}
    for r in res:
        by_a.setdefault(r["assay"], []).append(r["c_model"])
    c_all = float(np.mean(list(c_t.values())))
    ev = json.loads(Path(EVALSET).read_text())
    regs = [(str(c), int(s)) for c, s in ev["regions"]]
    rb = int(ev["region_bins"])
    tab = pd.read_csv(TABLE, sep="\t", dtype=str).set_index("measurement_id")
    with open(a.out, "a") as fo:
        for j in ev["imputation_targets"]:
            if j["regime"] != "leave_one_assay_out":
                continue
            mid, assay = j["target"], j["assay"]
            key = f"item/{mid}/leave_one_assay_out"
            z = np.load(Path(a.pred_store) / (hashlib.sha1(key.encode()).hexdigest() + ".npz"))
            mu = z["mu"].astype(np.float64).ravel()
            n = z["n"].astype(np.float64).ravel()
            cp = tab.loc[mid, "counts_path"]
            cp = cp[3:] if cp.startswith("h5:") else cp
            with h5py.File(cp, "r") as f:
                y = np.concatenate([np.asarray(f[c][s:s + rb], np.float64) for c, s in regs])
            if mid in c_t:
                c, src = c_t[mid], "measured"
            elif assay in by_a:
                c, src = float(np.mean(by_a[assay])), "assay mean"
            else:
                c, src = c_all, "all-target mean"
            r = {"target": mid, "assay": assay, "c": c, "c_source": src,
                 "D0": dev.deviance(y, np.full_like(y, max(y.mean(), dev.MU_MIN))),
                 "D_model": dev.deviance(y, mu),
                 "E_nb": float(dev.nb_expected_deviance(n, mu).sum()),
                 "E_nb_c": float(dev.nb_expected_deviance(n / c, mu).sum())}
            fo.write(json.dumps(r) + "\n")
            fo.flush()
            print(mid, assay, src, round(c, 3), flush=True)
    return 0


def cmd_calib_table(a) -> int:
    recs = _read(a.calib)

    def macro(key, assays):
        per = {}
        for s in assays:
            rs = [r for r in recs if r["assay"] == s]
            if rs:
                per[s] = np.mean([r["D_model"] / r["D0"] for r in rs]) / np.mean([r[key] / r["D0"] for r in rs])
        return per
    out = {"n_targets": len(recs)}
    for key in ("E_nb", "E_nb_c"):
        per = macro(key, SCORABLE)
        out[f"macro9_{key}"] = float(np.mean(list(per.values())))
        out[f"per_assay_{key}"] = {k: round(float(v), 4) for k, v in per.items()}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("skill")
    s.add_argument("--records", nargs="+", required=True)
    s.add_argument("--resid", nargs="+", required=True)
    s.add_argument("--scores", required=True)
    s.add_argument("--out", required=True)
    c = sp.add_parser("calib")
    c.add_argument("--pred-store", required=True)
    c.add_argument("--resid", nargs="+", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--nb-table", required=True)
    c.add_argument("--kit", default="/project/def-maxwl/mforooz/onion_kit/RS7")
    t = sp.add_parser("calib-table")
    t.add_argument("--calib", nargs="+", required=True)
    t.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"skill": cmd_skill, "calib": cmd_calib, "calib-table": cmd_calib_table}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
