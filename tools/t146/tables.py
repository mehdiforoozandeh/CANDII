"""t146 — per-arm CRPS tables from the scored runs (every CRPS with its oracle_scaled / scale_error).

    python tools/t146/tables.py <runs_dir> <out_dir>

Reads <runs_dir>/*/scores.json. **Pooling** (t118 / t128 readings): per (arm, weight, route, group,
knob, seed) the mean over a track's pairs, then the mean over tracks; the reading = the mean over
seeds; seed_spread = max - min over seeds of the per-seed pooled crps_all (the noise rule's spread).

**Groups.** trained / heldout / shuffle (the record kinds); law pairs: same (arm_src == arm_tgt, a
level change within one knob), cross (two knobs), never (both). knob: trained = the non-base arm;
same = the knob; cross = once under each of its two knobs; "all" = every pair once. ceiling rows
(h30) = each scored law pair (a, b) given its target b's ceiling record, grouped as the law pair.

Writes:
  records.tsv.gz  every record, flat (the input to any further reading)
  pooled.tsv      arm x weight x route x group x knob: crps_all, crps_oracle_scaled_all,
                  scale_error_all, crps_dec0 / dectop (the top non-empty decile) with their splits, seed_spread, per-seed crps
  cells.tsv       (h30) law pairs by (source knob, target knob) cell x route, pooled the same way
  deciles.tsv     (h31) per arm x seed x track x kind x direction x knob x level: crps per decile
  depthlaw.tsv    median |log2 scale error| per arm x weight x route x depth group, seed spread
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["crps_all", "crps_oracle_scaled_all", "scale_error_all",
           "crps_dec0", "crps_oracle_scaled_dec0", "scale_error_dec0",
           "crps_dectop", "crps_oracle_scaled_dectop", "scale_error_dectop"]
PAIR = ["source_pid", "target_pid"]


def load(runs_dir) -> pd.DataFrame:
    parts = []
    for f in sorted(Path(runs_dir).glob("*/scores.json")):
        s = json.loads(f.read_text())
        r = pd.DataFrame(s["records"])
        for k in ("run_name", "arm", "weight", "seed"):
            r[k] = s["run"][k]
        parts.append(r)
    if not parts:
        raise SystemExit(f"no scores.json under {runs_dir}")
    d = pd.concat(parts, ignore_index=True)
    d["weight"] = d["weight"].fillna(-1)
    for m in METRICS:
        if m not in d:
            d[m] = np.nan
        d[m] = pd.to_numeric(d[m], errors="coerce")
    return d


def label(d: pd.DataFrame) -> pd.DataFrame:
    law = d[d.kind == "law"].copy()
    ceil = d[d.kind == "ceiling"][["run_name", "target_pid"] + METRICS]
    if len(ceil):
        c = law[law.route == "direct"].drop(columns=METRICS).merge(
            ceil, on=["run_name", "target_pid"], how="inner")
        c["route"] = "ceiling"
        law = pd.concat([law, c], ignore_index=True)
    law["group"] = np.where(law.arm_src == law.arm_tgt, "same", "cross")
    a = law.assign(knob=law.arm_src)
    b = law[law.group == "cross"].assign(knob=lambda x: x.arm_tgt)
    once = law.assign(knob="all")
    lw = pd.concat([a, b, once], ignore_index=True)
    lw = pd.concat([lw, lw.drop_duplicates(["run_name", "route", "knob"] + PAIR)
                    .assign(group="never")], ignore_index=True)
    t = d[d.kind.isin(["trained", "heldout", "shuffle"])].copy()
    t["group"] = t.kind
    t["knob"] = np.where(t.direction == "base_to_arm", t.arm_tgt, t.arm_src)
    t = pd.concat([t, t.assign(knob="all")], ignore_index=True)
    return pd.concat([t, lw], ignore_index=True)


def pooled(p: pd.DataFrame, keys: list) -> pd.DataFrame:
    tr = p.groupby(keys + ["seed", "track"])[METRICS].mean()
    ps = tr.groupby(keys + ["seed"])[METRICS].mean().reset_index()
    n = p.drop_duplicates(keys + PAIR).groupby(keys).size().rename("n_pairs")
    out = ps.groupby(keys)[METRICS].mean()
    g = ps.groupby(keys)["crps_all"]
    out["seed_spread"] = g.max() - g.min()
    out["n_seeds"] = g.size()
    wide = ps.pivot_table(index=keys, columns="seed", values="crps_all")
    wide.columns = [f"crps_s{int(c)}" for c in wide.columns]
    return out.join(n).join(wide).reset_index()


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    runs_dir, out = Path(argv[0]), Path(argv[1])
    out.mkdir(parents=True, exist_ok=True)
    d = load(runs_dir)
    d.to_csv(out / "records.tsv.gz", sep="\t", index=False)
    p = label(d)
    pooled(p, ["arm", "weight", "route", "group", "knob"]).to_csv(
        out / "pooled.tsv", sep="\t", index=False, float_format="%.5f")
    law = p[p.group.isin(["same", "cross"]) & (p.knob != "all")].drop_duplicates(
        ["run_name", "route"] + PAIR)
    if len(law):
        pooled(law, ["arm", "weight", "route", "arm_src", "arm_tgt"]).to_csv(
            out / "cells.tsv", sep="\t", index=False, float_format="%.5f")
    dec = [c for c in d.columns if c.startswith("crps_dec") or c.startswith("n_dec")]
    if dec:
        keys = ["arm", "weight", "seed", "track", "kind", "route", "direction", "arm_src",
                "level_src", "arm_tgt", "level_tgt"]
        d[d.kind.isin(["trained", "heldout", "law"])].groupby(keys)[
            ["crps_all"] + sorted(set(dec) | set(METRICS[3:]))].mean().reset_index().to_csv(
            out / "deciles.tsv", sep="\t", index=False, float_format="%.5f")
    if "depth_log2_scale_pred" in d:
        z = d[d.depth_log2_scale_pred.notna()].copy()
        z["abs_err"] = (pd.to_numeric(z.depth_log2_scale_pred)
                        - pd.to_numeric(z.depth_log2_ratio_true)).abs()
        z["dgroup"] = np.where(z.kind == "law", "never", z.kind)
        ps = z.groupby(["arm", "weight", "route", "dgroup", "seed"]).abs_err.median().reset_index()
        g = ps.groupby(["arm", "weight", "route", "dgroup"]).abs_err
        pd.DataFrame({"median_abs_err": g.median(), "seed_spread": g.max() - g.min(),
                      "n_seeds": g.size()}).reset_index().to_csv(
            out / "depthlaw.tsv", sep="\t", index=False, float_format="%.5f")
    print(f"{d.run_name.nunique()} runs, {len(d)} records -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
