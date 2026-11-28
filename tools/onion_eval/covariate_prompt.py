"""Covariate-prompt test (PI, 2026-11-25): does the model follow a changed output covariate?

For a few imputed targets in held-out cells, predict with ONE output covariate (the target's C'
row) changed at a time, the rest kept, and compare the predicted mean and dispersion with the
native prediction. Directions are stated before the run in EXPECTED and printed in the report.

    python tools/onion_eval/covariate_prompt.py --runs <seed0> [<seed1> ...] \
        --calibration-evalset <recal_study forward dir>/calibration_evalset.json \
        --n-targets 8 --out <dir>
    -> <out>/covariate_prompt.json, <out>/covariate_prompt.md

Targets: the imputation jobs of the calibration eval set (regime in ensemble_store.FIT_REGIMES),
one per assay, assays in sorted order (only --assays when given), at most --n-targets. Every region of the eval set is used.
Per target, arm and seed, over all bins: total_ratio = sum mu_arm / sum mu_native; shape_r =
Pearson r of log1p(mu_arm) vs log1p(mu_native); n_ratio = median(n_arm) / median(n_native);
var_ratio = sum(mu + mu^2/n)_arm / sum(mu + mu^2/n)_native. mu is the NB mean, n its size
(variance mu + mu^2/n).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: C' column index of each META column (tools/minicandi/contract.py META_COLUMNS order)
COL = {"log2_depth": 0, "run_type_pe": 1, "read_length": 2, "dedup_on": 3, "mapq_threshold": 4,
       "pipeline_id": 5, "pipeline_version_id": 6, "lab_id": 7, "platform_id": 8,
       "antibody_id": 9, "donor_id": 10}
#: categorical columns tested, and the measurements.tsv field each id comes from
CAT_FIELD = {"lab_id": "lab", "platform_id": "platform", "antibody_id": "antibody"}

#: stated before the run. total = total_ratio band; n = n_ratio direction ("down": < 1)
EXPECTED = {
    "depth_up":      "total_ratio in [1.6, 2.5] (2x depth -> ~2x counts); shape_r > 0.98",
    "depth_down":    "total_ratio in [0.4, 0.625] (0.5x depth); shape_r > 0.98",
    "run_type_flip": "SE->PE at the same read depth: total_ratio < 1 (half the fragments); "
                     "PE->SE: total_ratio > 1",
    "read_length":   "36->100 bp: total_ratio >= 1 (more reads map); 100->36: total_ratio <= 1",
    "dedup_flip":    "dedup on->off: total_ratio > 1 (duplicates kept); off->on: < 1",
    "mapq_flip":     "MAPQ 30->0: total_ratio > 1 (multi-mapped reads kept); 0->30: < 1",
    "lab_other":     "another seen lab: |total_ratio - 1| < 0.2 (no direction)",
    "lab_unknown":   "unknown lab: |total_ratio - 1| < 0.2; n_ratio < 1 (less confident)",
    "platform_other": "another seen platform: |total_ratio - 1| < 0.2 (no direction)",
    "platform_unknown": "unknown platform: |total_ratio - 1| < 0.2; n_ratio < 1",
    "antibody_other": "another seen antibody (ChIP only): |total_ratio - 1| < 0.2 (no direction)",
    "antibody_unknown": "unknown antibody (ChIP only): |total_ratio - 1| < 0.2; n_ratio < 1",
}


def check(arm: str, native: np.ndarray, s: dict) -> bool | None:
    """True when the summary `s` of `arm` meets its EXPECTED rule; None when not defined."""
    t, n = s["total_ratio"], s["n_ratio"]
    if arm == "depth_up":
        return 1.6 <= t <= 2.5 and s["shape_r"] > 0.98
    if arm == "depth_down":
        return 0.4 <= t <= 0.625 and s["shape_r"] > 0.98
    if arm == "run_type_flip":
        return t < 1 if native[COL["run_type_pe"]] == 0 else t > 1
    if arm == "read_length":
        return t >= 1 if native[COL["read_length"]] < 68 else t <= 1
    if arm == "dedup_flip":
        return t > 1 if native[COL["dedup_on"]] == 1 else t < 1
    if arm == "mapq_flip":
        return t > 1 if native[COL["mapq_threshold"]] > 0 else t < 1
    if arm.endswith("_other"):
        return abs(t - 1) < 0.2
    if arm.endswith("_unknown"):
        return abs(t - 1) < 0.2 and n < 1
    return None


def arms(native: np.ndarray, other_ids: dict) -> dict:
    """{arm: C' row} for one target's native C' row (raw values). `other_ids`: {col: id or None},
    the seen id each categorical `_other` arm uses (None: arm skipped). Native id 0 (unknown)
    skips that column's `_unknown` arm."""
    out = {}

    def put(name, col, val):
        v = native.copy()
        v[COL[col]] = val
        out[name] = v
    put("depth_up", "log2_depth", native[0] + 1.0)
    put("depth_down", "log2_depth", native[0] - 1.0)
    put("run_type_flip", "run_type_pe", 0.0 if native[COL["run_type_pe"]] >= 0.5 else 1.0)
    put("read_length", "read_length", 100.0 if native[COL["read_length"]] < 68 else 36.0)
    put("dedup_flip", "dedup_on", 0.0 if native[COL["dedup_on"]] >= 0.5 else 1.0)
    put("mapq_flip", "mapq_threshold", 0.0 if native[COL["mapq_threshold"]] > 0 else 30.0)
    for col, field in CAT_FIELD.items():
        name = col[:-3]
        if other_ids.get(col) is not None:
            put(f"{name}_other", col, float(other_ids[col]))
        if native[COL[col]] != 0:
            put(f"{name}_unknown", col, 0.0)
    return out


def other_id(rows: dict, encoder, target_row: dict, col: str):
    """The most frequent id of `col` among training rows of the target's assay that differs from
    the target's own id (ties: smallest id); None when there is none. Id 0 (unknown) never."""
    field = CAT_FIELD[col]
    own = encoder.cat_id(field, target_row.get(field))
    c = Counter()
    for r in rows.values():
        if r.get("in_eic") != "train" or r.get("assay") != target_row.get("assay"):
            continue
        i = encoder.cat_id(field, r.get(field))
        if i not in (0, own):
            c[i] += 1
    if not c:
        return None
    top = max(c.values())
    return min(i for i, k in c.items() if k == top)


def summarize(n0, mu0, n1, mu1) -> dict:
    def var(n, mu):
        return mu + mu * mu / n
    a, b = np.log1p(mu0), np.log1p(mu1)
    r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    return {"total_ratio": float(mu1.sum() / mu0.sum()), "shape_r": r,
            "n_ratio": float(np.median(n1) / np.median(n0)),
            "var_ratio": float(var(n1, mu1).sum() / var(n0, mu0).sum())}


def run(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--calibration-evalset", required=True)
    ap.add_argument("--n-targets", type=int, default=8)
    ap.add_argument("--assays", nargs="*", default=None,
                    help="restrict targets to these assays (one target each, sorted order)")
    ap.add_argument("--infer-batch", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import torch
    from onion_eval import score as S
    from onion_eval import ensemble_store as ES
    from minicandi.registry import load_run
    S.set_threads()
    S.tf32_off()
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    shas = {S.run_tables_sha256(r) for r in a.runs}
    if len(shas) != 1:
        raise SystemExit("refused: the runs were trained on different tables")
    models = [load_run(r) for r in a.runs]
    sc = S.Scorer(a.calibration_evalset, main="model", model=models[0][0], config=models[0][1],
                  device="cpu", run_tables_sha256=shas.pop(), log=print, n_boot=0)
    preds = [S.ModelPredictor(m, "cpu") for m, _ in models]
    imp = [j for j in sc.jobs()[0] if j["regime"] in ES.FIT_REGIMES]
    seen, jobs = set(), []
    for j in sorted(imp, key=lambda j: (j["assay"], j["target"])):
        if a.assays and j["assay"] not in a.assays:
            continue
        if j["assay"] not in seen:
            seen.add(j["assay"])
            jobs.append(j)
    jobs = jobs[:a.n_targets]
    rows, enc = sc.ds.rows, sc.ds.encoder
    results = []
    for ji, job in enumerate(jobs):
        row = rows[job["target"]]
        t = sc.C.TRACK_INDEX[row["assay"]]
        native = np.asarray(enc.c_out(row, 1.0), dtype=np.float64)
        oth = {c: other_id(rows, enc, row, c) for c in CAT_FIELD}
        if row.get("assay_class", "").startswith("accessibility"):
            oth["antibody_id"] = None
        arm_rows = arms(native, oth)
        if native[COL["antibody_id"]] == 0 or row.get("assay_class", "").startswith("access"):
            arm_rows.pop("antibody_unknown", None)
        names = ["native"] + list(arm_rows)
        acc = {(nm, k): ([], []) for nm in names for k in range(len(preds))}
        for ri, tiles in sc._tiles(sc.regions):
            sc._set_region(*sc.regions[ri])
            for b0 in range(0, len(tiles), a.infer_batch):
                tl = tiles[b0:b0 + a.infer_batch]
                batch = sc.C.collate([sc.ds.window(c, s, job["inputs"], [job["target"]])
                                      for c, s in tl])
                for nm in names:
                    b = dict(batch)
                    if nm != "native":
                        c = batch["c_out"].clone()
                        c[:, t, :] = torch.as_tensor(arm_rows[nm], dtype=c.dtype)
                        b["c_out"] = c
                    for k, pr in enumerate(preds):
                        p = pr(b)
                        acc[(nm, k)][0].append(p["n"][:, :, t].reshape(-1))
                        acc[(nm, k)][1].append(p["mu"][:, :, t].reshape(-1))
        cat = {key: (np.concatenate(v[0]), np.concatenate(v[1])) for key, v in acc.items()}
        rec = {"target": job["target"], "assay": row["assay"], "cell": job.get("cell") or row["cell"],
               "native_c_out": native.tolist(), "other_ids": oth, "arms": {}}
        for nm in names[1:]:
            per_seed = []
            for k in range(len(preds)):
                n0, mu0 = cat[("native", k)]
                n1, mu1 = cat[(nm, k)]
                s = summarize(n0, mu0, n1, mu1)
                s["meets_expected"] = check(nm, native, s)
                per_seed.append(s)
            rec["arms"][nm] = {"c_out": arm_rows[nm].tolist(), "per_seed": per_seed}
        results.append(rec)
        print(f"covariate_prompt: target {ji + 1}/{len(jobs)} {row['assay']} done "
              f"({time.time() - t0:.0f} s)", flush=True)
    doc = {"runs": [str(Path(r).resolve()) for r in a.runs],
           "calibration_evalset": str(Path(a.calibration_evalset).resolve()),
           "n_regions": len(sc.regions), "expected": EXPECTED, "targets": results,
           "seconds": round(time.time() - t0, 1)}
    (out / "covariate_prompt.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    (out / "covariate_prompt.md").write_text(report(doc), encoding="utf-8")
    print(f"covariate_prompt: wrote {out} ({time.time() - t0:.0f} s)")
    return 0


def report(doc: dict) -> str:
    L = ["# Covariate-prompt test", "",
         f"{len(doc['targets'])} imputed targets in held-out cells, {doc['n_regions']} regions, "
         f"{len(doc['runs'])} seeds. One output covariate changed per arm; the rest kept.", "",
         "Expected directions, stated before the run:", ""]
    L += [f"- {k}: {v}" for k, v in doc["expected"].items()]
    L += ["", "Per target and arm: total ratio, shape r, n ratio (median over bins), mean over "
          "seeds [min, max]; met = seeds meeting the stated rule.", "",
          "| assay | target | arm | total ratio | shape r | n ratio | met |", "|---|---|---|---|---|---|---|"]
    for t in doc["targets"]:
        for nm, a in t["arms"].items():
            ps = a["per_seed"]

            def f(key):
                v = [s[key] for s in ps]
                return f"{np.mean(v):.3f} [{min(v):.3f}, {max(v):.3f}]"
            met = sum(bool(s["meets_expected"]) for s in ps)
            L.append(f"| {t['assay']} | {t['target']} | {nm} | {f('total_ratio')} | "
                     f"{f('shape_r')} | {f('n_ratio')} | {met}/{len(ps)} |")
    L += ["", "Summary per arm: targets where every seed met the rule.", "",
          "| arm | met / targets |", "|---|---|"]
    names = sorted({nm for t in doc["targets"] for nm in t["arms"]})
    for nm in names:
        ts = [t for t in doc["targets"] if nm in t["arms"]]
        ok = sum(all(bool(s["meets_expected"]) for s in t["arms"][nm]["per_seed"]) for t in ts)
        L.append(f"| {nm} | {ok} / {len(ts)} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(run())
