"""OF -- the oracle floor of the correction gap closed (headline 4; PI ruling 47, 2026-11-14).

    python tools/onion_eval/oracle_floor.py compute --evalset <json> --out <dir> [--tables <dir>]
    -> <out>/oracle_per_region.npz, <out>/oracle_floor.json (json last)
    python tools/onion_eval/oracle_floor.py report <scores.json> [...] [--oracle <dir>]
    -> per score: the gap closed with the oracle floor and with the half-read floor, both subsets

Definition (ruling 47). For one correction pair, the oracle O is one half of the mean absolute
difference between the two read halves (pr1, pr2) of the TARGET view, on the bins, regions and raw
counts of the scored CRPS (the scored target is the pr2 half; the source is the pr1 half of the
source view). For CRPS, 0.5 E|X - X'| is the expected score of the true distribution, so O
estimates the lowest expected CRPS that any forecast can reach. Then

    gap closed (oracle) = (CRPS depth-only - CRPS method) / (CRPS depth-only - O)

Aggregation (the one rule of this module; cruxvault/results/gap_closed_floor/GAP_FLOOR.md): per
pair, each CRPS is the sum of its per-region sums over the sum of its bins; per assay, the ratio of
the pair means (numerator and denominator are means over the assay's pairs); the macro is the mean
of the per-assay ratios over the correction-scorable assays (`scorable.correction` of the score).
The interval is a 95% cluster bootstrap over cells (200 resamples, seed 0; score.py `_boot`). The
macro numerator and the macro denominator (means over the same assays of the per-assay numerator and
denominator) are reported beside it; their quotient is NOT the macro. Per knob: the ratio of pair
means pooled over assays (score.py's per_knob rule).

Subsets. `all`: the headline pairs (never trained, not a subset pair, correction pairs only).
`no_nodedup_side`: the headline pairs where neither side is a view without duplicate removal
(measurements.tsv `dedup` != 0 on both sides). When one side keeps duplicates, copies of one DNA
fragment can fall into both the source pr1 and the target pr2, so O is not a strict bound.

The reference (`compute`). It reuses the scorer's loaders (score.Scorer: the eval set, its tables
check, its pair jobs and pair order, its pair regions, OnionDataset.read). Per pair (the order of
`pairs_per_region.npz`, correction then replicate pairs) and per pair region: `crps_sum_oracle` =
0.5 x sum |pr1 - pr2| of the target, `bins`, `sq_half_sum` = 0.5 x sum (pr1 - pr2)^2, `pr1_sum`,
`pr2_sum`; per pair `source_dedup`, `target_dedup`, `nodedup_side` and the labels of
pairs_per_region.npz. A pair whose target has no halves gets NaN rows. The json holds the eval set
and tables sha256, the code git sha and the counts. No scores.json, suffstats.npz or
pairs_per_region.npz is read or written by `compute`.

Report time (select_rule, arm_report). `find_oracle` looks for `oracle_floor/seed0` beside the
score's references: `<refs.dir>/oracle_floor/seed0` (a two-phase score), the baselines dir of a
baseline score (`<B>/<name>/seed0/[sub]/scores.json` -> `<B>/oracle_floor/seed0`), then
`<dir of the eval set>/baselines*/oracle_floor/seed0`; the first whose json carries the score's
`evalset_sha256` and `tables_sha256` is used. No reference found: the score is read as before
(composite version 1). A reference found but the score's per-pair records missing or of other
regions: the new metric is undefined with the reason; the old value never fills the new column.
This module imports no torch at import time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

ORACLE_SUBDIR = "oracle_floor/seed0"
ORACLE_NPZ = "oracle_per_region.npz"
ORACLE_JSON = "oracle_floor.json"
COMPOSITE_VERSION = 2
GAP_CLOSED_FLOOR = "oracle"
SUPERSEDED = "superseded on 2026-11-14 by ruling 47 (half-read floor as the top of the scale)"
SUBSETS = ("all", "no_nodedup_side")
REGION_FIELDS = ("crps_sum_oracle", "bins", "sq_half_sum", "pr1_sum", "pr2_sum")
META_STR = ("assay", "cell", "rung", "knob", "source", "target")
META_INT = ("never_trained", "subset_pair", "source_dedup", "target_dedup", "nodedup_side")
DEFINITION = ("gap closed = (CRPS depth-only - CRPS method) / (CRPS depth-only - O), O = 0.5 x mean "
              "over the scored bins of |pr1 - pr2| of the target view (ruling 47); per assay a ratio "
              "of pair means, macro = mean over the correction-scorable assays, 95% cluster "
              "bootstrap over cells")
N_BOOT, SEED = 200, 0
_SEED_DIR = re.compile(r"^seed\d+$")


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------ compute

def _dedup(row: dict) -> int:
    """1 when the view has duplicates removed, 0 when it keeps them (measurements.tsv `dedup`)."""
    return 0 if str(row.get("dedup", "1")).strip() in ("0", "false", "False") else 1


def compute(evalset, out, tables=None, log=print) -> dict:
    """Build the oracle reference of one eval set (see the module docstring)."""
    from onion_eval import score as S
    sc = S.Scorer(evalset, main="depth_only_scaling", region_cache=False, log=log)
    if tables is not None and Path(tables).resolve() != Path(sc.tables).resolve():
        raise S.ScoreRefused(f"--tables {tables} is not the eval set's tables {sc.tables}")
    _, cor, rep, _ = sc.jobs()
    jobs = cor + rep
    keys = [S._pair_key(j) for j in jobs]
    ridx = list(sc.pair_region_idx)
    Rp, rb = len(ridx), sc.rb
    per_target: dict = {}
    t0 = time.time()
    for j in jobs:
        t = j["target"]
        if t in per_target:
            continue
        row = sc.ds.rows[t]
        if not S.has_halves(row):
            per_target[t] = {f: np.full(Rp, np.nan) for f in REGION_FIELDS}
            continue
        v = {f: np.zeros(Rp) for f in REGION_FIELDS}
        for k, ri in enumerate(ridx):
            c, s = sc.regions[ri]
            y1 = sc._ds_read(t, c, s, rb, "pr1").astype(np.float64)
            y2 = sc._ds_read(t, c, s, rb, "pr2").astype(np.float64)
            d = y1 - y2
            v["crps_sum_oracle"][k] = 0.5 * np.abs(d).sum()
            v["sq_half_sum"][k] = 0.5 * (d * d).sum()
            v["pr1_sum"][k], v["pr2_sum"][k] = y1.sum(), y2.sum()
            v["bins"][k] = rb
        per_target[t] = v
        if len(per_target) % 25 == 0:
            log(f"oracle_floor: {len(per_target)} targets, {time.time() - t0:.0f} s")
    rows = sc.ds.rows
    lab = []
    for j in jobs:
        sd, td = _dedup(rows[j["source"]]), _dedup(rows[j["target"]])
        lab.append({"assay": j.get("assay") or "", "cell": j.get("cell") or "",
                    "rung": j.get("rung") or "", "knob": j.get("knob") or "",
                    "source": j["source"], "target": j["target"],
                    "never_trained": int(j.get("never_trained") or 0),
                    "subset_pair": int(j.get("subset_pair") or 0),
                    "source_dedup": sd, "target_dedup": td, "nodedup_side": int(not (sd and td))})
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / ORACLE_NPZ, pair_ids=np.array(keys),
                        region_indices=np.asarray(ridx, np.int64),
                        **{f: np.array([x[f] for x in lab]) for f in META_STR},
                        **{f: np.array([x[f] for x in lab], np.int64) for f in META_INT},
                        **{f: np.stack([per_target[j["target"]][f] for j in jobs]) if jobs
                           else np.zeros((0, Rp)) for f in REGION_FIELDS})
    head = [x for x in lab if x["rung"] == "" and x["never_trained"] == 1 and x["subset_pair"] == 0]
    info = {"definition": DEFINITION, "evalset": str(sc.evalset_path),
            "evalset_sha256": _sha256(sc.evalset_path), "tables": str(sc.tables),
            "tables_sha256": sc.tables_sha256, "git_sha": S.git_sha(),
            "npz_sha256": _sha256(out / ORACLE_NPZ), "n_pairs": len(keys),
            "n_correction_pairs": len(cor), "n_replicate_pairs": len(rep),
            "n_targets": len(per_target),
            "n_targets_without_halves": sum(1 for v in per_target.values()
                                            if np.isnan(v["bins"]).all()),
            "n_headline_pairs": len(head),
            "n_headline_pairs_no_nodedup_side": sum(1 for x in head if not x["nodedup_side"]),
            "pair_regions_n": Rp, "region_bins": rb,
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "wall_seconds": round(time.time() - t0, 1)}
    tmp = out / (ORACLE_JSON + ".tmp")
    tmp.write_text(json.dumps(info, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(out / ORACLE_JSON)
    return info


# ------------------------------------------------------------------------------------------ reading

def load_oracle(d) -> dict:
    """{"dir", "info", "row" {pair id: index}, arrays...} of one oracle reference directory."""
    d = Path(d)
    info = json.loads((d / ORACLE_JSON).read_text(encoding="utf-8"))
    with np.load(d / ORACLE_NPZ, allow_pickle=False) as z:
        o = {k: z[k] for k in z.files}
    o["dir"], o["info"] = str(d), info
    o["row"] = {str(k): i for i, k in enumerate(o["pair_ids"])}
    return o


def _candidates(score_path: Path, s: dict) -> list[Path]:
    out = []
    rd = (s.get("refs") or {}).get("dir")
    if rd:
        out.append(Path(rd) / ORACLE_SUBDIR)
    for a in Path(score_path).resolve().parents:
        if _SEED_DIR.match(a.name):
            out.append(a.parent.parent / ORACLE_SUBDIR)
            break
    ev = s.get("evalset")
    if ev:
        out += sorted(Path(ev).parent.glob(f"baselines*/{ORACLE_SUBDIR}"))
    return out


def find_oracle(score_path, s: dict):
    """The oracle reference directory for one scores.json (module docstring), or None."""
    for c in _candidates(Path(score_path), s):
        j = c / ORACLE_JSON
        if not j.is_file() or not (c / ORACLE_NPZ).is_file():
            continue
        try:
            info = json.loads(j.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (info.get("evalset_sha256") and info.get("evalset_sha256") == s.get("evalset_sha256")
                and info.get("tables_sha256") == s.get("tables_sha256")):
            return c
    return None


def pair_records(score_dir, oracle: dict):
    """(records, reason): the headline-eligible correction pair records of one score joined with the
    oracle, from `<score_dir>/pairs_per_region.npz`; (None, reason) when they cannot be read."""
    p = Path(score_dir) / "pairs_per_region.npz"
    if not p.is_file():
        return None, f"no per-pair records ({p.name} missing beside scores.json)"
    with np.load(p, allow_pickle=False) as z:
        need = ("pair_ids", "region_indices", "crps_sum_model", "crps_sum_depth_only",
                "crps_sum_floor", "bins", "rung", "never_trained", "subset_pair", "assay", "cell")
        miss = [k for k in need if k not in z.files]
        if miss:
            return None, f"{p.name} lacks {', '.join(miss)}"
        z = {k: z[k] for k in z.files}
    if [int(i) for i in z["region_indices"]] != [int(i) for i in oracle["region_indices"]]:
        return None, (f"{p.name} holds {len(z['region_indices'])} pair regions, the oracle reference "
                      f"{len(oracle['region_indices'])} (other regions)")
    recs, absent = [], []
    for i, k in enumerate(z["pair_ids"]):
        if str(z["rung"][i]) != "":
            continue                                  # correction pairs only
        j = oracle["row"].get(str(k))
        if j is None:
            absent.append(str(k))
            continue
        b = float(np.sum(z["bins"][i]))
        bo = float(np.sum(oracle["bins"][j]))
        if not b > 0 or b != bo:
            return None, f"pair {k}: {b:g} bins in the score, {bo:g} in the oracle reference"
        r = {"pair_id": str(k), "assay": str(z["assay"][i]), "cell": str(z["cell"][i]),
             "knob": str(z["knob"][i]) if "knob" in z else "",
             "never_trained": int(z["never_trained"][i]), "subset_pair": int(z["subset_pair"][i]),
             "nodedup_side": int(oracle["nodedup_side"][j]),
             "crps_model": float(np.sum(z["crps_sum_model"][i])) / b,
             "crps_depth_only": float(np.sum(z["crps_sum_depth_only"][i])) / b,
             "crps_floor": float(np.sum(z["crps_sum_floor"][i])) / b,
             "oracle": float(np.sum(oracle["crps_sum_oracle"][j])) / b}
        recs.append(r)
    if absent:
        return None, f"{len(absent)} correction pairs not in the oracle reference (first: {absent[0]})"
    return recs, ""


# ------------------------------------------------------------------------------------------ aggregation

def _finite(rs, keys):
    return all(math.isfinite(r[k]) for r in rs for k in keys)


def ratio(rs, floor_key="oracle", model_key="crps_model"):
    """(value or None, numerator, denominator, reason) of one group: a ratio of pair means."""
    if not rs:
        return None, None, None, "no pair"
    if not _finite(rs, (model_key, "crps_depth_only", floor_key)):
        return None, None, None, "a pair CRPS is undefined"
    m = float(np.mean([r[model_key] for r in rs]))
    b = float(np.mean([r["crps_depth_only"] for r in rs]))
    f = float(np.mean([r[floor_key] for r in rs]))
    if not b - f > 0:
        return None, b - m, b - f, "depth-only is not above the floor"
    return (b - m) / (b - f), b - m, b - f, ""


def macro(rs, scorable, floor_key="oracle", require_all=True):
    """{"macro", "numerator", "denominator", "per_assay", "missing_scorable", "reason"} of one
    record set: per-assay ratio of pair means, then the mean over the scorable assays."""
    by = defaultdict(list)
    for r in rs:
        if r["assay"] in scorable:
            by[r["assay"]].append(r)
    per = {a: ratio(by[a], floor_key) for a in sorted(by)}
    missing = sorted(set(scorable) - set(by))
    out = {"per_assay": {a: v[0] for a, v in per.items()},
           "per_assay_numerator": {a: v[1] for a, v in per.items()},
           "per_assay_denominator": {a: v[2] for a, v in per.items()},
           "missing_scorable": missing, "n_assays": len(per), "n_pairs": len(rs),
           "macro": None, "numerator": None, "denominator": None, "reason": ""}
    bad = [f"{a} ({v[3]})" for a, v in per.items() if v[0] is None]
    if not per:
        out["reason"] = "no scorable assay with a pair"
    elif bad:
        out["reason"] = "undefined for " + ", ".join(bad)
    elif require_all and missing:
        out["reason"] = "scorable assay with no pair: " + ", ".join(missing)
    else:
        out["macro"] = float(np.mean([v[0] for v in per.values()]))
        out["numerator"] = float(np.mean([v[1] for v in per.values()]))
        out["denominator"] = float(np.mean([v[2] for v in per.values()]))
    return out


def boot(rs, fn, n_boot=N_BOOT, seed=SEED):
    """score.py `_boot`: 95% cluster bootstrap over cells; [None, None] when too few draws."""
    cells = sorted({r["cell"] for r in rs})
    if len(cells) < 2 or n_boot <= 0:
        return [None, None]
    by = defaultdict(list)
    for r in rs:
        by[r["cell"]].append(r)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(len(cells), len(cells), replace=True)
        v = fn([r for i in pick for r in by[cells[i]]])
        if v is not None:
            vals.append(float(v))
    if len(vals) < max(10, n_boot // 2):
        return [None, None]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def subset(recs, name):
    head = [r for r in recs if r["never_trained"] == 1 and r["subset_pair"] == 0]
    return head if name == "all" else [r for r in head if not r["nodedup_side"]]


def gap_closed_block(recs, scorable, n_boot=N_BOOT, seed=SEED) -> dict:
    """The gap closed with the oracle floor (and the superseded half-read value) on both subsets."""
    out = {"definition": DEFINITION, "superseded_halfread": SUPERSEDED}
    for name in SUBSETS:
        rs = subset(recs, name)
        req = name == "all"               # the stratum lacks some assays by construction
        new = macro(rs, scorable, "oracle", req)
        old = macro(rs, scorable, "crps_floor", req)
        new["ci"] = (boot(rs, lambda x: macro(x, scorable, "oracle", req)["macro"], n_boot, seed)
                     if new["macro"] is not None else [None, None])
        dens = {a: (d / b if b else None) for a, (d, b) in (
            (a, (new["per_assay_denominator"][a],
                 float(np.mean([r["crps_depth_only"] for r in rs if r["assay"] == a]))))
            for a in new["per_assay_denominator"]) if d is not None}
        knobs = defaultdict(list)
        for r in rs:
            knobs[r["knob"]].append(r)
        pk = {}
        for k, xs in sorted(knobs.items()):
            v, n, d, why = ratio(xs, "oracle")
            pk[k] = {"value": v, "numerator": n, "denominator": d, "n_pairs": len(xs),
                     "halfread_superseded": ratio(xs, "crps_floor")[0], "reason": why}
        v, n, d, _ = ratio(rs, "oracle")
        out[name] = {
            "gap_closed_oracle": new["macro"], "ci": new["ci"], "numerator": new["numerator"],
            "denominator": new["denominator"], "per_assay": new["per_assay"],
            "per_assay_numerator": new["per_assay_numerator"],
            "per_assay_denominator": new["per_assay_denominator"],
            "denominator_relative_by_assay": dens,
            "denominator_relative_min": min(dens.values()) if dens and len(dens) == new["n_assays"]
            else None,
            "pooled_over_assays": {"value": v, "numerator": n, "denominator": d},
            "gap_closed_halfread_superseded": old["macro"],
            "excess_macro": _excess(rs, scorable),
            "crps_model_macro": _mean_macro(rs, scorable, "crps_model"),
            "oracle_macro": _mean_macro(rs, scorable, "oracle"),
            "n_pairs": len(rs), "n_assays": new["n_assays"], "n_cells": len({r["cell"] for r in rs}),
            "missing_scorable": new["missing_scorable"],
            "n_pairs_below_oracle": sum(1 for r in rs if r["crps_model"] < r["oracle"]),
            "per_knob": pk, "reason": new["reason"]}
    return out


def _mean_macro(rs, scorable, key):
    by = defaultdict(list)
    for r in rs:
        if r["assay"] in scorable:
            by[r["assay"]].append(r[key])
    return float(np.mean([np.mean(v) for v in by.values()])) if by else None


def _excess(rs, scorable):
    m, o = _mean_macro(rs, scorable, "crps_model"), _mean_macro(rs, scorable, "oracle")
    return None if m is None or o is None else m - o


def seed_terms(score_path, s: dict, oracle="auto") -> dict | None:
    """Report-time correction term of one scores.json. None: no oracle reference (read the score
    as before, composite version 1). Else {"composite_version": 2, "gap_closed_floor": "oracle",
    "value" (all headline pairs; None when undefined), "reason", "denominator_relative_min",
    "block" (gap_closed_block or None), "oracle_dir"}. `oracle`: "auto" (find_oracle), a directory,
    or None/False (never use one)."""
    if not oracle:
        return None
    d = find_oracle(score_path, s) if oracle == "auto" else Path(oracle)
    if d is None:
        return None
    out = {"composite_version": COMPOSITE_VERSION, "gap_closed_floor": GAP_CLOSED_FLOOR,
           "value": None, "reason": "", "denominator_relative_min": None, "block": None,
           "oracle_dir": str(d)}
    o = load_oracle(d)
    if o["info"].get("evalset_sha256") != s.get("evalset_sha256"):
        out["reason"] = f"oracle reference {d} is of another eval set"
        return out
    recs, why = pair_records(Path(score_path).parent, o)
    if recs is None:
        out["reason"] = "gap closed (oracle floor) undefined: " + why
        return out
    sc = (s.get("scorable") or {}).get("correction") or []
    blk = gap_closed_block(recs, sc)
    a = blk["all"]
    out.update(value=a["gap_closed_oracle"], block=blk,
               reason=("" if a["gap_closed_oracle"] is not None
                       else "gap closed (oracle floor) undefined: " + a["reason"]),
               denominator_relative_min=a["denominator_relative_min"])
    return out


# ------------------------------------------------------------------------------------------ CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("compute")
    c.add_argument("--evalset", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--tables", default=None)
    r = sp.add_parser("report")
    r.add_argument("scores", nargs="+")
    r.add_argument("--oracle", default="auto")
    a = ap.parse_args(argv)
    if a.cmd == "compute":
        info = compute(a.evalset, a.out, a.tables)
        print(json.dumps({k: info[k] for k in ("n_pairs", "n_correction_pairs", "n_targets",
                                               "n_headline_pairs", "n_headline_pairs_no_nodedup_side",
                                               "pair_regions_n")}))
        return 0
    f = lambda x: "" if x is None else f"{x:.6g}"          # noqa: E731
    print("\t".join(("scores", "subset", "n_pairs", "gap_closed_oracle", "ci_lo", "ci_hi",
                     "numerator", "denominator", "gap_closed_halfread_superseded", "excess_macro",
                     "crps_model_macro", "oracle_macro", "reason")))
    for p in a.scores:
        s = json.loads(Path(p).read_text(encoding="utf-8"))
        t = seed_terms(p, s, a.oracle)
        if t is None or t["block"] is None:
            print("\t".join((p, "", "", "", "", "", "", "", "", "", "", "",
                             "no oracle reference found" if t is None else t["reason"])))
            continue
        for name in SUBSETS:
            b = t["block"][name]
            print("\t".join((p, name, str(b["n_pairs"]), f(b["gap_closed_oracle"]), f(b["ci"][0]),
                             f(b["ci"][1]), f(b["numerator"]), f(b["denominator"]),
                             f(b["gap_closed_halfread_superseded"]), f(b["excess_macro"]),
                             f(b["crps_model_macro"]), f(b["oracle_macro"]), b["reason"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
