"""E5 -- report-time tables from stored scores: per stratum, robust variance ratio, by scale, by covariate.

Close-out item E5 (PI 2026-11-16). Every table here is computed from files a scoring already wrote
(`scores.json`, `suffstats.npz`) and the run's `measurements.tsv`; no model is run and no score is
rewritten.

    python tools/onion_eval/report_extras.py <table> --runs <runs root> --sub <score sub-dir>
        [--arms a,b,...] [--regime leave_one_assay_out] [--tables <tables dir>] --out <rows.tsv>
        [--summary <summary.tsv>]
    python tools/onion_eval/report_extras.py <table> --dirs <score dir> [<score dir> ...] --out ...

A score dir is `<runs>/<arm>/seed<k>/<sub>/` (or `<arm>/seed<k>/` with `--sub ''`). Comparability:
every scores.json read must carry one `evalset_sha256`, and the scores of each arm one value of
every select_rule comparability key (select_rule.check_scorer_keys), else the command refuses (exit
2). `--summary` writes one row per (arm, key columns): mean and range (max minus min) over the
seeds, n_seeds and n_defined; the mean and range are empty unless every seed is defined.

Tables (`<table>`):
  strata          PIT deviation per region stratum and assay, from suffstats.npz
                  `pit/<regime>/<target>/<stratum>` (21-point F_bar curve + bin count): per assay the
                  bin-weighted pooled curve of its targets (score.pooled_pit_deviation, the scorer's
                  own estimator), macro = mean over the scorable imputation assays (undefined when one
                  is). Equal to scores.json `pit_deviation_per_stratum` where a scoring wrote it
                  (column `stored`, for the check). The worst stratum of the scorer is the maximum.
  variance-ratio  Headline 7 at the pseudoreplicate rung per assay: the stored mean of per-target
                  ratios (`predicted_over_replicate_variance`), the median of the per-target ratios
                  (from `per_target.<t>.per_rung.pseudoreplicate`), and the ratio of means. The ratio
                  of means needs each target's mean predicted variance and mean replicate variance,
                  which only scorings with the E5 scorer store (`predicted_over_replicate_variance_
                  robust`); for an older score it is empty with the reason "not in the stored files".
  by-scale        Model error by Haar scale band (fine 25-75 bp, peak 100 bp-1.6 kb, domain 3.2-
                  25.6 kb, broad 51 kb and up; error_budget.BANDS), from suffstats.npz
                  `gram/<regime>/<assay>/{V,P,C,E}_{sh,sp}_by_level` [15 levels x targets]: per assay,
                  part (shared, specific) and band, the track means V (truth variance), E (error),
                  explained = C^2 / P (at most V) and added = E - (V - explained), Max's eue() with no
                  noise term (error_budget._eue, N = 0): the sampling noise is inside `unexplained`.
                  When V < explained (a negative specific part), eue() sets unexplained to 0 and
                  carries V - explained in `floor_correction` (`floor_flag` 1), so V = explained +
                  unexplained_with_noise + floor_correction and E = unexplained_with_noise +
                  floor_correction + added in every row.
                  Units: the truth's variance of each target (counts_log1p, standardized). The
                  scorer's error budget sums these bands and subtracts the noise from the halves.
  by-covariate    Per-target values grouped by the target's lab, platform, antibody and donor
                  (measurements.tsv of `--tables`, else the scores.json `tables`): n targets, mean
                  count CRPS skill against the position average (suffstats crps/<regime>/<target>),
                  mean PIT deviation of the target (suffstats pit, pooled stratum), mean variance
                  recovered and mean headline 7 ratio (scores.json per_target). One row per (column,
                  value, assay) and per (column, value) over all assays (`assay` = `__all__`). A column
                  that the table lacks, or that is empty for every scored target, gets one row with
                  `value` = "not available in this table version". These are artifact strata of the
                  model's errors; the variance by artifact source in the data is mixed_model.py
                  `--extra-factors`.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

TABLES = ("strata", "variance-ratio", "by-scale", "by-covariate")
POOLED = "__pooled__"
ALL = "__all__"
NOT_AVAILABLE = "not available in this table version"
COVARIATE_COLUMNS = ("lab", "platform", "antibody", "donor")
GRAM_NAMES = ("V", "P", "C", "E")
PARTS = (("sh", "shared"), ("sp", "specific"))
NOT_STORED = "not in the stored files (needs a scoring with the E5 scorer)"


class Refused(Exception):
    pass


# ------------------------------------------------------------------------------------------ inputs

def score_dirs(runs=None, sub="", arms=None, dirs=None) -> list[Path]:
    """The score dirs to read: `dirs`, or `<runs>/<arm>/seed<k>/<sub>` with a scores.json."""
    if dirs:
        return [Path(d) for d in dirs]
    want = set(arms) if arms else None
    out = []
    for arm in sorted(Path(runs).iterdir()):
        if not arm.is_dir() or (want is not None and arm.name not in want):
            continue
        for sd in sorted(arm.glob("seed*")):
            d = sd / sub if sub else sd
            if (d / "scores.json").is_file():
                out.append(d)
    return out


def arm_seed(d: Path) -> tuple[str, str]:
    """(arm, seed) of a score dir `<arm>/seed<k>[/<sub>]`."""
    p = d if d.name.startswith("seed") else d.parent
    return (p.parent.name, p.name) if p.name.startswith("seed") else (d.name, "")


def load(dirs) -> list[dict]:
    """[{dir, arm, seed, scores}] with the comparability checks of the module docstring."""
    from onion_eval import select_rule as SR
    items = []
    for d in dirs:
        s = json.loads((Path(d) / "scores.json").read_text(encoding="utf-8"))
        a, k = arm_seed(Path(d))
        items.append({"dir": Path(d), "arm": a, "seed": k, "scores": s})
    if not items:
        raise Refused("no scores.json found")
    ev = {i["scores"].get("evalset_sha256") for i in items}
    if len(ev) != 1 or not next(iter(ev)):
        raise Refused(f"the scores carry {len(ev)} evaluation-set hashes; compare one eval set")
    by = defaultdict(list)
    for i in items:
        by[i["arm"]].append(i)
    for arm, its in by.items():
        try:
            SR.check_scorer_keys([(str(i["dir"]), SR.scorer_keys(i["scores"])) for i in its])
        except SR.Refused as e:
            raise Refused(f"arm {arm}: {e}") from None
    return items


def _suff(d: Path):
    p = d / "suffstats.npz"
    return np.load(p, allow_pickle=False) if p.is_file() else None


def _f(x):
    """A JSON number or None -> float or None (NaN is None)."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _assays(blk: dict) -> dict:
    """{target: assay} of one imputation table's per_target block (headline 1, counts_log1p)."""
    pt = (blk.get("specific_variance_recovered_above_noise", {}).get("counts_log1p", {})
          .get("per_target") or {})
    return {t: e.get("assay") for t, e in pt.items() if not e.get("evaluation_only")}


def _table_block(s: dict, regime: str) -> dict:
    return s if regime == "leave_one_assay_out" else (s.get(regime) or {})


def _macro(per: dict, scorable) -> float | None:
    vals = [per.get(a) for a in sorted(scorable)]
    if not vals or any(v is None for v in vals):
        return None
    return float(np.mean(vals))


# ------------------------------------------------------------------------------------------ tables

def table_strata(item, regime="leave_one_assay_out") -> list[dict]:
    from onion_eval.score import pooled_pit_deviation
    s, z = item["scores"], _suff(item["dir"])
    if z is None:
        return [{"stratum": "", "assay": "", "value": None, "stored": None,
                 "reason": "no suffstats.npz"}]
    tgt = _assays(_table_block(s, regime))
    scorable = (s.get("scorable") or {}).get("imputation", [])
    curves = defaultdict(lambda: defaultdict(list))         # stratum -> assay -> [(fbar, n)]
    pre = f"pit/{regime}/"
    for k in z.files:
        if not k.startswith(pre):
            continue
        t, _, stratum = k[len(pre):].rpartition("/")
        if t not in tgt or stratum == POOLED:
            continue
        v = np.asarray(z[k], np.float64)
        curves[stratum][tgt[t]].append((v[:-1], int(v[-1])))
    stored = (_table_block(s, regime).get("pit_deviation_per_stratum") or {}).get("per_stratum") or {}
    rows = []
    for stratum in sorted(curves):
        per = {}
        for a, cs in curves[stratum].items():
            v = pooled_pit_deviation(cs)
            per[a] = _f(v)
        st = stored.get(stratum) or {}
        for a in sorted(per):
            rows.append({"stratum": stratum, "assay": a, "value": per[a],
                         "stored": _f((st.get("per_assay") or {}).get(a, (st.get("non_scorable_per_assay")
                                                                          or {}).get(a))),
                         "reason": "" if a in scorable else "non-scorable assay"})
        rows.append({"stratum": stratum, "assay": "__macro__", "value": _macro(per, scorable),
                     "stored": _f(st.get("macro")), "reason": ""})
    return rows


def table_variance_ratio(item, regime="leave_one_assay_out") -> list[dict]:
    s = item["scores"]
    blk = _table_block(s, regime)
    h7 = blk.get("predicted_over_replicate_variance") or {}
    rob = blk.get("predicted_over_replicate_variance_robust") or {}
    tgt = _assays(blk)
    scorable = (s.get("scorable") or {}).get("imputation", [])
    by = defaultdict(list)
    for t, e in (h7.get("per_target") or {}).items():
        if e.get("evaluation_only"):
            continue
        a = tgt.get(t)
        if a is None:
            continue
        by[a].append(_f((e.get("per_rung") or {}).get("pseudoreplicate")))
    med = {a: (float(np.median(v)) if v and all(x is not None for x in v) else None)
           for a, v in by.items()}
    rom = (rob.get("ratio_of_means") or {})
    rom_pa = {**(rom.get("non_scorable_per_assay") or {}), **(rom.get("per_assay") or {})}
    mean_pa = {**(h7.get("non_scorable_per_assay") or {}), **(h7.get("per_assay") or {})}
    rows = []
    for a in sorted(by):
        rows.append({"assay": a, "n_targets": len(by[a]), "mean_of_targets": _f(mean_pa.get(a)),
                     "median_of_targets": med[a],
                     "ratio_of_means": _f(rom_pa.get(a)) if rob else None,
                     "max_target": max((x for x in by[a] if x is not None), default=None),
                     "reason": "" if rob else NOT_STORED})
    rows.append({"assay": "__macro__", "n_targets": sum(len(v) for v in by.values()),
                 "mean_of_targets": _f(h7.get("macro")), "median_of_targets": _macro(med, scorable),
                 "ratio_of_means": _f(rom.get("macro")) if rob else None, "max_target": None,
                 "reason": "" if rob else NOT_STORED})
    return rows


def table_by_scale(item, regime="leave_one_assay_out") -> list[dict]:
    from onion_eval.error_budget import BANDS, _eue
    z = _suff(item["dir"])
    if z is None:
        return [{"assay": "", "part": "", "band": "", "reason": "no suffstats.npz"}]
    pre = f"gram/{regime}/"
    assays = sorted({k[len(pre):].split("/")[0] for k in z.files if k.startswith(pre)})
    rows = []
    for a in assays:
        for short, part in PARTS:
            g = {nm: np.asarray(z[f"{pre}{a}/{nm}_{short}_by_level"], np.float64)
                 for nm in GRAM_NAMES}
            ok = np.isfinite(g["V"]).all(axis=0)                 # targets with a defined split
            if not ok.any():
                rows.append({"assay": a, "part": part, "band": "", "n_targets": 0,
                             "reason": "no target with a shared/specific split (one cell?)"})
                continue
            for b, idx in BANDS.items():
                V, P, C, E = (float(g[nm][idx][:, ok].sum(0).mean()) for nm in GRAM_NAMES)
                r = _eue(V, C, P, E, 0.0)
                rows.append({"assay": a, "part": part, "band": b, "n_targets": int(ok.sum()),
                             "V": r["V"], "E": r["E"], "explained": r["explained"],
                             "unexplained_with_noise": r["unexplained"], "added": r["added"],
                             "floor_correction": r["noise"], "floor_flag": int(r["floor_flag"]),
                             "reason": ""})
    return rows


def _read_meas(tables) -> tuple[dict, list]:
    p = Path(tables) / "measurements.tsv"
    with open(p, newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        return {r["measurement_id"]: r for r in rd}, list(rd.fieldnames or [])


def table_by_covariate(item, regime="leave_one_assay_out", tables=None) -> list[dict]:
    from onion_eval.score import pooled_pit_deviation
    s, z = item["scores"], _suff(item["dir"])
    meas, header = _read_meas(tables or s.get("tables"))
    blk = _table_block(s, regime)
    h1 = ((blk.get("specific_variance_recovered_above_noise") or {}).get("counts_log1p") or {})
    h7 = blk.get("predicted_over_replicate_variance") or {}
    per = {}
    for t, e in (h1.get("per_target") or {}).items():
        if e.get("evaluation_only"):
            continue
        v = {"assay": e.get("assay"), "recovered": _f(e.get("recovered")),
             "h7": _f(((h7.get("per_target") or {}).get(t, {}).get("per_rung") or {})
                      .get("pseudoreplicate")), "skill": None, "pit": None}
        if z is not None:
            k = f"crps/{regime}/{t}"
            if k in z.files:
                c = np.asarray(z[k], np.float64)              # crps, oracle, crps_pa, bins
                v["skill"] = _f(c[2] - c[0])
            k = f"pit/{regime}/{t}/{POOLED}"
            if k in z.files:
                c = np.asarray(z[k], np.float64)
                v["pit"] = _f(pooled_pit_deviation([(c[:-1], int(c[-1]))]))
        per[t] = v
    rows = []
    for col in COVARIATE_COLUMNS:
        vals = {t: (meas.get(t, {}).get(col) or "").strip() for t in per}
        if col not in header or not any(vals.values()):
            why = f"column {col} absent" if col not in header else f"column {col} empty for every target"
            rows.append({"column": col, "value": NOT_AVAILABLE, "assay": ALL, "n_targets": len(per),
                         "reason": why})
            continue
        groups = defaultdict(list)
        for t, v in per.items():
            val = vals[t] or "(empty)"
            groups[(val, v["assay"])].append(v)
            groups[(val, ALL)].append(v)
        for (val, a), vs in sorted(groups.items()):
            row = {"column": col, "value": val, "assay": a, "n_targets": len(vs), "reason": ""}
            for k in ("skill", "pit", "recovered", "h7"):
                xs = [x[k] for x in vs if x[k] is not None]
                row[f"{k}_mean"] = float(np.mean(xs)) if xs else None
                row[f"{k}_n"] = len(xs)
            rows.append(row)
    return rows


# ------------------------------------------------------------------------------------------ output

KEYS = {"strata": ("stratum", "assay"), "variance-ratio": ("assay",),
        "by-scale": ("assay", "part", "band"), "by-covariate": ("column", "value", "assay")}


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def write_tsv(rows, path, first=("arm", "seed", "dir")) -> None:
    cols = list(first) + [c for c in dict.fromkeys(k for r in rows for k in r) if c not in first]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            w.writerow([_fmt(r.get(c)) for c in cols])


def summarize(rows, keys) -> list[dict]:
    """Mean and range over the seeds of each (arm, keys); a value column is any numeric column."""
    by = defaultdict(list)
    for r in rows:
        by[(r["arm"],) + tuple(r.get(k) for k in keys)].append(r)
    out = []
    for gk, rs in by.items():
        row = {"arm": gk[0], **dict(zip(keys, gk[1:])), "n_seeds": len(rs)}
        cols = [c for c in dict.fromkeys(k for r in rs for k in r)
                if c not in ("arm", "seed", "dir", *keys) and all(
                    isinstance(r.get(c), (int, float)) or r.get(c) is None for r in rs)
                and any(isinstance(r.get(c), (int, float)) for r in rs)]
        for c in cols:
            v = [r.get(c) for r in rs]
            ok = [float(x) for x in v if x is not None]
            row[f"{c}.n_defined"] = len(ok)
            row[f"{c}.mean"] = float(np.mean(ok)) if ok and len(ok) == len(v) else None
            row[f"{c}.range"] = float(max(ok) - min(ok)) if ok and len(ok) == len(v) else None
        out.append(row)
    return out


def run(table, dirs, regime="leave_one_assay_out", tables=None) -> list[dict]:
    if table not in TABLES:
        raise ValueError(f"table {table!r} not in {TABLES}")
    items = load(dirs)
    rows = []
    for it in items:
        if table == "strata":
            rs = table_strata(it, regime)
        elif table == "variance-ratio":
            rs = table_variance_ratio(it, regime)
        elif table == "by-scale":
            rs = table_by_scale(it, regime)
        else:
            rs = table_by_covariate(it, regime, tables)
        rows += [dict({"arm": it["arm"], "seed": it["seed"], "dir": str(it["dir"])}, **r) for r in rs]
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("table", choices=TABLES)
    ap.add_argument("--runs")
    ap.add_argument("--sub", default="")
    ap.add_argument("--arms", default="")
    ap.add_argument("--dirs", nargs="*")
    ap.add_argument("--regime", default="leave_one_assay_out",
                    choices=("leave_one_assay_out", "accessibility_only"))
    ap.add_argument("--tables", default=None,
                    help="by-covariate: the tables dir (default: scores.json `tables`)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", default=None)
    a = ap.parse_args(argv)
    if not (a.runs or a.dirs):
        ap.error("give --runs or --dirs")
    try:
        dirs = score_dirs(a.runs, a.sub, [x for x in a.arms.split(",") if x], a.dirs)
        rows = run(a.table, dirs, a.regime, a.tables)
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    write_tsv(rows, a.out)
    if a.summary:
        write_tsv(summarize(rows, KEYS[a.table]), a.summary, first=("arm",) + KEYS[a.table])
    print(f"report_extras {a.table}: {len(rows)} rows from {len(dirs)} score dirs -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
