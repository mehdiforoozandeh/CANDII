"""C17 -- the PI's metric panel (D15, D23) from existing score outputs; one TSV, no scoring.

    python tools/onion_eval/max_panel.py --recipe <runs dir>/<arm> [--nocov <runs dir>/<arm>]
        [--foreign-sub p16_foreign_out_keep_depth ...] [--foreign-ref-sub <sub>]
        --sub p256_sv --out <tsv>

Each arm directory holds `seed<k>/<sub>/{scores.json, suffstats.npz, SCORED}` (score.py). A seed
counts when select_rule.seed_state is `done` in that sub-directory (a CAPPED run never counts).

Panel rows (column `panel_row`):
  1. `r2_test_tracks`: R2 = 1 - SSE / SST per validation leave-one-assay-out target, in log1p
     counts at 25 bp over the eval regions, mean over targets; then mean and range over seeds.
  2. `covariates_gain`: recipe minus the no-covariates arm, on R2 and on the three terms of
     select_rule arm-summary (imputation_raw, correction_v3_raw, uncertainty_raw).
  3. `real_vs_foreign`: true covariates minus foreign covariates on the same quantities, one
     block per `--foreign-sub`, with `n_changed_by_column` from the foreign scores.json.
A difference is mean(a) - mean(b) over each side's seeds; `seed_range` is the larger of the two
seed ranges (max - min). Every row carries n_seeds (and n_seeds_ref) and the sub-directory.

Where R2 comes from. suffstats.npz holds no SSE or SST array. It holds, per (regime, assay), the
Haar Gram matrices of metrics.shared_specific: truth and the monotone-calibrated prediction, both
standardized by the truth's SD, so SST is the unit (Var(x) per bin). Their error part E_sh + E_sp,
summed over the Haar levels, is the calibrated residual MSE / Var(x) (errdecomp.calibrate
`residual`); scores.json per_target `level` (= offset + scale + nonlinear) is the error the
calibration removed. So per target SSE / SST = level + sum_k (E_sh + E_sp), exactly (errdecomp:
mse = offset + scale + nonlinear + residual). The Gram columns carry no target name, so the mean
is taken per assay (mean level + mean E over the same targets) and pooled with weights n_targets.
An assay is not derivable when a target has no target of another cell in its assay (E_sh, hence
E_sp, is NaN there) or when its column count differs from its count of targets with a level; it is
then left out and counted in `n_targets_not_derivable`. When any seed leaves a target out, the
panel also writes the fallback quantity `r_spec_sq_fallback`: the mean over scorable assays of the
square of scores.json `r_spec.counts_log1p.per_assay` (per-target r_spec is not stored; this is
the square of the per-assay mean r_spec, a different quantity from R2, labelled so).

A comparison whose two sides differ in evalset_sha256 or tables_sha256 gets no difference and a
`note` that says which key differs. The foreign rows compare each foreign sub with the true
covariates of the same arm in `--foreign-ref-sub` (default: the text before `_foreign` plus `_sv`,
e.g. `p16_foreign_out_keep_depth` -> `p16_sv`, the screen tier the foreign scoring uses).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))
from onion_eval import select_rule as SR  # noqa: E402

REGIME = "leave_one_assay_out"
SPACE = "counts_log1p"
H1_KEY = "specific_variance_recovered_above_noise"
TERMS = {"imputation_raw": "imputation_raw",
         "correction_v3_raw": "correction_gap_closed_oracle_no_nodedup_side",
         "uncertainty_raw": "uncertainty_raw"}
R2 = "r2_log1p"
FALLBACK = "r_spec_sq_fallback"
COLUMNS = ("panel_row", "comparison", "quantity", "value", "seed_range", "mean_a", "range_a",
           "mean_b", "range_b", "n_seeds", "n_seeds_ref", "arm", "sub", "arm_ref", "sub_ref",
           "n_targets", "n_targets_not_derivable", "n_changed_by_column", "source", "note")
R2_SOURCE = ("suffstats gram E_sh+E_sp over Haar levels (calibrated residual / Var(x)) + "
             "scores.json per_target level; SST = Var(x) per target")
FALLBACK_SOURCE = ("FALLBACK: mean over scorable assays of r_spec.counts_log1p.per_assay squared "
                   "(SSE/SST not derivable for some targets; per-target r_spec not stored)")
TERM_SOURCE = "select_rule.table (composite version 3), as arm-summary"


def _finite(x):
    return x is not None and isinstance(x, (int, float)) and not isinstance(x, bool) \
        and math.isfinite(x)


def r2_from_score(score_dir) -> dict:
    """{"r2", "n_targets", "n_targets_not_derivable", "assays_not_derivable", "r_spec_sq"} of one
    score directory (module docstring)."""
    d = Path(score_dir)
    s = json.loads((d / "scores.json").read_text(encoding="utf-8"))
    per = ((s.get(H1_KEY) or {}).get(SPACE) or {}).get("per_target") or {}
    levels: dict = {}
    n_all: dict = {}
    for t, e in per.items():
        if e.get("evaluation_only"):
            continue
        a = e.get("assay")
        n_all[a] = n_all.get(a, 0) + 1
        if _finite(e.get("level")):
            levels.setdefault(a, []).append(float(e["level"]))
    z = np.load(d / "suffstats.npz")
    num, n_used, bad = 0.0, 0, {}
    for a in sorted(n_all):
        kh, kp = f"gram/{REGIME}/{a}/E_sh_by_level", f"gram/{REGIME}/{a}/E_sp_by_level"
        lv = levels.get(a, [])
        if kh not in z.files or kp not in z.files:
            bad[a] = "no gram in suffstats"
            continue
        e = np.asarray(z[kh], np.float64) + np.asarray(z[kp], np.float64)   # [J+1, T]
        if e.ndim != 2 or e.shape[1] != len(lv) or len(lv) != n_all[a]:
            bad[a] = (f"{e.shape[-1]} gram columns, {len(lv)} targets with a level, "
                      f"{n_all[a]} targets")
            continue
        et = e.sum(0)
        if not np.all(np.isfinite(et)):
            bad[a] = "E undefined for a target (no target of another cell in this assay)"
            continue
        num += float(np.sum(1.0 - np.asarray(lv) - et))
        n_used += len(lv)
    rs = ((s.get("r_spec") or {}).get(SPACE) or {}).get("per_assay") or {}
    sc = set((s.get("scorable") or {}).get("imputation") or rs)
    vals = [float(v) ** 2 for a, v in rs.items() if a in sc and _finite(v)]
    return {"r2": num / n_used if n_used else None, "n_targets": n_used,
            "n_targets_not_derivable": sum(n_all[a] for a in bad), "assays_not_derivable": bad,
            "r_spec_sq": sum(vals) / len(vals) if vals else None,
            "evalset_sha256": s.get("evalset_sha256") or "",
            "tables_sha256": s.get("tables_sha256") or "",
            "n_changed_by_column": (s.get("foreign_covariates") or {}).get("n_changed_by_column")}


def arm_side(arm, sub: str, oracle="auto") -> dict:
    """Per-quantity seed values of one arm in one sub-directory, over its done seeds."""
    arm = Path(arm)
    rows = {r["seed"]: r for r in SR.table(arm.parent, arm.name, sub, oracle, 3)
            if r["arm"] == arm.name and r["state"] == "done"}
    vals = {q: [] for q in (R2, FALLBACK, *TERMS)}
    info = {"n_targets": [], "n_not": [], "sha": set(), "nchg": {}}
    for seed in sorted(rows):
        r = rows[seed]
        sd = arm / f"seed{seed}" / sub if sub else arm / f"seed{seed}"
        x = r2_from_score(sd)
        vals[R2].append(x["r2"])
        vals[FALLBACK].append(x["r_spec_sq"])
        for q, k in TERMS.items():
            vals[q].append(r.get(k))
        info["n_targets"].append(x["n_targets"])
        info["n_not"].append(x["n_targets_not_derivable"])
        info["sha"].add((x["evalset_sha256"], x["tables_sha256"]))
        info["nchg"][seed] = x["n_changed_by_column"]
    return {"arm": arm.name, "sub": sub, "n_seeds": len(rows), "vals": vals, **info}


def _agg(v: list):
    """(mean, range) over seeds; None when no seed or any seed undefined (as arm-summary)."""
    if not v or any(x is None for x in v):
        return None, None
    return SR._mean(v), SR.spread(v)


def _fmt(x):
    return "" if x is None else f"{x:.6g}" if isinstance(x, float) else str(x)


def _span(v: list) -> str:
    return "" if not v else str(v[0]) if min(v) == max(v) else f"{min(v)}-{max(v)}"


def _quantities(*sides) -> list:
    """R2 and the three terms; the fallback only when some seed left a target out of R2."""
    q = [R2]
    if any(n for s in sides for n in s["n_not"]):
        q.append(FALLBACK)
    return q + list(TERMS)


def _row(panel_row, comparison, q, a, b=None, note="", nchg=""):
    ma, ra = _agg(a["vals"][q])
    r = {"panel_row": panel_row, "comparison": comparison, "quantity": q, "mean_a": ma,
         "range_a": ra, "n_seeds": a["n_seeds"], "arm": a["arm"], "sub": a["sub"],
         "source": R2_SOURCE if q == R2 else FALLBACK_SOURCE if q == FALLBACK else TERM_SOURCE,
         "n_changed_by_column": nchg, "note": note}
    if q in (R2, FALLBACK):
        r["n_targets"] = _span(a["n_targets"] + (b["n_targets"] if b else []))
        r["n_targets_not_derivable"] = _span(a["n_not"] + (b["n_not"] if b else []))
    if b is None:
        r.update(value=ma, seed_range=ra)
        return r
    mb, rb = _agg(b["vals"][q])
    r.update(mean_b=mb, range_b=rb, n_seeds_ref=b["n_seeds"], arm_ref=b["arm"], sub_ref=b["sub"])
    if not note and None not in (ma, mb):
        r.update(value=ma - mb, seed_range=max(ra, rb))
    return r


def _mismatch(a: dict, b: dict) -> str:
    sh = a["sha"] | b["sha"]
    if not a["sha"] or not b["sha"]:
        return "no scored seed on one side"
    why = [k for i, k in enumerate(("evalset_sha256", "tables_sha256"))
           if len({x[i] for x in sh}) > 1]
    return ("not comparable: " + ", ".join(why) + " differs") if why else ""


def foreign_ref_sub(foreign_sub: str, given: str | None) -> str:
    if given:
        return given
    i = foreign_sub.find("_foreign")
    if i <= 0:
        raise SystemExit(f"--foreign-sub {foreign_sub!r}: no tier before '_foreign'; "
                         f"give --foreign-ref-sub")
    return foreign_sub[:i] + "_sv"


def panel(recipe, sub: str, nocov=None, foreign_subs=(), foreign_ref=None, oracle="auto") -> list:
    rec = arm_side(recipe, sub, oracle)
    rows = [_row("r2_test_tracks", "recipe", q, rec) for q in _quantities(rec)
            if q in (R2, FALLBACK)]
    if nocov is not None:
        nc = arm_side(nocov, sub, oracle)
        note = _mismatch(rec, nc)
        rows += [_row("covariates_gain", "recipe - nocov", q, rec, nc, note)
                 for q in _quantities(rec, nc)]
    refs: dict = {}
    for fs in foreign_subs:
        rsub = foreign_ref_sub(fs, foreign_ref)
        tr = refs.setdefault(rsub, arm_side(recipe, rsub, oracle))
        fo = arm_side(recipe, fs, oracle)
        note = _mismatch(tr, fo)
        ch = {json.dumps(v, sort_keys=True) for v in fo["nchg"].values()}
        nchg = (next(iter(ch)) if len(ch) == 1 else
                json.dumps({f"seed{k}": v for k, v in sorted(fo["nchg"].items())}, sort_keys=True))
        nchg = nchg.replace("\t", " ") if fo["nchg"] else ""
        rows += [_row("real_vs_foreign", f"true - foreign:{fs}", q, tr, fo, note, nchg)
                 for q in _quantities(tr, fo)]
    return rows


def write_tsv(rows: list, out) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(COLUMNS)
        for r in rows:
            w.writerow([_fmt(r.get(c)) for c in COLUMNS])
    tmp.replace(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--recipe", required=True, type=Path, help="<runs dir>/<arm> of the recipe")
    ap.add_argument("--nocov", type=Path, help="<runs dir>/<arm> of the no-covariates arm")
    ap.add_argument("--foreign-sub", action="append", default=[], type=SR.sub_name,
                    help="a foreign-covariate score sub-directory of the recipe (repeatable)")
    ap.add_argument("--foreign-ref-sub", type=SR.sub_name, default=None,
                    help="the true-covariate sub the foreign subs are compared with "
                         "(default: <tier>_sv from the foreign sub name)")
    ap.add_argument("--sub", required=True, type=SR.sub_name, help="score sub-directory, e.g. p256_sv")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    rows = panel(a.recipe, a.sub, a.nocov, a.foreign_sub, a.foreign_ref_sub)
    write_tsv(rows, a.out)
    print(f"max_panel: {len(rows)} rows -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
