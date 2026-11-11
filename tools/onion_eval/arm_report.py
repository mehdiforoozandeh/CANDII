"""RP -- arm report: the tables every settle chunk and both reports quote, from scores.json only.

    python tools/onion_eval/arm_report.py headline --runs <dir> [--prefix p] --baselines <dir>
        --out <tsv> [--sub blind]
    python tools/onion_eval/arm_report.py dev-evals --runs <dir> [--prefix p] [--baselines <dir>]
        --out <tsv> [--sub blind]
    python tools/onion_eval/arm_report.py accessibility --runs <dir> [--prefix p] [--baselines <dir>]
        --out <tsv> [--sub blind]
    python tools/onion_eval/arm_report.py per-knob --runs <dir> [--prefix p] --out <tsv> [--sub blind]
    python tools/onion_eval/arm_report.py shared-specific --runs <dir> --arms <a,b,...>
        [--baselines <dir>] [--out <tsv>] [--sub blind]

Inputs. An arm is a directory `<runs>/<arm>/seed<k>/scores.json` (score.py; select_rule's layout).
`--prefix` keeps the arms whose name starts with it; a directory with no `seed<k>` subdirectory is
not an arm (e.g. `settlements/`). A baseline is `<baselines>/<name>/seed0/scores.json`
(`eval/<tables>/baselines/`, as select_rule check-done reads it); every baseline found is a row
with `kind = baseline`. `--sub blind` reads `seed<k>/blind/scores.json` instead. Only scores.json
files are read. A seed directory without scores.json is not counted (n_seeds counts scored seeds).

Comparability. Every scores.json one command reads (arms and baselines) must carry the same
non-empty `evalset_sha256`; otherwise the command REFUSES: it prints the reason, writes no file and
exits 2.

Statistics over the seeds of one arm, for one quantity (a JSON number, or null when the scorer
marked it undefined): `n_seeds` = scored seeds, `n_defined` = seeds where it is defined, `mean`
and `spread` (max minus min over seeds, PI ruling 7) only when n_defined == n_seeds >= 1, else
empty. The spread of a baseline (one seed) is 0. Empty cell = undefined or absent.

headline. One row per arm, then one row per baseline. For each headline quantity Q (column name =
its key path in scores.json, `.` between levels, the `macro` value), the columns are `Q.mean`,
`Q.spread`, `Q.n_defined`, `Q.reference` and `Q.reference_source`. The reference is the best
training-free baseline (select_rule.TRAINING_FREE found under --baselines; "best" = highest for a
quantity where higher is better, lowest where lower is better, closest to 1 for the variance
ratio), except for the correction CRPS of the model, whose reference is the pseudoreplicate floor
(`correction_crps_pseudoreplicate_floor`, the arm's own seed mean): headline 1 is already "above
the noise floor" and headline 4 is already scaled between depth-only and the floor, and both of
those floors are their own columns. Every CRPS appears with its `oracle_scaled` and `scale_error`
(crps = oracle_scaled + scale_error, metrics.crps_split); score.py writes no scale_error for the
correction CRPS of the model, so `correction_crps_model_scale_error` is derived per seed as
`correction_crps_model - correction_crps_model_oracle_scaled` (macro minus macro).

dev-evals. Max's error budget (D12 development eval) from `error_budget`: one row per (arm, assay)
plus `assay = __macro__` (the scorer's mean over scorable assays), columns `level, shared,
specific, explained, unexplained, noise, added`, each `.mean` and `.spread`. `band` is `all`: the
scorer writes the budget summed over the four Haar bands (error_budget.BANDS) only, so no per-band
row exists in scores.json (score.py BUDGET_KEYS).

accessibility. The D11b second table: one row per (arm, assay, quantity) for the imputation
quantities, from the `accessibility_only` key, beside the same quantity of the leave-one-assay-out
regime (the top-level keys). Per-assay values come from `per_assay` and, for assays outside the
scorable set, `non_scorable_per_assay` (column `scorable` = 0); `assay = __macro__` is the macro.

per-knob. The correction gap closed per knob (headline 4, never-trained pairs with
`subset_pair = 0`, `correction_gap_closed_vs_depth_only_never_trained.per_knob`), one row per (arm,
knob) with `pair_kind = never_trained_non_subset`, then `knob = __macro__`; the subset (pure depth
thinning) pairs are apart, one row per arm with `pair_kind = subset` from
`subset_pairs_gap_closed` (score.py writes no per-knob split of them).

shared-specific. Each arm's gain in shared and in specific error (error_budget `shared`,
`specific`: error energies, lower is better) against the first arm of `--arms`: gain = reference
mean minus arm mean, so a positive gain is less error. `gain_spread` is the larger of the two arms'
seed spreads (ruling 7). One row per (arm, assay) plus `__macro__`. An arm name is looked up as
`<runs>/<name>`, then `<baselines>/<name>`.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))
from onion_eval.select_rule import TRAINING_FREE, seed_dirs  # noqa: E402

MACRO = "__macro__"
SPACES = ("counts_log1p", "signal_arcsinh", "signal_arcsinh_plug_in")
BUDGET_KEYS = ("level", "shared", "specific", "explained", "unexplained", "noise", "added")

# (key path, direction: "high" | "low" | "one" | None = no baseline reference)
HEADLINE = (
    *((("specific_variance_recovered_above_noise", s), "high") for s in SPACES),
    *((("r_spec", s), "high") for s in SPACES),
    (("count_crps_skill_vs_position_average",), "high"),
    (("count_crps",), "low"),
    (("count_crps_oracle_scaled",), "low"),
    (("count_crps_scale_error",), "low"),
    (("correction_gap_closed_vs_depth_only_never_trained",), "high"),
    (("correction_crps_model",), "floor"),
    (("correction_crps_model_oracle_scaled",), None),
    (("correction_crps_model_scale_error",), None),
    (("correction_crps_depth_only",), None),
    (("correction_crps_pseudoreplicate_floor",), None),
    (("subset_pairs_gap_closed",), "high"),
    (("depth_law_abs_log2_error_never_trained",), "low"),
    (("pit_deviation_pooled",), "low"),
    (("pit_deviation_worst_stratum",), "low"),
    (("predicted_over_replicate_variance",), "one"),
)
DERIVED_SCALE_ERROR = ("correction_crps_model_scale_error",)

# imputation quantities of the accessibility-only table (keys inside one regime's metric block)
IMPUTATION = (
    ("specific_variance_recovered_above_noise", "counts_log1p"),
    ("specific_variance_recovered_above_noise", "signal_arcsinh"),
    ("r_spec", "counts_log1p"),
    ("r_spec", "signal_arcsinh"),
    ("count_crps_skill_vs_position_average",),
    ("count_crps",),
    ("count_crps_oracle_scaled",),
    ("count_crps_scale_error",),
    ("pit_deviation_pooled",),
    ("predicted_over_replicate_variance",),
)


class Refused(Exception):
    """The scores read by one command are not comparable."""


# ------------------------------------------------------------------------------------------ reading

def _num(x):
    if isinstance(x, bool) or x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _get(d, path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _scores_path(seed_dir: Path, sub: str) -> Path:
    return seed_dir / sub / "scores.json" if sub else seed_dir / "scores.json"


def load_scores(arm: Path, sub: str = "") -> list[dict]:
    out = []
    for d in seed_dirs(arm):
        p = _scores_path(d, sub)
        if p.is_file():
            s = json.loads(p.read_text(encoding="utf-8"))
            s["__path__"] = str(p)
            out.append(s)
    return out


def find_arms(runs, prefix: str = "") -> list[Path]:
    runs = Path(runs)
    if not runs.is_dir():
        raise Refused(f"--runs {runs}: not a directory")
    return [p for p in sorted(runs.iterdir())
            if p.is_dir() and p.name.startswith(prefix) and seed_dirs(p)]


def find_baselines(baselines) -> list[Path]:
    if not baselines:
        return []
    b = Path(baselines)
    if not b.is_dir():
        raise Refused(f"--baselines {b}: not a directory")
    return [p for p in sorted(b.iterdir()) if p.is_dir() and (p / "seed0").is_dir()]


def check_evalset(groups: dict) -> str:
    """The one evalset_sha256 of every scores.json in `groups` ({name: [scores]}); Refused else."""
    shas, bad = {}, []
    for name, seeds in groups.items():
        for s in seeds:
            v = s.get("evalset_sha256") or ""
            if not v:
                bad.append(f"{s['__path__']}: no evalset_sha256")
            else:
                shas.setdefault(v, []).append(name)
    if len(shas) > 1:
        bad.append("evalset_sha256 differs: " + "; ".join(
            f"{k[:12]} ({', '.join(sorted(set(v)))})" for k, v in sorted(shas.items())))
    if bad:
        raise Refused("scores not comparable: " + "; ".join(bad[:6])
                      + (f"; ... {len(bad) - 6} more" if len(bad) > 6 else ""))
    return next(iter(shas), "")


# ------------------------------------------------------------------------------------------ statistics

def stats(vals: list) -> dict:
    """{n_seeds, n_defined, mean, spread}; mean and spread only when every seed is defined."""
    v = [_num(x) for x in vals]
    d = [x for x in v if x is not None]
    ok = bool(v) and len(d) == len(v)
    return {"n_seeds": len(v), "n_defined": len(d),
            "mean": sum(d) / len(d) if ok else None,
            "spread": (max(d) - min(d)) if ok else None}


def _fmt(x) -> str:
    if x is None:
        return ""
    if isinstance(x, bool):
        return str(int(x))
    if isinstance(x, float):
        return f"{x:.6g}"
    return str(x)


def write_tsv(rows: list[dict], columns: list[str], out) -> None:
    text = "\t".join(columns) + "\n" + "".join(
        "\t".join(_fmt(r.get(c)) for c in columns) + "\n" for r in rows)
    if out is None or str(out) == "-":
        sys.stdout.write(text)
        return
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(out)


def _load_groups(runs, prefix, baselines, sub):
    arms = {p.name: load_scores(p, sub) for p in find_arms(runs, prefix)}
    base = {p.name: load_scores(p, sub)[:1] for p in find_baselines(baselines)}
    return arms, base


def _seed_value(s: dict, path: tuple):
    """The macro of the quantity at `path` in one scores.json (derived scale error included)."""
    if path[0] in DERIVED_SCALE_ERROR:
        c = _num(_get(s, ("correction_crps_model", "macro")))
        o = _num(_get(s, ("correction_crps_model_oracle_scaled", "macro")))
        return None if c is None or o is None else c - o
    return _get(s, (*path, "macro"))


# ------------------------------------------------------------------------------------------ headline

def _best(direction, cands: dict):
    """(value, name) of the best defined baseline value."""
    vals = {k: v for k, v in cands.items() if v is not None}
    if not vals:
        return None, ""
    key = {"high": lambda k: -vals[k], "low": lambda k: vals[k],
           "one": lambda k: abs(vals[k] - 1.0)}[direction]
    name = min(sorted(vals), key=key)
    return vals[name], name


def headline(runs, prefix, baselines, sub="") -> tuple[list[dict], list[str]]:
    arms, base = _load_groups(runs, prefix, baselines, sub)
    check_evalset({**arms, **base})
    cols = ["arm", "kind", "n_seeds"]
    for path, _ in HEADLINE:
        q = ".".join(path)
        cols += [f"{q}.mean", f"{q}.spread", f"{q}.n_defined", f"{q}.reference",
                 f"{q}.reference_source"]
    base_means = {b: {path: _num(_seed_value(ss[0], path)) if ss else None for path, _ in HEADLINE}
                  for b, ss in base.items()}
    rows = []
    for kind, group in (("arm", arms), ("baseline", base)):
        for name, seeds in group.items():
            r = {"arm": name, "kind": kind, "n_seeds": len(seeds)}
            means = {}
            for path, direction in HEADLINE:
                q = ".".join(path)
                st = stats([_seed_value(s, path) for s in seeds])
                means[path] = st["mean"]
                r.update({f"{q}.mean": st["mean"], f"{q}.spread": st["spread"],
                          f"{q}.n_defined": st["n_defined"]})
            for path, direction in HEADLINE:
                q = ".".join(path)
                ref, src = None, ""
                if direction == "floor":
                    ref = means[("correction_crps_pseudoreplicate_floor",)]
                    src = "correction_crps_pseudoreplicate_floor" if ref is not None else ""
                elif direction is not None:
                    ref, src = _best(direction, {b: base_means[b][path] for b in base_means
                                                 if b in TRAINING_FREE})
                r[f"{q}.reference"], r[f"{q}.reference_source"] = ref, src
            rows.append(r)
    return rows, cols


# ------------------------------------------------------------------------------------------ dev evals

def dev_evals(runs, prefix, baselines=None, sub="") -> tuple[list[dict], list[str]]:
    arms, base = _load_groups(runs, prefix, baselines, sub)
    check_evalset({**arms, **base})
    cols = ["arm", "kind", "assay", "band", "n_seeds"] + [
        f"{k}.{x}" for k in BUDGET_KEYS for x in ("mean", "spread")]
    rows = []
    for kind, group in (("arm", arms), ("baseline", base)):
        for name, seeds in group.items():
            assays = sorted({a for s in seeds for a in (_get(s, ("error_budget", "per_assay")) or {})})
            for assay in assays + [MACRO]:
                r = {"arm": name, "kind": kind, "assay": assay, "band": "all", "n_seeds": len(seeds)}
                for k in BUDGET_KEYS:
                    p = ("error_budget", k) if assay == MACRO else ("error_budget", "per_assay", assay, k)
                    st = stats([_get(s, p) for s in seeds])
                    r[f"{k}.mean"], r[f"{k}.spread"] = st["mean"], st["spread"]
                rows.append(r)
    return rows, cols


# ------------------------------------------------------------------------------------------ accessibility

def _per_assay(block, path, assay):
    """(value, scorable 1|0|None) of one assay in a metric block."""
    m = _get(block, path) or {}
    if assay == MACRO:
        return m.get("macro"), None
    if assay in (m.get("per_assay") or {}):
        return m["per_assay"][assay], 1
    if assay in (m.get("non_scorable_per_assay") or {}):
        return m["non_scorable_per_assay"][assay], 0
    return None, None


def accessibility(runs, prefix, baselines=None, sub="") -> tuple[list[dict], list[str]]:
    arms, base = _load_groups(runs, prefix, baselines, sub)
    check_evalset({**arms, **base})
    cols = ["arm", "kind", "assay", "quantity", "scorable", "n_seeds",
            "accessibility_only.mean", "accessibility_only.spread", "accessibility_only.n_defined",
            "leave_one_assay_out.mean", "leave_one_assay_out.spread", "leave_one_assay_out.n_defined"]
    rows = []
    for kind, group in (("arm", arms), ("baseline", base)):
        for name, seeds in group.items():
            assays = set()
            for s in seeds:
                for path in IMPUTATION:
                    m = _get(s.get("accessibility_only") or {}, path) or {}
                    assays |= set(m.get("per_assay") or {}) | set(m.get("non_scorable_per_assay") or {})
            for assay in sorted(assays) + [MACRO]:
                for path in IMPUTATION:
                    acc = [_per_assay(s.get("accessibility_only") or {}, path, assay) for s in seeds]
                    loo = [_per_assay(s, path, assay) for s in seeds]
                    sa, sl = stats([v for v, _ in acc]), stats([v for v, _ in loo])
                    sc = {f for _, f in acc if f is not None}
                    rows.append({"arm": name, "kind": kind, "assay": assay, "quantity": ".".join(path),
                                 "scorable": sc.pop() if len(sc) == 1 else None,
                                 "n_seeds": len(seeds),
                                 "accessibility_only.mean": sa["mean"],
                                 "accessibility_only.spread": sa["spread"],
                                 "accessibility_only.n_defined": sa["n_defined"],
                                 "leave_one_assay_out.mean": sl["mean"],
                                 "leave_one_assay_out.spread": sl["spread"],
                                 "leave_one_assay_out.n_defined": sl["n_defined"]})
    return rows, cols


# ------------------------------------------------------------------------------------------ per knob

def per_knob(runs, prefix, baselines=None, sub="") -> tuple[list[dict], list[str]]:
    arms, base = _load_groups(runs, prefix, baselines, sub)
    check_evalset({**arms, **base})
    cols = ["arm", "kind", "pair_kind", "knob", "n_seeds", "n_defined", "mean", "spread"]
    key = "correction_gap_closed_vs_depth_only_never_trained"
    rows = []
    for kind, group in (("arm", arms), ("baseline", base)):
        for name, seeds in group.items():
            knobs = sorted({k for s in seeds for k in (_get(s, (key, "per_knob")) or {})})
            for knob, path in [(k, (key, "per_knob", k)) for k in knobs] + [(MACRO, (key, "macro"))]:
                st = stats([_get(s, path) for s in seeds])
                rows.append({"arm": name, "kind": kind, "pair_kind": "never_trained_non_subset",
                             "knob": knob, **st})
            st = stats([_get(s, ("subset_pairs_gap_closed", "macro")) for s in seeds])
            rows.append({"arm": name, "kind": kind, "pair_kind": "subset", "knob": MACRO, **st})
    return rows, cols


# ------------------------------------------------------------------------------------------ shared / specific

def _resolve(name, runs, baselines) -> tuple[Path, bool]:
    """(arm dir, is a baseline)."""
    for root, is_b in ((runs, False), (baselines, True)):
        if root and (Path(root) / name).is_dir():
            return Path(root) / name, is_b
    raise Refused(f"arm {name!r} not found under --runs {runs} or --baselines {baselines}")


def shared_specific(runs, arms: list[str], baselines=None, sub="") -> tuple[list[dict], list[str]]:
    if len(arms) < 2:
        raise Refused("--arms needs at least two arms: the reference first")
    groups = {}
    for a in arms:
        p, is_b = _resolve(a, runs, baselines)
        seeds = load_scores(p, sub)[:1] if is_b else load_scores(p, sub)
        if not seeds:
            raise Refused(f"arm {a!r}: no scored seed under {p}")
        groups[a] = seeds
    check_evalset(groups)
    ref = arms[0]
    assays = sorted({x for ss in groups.values() for s in ss
                     for x in (_get(s, ("error_budget", "per_assay")) or {})})
    cols = ["arm", "reference_arm", "assay", "n_seeds"]
    for k in ("shared", "specific"):
        cols += [f"{k}.mean", f"{k}.spread", f"{k}.gain", f"{k}.gain_spread"]

    def st(seeds, assay, k):
        p = ("error_budget", k) if assay == MACRO else ("error_budget", "per_assay", assay, k)
        return stats([_get(s, p) for s in seeds])

    rows = []
    for a in arms:
        for assay in assays + [MACRO]:
            r = {"arm": a, "reference_arm": ref, "assay": assay, "n_seeds": len(groups[a])}
            for k in ("shared", "specific"):
                sa, sr = st(groups[a], assay, k), st(groups[ref], assay, k)
                ok = sa["mean"] is not None and sr["mean"] is not None
                r.update({f"{k}.mean": sa["mean"], f"{k}.spread": sa["spread"],
                          f"{k}.gain": sr["mean"] - sa["mean"] if ok else None,
                          f"{k}.gain_spread": max(sa["spread"], sr["spread"]) if ok else None})
            rows.append(r)
    return rows, cols


# ------------------------------------------------------------------------------------------ entry

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("headline", "dev-evals", "accessibility", "per-knob"):
        p = sp.add_parser(name)
        p.add_argument("--runs", required=True)
        p.add_argument("--prefix", default="")
        p.add_argument("--baselines", required=(name == "headline"), default=None)
        p.add_argument("--out", required=True)
        p.add_argument("--sub", default="", choices=("", "blind"))
    p = sp.add_parser("shared-specific")
    p.add_argument("--runs", required=True)
    p.add_argument("--arms", required=True, help="comma-separated; the first is the reference")
    p.add_argument("--baselines", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--sub", default="", choices=("", "blind"))
    a = ap.parse_args(argv)
    fn = {"headline": headline, "dev-evals": dev_evals, "accessibility": accessibility,
          "per-knob": per_knob}
    try:
        if a.cmd == "shared-specific":
            rows, cols = shared_specific(a.runs, [x for x in a.arms.split(",") if x], a.baselines,
                                         a.sub)
        else:
            rows, cols = fn[a.cmd](a.runs, a.prefix, a.baselines, a.sub)
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    write_tsv(rows, cols, a.out)
    if a.out and a.out != "-":
        print(f"{a.cmd}: {len(rows)} rows -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
