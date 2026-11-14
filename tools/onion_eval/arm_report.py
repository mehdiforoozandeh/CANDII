"""RP -- arm report: the tables every settle chunk and both reports quote, from scores.json only.

    python tools/onion_eval/arm_report.py headline --runs <dir> [--prefix p] --baselines <dir>
        --out <tsv> [--sub blind] [--addon ox] [--pairs <pairs.tsv>]
    python tools/onion_eval/arm_report.py explained --runs <dir> [--prefix p] [--baselines <dir>]
        --out <tsv> [--sub blind] [--addon ox]
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
with `kind = baseline`. `--sub blind` reads `seed<k>/blind/scores.json` instead. Any other `--sub
SUB` (Amendment 10, interface M: `p16`, `p256`) reads the arms' `seed<k>/<SUB>/scores.json` and a
baseline's `seed0/<SUB>/scores.json` if present, else its `seed0/scores.json` (`baselines_p16`
layout). Only scores.json files are read. A seed directory without scores.json is not counted (n_seeds counts scored seeds).

Comparability. Every scores.json one command reads (arms and baselines) must carry the same
non-empty `evalset_sha256`; otherwise the command REFUSES: it prints the reason, writes no file and
exits 2. Every ARM score must also share the scorer keys of select_rule (`tables_sha256`,
`scorer_version`, `scorer_settings.device`, `scorer_settings.infer_batch`, `refs.sha256_*`), else
REFUSED; baselines are references and are exempt (a scorer_version 1 baseline is comparable with
version 2 runs). `--allow-mix <key>` exempts the device or the batch only.

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

headline, metric 7 (PI ruling 31). The block `predicted_over_replicate_variance` is the scorer's old
scalar (the pseudoreplicate ratio), now read as the lower-bound check: its columns keep their names,
and its column `predicted_over_replicate_variance.target` says `none (lower_bound_check)`. Four
quantities follow, each with `.mean`, `.spread`, `.n_defined` and `.target`:
  `predicted_over_replicate_variance.per_rung.pseudoreplicate` (target `>= 1`; from scores.json),
  `predicted_over_replicate_variance.per_rung.other_experiment` (target `<= 1`; the upper bound),
  `predicted_over_replicate_variance.pass_rate` (target `high`; share of targets that pass both),
  `predicted_over_replicate_variance.n_targets_with_both_bounds` (no target; per seed, the count of
  `per_target` entries whose `pass` is defined, i.e. both bounds are defined).
The last three come from the add-on score `<seed dir>/<addon>/scores.json` (`--addon`, default `ox`;
with `--sub blind`, `<seed dir>/blind/<addon>/scores.json`), the score of the add-on eval set that
holds the second experiments (PI ruling 33). Every add-on file one command reads must carry one
non-empty `evalset_sha256` (REFUSED else), which may differ from the frozen one. An arm or baseline
with no add-on file for any seed prints `not populated` in `.mean`, `.spread` and `.n_defined` of
those three; that is not an error. `pass_rate` has a reference (best training-free baseline, highest
is best); the bounds and the count have none. `--pairs <pairs.tsv>`: the `other_experiment` pairs give
each target (side `a` or `b`) its pair class, the `experiment`, `lab`, `donor`, `donor?` part of
`knobs_differing` (`mixed` when a target sits in pairs of two classes). The columns
`predicted_over_replicate_variance.by_class.<class>.{upper_bound,pass_rate,n_targets}.{mean,spread,
n_defined}` are added: per seed, the plain mean over the class's targets in `per_target` (the scorer
writes no assay per target, so this is not the macro over assays), Undefined when one of them is
Undefined (the scorer's rule, score._mean_of); `n_targets` counts them. With `--pairs`, a target that is the `b`
side (the second experiment) of an `other_experiment` pair is left out of the count and of the class
columns: it is an evaluation-only target, not an EIC target.

explained. The explained fraction per layer (PI ruling 32; reported only), from
`explained_fraction_per_layer.per_assay.<assay>.rows` of the add-on score. One row per (arm, assay,
rung, layer, pair_class) over the seeds: `explained_fraction`, `captured`, `layer_term`, each `.mean`,
`.spread`, `.n_defined`; `n_pairs` and `n_cells` (max over seeds) and `flag` (the distinct flags, `; `
joined). An assay with no row in any seed gives one row with an empty layer and its reason as the
flag; an arm with no add-on file gives one row with flag `not populated`.

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
from onion_eval import select_rule as _SR  # noqa: E402

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

# headline 7 with two bounds (PI ruling 31): (column, target text, reference direction, from add-on)
H7 = "predicted_over_replicate_variance"
H7_OLD_TARGET = "none (lower_bound_check)"
H7_COLUMNS = (
    (f"{H7}.per_rung.pseudoreplicate", ">= 1", None, False),
    (f"{H7}.per_rung.other_experiment", "<= 1", None, True),
    (f"{H7}.pass_rate", "high", "high", True),
    (f"{H7}.n_targets_with_both_bounds", "", None, True),
)
H7_UPPER = "other_experiment"
NOT_POPULATED = "not populated"
ADDON = "ox"
CLASS_TOKENS = ("experiment", "lab", "donor", "donor?")
EXPLAINED_KEYS = ("explained_fraction", "captured", "layer_term")

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


def load_baseline(b: Path, sub: str = "") -> list[dict]:
    """The one scores.json of a baseline. With a score subdirectory other than `blind` (interface
    M: `p16`, `p256`), `seed0/<sub>/scores.json` when it exists, else `seed0/scores.json` (a
    baseline of a derived evaluation set lives in its own baselines dir, e.g. baselines_p16)."""
    if sub and sub != "blind" and not any(_scores_path(d, sub).is_file() for d in seed_dirs(b)):
        sub = ""
    return load_scores(b, sub)[:1]


def check_arm_keys(arms: dict, allow=()) -> None:
    """Refused unless every arm score shares the scorer keys of select_rule.SCORER_KEYS (Amendment
    10); baselines are references and are not passed here."""
    try:
        _SR.check_scorer_keys([(s["__path__"], _SR.scorer_keys(s))
                               for seeds in arms.values() for s in seeds], allow)
    except _SR.Refused as e:
        raise Refused(str(e)) from None


def _load_groups(runs, prefix, baselines, sub, allow_mix=()):
    arms = {p.name: load_scores(p, sub) for p in find_arms(runs, prefix)}
    base = {p.name: load_baseline(p, sub) for p in find_baselines(baselines)}
    check_arm_keys(arms, allow_mix)
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


def load_addon(seeds: list[dict], addon: str) -> list:
    """The add-on score beside each scores.json of `seeds` (`<its dir>/<addon>/scores.json`), or None."""
    out = []
    for s in seeds:
        p = Path(s["__path__"]).parent / addon / "scores.json"
        if p.is_file():
            x = json.loads(p.read_text(encoding="utf-8"))
            x["__path__"] = str(p)
            out.append(x)
        else:
            out.append(None)
    return out


def _check_addon(add: dict) -> str:
    """One evalset_sha256 over every add-on score present ({kind: {name: [score | None]}}); Refused
    else."""
    present = {f"{kind} {k}": [x for x in v if x is not None]
               for kind, g in add.items() for k, v in g.items()}
    return check_evalset({k: v for k, v in present.items() if v})


def pair_class(knobs: str) -> str:
    """The `experiment`, `lab`, `donor` part of `knobs_differing` (technical knobs dropped)."""
    return ";".join(t for t in (knobs or "").split(";") if t in CLASS_TOKENS)


def read_pair_classes(path) -> tuple[dict, set]:
    """({target id: pair class}, {second-experiment ids}) from the other_experiment rows of pairs.tsv."""
    p = Path(path)
    if not p.is_file():
        raise Refused(f"--pairs {p}: not a file")
    with open(p, newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        need = {"a", "b", "rung", "knobs_differing"}
        if not need <= set(rd.fieldnames or ()):
            raise Refused(f"--pairs {p}: needs the columns {', '.join(sorted(need))}")
        classes, second = {}, set()
        for r in rd:
            if r["rung"] != H7_UPPER:
                continue
            c = pair_class(r["knobs_differing"])
            for side in ("a", "b"):
                classes.setdefault(r[side], set()).add(c)
            second.add(r["b"])
    return {t: (c.pop() if len(c) == 1 else "mixed") for t, c in classes.items()}, second


def _h7_seed(x, second: set, cls_of: dict, classes: list) -> dict:
    """Add-on values of one seed: {column or (class, quantity): value}; None when x is None."""
    if x is None:
        return {}
    per_target = _get(x, (H7, "per_target")) or {}
    targets = {t: v for t, v in per_target.items() if t not in second and isinstance(v, dict)}
    out = {f"{H7}.per_rung.{H7_UPPER}": _get(x, (H7, "per_rung", H7_UPPER, "macro")),
           f"{H7}.pass_rate": _get(x, (H7, "pass_rate", "macro")),
           f"{H7}.n_targets_with_both_bounds":
               sum(1 for v in targets.values() if _num(v.get("pass")) is not None)}
    for c in classes:
        ts = [v for t, v in targets.items() if cls_of.get(t) == c]
        ub = [_num((v.get("per_rung") or {}).get(H7_UPPER)) for v in ts]
        ps = [_num(v.get("pass")) for v in ts]
        out[(c, "upper_bound")] = sum(ub) / len(ub) if ub and None not in ub else None
        out[(c, "pass_rate")] = sum(ps) / len(ps) if ps and None not in ps else None
        out[(c, "n_targets")] = len(ts)
    return out


def headline(runs, prefix, baselines, sub="", addon=ADDON, pairs=None, allow_mix=()) -> tuple[list[dict], list[str]]:
    arms, base = _load_groups(runs, prefix, baselines, sub, allow_mix)
    check_evalset({**arms, **base})
    add = {"arm": {k: load_addon(v, addon) for k, v in arms.items()},
           "baseline": {k: load_addon(v, addon) for k, v in base.items()}}
    _check_addon(add)
    cls_of, second = read_pair_classes(pairs) if pairs else ({}, set())
    classes = sorted(set(cls_of.values()))
    cols = ["arm", "kind", "n_seeds"]
    for path, _ in HEADLINE:
        q = ".".join(path)
        cols += [f"{q}.mean", f"{q}.spread", f"{q}.n_defined", f"{q}.reference",
                 f"{q}.reference_source"]
    cols.append(f"{H7}.target")
    for q, _, ref_dir, _ in H7_COLUMNS:
        cols += [f"{q}.mean", f"{q}.spread", f"{q}.n_defined", f"{q}.target"]
        if ref_dir:
            cols += [f"{q}.reference", f"{q}.reference_source"]
    for c in classes:
        for k in ("upper_bound", "pass_rate", "n_targets"):
            cols += [f"{H7}.by_class.{c}.{k}.{x}" for x in ("mean", "spread", "n_defined")]
    h7_seeds = {kind: {k: [_h7_seed(x, second, cls_of, classes) for x in v] for k, v in g.items()}
                for kind, g in add.items()}
    pr_path = (H7, "per_rung", "pseudoreplicate", "macro")
    base_h7 = {b: (h7_seeds["baseline"][b][0] if h7_seeds["baseline"][b] else {}) for b in base}
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
            r[f"{H7}.target"] = H7_OLD_TARGET
            hs = h7_seeds[kind][name]
            populated = any(x for x in hs)
            for q, target, ref_dir, from_addon in H7_COLUMNS:
                r[f"{q}.target"] = target
                if from_addon and not populated:
                    r.update({f"{q}.{x}": NOT_POPULATED for x in ("mean", "spread", "n_defined")})
                else:
                    st = stats([h.get(q) for h in hs] if from_addon
                               else [_get(s, pr_path) for s in seeds])
                    r.update({f"{q}.mean": st["mean"], f"{q}.spread": st["spread"],
                              f"{q}.n_defined": st["n_defined"]})
                if ref_dir:
                    r[f"{q}.reference"], r[f"{q}.reference_source"] = _best(
                        ref_dir, {b: _num(base_h7[b].get(q)) for b in base_h7 if b in TRAINING_FREE})
            for c in classes:
                for k in ("upper_bound", "pass_rate", "n_targets"):
                    p = f"{H7}.by_class.{c}.{k}"
                    if not populated:
                        r.update({f"{p}.{x}": NOT_POPULATED for x in ("mean", "spread", "n_defined")})
                        continue
                    st = stats([h.get((c, k)) for h in hs])
                    r.update({f"{p}.mean": st["mean"], f"{p}.spread": st["spread"],
                              f"{p}.n_defined": st["n_defined"]})
            rows.append(r)
    return rows, cols


# ------------------------------------------------------------------------------------------ explained

def explained(runs, prefix, baselines=None, sub="", addon=ADDON, allow_mix=()) -> tuple[list[dict], list[str]]:
    arms, base = _load_groups(runs, prefix, baselines, sub, allow_mix)
    add = {"arm": {k: load_addon(v, addon) for k, v in arms.items()},
           "baseline": {k: load_addon(v, addon) for k, v in base.items()}}
    _check_addon(add)
    cols = ["arm", "kind", "assay", "rung", "layer", "pair_class", "increment_over", "n_seeds"]
    cols += [f"{k}.{x}" for k in EXPLAINED_KEYS for x in ("mean", "spread", "n_defined")]
    cols += ["n_pairs", "n_cells", "flag"]
    rows = []
    for kind, group in add.items():
        for name, xs in group.items():
            xs = [x for x in xs if x is not None]
            if not xs:
                rows.append({"arm": name, "kind": kind, "n_seeds": 0, "flag": NOT_POPULATED})
                continue
            per = [_get(x, ("explained_fraction_per_layer", "per_assay")) or {} for x in xs]
            for assay in sorted({a for p in per for a in p}):
                blocks = [p.get(assay) or {} for p in per]
                keys, seen = [], {}
                for i, blk in enumerate(blocks):
                    for row in blk.get("rows") or []:
                        key = (row.get("rung", ""), row.get("layer", ""), row.get("pair_class", ""))
                        if key not in seen:
                            keys.append(key)
                            seen[key] = row.get("increment_over", "")
                if not keys:
                    why = sorted({blk.get("reason", "") for blk in blocks if blk.get("reason")})
                    rows.append({"arm": name, "kind": kind, "assay": assay, "n_seeds": len(xs),
                                 "flag": "; ".join(why) or "no row"})
                    continue
                for key in keys:
                    got = [next((rw for rw in blk.get("rows") or []
                                 if (rw.get("rung", ""), rw.get("layer", ""),
                                     rw.get("pair_class", "")) == key), None) for blk in blocks]
                    r = {"arm": name, "kind": kind, "assay": assay, "rung": key[0], "layer": key[1],
                         "pair_class": key[2], "increment_over": seen[key], "n_seeds": len(xs)}
                    for k in EXPLAINED_KEYS:
                        st = stats([(g or {}).get(k) for g in got])
                        r.update({f"{k}.mean": st["mean"], f"{k}.spread": st["spread"],
                                  f"{k}.n_defined": st["n_defined"]})
                    for k in ("n_pairs", "n_cells"):
                        v = [_num((g or {}).get(k)) for g in got]
                        v = [x for x in v if x is not None]
                        r[k] = int(max(v)) if v else None
                    r["flag"] = "; ".join(sorted({g.get("flag", "") for g in got
                                                  if g and g.get("flag")}))
                    rows.append(r)
    return rows, cols


# ------------------------------------------------------------------------------------------ dev evals

def dev_evals(runs, prefix, baselines=None, sub="", allow_mix=()) -> tuple[list[dict], list[str]]:
    arms, base = _load_groups(runs, prefix, baselines, sub, allow_mix)
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


def accessibility(runs, prefix, baselines=None, sub="", allow_mix=()) -> tuple[list[dict], list[str]]:
    arms, base = _load_groups(runs, prefix, baselines, sub, allow_mix)
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

def per_knob(runs, prefix, baselines=None, sub="", allow_mix=()) -> tuple[list[dict], list[str]]:
    arms, base = _load_groups(runs, prefix, baselines, sub, allow_mix)
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


def shared_specific(runs, arms: list[str], baselines=None, sub="",
                    allow_mix=()) -> tuple[list[dict], list[str]]:
    if len(arms) < 2:
        raise Refused("--arms needs at least two arms: the reference first")
    groups, trained = {}, {}
    for a in arms:
        p, is_b = _resolve(a, runs, baselines)
        seeds = load_baseline(p, sub) if is_b else load_scores(p, sub)
        if not seeds:
            raise Refused(f"arm {a!r}: no scored seed under {p}")
        groups[a] = seeds
        if not is_b:
            trained[a] = seeds
    check_evalset(groups)
    check_arm_keys(trained, allow_mix)
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
        p.add_argument("--sub", default="", type=_SR.sub_name)
        p.add_argument("--allow-mix", action="append", default=[], choices=_SR.MIXABLE)
        if name == "headline":
            p.add_argument("--addon", default=ADDON)
            p.add_argument("--pairs", default=None)
    p = sp.add_parser("explained")
    p.add_argument("--runs", required=True)
    p.add_argument("--prefix", default="")
    p.add_argument("--baselines", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--sub", default="", type=_SR.sub_name)
    p.add_argument("--allow-mix", action="append", default=[], choices=_SR.MIXABLE)
    p.add_argument("--addon", default=ADDON)
    p = sp.add_parser("shared-specific")
    p.add_argument("--runs", required=True)
    p.add_argument("--arms", required=True, help="comma-separated; the first is the reference")
    p.add_argument("--baselines", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--sub", default="", type=_SR.sub_name)
    p.add_argument("--allow-mix", action="append", default=[], choices=_SR.MIXABLE)
    a = ap.parse_args(argv)
    fn = {"headline": headline, "dev-evals": dev_evals, "accessibility": accessibility,
          "per-knob": per_knob}
    try:
        if a.cmd == "shared-specific":
            rows, cols = shared_specific(a.runs, [x for x in a.arms.split(",") if x], a.baselines,
                                         a.sub, allow_mix=tuple(a.allow_mix))
        elif a.cmd == "headline":
            rows, cols = headline(a.runs, a.prefix, a.baselines, a.sub, a.addon, a.pairs,
                                  allow_mix=tuple(a.allow_mix))
        elif a.cmd == "explained":
            rows, cols = explained(a.runs, a.prefix, a.baselines, a.sub, a.addon,
                                   allow_mix=tuple(a.allow_mix))
        else:
            rows, cols = fn[a.cmd](a.runs, a.prefix, a.baselines, a.sub,
                                   allow_mix=tuple(a.allow_mix))
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    write_tsv(rows, cols, a.out)
    if a.out and a.out != "-":
        print(f"{a.cmd}: {len(rows)} rows -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
