"""E6 -- the onion scorer: a miniCANDI run or a training-free baseline -> `scores.json` (pinned interface D).

    python tools/onion_eval/score.py (--run <run dir> | --baseline <name>) --evalset <json> --out <dir>
        [--split val|blind] [--allow-blind] [--repeat-blind-reason "<text>"] [--no-halves]
        [--dnase-route hotspot2_density|macs2_no_control] [--device cpu] [--n-boot 200]
        [--guardrail-windows 256] [--force] [--infer-batch N] [--no-region-cache] [--addon]
    -> <out>/scores.json, <out>/suffstats.npz, <out>/SCORED (SCORED last; an existing SCORED skips
       the job unless --force). The scorer never reads or writes DONE: training's DONE in a run dir
       means trained, not scored (finding 3). A pair-only predictor (`depth_only_scaling`,
       `pseudoreplicate_floor`) on an eval set with no correction or replicate pair on the split
       writes only <out>/SKIPPED (the reason) and exits 0 (F3, DEFECTS item 4).

Cost (F3). Each tile read inside the current evalset region is a slice of one whole-region read
per (measurement, file) and of one region of DNA (`--no-region-cache` restores tile reads; the
scores are identical). `--infer-batch` sets the windows per forward pass while scoring targets.
`timing_seconds` in scores.json splits the wall time by phase.

A baseline is scored into `eval/<tables>/baselines/<name>/seed0/` so `select_rule.py table` reads it
like a run. Baselines: `marginal_nb`, `position_average`, `depth_only_scaling` (E4, the three
training-free baselines of D14) and `pseudoreplicate_floor` (the half-against-half floor of headline
4; a reference, not a D14 baseline). By definition depth-only scores a correction gap closed of
exactly 0, the floor exactly 1, and marginal NB an uncertainty term of exactly 0.

What is scored (validation split by default):
  imputation targets  per regime (`leave_one_assay_out` is the headline; `accessibility_only`,
                      D11b.2, is the second table under the key `accessibility_only`). The ChIP-seq
                      control is never an imputation target; when the model predicts the control
                      (`predict_control`), the evalset's `control_targets` are scored in a separate
                      table under `predicted_control` (finding 16)
  correction pairs    source pr1 -> target pr2 (PI ruling 6); model, depth-only scaling and the
                      floor (target pr1 -> target pr2, one fitted NB size) all predict the same pr2
  replicate pairs     the evalset's replicate-ladder pairs of the scored split, same layout
  copy-leak           the guardrail (`guardrail.copy_leak`) on validation reconstruction
                      windows, over 3 mask seeds; each mask seed draws its own random sample of
                      the reconstruction items (`guardrail_items`, F3) with its own bin masks, so
                      the spread across mask seeds includes window sampling. `copy_leak_pass` (the hard gate, D13) is true only
                      when every mask seed passes part 1 of the PI's definition: perturbing hidden
                      bins or unavailable tracks moves no output (finding 2). Part 2 of the PI's
                      definition is this scorer's rule: same-view reconstruction is scored on
                      hidden bins only, denoising and correction against an independent draw.
                      The same-view copy gain (ours) is written beside it and does not gate:
                      `same_view_copy_gain`, `same_view_copy_gain_relative` (max over mask seeds),
                      `..._relative_by_mask_seed`, `..._relative_spread` (max - min over mask
                      seeds) and `same_view_copy_gain_flag` (relative gain above the threshold on
                      any mask seed). With `guardrail.GATE_ON_COPY_GAIN` True the gain gates too,
                      and `same_view_copy_gain_spread_ok` (spread at most SPREAD_MAX_FRACTION x
                      the threshold, addendum) must hold or select_rule escalates

Blind discipline (gate G4; finding 9 and 14). `--split blind` needs `--allow-blind`, an eval set
built with `--include-blind` (pinned name `evalset_blind.json`) and `--out` ending in `/blind`
(`<runs>/<arm>/seed<k>/blind`, `eval/<tables>/baselines/<name>/seed0/blind`). Blind rows are
accepted only when `split == blind`. Every blind pass is appended to the append-only ledger
`BLIND_PASSES.tsv` beside the eval set (time, run, evalset sha256, tables sha256, git sha, out,
status, repeat reason); a second pass for the same (run, eval set) is refused unless
`--repeat-blind-reason "<text>"` is given, and the reason is recorded.

Tables check (finding 9). The eval set's `tables_sha256` must equal `minicandi.dataset.tables_sha256`
of the tables on disk, and on validation the run checkpoint's `tables_sha256` too (the blind pass
reads eic_v2, a superset of the training tables, so there the checkpoint's value is recorded as
`run_tables_sha256`, not required equal).

Pseudoreplicate halves (finding 11). A target without halves (`has_pseudoreps` not 1, or a
`store:` path) gets noise floor N = Undefined with the reason, is kept out of headline 1 and counted
(`n_targets_without_halves`). Headline 1 is Undefined unless E_sp, V_sp and N are all defined on
every included target (finding 6). `--no-halves` (the pilot on `store:` tables): no half is ever
read; `specific_variance_recovered_above_noise` is null with the reason, and headline 1 without a
noise floor is written under `specific_variance_recovered_NO_NOISE_FLOOR_pilot_only`;
`noise_floor = "none_pilot"`, which select_rule refuses. The pilot also gets the uncertainty term
over every assay with a target under `uncertainty_gap_closed_all_assays_pilot_only` (never a
selection input; F3, DEFECTS item 3).

Noise floor N in counts_log1p (TVf). Each half is rescaled to the target's depth by 1 / its real
share of the reads (the `depth_reads` attributes of pr1.h5 and pr2.h5; 0.5 each without them), and N
uses the unequal-length Spearman-Brown form of `metrics.noise_floor(shares=...)`. The error budget
still takes the halves rescaled by 2.0. Headline 1's `per_target` also carries `depth_reads` (the
target's depth: a log1p layer of layers.tsv is comparable with it only at that depth) and
`N_log1p_bias_sim` (`metrics.noise_floor_log1p_bias` at the target's mean count: the simulated bias of
N; a diagnostic, never applied).

Headline 1 per assay (F3, DEFECTS item 5). `selection.headline1_per_assay_status` (also under
headline 1's `per_assay_status`) says per assay: the targets, those with a noise floor, whether
headline 1 is defined and why not, whether the assay is in the composite's imputation term, and
the fallback used. No fallback value exists: an assay outside the scorable set (PI ruling 4) never
enters the term; a scorable assay that is undefined makes the term undefined. The definition is
written beside the term (`selection.imputation_term_definition`).

Every metric is computed by the merged library (`metrics.py`, `error_budget.py`), once per target
track over all of its bins (addendum): the scorer streams the predictor region by region and tile by
tile, assembles each target's vectors over the evalset regions (whole 2^14-bin blocks, so the Haar
transform never mixes two regions), computes that target's scalars, and keeps the vectors only
while its assay group is open (headline 1 and the error budget split shared from specific across the
targets of one assay). `suffstats.npz` keeps the sufficient statistics: the Gram matrices per Haar
scale (shared and specific parts of V, P, C, E), the PIT curves (F_bar on the 21-point grid, with bin
counts, per target and stratum), and the CRPS sums per target and pair.

Space of headline 1 and 2 for selection: always `counts_log1p` (ruling 13; there is no option).
`signal_arcsinh` is computed through `signal_from_counts.signal_from_nb` (X2), imported lazily, with
the dataset, the region list and the region length; a ChIP-seq target without an observed control
uses the model's predicted control (a second forward pass with the control channel asked for at
the target's own C'). `lambda_source` per target is written under `per_target`. `--dnase-route`
selects the DNase-seq route (default `macs2_no_control`, PI ruling 19: the EIC DNase-seq truth is a
MACS2 track; `hotspot2_density` is the native DNase-seq view). The headline signal is the expectation
of the MACS2 score under the predicted NB (`signal_arcsinh`); the plug-in variant, the score of the NB
mean (PI ruling 20), is written beside it under the same keys with the suffix `_plug_in`
(`signal_arcsinh_plug_in` in headline 1 and in r_spec). Without the module the signal space is NaN
with the reason.

Options of chunk SC (each off by default; a default scores.json is unchanged):
  --lambda-source auto|observed_control|predicted_control. `auto` is the route above.
      `observed_control` never uses the predicted control (a ChIP-seq target without an observed
      control is undefined in the signal space, with the reason). `predicted_control` keeps the
      `auto` keys and also scores every ChIP-seq target with the model's predicted control, even
      where an observed control exists, under `signal_arcsinh_predicted_control` (headline 1 and
      r_spec; the expectation mode; the training average `a` is the `auto` one); per target it
      records `lambda_source`. The option is written as `lambda_source_option`.
  score(..., signal_override=f) (chunk XE): an external entry. f(target row, regions) gives the
      entry's signal per region; it replaces the derived signal in the `signal_arcsinh` path (the
      plug-in variant is undefined), and every count-space metric is undefined with the reason
      `external entry: signal only`. No pair is scored.
  A correction pair flagged `read_disjoint: 0` by evalset.py (two sides can share reads) is never
      scored; their count is `correction_pairs_not_read_disjoint`.
  A blind eval set may be `evalset_blind.json` or `evalset_blind_<tag>.json`; the scorer reads
      `include_blind` from the eval set, not its name.

Add-on scoring (OXe, Amendment 4, PI ruling 33). `--addon` scores an add-on eval set
(`evalset.py addon`, key `addon_of`) and is required for one; it is refused on any other eval set
and on the blind split. The form is `score.sh $K $EVX <seed dir>/ox run <seed dir> --addon`: the
add-on score goes to `<seed dir>/ox/scores.json` and never over a `scores.json` of another eval set
(an output dir whose `scores.json` has another `evalset_sha256` is refused). On validation the
run's `tables_sha256` must equal `build_info.json::addon_of_tables_sha256` of the add-on tables (the
tables the add-on extends). A target flagged `evaluation_only` (a second experiment) is predicted
with its EIC track's inputs and enters only `explained_fraction_per_layer` (the experiment row needs
both members of the pair predicted) and the `per_target` blocks of headline 1 and headline 7, with
`"evaluation_only": true`; no headline block, no `selection` term and no `n_targets` count reads it,
and it stays out of the shared/specific split, the error budget and the signal space of its assay.
The copy-leak draw leaves out every reconstruction item of a flagged target. The explained
fraction also reads the isogenic and anisogenic pairs of the scored split from `pairs.tsv` (they are
the term the experiment layer subtracts; the eval set lists only the other_experiment pair ids, so
they are never pair jobs). Headline 7 in an add-on score: `per_target` gives `per_rung_reason`
(the reason of every Undefined ratio, the upper-bound rung included when it is not populated) and
the block gains `n_undefined_upper_bound` (EIC targets whose upper bound is Undefined). Without
`--addon` every output is unchanged.

Per-metric aggregation. Each metric is {"macro", "per_assay", "ci", "non_scorable_per_assay",
"missing_scorable", "reason"}: per assay over that assay's targets (headline 1 and 4 as ratios of
means, the others as means), macro = mean over the scorable assays (evalset `scorable`), undefined
when any scorable assay is undefined, and in the leave-one-assay-out headline and the correction
metrics also when a scorable assay has no target (finding 10). `ci` is a 95% cluster bootstrap over
cells (targets resampled with their cell; the per-target components are held fixed, so headline 1's
shared estimate is not refitted per draw).

Selection denominators (finding 4). `selection.denominators` holds the marginal-NB PIT deviation
(uncertainty) and, per correction-scorable assay, (depth-only CRPS - floor CRPS) / depth-only CRPS,
with its minimum; select_rule escalates a term whose denominator is under its tolerance. The goal
terms and `composite` here are raw; select_rule clips each term to [0, 1] inside the composite (PI
ruling 18) and keeps the raw terms for the per-goal veto.

Baseline fitting (agent decisions, logged): marginal NB per assay and the NB size of depth-only
scaling and of the floor are fitted on the evalset's `fit_regions` (never a scored bin); the
position average's leave-one-cell-out table and its NB size are fitted per (assay, region) on the
training cells' base views of that region (the table must cover the scored bins and the genome-wide
table does not fit in memory), so its NB size is one per assay per region. The position average is
also the training average `a` of r_spec.

Training-average control (chunk F4, foreman call logged in the notepad): in the signal space, the
position average of a ChIP-seq target with no observed control takes its MACS2 lambda from the
average of the training controls, i.e. the position average of the `chipseq-control` track,
fitted like every position average (the training-track base views, `train_base`, so a model trained
on train+val uses its validation controls too; never a blind control), leave-one-cell-out, scaled
to the target's own C' depth. Such a record carries `lambda_source_training_average =
training_average_control` (and, under `per_target`, the same key); when the position average is the
main predictor its `lambda_source` is `training_average_control`. Targets with an observed control,
and ATAC-seq and DNase-seq targets, are unchanged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import zlib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from onion_eval import metrics as Mx  # noqa: E402

SPACE = "counts_log1p"                               # ruling 13: the selection space, fixed
REFS = ("marginal_nb", "position_average", "depth_only_scaling", "pseudoreplicate_floor")
TRAINING_FREE = ("marginal_nb", "position_average", "depth_only_scaling")
WEIGHTS = {"imputation": 0.5, "correction": 0.25, "uncertainty": 0.25}
REGIMES = ("leave_one_assay_out", "accessibility_only")
REPLICATE_RUNGS = ("pseudoreplicate", "sequencing", "technical", "isogenic", "anisogenic",
                   "other_experiment", "other_cell_type")
#: headline 7 upper bound (PI ruling 31): the rung whose pair shares only what the model knows. An
#: imputed target: another experiment of the same cell type. A reconstructed target (its own reads at
#: other bins are an input) and a covariate-corrected target (source pr1 -> target pr2 of one track) share
#: every layer down to the sampling term with their input, so their upper-bound rung is the
#: pseudoreplicate rung itself: both bounds meet and a calibrated model scores 1 there. score.py scores
#: headline 7 on imputed targets only.
H7_UPPER_RUNG = {"imputation": "other_experiment", "reconstruction": "pseudoreplicate",
                 "correction": "pseudoreplicate"}
#: explained fraction per layer (PI ruling 32): rungs whose members are predicted (outer) and rungs within one
#: experiment whose prediction difference is 0 for an imputed target (zero)
EXPLAINED_OUTER = ("other_experiment", "other_cell_type")
EXPLAINED_ZERO = ("isogenic", "anisogenic")
EXPLAINED_RUNGS = EXPLAINED_OUTER + EXPLAINED_ZERO
H7_DESCRIPTION = ("lower-bound check: mean predicted NB variance over the pair value of the target's own "
                  "pseudoreplicate halves (the sampling term); a calibrated model scores >= 1 here. "
                  "per_rung gives the ratio at every populated rung; pass = ratio >= 1 at "
                  "pseudoreplicate and <= 1 at upper_bound_rung (PI ruling 31). PIT stays the "
                  "selection metric.")
POOLED = "__pooled__"
#: OXe: why an evaluation-only target (a second experiment) has no assay-level value
FLAGGED_REASON = ("evaluation-only target (a second experiment, PI ruling 33): outside the "
                  "shared/specific split, the error budget and every headline block")
BUDGET_KEYS = ("level", "shared", "specific", "explained", "unexplained", "noise", "added")
CONTROL = "chipseq-control"
NOISE_FLOOR_HALVES = "pseudoreplicate_halves"
NOISE_FLOOR_NONE = "none_pilot"
PILOT_H1_KEY = "specific_variance_recovered_NO_NOISE_FLOOR_pilot_only"
PILOT_UNC_KEY = "uncertainty_gap_closed_all_assays_pilot_only"
#: the composite's imputation term (D13, plan E6, PI ruling 4), written beside it in scores.json
IMPUTATION_TERM_DEFINITION = (
    "macro mean, over the eval set's imputation-scorable assays (>= 3 validation tracks with "
    "pseudoreplicate halves from >= 3 cells, evalset.py), of headline 1 per assay in counts_log1p "
    "(leave-one-assay-out, validation). It is undefined when any scorable assay is undefined or has "
    "no target. A non-scorable assay never enters it, so it needs no fallback; its headline 1 is "
    "reported apart, per assay, with its floor status. No floor and no value is ever invented.")
#: same-view copy gain: its relative spread across mask seeds must be at most this fraction of
#: the threshold ("well under", addendum), else `same_view_copy_gain_spread_ok` is false; it
#: matters only when guardrail.GATE_ON_COPY_GAIN is True
SPREAD_MAX_FRACTION = 0.5
BLIND_LEDGER = "BLIND_PASSES.tsv"
LEDGER_COLUMNS = ("time", "run", "evalset_sha256", "tables_sha256", "git_sha", "out", "status",
                  "repeat_reason")
SCORED = "SCORED"
#: a pair-only predictor with no pair to score writes this (the reason) instead of NaN rows
SKIPPED = "SKIPPED"
PAIR_ONLY = ("depth_only_scaling", "pseudoreplicate_floor")
#: NB -> signal modes (PI ruling 20): record-key prefix of each; the headline is `expectation`
SIGNAL_PREFIX = {"expectation": "sig_", "plug_in": "sig_pi_"}
#: suffix of the signal-space keys that hold the plug-in variant
PLUG_IN = "_plug_in"
#: --lambda-source (SC): `auto` is the behavior before SC and the default
LAMBDA_SOURCE_OPTIONS = ("auto", "observed_control", "predicted_control")
#: record prefix and key suffix of the `predicted_control` signal route (every ChIP target)
PC_PREFIX = "sig_pc_"
PREDICTED_CONTROL = "_predicted_control"
#: lambda source of the position average on a ChIP target with no observed control (F4)
TRAINING_AVERAGE_CONTROL = "training_average_control"
#: an external entry (chunk XE): the main predictor gives a signal only, through signal_override
EXTERNAL = "external"
EXTERNAL_REASON = "external entry: signal only"
EXTERNAL_PLUG_IN_REASON = ("external entry: signal only (no NB prediction, so no plug-in "
                           "variant)")


class ScoreRefused(Exception):
    """The scorer may not run on these inputs (blind discipline, tables mismatch, layout)."""


# ------------------------------------------------------------------------------------------ helpers

def _undef(x) -> bool:
    return x is None or not np.isfinite(float(x))


def _j(x):
    """JSON value: finite float, or None for an undefined one."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _why(x) -> str:
    return getattr(x, "reason", "") if x is not None else "missing"


def composite(imp, cor, unc):
    """D13: 0.5 imputation + 0.25 correction + 0.25 uncertainty; Undefined if a term is."""
    terms = {"imputation": imp, "correction": cor, "uncertainty": unc}
    bad = [k for k, v in terms.items() if _undef(v)]
    if bad:
        return Mx.Undefined("undefined goal term: " + ", ".join(bad))
    return float(sum(WEIGHTS[k] * float(v) for k, v in terms.items()))


def _macro(per: dict):
    if not per:
        return Mx.Undefined("no scorable assay with a target")
    bad = [a for a, v in per.items() if _undef(v)]
    if bad:
        return Mx.Undefined("undefined for " + ", ".join(f"{a} ({_why(per[a])})" for a in bad))
    return float(np.mean([float(v) for v in per.values()]))


def _boot(records, agg, n_boot, seed):
    cells = sorted({r["cell"] for r in records})
    if len(cells) < 2 or n_boot <= 0:
        return [None, None]
    by = defaultdict(list)
    for r in records:
        by[r["cell"]].append(r)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(len(cells), len(cells), replace=True)
        v = agg([r for i in pick for r in by[cells[i]]])
        if not _undef(v):
            vals.append(float(v))
    if len(vals) < max(10, n_boot // 2):
        return [None, None]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def metric(records, fn, scorable, n_boot=200, seed=0, undefined_reason=None,
           require_all=False) -> dict:
    """{"macro", "per_assay", "ci", "non_scorable_per_assay", "missing_scorable", "n", "reason"}
    from per-target records. `require_all`: the macro is Undefined when a scorable assay has no
    record (finding 10); `missing_scorable` lists such assays in every case."""
    if undefined_reason:
        return {"macro": None, "per_assay": {}, "ci": [None, None], "non_scorable_per_assay": {},
                "missing_scorable": [], "n": 0, "reason": undefined_reason}
    scorable = set(scorable)

    def per_assay(recs):
        by = defaultdict(list)
        for r in recs:
            by[r["assay"]].append(r)
        return {a: fn(rs) for a, rs in sorted(by.items())}

    def agg(recs):
        return _macro({a: v for a, v in per_assay(recs).items() if a in scorable})

    per = per_assay(records)
    sc = {a: v for a, v in per.items() if a in scorable}
    missing = sorted(scorable - set(per))
    macro = _macro(sc)
    if require_all and missing and not _undef(macro):
        macro = Mx.Undefined("scorable assay with no target in this split and regime: "
                             + ", ".join(missing))
    return {"macro": _j(macro), "per_assay": {a: _j(v) for a, v in sc.items()},
            "ci": _boot(records, agg, n_boot, seed) if not _undef(macro) else [None, None],
            "non_scorable_per_assay": {a: _j(v) for a, v in per.items() if a not in scorable},
            "per_assay_reason": {a: _why(v) for a, v in per.items() if _undef(v)},
            "missing_scorable": missing,
            "n": len(records), "reason": _why(macro) if _undef(macro) else ""}


def _mean_of(key):
    def fn(rs):
        v = [r[key] for r in rs if not _undef(r[key])]
        if len(v) < len(rs):
            bad = [r for r in rs if _undef(r[key])]
            return Mx.Undefined(f"{len(bad)} of {len(rs)} targets undefined: {_why(bad[0][key])}")
        return float(np.mean(v))
    return fn


def recovered_fn(prefix: str = "", no_floor: bool = False):
    """Headline 1 of one assay's records (findings 6 and 11). Targets without pseudoreplicate halves
    are left out; every included target needs E_sp, V_sp and N defined, else Undefined with the
    reasons. `no_floor`: the pilot form, N = 0 for every target."""
    kE, kV, kN = f"{prefix}E_sp", f"{prefix}V_sp", f"{prefix}N"

    def fn(rs):
        inc = rs if no_floor else [r for r in rs if r.get("has_halves", True)]
        if not inc:
            return Mx.Undefined(f"no target with pseudoreplicate halves ({len(rs)} without)")
        keys = (kE, kV) if no_floor else (kE, kV, kN)
        bad = sorted({f"{k}: {_why(r[k])}" for r in inc for k in keys if _undef(r[k])})
        if bad:
            return Mx.Undefined("; ".join(bad))
        N = [0.0] * len(inc) if no_floor else [r[kN] for r in inc]
        return Mx.specific_variance_recovered([r[kE] for r in inc], [r[kV] for r in inc], N)
    return fn


def pit_curve(n, mu, y):
    from candi.bench.distributional import p_from_mu
    from candi.metrics import PIT_GRID, calibration_pit_curve
    _, fbar = calibration_pit_curve(n, p_from_mu(n, mu), y, PIT_GRID)
    return np.asarray(fbar, np.float64)


def pooled_pit_deviation(curves):
    """Mean |F_bar(u) - u| of the bin-weighted mean of several F_bar curves [(fbar, count)].
    Exactly `metrics.pit_deviation(...)["pooled"]` on the concatenated bins (F_bar is a bin mean)."""
    from candi.metrics import PIT_GRID
    curves = [(f, c) for f, c in curves if c > 0]
    if not curves:
        return Mx.Undefined("no bins")
    w = np.array([c for _, c in curves], np.float64)
    f = np.stack([f for f, _ in curves]) * w[:, None]
    return float(np.mean(np.abs(f.sum(0) / w.sum() - np.asarray(PIT_GRID))))


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_sha() -> str:
    try:
        return subprocess.run(["git", "-C", str(_TOOLS), "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:  # pragma: no cover - no git on the node
        return "unknown"


def _half_shares(row: dict) -> tuple:
    """(f1, f2): the real shares of the target's reads in pr1.h5 and pr2.h5 (their `depth_reads`
    attributes, via `layers.half_depths`); (0.5, 0.5) when the attributes are missing."""
    from onion_eval.layers import half_depths
    h = half_depths(row)
    if h is None:
        return (0.5, 0.5)
    d1, d2 = h[0][1], h[1][1]
    return (d1 / (d1 + d2), d2 / (d1 + d2))


def has_halves(row: dict) -> bool:
    """The row's pseudoreplicate halves exist on disk (a `store:` path never has them)."""
    return (str(row.get("has_pseudoreps", "")).strip() in ("1", "true", "True")
            and str(row.get("counts_path", "")).startswith("h5:"))


#: how the copy-leak diagnostic draws its windows per mask seed (F3, DEFECTS item 9)
GUARD_DRAW = ("per mask seed s: a seeded random sample (rng [s, 9]) of the validation "
              "reconstruction items, bin masks from rng [s, item, 3]")


def guardrail_items(ds, n: int, seed: int, exclude=()) -> tuple[list, int]:
    """(`n` validation reconstruction items drawn at random with mask seed `seed`, in item order;
    the pool size). The old draw took the first `n` items for every seed:
    the same windows of one target, with only the bin masks changing, so its spread across mask
    seeds (0.003-0.013 in the pilot) missed the window-sampling noise (0.045-0.12 across train-split
    seeds). Masks come from the dataset seed, so every seed also gets independent masks."""
    pool = [it for it in ds.evalset_items() if it["task"] == "reconstruction"]
    if exclude:              # OXe: never a second experiment in the copy-leak draw
        ex = set(exclude)
        pool = [it for it in pool if not (set(it["targets"]) | set(it["inputs"])) & ex]
    k = min(int(n), len(pool))
    pick = np.sort(np.random.default_rng([int(seed), 9]).choice(len(pool), k, replace=False))
    return [pool[i] for i in pick], len(pool)


def _signal_module():
    try:
        from onion_eval import signal_from_counts as S  # noqa: F401  (X2, lazy)
    except ImportError as e:
        return None, f"tools/onion_eval/signal_from_counts.py (X2) not importable: {e}"
    if not hasattr(S, "signal_from_nb"):
        return None, "signal_from_counts has no signal_from_nb(n, mu, row, *, ds, regions, rb)"
    return S, ""


# ------------------------------------------------------------------------------------------ predictors

class ModelPredictor:
    def __init__(self, model, device="cpu"):
        self.model, self.device = model.to(device).eval(), device

    def __call__(self, batch):
        b = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.no_grad():
            o = self.model(b)
        return {k: o[k].detach().cpu().double().numpy() for k in ("n", "mu")}


def _np_out(o):
    return {k: (o[k].detach().cpu().double().numpy() if torch.is_tensor(o[k])
                else np.asarray(o[k], np.float64)) for k in ("n", "mu")}


# ------------------------------------------------------------------------------------------ scorer

class Scorer:
    def __init__(self, evalset, *, main, model=None, config=None, split="val", allow_blind=False,
                 n_boot=200, seed=0, guardrail_windows=256, guardrail_seeds=(0, 1, 2),
                 device="cpu", no_halves=False, dnase_route=None, run_tables_sha256=None,
                 log=print, region_cache=True, infer_batch=None, lambda_source="auto",
                 signal_override=None, addon=False, dump_signal=None, dump_targets=None,
                 dump_method=None):
        from minicandi import contract as C
        from minicandi.dataset import OnionDataset, tables_sha256
        from onion_eval.evalset import tables_dir_of
        if split not in ("val", "blind"):
            raise ValueError("split must be val or blind")
        if split == "blind" and not allow_blind:
            raise PermissionError("scoring blind targets needs --allow-blind (gate G4)")
        if lambda_source not in LAMBDA_SOURCE_OPTIONS:
            raise ValueError(f"lambda_source {lambda_source!r} not in {LAMBDA_SOURCE_OPTIONS}")
        if (signal_override is not None) != (main == EXTERNAL):
            raise ValueError(f"main {EXTERNAL!r} goes with a signal_override, and only it")
        if signal_override is not None and model is not None:
            raise ValueError("signal_override replaces the model: give no model")
        if main not in REFS and model is None and main != EXTERNAL:
            raise ValueError(f"main {main!r}: give a model or one of {REFS}")
        self.lambda_source = lambda_source
        self.signal_override = signal_override
        self.external = signal_override is not None
        self.n_pairs_not_read_disjoint = 0
        self.C, self.log = C, log
        self.evalset_path = Path(evalset).resolve()
        self.ev = json.loads(self.evalset_path.read_text(encoding="utf-8"))
        if split == "blind" and not self.ev.get("include_blind"):
            raise ScoreRefused(f"{self.evalset_path}: not a blind eval set (built without "
                               f"--include-blind)")
        self.tables = tables_dir_of(self.ev, self.evalset_path)
        self.tables_sha256 = tables_sha256(self.tables)
        if not self.ev.get("tables_sha256"):
            raise ScoreRefused(f"{self.evalset_path}: no tables_sha256 (rebuild the eval set)")
        if self.ev["tables_sha256"] != self.tables_sha256:
            raise ScoreRefused(f"tables {self.tables} changed since the eval set was built "
                               f"(sha256 {self.tables_sha256[:12]} vs {self.ev['tables_sha256'][:12]})")
        self.run_tables_sha256 = run_tables_sha256
        self.addon = bool(addon)
        self.addon_of_tables_sha256 = None
        if self.ev.get("addon_of") and not self.addon:
            raise ScoreRefused(f"{self.evalset_path}: an add-on eval set (addon_of); score it with "
                               f"--addon into <seed dir>/ox")
        if self.addon:
            if not self.ev.get("addon_of"):
                raise ScoreRefused(f"--addon: {self.evalset_path} is not an add-on eval set (no "
                                   f"addon_of; build it with evalset.py addon)")
            if split != "val":
                raise ScoreRefused("--addon scores the validation split only")
            from onion_eval.evalset import addon_info
            self.addon_of_tables_sha256 = addon_info(self.tables).get("addon_of_tables_sha256")
            if not self.addon_of_tables_sha256:
                raise ScoreRefused(f"{self.tables}/build_info.json: no addon_of_tables_sha256 (not "
                                   f"add-on tables)")
        #: second experiments flagged evaluation_only (add-on eval set; empty otherwise)
        self.flagged = {t["target"] for t in self.ev["imputation_targets"]
                        if t.get("evaluation_only")} if self.addon else set()
        want_run = self.addon_of_tables_sha256 if self.addon else self.tables_sha256
        if model is not None and split == "val" and run_tables_sha256 != want_run:
            raise ScoreRefused(f"the run was trained on tables sha256 {str(run_tables_sha256)[:12]}, "
                               + (f"the add-on tables extend {want_run[:12]} "
                                  f"(build_info.json::addon_of_tables_sha256)" if self.addon else
                                  f"the eval set's tables are {self.tables_sha256[:12]}"))
        # finding 14: blind rows only on a blind pass
        self.split, self.allow_blind, self.n_boot, self.seed = split, split == "blind", n_boot, seed
        self.no_halves = bool(no_halves)
        self.main, self.model_pred = main, (ModelPredictor(model, device) if model is not None else None)
        self.device = device
        self.guardrail_windows, self.guardrail_seeds = guardrail_windows, tuple(guardrail_seeds)
        self.model = model
        self.rb = int(self.ev.get("region_bins", 2 ** 14))
        cfg = dict(config or {})
        L = int(cfg.get("context_bins") or min(1024, self.rb))
        if self.rb % L:
            raise ValueError(f"region_bins {self.rb} is not a multiple of context_bins {L}")
        cfg.update({"context_bins": L, "evalset": str(self.evalset_path), "region_bins": self.rb,
                    "allow_blind": self.allow_blind})
        cfg.setdefault("batch_size", 8)
        self.cfg, self.L, self.bs = cfg, L, int(cfg["batch_size"])
        #: windows per forward pass while scoring targets (F3); the guardrail keeps batch_size
        self.tile_bs = int(infer_batch or self.bs)
        self._OnionDataset = OnionDataset
        self.ds = OnionDataset(self.tables, split, cfg, seed=0)
        self.predict_control = bool(self.ds.config.get("predict_control")) and model is not None
        self.regions = [(c, int(s)) for c, s in self.ev["regions"]]
        fr = [(c, int(s)) for c, s in self.ev.get("fit_regions", [])]
        self.fit_on = "fit_regions" if fr else "eval_regions (no fit_regions in the evalset)"
        self.fit_regions = fr or self.regions
        for c, s in self.regions + self.fit_regions:
            if s + self.rb > self.ds.n_bins.get(c, 0):
                raise ValueError(f"region {c}:{s} runs past the chromosome end")
        self.strata = self._strata()
        self.sig, self.sig_reason = _signal_module()
        if self.sig is not None:
            self.dnase_route = dnase_route or self.sig.DNASE_ROUTE_DEFAULT
            if self.dnase_route not in self.sig.DNASE_ROUTES:
                raise ValueError(f"dnase_route {self.dnase_route!r} not in {self.sig.DNASE_ROUTES}")
        else:
            self.dnase_route = dnase_route
        self._pa_cache: dict = {}
        self._pa_ctl_rate: dict = {}               # (region index, cell) -> average control rate
        self.suff: dict = {}
        self._rung_pair_index = None              # (assay, cell) -> rung -> pairs (headline 7)
        self.timing: dict = defaultdict(float)     # wall seconds per phase, in scores.json
        # F3 region cache: a tile read inside the current evalset region is a slice of one
        # whole-region read per (measurement, file); see _cached_read
        self.region_cache = bool(region_cache)
        self._region, self._rc = None, {}
        self._ds_read, self._ds_dna = self.ds.read, self.ds.dna
        if self.region_cache:
            self.ds.read, self.ds.dna = self._cached_read, self._cached_dna
        self._init_dump(dump_signal, dump_targets, dump_method, config)

    # -------------------------------------------------------------- AGd dump hook (interface F)
    def _init_dump(self, dump_signal, dump_targets, dump_method, config) -> None:
        """Method track files (plan Amendment 5, pinned interface F): with `dump_signal`, every
        scored imputation target of each regime writes its per-bin arrays under
        `<dump_signal>/<regime>/`. Nothing else in the scorer reads them, so scores.json is the same."""
        self.dump_dir = Path(dump_signal) if dump_signal else None
        self.dump_targets = set(dump_targets) if dump_targets is not None else None
        self._dump_ledgered = False        # set by score() after the blind ledger's `started` row
        self._dump_meta: dict = {}
        self._dump_ctl = None
        self.dump_kind = ("external" if self.external else "run" if self.main == "model"
                          else "baseline")
        self.dump_method = dump_method or self.main
        self.trained_on_roles = None
        if self.dump_dir is None:
            return
        if self.dump_kind == "run":       # the run's own record of its training split
            cfg = dict(config or {})
            if "train_on_validation" not in cfg:
                raise ScoreRefused("--dump-signal: the run's config records no train_on_validation, "
                                   "so the roles it trained on (tracks.json trained_on_roles) are "
                                   "unknown")
            self.trained_on_roles = ["train", "val"] if cfg["train_on_validation"] else ["train"]
        elif self.dump_kind == "baseline":  # fitted on the scorer's own training tracks
            self.trained_on_roles = list(self.ds.train_roles)

    def _dump_wants_avg_ctl(self, job) -> bool:
        """A baseline under --lambda-source predicted_control: its predicted-control substitute is
        the training-average control (F4), here for every ChIP target with signal tracks."""
        if (self.dump_dir is None or self.dump_kind != "baseline" or self.sig is None
                or self.lambda_source != "predicted_control" or job["kind"] != "imputation"
                or job.get("regime") == "control"):
            return False
        row = self.ds.rows[job["target"]]
        try:
            route = self.sig.route_of(row["assay_class"] or row["assay"])
        except ValueError:
            return False
        return route == "chip" and self._signal_files(row)[0] is not None

    def _dump_target(self, job, regime, n, mu, A, defined, sig) -> None:
        """Write one target's arrays of interface F; record why an array is missing."""
        from onion_eval import ag2_space as AS
        mid = job["target"]
        if regime == "control" or (self.dump_targets is not None and mid not in self.dump_targets):
            return
        m = self._dump_meta.setdefault(regime, {"targets": [], "left_out": {}, "lambda_sources": {},
                                                "signal_pc_left_out": {}, "native_left_out": {},
                                                "peakprob_left_out": {}})
        if mid in self.flagged:
            m["left_out"][mid] = FLAGGED_REASON
            return
        rec = sig["record"]
        pred = sig.get(f"{SIGNAL_PREFIX['expectation']}pred")
        if pred is None:
            m["left_out"][mid] = rec.get("lambda_reason") or "no signal prediction for this target"
            return
        row = self._row(mid)
        want = (len(self.regions) * self.rb,)
        arrays = {"signal": pred}
        src = {"signal": rec.get("lambda_source")}
        if not self.external:
            arrays["truth"] = sig[f"{SIGNAL_PREFIX['expectation']}truth"]
            kw = dict(ds=self.ds, regions=self.regions, rb=self.rb)
            if self.lambda_source == "predicted_control":
                pc = sig.get(f"{PC_PREFIX}pred")
                pc_src = rec.get("lambda_source" + PREDICTED_CONTROL)
                why = rec.get("lambda_reason" + PREDICTED_CONTROL) or ""
                if pc is None and defined and A.get("dump.avg_ctl_mu") is not None:
                    try:
                        sp, _ = self.sig.signal_from_nb(
                            n, mu, dict(row, control_measurement_id=""), mode="expectation",
                            predicted_control=A["dump.avg_ctl_mu"], dnase_route=self.dnase_route,
                            **kw)
                        pc = Mx.to_space(np.asarray(sp, np.float64), "signal_arcsinh")
                        pc_src = TRAINING_AVERAGE_CONTROL
                    except (ValueError, KeyError) as e:
                        why = f"no signal from the training-average control: {e}"
                if pc is not None:
                    arrays["signal_pc"] = pc
                    src["signal_pc"] = pc_src
                else:
                    m["signal_pc_left_out"][mid] = why or "no predicted control for this target"
            if defined:
                ctl = self._dump_ctl
                try:
                    arrays["native"], src["native"] = AS.native_from_nb(
                        n, mu, row, predicted_control=ctl, **kw)
                    arrays["native_truth"], _ = AS.native_from_nb(
                        np.ones_like(A["y"]), A["y"], row, predicted_control=ctl, **kw)
                except (ValueError, KeyError) as e:
                    arrays.pop("native", None)
                    m["native_left_out"][mid] = f"no native-space track: {e}"
                try:
                    arrays["peakprob"] = AS.peak_probability_from_nb(
                        n, mu, row, predicted_control=ctl, dnase_route=self.dnase_route, **kw)
                except (ValueError, KeyError) as e:
                    m["peakprob_left_out"][mid] = f"no peak probability: {e}"
            else:
                m["native_left_out"][mid] = m["peakprob_left_out"][mid] = (
                    "the predictor is undefined on this target")
        d = self.dump_dir / regime
        d.mkdir(parents=True, exist_ok=True)
        for name, a in arrays.items():
            a = np.asarray(a, np.float32)
            if a.shape != want:
                raise ValueError(f"dump {mid}.{name}: shape {a.shape}, want {want}")
            np.save(d / f"{mid}.{name}.npy", a)
        m["targets"].append(mid)
        m["lambda_sources"][mid] = src

    def _dump_finish(self) -> None:
        """tracks.json of each regime directory (interface F keys first)."""
        from onion_eval.ag2_baselines import regions_sha256
        esha, rsha = sha256_file(self.evalset_path), regions_sha256(self.regions, self.rb)
        for regime, m in sorted(self._dump_meta.items()):
            meta = {"method": self.dump_method, "kind": self.dump_kind, "evalset_sha256": esha,
                    "regions_sha256": rsha, "split": self.split, "regime": regime,
                    "region_bins": self.rb, "n_regions": len(self.regions),
                    "lambda_source": self.lambda_source, "targets": m["targets"],
                    "space": "signal_arcsinh", "dnase_route": self.dnase_route,
                    "left_out": m["left_out"],
                    "dump_targets_given": self.dump_targets is not None,
                    "evalset": str(self.evalset_path)}
            # per track, the lambda source of each array as signal_from_nb (or the scorer) reports it
            for arr in ("signal", "signal_pc", "native"):
                key = "lambda_source_by_target" + ("" if arr == "signal" else f"_{arr}")
                by = {t: v[arr] for t, v in m["lambda_sources"].items() if arr in v}
                if arr == "signal" or by:
                    meta[key] = by
            if self.lambda_source == "predicted_control":
                meta["lambda_source_predicted_control"] = "predicted_control"
            if self.trained_on_roles is not None:
                meta["trained_on_roles"] = self.trained_on_roles
            if not self.external:
                meta["native_left_out"] = m["native_left_out"]
                meta["peakprob_left_out"] = m["peakprob_left_out"]
                if self.lambda_source == "predicted_control":
                    meta["signal_pc_left_out"] = m["signal_pc_left_out"]
                else:
                    meta["signal_pc_reason"] = ("no signal_pc: the scorer ran with --lambda-source "
                                                f"{self.lambda_source}, not predicted_control")
            d = self.dump_dir / regime
            d.mkdir(parents=True, exist_ok=True)
            tmp = d / "tracks.json.tmp"
            tmp.write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n", encoding="utf-8")
            tmp.replace(d / "tracks.json")

    # -------------------------------------------------------------- data
    def _set_region(self, chrom, start) -> None:
        """Make (chrom, start) the cached region; the previous region's arrays are dropped."""
        if self._region != (chrom, int(start)):
            self._region, self._rc = (chrom, int(start)), {}

    def _in_region(self, chrom, start, L) -> bool:
        r = self._region
        return r is not None and chrom == r[0] and r[1] <= start and start + L <= r[1] + self.rb

    def _cacheable(self, mid) -> bool:
        """A slice of a whole-region read equals a direct read: true for h5: paths and for store:
        views that are not depth-thinned (those thin with a seed that depends on the start bin)."""
        from minicandi.dataset import view_of
        row = self.ds.check_row(mid)
        p = row["counts_path"]
        return p.startswith("h5:") or (p.startswith("store:") and view_of(row)[0] != "depth")

    def _cached_read(self, mid, chrom, start, L=None, which="counts"):
        L = self.ds.L if L is None else int(L)
        start = int(start)
        if not (self._in_region(chrom, start, L) and self._cacheable(mid)):
            return self._ds_read(mid, chrom, start, L, which)
        key = (mid, which)
        arr = self._rc.get(key)
        if arr is None:
            arr = self._rc[key] = self._ds_read(mid, chrom, self._region[1], self.rb, which)
        o = start - self._region[1]
        return arr[o:o + L].copy()

    def _cached_dna(self, chrom, start, L=None):
        L = self.ds.L if L is None else int(L)
        start = int(start)
        if not self._in_region(chrom, start, L):
            return self._ds_dna(chrom, start, L)
        arr = self._rc.get(("__dna__",))
        if arr is None:
            arr = self._rc[("__dna__",)] = self._ds_dna(chrom, self._region[1], self.rb)
        o = (start - self._region[1]) * self.C.RES
        return arr[o:o + L * self.C.RES].copy()

    def _strata(self) -> dict:
        names = self.ev.get("strata") or []
        path = self.ev.get("strata_path") or None
        if not names or not path:
            return {}
        from onion_eval.layers import load_strata_regions
        m = load_strata_regions(path, names, self.regions, self.rb)
        return {k: v.reshape(-1) for k, v in m.items()}

    def _tiles(self, regions):
        for ri, (c, s) in enumerate(regions):
            yield ri, [(c, s + k * self.L) for k in range(self.rb // self.L)]

    def _row(self, mid):
        return self.ds.check_row(mid)

    def _encoder_depth(self, row, scale=1.0):
        return self.ds.encoder.c_out(row, scale)

    def _halves_ok(self, row) -> bool:
        return (not self.no_halves) and has_halves(row)

    def _signal_files(self, row):
        """Paths of signal.h5, pr1_signal.h5, pr2_signal.h5, or (None, reason)."""
        if not row["counts_path"].startswith("h5:"):
            return None, f"{row['measurement_id']}: a store: path has no pipeline signal tracks"
        d = Path(row["counts_path"][3:]).parent
        ps = {nm: d / f"{nm}.h5" for nm in ("signal", "pr1_signal", "pr2_signal")}
        miss = [p.name for p in ps.values() if not p.is_file()]
        if miss:
            return None, f"{', '.join(miss)} missing for {row['measurement_id']}"
        return ps, ""

    def _needs_predicted_control(self, job) -> bool:
        if self.sig is None or job["kind"] != "imputation" or job.get("regime") == "control":
            return False
        if self.lambda_source == "observed_control":
            return False
        row = self.ds.rows[job["target"]]
        try:
            route = self.sig.route_of(row["assay_class"] or row["assay"])
        except ValueError:
            return False
        observed = bool((row.get("control_measurement_id") or "").strip())
        if self.lambda_source == "predicted_control":
            observed = False          # the predicted-control route runs on every ChIP target
        return (route == "chip" and not observed
                and self.predict_control and self._signal_files(row)[0] is not None)

    def _needs_average_control(self, job) -> bool:
        """F4: a ChIP-seq imputation target with no observed control and pipeline signal tracks;
        its position average takes lambda from the training-average control."""
        if self.sig is None or job["kind"] != "imputation" or job.get("regime") == "control":
            return False
        row = self.ds.rows[job["target"]]
        try:
            route = self.sig.route_of(row["assay_class"] or row["assay"])
        except ValueError:
            return False
        return (route == "chip" and not (row.get("control_measurement_id") or "").strip()
                and self._signal_files(row)[0] is not None)

    def _average_control_rate(self, ri, cell) -> np.ndarray:
        """Leave-one-cell-out mean rate (count / depth) of the training controls over region ri,
        fitted like the position average (`train_base`); NaN where undefined."""
        key = (ri, cell)
        if key not in self._pa_ctl_rate:
            from onion_eval.baselines import PositionAverage
            c, s = self.regions[ri]
            mids = [m for (_, a), m in sorted(self.ds.train_base.items()) if a == CONTROL]
            ci = self.C.TRACK_INDEX[CONTROL]
            bins = int(s) + np.arange(self.rb, dtype=np.int64)
            if mids:
                self._set_region(c, s)
                pa = PositionAverage().fit([self._region_batch(mids, c, s)])
                rate = pa.loo_rate(c, ci, cell, bins)
            else:
                rate = np.full(self.rb, np.nan)
            self._pa_ctl_rate[key] = rate
        return self._pa_ctl_rate[key]

    def _sig_prefixes(self) -> list:
        """Record prefixes of the signal variants: the two modes, plus the predicted-control route."""
        return list(SIGNAL_PREFIX.values()) + (
            [PC_PREFIX] if self.lambda_source == "predicted_control" else [])

    # -------------------------------------------------------------- reference predictors
    def _region_batch(self, mids, chrom, start, which="counts"):
        """A compact batch of whole-region target vectors (one window per measurement)."""
        C = self.C
        Lr = self.rb
        y = np.stack([self.ds.read(m, chrom, start, Lr, which) for m in mids])
        a_idx = [C.TRACK_INDEX[self.ds.rows[m]["assay"]] for m in mids]
        avail = np.zeros((len(mids), C.A), bool)
        c_out = np.zeros((len(mids), C.A, C.K_META), np.float32)
        ycols = np.zeros((len(mids), Lr, C.A), np.int32)
        for b, (m, a) in enumerate(zip(mids, a_idx)):
            avail[b, a] = True
            c_out[b, a] = self._encoder_depth(self.ds.rows[m])
            ycols[b, :, a] = y[b]
        meta = [{"cell": self.ds.rows[m]["cell"], "chrom": chrom, "start_bin": start} for m in mids]
        return {"y_counts": ycols, "y_avail": avail, "c_out": c_out, "meta": meta}

    def _fit_marginal(self, assays):
        from onion_eval.baselines import MarginalNB
        mnb = MarginalNB()
        for a in sorted(assays):
            mids = [m for (c, aa), m in sorted(self.ds.train_base.items()) if aa == a]
            if not mids:
                continue
            batches = [self._region_batch(mids, c, s) for c, s in self.fit_regions]
            part = MarginalNB().fit(batches)
            mnb.params.update(part.params)
        return mnb

    def _pa(self, assay, ri):
        key = (assay, ri)
        if key not in self._pa_cache:
            from onion_eval.baselines import PositionAverage
            if self._pa_cache and next(iter(self._pa_cache))[0] != assay:
                self._pa_cache.clear()
            mids = [m for (c, a), m in sorted(self.ds.train_base.items()) if a == assay]
            pa = PositionAverage()
            if mids:
                c, s = self.regions[ri]
                self._set_region(c, s)
                pa.fit([self._region_batch(mids, c, s)])
            self._pa_cache[key] = pa
        return self._pa_cache[key]

    def _pair_window(self, job, c, s):
        return self.ds.window(c, s, [job["source"]], [job["target"]], input_file="pr1",
                              target_file="pr2")

    def _fit_pair_refs(self, job):
        """Depth-only scaling and the floor of one pair, NB sizes fitted on the fit regions."""
        from onion_eval.baselines import PRED_FLOOR, DepthOnlyScaling, _fit_size
        wins = [self._pair_window(job, c, s) for _, tl in self._tiles(self.fit_regions) for c, s in tl]
        batches = [self.C.collate(wins[i:i + self.bs]) for i in range(0, len(wins), self.bs)]
        dos = DepthOnlyScaling().fit(batches)
        pr1 = np.concatenate([self.ds.read(job["target"], c, s, self.rb, "pr1")
                              for c, s in self.fit_regions]).astype(np.float64)
        pr2 = np.concatenate([self.ds.read(job["target"], c, s, self.rb, "pr2")
                              for c, s in self.fit_regions]).astype(np.float64)
        return dos, _fit_size(np.maximum(pr1, PRED_FLOOR), pr2)

    def _predicted_control(self, batch, t):
        """The model's control mean [B, L] with the control channel asked for at track t's C'
        (a separate forward pass, so the target's own prediction is untouched)."""
        ci = self.C.TRACK_INDEX[CONTROL]
        b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
        b2["c_out"][:, ci] = b2["c_out"][:, t]
        b2["y_avail"][:, ci] = True
        b2["y_kind"][:, ci] = 1
        return self.model_pred(b2)["mu"][:, :, ci]

    # -------------------------------------------------------------- one job
    def _job_arrays(self, job, refs):
        """Vectors of one job's target track over every evalset region, in region then tile order."""
        from onion_eval.baselines import PRED_FLOOR
        tgt = self._row(job["target"])
        t = self.C.TRACK_INDEX[tgt["assay"]]
        pair = job["kind"] in ("correction", "replicate")
        halves = (not pair) and self._halves_ok(tgt)
        want_ctl = (not pair) and self.model_pred is not None and self._needs_predicted_control(job)
        if pair:
            dos, floor_n = self._fit_pair_refs(job)
        names = tuple(refs) + ((self.main,) if self.main in REFS and self.main not in refs else ())
        want_avg_ctl = ((not pair) and "position_average" in names
                        and self._needs_average_control(job))
        want_dump_ctl = (not pair) and self._dump_wants_avg_ctl(job)    # AGd; read by the dump only
        out = defaultdict(list)
        for ri, tiles in self._tiles(self.regions):
            self._set_region(*self.regions[ri])
            for b0 in range(0, len(tiles), self.tile_bs):
                tl = tiles[b0:b0 + self.tile_bs]
                t0 = time.perf_counter()
                if pair:
                    wins = [self._pair_window(job, c, s) for c, s in tl]
                else:
                    wins = [self.ds.window(c, s, job["inputs"], [job["target"]]) for c, s in tl]
                batch = self.C.collate(wins)
                t1 = time.perf_counter()
                self.timing["windows"] += t1 - t0
                preds = {}
                if self.external:      # signal only: the count-space prediction is undefined
                    nanv = np.full(batch["y_counts"].shape, np.nan)
                    preds[EXTERNAL] = {"n": nanv, "mu": nanv}
                for name in names:
                    if name in ("depth_only_scaling", "pseudoreplicate_floor") and not pair:
                        nanv = np.full(batch["y_counts"].shape, np.nan)
                        preds[name] = {"n": nanv, "mu": nanv}   # defined on pairs only
                    elif name == "marginal_nb":
                        preds[name] = _np_out(self.marg.predict(batch))
                    elif name == "position_average":
                        preds[name] = _np_out(self._pa(tgt["assay"], ri).predict(batch))
                        if want_avg_ctl:   # F4: the training-average control at the target's C'
                            r0 = self.regions[ri][1]
                            depth = 2.0 ** np.asarray(batch["c_out"][:, t, 0], np.float64)
                            rate = self._average_control_rate(ri, tgt["cell"])
                            out["position_average.ctl_mu"].append(np.concatenate([
                                np.maximum(rate[s - r0:s - r0 + self.L] * depth[b], PRED_FLOOR)
                                for b, (_, s) in enumerate(tl)]))
                    elif name == "depth_only_scaling":
                        preds[name] = _np_out(dos.predict(batch))
                    elif name == "pseudoreplicate_floor":
                        p1 = np.stack([self.ds.read(job["target"], c, s, self.L, "pr1") for c, s in tl])
                        mu = np.zeros(batch["y_counts"].shape, np.float64)
                        n = np.full(mu.shape, np.nan)
                        mu[:, :, t] = np.maximum(p1, PRED_FLOOR)
                        n[:, :, t] = floor_n
                        preds[name] = {"n": n, "mu": mu}
                t2 = time.perf_counter()
                self.timing["reference_predictors"] += t2 - t1
                preds["main"] = preds[self.main] if self.main in preds else self.model_pred(batch)
                if want_ctl:
                    out["main.ctl_mu"].append(self._predicted_control(batch, t).reshape(-1))
                self.timing["model_forward"] += time.perf_counter() - t2
                for name, p in preds.items():
                    out[f"{name}.n"].append(p["n"][:, :, t].reshape(-1))
                    out[f"{name}.mu"].append(p["mu"][:, :, t].reshape(-1))
                if want_dump_ctl:      # AGd: the training-average control at the target's C'
                    r0 = self.regions[ri][1]
                    depth = 2.0 ** np.asarray(batch["c_out"][:, t, 0], np.float64)
                    rate = self._average_control_rate(ri, tgt["cell"])
                    out["dump.avg_ctl_mu"].append(np.concatenate([
                        np.maximum(rate[s - r0:s - r0 + self.L] * depth[b], PRED_FLOOR)
                        for b, (_, s) in enumerate(tl)]))
                out["y"].append(batch["y_counts"][:, :, t].numpy().reshape(-1).astype(np.float64))
                if pair:
                    out["x"].append(batch["x_counts"][:, :, t].numpy().reshape(-1).astype(np.float64))
                    if "log2r" not in out:
                        out["log2r"] = [float(batch["c_out"][0, t, 0] - batch["c_in"][0, t, 0])]
                elif halves:
                    for h in ("pr1", "pr2"):
                        out[h].append(np.concatenate([self.ds.read(job["target"], c, s, self.L, h)
                                                      for c, s in tl]).astype(np.float64))
        res = {k: (np.concatenate(v) if k != "log2r" else v[0]) for k, v in out.items()}
        if "position_average.ctl_mu" in res and not np.isfinite(res["position_average.ctl_mu"]).all():
            del res["position_average.ctl_mu"]     # no training control covers a bin: as before F4
        if "dump.avg_ctl_mu" in res and not np.isfinite(res["dump.avg_ctl_mu"]).all():
            del res["dump.avg_ctl_mu"]
        return res

    # -------------------------------------------------------------- headline 7 per rung
    def _rung_pairs(self, assay, cell):
        """{rung: [(a, b)]}: the replicate-ladder pairs of pairs.tsv (other than pseudoreplicate rows)
        of this assay with a side in this cell, of the scored split, restricted to the eval set's
        `replicate_pairs` when it lists any."""
        if self._rung_pair_index is None:
            from onion_eval.evalset import read_tsv
            idx = defaultdict(lambda: defaultdict(list))
            want = set(self.ev.get("replicate_pairs") or [])
            path = self.tables / "pairs.tsv"
            meas = self.ds.rows
            # OXe: an add-on score also reads the zero rungs of the explained fraction (isogenic,
            # anisogenic) of its split; the add-on eval set lists only other_experiment pair ids
            free = EXPLAINED_ZERO if self.addon else ()
            for p in (read_tsv(path) if path.exists() else []):
                if (p["rung"] not in REPLICATE_RUNGS[1:] or p["a"] == p["b"] or p.get("split") != self.split
                        or (want and p["pair_id"] not in want and p["rung"] not in free)
                        or p["a"] not in meas or p["b"] not in meas):
                    continue
                for c in {meas[p["a"]]["cell"], meas[p["b"]]["cell"]}:
                    idx[(p["assay"], c)][p["rung"]].append((p["a"], p["b"]))
            self._rung_pair_index = idx
        return self._rung_pair_index.get((assay, cell), {})

    def _h7_per_rung(self, job, row, n, mu, defined, vr_pr, size):
        """Headline 7 at every populated rung of the target's cell and assay, and the two-bound pass
        flag (PI ruling 31). The prediction NB(n, mu) is for the target's depth D; against a pair (a, b)
        it is thinned to a's depth (mu x D_a / D) and b is depth-matched to a by `variance_ratio`. A
        rung's ratio is the harmonic mean of its pairs' pooled ratios."""
        per = {"pseudoreplicate": vr_pr}
        why_rung = {}
        if defined and not self.external:
            D = float(row.get("depth_reads") or row.get("depth_fragments") or "nan")
            for rung, pairs in sorted(self._rung_pairs(job["assay"], job["cell"]).items()):
                vals, why = [], []
                for a, b in pairs:
                    ra, rb = self._row(a), self._row(b)
                    Da = float(ra.get("depth_reads") or ra.get("depth_fragments") or "nan")
                    Db = float(rb.get("depth_reads") or rb.get("depth_fragments") or "nan")
                    if not (np.isfinite(D) and Da > 0 and Db > 0):
                        why.append(f"{a}:{b}: a depth is missing")
                        continue
                    xa, xb = (np.concatenate([self._ds_read(m, c, s, self.rb) for c, s in self.regions])
                              .astype(np.float64) for m in (a, b))
                    if xa.size != size:
                        why.append(f"{a}:{b}: the pair covers other bins than the target")
                        continue
                    v = Mx.variance_ratio(n, mu * Da / D, xa, xb, depth_ratio_b=Da / Db,
                                          seed=zlib.crc32(f"h7:{a}:{b}".encode()))["pooled"]
                    if not _undef(v) and v > 0:
                        vals.append(float(v))
                    else:
                        why.append(f"{a}:{b}: " + (_why(v) if _undef(v) else f"ratio {v}"))
                per[rung] = (float(1.0 / np.mean(1.0 / np.array(vals))) if vals else
                             Mx.Undefined(f"no pair of rung {rung} gave a defined ratio"))
                if not vals:
                    why_rung[rung] = f"no pair of rung {rung} gave a defined ratio: " + "; ".join(why)
        lo, hi = per.get("pseudoreplicate"), per.get(H7_UPPER_RUNG["imputation"])
        if _undef(lo) or _undef(hi):
            why = ("no pseudoreplicate ratio" if _undef(lo) else
                   f"rung {H7_UPPER_RUNG['imputation']} not populated for this cell and assay")
            ok = Mx.Undefined(why)
        else:
            ok = float(lo >= 1.0 and hi <= 1.0)
        out = {"variance_ratio_per_rung": per, "variance_ratio_pass": ok}
        if getattr(self, "addon", False):          # OXe: the reason text of every Undefined ratio
            reasons = {g: why_rung.get(g) or _why(v) for g, v in per.items() if _undef(v)}
            up = H7_UPPER_RUNG["imputation"]
            if up not in per:
                reasons[up] = (f"rung {up} not populated for this cell and assay" if defined
                               else "the predictor is undefined on this target")
            out["variance_ratio_per_rung_reason"] = reasons
        return out

    def _h7_metric(self, recs, scorable, require_all=False):
        """Headline 7 block: the old scalar (the pseudoreplicate ratio, now read as the lower-bound
        check), plus `per_rung` (one metric block per rung), `pass_rate` (share of targets that pass
        both bounds) and `per_target`."""
        nb, sd = self.n_boot, self.seed
        out = metric(recs, _mean_of("variance_ratio"), scorable, nb, sd, require_all=require_all)
        out["description"] = H7_DESCRIPTION
        rungs = sorted({k for r in recs for k in (r.get("variance_ratio_per_rung") or {})},
                       key=REPLICATE_RUNGS.index)

        def rung_fn(rung):
            def fn(rs):
                return _mean_of("_v")([{"_v": (r.get("variance_ratio_per_rung") or {}).get(
                    rung, Mx.Undefined(f"rung {rung} not populated"))} for r in rs])
            return fn
        out["per_rung"] = {g: metric(recs, rung_fn(g), scorable, nb, sd) for g in rungs}
        out["pass_rate"] = metric(recs, _mean_of("variance_ratio_pass"), scorable, nb, sd)
        out["upper_bound_rung"] = dict(H7_UPPER_RUNG)
        out["per_target"] = {r["target"]: Scorer._h7_target(self, r) for r in recs}  # duck-typed callers (tests)
        if getattr(self, "addon", False):
            up = H7_UPPER_RUNG["imputation"]
            out["n_undefined_upper_bound"] = sum(
                1 for r in recs if _undef((r.get("variance_ratio_per_rung") or {}).get(up)))
        return out

    def _h7_target(self, r) -> dict:
        """Headline 7's per_target entry of one record."""
        e = {"per_rung": {g: _j(v) for g, v in (r.get("variance_ratio_per_rung") or {}).items()},
             "pass": _j(r.get("variance_ratio_pass")), "reason": _why(r.get("variance_ratio_pass"))
             if _undef(r.get("variance_ratio_pass")) else ""}
        if getattr(self, "addon", False):
            e["per_rung_reason"] = dict(r.get("variance_ratio_per_rung_reason") or {})
        return e

    @staticmethod
    def _h1_target(r) -> dict:
        """Headline 1's (counts_log1p) per_target entry of one record."""
        return {"assay": r["assay"], "cell": r["cell"], "level": _j(r["level"]),
                "E_sp": _j(r["E_sp"]), "V_sp": _j(r["V_sp"]), "N": _j(r["N"]),
                "N_log1p_bias_sim": _j(r.get("N_log1p_bias_sim")),
                "depth_reads": r.get("depth_reads"), "has_halves": r["has_halves"],
                "recovered": _j(r["recovered"])}

    # -------------------------------------------------------------- imputation
    def _imputation_group(self, jobs, regime):
        """Per-target records of one (regime, assay) group, and the assay-level shared/specific."""
        from onion_eval import error_budget as EB
        recs, arrays = [], []
        for job in jobs:
            A = self._job_arrays(job, ("marginal_nb", "position_average"))
            n, mu, y = A["main.n"], A["main.mu"], A["y"]
            row = self._row(job["target"])
            halves = self._halves_ok(row)
            r = {"target": job["target"], "assay": job["assay"], "cell": job["cell"],
                 "has_halves": halves}
            no_halves = Mx.Undefined("no pseudoreplicate halves for this target ("
                                     + ("--no-halves mode" if self.no_halves else
                                        "has_pseudoreps != 1 or a store: path") + ")")
            defined = bool(np.isfinite(n).all() and np.isfinite(mu).all() and (mu > 0).all()
                           and (n > 0).all())
            pa_ok = bool(np.isfinite(A["position_average.mu"]).all()
                         and np.isfinite(A["position_average.n"]).all())
            nan = Mx.Undefined(EXTERNAL_REASON if self.external else
                               "the predictor is undefined on this target (NaN prediction)")
            if self.main == "position_average":
                pa_c = Mx.crps_split(A["position_average.n"], A["position_average.mu"], y)
            elif pa_ok:                # the skill needs only the reference's CRPS (same estimator)
                from candi.bench.distributional import nb_crps_mean
                pa_c = {"crps": nb_crps_mean(A["position_average.n"], A["position_average.mu"], y),
                        "oracle_scaled": float("nan")}
            else:
                pa_c = {k: Mx.Undefined("position average undefined") for k in ("crps", "oracle_scaled")}
            m_c = (pa_c if self.main == "position_average" else Mx.crps_split(n, mu, y)) if defined else {k: nan for k in ("crps", "oracle_scaled", "scale_error")}
            r.update({"crps": m_c["crps"], "oracle_scaled": m_c["oracle_scaled"],
                      "scale_error": m_c["scale_error"], "crps_pa": pa_c["crps"],
                      "skill": (float(pa_c["crps"] - m_c["crps"]) if not (_undef(pa_c["crps"]) or _undef(m_c["crps"]))
                                else Mx.Undefined("CRPS undefined for the model or the position average"))})
            strata = {POOLED: np.ones(y.size, bool), **self.strata}
            r["pit"] = {k: ((pit_curve(n[m], mu[m], y[m]), int(m.sum())) if defined else None)
                        for k, m in strata.items() if m.any()}
            mn, mm = A["marginal_nb.n"], A["marginal_nb.mu"]
            r["pit_marginal"] = ((pit_curve(mn, mm, y), y.size)
                                 if np.isfinite(mn).all() and np.isfinite(mm).all() else None)
            f = _half_shares(row) if halves else (0.5, 0.5)
            if not halves:
                vr = no_halves
            else:                  # the prediction thinned to pr1's share; pr2 matched to pr1's depth
                vr = (Mx.variance_ratio(n, f[0] * mu, A["pr1"], A["pr2"], self.strata or None,
                                        depth_ratio_b=f[0] / f[1])["pooled"] if defined else nan)
            r["variance_ratio"] = vr
            r.update(self._h7_per_rung(job, row, n, mu, defined, vr, y.size))
            # headline 1 / 2 in counts_log1p: prediction is the NB mean at the target's own C'
            ts = Mx.to_space(y, SPACE)
            h1 = h2 = None
            dr = row.get("depth_reads") or row.get("depth_fragments")
            r["depth_reads"] = float(dr) if dr else None
            if halves:             # each half rescaled to the target's depth by 1 / its read share
                h1, h2 = (Mx.to_space(A[h], SPACE, rescale=1.0 / fh) for h, fh in zip(("pr1", "pr2"), f))
                r["N"] = Mx.noise_floor(h1, h2, ts, shares=f)
                r["N_log1p_bias_sim"] = Mx.noise_floor_log1p_bias(
                    y, f, seed=zlib.crc32(f"nfb:{job['target']}".encode()))
            else:
                r["N"] = no_halves
                r["N_log1p_bias_sim"] = no_halves
            ps = Mx.to_space(mu, SPACE) if defined else None
            a_s = Mx.to_space(A["position_average.mu"], SPACE) if pa_ok else None
            r["r_spec"] = (Mx.r_spec(ts, ps, a_s) if (defined and pa_ok)
                           else Mx.Undefined("prediction or training average undefined"))
            sig = self._signal_target(job, n, mu, A, defined, pa_ok)
            if self.dump_dir is not None:
                self._dump_target(job, regime, n, mu, A, defined, sig)
            r.update(sig.pop("record"))
            arrays.append({"defined": defined, "truth": ts, "pred": ps, "h1": h1, "h2": h2, "shares": f,
                           "n": n, "mu": mu, "y": y, "target": job["target"], **sig})
            recs.append(r)
            self.suff[f"crps/{regime}/{job['target']}"] = np.array(
                [_j(m_c["crps"]) or np.nan, _j(m_c["oracle_scaled"]) or np.nan,
                 _j(pa_c["crps"]) or np.nan, y.size], np.float64)
        # assay-level shared / specific split (headline 1) and error budget, over defined targets;
        # a target without halves joins the shared estimate (it needs no N) but not headline 1
        # OXe: a flagged (evaluation-only) target stays out of the assay-level split, the error
        # budget and the signal space, so its EIC targets keep their values
        ok = [i for i, a in enumerate(arrays) if a["defined"] and recs[i]["target"] not in self.flagged]
        assay = jobs[0]["assay"]
        for i, r in enumerate(recs):
            for k in ("E_sp", "V_sp", "level"):
                r[k] = Mx.Undefined(FLAGGED_REASON if r["target"] in self.flagged else
                                    "the predictor is undefined on this target")
            r["budget"] = None
        if ok:
            T = np.stack([arrays[i]["truth"] for i in ok])
            P = np.stack([arrays[i]["pred"] for i in ok])
            cells = np.array([recs[i]["cell"] for i in ok])
            N = None if self.no_halves else np.array([recs[i]["N"] for i in ok], np.float64)
            ss = Mx.shared_specific(T, P, cells, N)
            for j, i in enumerate(ok):
                reason = ss["reason"][j]
                for k in ("E_sp", "V_sp", "level"):
                    v = ss[k][j]
                    recs[i][k] = Mx.Undefined(reason) if (reason and k != "level") else float(v)
            for nm in ("V", "P", "C", "E"):
                for part in ("sh", "sp"):
                    self.suff[f"gram/{regime}/{assay}/{nm}_{part}_by_level"] = ss["by_level"][f"{nm}_{part}"]
            eb = ok if self.no_halves else [i for i in ok if recs[i]["has_halves"]]
            if eb:
                nbl = (None if self.no_halves else
                       EB.noise_share_from_halves(np.stack([arrays[i]["h1"] for i in eb]),
                                                  np.stack([arrays[i]["h2"] for i in eb]),
                                                  shares=np.array([arrays[i]["shares"] for i in eb])))
                bud = EB.error_budget(np.stack([arrays[i]["truth"] for i in eb]),
                                      np.stack([arrays[i]["pred"] for i in eb]),
                                      np.array([recs[i]["cell"] for i in eb]), noise_by_level=nbl)
                for i in eb:
                    recs[i]["budget"] = {k: bud[k] for k in BUDGET_KEYS}
            self._signal_group(recs, arrays, ok, regime, assay)
        elif self.external:            # no count prediction; the signal comes from the override
            self._signal_group(recs, arrays, list(range(len(recs))), regime, assay)
        elif self.flagged:
            for r in recs:
                for pre in self._sig_prefixes():
                    r[f"{pre}E_sp"] = r[f"{pre}V_sp"] = r[f"{pre}level"] = Mx.Undefined(
                        FLAGGED_REASON if r["target"] in self.flagged else
                        "the predictor is undefined on this target")
        else:
            for r in recs:
                for pre in self._sig_prefixes():
                    r[f"{pre}E_sp"] = r[f"{pre}V_sp"] = r[f"{pre}level"] = Mx.Undefined(
                        "the predictor is undefined on this target")
        ef = self._explained_per_layer(jobs[0]["assay"], arrays)
        for r in recs:
            r["recovered"] = recovered_fn(no_floor=self.no_halves)([r])
            r["_explained_per_layer"] = ef
            for k, v in r["pit"].items():
                if v is not None:
                    self.suff[f"pit/{regime}/{r['target']}/{k}"] = np.append(v[0], v[1])
        return recs

    # -------------------------------------------------------------- explained fraction per layer
    def _explained_per_layer(self, assay, arrays) -> dict:
        """Explained fraction per layer of one assay (PI ruling 32; reported only, never a selection
        input; `layers.explained_table`, `onion total variance.md` explained-per-layer).

        Filled where the two members of a pair both have a prediction: an other_experiment or
        other_cell_type pair of `pairs.tsv` whose two sides are both imputation targets of this regime.
        The members of a pair within one experiment (isogenic, anisogenic) are never predicted; for an
        imputed target they get the same inputs (the cell's other assays) and covariates up to depth, so
        their prediction difference is 0 at the common depth and their captured part is 0
        (`zero_by_shared_inputs`); their pair value comes from their reads. Layers reported: experiment
        (other_experiment over isogenic) and cell type (over anisogenic + other_experiment - isogenic, or
        the fallback that `layers.lower_terms` names)."""
        from onion_eval import layers as LY
        if self.external:
            return {"rows": [], "reason": EXTERNAL_REASON}
        pred = {a["target"]: a for a in arrays if a["defined"]}
        self._rung_pairs(assay, "")                                 # builds the index
        seen, pairs = set(), []
        for (ass, _), by in self._rung_pair_index.items():
            if ass != assay:
                continue
            for rung, lst in by.items():
                if rung not in EXPLAINED_RUNGS:
                    continue
                for a, b in lst:
                    if (rung, a, b) not in seen:
                        seen.add((rung, a, b))
                        pairs.append((rung, a, b))
        outer = [p for p in pairs if p[0] in EXPLAINED_OUTER and p[1] in pred and p[2] in pred]
        if not outer:
            return {"rows": [], "zero_by_shared_inputs": list(EXPLAINED_ZERO), "reason":
                    "no other_experiment or other_cell_type pair with a prediction for both members"}
        cells = {self._row(m)["cell"] for _, a, b in outer for m in (a, b)}
        inner = [p for p in pairs if p[0] in EXPLAINED_ZERO and self._row(p[1])["cell"] in cells]

        def depth(m):
            row = self._row(m)
            return float(row.get("depth_reads") or row.get("depth_fragments") or "nan")

        used = outer + inner
        D = min(min(depth(a), depth(b)) for _, a, b in used)

        def counts(m):
            if m in pred:
                return pred[m]["y"]
            return np.concatenate([self._ds_read(m, c, s_, self.rb) for c, s_ in self.regions]).astype(np.float64)

        recs = []
        for rung, a, b in used:
            xa, xb = counts(a), counts(b)
            if rung in EXPLAINED_OUTER:
                ea, eb = (LY.predicted_layer_mean(pred[m]["n"], pred[m]["mu"], D / depth(m)) for m in (a, b))
            else:
                ea = eb = np.zeros(xa.size)
            got = LY.pair_explained(xa, xb, ea, eb, depth(a), depth(b), depth=D,
                                    seed=zlib.crc32(f"ef:{rung}:{a}:{b}".encode()))
            recs.append(dict(rung=rung, cell=self._row(a)["cell"], captured=got["captured"],
                             pair_value=got["pair_value"]))
        rows = [r for r in LY.explained_table(recs, n_boot=self.n_boot, seed=self.seed)
                if r["rung"] in EXPLAINED_OUTER]
        return {"rows": [{k: (_j(v) if isinstance(v, float) else v) for k, v in r.items()} for r in rows],
                "depth": D, "zero_by_shared_inputs": list(EXPLAINED_ZERO), "reason": ""}

    # -------------------------------------------------------------- signal space (X2)
    def _signal_target(self, job, n, mu, A, defined, pa_ok):
        """Signal-space records of one target, once per NB-to-signal mode (SIGNAL_PREFIX): the
        headline `expectation` under `sig_*`, the `plug_in` variant under `sig_pi_*` (ruling 20)."""
        pc = self.lambda_source == "predicted_control"
        self._dump_ctl = None        # AGd: the control of the headline signal, for the dump only

        def undef(reason):
            u = Mx.Undefined(reason)
            rec = {"lambda_source": None, "lambda_reason": reason}
            for pre in self._sig_prefixes():
                rec.update({f"{pre}N": u, f"{pre}r_spec": u})
            if pc:
                rec.update({"lambda_source" + PREDICTED_CONTROL: None,
                            "lambda_reason" + PREDICTED_CONTROL: reason})
            return {"record": rec}
        if self.sig is None:
            return undef(self.sig_reason)
        row = self._row(job["target"])
        ps_files, why = self._signal_files(row)
        if ps_files is None:
            return undef(why)
        import h5py
        files = {}
        for nm, p in ps_files.items():
            with h5py.File(p, "r") as f:
                files[nm] = np.concatenate([np.asarray(f[c][s:s + self.rb], np.float64)
                                            for c, s in self.regions])
        ts = Mx.to_space(files["signal"], "signal_arcsinh")
        h1 = Mx.to_space(files["pr1_signal"], "signal_arcsinh")
        h2 = Mx.to_space(files["pr2_signal"], "signal_arcsinh")
        N = Mx.noise_floor(h1, h2, ts, 2.0)
        kw = dict(ds=self.ds, regions=self.regions, rb=self.rb, dnase_route=self.dnase_route)
        rec = {"lambda_source": None, "lambda_reason": ""}
        out = {}
        variants = [(mode, pre, row) for mode, pre in SIGNAL_PREFIX.items()]
        if pc:    # the predicted-control route: the observed control is ignored (SC)
            variants.append(("expectation", PC_PREFIX, dict(row, control_measurement_id="")))
        a_by_mode = {}
        for mode, pre, row_v in variants:
            src, lam_reason, ps, pa_reason = None, "", None, ""
            if self.external and pre == SIGNAL_PREFIX["expectation"]:
                ps = self._override_signal(row)
                src = "external_entry"
            elif self.external:
                lam_reason = EXTERNAL_PLUG_IN_REASON
            elif pre == PC_PREFIX and A.get("main.ctl_mu") is None:
                lam_reason = ("the predictor gives no control prediction (predict_control off, or "
                              "a training-free baseline)")
            elif defined:
                ctl, avg = A.get("main.ctl_mu"), False
                if (ctl is None and pre != PC_PREFIX and self.main == "position_average"
                        and self.lambda_source != "observed_control"
                        and A.get("position_average.ctl_mu") is not None):
                    ctl, avg = A["position_average.ctl_mu"], True     # F4
                if pre == SIGNAL_PREFIX["expectation"]:
                    self._dump_ctl = ctl
                try:
                    sp, src = self.sig.signal_from_nb(n, mu, row_v, predicted_control=ctl,
                                                      mode=mode, **kw)
                    src = TRAINING_AVERAGE_CONTROL if avg else src
                    ps = Mx.to_space(np.asarray(sp, np.float64), "signal_arcsinh")
                except (ValueError, KeyError) as e:
                    lam_reason = f"no signal from the prediction: {e}"
            else:
                lam_reason = "the predictor is undefined on this target"
            if mode in a_by_mode:         # the predicted-control route reuses the training average
                a_s, pa_reason = a_by_mode[mode]
            else:
                a_s = None
                if pa_ok:
                    actl = A.get("position_average.ctl_mu")     # F4: None unless no observed control
                    try:
                        sa, _ = self.sig.signal_from_nb(A["position_average.n"], A["position_average.mu"],
                                                        row, predicted_control=actl, mode=mode, **kw)
                        a_s = Mx.to_space(np.asarray(sa, np.float64), "signal_arcsinh")
                        if actl is not None:
                            rec["lambda_source_training_average"] = TRAINING_AVERAGE_CONTROL
                    except (ValueError, KeyError) as e:
                        a_s, pa_reason = None, f"no signal from the position average: {e}"
                a_by_mode[mode] = (a_s, pa_reason)
            if ps is not None and a_s is not None:
                rs = Mx.r_spec(ts, ps, a_s)
            else:
                rs = Mx.Undefined(lam_reason or (pa_reason if pa_ok else "position average undefined"))
            rec.update({f"{pre}N": N, f"{pre}r_spec": rs})
            if pre == SIGNAL_PREFIX["expectation"]:
                rec.update(lambda_source=src, lambda_reason=lam_reason)
            elif pre == PC_PREFIX:
                rec.update({"lambda_source" + PREDICTED_CONTROL: src,
                            "lambda_reason" + PREDICTED_CONTROL: lam_reason})
            out.update({f"{pre}truth": ts, f"{pre}pred": ps})
        return {"record": rec, **out}

    def _override_signal(self, row) -> np.ndarray:
        """The external entry's signal (raw units, like signal.h5) of one target over the eval
        regions, in the scorer's order, mapped to signal_arcsinh."""
        got = self.signal_override(row, list(self.regions))
        if isinstance(got, (list, tuple)):        # one array per region
            sig = np.concatenate([np.asarray(p, np.float32).reshape(-1) for p in got])
        else:                                     # the regions end to end
            sig = np.asarray(got, np.float32).reshape(-1)
        want = len(self.regions) * self.rb
        if sig.shape != (want,):
            raise ValueError(f"signal_override for {row['measurement_id']}: {sig.shape[0]} bins, "
                             f"want {len(self.regions)} regions x {self.rb}")
        if not np.isfinite(sig).all():
            raise ValueError(f"signal_override for {row['measurement_id']}: non-finite values")
        return Mx.to_space(sig.astype(np.float64), "signal_arcsinh")

    def _signal_group(self, recs, arrays, ok, regime, assay):
        for pre in self._sig_prefixes():
            for r in recs:
                r[f"{pre}E_sp"] = r[f"{pre}V_sp"] = r[f"{pre}level"] = Mx.Undefined(
                    "the predictor is undefined on this target")
            if self.sig is None or any(arrays[i].get(f"{pre}pred") is None for i in ok):
                rkey = "lambda_reason" + (PREDICTED_CONTROL if pre == PC_PREFIX else "")
                reasons = sorted({recs[i].get(rkey) or "" for i in ok} - {""})
                if self.external and pre != SIGNAL_PREFIX["expectation"]:
                    reasons = [EXTERNAL_PLUG_IN_REASON]
                reason = self.sig_reason or ("; ".join(reasons) if reasons
                                             else "signal prediction or truth missing")
                for r in recs:
                    r[f"{pre}E_sp"] = r[f"{pre}V_sp"] = r[f"{pre}level"] = Mx.Undefined(reason)
                continue
            T = np.stack([arrays[i][f"{pre}truth"] for i in ok])
            P = np.stack([arrays[i][f"{pre}pred"] for i in ok])
            ss = Mx.shared_specific(T, P, np.array([recs[i]["cell"] for i in ok]),
                                    np.array([recs[i][f"{pre}N"] for i in ok], np.float64))
            for j, i in enumerate(ok):
                for k in ("E_sp", "V_sp"):
                    recs[i][f"{pre}{k}"] = (Mx.Undefined(ss["reason"][j]) if ss["reason"][j]
                                            else float(ss[k][j]))
                recs[i][f"{pre}level"] = float(ss["level"][j])  # finding 15: signal-space level

    # -------------------------------------------------------------- pairs
    def _pair_record(self, job):
        A = self._job_arrays(job, ("depth_only_scaling", "pseudoreplicate_floor"))
        y = A["y"]
        out = {k: job.get(k) for k in ("pair_id", "assay", "cell", "knob", "never_trained",
                                       "subset_pair", "rung", "source", "target")}
        for nm, key in (("main", "crps_model"), ("depth_only_scaling", "crps_depth_only"),
                        ("pseudoreplicate_floor", "crps_floor")):
            n, mu = A[f"{nm}.n"], A[f"{nm}.mu"]
            if np.isfinite(n).all() and np.isfinite(mu).all():
                if nm == "main":       # the oracle split is quoted for the model's CRPS only
                    c = Mx.crps_split(n, mu, y)
                    out[key], out[key + "_oracle_scaled"] = c["crps"], c["oracle_scaled"]
                else:                  # same estimator as crps_split's crps (closed form)
                    from candi.bench.distributional import nb_crps_mean
                    out[key] = nb_crps_mean(n, mu, y)
                    out[key + "_oracle_scaled"] = None
            else:
                out[key] = out[key + "_oracle_scaled"] = Mx.Undefined(f"{nm} undefined on this pair")
        mu = A["main.mu"]
        out["pred_total"] = float(mu.sum()) if np.isfinite(mu).all() else float("nan")
        out["source_total"] = float(A["x"].sum())
        out["log2_depth_ratio"] = A["log2r"]
        self.suff[f"pair/{job.get('pair_id') or job['source'] + '>' + job['target']}"] = np.array(
            [_j(out["crps_model"]) or np.nan, _j(out["crps_depth_only"]) or np.nan,
             _j(out["crps_floor"]) or np.nan, y.size], np.float64)
        return out

    # -------------------------------------------------------------- guardrail
    def _guard_model(self):
        """The model as the guardrail calls it: CPU batches in, on the scoring device inside."""
        m, dev = self.model_pred.model, self.device
        if str(dev) == "cpu":
            return m
        return lambda b: m({k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()})

    def _copy_leak(self):
        if self.model is None:
            return {"hidden_input_max_abs_change": None, "unavailable_input_max_abs_change": None,
                    "same_view_copy_gain": None, "same_view_copy_gain_flag": None,
                    "copy_leak_pass": None, "same_view_copy_gain_spread_ok": None,
                    "part1_wiring_pass": None,
                    "reason": "not applicable: a training-free baseline reads no hidden input"}
        from onion_eval import guardrail as G
        GAIN_THRESHOLD, gate_on = G.GAIN_THRESHOLD, bool(G.GATE_ON_COPY_GAIN)
        per_seed, n_win, n_pool, n_targets = [], 0, 0, []
        for s in self.guardrail_seeds:
            ds = self._OnionDataset(self.tables, self.split, self.cfg, seed=s)
            items, n_pool = guardrail_items(ds, self.guardrail_windows, s, self.flagged)
            n_targets.append(len({it["targets"][0] for it in items}))
            wins = [ds.item_window(it) for it in items]
            n_win = len(wins)
            batches = [self.C.collate(wins[i:i + self.bs]) for i in range(0, len(wins), self.bs)]
            per_seed.append(G.copy_leak(self._guard_model(), batches, seed=s))
        if not per_seed:
            return {"copy_leak_pass": None, "same_view_copy_gain_spread_ok": None,
                    "part1_wiring_pass": None, "same_view_copy_gain_flag": None,
                    "gate_on_copy_gain": gate_on, "reason": "no mask seed"}
        rel = [_j(r["same_view_copy_gain_relative"]) for r in per_seed]
        fin = [v for v in rel if v is not None]
        spread = (max(fin) - min(fin)) if len(fin) >= 2 else None
        lim = SPREAD_MAX_FRACTION * GAIN_THRESHOLD
        spread_ok = bool(spread is not None and len(fin) == len(rel) and spread <= lim)
        flags = [bool(v is not None and v > GAIN_THRESHOLD) for v in rel]
        part1 = all(r["part1_wiring_pass"] is True for r in per_seed)
        why = [] if part1 else ["part 1 failed on a mask seed: perturbing hidden bins or "
                                "unavailable tracks moved an output, or no hidden bin was checked"]
        if gate_on and not spread_ok:
            why.append(f"the relative gain moved {spread} across mask seeds (limit {lim:g}); score "
                       f"with more --guardrail-windows" if spread is not None else
                       "the relative gain is undefined on a mask seed, or fewer than 2 mask seeds")

        def mx(k):
            v = [_j(r[k]) for r in per_seed]
            return None if any(x is None for x in v) else max(v)
        return {
            "hidden_input_max_abs_change": mx("hidden_input_max_abs_change"),
            "unavailable_input_max_abs_change": mx("unavailable_input_max_abs_change"),
            "same_view_copy_gain": mx("same_view_copy_gain"),
            "same_view_copy_gain_relative": mx("same_view_copy_gain_relative"),
            "copy_leak_pass": all(r["copy_leak_pass"] is True for r in per_seed),
            "part1_wiring_pass": part1,
            "gate_on_copy_gain": gate_on,
            "copy_leak_pass_by_mask_seed": [bool(r["copy_leak_pass"]) for r in per_seed],
            "hidden_input_max_abs_change_by_mask_seed": [_j(r["hidden_input_max_abs_change"])
                                                         for r in per_seed],
            "unavailable_input_max_abs_change_by_mask_seed": [
                _j(r["unavailable_input_max_abs_change"]) for r in per_seed],
            "same_view_copy_gain_relative_by_mask_seed": rel,
            "same_view_copy_gain_relative_spread": spread,
            "same_view_copy_gain_flag": any(flags),
            "same_view_copy_gain_flag_by_mask_seed": flags,
            "same_view_copy_gain_spread_ok": spread_ok,
            "same_view_copy_gain_spread_limit_relative": lim,
            "same_view_copy_gain_threshold_relative": GAIN_THRESHOLD,
            "mask_seeds": list(self.guardrail_seeds), "n_windows_per_seed": n_win,
            "window_draw": GUARD_DRAW, "n_reconstruction_items": n_pool,
            "n_targets_by_mask_seed": n_targets,
            "reason": "; ".join(why),
        }

    # -------------------------------------------------------------- run
    def jobs(self):
        meas = self.ds.rows
        imp = [dict(t, kind="imputation") for t in self.ev["imputation_targets"]
               if t["split"] == self.split]
        ctl = [dict(t, kind="imputation", regime="control")
               for t in self.ev.get("control_targets", []) if t["split"] == self.split]
        for t in imp + ctl:
            t.setdefault("assay", meas[t["target"]]["assay"])
            t.setdefault("cell", meas[t["target"]]["cell"])
        bad_ctl = [t["target"] for t in imp if t["assay"] == CONTROL]
        if bad_ctl:
            raise ValueError(f"control tracks listed as imputation targets: {bad_ctl[:3]} "
                             f"(rebuild the eval set)")
        cor = [dict(p, kind="correction") for p in self.ev["correction_pairs"]
               if p["split"] == self.split]
        for p in cor:
            p.setdefault("assay", meas[p["target"]]["assay"])
            p.setdefault("cell", meas[p["target"]]["cell"])
        rep = []
        want = set(self.ev.get("replicate_pairs") or [])
        if want:
            from onion_eval.evalset import read_tsv
            for p in read_tsv(self.tables / "pairs.tsv"):
                if (p["pair_id"] in want and p["split"] == self.split and p["rung"] in REPLICATE_RUNGS
                        and p["a"] != p["b"]
                        and all(has_halves(meas.get(m, {})) for m in (p["a"], p["b"]))):
                    rep.append({"kind": "replicate", "source": p["a"], "target": p["b"],
                                "pair_id": p["pair_id"], "rung": p["rung"], "assay": p["assay"],
                                "cell": p["cell"], "never_trained": int(p["never_trained"] or 0),
                                "subset_pair": int(p["subset_pair"] or 0),
                                "knob": p.get("knobs_differing", "")})
        if self.no_halves or self.external:   # every pair reads halves; a pair is count-space
            cor, rep = [], []
        # DT, Log "FOR E7a": a pair whose two sides can share reads is never scored
        self.n_pairs_not_read_disjoint = sum(1 for p in cor if p.get("read_disjoint", 1) == 0)
        cor = [p for p in cor if p.get("read_disjoint", 1) != 0]
        for j in imp + ctl + cor + rep:
            for m in [j["target"]] + j.get("inputs", []) + ([j["source"]] if "source" in j else []):
                if m not in meas:
                    raise KeyError(f"measurement {m!r} not in measurements.tsv")
                if meas[m]["in_eic"] == "blind" and self.split != "blind":
                    raise PermissionError(f"{m} is a blind track; blind rows are scored only on "
                                          f"--split blind --allow-blind")
                self.ds.check_row(m)
        if not self.predict_control:
            ctl = []
        return imp, cor, rep, ctl

    def run(self) -> dict:
        t0 = time.time()
        if self.dump_dir is not None and self.split == "blind" and not self._dump_ledgered:
            raise ScoreRefused("--dump-signal on the blind split: only inside a blind pass that "
                               "the ledger records (score())")
        imp, cor, rep, ctl = self.jobs()
        sc = self.ev.get("scorable", {})
        sc_imp, sc_cor = sc.get("imputation", []), sc.get("correction", [])
        tq = time.perf_counter()
        self.marg = self._fit_marginal({j["assay"] for j in imp + ctl})
        self.timing["fit_marginal"] += time.perf_counter() - tq
        scores = {}
        recs_by_regime, flagged_by_regime = {}, {}
        for regime in REGIMES + ("control",):
            groups = defaultdict(list)
            for j in (ctl if regime == "control" else imp):
                if j["regime"] == regime:
                    groups[j["assay"]].append(j)
            recs = []
            for a in sorted(groups):
                self.log(f"score: {regime} {a}: {len(groups[a])} targets")
                tq = time.perf_counter()
                recs += self._imputation_group(groups[a], regime)
                self.timing["targets_total"] += time.perf_counter() - tq
            self._pa_cache.clear()
            flagged_by_regime[regime] = [r for r in recs if r["target"] in self.flagged]
            recs_by_regime[regime] = [r for r in recs if r["target"] not in self.flagged]
        if self.dump_dir is not None:
            self._dump_finish()
        main_recs = recs_by_regime["leave_one_assay_out"]
        ta = time.perf_counter()
        scores.update(self._imputation_metrics(main_recs, sc_imp, require_all=True))
        if self.addon:
            self._add_flagged(scores, flagged_by_regime["leave_one_assay_out"])
        scores["accessibility_only"] = self._imputation_metrics(
            recs_by_regime["accessibility_only"], sc_imp)
        scores["accessibility_only"]["regime"] = "accessibility_only"
        if ctl:
            pc = self._imputation_metrics(recs_by_regime["control"], [CONTROL])
            pc["regime"] = "control"
            pc["n_targets"] = len(recs_by_regime["control"])
            scores["predicted_control"] = pc
        else:
            scores["predicted_control"] = {"n_targets": 0, "reason": (
                "predict_control is off or the predictor is a training-free baseline"
                if not self.predict_control else "no control target in this split")}
        self.timing["assembly"] += time.perf_counter() - ta
        self.log(f"score: {len(cor)} correction pairs, {len(rep)} replicate pairs")
        tq = time.perf_counter()
        cor_recs = [self._pair_record(j) for j in cor]
        rep_recs = [self._pair_record(j) for j in rep]
        self.timing["pairs_total"] += time.perf_counter() - tq
        ta = time.perf_counter()
        scores.update(self._correction_metrics(cor_recs, sc_cor))
        scores["replicate_pair_gap_closed_vs_depth_only"] = {"per_rung": {
            rung: metric([r for r in rep_recs if r["rung"] == rung], _gap_fn, set(
                (sc.get("layers") or {}).get(rung, [])), self.n_boot, self.seed)
            for rung in sorted({r["rung"] for r in rep_recs})}}
        self.timing["assembly"] += time.perf_counter() - ta
        tg = time.perf_counter()
        scores["copy_leak"] = self._copy_leak()
        self.timing["guardrail"] += time.perf_counter() - tg
        if self.external:
            for k in list(scores):
                if k.startswith(("correction_", "subset_pairs_", "depth_law_")):
                    scores[k] = metric([], None, sc_cor, undefined_reason=EXTERNAL_REASON)
            scores["correction_gap_closed_vs_depth_only_never_trained"].update(
                per_knob={}, n_pairs=0, denominator_relative_by_assay={})
            scores["replicate_pair_gap_closed_vs_depth_only"] = {"per_rung": {},
                                                                 "reason": EXTERNAL_REASON}
        if self.no_halves:
            imp_t = Mx.Undefined("--no-halves mode: headline 1 has no noise floor and is not a "
                                 "selection term")
        elif self.external:
            imp_t = Mx.Undefined(EXTERNAL_REASON)
        else:
            imp_t = scores["specific_variance_recovered_above_noise"][SPACE]["macro"]
        cor_m = scores["correction_gap_closed_vs_depth_only_never_trained"]
        cor_t = Mx.Undefined(EXTERNAL_REASON) if self.external else cor_m["macro"]
        h1_status = scores["specific_variance_recovered_above_noise"][SPACE]["per_assay_status"]
        if self.external:
            unc, unc_den = Mx.Undefined(EXTERNAL_REASON), None
        else:
            unc, unc_den = self._uncertainty(main_recs, sc_imp)
        comp = composite(imp_t, cor_t, unc)
        den_rel = [v for v in cor_m["denominator_relative_by_assay"].values() if v is not None]
        scores["selection"] = {
            "space": SPACE, "imputation_gap_closed": _j(imp_t),
            "correction_gap_closed": _j(cor_t), "uncertainty_gap_closed": _j(unc),
            "composite": _j(comp), "reason": _why(comp) if _undef(comp) else "",
            "imputation_reason": _why(imp_t) if _undef(imp_t) else "",
            "uncertainty_reason": _why(unc) if _undef(unc) else "",
            "correction_reason": cor_m.get("reason", "") if _undef(cor_t) else "",
            "imputation_term_definition": IMPUTATION_TERM_DEFINITION,
            "imputation_assays_in_term": sorted(
                a for a, v in h1_status.items() if v["in_imputation_term"]),
            "headline1_per_assay_status": h1_status,
            "denominators": {
                "uncertainty_marginal_pit_deviation": _j(unc_den),
                "correction_baseline_minus_floor_relative_min": (min(den_rel) if den_rel and len(
                    den_rel) == len(cor_m["denominator_relative_by_assay"]) else None),
                "correction_baseline_minus_floor_relative_by_assay":
                    cor_m["denominator_relative_by_assay"]}}
        n_wo = sum(1 for r in main_recs if not r["has_halves"])
        scores.update({"n_targets": len(main_recs), "n_pairs": len(cor_recs),
                       "n_targets_without_halves": n_wo,
                       "n_replicate_pairs": len(rep_recs), "regime": "leave_one_assay_out",
                       "split": self.split, "scorable": sc, "predictor": self.main,
                       "noise_floor": NOISE_FLOOR_NONE if self.no_halves else NOISE_FLOOR_HALVES,
                       "evalset": str(self.evalset_path), "evalset_sha256": sha256_file(self.evalset_path),
                       "tables": str(self.tables), "tables_sha256": self.tables_sha256,
                       "run_tables_sha256": self.run_tables_sha256,
                       "dnase_route": self.dnase_route, "baseline_fit_on": self.fit_on,
                       "n_regions": len(self.regions), "region_bins": self.rb,
                       "context_bins": self.L, "wall_seconds": round(time.time() - t0, 1),
                       "infer_batch": self.tile_bs, "region_cache": self.region_cache,
                       "timing_seconds": {k: round(v, 1) for k, v in sorted(self.timing.items())},
                       "torch_threads": torch.get_num_threads(), "device": str(self.device)})
        # SC: keys only off the default, so a default scores.json is unchanged
        if self.lambda_source != "auto":
            scores["lambda_source_option"] = self.lambda_source
        if self.external:
            scores["signal_source"] = "external entry (signal_override)"
            scores["count_space_reason"] = EXTERNAL_REASON
        if self.n_pairs_not_read_disjoint:
            scores["correction_pairs_not_read_disjoint"] = self.n_pairs_not_read_disjoint
        if self.addon:
            fl = sorted(self.flagged)
            scores["addon"] = {
                "addon_of": self.ev["addon_of"], "addon_of_path": self.ev.get("addon_of_path", ""),
                "addon_of_tables_sha256": self.addon_of_tables_sha256,
                "evaluation_only_targets": fl, "n_evaluation_only_targets": len(fl),
                "note": "add-on score (Amendment 4): reported only; evaluation-only targets enter "
                        "explained_fraction_per_layer and per_target only, never a headline block, "
                        "a selection term or n_targets"}
        return scores

    def _add_flagged(self, scores, recs) -> None:
        """OXe: the evaluation-only targets' per_target entries (headline 1 in counts_log1p and
        headline 7), each marked `evaluation_only`; no aggregate reads them."""
        h1 = scores["specific_variance_recovered_above_noise"][SPACE]
        h7 = scores["predicted_over_replicate_variance"]
        for r in recs:
            h1.setdefault("per_target", {})[r["target"]] = dict(self._h1_target(r), evaluation_only=True)
            h7.setdefault("per_target", {})[r["target"]] = dict(self._h7_target(r), evaluation_only=True)

    # -------------------------------------------------------------- metric assembly
    def _imputation_metrics(self, recs, scorable, require_all=False) -> dict:
        nb, sd = self.n_boot, self.seed
        ra = require_all
        wo = sorted(r["target"] for r in recs if not r["has_halves"])
        if self.no_halves:
            h1 = metric(recs, None, scorable, undefined_reason=(
                f"--no-halves mode: no pseudoreplicate noise floor; headline 1 without a floor is "
                f"under {PILOT_H1_KEY} and is never a selection input"))
            pilot = metric(recs, recovered_fn(no_floor=True), scorable, nb, sd, require_all=ra)
        else:
            h1 = metric(recs, recovered_fn(), scorable, nb, sd, require_all=ra)
        h1["n_targets_without_halves"] = len(wo)
        h1["targets_without_halves"] = wo
        h1["per_target"] = {r["target"]: self._h1_target(r) for r in recs}
        h1s = metric(recs, recovered_fn("sig_"), scorable, nb, sd, require_all=ra)
        h1s["per_target"] = {r["target"]: {"assay": r["assay"], "cell": r["cell"],
                                           "level": _j(r["sig_level"]), "N": _j(r["sig_N"]),
                                           "lambda_source": r.get("lambda_source"),
                                           "lambda_reason": r.get("lambda_reason", ""),
                                           **({"lambda_source_training_average":
                                               r["lambda_source_training_average"]}
                                              if "lambda_source_training_average" in r else {})}
                             for r in recs}
        h1s["dnase_route"] = self.dnase_route
        h1s["signal_mode"] = "expectation"
        h1p = metric(recs, recovered_fn(SIGNAL_PREFIX["plug_in"]), scorable, nb, sd, require_all=ra)
        h1p["per_target"] = {r["target"]: {"assay": r["assay"], "cell": r["cell"],
                                           "level": _j(r["sig_pi_level"]), "N": _j(r["sig_pi_N"]),
                                           "lambda_source": r.get("lambda_source"),
                                           **({"lambda_source_training_average":
                                               r["lambda_source_training_average"]}
                                              if "lambda_source_training_average" in r else {})}
                             for r in recs}
        h1p["dnase_route"] = self.dnase_route
        h1p["signal_mode"] = "plug_in"

        def pit_fn(key):
            def fn(rs):
                cur = [r["pit"].get(key) for r in rs]
                if any(c is None for c in cur) and key == POOLED:
                    return Mx.Undefined("the predictor is undefined on a target")
                return pooled_pit_deviation([c for c in cur if c is not None])
            return fn

        def worst_fn(rs):
            if not self.strata:
                return Mx.Undefined("no region strata in the evalset")
            vals = [pit_fn(k)(rs) for k in self.strata]
            vals = [v for v in vals if not _undef(v)]
            return max(vals) if vals else Mx.Undefined("no stratum with bins")

        budget = {}
        per_assay_budget = {}
        by = defaultdict(list)
        for r in recs:
            if r.get("budget") is not None:
                by[r["assay"]].append(r["budget"])
        for a, bs in by.items():
            per_assay_budget[a] = {k: _j(bs[0][k]) for k in BUDGET_KEYS}
        for k in BUDGET_KEYS:
            vals = [per_assay_budget[a][k] for a in scorable if a in per_assay_budget]
            budget[k] = (float(np.mean(vals)) if vals and all(v is not None for v in vals)
                         and len(vals) == len([a for a in scorable if a in {r["assay"] for r in recs}])
                         else None)
        budget["per_assay"] = per_assay_budget
        out = {
            "specific_variance_recovered_above_noise": {SPACE: h1, "signal_arcsinh": h1s,
                                                        "signal_arcsinh" + PLUG_IN: h1p},
            "r_spec": {SPACE: metric(recs, _mean_of("r_spec"), scorable, nb, sd, require_all=ra),
                       "signal_arcsinh": metric(recs, _mean_of("sig_r_spec"), scorable, nb, sd,
                                                require_all=ra),
                       "signal_arcsinh" + PLUG_IN: metric(recs, _mean_of("sig_pi_r_spec"), scorable,
                                                          nb, sd, require_all=ra)},
            "count_crps_skill_vs_position_average": metric(recs, _mean_of("skill"), scorable, nb, sd,
                                                           require_all=ra),
            "count_crps": metric(recs, _mean_of("crps"), scorable, nb, sd, require_all=ra),
            "count_crps_oracle_scaled": metric(recs, _mean_of("oracle_scaled"), scorable, nb, sd,
                                               require_all=ra),
            "count_crps_scale_error": metric(recs, _mean_of("scale_error"), scorable, nb, sd,
                                             require_all=ra),
            "pit_deviation_pooled": metric(recs, pit_fn(POOLED), scorable, nb, sd, require_all=ra),
            "pit_deviation_worst_stratum": metric(recs, worst_fn, scorable, nb, sd, require_all=ra),
            "predicted_over_replicate_variance": self._h7_metric(recs, scorable, require_all=ra),
            "explained_fraction_per_layer": {
                "description": "reported only, never a selection input (PI ruling 32): per layer, the share of "
                               "the layer term that the prediction difference of the pair captures, "
                               "layers.explained_table",
                "per_assay": {r["assay"]: r["_explained_per_layer"] for r in recs
                              if r.get("_explained_per_layer") is not None}},
            "error_budget": budget,
        }
        if self.lambda_source == "predicted_control":
            sk = "signal_arcsinh" + PREDICTED_CONTROL
            h1pc = metric(recs, recovered_fn(PC_PREFIX), scorable, nb, sd, require_all=ra)
            h1pc["per_target"] = {r["target"]: {
                "assay": r["assay"], "cell": r["cell"], "level": _j(r[f"{PC_PREFIX}level"]),
                "N": _j(r[f"{PC_PREFIX}N"]),
                "lambda_source": r.get("lambda_source" + PREDICTED_CONTROL),
                "lambda_reason": r.get("lambda_reason" + PREDICTED_CONTROL, "")} for r in recs}
            h1pc["dnase_route"] = self.dnase_route
            h1pc["signal_mode"] = "expectation"
            h1pc["lambda_source_option"] = self.lambda_source
            out["specific_variance_recovered_above_noise"][sk] = h1pc
            out["r_spec"][sk] = metric(recs, _mean_of(f"{PC_PREFIX}r_spec"), scorable, nb, sd,
                                       require_all=ra)
        if self.no_halves:
            pilot["per_target"] = h1["per_target"]
            out[PILOT_H1_KEY] = pilot
        h1["per_assay_status"] = headline1_status(recs, scorable, pilot if self.no_halves else h1,
                                                  self.no_halves, in_composite=ra)
        if self.no_halves:
            pilot["per_assay_status"] = h1["per_assay_status"]
            unc, den = self._uncertainty_all(recs)
            out[PILOT_UNC_KEY] = {"value": _j(unc), "marginal_pit_deviation": _j(den),
                                  "assays": sorted({r["assay"] for r in recs}),
                                  "reason": _why(unc) if _undef(unc) else "",
                                  "note": "pilot only: the uncertainty term over every assay with a "
                                          "target (no scorable assay exists without halves); never "
                                          "a selection input"}
        if self.external:
            self._blank_count_space(out, recs, scorable)
        return out

    #: the count-space metrics of an imputation table (an external entry writes each undefined)
    COUNT_SPACE_KEYS = ("count_crps_skill_vs_position_average", "count_crps",
                        "count_crps_oracle_scaled", "count_crps_scale_error", "pit_deviation_pooled",
                        "pit_deviation_worst_stratum", "predicted_over_replicate_variance")

    def _blank_count_space(self, out, recs, scorable) -> None:
        """External entry (XE): every count-space metric undefined with EXTERNAL_REASON."""
        def u():
            return metric(recs, None, scorable, undefined_reason=EXTERNAL_REASON)
        for k in self.COUNT_SPACE_KEYS:
            out[k] = u()
        h1 = u()
        h1.update(per_assay_status={}, per_target={}, n_targets_without_halves=0,
                  targets_without_halves=[])
        out["specific_variance_recovered_above_noise"][SPACE] = h1
        out["r_spec"][SPACE] = u()
        out["error_budget"] = {**{k: None for k in BUDGET_KEYS}, "per_assay": {},
                               "reason": EXTERNAL_REASON}
        for k in (PILOT_H1_KEY, PILOT_UNC_KEY):
            out.pop(k, None)

    def _uncertainty_all(self, recs):
        """The uncertainty term over every assay with a target (the pilot form, DEFECTS item 3)."""
        return self._uncertainty(recs, sorted({r["assay"] for r in recs}))

    def _uncertainty(self, recs, scorable):
        """(uncertainty term, its denominator = mean marginal-NB PIT deviation over the assays)."""
        by = defaultdict(list)
        for r in recs:
            if r["assay"] in scorable:
                by[r["assay"]].append(r)
        if not by:
            return Mx.Undefined("no scorable assay with a target"), None
        missing = sorted(set(scorable) - set(by))
        if missing:
            return Mx.Undefined("scorable assay with no target: " + ", ".join(missing)), None
        arm, marg = [], []
        for a, rs in sorted(by.items()):
            if any(r["pit"].get(POOLED) is None for r in rs):
                return Mx.Undefined(f"PIT undefined for the arm on {a}"), None
            if any(r["pit_marginal"] is None for r in rs):
                return Mx.Undefined(f"marginal NB undefined on {a}"), None
            arm.append(pooled_pit_deviation([r["pit"][POOLED] for r in rs]))
            marg.append(pooled_pit_deviation([r["pit_marginal"] for r in rs]))
        return Mx.uncertainty_gap_closed(arm, marg), float(np.mean(marg))

    def _correction_metrics(self, recs, scorable) -> dict:
        nb, sd = self.n_boot, self.seed
        head = [r for r in recs if r["never_trained"] == 1 and r["subset_pair"] == 0]
        sub = [r for r in recs if r["subset_pair"] == 1]        # pure depth thinning, any pair
        m = metric(head, _gap_fn, scorable, nb, sd, require_all=True)
        knobs = defaultdict(list)
        for r in head:
            knobs[r.get("knob") or ""].append(r)
        m["per_knob"] = {k: _j(_gap_fn(rs)) for k, rs in sorted(knobs.items())}
        m["n_pairs"] = len(head)
        den = {}
        for a in sorted(scorable):
            rs = [r for r in head if r["assay"] == a]
            b = [r["crps_depth_only"] for r in rs]
            f = [r["crps_floor"] for r in rs]
            if rs and not any(_undef(x) for x in b + f) and np.mean(b) > 0:
                den[a] = float((np.mean(b) - np.mean(f)) / np.mean(b))
            else:
                den[a] = None
        m["denominator_relative_by_assay"] = den

        def law(rs):
            bad = [r for r in rs if any(_undef(r[k]) for k in ("pred_total", "source_total",
                                                               "log2_depth_ratio"))]
            if bad:
                return Mx.Undefined(f"{len(bad)} of {len(rs)} pairs have an undefined total")
            return Mx.depth_law_error([r["pred_total"] for r in rs], [r["source_total"] for r in rs],
                                      [r["log2_depth_ratio"] for r in rs])
        return {
            "correction_gap_closed_vs_depth_only_never_trained": m,
            "correction_crps_model": metric(head, _mean_of("crps_model"), scorable, nb, sd),
            "correction_crps_model_oracle_scaled": metric(head, _mean_of("crps_model_oracle_scaled"),
                                                          scorable, nb, sd),
            "correction_crps_depth_only": metric(head, _mean_of("crps_depth_only"), scorable, nb, sd),
            "correction_crps_pseudoreplicate_floor": metric(head, _mean_of("crps_floor"), scorable, nb, sd),
            "subset_pairs_gap_closed": metric(sub, _gap_fn, scorable, nb, sd),
            "depth_law_abs_log2_error_never_trained": metric(head, law, scorable, nb, sd),
        }


FALLBACK_NONE_NEEDED = "none needed: defined, in the imputation term"
FALLBACK_NON_SCORABLE = ("none: non-scorable (PI ruling 4), so outside the macro and the composite; "
                         "reported apart")
FALLBACK_UNDEFINED = ("none: a scorable assay is undefined, so the macro and the composite's "
                      "imputation term are undefined (select_rule escalates)")
FALLBACK_PILOT = (f"pilot --no-halves: headline 1 with N = 0 under {PILOT_H1_KEY}; never a "
                  f"selection input")


def headline1_status(recs, scorable, h1m: dict, no_halves: bool, in_composite: bool) -> dict:
    """Per assay of one regime (DEFECTS item 5): whether the pseudoreplicate noise floor exists,
    whether headline 1 is defined (and why not), whether the assay is in the composite's imputation
    term, and the fallback used. `h1m` is the headline-1 metric dict the values come from (the
    pilot form in --no-halves mode). No fallback value is ever computed: there is none."""
    by = defaultdict(list)
    for r in recs:
        by[r["assay"]].append(r)
    vals = dict(h1m.get("per_assay") or {}, **(h1m.get("non_scorable_per_assay") or {}))
    reasons = h1m.get("per_assay_reason") or {}
    out = {}
    for a, rs in sorted(by.items()):
        n_floor = sum(1 for r in rs if r["has_halves"] and not _undef(r["N"]))
        defined = vals.get(a) is not None
        sc = a in set(scorable)
        if no_halves:
            fb = FALLBACK_PILOT
        elif not sc:
            fb = FALLBACK_NON_SCORABLE
        else:
            fb = FALLBACK_NONE_NEEDED if defined else FALLBACK_UNDEFINED
        why = "" if defined else reasons.get(a, "")
        if not defined and len(rs) < 2:
            why = (f"{len(rs)} validation target: the shared/specific split needs >= 2 targets of "
                   f"the assay" + (f" ({why})" if why else ""))
        out[a] = {"n_targets": len(rs), "n_targets_with_floor": n_floor,
                  "floor_exists": n_floor > 0 and not no_halves, "headline1_defined": defined,
                  "value": vals.get(a), "reason": why, "scorable": sc,
                  "in_imputation_term": bool(in_composite and sc and not no_halves),
                  "fallback": fb}
    return out


def _gap_fn(rs):
    keys = ("crps_model", "crps_depth_only", "crps_floor")
    bad = [r for r in rs if any(_undef(r[k]) for k in keys)]
    if bad:
        k = next(k for k in keys if _undef(bad[0][k]))
        return Mx.Undefined(f"{len(bad)} of {len(rs)} pairs undefined: {_why(bad[0][k])}")
    return Mx.gap_closed([r["crps_model"] for r in rs], [r["crps_depth_only"] for r in rs],
                         [r["crps_floor"] for r in rs])


# ------------------------------------------------------------------------------------------ blind ledger

def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ledger_rows(path) -> list[dict]:
    if not Path(path).is_file():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return [dict(r) for r in csv.DictReader(fh, delimiter="\t")]


def ledger_append(path, row: dict) -> None:
    """Append-only: one line per event, never rewritten."""
    path = Path(path)
    new = not path.is_file()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(LEDGER_COLUMNS), delimiter="\t",
                           lineterminator="\n")
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LEDGER_COLUMNS})
        fh.flush()
        os.fsync(fh.fileno())


# ------------------------------------------------------------------------------------------ entry

def write_outputs(scores: dict, suff: dict, out) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "scores.json.tmp"
    tmp.write_text(json.dumps(scores, indent=1, sort_keys=True, allow_nan=False) + "\n",
                   encoding="utf-8")
    np.savez_compressed(out / "suffstats.npz", **{k: np.asarray(v) for k, v in suff.items()})
    tmp.replace(out / "scores.json")
    (out / SCORED).write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n", encoding="utf-8")
    return out / "scores.json"


def write_skipped(out, reason: str) -> Path:
    """`<out>/SKIPPED` (the reason): the job had nothing to score; no scores.json is written."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / SKIPPED).write_text(reason + "\n", encoding="utf-8")
    return out / SKIPPED


def run_tables_sha256(run) -> str:
    """tables_sha256 recorded in the run's checkpoint (ckpt.pt; T1's pinned dict)."""
    p = Path(run) / "ckpt.pt"
    if not p.is_file():
        raise ScoreRefused(f"{p}: no checkpoint; cannot check the run's tables")
    ck = torch.load(p, map_location="cpu", weights_only=False)
    if not ck.get("tables_sha256"):
        raise ScoreRefused(f"{p}: no tables_sha256 in the checkpoint")
    return str(ck["tables_sha256"])


def score(evalset, out, *, run=None, baseline=None, model=None, config=None,
          repeat_blind_reason=None, signal_override=None, **kw) -> dict:
    """Score a run dir, an in-memory model, a baseline name, or an external entry; writes the
    output files. `signal_override` (chunk XE): a callable (target row, regions) -> the entry's
    float32 signal (raw units, like the pipeline's signal.h5) per region, either a list of one
    array per region or the regions end to end; it replaces the model's derived signal in the
    `signal_arcsinh` path, and every count-space metric is written undefined with the reason
    `external entry: signal only`."""
    if sum(x is not None for x in (run, baseline, model, signal_override)) != 1:
        raise ValueError("give exactly one of run, baseline, model, signal_override")
    split = kw.get("split", "val")
    out = Path(out)
    if (split == "blind") != (out.name == "blind"):
        raise ScoreRefused(f"--out {out}: a blind pass writes into a dir named 'blind' "
                           f"(<seed dir>/blind), and only a blind pass does")
    if kw.get("addon"):     # OXe: never over a scores.json of another eval set
        prev = out / "scores.json"
        if prev.is_file():
            old = json.loads(prev.read_text(encoding="utf-8")).get("evalset_sha256")
            if old != sha256_file(evalset):
                raise ScoreRefused(f"{prev} holds a score of another eval set (evalset_sha256 "
                                   f"{str(old)[:12]}); an add-on score goes to <seed dir>/ox")
    run_sha = kw.pop("run_tables_sha256", None)
    if run is not None:
        run_sha = run_tables_sha256(run)
        from minicandi.registry import load_run          # lazy: chunk T1 owns it
        model, config = load_run(run)
    main = (baseline if baseline is not None else EXTERNAL if signal_override is not None
            else "model")
    if baseline is not None and baseline not in REFS:
        raise ValueError(f"unknown baseline {baseline!r}; one of {REFS}")
    if signal_override is not None:
        kw["signal_override"] = signal_override
    if run is not None and kw.get("dump_signal"):
        kw.setdefault("dump_method", str(Path(run).resolve()))
    sc = Scorer(evalset, main=main, model=model, config=config, run_tables_sha256=run_sha, **kw)
    if main in PAIR_ONLY:
        _, cor, rep, _ = sc.jobs()
        if not cor and not rep:
            reason = (f"{main} is defined on pairs only and this eval set has 0 correction and 0 "
                      f"replicate pairs on split {split}"
                      + (" (--no-halves drops every pair)" if sc.no_halves else ""))
            write_skipped(out, reason)
            sc.log(f"skip: {reason}; wrote {out / SKIPPED}")
            return {"skipped": reason}
    ledger, entry = None, None
    if split == "blind":
        ledger = sc.evalset_path.parent / BLIND_LEDGER
        key = (str(Path(run).resolve()) if run is not None else
               f"baseline:{baseline}" if baseline is not None else
               f"external:{out.resolve()}" if signal_override is not None else
               f"model:{out.resolve()}")
        esha = sha256_file(sc.evalset_path)
        prior = [r for r in ledger_rows(ledger) if r["run"] == key and r["evalset_sha256"] == esha]
        reason = (repeat_blind_reason or "").strip()
        if prior and not reason:
            raise ScoreRefused(f"{key} already had {len(prior)} blind-pass ledger rows for this eval "
                               f"set ({ledger}); a repeat needs --repeat-blind-reason \"<text>\"")
        entry = {"run": key, "evalset_sha256": esha, "tables_sha256": sc.tables_sha256,
                 "git_sha": git_sha(), "out": str(out.resolve()),
                 "repeat_reason": reason if prior else ""}
        ledger_append(ledger, dict(entry, time=_utc(), status="started"))
        sc._dump_ledgered = True
    scores = sc.run()
    if run is not None:
        scores["run"] = str(Path(run).resolve())
    if ledger is not None:
        scores["blind_ledger"] = str(ledger)
    write_outputs(scores, sc.suff, out)
    if ledger is not None:
        ledger_append(ledger, dict(entry, time=_utc(), status="scored"))
    return scores


def allocated_cpus() -> int:
    """CPUs this process may run on (the Slurm allocation, not the node's core count)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:                       # pragma: no cover - macOS
        return max(1, os.cpu_count() or 1)


def set_threads() -> int:
    """Torch threads = OMP_NUM_THREADS if set, else the allocated CPUs. Without it torch took the
    node's 192 cores inside a 4-CPU job on Fir (F3: run.sh sets no OMP_NUM_THREADS)."""
    n = int(os.environ.get("OMP_NUM_THREADS") or allocated_cpus())
    torch.set_num_threads(max(1, n))
    return torch.get_num_threads()


def add_dump_arguments(ap) -> None:
    """AGd: --dump-signal <dir> [--dump-targets <file>] (score.py, external_signal.py score,
    baseline_retrieval.py run)."""
    ap.add_argument("--dump-signal", default=None, metavar="DIR",
                    help="also write method track files (plan Amendment 5, interface F) under "
                         "DIR/<regime>/; scores.json is unchanged")
    ap.add_argument("--dump-targets", default=None, metavar="FILE",
                    help="with --dump-signal: dump only these measurement ids (one per line)")


def dump_kwargs(a) -> dict:
    """The Scorer arguments of add_dump_arguments; {} without --dump-signal."""
    if a.dump_targets and not a.dump_signal:
        raise SystemExit("--dump-targets needs --dump-signal")
    if not a.dump_signal:
        return {}
    ids = None
    if a.dump_targets:
        ids = {ln.strip() for ln in Path(a.dump_targets).read_text(encoding="utf-8").splitlines()
               if ln.strip() and not ln.startswith("#")}
    return {"dump_signal": a.dump_signal, "dump_targets": ids}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run")
    g.add_argument("--baseline", choices=REFS)
    ap.add_argument("--evalset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="val", choices=("val", "blind"))
    ap.add_argument("--allow-blind", action="store_true")
    ap.add_argument("--repeat-blind-reason", default=None)
    ap.add_argument("--no-halves", action="store_true",
                    help="pilot tables without pseudoreplicate halves: headline 1 without a noise "
                         "floor, under a key the selection rule refuses")
    ap.add_argument("--dnase-route", default=None,
                    choices=("hotspot2_density", "macs2_no_control"))
    ap.add_argument("--lambda-source", default="auto", choices=LAMBDA_SOURCE_OPTIONS,
                    help="auto (default): the observed control where it exists, else the model's "
                         "predicted control; observed_control: never the predicted control; "
                         "predicted_control: also score every ChIP target with the predicted "
                         "control, under keys with the suffix _predicted_control")
    ap.add_argument("--device", default="auto",
                    help="auto (default: cuda when a GPU is visible, else cpu), cpu or cuda")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--guardrail-windows", type=int, default=256)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--infer-batch", type=int, default=None,
                    help="windows per forward pass while scoring targets (default: the run's "
                         "batch_size); the guardrail keeps batch_size")
    ap.add_argument("--no-region-cache", action="store_true",
                    help="read every tile from disk (the pre-F3 path; for equality checks)")
    ap.add_argument("--addon", action="store_true",
                    help="score an add-on eval set (evalset.py addon; required for one) into "
                         "<seed dir>/ox; the run must be trained on the tables the add-on extends")
    add_dump_arguments(ap)
    a = ap.parse_args(argv)
    if (Path(a.out) / SCORED).is_file() and not a.force:
        print(f"skip: {a.out}/{SCORED} exists")
        return 0
    set_threads()
    device = a.device
    if device == "auto":
        device = "cuda" if (a.run and torch.cuda.is_available()) else "cpu"
    try:
        s = score(a.evalset, a.out, run=a.run, baseline=a.baseline, split=a.split,
                  allow_blind=a.allow_blind, device=device, n_boot=a.n_boot,
                  guardrail_windows=a.guardrail_windows, no_halves=a.no_halves,
                  dnase_route=a.dnase_route, repeat_blind_reason=a.repeat_blind_reason,
                  infer_batch=a.infer_batch, region_cache=not a.no_region_cache,
                  lambda_source=a.lambda_source, **({"addon": True} if a.addon else {}),
                  **dump_kwargs(a))
    except (PermissionError, ScoreRefused) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    if "skipped" in s:
        print(f"skipped: {s['skipped']} ({a.out}/{SKIPPED})")
        return 0
    sel = s["selection"]
    print(f"scores -> {a.out}/scores.json; composite {sel['composite']} (imputation "
          f"{sel['imputation_gap_closed']}, correction {sel['correction_gap_closed']}, uncertainty "
          f"{sel['uncertainty_gap_closed']}); copy_leak_pass {s['copy_leak'].get('copy_leak_pass')}; "
          f"same_view_copy_gain_flag {s['copy_leak'].get('same_view_copy_gain_flag')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
