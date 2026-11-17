"""E6 -- the selection rule D13 and the done rule D14 as code.

    python tools/onion_eval/select_rule.py compare --incumbent <arm dir> --challenger <arm dir>
        [--simpler incumbent|challenger|none] [--name <comparison>] [--settlements <tsv>]
        [--sub SUB] [--allow-mix scorer_settings.device|scorer_settings.infer_batch|fraglen_table.sha256|input_depth_at_scoring|signal_pileup|signal_box_dependence ...]
    python tools/onion_eval/select_rule.py table --runs <dir> [--prefix p] [--sub SUB]
        [--composite-version 2|3]
        (columns: arm, seed, state, composite (from the clipped terms while CLIP_GOAL_TERMS),
        <goal>_raw and <goal>_clipped for each goal, copy_leak_pass (the gate as applied below:
        True, False or unreliable), same_view_copy_gain_flag)
    python tools/onion_eval/select_rule.py arm-summary --runs <dir> [--prefix p] [--sub SUB]
        (per arm, over its scored seeds: n_seeds, then mean and range (max - min) of the clipped
        composite of version 2 and of version 3 and of each raw goal term, the correction term of
        both versions; plus the correction-only term when the scores hold it; CM)
    python tools/onion_eval/select_rule.py check-rung --runs <dir> --rung <r> --ladder <json>
    python tools/onion_eval/select_rule.py check-done --runs <dir> --ladder <json>
        [--final <arm dir>] [--baselines <dir>] [--blind-baselines <dir>]
    python tools/onion_eval/select_rule.py h1-status <scores.json> [...]
        (per assay: n_targets, n_targets_with_floor, floor_exists, headline1_defined, value,
        scorable, in_imputation_term, fallback, reason; DEFECTS item 5)
    python tools/onion_eval/select_rule.py ladder-add --ladder <json> --name <n> --incumbent <arm>
        --challenger <arm> --simpler <s> --d14 true|false [--verdict <v>] [--notepad-ref <ref>]

An arm dir holds `seed<k>/scores.json` (score.py). Each seed's goal terms are read from
`scores.json["selection"]` (imputation, correction and uncertainty gap closed, on validation,
leave-one-assay-out, counts_log1p, scorable assays); the composite is recomputed from them as
0.5 imputation + 0.25 correction + 0.25 uncertainty, each term clipped to [0, 1] first (PI ruling 18,
see "Goal terms").

Comparability (finding 1). Every seed of both arms (and, for D14 condition 2, every baseline) must
carry the same non-empty `evalset_sha256` and `tables_sha256`, `regime == leave_one_assay_out`,
`selection.space == counts_log1p`, `noise_floor == pseudoreplicate_halves` (a pilot scored with
`score.py --no-halves` is refused), and `split == val` (`split == blind` for the blind pass).
Otherwise the command REFUSES (exit 2, no row written).
Scorer keys (Amendment 10, interface M): every seed of the compared ARMS (both arms of `compare`, the
final design of `check-done`) must also share `scorer_version` (absent = 1), `scorer_settings.device`,
`scorer_settings.infer_batch` and each `refs.sha256_*` (absent = None); a mix is REFUSED. Baselines
are references, not arms: a scorer_version 1 baseline is comparable with version 2 runs.
`--allow-mix <key>` exempts the device, the batch, `fraglen_table.sha256`, `input_depth_at_scoring` (A13:
a native-depth score of a fixed-input-depth run against a normal one, the deployment comparison),
`signal_pileup` or `signal_box_dependence` (E2: the pileup form of the signal routes, absent =
`inbin`, and the bin dependence of the box_d expectation). `--sub SUB` (compare, table) reads
`<arm>/seed<k>/<SUB>/scores.json` (`SCORE_SUBDIR`); table then reads SCORED and SKIPPED there.

D13 (`compare`). Seed spread of a quantity = max - min over seeds, the larger of the two arms
(ruling 7); each arm needs >= 3 seeds, else REFUSED. A spread of exactly 0 (composite or a goal)
over >= 3 seeds means the seeds do not vary: REFUSED. Deltas are challenger mean minus incumbent
mean over seeds. The composite's delta and seed spread use the clipped composite (agent call, see
"Goal terms"); every per-goal delta and spread of the veto uses the raw goal terms.
  RULE_NOT_APPLICABLE_ASK_PI  a goal term is undefined in any seed of either arm; a goal term's
                              denominator is under its tolerance; (only with CLIP_GOAL_TERMS False)
                              a goal term is outside GOAL_BAND (see "Goal terms"); both arms fail
                              the copy-leak gate; or (only with the copy gain gating, see below)
                              the gate was not measured reliably
  INCUMBENT                   the challenger fails the copy-leak gate (it is out); or the
                              incumbent's composite is higher by more than 2x the spread and no goal
                              of the incumbent is lower by more than 2x that goal's spread; or the
                              difference is within the bars (or a win is vetoed by a goal drop) and
                              --simpler incumbent
  CHALLENGER                  the incumbent fails the copy-leak gate and the challenger passes; or
                              the challenger's composite is higher by more than 2x the spread and no
                              goal drops by more than 2x that goal's own spread
  CHALLENGER_AS_SIMPLER       within the bars (or a vetoed win) and --simpler challenger
  TIE_ASK_PI                  within the bars (or a vetoed win) and --simpler none (prints ESCALATE)
Copy-leak gate per arm (the PI's definition, confirmed by PI ruling 17 of 2026-11-11: the hard gate
of D13 is the wiring check only; an arm that fails it is out): `fail` when a seed
has no result (the check did not run) or a seed fails part 1 (`part1_wiring_pass`: perturbing hidden
bins or unavailable tracks moved an output, on any mask seed); else `pass`. The same-view copy gain
(`same_view_copy_gain_flag`, its relative value and mask-seed spread) is shown by `table` and never
gates, escalates or refuses, unless the scores.json says `gate_on_copy_gain` true (written when
guardrail.GATE_ON_COPY_GAIN is True). Then the old gate applies: `unreliable` when a seed's
`same_view_copy_gain_spread_ok` is not true (the gain moved too much across mask seeds to trust
it), else `pass` only when every seed's `copy_leak_pass` is true. A scores.json written before the
switch existed has no `gate_on_copy_gain` and is read with the gate off (part 1 only).

`--simpler` (finding 12 and the simplicity gap). It may be omitted only when the incumbent is the
challenger's own rung base (`rung<k>_base` against `rung<k>_<arm>`); it is then read from the
challenger's `simpler_than_base` (seed0/run.json config, else seed0/config.json), which must be
true, false or null (true -> challenger, false -> incumbent, null -> none). For any other comparison
(cross-rung, arm against arm, a final design) `--simpler` must be given, else REFUSED. Every
verdict appends one row to `settlements.tsv` (default: in the parent dir of the challenger arm)
with a UTC `time` column; the latest row of a comparison is the one with the greatest time. After
`time` the row carries `clip_goal_terms` and every seed's goal terms, raw and clipped
(`goal_terms_raw`, `goal_terms_clipped`: JSON {arm side: {goal: [per seed]}}); these columns come
after `time`, so a settlements.tsv written before them still reads by its header.

Goal terms (finding 4; PI ruling 18 of 2026-11-11 on D13's "0-1 gap closed scale"). Constants:
GOAL_BAND, UNCERTAINTY_DENOM_TOL, CORRECTION_DENOM_REL_TOL, CLIP_GOAL_TERMS, CLIP_RANGE.
CLIP_GOAL_TERMS True (default, the ruling): each goal term is clipped to CLIP_RANGE = [0, 1] INSIDE
THE COMPOSITE only. The per-goal veto ("no single goal drops by more than 2x that goal's own seed
spread") and every seed spread a veto uses are computed on the RAW terms. The raw out-of-band check
(GOAL_BAND) does not escalate in this mode: clipping is the PI's answer to it. Agent call (the ruling
leaves it open): the composite's seed spread, the bar of condition 1 of D13, is the spread of the
clipped composite, because it measures the noise of the quantity being compared. With
CLIP_GOAL_TERMS False (the pre-ruling mode, kept for tests and reanalysis) the composite uses the raw
terms and a term outside GOAL_BAND escalates with the term and its value. In both modes a term whose
denominator is under its tolerance escalates: the marginal-NB PIT deviation (uncertainty) and
(depth-only CRPS - floor CRPS) / depth-only CRPS, the smallest over scorable assays (correction);
score.py writes both under `selection.denominators`.

Oracle floor (OF, PI ruling 47 of 2026-11-14; tools/onion_eval/oracle_floor.py). When a score has an
oracle-floor reference (`oracle_floor/seed0` beside its baselines, same evalset and tables sha256),
its correction term is recomputed at report time from its `pairs_per_region.npz`: the gap closed
against depth-only scaling with the ORACLE as the top of the scale, (depth-only - method) /
(depth-only - O), O = 0.5 x mean |pr1 - pr2| of the target view; per assay a ratio of pair means,
macro over the correction-scorable assays (all headline pairs). The seed then carries
`composite_version` 2 and `gap_closed_floor` "oracle" (also in its scorer keys); its denominator
check reads (depth-only - O) / depth-only, smallest over assays, against CORRECTION_DENOM_REL_TOL.
The scored half-read term stays as `correction_gap_closed_halfread_superseded` (in `table` too). A reference found but no per-pair records: the term is
undefined with the reason (never the old value). No reference: the seed is read as before
(version 1). Every command REFUSES a mix of composite versions (arms and baselines alike). Clipping
to [0, 1] inside the composite is kept: the marginal NB still scores below 0 on the new scale.

Composite version 3 (CM, PI ruling of 2026-11-16; cruxvault/results/correction_metric_review). The
correction term is the gap closed with the oracle floor on the headline pairs with NO
duplicate-keeping side (`oracle_floor` subset `no_nodedup_side`: the crop and crop + single-end
pairs), same per-assay ratio of pair means and the same macro, over the assays that have such a
pair; its denominator check reads that subset's (depth-only - O) / depth-only. The term is
undefined, with the reason, when a correction-scorable assay of the seed that has such a pair in
the eval set (the oracle reference's labels) does not enter it (no per-pair record); the
assays and pairs that entered are reported beside it (`correction_n_assays`, `correction_n_pairs`).
Weights unchanged (0.5, 0.25, 0.25). It is computed at report time only, from the stored
`pairs_per_region.npz`, so every score of version 2 reads as version 3 with no rescoring and
score.py writes no version key. Every command and every python function reads version 3 by
default (DEFAULT_COMPOSITE_VERSION); `--composite-version 2` / `composite_version=2` keeps version 2.
Version 1 (no oracle reference) is unchanged by the option. A seed read as version 2 or 3 also
carries both correction terms (`correction_gap_closed_oracle_all`, `..._no_nodedup_side`) and
`table` prints both composites (`composite_v2`, `composite_v3`).
Settlements by version (CM). Version 3 `compare` writes `settlements_v3.tsv` (beside where
settlements.tsv goes; SETTLEMENT_COLUMNS plus `composite_version`) and refuses, writing nothing, an
arm with a seed that has no oracle reference. `check-rung` and `check-done` under version 3 read
settlements_v3.tsv only; a comparison with no row there is MISSING ("no version-3 comparison
yet"), never FAIL. The legacy settlements.tsv is never touched by version 3: with
`--composite-version 2` every reader and writer is the pre-CM code (same file, columns and
output).

Arm naming and the ladder (pinned). An arm of the ladder is named `rung<k>_<arm>` (k = 1..6; arm
in [a-z0-9_], the rung base is `rung<k>_base`); `_rung_of` reads k from the incumbent.
`ladder.json` schema:

    {"comparisons": [{"name": str,                 unique; the settlements.tsv `comparison`
                      "incumbent": "rung<k>_<arm>",
                      "challenger": "rung<k>_<arm>",
                      "simpler": "incumbent"|"challenger"|"none",
                      "d14_condition_1": true|false,  false for agent extras (ruling 14)
                      "verdict": str,                 "" = take the latest settlements.tsv row
                      "notepad_ref": str}]}           where the notepad records a PI ruling or a
                                                      not-applicable call; "" otherwise

A ladder verdict is accepted (finding 8) when it is "" and a settlements.tsv row exists; a D13
verdict equal to the latest settlements.tsv row of that comparison; `PI_(CHALLENGER|INCUMBENT|
CHALLENGER_AS_SIMPLER)` with a non-empty notepad_ref; or `NOT_APPLICABLE_BY_CONSTRUCTION` with a
non-empty notepad_ref. Anything else is not a settlement.

D14 (`check-done`). Condition 1: every `ladder.json` comparison with `d14_condition_1` true has an
accepted settled verdict (CHALLENGER, INCUMBENT, CHALLENGER_AS_SIMPLER, a PI ruling, or
NOT_APPLICABLE_BY_CONSTRUCTION) and both arms with >= 3 done seeds, and rungs 1 to 6 each have at
least one such comparison. Condition 2: the final design (`--final`, default `<runs>/final`, >= 3
seeds) beats the strongest training-free baseline (marginal_nb, position_average,
depth_only_scaling: the highest defined value of each goal) on EACH goal by more than 2x the final
design's seed spread of that goal; first on validation, then once on blind. Condition 2 compares
goal by goal on the RAW terms (final design and baselines), never clipped: a goal term above 1 or
below 0 against a baseline is information, not noise. The clipped values are printed beside them.
States per split: PASS; FAIL (a defined margin is <= 2x the spread; the run stops); ESCALATE (a lead
metric is undefined, a denominator is under its tolerance, or, only with CLIP_GOAL_TERMS False, a
term is out of band); REFUSED (scores not comparable, or a zero seed spread);
MISSING (not scored yet). Condition 2 overall: FAIL if either split fails, else the validation state
unless it passed, then PASS, or PENDING while the blind pass is missing. PI ruling 24: the blind
state is computed whenever blind scores exist, also after a validation FAIL (the blind tracks are
still scored once); `check-done` prints `validation: <state>` and `blind: <state>`, each with its
per-goal margins and seed spreads.

Blind layout (pinned). The blind eval set is `eval/eic_v2/evalset_blind.json` (built with
`evalset.py build --include-blind` on `tables/eic_v2`). A run's blind pass is scored into
`<runs>/<arm>/seed<k>/blind/` and a baseline's into `eval/eic_v2/baselines/<name>/seed0/blind/`.
`--baselines` defaults to `<dir of the validation evalset>/baselines` and `--blind-baselines` to
`<dir of the blind evalset>/baselines`, each read from the final design's seed0 scores.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

GOALS = ("imputation", "correction", "uncertainty")
WEIGHTS = {"imputation": 0.5, "correction": 0.25, "uncertainty": 0.25}
GOAL_KEYS = {"imputation": "imputation_gap_closed", "correction": "correction_gap_closed",
             "uncertainty": "uncertainty_gap_closed"}
MIN_SEEDS = 3
BAR = 2.0
TRAINING_FREE = ("marginal_nb", "position_average", "depth_only_scaling")
VERDICTS = ("CHALLENGER", "INCUMBENT", "CHALLENGER_AS_SIMPLER", "TIE_ASK_PI",
            "RULE_NOT_APPLICABLE_ASK_PI")
ESCALATED = ("TIE_ASK_PI", "RULE_NOT_APPLICABLE_ASK_PI")
NOT_APPLICABLE = "NOT_APPLICABLE_BY_CONSTRUCTION"
SETTLED = ("CHALLENGER", "INCUMBENT", "CHALLENGER_AS_SIMPLER", NOT_APPLICABLE)
PI_RE = re.compile(r"^PI_(CHALLENGER|INCUMBENT|CHALLENGER_AS_SIMPLER)$")
ARM_RE = re.compile(r"^rung([1-6])_([a-z0-9][a-z0-9_]*)$")
SETTLEMENT_COLUMNS = ("comparison", "incumbent", "challenger", "simpler", "composite_delta",
                      "seed_spread", "per_goal_deltas", "per_goal_spreads", "copy_leak", "verdict",
                      "time", "clip_goal_terms", "goal_terms_raw", "goal_terms_clipped")
#: CM: the settlements of composite version 3 live in their own file, with one more column; the
#: legacy settlements.tsv (version 2 and before) is never read or written by version 3
SETTLEMENTS_FILE = {2: "settlements.tsv", 3: "settlements_v3.tsv"}
SETTLEMENT_COLUMNS_V3 = SETTLEMENT_COLUMNS + ("composite_version",)
V3_MISSING = "no version-3 comparison yet"
LADDER_KEYS = ("name", "incumbent", "challenger", "simpler", "d14_condition_1", "verdict",
               "notepad_ref")
SEED_RE = re.compile(r"^seed(\d+)$")
SKIPPED = "SKIPPED"                 # score.py: a pair-only baseline with no pair to score
#: train.py: the run stopped at --max-windows short of its budget (t209 ASf); never a full run
CAPPED = "CAPPED"
#: any of these in a seed dir means the run has started (seed_state `running`)
STARTED_FILES = ("run.json", "ckpt.pt", "ckpt.pt.tmp", "train_state.pt", "log.jsonl", "DONE",
                 "RUNNING", "scores.json")

# What every compared scores.json must say (finding 1).
EXPECT_REGIME = "leave_one_assay_out"
EXPECT_SPACE = "counts_log1p"
EXPECT_NOISE_FLOOR = "pseudoreplicate_halves"

# Goal terms (finding 4). PI ruling 18 (2026-11-11): clip each term to [0, 1] inside the composite;
# the per-goal veto uses raw terms; a near-zero denominator still escalates.
CLIP_GOAL_TERMS = True
CLIP_RANGE = (0.0, 1.0)
GOAL_BAND = (-1.0, 1.5)            # escalate outside this band only when not clipping
UNCERTAINTY_DENOM_TOL = 0.005      # marginal-NB PIT deviation (mean |F_bar(u) - u|) below this
CORRECTION_DENOM_REL_TOL = 0.01    # (depth-only CRPS - floor CRPS) / depth-only CRPS below this
DENOM_KEYS = {"uncertainty": ("uncertainty_marginal_pit_deviation", UNCERTAINTY_DENOM_TOL),
              "correction": ("correction_baseline_minus_floor_relative_min", CORRECTION_DENOM_REL_TOL)}

# Composite versions with an oracle reference (CM): the oracle_floor subset of the correction term.
CORRECTION_SUBSET = {2: "all", 3: "no_nodedup_side"}
COMPOSITE_VERSIONS = tuple(CORRECTION_SUBSET)
#: the version the command line reads by default (PI ruling 2026-11-16)
DEFAULT_COMPOSITE_VERSION = 3
CLI_COMPOSITE_VERSION = DEFAULT_COMPOSITE_VERSION
#: score.py's correction-only term (CM; reported beside the composite, never inside it)
CORRECTION_ONLY_KEY = "correction_only_covariate_cost_closed"


class Refused(Exception):
    """The rule may not be applied (e.g. fewer than 3 seeds, scores not comparable)."""


class Missing(Refused):
    """Fewer scored seeds than the rule needs (not scored yet)."""


def _num(x):
    if isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def seed_dirs(arm: Path) -> list[Path]:
    return sorted((p for p in Path(arm).iterdir() if p.is_dir() and SEED_RE.match(p.name)),
                  key=lambda p: int(SEED_RE.match(p.name).group(1))) if Path(arm).is_dir() else []


def _composite(g: dict):
    return None if any(g[k] is None for k in GOALS) else sum(WEIGHTS[k] * g[k] for k in GOALS)


def load_seed(path: Path, oracle="auto", composite_version: int = DEFAULT_COMPOSITE_VERSION) -> dict:
    """Goal terms, recomputed composite, copy-leak results, denominators and the comparability
    fields of one scores.json. With an oracle-floor reference (`oracle_floor.find_oracle`, or the
    directory `oracle`; None never uses one) the correction term is the gap closed with the
    oracle floor (PI ruling 47), recomputed from the score's per-pair records, and the seed carries
    `composite_version` 2 (all headline pairs) or 3 (`composite_version=3`: the pairs with no
    duplicate-keeping side, CM); the scored (half-read) term is kept as
    `correction_gap_closed_halfread_superseded`. Without one the seed is read as before
    (`composite_version` 1)."""
    s = json.loads(Path(path).read_text(encoding="utf-8"))
    return seed_from_scores(s, path, oracle, composite_version)


def seed_from_scores(s: dict, path, oracle="auto", composite_version: int = DEFAULT_COMPOSITE_VERSION) -> dict:
    """load_seed on a scores dict already read (score.py calls it before writing scores.json, with
    `path` the scores.json it will write: the per-pair records beside it must exist)."""
    if composite_version not in COMPOSITE_VERSIONS:
        raise ValueError(f"composite_version {composite_version!r}: one of {COMPOSITE_VERSIONS}")
    sel = s.get("selection") or {}
    g = {k: _num(sel.get(GOAL_KEYS[k])) for k in GOALS}
    reasons = {k: sel.get(f"{k}_reason") or "" for k in GOALS}
    cl = s.get("copy_leak") or {}
    den = sel.get("denominators") or {}
    dens = {k: _num(den.get(DENOM_KEYS[k][0])) for k in DENOM_KEYS}
    keys = scorer_keys(s)
    ver, extra = 1, {}
    t = _oracle_terms(path, s, oracle)
    if t is not None:
        ver = composite_version
        blk = t["block"] or {}
        both = {v: _num((blk.get(sub) or {}).get("gap_closed_oracle")) for v, sub in
                CORRECTION_SUBSET.items()}
        nnd_raw = both[3]
        # version 3 needs every correction-scorable assay of the seed that has a pair with no
        # duplicate-keeping side in the eval set
        want = (set(t.get("evalset_assays_no_nodedup_side") or ())
                & set((s.get("scorable") or {}).get("correction") or ()))
        got = set((blk.get(CORRECTION_SUBSET[3]) or {}).get("per_assay") or {})
        v3_why = ""
        if t["block"] is not None and want - got:
            both[3] = None
            v3_why = ("gap closed (oracle floor, pairs with no duplicate-keeping side) undefined: "
                      f"{len(got)} of the {len(want)} correction-scorable assays that have such a "
                      "pair in the eval set entered; missing (no per-pair record): "
                      + ", ".join(sorted(want - got)))
        den_name = "correction_depth_only_minus_oracle_relative_min"
        sb = blk.get(CORRECTION_SUBSET[ver]) or {}
        extra = {"correction_gap_closed_halfread_superseded": g["correction"], "oracle_block": t["block"],
                 "correction_n_assays": sb.get("n_assays"), "correction_n_pairs": sb.get("n_pairs"),
                 "correction_numerator": sb.get("numerator"),
                 "correction_denominator": sb.get("denominator"),
                 "oracle_dir": t["oracle_dir"],
                 "correction_gap_closed_oracle_all": both[2],
                 "correction_gap_closed_oracle_no_nodedup_side": nnd_raw,
                 "correction_n_assays_expected": len(want),
                 "correction_by_version": both,
                 "denominator_names": {"correction": den_name if ver == 2
                                       else den_name + "_no_nodedup_side"}}
        if ver == 2:                     # composite version 2, exactly as before CM
            g["correction"] = _num(t["value"])
            reasons["correction"] = t["reason"]
            dens["correction"] = _num(t["denominator_relative_min"])
        else:                            # version 3: the subset with no duplicate-keeping side
            b = blk.get(CORRECTION_SUBSET[ver])
            g["correction"] = both[ver]
            reasons["correction"] = (t["reason"] if b is None else v3_why or (
                "" if both[ver] is not None else "gap closed (oracle floor, pairs with no "
                "duplicate-keeping side) undefined: " + (b.get("reason") or "")))
            dens["correction"] = None if b is None else _num(b.get("denominator_relative_min"))
        keys.update(composite_version=ver, gap_closed_floor=t["gap_closed_floor"])
    return {**extra, "composite_version": ver,
            "goals": g, "composite": _composite(g), "copy_leak_pass": cl.get("copy_leak_pass"),
            "copy_leak_part1": cl.get("part1_wiring_pass"),
            "gate_on_copy_gain": cl.get("gate_on_copy_gain") is True,
            "copy_gain_spread_ok": cl.get("same_view_copy_gain_spread_ok",
                                          cl.get("copy_leak_spread_ok")),
            "copy_gain_flag": cl.get("same_view_copy_gain_flag"),
            "denominators": dens,
            "reason": sel.get("reason", ""), "goal_reasons": reasons, "path": str(path),
            "evalset_sha256": s.get("evalset_sha256") or "",
            "tables_sha256": s.get("tables_sha256") or "", "split": s.get("split"),
            "regime": s.get("regime"), "space": sel.get("space"),
            "noise_floor": s.get("noise_floor"), "evalset": s.get("evalset", ""),
            "scorer_keys": keys, "correction_only": _num((s.get(CORRECTION_ONLY_KEY) or {}).get("value"))}


def _oracle_terms(path, s: dict, oracle):
    """oracle_floor.seed_terms (None without a reference); numpy is imported only here."""
    if not oracle:
        return None
    tools = str(Path(__file__).resolve().parents[1])      # run as a script: tools/ is not on the path
    if tools not in sys.path:
        sys.path.insert(0, tools)
    from onion_eval import oracle_floor as OF
    try:
        return OF.seed_terms(path, s, oracle)
    except OF.OracleRefused as e:
        raise Refused(str(e)) from None


def check_composite_versions(seeds: list[dict]) -> None:
    """Refused when the seeds (arms AND baselines) carry composites of more than one version: a
    version 3 composite's correction term is the oracle-floor gap closed on the pairs with no
    duplicate-keeping side (CM), version 2's on all headline pairs (PI ruling 47), version 1's the
    half-read floor."""
    vals: dict = {}
    for s in seeds:
        vals.setdefault(s.get("composite_version", 1), []).append(s["path"])
    if len(vals) > 1:
        raise Refused("composite versions differ (version 3: oracle floor, pairs with no "
                      "duplicate-keeping side; version 2: correction gap closed with the oracle "
                      "floor; version 1: the half-read floor, superseded): " + "; ".join(
                          f"version {v}: {len(ps)} scores, e.g. {ps[0]}"
                          for v, ps in sorted(vals.items())))


def load_arm(arm, sub: str = "", min_seeds: int = MIN_SEEDS, oracle="auto",
             composite_version: int = DEFAULT_COMPOSITE_VERSION) -> list[dict]:
    """Every seed of an arm with a scores.json (under `sub` inside each seed dir) and no CAPPED;
    Missing when fewer than min_seeds."""
    seeds = []
    for d in seed_dirs(Path(arm)):
        p = d / sub / "scores.json" if sub else d / "scores.json"
        if p.is_file() and not (d / CAPPED).exists():      # a capped run never counts as a seed
            seeds.append(load_seed(p, oracle, composite_version))
    if len(seeds) < min_seeds:
        raise Missing(f"{arm}: {len(seeds)} scored seeds; the rule needs at least {min_seeds}")
    return seeds


def check_comparable(seeds: list[dict], split: str = "val") -> None:
    """Refused unless every scores.json comes from the same eval set and tables, on `split`, in
    the headline regime and space, with the pseudoreplicate noise floor (finding 1)."""
    bad = []
    for s in seeds:
        p = s["path"]
        if s["split"] != split:
            bad.append(f"{p}: split {s['split']!r}, need {split!r}")
        if s["regime"] != EXPECT_REGIME:
            bad.append(f"{p}: regime {s['regime']!r}, need {EXPECT_REGIME!r}")
        if s["space"] != EXPECT_SPACE:
            bad.append(f"{p}: selection.space {s['space']!r}, need {EXPECT_SPACE!r}")
        if s["noise_floor"] != EXPECT_NOISE_FLOOR:
            bad.append(f"{p}: noise_floor {s['noise_floor']!r}, need {EXPECT_NOISE_FLOOR!r} "
                       f"(a --no-halves pilot score is never a D13/D14 input)")
        for k in ("evalset_sha256", "tables_sha256"):
            if not s[k]:
                bad.append(f"{p}: no {k}")
    for k in ("evalset_sha256", "tables_sha256"):
        vals = sorted({s[k] for s in seeds if s[k]})
        if len(vals) > 1:
            bad.append(f"{k} differs across the compared scores: " + ", ".join(v[:12] for v in vals))
    if bad:
        raise Refused("scores not comparable: " + "; ".join(bad[:6])
                      + (f"; ... {len(bad) - 6} more" if len(bad) > 6 else ""))


# Comparability keys (Amendment 10, pinned interface M; SP2c). Every compared ARM score must carry
# one value of each key; a key absent from a score is the value None, and `scorer_version` absent is
# 1 (the legacy scorer). Reference directories (baselines) are not arms: they are checked by
# check_comparable only, so a scorer_version 1 baseline is a valid reference for version 2 runs.
SCORER_KEYS = ("evalset_sha256", "tables_sha256", "scorer_version", "scorer_settings.device",
               "scorer_settings.infer_batch", "refs.sha256_*", "fraglen_table.sha256",
               "foreign_covariates", "input_depth_at_scoring", "signal_pileup",
               "signal_box_dependence")
#: FL: the fragment-length side table of `score.py --fraglen-table`; absent (a score made without
#: the option, or before the key existed) is None
FRAGLEN_KEY = "fraglen_table.sha256"
#: E2 (PI ruling 2026-11-17): the pileup form of the signal routes; a score made before the key
#: existed used the bin's own count, so an absent key reads `inbin`
PILEUP_KEY = "signal_pileup"
PILEUP_ABSENT = "inbin"
PILEUP_MIX_NOTE = ("signal-space values differ between the box_d and inbin pileup forms; "
                   "count-space values do not depend on it")
#: E2 review: the bin dependence of the box_d expectation (comonotone | independent); absent
#: (inbin, or a score made before the key) is None
DEPENDENCE_KEY = "signal_box_dependence"
#: E5: the comparability key of a score made with foreign covariates at scoring (`mode:seed<n>`)
FOREIGN_KEY = "foreign_covariates"
#: A13: the key of a native-depth score of a fixed-input-depth run (score.py INPUT_DEPTH_KEY); it is
#: written only when present and is mixable, so the deployment comparison (native-depth arm against
#: a normal reference) needs an explicit --allow-mix input_depth_at_scoring
INPUT_DEPTH_KEY = "input_depth_at_scoring"
#: the keys `--allow-mix` may exempt (a cross-rung comparison of a GPU and a CPU forward; FL: a
#: count-space comparison across scores with and without the fragment-length table)
MIXABLE = ("scorer_settings.device", "scorer_settings.infer_batch", FRAGLEN_KEY, INPUT_DEPTH_KEY,
           PILEUP_KEY, DEPENDENCE_KEY)
FRAGLEN_MIX_NOTE = ("signal-space values differ between scores with and without the fragment-length "
                    "table (estimated frag_len for ChIP-seq tracks whose frag_len is empty); "
                    "count-space values do not depend on it")
SUB_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def scorer_keys(s: dict) -> dict:
    """{key: value} of the comparability keys of one scores.json (SCORER_KEYS)."""
    st = s.get("scorer_settings") or {}
    out = {"evalset_sha256": s.get("evalset_sha256") or "",
           "tables_sha256": s.get("tables_sha256") or "",
           "scorer_version": s.get("scorer_version", 1),
           "scorer_settings.device": st.get("device"),
           "scorer_settings.infer_batch": st.get("infer_batch"),
           FRAGLEN_KEY: (s.get("fraglen_table") or {}).get("sha256"),
           PILEUP_KEY: s.get(PILEUP_KEY) or PILEUP_ABSENT,
           DEPENDENCE_KEY: s.get(DEPENDENCE_KEY)}
    refs = s.get("refs") or {}
    out.update({f"refs.{k}": v for k, v in sorted(refs.items()) if k.startswith("sha256_")})
    # E5: a score made with foreign covariates at scoring never compares with a normal score; the
    # key is written only when present, so the keys of every normal score are unchanged
    fc = s.get("foreign_covariates")
    if fc:
        out[FOREIGN_KEY] = f"{fc.get('mode')}:seed{fc.get('seed')}"
    idm = s.get(INPUT_DEPTH_KEY)            # A13: only with the switch
    if idm:
        out[INPUT_DEPTH_KEY] = idm
    t = s.get("__oracle__")             # arm_report: read with an oracle-floor reference (OF)
    if t:
        out.update(composite_version=t["composite_version"], gap_closed_floor=t["gap_closed_floor"])
    return out


def check_scorer_keys(items, allow=()) -> None:
    """Refused unless every arm score in `items` ([(path, scorer_keys dict)]) has one value of each
    comparability key, apart from the keys in `allow` (a subset of MIXABLE)."""
    bad_allow = [k for k in allow if k not in MIXABLE]
    if bad_allow:
        raise Refused(f"--allow-mix {bad_allow}: only {', '.join(MIXABLE)} may be mixed")
    keys = sorted({k for _, kv in items for k in kv})
    bad = []
    for k in keys:
        if k in allow:
            if k == FRAGLEN_KEY and len({json.dumps(kv.get(k)) for _, kv in items}) > 1:
                print(f"warning: --allow-mix {FRAGLEN_KEY}: {FRAGLEN_MIX_NOTE}", file=sys.stderr)
            if k in (PILEUP_KEY, DEPENDENCE_KEY) and len({json.dumps(kv.get(k)) for _, kv in items}) > 1:
                print(f"warning: --allow-mix {PILEUP_KEY}: {PILEUP_MIX_NOTE}", file=sys.stderr)
            continue
        vals: dict = {}
        for p, kv in items:
            vals.setdefault(json.dumps(kv.get(k)), []).append(str(p))
        if len(vals) > 1:
            bad.append(f"{k} differs across the compared arms: " + "; ".join(
                f"{v[:24]} ({len(ps)} scores, e.g. {ps[0]})" for v, ps in sorted(vals.items()))
                + (f" (null = scored without --fraglen-table; {FRAGLEN_MIX_NOTE}; "
                   f"--allow-mix {FRAGLEN_KEY} compares them anyway)" if k == FRAGLEN_KEY else "")
                + (f" ({PILEUP_MIX_NOTE}; --allow-mix {PILEUP_KEY} compares them anyway)"
                   if k in (PILEUP_KEY, DEPENDENCE_KEY) else ""))
    if bad:
        raise Refused("scorer settings not comparable: " + "; ".join(bad))


def sub_name(v: str) -> str:
    """argparse type of `--sub`: one directory name, no path ('' = the seed dir itself)."""
    if v and not SUB_RE.match(v):
        raise argparse.ArgumentTypeError(f"--sub {v!r}: one directory name ([A-Za-z0-9_.-])")
    return v


def _clip(v):
    return None if v is None else min(max(v, CLIP_RANGE[0]), CLIP_RANGE[1])


def screen_terms(seeds: list[dict], clip: bool | None = None) -> tuple[list[tuple], list[dict]]:
    """(problems [(goal, message)], seeds) after the goal-term checks of finding 4 and ruling 18.
    Undefined terms are left for the caller. Every returned seed carries `raw_goals` (as scored),
    `clipped_goals` (each term clipped to CLIP_RANGE), `goals` (the clipped terms with clipping,
    else the raw ones) and `composite` recomputed from `goals`."""
    clip = CLIP_GOAL_TERMS if clip is None else clip
    problems, out = [], []
    for s in seeds:
        raw = dict(s.get("raw_goals", s["goals"]))
        g = dict(raw)
        for k, (key, tol) in DENOM_KEYS.items():
            if g[k] is None:
                continue
            d = s["denominators"].get(k)
            if d is None:
                problems.append((k, f"{s['path']}: {k} denominator ({key}) not recorded"))
            elif d < tol:
                key = (s.get("denominator_names") or {}).get(k, key)
                problems.append((k, f"{s['path']}: {k} denominator {key} = {d:.4g} < tolerance {tol:g}"))
        clipped = {k: _clip(raw[k]) for k in GOALS}
        for k in GOALS:
            if g[k] is None:
                continue
            if clip:
                g[k] = clipped[k]
            elif not GOAL_BAND[0] <= g[k] <= GOAL_BAND[1]:
                problems.append((k, f"{s['path']}: {k} term = {g[k]:.6g} outside the sanity band "
                                    f"[{GOAL_BAND[0]:g}, {GOAL_BAND[1]:g}]"))
        out.append(dict(s, raw_goals=raw, clipped_goals=clipped, goals=g, composite=_composite(g)))
    return problems, out


def _terms_by_side(inc: list[dict], ch: list[dict], key: str) -> dict:
    return {side: {k: [s[key][k] for s in arm] for k in GOALS}
            for side, arm in (("incumbent", inc), ("challenger", ch))}


def spread(vals) -> float:
    return max(vals) - min(vals)


def _mean(vals) -> float:
    return sum(vals) / len(vals)


def gate_status(seeds: list[dict]) -> str:
    """pass | fail | unreliable, the copy-leak gate of one arm (see the module docstring)."""
    if any(s["copy_leak_pass"] is None for s in seeds):
        return "fail"
    if any(s["copy_leak_part1"] is False for s in seeds):
        return "fail"
    if any(s["gate_on_copy_gain"] for s in seeds):          # the old gate, switched on
        if any(s["copy_gain_spread_ok"] is not True for s in seeds):
            return "unreliable"
        return "pass" if all(s["copy_leak_pass"] is True for s in seeds) else "fail"
    ok = [s["copy_leak_part1"] if s["copy_leak_part1"] is not None else s["copy_leak_pass"]
          for s in seeds]
    return "pass" if all(v is True for v in ok) else "fail"


def decide(inc: list[dict], ch: list[dict], simpler: str, *, clip: bool | None = None,
           allow_mix=()) -> dict:
    """D13 on two arms' seed lists. Returns {verdict, reason, composite_delta, seed_spread,
    per_goal_deltas, per_goal_spreads, copy_leak}. Raises Refused when the rule may not run."""
    if simpler not in ("incumbent", "challenger", "none"):
        raise ValueError("simpler must be incumbent, challenger or none")
    if len(inc) < MIN_SEEDS or len(ch) < MIN_SEEDS:
        raise Refused(f"seeds: incumbent {len(inc)}, challenger {len(ch)}; D13 needs at least {MIN_SEEDS}")
    check_comparable(inc + ch, "val")
    check_composite_versions(inc + ch)
    check_scorer_keys([(s["path"], s.get("scorer_keys") or {}) for s in inc + ch], allow_mix)
    clip = CLIP_GOAL_TERMS if clip is None else clip
    out = {"composite_delta": None, "seed_spread": None, "per_goal_deltas": {}, "per_goal_spreads": {},
           "clip_goal_terms": bool(clip)}
    _, si = screen_terms(inc, clip)          # the terms as scored and clipped, for the record
    _, sc = screen_terms(ch, clip)
    out["goal_terms_raw"] = _terms_by_side(si, sc, "raw_goals")
    out["goal_terms_clipped"] = _terms_by_side(si, sc, "clipped_goals")
    gi, gc = gate_status(inc), gate_status(ch)
    out["copy_leak"] = f"incumbent={gi};challenger={gc}"
    if gi == "fail" and gc == "fail":
        return dict(out, verdict="RULE_NOT_APPLICABLE_ASK_PI",
                    reason="both arms fail the copy-leak gate")
    if gc == "fail":
        return dict(out, verdict="INCUMBENT", reason="the challenger fails the copy-leak gate (out)")
    if "unreliable" in (gi, gc):
        return dict(out, verdict="RULE_NOT_APPLICABLE_ASK_PI",
                    reason="copy-leak gate not measured reliably (gate_on_copy_gain): the same-view "
                           "copy gain moved too much across mask seeds "
                           "(same_view_copy_gain_spread_ok false); rescore with more windows")
    undefined = [f"{who} {s['path']}: {k}" for who, arm in (("incumbent", inc), ("challenger", ch))
                 for s in arm for k in GOALS if s["goals"][k] is None]
    if undefined:
        return dict(out, verdict="RULE_NOT_APPLICABLE_ASK_PI",
                    reason="lead metric undefined: " + "; ".join(undefined[:6]))
    p_i, inc = screen_terms(inc, clip)
    p_c, ch = screen_terms(ch, clip)
    if p_i or p_c:
        probs = [m for _, m in p_i + p_c]
        return dict(out, verdict="RULE_NOT_APPLICABLE_ASK_PI",
                    reason="goal term not usable on the D13 gap-closed scale: " + "; ".join(probs[:6]))
    if gi == "fail":
        return dict(out, verdict="CHALLENGER", reason="the incumbent fails the copy-leak gate (out)")
    # composite (clipped terms while clipping): its delta and its seed spread (agent call)
    ci, cc = [s["composite"] for s in inc], [s["composite"] for s in ch]
    d = _mean(cc) - _mean(ci)
    sp = max(spread(ci), spread(cc))
    # per-goal veto: raw terms, raw seed spreads (ruling 18)
    gd = {k: _mean([s["raw_goals"][k] for s in ch]) - _mean([s["raw_goals"][k] for s in inc])
          for k in GOALS}
    gs = {k: max(spread([s["raw_goals"][k] for s in inc]), spread([s["raw_goals"][k] for s in ch]))
          for k in GOALS}
    zero = (["composite"] if sp == 0 else []) + [k for k in GOALS if gs[k] == 0]
    if zero:
        raise Refused(f"seed spread is exactly 0 for {', '.join(zero)} over {len(inc)} and "
                      f"{len(ch)} seeds: the seeds do not vary, so the 2x-spread bars are void")
    out.update(composite_delta=d, seed_spread=sp, per_goal_deltas=gd, per_goal_spreads=gs)
    ch_drops = [k for k in GOALS if -gd[k] > BAR * gs[k]]
    in_drops = [k for k in GOALS if gd[k] > BAR * gs[k]]
    if d > BAR * sp and not ch_drops:
        return dict(out, verdict="CHALLENGER",
                    reason=f"composite +{d:.4g} > 2 x spread {sp:.4g}; no goal drops beyond 2x its spread")
    if -d > BAR * sp and not in_drops:
        return dict(out, verdict="INCUMBENT",
                    reason=f"composite {d:.4g} < -2 x spread {sp:.4g}; no goal of the incumbent drops")
    if d > BAR * sp:
        why = f"challenger's composite win is vetoed: {', '.join(ch_drops)} drops beyond 2x its spread"
    elif -d > BAR * sp:
        why = f"incumbent's composite win is vetoed: {', '.join(in_drops)} lower beyond 2x its spread"
    else:
        why = f"composite delta {d:.4g} within 2 x spread {sp:.4g}"
    if simpler == "incumbent":
        return dict(out, verdict="INCUMBENT", reason=why + "; the simpler incumbent stays")
    if simpler == "challenger":
        return dict(out, verdict="CHALLENGER_AS_SIMPLER", reason=why + "; the simpler challenger stays")
    return dict(out, verdict="TIE_ASK_PI", reason=why + "; neither arm is clearly simpler")


def is_own_rung_base(incumbent: str, challenger: str) -> bool:
    m = ARM_RE.match(challenger)
    return bool(m) and incumbent == f"rung{m.group(1)}_base"


def simpler_from_config(challenger: Path) -> str:
    for d in seed_dirs(challenger)[:1]:
        for name in ("run.json", "config.json"):
            p = d / name
            if p.is_file():
                j = json.loads(p.read_text(encoding="utf-8"))
                cfg = j.get("config", j)
                if "simpler_than_base" in cfg:
                    v = cfg["simpler_than_base"]
                    if not (v is None or isinstance(v, bool)):
                        raise Refused(f"{p}: simpler_than_base = {v!r} ({type(v).__name__}); it "
                                      f"must be true, false or null; pass --simpler")
                    return {True: "challenger", False: "incumbent", None: "none"}[v]
    raise Refused(f"{challenger}: no simpler_than_base in seed0/run.json; pass --simpler")


def append_settlement(path: Path, row: dict, columns=SETTLEMENT_COLUMNS) -> None:
    path = Path(path)
    new = not path.is_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), delimiter="\t",
                           lineterminator="\n", extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def _fmt_terms(x) -> str:
    if not x:
        return ""
    return json.dumps({side: {k: [None if v is None else round(v, 6) for v in vals]
                              for k, vals in goals.items()} for side, goals in x.items()},
                      sort_keys=True, separators=(",", ":"))


def _fmt(x):
    if isinstance(x, dict):
        return json.dumps({k: (None if v is None else round(v, 6)) for k, v in x.items()},
                          sort_keys=True, separators=(",", ":"))
    return "" if x is None else f"{x:.6g}"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def compare(incumbent, challenger, simpler=None, name=None, settlements=None, *, clip=None,
            sub: str = "", allow_mix=(), composite_version: int = DEFAULT_COMPOSITE_VERSION) -> dict:
    inc_dir, ch_dir = Path(incumbent), Path(challenger)
    if simpler is None:
        if not is_own_rung_base(inc_dir.name, ch_dir.name):
            raise Refused(f"{inc_dir.name} is not the rung base of {ch_dir.name}: simpler_than_base "
                          f"is relative to the challenger's own rung base, so pass --simpler")
        simpler = simpler_from_config(ch_dir)
    inc = load_arm(inc_dir, sub, composite_version=composite_version)
    ch = load_arm(ch_dir, sub, composite_version=composite_version)
    v3 = int(composite_version) == 3
    if v3:                               # CM: version 3 needs the oracle reference of every seed
        bad = [s["path"] for s in inc + ch if s.get("composite_version") != 3]
        if bad:
            raise Refused(f"composite version 3 needs an oracle-floor reference for every seed; "
                          f"{len(bad)} seeds have none (e.g. {bad[0]}); nothing written. Use "
                          f"--composite-version 2 for the legacy comparison")
    if settlements:                      # an explicit path must not cross the version boundary
        base = Path(settlements).name
        if v3 and base != SETTLEMENTS_FILE[3]:
            raise Refused(f"composite version 3 writes only a file named {SETTLEMENTS_FILE[3]}; "
                          f"got --settlements {base}; nothing written")
        if not v3 and base == SETTLEMENTS_FILE[3]:
            raise Refused(f"{SETTLEMENTS_FILE[3]} holds version-3 rows only; a version-2 "
                          f"comparison cannot write there; nothing written")
    res = decide(inc, ch, simpler, clip=clip, allow_mix=allow_mix)
    name = name or f"{inc_dir.name}__vs__{ch_dir.name}"
    row = {"comparison": name, "incumbent": inc_dir.name, "challenger": ch_dir.name,
           "simpler": simpler, "composite_delta": _fmt(res["composite_delta"]),
           "seed_spread": _fmt(res["seed_spread"]), "per_goal_deltas": _fmt(res["per_goal_deltas"]),
           "per_goal_spreads": _fmt(res["per_goal_spreads"]), "copy_leak": res["copy_leak"],
           "verdict": res["verdict"], "time": _utc(),
           "clip_goal_terms": str(res["clip_goal_terms"]).lower(),
           "goal_terms_raw": _fmt_terms(res.get("goal_terms_raw")),
           "goal_terms_clipped": _fmt_terms(res.get("goal_terms_clipped"))}
    if v3:
        row["composite_version"] = "3"
        append_settlement(Path(settlements) if settlements else ch_dir.parent / SETTLEMENTS_FILE[3],
                          row, SETTLEMENT_COLUMNS_V3)
    else:                                # legacy: the pre-CM file, columns and row
        append_settlement(Path(settlements) if settlements else ch_dir.parent / "settlements.tsv", row)
    res["comparison"] = name
    res["composite_version"] = inc[0].get("composite_version", 1)
    return res


# ------------------------------------------------------------------------------------------ table

def seed_state(d: Path, sub: str = "") -> str:
    """capped | failed | done (scored: SCORED and scores.json) | skipped | running | queued.
    `capped`: train.py wrote CAPPED (stopped at --max-windows short of the budget); checked first,
    so a capped run is never `done`, whatever else it holds (t209 ASf). Training's DONE
    alone means trained, not scored, so the seed is still `running` (finding 3). A run that has
    started (train.py's log.jsonl or train_state.pt, a checkpoint, or a job log `*.out`) is
    `running` before its first run.json or DONE exists. `skipped`: score.py wrote SKIPPED (a
    pair-only baseline on an eval set with no pair); it has no scores and never counts as done.
    With `sub` (a score subdirectory, interface M) `done`, `skipped` and a score's FAILED are read
    in `<d>/<sub>`; the training states are read in `d` as without it."""
    s = d / sub if sub else d
    if (d / CAPPED).exists():
        return "capped"
    if (d / "FAILED").exists() or (s / "FAILED").exists():
        return "failed"
    if (s / "SCORED").exists() and (s / "scores.json").is_file():
        return "done"
    if (s / SKIPPED).is_file():
        return "skipped"
    if any((d / f).exists() for f in STARTED_FILES) or any(d.glob("*.out")):
        return "running"
    return "queued"


def table(runs, prefix: str = "", sub: str = "", oracle="auto", composite_version: int = DEFAULT_COMPOSITE_VERSION) -> list[dict]:
    """One row per seed. A seed read with an oracle-floor reference (composite version 2 or 3) also
    gets `composite_version`, `correction_gap_closed_halfread_superseded` (the old term, PI ruling 47),
    `correction_gap_closed_oracle_no_nodedup_side`, `correction_numerator` and
    `correction_denominator` (macro over assays of the version's own term: all headline pairs in
    version 2, the pairs with no duplicate-keeping side in version 3), `correction_n_assays`,
    `correction_n_pairs` (what entered that term), and (CM)
    `correction_gap_closed_oracle_all`, `composite_v2` and `composite_v3` (clipped composites with
    either correction term). `correction_only` (CM) when the scores.json holds that term."""
    rows, fc_items = [], []
    for arm in sorted(p for p in Path(runs).iterdir() if p.is_dir() and p.name.startswith(prefix)):
        for d in seed_dirs(arm):
            st = seed_state(d, sub)
            comp, cl, flag = None, None, None
            fc = None
            raw = clipped = dict.fromkeys(GOALS)
            v2 = {}
            sp = d / sub / "scores.json" if sub else d / "scores.json"
            if sp.is_file():
                s = screen_terms([load_seed(sp, oracle, composite_version)])[1][0]
                if s["composite_version"] != 1:
                    blk = s.get("oracle_block") or {}
                    a, b = blk.get("all") or {}, blk.get("no_nodedup_side") or {}
                    v2 = {"composite_version": s["composite_version"],
                          "correction_gap_closed_halfread_superseded":
                              s["correction_gap_closed_halfread_superseded"],
                          "correction_gap_closed_oracle_no_nodedup_side": b.get("gap_closed_oracle"),
                          "correction_numerator": s["correction_numerator"],
                          "correction_denominator": s["correction_denominator"],
                          "correction_n_assays": s["correction_n_assays"],
                          "correction_n_pairs": s["correction_n_pairs"],
                          "correction_gap_closed_oracle_all": a.get("gap_closed_oracle"),
                          **{f"composite_v{v}": _composite(dict(s["clipped_goals"],
                                                                correction=_clip(c)))
                             for v, c in s["correction_by_version"].items()}}
                if s.get("correction_only") is not None:
                    v2["correction_only"] = s["correction_only"]
                fc = (s.get("scorer_keys") or {}).get(FOREIGN_KEY)
                fc_items.append((str(sp), {FOREIGN_KEY: fc}))
                g = gate_status([s])            # the gate as select_rule applies it (old scores too)
                comp, flag = s["composite"], s["copy_gain_flag"]
                raw, clipped = s["raw_goals"], s["clipped_goals"]
                cl = {"pass": True, "fail": False}.get(g, g)
            rows.append({"arm": arm.name, "seed": int(SEED_RE.match(d.name).group(1)), "state": st,
                         "composite": comp,
                         **{f"{k}_raw": raw[k] for k in GOALS},
                         **{f"{k}_clipped": clipped[k] for k in GOALS},
                         "copy_leak_pass": cl, "same_view_copy_gain_flag": flag, **v2,
                         **({FOREIGN_KEY: fc} if fc else {})})
    # E5: one table never holds scores made with foreign covariates beside normal ones, or two
    # foreign modes or donor seeds (only this key is checked here, so the refusals of every
    # existing normal table are unchanged)
    check_scorer_keys(fc_items)
    return rows


#: the extra `table` columns of a seed read with an oracle-floor reference
TABLE_V2_COLUMNS = ("composite_version", "correction_gap_closed_halfread_superseded",
                    "correction_gap_closed_oracle_no_nodedup_side", "correction_numerator",
                    "correction_denominator", "correction_n_assays", "correction_n_pairs",
                    "correction_gap_closed_oracle_all", "composite_v2", "composite_v3")


def arm_summary(runs, prefix: str = "", sub: str = "", oracle="auto") -> list[dict]:
    """CM: per arm, over its done seeds (`table` state done), n_seeds and the mean and range
    (max - min) of: composite_v2, composite_v3 (clipped composites), imputation_raw,
    correction_v2_raw, correction_v3_raw, uncertainty_raw, and correction_only when present. A
    quantity undefined in any seed has mean and range None and its n_defined below n_seeds."""
    rows = [r for r in table(runs, prefix, sub, oracle, 3) if r["state"] == "done"]
    cols = {"composite_v2": "composite_v2", "composite_v3": "composite_v3",
            "imputation_raw": "imputation_raw",
            "correction_v2_raw": "correction_gap_closed_oracle_all",
            "correction_v3_raw": "correction_gap_closed_oracle_no_nodedup_side",
            "uncertainty_raw": "uncertainty_raw", "correction_only": "correction_only"}
    out = []
    for arm in sorted({r["arm"] for r in rows}):
        rs = [r for r in rows if r["arm"] == arm]
        o = {"arm": arm, "n_seeds": len(rs),
             "composite_version": ",".join(sorted({str(r.get("composite_version", 1)) for r in rs}))}
        for name, key in cols.items():
            v = [r.get(key) for r in rs]
            ok = [x for x in v if x is not None]
            o[f"{name}.n_defined"] = len(ok)
            full = len(ok) == len(v)
            o[f"{name}.mean"] = _mean(ok) if ok and full else None
            o[f"{name}.range"] = spread(ok) if ok and full else None
        fcs = {r.get(FOREIGN_KEY) for r in rs} - {None}
        if fcs:                         # E5: only when a score carries it (table refuses a mix)
            o[FOREIGN_KEY] = ",".join(sorted(fcs))
        out.append(o)
    return out


SUMMARY_QUANTITIES = ("composite_v2", "composite_v3", "imputation_raw", "correction_v2_raw",
                      "correction_v3_raw", "uncertainty_raw", "correction_only")


H1_COLUMNS = ("assay", "n_targets", "n_targets_with_floor", "floor_exists", "headline1_defined",
              "value", "scorable", "in_imputation_term", "fallback", "reason")
PILOT_H1_KEY = "specific_variance_recovered_NO_NOISE_FLOOR_pilot_only"


def h1_status(s: dict) -> dict:
    """Per-assay headline-1 status of one scores.json (DEFECTS item 5): whether the noise floor
    exists, whether headline 1 is defined, whether the assay is in the composite's imputation term
    and which fallback is used. score.py writes it under `selection.headline1_per_assay_status`;
    for a scores.json written before that, it is derived from the per-target records (same rules)."""
    st = (s.get("selection") or {}).get("headline1_per_assay_status")
    if st is not None:
        return st
    h1 = (s.get("specific_variance_recovered_above_noise") or {}).get(EXPECT_SPACE) or {}
    pilot = s.get("noise_floor") != EXPECT_NOISE_FLOOR
    src = s.get(PILOT_H1_KEY, h1) if pilot else h1
    vals = dict(src.get("per_assay") or {}, **(src.get("non_scorable_per_assay") or {}))
    reasons = src.get("per_assay_reason") or {}
    scorable = set((s.get("scorable") or {}).get("imputation") or [])
    by: dict = {}
    for t in (h1.get("per_target") or {}).values():
        by.setdefault(t["assay"], []).append(t)
    out = {}
    for a, ts in sorted(by.items()):
        nf = sum(1 for t in ts if t.get("has_halves") and t.get("N") is not None)
        d = vals.get(a) is not None
        fb = ("pilot --no-halves: headline 1 with N = 0; never a selection input" if pilot else
              "none: non-scorable (PI ruling 4)" if a not in scorable else
              "none needed" if d else "none: the imputation term is undefined")
        why = "" if d else reasons.get(a, "")
        if not d and len(ts) < 2:
            why = f"{len(ts)} validation target: the shared/specific split needs >= 2 targets"
        out[a] = {"n_targets": len(ts), "n_targets_with_floor": nf,
                  "floor_exists": nf > 0 and not pilot, "headline1_defined": d, "value": vals.get(a),
                  "scorable": a in scorable, "in_imputation_term": a in scorable and not pilot,
                  "fallback": fb, "reason": why}
    return out


# ------------------------------------------------------------------------------------------ ladder

def read_ladder(path) -> dict:
    p = Path(path)
    lad = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {"comparisons": []}
    for c in lad.get("comparisons", []):
        miss = [k for k in ("name", "incumbent", "challenger") if not c.get(k)]
        if miss:
            raise ValueError(f"{path}: a comparison lacks {miss}: {c}")
    return lad


def latest_settlements(runs, composite_version: int = DEFAULT_COMPOSITE_VERSION) -> dict:
    """comparison -> verdict of its latest row (greatest `time`; a row without time is oldest;
    ties keep the later row read), from the settlements files of `composite_version` only:
    settlements_v3.tsv for version 3, the legacy settlements.tsv for version 2 (CM)."""
    best = {}
    k = 0
    for p in sorted(Path(runs).glob(f"**/{SETTLEMENTS_FILE[int(composite_version)]}")):
        with open(p, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                k += 1
                key = (r.get("time") or "", k)
                if r["comparison"] not in best or key >= best[r["comparison"]][0]:
                    best[r["comparison"]] = (key, r["verdict"])
    return {c: v for c, (_, v) in best.items()}


def ladder_verdict(c: dict, settled_rows: dict, v3: bool = False) -> tuple[str, str]:
    """(verdict, problem) of one ladder comparison; problem '' when the verdict is accepted. With
    `v3` (CM) a comparison that needs a settlements row and has none in settlements_v3.tsv gets the
    problem V3_MISSING."""
    v = (c.get("verdict") or "").strip()
    s = (settled_rows.get(c["name"]) or "").strip()
    ref = (c.get("notepad_ref") or "").strip()
    if v3 and not s and (not v or v in VERDICTS):
        return v, V3_MISSING
    if not v:
        return (s, "") if s else ("", "no verdict in the ladder and no settlements.tsv row")
    if PI_RE.match(v) or v == NOT_APPLICABLE:
        return (v, "") if ref else (v, f"verdict {v} has no notepad_ref")
    if v in VERDICTS:
        if v != s:
            return v, f"ladder verdict {v} does not match the latest settlements.tsv row ({s or 'none'})"
        return v, ""
    return v, f"verdict {v!r} is not a D13 verdict, a PI_ ruling or {NOT_APPLICABLE}"


def _rung_of(c: dict):
    m = ARM_RE.match(c.get("incumbent", ""))
    return int(m.group(1)) if m else None


def check_rung(runs, rung: int, ladder,
               composite_version: int = DEFAULT_COMPOSITE_VERSION) -> list[str]:
    lad = read_ladder(ladder)
    rows = latest_settlements(runs, composite_version)
    comps = [c for c in lad.get("comparisons", []) if _rung_of(c) == int(rung)]
    if not comps:
        return [f"no comparison of rung {rung} in {ladder}"]
    out = []
    for c in comps:
        v, prob = ladder_verdict(c, rows, int(composite_version) == 3)
        if prob:
            out.append(f"{c['name']} ({prob})")
        elif not (v in SETTLED + ESCALATED or PI_RE.match(v)):
            out.append(f"{c['name']} (verdict {v})")
    return out


def ladder_add(ladder, name, incumbent, challenger, simpler, d14: bool, verdict: str = "",
               notepad_ref: str = "") -> dict:
    verdict, notepad_ref = (verdict or "").strip(), (notepad_ref or "").strip()
    if simpler not in ("incumbent", "challenger", "none"):
        raise Refused(f"simpler {simpler!r} must be incumbent, challenger or none")
    if verdict and not (verdict in VERDICTS or verdict == NOT_APPLICABLE or PI_RE.match(verdict)):
        raise Refused(f"verdict {verdict!r}: use one of {VERDICTS}, {NOT_APPLICABLE}, "
                      f"PI_CHALLENGER, PI_INCUMBENT, PI_CHALLENGER_AS_SIMPLER, or '' (settlements)")
    if (PI_RE.match(verdict) or verdict == NOT_APPLICABLE) and not notepad_ref:
        raise Refused(f"verdict {verdict} needs --notepad-ref (where the notepad records it)")
    if d14:
        bad = [a for a in (incumbent, challenger) if not ARM_RE.match(a)]
        if bad:
            raise Refused(f"arm names {bad} do not follow rung<k>_<arm> (k = 1..6)")
    lad = read_ladder(ladder)
    entry = {"name": name, "incumbent": incumbent, "challenger": challenger, "simpler": simpler,
             "d14_condition_1": bool(d14), "verdict": verdict, "notepad_ref": notepad_ref}
    comps = [c for c in lad.get("comparisons", []) if c["name"] != name] + [entry]
    lad["comparisons"] = comps
    Path(ladder).parent.mkdir(parents=True, exist_ok=True)
    Path(ladder).write_text(json.dumps(lad, indent=1) + "\n", encoding="utf-8")
    return entry


def done_condition_1(runs, ladder,
                     composite_version: int = DEFAULT_COMPOSITE_VERSION) -> tuple[bool, list[str]]:
    runs = Path(runs)
    lad = read_ladder(ladder)
    rows = latest_settlements(runs, composite_version)
    listed = [c for c in lad.get("comparisons", []) if c.get("d14_condition_1")]
    problems = []
    for r in range(1, 7):
        if not any(_rung_of(c) == r for c in listed):
            problems.append(f"rung {r}: no listed comparison")
    for c in listed:
        v, prob = ladder_verdict(c, rows, int(composite_version) == 3)
        if prob:
            problems.append(f"{c['name']}: {prob}; not a settlement")
            continue
        if v == NOT_APPLICABLE:
            continue
        if not (v in SETTLED or PI_RE.match(v)):
            problems.append(f"{c['name']}: verdict {v or 'missing'} is not a settlement")
        for side in ("incumbent", "challenger"):
            arm = runs / c[side]
            n = sum(1 for d in seed_dirs(arm) if seed_state(d) == "done")
            if n < MIN_SEEDS:
                problems.append(f"{c['name']}: {side} {c[side]} has {n} done seeds (need {MIN_SEEDS})")
    return not problems, problems


def done_condition_2(final, baselines, sub: str = "", *, clip: bool | None = None,
                     composite_version: int = DEFAULT_COMPOSITE_VERSION) -> tuple[str, list[str]]:
    """(PASS|FAIL|ESCALATE|REFUSED|MISSING, lines) on one split (sub '' = validation, 'blind')."""
    split = "blind" if sub else "val"
    try:
        seeds = load_arm(final, sub, composite_version=composite_version)
    except Missing as e:
        return "MISSING", [str(e)]
    base, skipped = {}, []
    for b in TRAINING_FREE:
        p = Path(baselines) / b / "seed0" / sub / "scores.json" if sub else \
            Path(baselines) / b / "seed0" / "scores.json"
        if not p.is_file() and (p.parent / SKIPPED).is_file():
            why = (p.parent / SKIPPED).read_text(encoding="utf-8").strip()
            skipped.append(f"  baseline {b}: skipped by score.py ({why}); left out of the maximum")
            continue
        if not p.is_file():
            return "MISSING", [f"baseline {b}: {p} missing"]
        base[b] = load_seed(p, composite_version=composite_version)
    try:
        check_comparable(seeds + list(base.values()), split)
        check_composite_versions(seeds + list(base.values()))
    except Refused as e:
        return "REFUSED", [str(e)]
    lines, states = list(skipped), []
    lines.append(f"  composite_version: {seeds[0].get('composite_version', 1)}")
    try:                       # the final design's seeds are arms; the baselines are references
        check_scorer_keys([(s["path"], s["scorer_keys"]) for s in seeds])
    except Refused as e:
        return "REFUSED", [str(e)]
    p_f, seeds = screen_terms(seeds, clip)
    p_b, bl = screen_terms(list(base.values()), clip)
    base = dict(zip(base, bl))
    for g in GOALS:            # raw terms, goal by goal (ruling 18: a term outside [0, 1] is information)
        vals = [s["raw_goals"][g] for s in seeds]
        if any(v is None for v in vals):
            states.append("ESCALATE")
            lines.append(f"  {g}: the final design's lead metric is undefined; ESCALATE")
            continue
        probs = [m for k, m in p_f + p_b if k == g]
        if probs:
            states.append("ESCALATE")
            lines.append(f"  {g}: {probs[0]}; ESCALATE")
            continue
        sp = spread(vals)
        if sp == 0:
            states.append("REFUSED")
            lines.append(f"  {g}: seed spread exactly 0 over {len(vals)} seeds: the seeds do not "
                         f"vary; REFUSED")
            continue
        best, best_name = None, ""
        for b, s in base.items():
            v = s["raw_goals"][g]
            if v is not None and (best is None or v > best):
                best, best_name = v, b
        if best is None:
            states.append("ESCALATE")
            lines.append(f"  {g}: no training-free baseline is defined; ESCALATE")
            continue
        margin = _mean(vals) - best
        good = margin > BAR * sp
        states.append("PASS" if good else "FAIL")
        cf = _mean([s["clipped_goals"][g] for s in seeds])
        cb = base[best_name]["clipped_goals"][g]
        lines.append(f"  {g}: final {_mean(vals):.4f} vs {best_name} {best:.4f} (raw): margin "
                     f"{margin:+.4f}, 2 x seed spread {BAR * sp:.4f} -> "
                     f"{'beats' if good else 'does not beat'}; clipped {cf:.4f} vs {cb:.4f}")
    for st in ("REFUSED", "FAIL", "ESCALATE"):
        if st in states:
            return st, lines
    return "PASS", lines


def _baselines_from(final: Path, sub: str = "") -> str:
    for d in seed_dirs(final)[:1]:
        p = d / sub / "scores.json" if sub else d / "scores.json"
        if p.is_file():
            ev = json.loads(p.read_text(encoding="utf-8")).get("evalset", "")
            return str(Path(ev).parent / "baselines") if ev else ""
    return ""


def check_done(runs, ladder, final=None, baselines=None, blind_baselines=None, *, clip=None,
               composite_version: int = DEFAULT_COMPOSITE_VERSION) -> dict:
    runs = Path(runs)
    final = Path(final) if final else runs / "final"
    c1, p1 = done_condition_1(runs, ladder, composite_version)
    baselines = str(baselines) if baselines else _baselines_from(final)
    sv, lv = (done_condition_2(final, baselines, clip=clip, composite_version=composite_version)
              if baselines
              else ("MISSING", ["no baselines dir (no scored final seed0)"]))
    # PI ruling 24: the blind pass runs once also after a validation failure, so the blind state is
    # computed whenever blind scores exist, whatever the validation state.
    bb = str(blind_baselines) if blind_baselines else _baselines_from(final, "blind")
    sb, lb = (done_condition_2(final, bb, "blind", clip=clip, composite_version=composite_version)
              if bb
              else ("MISSING", ["no blind pass of the final design yet"]))
    if "FAIL" in (sv, sb):
        c2 = "FAIL"
    elif sv != "PASS":
        c2 = sv
    elif sb == "PASS":
        c2 = "PASS"
    elif sb == "MISSING":
        c2 = "PENDING"
    else:
        c2 = sb
    # CM: under version 3 a condition 1 whose only problems are comparisons with no version-3 row
    # yet is MISSING, not FAIL
    st1 = ("PASS" if c1 else "MISSING" if int(composite_version) == 3 and p1
           and all(V3_MISSING in x for x in p1) else "FAIL")
    out = {"condition_1": st1, "problems_1": p1, "condition_2": c2,
           "validation": (sv, lv), "blind": (sb, lb)}
    if int(composite_version) == 3:     # the legacy (version 2) result is the pre-CM dict
        out["composite_version"] = 3
    return out


# ------------------------------------------------------------------------------------------ CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--incumbent", required=True)
    c.add_argument("--challenger", required=True)
    c.add_argument("--simpler", choices=("incumbent", "challenger", "none"))
    c.add_argument("--name")
    c.add_argument("--settlements")
    c.add_argument("--sub", default="", type=sub_name,
                   help="read <arm>/seed<s>/<SUB>/scores.json (interface M)")
    c.add_argument("--allow-mix", action="append", default=[], choices=MIXABLE,
                   help="accept a mix of this scorer setting across the two arms")
    c.add_argument("--composite-version", type=int, choices=COMPOSITE_VERSIONS,
                   default=CLI_COMPOSITE_VERSION)
    t = sub.add_parser("table")
    t.add_argument("--runs", required=True)
    t.add_argument("--prefix", default="")
    t.add_argument("--sub", default="", type=sub_name)
    t.add_argument("--composite-version", type=int, choices=COMPOSITE_VERSIONS,
                   default=CLI_COMPOSITE_VERSION,
                   help="correction term with an oracle reference: 3 (default, pairs with no "
                        "duplicate-keeping side) or 2 (all headline pairs)")
    asum = sub.add_parser("arm-summary")
    asum.add_argument("--runs", required=True)
    asum.add_argument("--prefix", default="")
    asum.add_argument("--sub", default="", type=sub_name)
    r = sub.add_parser("check-rung")
    r.add_argument("--runs", required=True)
    r.add_argument("--rung", type=int, required=True)
    r.add_argument("--ladder", required=True)
    r.add_argument("--composite-version", type=int, choices=COMPOSITE_VERSIONS,
                   default=CLI_COMPOSITE_VERSION,
                   help="3 (default): settlements_v3.tsv only; 2: the legacy settlements.tsv")
    d = sub.add_parser("check-done")
    d.add_argument("--runs", required=True)
    d.add_argument("--ladder", required=True)
    d.add_argument("--final")
    d.add_argument("--baselines")
    d.add_argument("--blind-baselines")
    d.add_argument("--composite-version", type=int, choices=COMPOSITE_VERSIONS,
                   default=CLI_COMPOSITE_VERSION)
    h = sub.add_parser("h1-status")
    h.add_argument("scores", nargs="+", help="scores.json files")
    la = sub.add_parser("ladder-add")
    la.add_argument("--ladder", required=True)
    la.add_argument("--name", required=True)
    la.add_argument("--incumbent", required=True)
    la.add_argument("--challenger", required=True)
    la.add_argument("--simpler", required=True, choices=("incumbent", "challenger", "none"))
    la.add_argument("--d14", required=True, choices=("true", "false"))
    la.add_argument("--verdict", default="")
    la.add_argument("--notepad-ref", default="")
    a = ap.parse_args(argv)

    if a.cmd == "compare":
        try:
            res = compare(a.incumbent, a.challenger, a.simpler, a.name, a.settlements, sub=a.sub,
                          allow_mix=tuple(a.allow_mix), composite_version=a.composite_version)
        except Refused as e:
            print(f"REFUSED: {e}")
            return 2
        print(f"composite_version: {res['composite_version']}")
        print(f"{res['comparison']}: {res['verdict']} ({res['reason']})")
        if res["verdict"] in ESCALATED:
            print(f"ESCALATE: {res['reason']}")
        return 0
    if a.cmd == "table":
        terms = [f"{k}_{v}" for v in ("raw", "clipped") for k in GOALS]
        try:
            rows = table(a.runs, a.prefix, a.sub, composite_version=a.composite_version)
        except Refused as e:
            print(f"REFUSED: {e}")
            return 2
        vers = {r["composite_version"] for r in rows if "composite_version" in r}
        # composite_version is always printed (1 or 2); the other v2 columns only with a reference
        extra = list(TABLE_V2_COLUMNS) if vers else ["composite_version"]
        if any("correction_only" in r for r in rows):
            extra.append("correction_only")
        if any(FOREIGN_KEY in r for r in rows):      # E5: absent for normal scores
            extra.append(FOREIGN_KEY)
        done = [r for r in rows if r["state"] == "done"]
        if len({r.get("composite_version", 1) for r in done}) > 1:
            print("warning: composite versions differ across rows (version 1 rows have no oracle "
                  "reference); never compare them", file=sys.stderr)
        print("\t".join(["arm", "seed", "state", "composite", *terms, "copy_leak_pass",
                         "same_view_copy_gain_flag", *extra]))
        f = lambda x: "" if x is None else f"{x:.6g}"          # noqa: E731
        for row in rows:
            print("\t".join([row["arm"], str(row["seed"]), row["state"], f(row["composite"]),
                             *(f(row[t]) for t in terms), str(row["copy_leak_pass"]),
                             str(row["same_view_copy_gain_flag"]),
                             *((str(row.get(c, 1)) if row["state"] == "done" or c in row else "")
                               if c == "composite_version" else
                               (row.get(c) or "") if c == FOREIGN_KEY else f(row.get(c))
                               for c in extra)]))
        return 0
    if a.cmd == "arm-summary":
        try:
            rows = arm_summary(a.runs, a.prefix, a.sub)
        except Refused as e:
            print(f"REFUSED: {e}")
            return 2
        cols = ["arm", "n_seeds", "composite_version"] + [
            f"{q}.{k}" for q in SUMMARY_QUANTITIES for k in ("mean", "range", "n_defined")]
        if any(FOREIGN_KEY in r for r in rows):      # E5: absent for normal scores
            cols.append(FOREIGN_KEY)
        print("\t".join(cols))
        f = lambda x: "" if x is None else (str(x) if isinstance(x, (int, str)) else f"{x:.4f}")  # noqa: E731
        for r in rows:
            print("\t".join(f(r.get(c)) for c in cols))
        return 0
    if a.cmd == "h1-status":
        print("\t".join(("scores",) + H1_COLUMNS))
        for p in a.scores:
            st = h1_status(json.loads(Path(p).read_text(encoding="utf-8")))
            for assay, v in st.items():
                print("\t".join([p, assay] + ["" if v.get(c) is None else
                                              (f"{v[c]:.6g}" if isinstance(v[c], float) else str(v[c]))
                                              for c in H1_COLUMNS[1:]]))
        return 0
    if a.cmd == "check-rung":
        if a.composite_version == 3:     # version 2: the pre-CM output, line for line
            print(f"composite_version: {a.composite_version}")
        missing = check_rung(a.runs, a.rung, a.ladder, a.composite_version)
        if missing and a.composite_version == 3 and all(V3_MISSING in m for m in missing):
            print(f"rung {a.rung}: MISSING: {', '.join(missing)}")
            return 1
        if missing:
            print(f"rung {a.rung}: not settled: {', '.join(missing)}")
            return 1
        print(f"rung {a.rung}: all comparisons settled or escalated")
        return 0
    if a.cmd == "check-done":
        try:
            res = check_done(a.runs, a.ladder, a.final, a.baselines, a.blind_baselines,
                             composite_version=a.composite_version)
        except Refused as e:
            print(f"REFUSED: {e}")
            return 2
        if "composite_version" in res:
            print(f"composite_version: {res['composite_version']}")
        print(f"D14 condition 1: {res['condition_1']}")
        for p in res["problems_1"]:
            print(f"  {p}")
        print(f"D14 condition 2: {res['condition_2']}")
        for split, (st, lines) in (("validation", res["validation"]), ("blind", res["blind"])):
            print(f"  {split}: {st}")
            for ln in lines:
                print(f"  {ln}")
        return 0 if (res["condition_1"] == "PASS" and res["condition_2"] == "PASS") else 1
    if a.cmd == "ladder-add":
        try:
            e = ladder_add(a.ladder, a.name, a.incumbent, a.challenger, a.simpler, a.d14 == "true",
                           a.verdict, a.notepad_ref)
        except Refused as ex:
            print(f"REFUSED: {ex}")
            return 2
        print(f"ladder: {e['name']} -> {e['verdict'] or '(from settlements.tsv)'}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
