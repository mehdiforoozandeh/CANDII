"""E6 -- the selection rule D13 and the done rule D14 as code.

    python tools/onion_eval/select_rule.py compare --incumbent <arm dir> --challenger <arm dir>
        [--simpler incumbent|challenger|none] [--name <comparison>] [--settlements <tsv>]
    python tools/onion_eval/select_rule.py table --runs <dir> [--prefix p]
        (columns: arm, seed, state, composite (from the clipped terms while CLIP_GOAL_TERMS),
        <goal>_raw and <goal>_clipped for each goal, copy_leak_pass (the gate as applied below:
        True, False or unreliable), same_view_copy_gain_flag)
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
unless it passed, then PASS, or PENDING while the blind pass is missing.

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
LADDER_KEYS = ("name", "incumbent", "challenger", "simpler", "d14_condition_1", "verdict",
               "notepad_ref")
SEED_RE = re.compile(r"^seed(\d+)$")
SKIPPED = "SKIPPED"                 # score.py: a pair-only baseline with no pair to score
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


def load_seed(path: Path) -> dict:
    """Goal terms, recomputed composite, copy-leak results, denominators and the comparability
    fields of one scores.json."""
    s = json.loads(Path(path).read_text(encoding="utf-8"))
    sel = s.get("selection") or {}
    g = {k: _num(sel.get(GOAL_KEYS[k])) for k in GOALS}
    reasons = {k: sel.get(f"{k}_reason") or "" for k in GOALS}
    cl = s.get("copy_leak") or {}
    den = sel.get("denominators") or {}
    return {"goals": g, "composite": _composite(g), "copy_leak_pass": cl.get("copy_leak_pass"),
            "copy_leak_part1": cl.get("part1_wiring_pass"),
            "gate_on_copy_gain": cl.get("gate_on_copy_gain") is True,
            "copy_gain_spread_ok": cl.get("same_view_copy_gain_spread_ok",
                                          cl.get("copy_leak_spread_ok")),
            "copy_gain_flag": cl.get("same_view_copy_gain_flag"),
            "denominators": {k: _num(den.get(DENOM_KEYS[k][0])) for k in DENOM_KEYS},
            "reason": sel.get("reason", ""), "goal_reasons": reasons, "path": str(path),
            "evalset_sha256": s.get("evalset_sha256") or "",
            "tables_sha256": s.get("tables_sha256") or "", "split": s.get("split"),
            "regime": s.get("regime"), "space": sel.get("space"),
            "noise_floor": s.get("noise_floor"), "evalset": s.get("evalset", "")}


def load_arm(arm, sub: str = "", min_seeds: int = MIN_SEEDS) -> list[dict]:
    """Every seed of an arm with a scores.json (under `sub` inside each seed dir); Missing when
    fewer than min_seeds."""
    seeds = []
    for d in seed_dirs(Path(arm)):
        p = d / sub / "scores.json" if sub else d / "scores.json"
        if p.is_file():
            seeds.append(load_seed(p))
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


def decide(inc: list[dict], ch: list[dict], simpler: str, *, clip: bool | None = None) -> dict:
    """D13 on two arms' seed lists. Returns {verdict, reason, composite_delta, seed_spread,
    per_goal_deltas, per_goal_spreads, copy_leak}. Raises Refused when the rule may not run."""
    if simpler not in ("incumbent", "challenger", "none"):
        raise ValueError("simpler must be incumbent, challenger or none")
    if len(inc) < MIN_SEEDS or len(ch) < MIN_SEEDS:
        raise Refused(f"seeds: incumbent {len(inc)}, challenger {len(ch)}; D13 needs at least {MIN_SEEDS}")
    check_comparable(inc + ch, "val")
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


def append_settlement(path: Path, row: dict) -> None:
    path = Path(path)
    new = not path.is_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(SETTLEMENT_COLUMNS), delimiter="\t",
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


def compare(incumbent, challenger, simpler=None, name=None, settlements=None, *, clip=None) -> dict:
    inc_dir, ch_dir = Path(incumbent), Path(challenger)
    if simpler is None:
        if not is_own_rung_base(inc_dir.name, ch_dir.name):
            raise Refused(f"{inc_dir.name} is not the rung base of {ch_dir.name}: simpler_than_base "
                          f"is relative to the challenger's own rung base, so pass --simpler")
        simpler = simpler_from_config(ch_dir)
    inc = load_arm(inc_dir)
    ch = load_arm(ch_dir)
    res = decide(inc, ch, simpler, clip=clip)
    name = name or f"{inc_dir.name}__vs__{ch_dir.name}"
    row = {"comparison": name, "incumbent": inc_dir.name, "challenger": ch_dir.name,
           "simpler": simpler, "composite_delta": _fmt(res["composite_delta"]),
           "seed_spread": _fmt(res["seed_spread"]), "per_goal_deltas": _fmt(res["per_goal_deltas"]),
           "per_goal_spreads": _fmt(res["per_goal_spreads"]), "copy_leak": res["copy_leak"],
           "verdict": res["verdict"], "time": _utc(),
           "clip_goal_terms": str(res["clip_goal_terms"]).lower(),
           "goal_terms_raw": _fmt_terms(res.get("goal_terms_raw")),
           "goal_terms_clipped": _fmt_terms(res.get("goal_terms_clipped"))}
    append_settlement(Path(settlements) if settlements else ch_dir.parent / "settlements.tsv", row)
    res["comparison"] = name
    return res


# ------------------------------------------------------------------------------------------ table

def seed_state(d: Path) -> str:
    """failed | done (scored: SCORED and scores.json) | skipped | running | queued. Training's DONE
    alone means trained, not scored, so the seed is still `running` (finding 3). A run that has
    started (train.py's log.jsonl or train_state.pt, a checkpoint, or a job log `*.out`) is
    `running` before its first run.json or DONE exists. `skipped`: score.py wrote SKIPPED (a
    pair-only baseline on an eval set with no pair); it has no scores and never counts as done."""
    if (d / "FAILED").exists():
        return "failed"
    if (d / "SCORED").exists() and (d / "scores.json").is_file():
        return "done"
    if (d / SKIPPED).is_file():
        return "skipped"
    if any((d / f).exists() for f in STARTED_FILES) or any(d.glob("*.out")):
        return "running"
    return "queued"


def table(runs, prefix: str = "") -> list[dict]:
    rows = []
    for arm in sorted(p for p in Path(runs).iterdir() if p.is_dir() and p.name.startswith(prefix)):
        for d in seed_dirs(arm):
            st = seed_state(d)
            comp, cl, flag = None, None, None
            raw = clipped = dict.fromkeys(GOALS)
            if (d / "scores.json").is_file():
                s = screen_terms([load_seed(d / "scores.json")])[1][0]
                g = gate_status([s])            # the gate as select_rule applies it (old scores too)
                comp, flag = s["composite"], s["copy_gain_flag"]
                raw, clipped = s["raw_goals"], s["clipped_goals"]
                cl = {"pass": True, "fail": False}.get(g, g)
            rows.append({"arm": arm.name, "seed": int(SEED_RE.match(d.name).group(1)), "state": st,
                         "composite": comp,
                         **{f"{k}_raw": raw[k] for k in GOALS},
                         **{f"{k}_clipped": clipped[k] for k in GOALS},
                         "copy_leak_pass": cl, "same_view_copy_gain_flag": flag})
    return rows


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


def latest_settlements(runs) -> dict:
    """comparison -> verdict of its latest row (greatest `time`; a row without time is oldest;
    ties keep the later row read)."""
    best = {}
    k = 0
    for p in sorted(Path(runs).glob("**/settlements.tsv")):
        with open(p, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                k += 1
                key = (r.get("time") or "", k)
                if r["comparison"] not in best or key >= best[r["comparison"]][0]:
                    best[r["comparison"]] = (key, r["verdict"])
    return {c: v for c, (_, v) in best.items()}


def ladder_verdict(c: dict, settled_rows: dict) -> tuple[str, str]:
    """(verdict, problem) of one ladder comparison; problem '' when the verdict is accepted."""
    v = (c.get("verdict") or "").strip()
    s = (settled_rows.get(c["name"]) or "").strip()
    ref = (c.get("notepad_ref") or "").strip()
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


def check_rung(runs, rung: int, ladder) -> list[str]:
    lad = read_ladder(ladder)
    rows = latest_settlements(runs)
    comps = [c for c in lad.get("comparisons", []) if _rung_of(c) == int(rung)]
    if not comps:
        return [f"no comparison of rung {rung} in {ladder}"]
    out = []
    for c in comps:
        v, prob = ladder_verdict(c, rows)
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


def done_condition_1(runs, ladder) -> tuple[bool, list[str]]:
    runs = Path(runs)
    lad = read_ladder(ladder)
    rows = latest_settlements(runs)
    listed = [c for c in lad.get("comparisons", []) if c.get("d14_condition_1")]
    problems = []
    for r in range(1, 7):
        if not any(_rung_of(c) == r for c in listed):
            problems.append(f"rung {r}: no listed comparison")
    for c in listed:
        v, prob = ladder_verdict(c, rows)
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


def done_condition_2(final, baselines, sub: str = "", *, clip: bool | None = None) -> tuple[str, list[str]]:
    """(PASS|FAIL|ESCALATE|REFUSED|MISSING, lines) on one split (sub '' = validation, 'blind')."""
    split = "blind" if sub else "val"
    try:
        seeds = load_arm(final, sub)
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
        base[b] = load_seed(p)
    try:
        check_comparable(seeds + list(base.values()), split)
    except Refused as e:
        return "REFUSED", [str(e)]
    lines, states = list(skipped), []
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


def check_done(runs, ladder, final=None, baselines=None, blind_baselines=None, *, clip=None) -> dict:
    runs = Path(runs)
    final = Path(final) if final else runs / "final"
    c1, p1 = done_condition_1(runs, ladder)
    baselines = str(baselines) if baselines else _baselines_from(final)
    sv, lv = (done_condition_2(final, baselines, clip=clip) if baselines
              else ("MISSING", ["no baselines dir (no scored final seed0)"]))
    if sv == "PASS":
        bb = str(blind_baselines) if blind_baselines else _baselines_from(final, "blind")
        sb, lb = (done_condition_2(final, bb, "blind", clip=clip) if bb
                  else ("MISSING", ["no blind pass of the final design yet"]))
    else:
        sb, lb = "MISSING", ["blind pass waits for validation"]
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
    return {"condition_1": "PASS" if c1 else "FAIL", "problems_1": p1, "condition_2": c2,
            "validation": (sv, lv), "blind": (sb, lb)}


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
    t = sub.add_parser("table")
    t.add_argument("--runs", required=True)
    t.add_argument("--prefix", default="")
    r = sub.add_parser("check-rung")
    r.add_argument("--runs", required=True)
    r.add_argument("--rung", type=int, required=True)
    r.add_argument("--ladder", required=True)
    d = sub.add_parser("check-done")
    d.add_argument("--runs", required=True)
    d.add_argument("--ladder", required=True)
    d.add_argument("--final")
    d.add_argument("--baselines")
    d.add_argument("--blind-baselines")
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
            res = compare(a.incumbent, a.challenger, a.simpler, a.name, a.settlements)
        except Refused as e:
            print(f"REFUSED: {e}")
            return 2
        print(f"{res['comparison']}: {res['verdict']} ({res['reason']})")
        if res["verdict"] in ESCALATED:
            print(f"ESCALATE: {res['reason']}")
        return 0
    if a.cmd == "table":
        terms = [f"{k}_{v}" for v in ("raw", "clipped") for k in GOALS]
        print("\t".join(["arm", "seed", "state", "composite", *terms, "copy_leak_pass",
                         "same_view_copy_gain_flag"]))
        f = lambda x: "" if x is None else f"{x:.6g}"          # noqa: E731
        for row in table(a.runs, a.prefix):
            print("\t".join([row["arm"], str(row["seed"]), row["state"], f(row["composite"]),
                             *(f(row[t]) for t in terms), str(row["copy_leak_pass"]),
                             str(row["same_view_copy_gain_flag"])]))
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
        missing = check_rung(a.runs, a.rung, a.ladder)
        if missing:
            print(f"rung {a.rung}: not settled: {', '.join(missing)}")
            return 1
        print(f"rung {a.rung}: all comparisons settled or escalated")
        return 0
    if a.cmd == "check-done":
        res = check_done(a.runs, a.ladder, a.final, a.baselines, a.blind_baselines)
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
