"""E6 -- the selection rule D13 and the done rule D14 as code.

    python tools/onion_eval/select_rule.py compare --incumbent <arm dir> --challenger <arm dir>
        [--simpler incumbent|challenger|none] [--name <comparison>] [--settlements <tsv>]
    python tools/onion_eval/select_rule.py table --runs <dir> [--prefix p]
    python tools/onion_eval/select_rule.py check-rung --runs <dir> --rung <r> --ladder <json>
    python tools/onion_eval/select_rule.py check-done --runs <dir> --ladder <json>
        [--final <arm dir>] [--baselines <dir>]
    python tools/onion_eval/select_rule.py ladder-add --ladder <json> --name <n> --incumbent <arm>
        --challenger <arm> --simpler <s> --d14 true|false --verdict <v>

An arm dir holds `seed<k>/scores.json` (score.py). Each seed's goal terms are read from
`scores.json["selection"]` (imputation, correction and uncertainty gap closed, on validation,
leave-one-assay-out, counts_log1p, scorable assays); the composite is recomputed from them as
0.5 imputation + 0.25 correction + 0.25 uncertainty.

D13 (`compare`). Seed spread of a quantity = max - min over seeds, the larger of the two arms
(ruling 7); each arm needs >= 3 seeds, else the command REFUSES (exit 2, no row written). Deltas are
challenger mean minus incumbent mean over seeds.
  RULE_NOT_APPLICABLE_ASK_PI  a goal term or the composite is undefined in any seed of either arm,
                              or both arms fail the copy-leak gate (prints ESCALATE and the reason)
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
The copy-leak gate passes for an arm when `copy_leak.copy_leak_pass` is true in every seed; a
missing or null value fails it (the check was not run). `--simpler` defaults to the challenger's
`simpler_than_base` (seed0/run.json config, else seed0/config.json): true -> challenger, false ->
incumbent, null -> none. Every verdict appends one row to `settlements.tsv` (default: in the parent
dir of the challenger arm).

D14 (`check-done`). Condition 1: every `ladder.json` comparison with `d14_condition_1` true has both
arms with >= 3 done seeds and a settled verdict (CHALLENGER, INCUMBENT, CHALLENGER_AS_SIMPLER,
NOT_APPLICABLE_BY_CONSTRUCTION, or a PI ruling recorded as `PI_<verdict>`), and rungs 1 to 6 each have
at least one such comparison. An escalated comparison (TIE_ASK_PI, RULE_NOT_APPLICABLE_ASK_PI) is not
settled until the PI's ruling is recorded. Condition 2: the final design (`--final`, default
`<runs>/final`, >= 3 seeds) beats the strongest training-free baseline (marginal_nb,
position_average, depth_only_scaling: the highest defined value of each goal) on EACH goal by more
than 2x the final design's seed spread of that goal; on validation (`seed<k>/scores.json`,
`<baselines>/<name>/seed0/scores.json`) and then on blind (`seed<k>/blind/scores.json`,
`<baselines>/<name>/seed0/blind/scores.json`). Prints PASS, FAIL, or PENDING (validation passes and
the blind pass is not scored yet). `--baselines` defaults to `<dir of the evalset>/baselines`, read
from the final design's seed0 scores.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
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
SETTLED = ("CHALLENGER", "INCUMBENT", "CHALLENGER_AS_SIMPLER", "NOT_APPLICABLE_BY_CONSTRUCTION")
SETTLEMENT_COLUMNS = ("comparison", "incumbent", "challenger", "simpler", "composite_delta",
                      "seed_spread", "per_goal_deltas", "per_goal_spreads", "copy_leak", "verdict")
SEED_RE = re.compile(r"^seed(\d+)$")


class Refused(Exception):
    """The rule may not be applied (e.g. fewer than 3 seeds)."""


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def seed_dirs(arm: Path) -> list[Path]:
    return sorted((p for p in Path(arm).iterdir() if p.is_dir() and SEED_RE.match(p.name)),
                  key=lambda p: int(SEED_RE.match(p.name).group(1))) if Path(arm).is_dir() else []


def load_seed(path: Path) -> dict:
    """Goal terms, recomputed composite and copy-leak pass of one scores.json."""
    s = json.loads(Path(path).read_text(encoding="utf-8"))
    sel = s.get("selection") or {}
    g = {k: _num(sel.get(GOAL_KEYS[k])) for k in GOALS}
    comp = (None if any(v is None for v in g.values())
            else sum(WEIGHTS[k] * g[k] for k in GOALS))
    reasons = {k: (sel.get("uncertainty_reason") if k == "uncertainty" else "") for k in GOALS}
    cl = (s.get("copy_leak") or {}).get("copy_leak_pass")
    return {"goals": g, "composite": comp, "copy_leak_pass": cl, "reason": sel.get("reason", ""),
            "goal_reasons": reasons, "path": str(path)}


def load_arm(arm, sub: str = "", min_seeds: int = MIN_SEEDS) -> list[dict]:
    """Every seed of an arm with a scores.json (under `sub` inside each seed dir); Refused when
    fewer than min_seeds."""
    seeds = []
    for d in seed_dirs(Path(arm)):
        p = d / sub / "scores.json" if sub else d / "scores.json"
        if p.is_file():
            seeds.append(load_seed(p))
    if len(seeds) < min_seeds:
        raise Refused(f"{arm}: {len(seeds)} scored seeds; the rule needs at least {min_seeds}")
    return seeds


def spread(vals) -> float:
    return max(vals) - min(vals)


def _mean(vals) -> float:
    return sum(vals) / len(vals)


def decide(inc: list[dict], ch: list[dict], simpler: str) -> dict:
    """D13 on two arms' seed lists. Returns {verdict, reason, composite_delta, seed_spread,
    per_goal_deltas, per_goal_spreads, copy_leak}."""
    if simpler not in ("incumbent", "challenger", "none"):
        raise ValueError("simpler must be incumbent, challenger or none")
    if len(inc) < MIN_SEEDS or len(ch) < MIN_SEEDS:
        raise Refused(f"seeds: incumbent {len(inc)}, challenger {len(ch)}; D13 needs at least {MIN_SEEDS}")
    out = {"composite_delta": None, "seed_spread": None, "per_goal_deltas": {}, "per_goal_spreads": {}}
    pass_i = all(s["copy_leak_pass"] is True for s in inc)
    pass_c = all(s["copy_leak_pass"] is True for s in ch)
    out["copy_leak"] = f"incumbent={'pass' if pass_i else 'fail'};challenger={'pass' if pass_c else 'fail'}"
    if not pass_i and not pass_c:
        return dict(out, verdict="RULE_NOT_APPLICABLE_ASK_PI",
                    reason="both arms fail the copy-leak gate")
    if not pass_c:
        return dict(out, verdict="INCUMBENT", reason="the challenger fails the copy-leak gate (out)")
    undefined = [f"{who} {s['path']}: {k}" for who, arm in (("incumbent", inc), ("challenger", ch))
                 for s in arm for k in GOALS if s["goals"][k] is None]
    if undefined:
        return dict(out, verdict="RULE_NOT_APPLICABLE_ASK_PI",
                    reason="lead metric undefined: " + "; ".join(undefined[:6]))
    if not pass_i:
        return dict(out, verdict="CHALLENGER", reason="the incumbent fails the copy-leak gate (out)")
    ci, cc = [s["composite"] for s in inc], [s["composite"] for s in ch]
    d = _mean(cc) - _mean(ci)
    sp = max(spread(ci), spread(cc))
    gd = {k: _mean([s["goals"][k] for s in ch]) - _mean([s["goals"][k] for s in inc]) for k in GOALS}
    gs = {k: max(spread([s["goals"][k] for s in inc]), spread([s["goals"][k] for s in ch]))
          for k in GOALS}
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


def simpler_from_config(challenger: Path) -> str:
    for d in seed_dirs(challenger)[:1]:
        for name in ("run.json", "config.json"):
            p = d / name
            if p.is_file():
                j = json.loads(p.read_text(encoding="utf-8"))
                cfg = j.get("config", j)
                if "simpler_than_base" in cfg:
                    v = cfg["simpler_than_base"]
                    return {True: "challenger", False: "incumbent", None: "none"}[v]
    raise Refused(f"{challenger}: no simpler_than_base in seed0/run.json; pass --simpler")


def append_settlement(path: Path, row: dict) -> None:
    path = Path(path)
    new = not path.is_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(SETTLEMENT_COLUMNS), delimiter="\t",
                           lineterminator="\n")
        if new:
            w.writeheader()
        w.writerow(row)


def _fmt(x):
    if isinstance(x, dict):
        return json.dumps({k: (None if v is None else round(v, 6)) for k, v in x.items()},
                          sort_keys=True, separators=(",", ":"))
    return "" if x is None else f"{x:.6g}"


def compare(incumbent, challenger, simpler=None, name=None, settlements=None) -> dict:
    inc_dir, ch_dir = Path(incumbent), Path(challenger)
    if simpler is None:
        simpler = simpler_from_config(ch_dir)
    inc = load_arm(inc_dir)
    ch = load_arm(ch_dir)
    res = decide(inc, ch, simpler)
    name = name or f"{inc_dir.name}__vs__{ch_dir.name}"
    row = {"comparison": name, "incumbent": inc_dir.name, "challenger": ch_dir.name,
           "simpler": simpler, "composite_delta": _fmt(res["composite_delta"]),
           "seed_spread": _fmt(res["seed_spread"]), "per_goal_deltas": _fmt(res["per_goal_deltas"]),
           "per_goal_spreads": _fmt(res["per_goal_spreads"]), "copy_leak": res["copy_leak"],
           "verdict": res["verdict"]}
    append_settlement(Path(settlements) if settlements else ch_dir.parent / "settlements.tsv", row)
    res["comparison"] = name
    return res


# ------------------------------------------------------------------------------------------ table

def seed_state(d: Path) -> str:
    if (d / "FAILED").exists():
        return "failed"
    if (d / "DONE").exists() and (d / "scores.json").is_file():
        return "done"
    if any((d / f).exists() for f in ("run.json", "ckpt.pt", "DONE", "RUNNING")):
        return "running"
    return "queued"


def table(runs, prefix: str = "") -> list[dict]:
    rows = []
    for arm in sorted(p for p in Path(runs).iterdir() if p.is_dir() and p.name.startswith(prefix)):
        for d in seed_dirs(arm):
            st = seed_state(d)
            comp, cl = None, None
            if (d / "scores.json").is_file():
                s = load_seed(d / "scores.json")
                comp, cl = s["composite"], s["copy_leak_pass"]
            rows.append({"arm": arm.name, "seed": int(SEED_RE.match(d.name).group(1)), "state": st,
                         "composite": comp, "copy_leak_pass": cl})
    return rows


# ------------------------------------------------------------------------------------------ ladder

def read_ladder(path) -> dict:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {"comparisons": []}


def latest_settlements(runs) -> dict:
    out = {}
    for p in sorted(Path(runs).glob("**/settlements.tsv")):
        with open(p, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                out[r["comparison"]] = r["verdict"]
    return out


def verdict_of(c: dict, settled_rows: dict) -> str:
    return (c.get("verdict") or settled_rows.get(c["name"]) or "").strip()


def _rung_of(c: dict):
    m = re.match(r"^rung(\d+)_", c.get("incumbent", ""))
    return int(m.group(1)) if m else None


def check_rung(runs, rung: int, ladder) -> list[str]:
    lad = read_ladder(ladder)
    rows = latest_settlements(runs)
    comps = [c for c in lad.get("comparisons", []) if _rung_of(c) == int(rung)]
    if not comps:
        return [f"no comparison of rung {rung} in {ladder}"]
    return [c["name"] for c in comps
            if not (verdict_of(c, rows) in SETTLED + ESCALATED or verdict_of(c, rows).startswith("PI_"))]


def ladder_add(ladder, name, incumbent, challenger, simpler, d14: bool, verdict: str) -> dict:
    lad = read_ladder(ladder)
    entry = {"name": name, "incumbent": incumbent, "challenger": challenger, "simpler": simpler,
             "d14_condition_1": bool(d14), "verdict": verdict}
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
        v = verdict_of(c, rows)
        if v == "NOT_APPLICABLE_BY_CONSTRUCTION":
            continue
        if not (v in SETTLED or v.startswith("PI_")):
            problems.append(f"{c['name']}: verdict {v or 'missing'} is not a settlement")
        for side in ("incumbent", "challenger"):
            arm = runs / c[side]
            n = sum(1 for d in seed_dirs(arm) if seed_state(d) == "done")
            if n < MIN_SEEDS:
                problems.append(f"{c['name']}: {side} {c[side]} has {n} done seeds (need {MIN_SEEDS})")
    return not problems, problems


def done_condition_2(final, baselines, sub: str = "") -> tuple[str, list[str]]:
    """('PASS'|'FAIL'|'MISSING', lines) on one split (sub '' = validation, 'blind')."""
    try:
        seeds = load_arm(final, sub)
    except Refused as e:
        return "MISSING", [str(e)]
    lines, ok = [], True
    for g in GOALS:
        vals = [s["goals"][g] for s in seeds]
        if any(v is None for v in vals):
            return "FAIL", lines + [f"{g}: the final design's lead metric is undefined; ESCALATE"]
        sp = spread(vals)
        best, best_name = None, ""
        for b in TRAINING_FREE:
            p = Path(baselines) / b / "seed0" / sub / "scores.json" if sub else \
                Path(baselines) / b / "seed0" / "scores.json"
            if not p.is_file():
                return "MISSING", lines + [f"baseline {b}: {p} missing"]
            v = load_seed(p)["goals"][g]
            if v is not None and (best is None or v > best):
                best, best_name = v, b
        if best is None:
            return "FAIL", lines + [f"{g}: no training-free baseline is defined; ESCALATE"]
        margin = _mean(vals) - best
        good = margin > BAR * sp
        ok &= good
        lines.append(f"  {g}: final {_mean(vals):.4f} vs {best_name} {best:.4f}: margin {margin:+.4f}, "
                     f"2 x seed spread {BAR * sp:.4f} -> {'beats' if good else 'does not beat'}")
    return ("PASS" if ok else "FAIL"), lines


def check_done(runs, ladder, final=None, baselines=None) -> dict:
    runs = Path(runs)
    final = Path(final) if final else runs / "final"
    c1, p1 = done_condition_1(runs, ladder)
    if baselines is None:
        baselines = ""
        for d in seed_dirs(final)[:1]:
            if (d / "scores.json").is_file():
                ev = json.loads((d / "scores.json").read_text(encoding="utf-8")).get("evalset", "")
                baselines = str(Path(ev).parent / "baselines") if ev else ""
    sv, lv = done_condition_2(final, baselines) if baselines else ("MISSING", ["no baselines dir"])
    sb, lb = (done_condition_2(final, baselines, "blind") if sv == "PASS"
              else ("MISSING", ["blind pass waits for validation"]))
    if sv == "FAIL" or sb == "FAIL":
        c2 = "FAIL"
    elif sv == "PASS" and sb == "PASS":
        c2 = "PASS"
    elif sv == "PASS":
        c2 = "PENDING"
    else:
        c2 = "FAIL"
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
    la = sub.add_parser("ladder-add")
    la.add_argument("--ladder", required=True)
    la.add_argument("--name", required=True)
    la.add_argument("--incumbent", required=True)
    la.add_argument("--challenger", required=True)
    la.add_argument("--simpler", required=True, choices=("incumbent", "challenger", "none"))
    la.add_argument("--d14", required=True, choices=("true", "false"))
    la.add_argument("--verdict", required=True)
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
        print("arm\tseed\tstate\tcomposite\tcopy_leak_pass")
        for row in table(a.runs, a.prefix):
            comp = "" if row["composite"] is None else f"{row['composite']:.6g}"
            print(f"{row['arm']}\t{row['seed']}\t{row['state']}\t{comp}\t{row['copy_leak_pass']}")
        return 0
    if a.cmd == "check-rung":
        missing = check_rung(a.runs, a.rung, a.ladder)
        if missing:
            print(f"rung {a.rung}: not settled: {', '.join(missing)}")
            return 1
        print(f"rung {a.rung}: all comparisons settled or escalated")
        return 0
    if a.cmd == "check-done":
        res = check_done(a.runs, a.ladder, a.final, a.baselines)
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
        e = ladder_add(a.ladder, a.name, a.incumbent, a.challenger, a.simpler, a.d14 == "true",
                       a.verdict)
        print(f"ladder: {e['name']} -> {e['verdict']}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
