"""E6 selection rule D13 and done rule D14 on hand-built score tables (no model, no data)."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from onion_eval import select_rule as R  # noqa: E402


def write_seed(d: Path, imp, cor, unc, leak=True, sub="", simpler=None, done=True, **meta):
    """A hand-built scores.json with every field the rule checks; `meta` overrides top-level keys
    (evalset_sha256, tables_sha256, split, regime, noise_floor), `space`, `denominators` and
    `spread_ok`."""
    d.mkdir(parents=True, exist_ok=True)
    tgt = d / sub if sub else d
    tgt.mkdir(parents=True, exist_ok=True)
    comp = None if None in (imp, cor, unc) else 0.5 * imp + 0.25 * cor + 0.25 * unc
    den = meta.pop("denominators", {"uncertainty_marginal_pit_deviation": 0.1,
                                    "correction_baseline_minus_floor_relative_min": 0.2})
    s = {"selection": {"space": meta.pop("space", "counts_log1p"), "imputation_gap_closed": imp,
                       "correction_gap_closed": cor, "uncertainty_gap_closed": unc,
                       "composite": comp, "reason": "" if comp is not None else "undefined",
                       "denominators": den},
         "copy_leak": {"copy_leak_pass": leak, "part1_wiring_pass": leak,
                       "copy_leak_spread_ok": meta.pop("spread_ok", True)},
         "evalset_sha256": "E" * 64 if not sub else "B" * 64, "tables_sha256": "T" * 64,
         "split": "blind" if sub == "blind" else "val", "regime": "leave_one_assay_out",
         "noise_floor": "pseudoreplicate_halves",
         "evalset": "/x/eval/eic_v2/evalset_blind.json" if sub else "/x/eval/eic_v1/evalset.json"}
    s.update(meta)
    (tgt / "scores.json").write_text(json.dumps(s))
    (tgt / "SCORED").write_text("x")
    if done:
        (d / "DONE").write_text("x")
    if simpler is not None or simpler is None:
        (d / "run.json").write_text(json.dumps({"config": {"simpler_than_base": simpler}}))


def make_arm(root: Path, name: str, goals, leak=True, simpler=None, sub="", **meta):
    """goals: list over seeds of (imputation, correction, uncertainty)."""
    arm = root / name
    for k, g in enumerate(goals):
        write_seed(arm / f"seed{k}", *g, leak=leak if not isinstance(leak, list) else leak[k],
                   simpler=simpler, sub=sub, **meta)
    return arm


BASE = [(0.30, 0.40, 0.50), (0.31, 0.41, 0.50), (0.30, 0.40, 0.51)]


def shift(goals, di=0.0, dc=0.0, du=0.0):
    return [(i + di, c + dc, u + du) for i, c, u in goals]


def run_compare(tmp_path, inc, ch, simpler):
    return R.compare(inc, ch, simpler, settlements=tmp_path / "settlements.tsv")


def test_clear_win(tmp_path):
    inc = make_arm(tmp_path, "rung1_base", BASE)
    ch = make_arm(tmp_path, "rung1_x", shift(BASE, di=0.10))
    res = run_compare(tmp_path, inc, ch, "incumbent")
    assert res["verdict"] == "CHALLENGER"
    assert res["composite_delta"] == pytest.approx(0.05)
    rows = list(csv.DictReader(open(tmp_path / "settlements.tsv"), delimiter="\t"))
    assert tuple(rows[0].keys()) == R.SETTLEMENT_COLUMNS and rows[0]["verdict"] == "CHALLENGER"


def test_clear_loss(tmp_path):
    inc = make_arm(tmp_path, "rung1_base", BASE)
    ch = make_arm(tmp_path, "rung1_x", shift(BASE, di=-0.10))
    assert run_compare(tmp_path, inc, ch, "challenger")["verdict"] == "INCUMBENT"


@pytest.mark.parametrize("simpler,verdict", [("incumbent", "INCUMBENT"),
                                             ("challenger", "CHALLENGER_AS_SIMPLER"),
                                             ("none", "TIE_ASK_PI")])
def test_tie_uses_simpler(tmp_path, simpler, verdict):
    inc = make_arm(tmp_path, "rung1_base", BASE)
    ch = make_arm(tmp_path, "rung1_x", shift(BASE, di=0.004))   # within 2x spread (0.005 x 2)
    res = run_compare(tmp_path, inc, ch, simpler)
    assert res["verdict"] == verdict


def test_tie_escalate_prints(tmp_path, capsys):
    inc = make_arm(tmp_path, "rung1_base", BASE)
    ch = make_arm(tmp_path, "rung1_x", shift(BASE, di=0.004))
    rc = R.main(["compare", "--incumbent", str(inc), "--challenger", str(ch), "--simpler", "none",
                 "--settlements", str(tmp_path / "s.tsv")])
    out = capsys.readouterr().out
    assert rc == 0 and "TIE_ASK_PI" in out and "ESCALATE:" in out


def test_one_goal_drop_vetoes_a_composite_win(tmp_path):
    inc = make_arm(tmp_path, "rung1_base", BASE)
    # imputation +0.2 (composite +0.1 - 0.0375 = +0.0625) but correction drops by 0.15 >> 2 x 0.01
    ch = make_arm(tmp_path, "rung1_x", shift(BASE, di=0.20, dc=-0.15))
    res = run_compare(tmp_path, inc, ch, "incumbent")
    assert res["composite_delta"] > 2 * res["seed_spread"]
    assert res["verdict"] == "INCUMBENT" and "vetoed" in res["reason"]
    res2 = R.decide(R.load_arm(inc), R.load_arm(ch), "none")
    assert res2["verdict"] == "TIE_ASK_PI"


def test_goal_drop_within_its_spread_does_not_veto(tmp_path):
    inc = make_arm(tmp_path, "rung1_base", [(0.30, 0.40, 0.5), (0.31, 0.50, 0.5), (0.30, 0.45, 0.51)])
    ch = make_arm(tmp_path, "rung1_x", [(0.50, 0.35, 0.51), (0.51, 0.45, 0.5), (0.50, 0.40, 0.5)])
    # correction drops 0.05 < 2 x spread 0.10; composite +0.0875 > 2 x composite spread 0.025 x 2
    assert run_compare(tmp_path, inc, ch, "incumbent")["verdict"] == "CHALLENGER"


def test_seed_spread_is_max_minus_min_larger_arm(tmp_path):
    # correction + uncertainty sum to 0.91 in every seed, so only imputation moves the composite
    inc = make_arm(tmp_path, "rung1_base", [(0.30, 0.40, 0.51), (0.30, 0.41, 0.50), (0.30, 0.40, 0.51)])
    ch = make_arm(tmp_path, "rung1_x", [(0.36, 0.40, 0.51), (0.40, 0.41, 0.50), (0.44, 0.40, 0.51)])
    res = R.decide(R.load_arm(inc), R.load_arm(ch), "incumbent")
    assert res["seed_spread"] == pytest.approx(0.5 * 0.08)          # challenger's range, not SD
    assert res["per_goal_spreads"]["imputation"] == pytest.approx(0.08)
    # delta = 0.05 composite, 2 x 0.04 = 0.08: within the bars, so the simpler incumbent stays
    assert res["verdict"] == "INCUMBENT" and "within" in res["reason"]
    ch2 = make_arm(tmp_path, "rung1_y", [(0.46, 0.40, 0.51), (0.50, 0.41, 0.50), (0.54, 0.40, 0.51)])
    assert R.decide(R.load_arm(inc), R.load_arm(ch2), "incumbent")["verdict"] == "CHALLENGER"


def test_guardrail_fail_is_out(tmp_path):
    inc = make_arm(tmp_path, "rung1_base", BASE)
    ch = make_arm(tmp_path, "rung1_x", shift(BASE, di=0.30), leak=[True, False, True])
    res = run_compare(tmp_path, inc, ch, "challenger")
    assert res["verdict"] == "INCUMBENT" and "copy-leak" in res["reason"]
    inc2 = make_arm(tmp_path, "rung1_b2", BASE, leak=False)
    ch2 = make_arm(tmp_path, "rung1_y", shift(BASE, di=-0.1))
    assert R.decide(R.load_arm(inc2), R.load_arm(ch2), "incumbent")["verdict"] == "CHALLENGER"
    both = R.decide(R.load_arm(inc2), R.load_arm(ch), "incumbent")
    assert both["verdict"] == "RULE_NOT_APPLICABLE_ASK_PI"


def test_missing_guardrail_fails_closed(tmp_path):
    inc = make_arm(tmp_path, "rung1_base", BASE)
    ch = make_arm(tmp_path, "rung1_x", shift(BASE, di=0.3), leak=None)
    assert run_compare(tmp_path, inc, ch, "challenger")["verdict"] == "INCUMBENT"


def test_undefined_metric_escalates(tmp_path, capsys):
    inc = make_arm(tmp_path, "rung1_base", BASE)
    ch = make_arm(tmp_path, "rung1_x", [(0.4, None, 0.5), (0.4, 0.4, 0.5), (0.4, 0.4, 0.5)])
    rc = R.main(["compare", "--incumbent", str(inc), "--challenger", str(ch), "--simpler",
                 "challenger", "--settlements", str(tmp_path / "s.tsv")])
    out = capsys.readouterr().out
    assert rc == 0 and "RULE_NOT_APPLICABLE_ASK_PI" in out and "ESCALATE:" in out
    assert "correction" in out


def test_fewer_than_three_seeds_refused(tmp_path, capsys):
    inc = make_arm(tmp_path, "rung1_base", BASE)
    ch = make_arm(tmp_path, "rung1_x", shift(BASE, di=0.3)[:2])
    rc = R.main(["compare", "--incumbent", str(inc), "--challenger", str(ch), "--simpler",
                 "incumbent", "--settlements", str(tmp_path / "s.tsv")])
    assert rc == 2 and "REFUSED" in capsys.readouterr().out
    assert not (tmp_path / "s.tsv").exists()
    with pytest.raises(R.Refused):
        R.decide(R.load_arm(inc), R.load_arm(inc, min_seeds=1)[:2], "incumbent")


def test_simpler_read_from_config(tmp_path):
    inc = make_arm(tmp_path, "rung1_base", BASE)
    ch = make_arm(tmp_path, "rung1_dna_off", shift(BASE, di=0.002), simpler=True)
    assert R.simpler_from_config(ch) == "challenger"
    res = R.compare(inc, ch, None, settlements=tmp_path / "s.tsv")
    assert res["verdict"] == "CHALLENGER_AS_SIMPLER"
    ch2 = make_arm(tmp_path, "rung1_tokens", shift(BASE, di=0.002), simpler=False)
    assert R.compare(inc, ch2, None, settlements=tmp_path / "s.tsv")["verdict"] == "INCUMBENT"


def test_table_states(tmp_path, capsys):
    make_arm(tmp_path, "rung1_base", BASE)
    (tmp_path / "rung1_q" / "seed0").mkdir(parents=True)
    (tmp_path / "rung1_r" / "seed0").mkdir(parents=True)
    (tmp_path / "rung1_r" / "seed0" / "ckpt.pt").write_text("x")
    (tmp_path / "rung1_f" / "seed0").mkdir(parents=True)
    (tmp_path / "rung1_f" / "seed0" / "FAILED").write_text("x")
    rows = R.table(tmp_path, "rung1_")
    st = {(r["arm"], r["seed"]): r["state"] for r in rows}
    assert st[("rung1_base", 0)] == "done" and st[("rung1_q", 0)] == "queued"
    assert st[("rung1_r", 0)] == "running" and st[("rung1_f", 0)] == "failed"
    base = [r for r in rows if r["arm"] == "rung1_base"][0]
    assert base["composite"] == pytest.approx(0.5 * 0.3 + 0.25 * 0.4 + 0.25 * 0.5)
    assert base["copy_leak_pass"] is True


def _ladder(tmp_path, comps):
    p = tmp_path / "ladder.json"
    p.write_text(json.dumps({"comparisons": comps}))
    return p


def test_check_rung(tmp_path):
    runs = tmp_path / "runs"
    lad = _ladder(tmp_path, [
        {"name": "r1_dna", "incumbent": "rung1_base", "challenger": "rung1_dna_off",
         "simpler": "challenger", "d14_condition_1": True, "verdict": "TIE_ASK_PI"},
        {"name": "r1_tok", "incumbent": "rung1_base", "challenger": "rung1_tokens",
         "simpler": "incumbent", "d14_condition_1": True, "verdict": ""}])
    runs.mkdir()
    missing = R.check_rung(runs, 1, lad)
    assert [m.split(" ")[0] for m in missing] == ["r1_dna", "r1_tok"]   # no settlements row yet
    with open(runs / "settlements.tsv", "w") as fh:
        fh.write("\t".join(R.SETTLEMENT_COLUMNS) + "\n")
        fh.write("\t".join(["r1_tok", "a", "b", "c", "", "", "", "", "", "INCUMBENT", "t1"]) + "\n")
        fh.write("\t".join(["r1_dna", "a", "b", "c", "", "", "", "", "", "TIE_ASK_PI", "t1"]) + "\n")
    assert R.check_rung(runs, 1, lad) == []


def _full_ladder(tmp_path, runs, verdict="INCUMBENT", final_goals=None, base_goals=None):
    """Six rungs, one listed comparison each; a D13 verdict also gets its settlements.tsv row, a PI
    ruling its notepad reference."""
    comps = []
    for r in range(1, 7):
        make_arm(runs, f"rung{r}_base", BASE)
        make_arm(runs, f"rung{r}_dna_off", BASE)
        comps.append({"name": f"r{r}_dna", "incumbent": f"rung{r}_base",
                      "challenger": f"rung{r}_dna_off", "simpler": "challenger",
                      "d14_condition_1": True, "verdict": verdict,
                      "notepad_ref": "notepad#pi-ruling" if verdict.startswith("PI_") else ""})
        if verdict in R.VERDICTS or verdict.startswith("PI_"):
            R.append_settlement(runs / "settlements.tsv", {
                "comparison": f"r{r}_dna", "verdict": verdict if verdict in R.VERDICTS else "TIE_ASK_PI",
                "time": "2026-11-11T00:00:00+00:00"})
    comps.append({"name": "extra", "incumbent": "rung5_base", "challenger": "rung5_axis",
                  "simpler": "none", "d14_condition_1": False, "verdict": "TIE_ASK_PI",
                  "notepad_ref": ""})
    return _ladder(tmp_path, comps)


def _baselines(root: Path, goals: dict, sub=""):
    for name, g in goals.items():
        write_seed(root / name / "seed0", *g, leak=None, sub=sub)


GOOD_BASELINES = {"marginal_nb": (0.05, 0.10, 0.0), "position_average": (0.20, 0.15, 0.30),
                  "depth_only_scaling": (None, 0.0, None)}


def test_check_done_both_conditions(tmp_path, capsys):
    runs = tmp_path / "runs"
    lad = _full_ladder(tmp_path, runs)
    make_arm(runs, "final", BASE)
    bl = tmp_path / "baselines"
    _baselines(bl, GOOD_BASELINES)
    res = R.check_done(runs, lad, baselines=bl)
    assert res["condition_1"] == "PASS"
    assert res["validation"][0] == "PASS" and res["condition_2"] == "PENDING"   # blind not scored
    for k in range(3):
        write_seed(runs / "final" / f"seed{k}", *BASE[k], sub="blind")
    _baselines(bl, GOOD_BASELINES, sub="blind")
    res = R.check_done(runs, lad, baselines=bl)
    assert res["condition_2"] == "PENDING"      # blind baselines default beside the blind evalset
    res = R.check_done(runs, lad, baselines=bl, blind_baselines=bl)
    assert res["condition_2"] == "PASS"
    rc = R.main(["check-done", "--runs", str(runs), "--ladder", str(lad), "--baselines", str(bl),
                 "--blind-baselines", str(bl)])
    out = capsys.readouterr().out
    assert rc == 0 and "D14 condition 1: PASS" in out and "D14 condition 2: PASS" in out
    assert "margin" in out


def test_check_done_condition_2_fails_on_one_goal(tmp_path):
    runs = tmp_path / "runs"
    lad = _full_ladder(tmp_path, runs)
    make_arm(runs, "final", BASE)
    bl = tmp_path / "baselines"
    # position average's uncertainty 0.49: final 0.503 - 0.49 = 0.013 < 2 x spread 0.01 x 2 = 0.02
    _baselines(bl, dict(GOOD_BASELINES, position_average=(0.20, 0.15, 0.49)))
    res = R.check_done(runs, lad, baselines=bl)
    assert res["validation"][0] == "FAIL" and res["condition_2"] == "FAIL"
    assert any("uncertainty" in ln and "does not beat" in ln for ln in res["validation"][1])


def test_check_done_condition_1_fails(tmp_path):
    runs = tmp_path / "runs"
    lad = _full_ladder(tmp_path, runs, verdict="TIE_ASK_PI")           # escalated, not settled
    res = R.check_done(runs, lad, baselines=tmp_path / "none")
    assert res["condition_1"] == "FAIL"
    assert any("not a settlement" in p for p in res["problems_1"])
    lad2 = _full_ladder(tmp_path, runs, verdict="PI_INCUMBENT")
    assert R.check_done(runs, lad2, baselines=tmp_path / "none")["condition_1"] == "PASS"
    # a listed arm with 2 seeds fails condition 1
    import shutil
    shutil.rmtree(runs / "rung3_dna_off" / "seed2")
    res = R.check_done(runs, lad2, baselines=tmp_path / "none")
    assert res["condition_1"] == "FAIL" and any("rung3_dna_off" in p for p in res["problems_1"])
    # a missing rung fails condition 1
    lad3 = _ladder(tmp_path, [c for c in json.loads(lad2.read_text())["comparisons"]
                              if c["incumbent"] != "rung6_base"])
    assert any("rung 6" in p for p in R.check_done(runs, lad3, baselines=tmp_path / "x")["problems_1"])


def test_ladder_add_replaces_by_name(tmp_path):
    lad = tmp_path / "ladder.json"
    R.ladder_add(lad, "r1_dna", "rung1_base", "rung1_dna_off", "challenger", True, "TIE_ASK_PI")
    R.ladder_add(lad, "r1_dna", "rung1_base", "rung1_dna_off", "challenger", True, "PI_INCUMBENT",
                 "notepad#r1-dna")
    comps = json.loads(lad.read_text())["comparisons"]
    assert len(comps) == 1 and comps[0]["verdict"] == "PI_INCUMBENT"
    assert tuple(comps[0]) == R.LADDER_KEYS and comps[0]["notepad_ref"] == "notepad#r1-dna"


# ------------------------------------------------------------------ review E6 regression tests

def _rows(path):
    return list(csv.DictReader(open(path), delimiter="\t")) if Path(path).is_file() else []


def test_f1_scores_from_different_evalsets_or_splits_refused(tmp_path, capsys):
    """Finding 1 repro: incumbent on evalset A / val, challenger on evalset B / blind /
    accessibility_only used to give CHALLENGER."""
    inc = make_arm(tmp_path, "rung1_base", BASE, evalset_sha256="A" * 64)
    ch = make_arm(tmp_path, "rung1_x", shift(BASE, di=0.10), evalset_sha256="B" * 64,
                  split="blind", regime="accessibility_only")
    s = tmp_path / "s.tsv"
    with pytest.raises(R.Refused, match="not comparable"):
        R.compare(inc, ch, "none", settlements=s)
    assert not s.exists()
    rc = R.main(["compare", "--incumbent", str(inc), "--challenger", str(ch), "--simpler", "none",
                 "--settlements", str(s)])
    assert rc == 2 and "REFUSED" in capsys.readouterr().out and not s.exists()
    # each field on its own
    for k, v in (("evalset_sha256", "C" * 64), ("tables_sha256", "U" * 64), ("split", "blind"),
                 ("regime", "accessibility_only"), ("space", "signal_arcsinh"), ("evalset_sha256", "")):
        ch2 = make_arm(tmp_path, f"rung1_{k}{len(v)}", shift(BASE, di=0.10), **{k: v})
        with pytest.raises(R.Refused):
            R.decide(R.load_arm(inc), R.load_arm(ch2), "none")


def test_f1_done_condition_2_checks_comparability(tmp_path):
    runs = tmp_path / "runs"
    make_arm(runs, "final", BASE)
    bl = tmp_path / "baselines"
    _baselines(bl, GOOD_BASELINES)
    write_seed(bl / "position_average" / "seed0", 0.2, 0.15, 0.3, leak=None, evalset_sha256="Z" * 64)
    st, lines = R.done_condition_2(runs / "final", bl)
    assert st == "REFUSED" and "evalset_sha256" in lines[0]
    # a blind pass that says split val is refused
    bl2 = tmp_path / "bl2"
    _baselines(bl2, GOOD_BASELINES)
    for k in range(3):
        write_seed(runs / "final" / f"seed{k}", *BASE[k], sub="blind", split="val")
    _baselines(bl2, GOOD_BASELINES, sub="blind")
    st, lines = R.done_condition_2(runs / "final", bl2, "blind")
    assert st == "REFUSED" and "split" in lines[0]


def test_f11_pilot_scores_without_noise_floor_refused(tmp_path):
    inc = make_arm(tmp_path, "rung1_base", BASE, noise_floor="none_pilot")
    ch = make_arm(tmp_path, "rung1_x", shift(BASE, di=0.10), noise_floor="none_pilot")
    with pytest.raises(R.Refused, match="no-halves"):
        R.compare(inc, ch, "none", settlements=tmp_path / "s.tsv")


def test_f4_goal_term_out_of_band_escalates(tmp_path):
    """Finding 4 repro: incumbent (0.5, 0.5, 0.5) vs challenger (0.9, 0.9, -20) with --simpler
    challenger gave CHALLENGER_AS_SIMPLER; (0.5, 0.5, 5.0) won CHALLENGER."""
    jit = [(0.0, 0.0, 0.0), (0.01, 0.01, 0.01), (0.0, 0.01, 0.0)]
    inc = make_arm(tmp_path, "rung1_base", [(0.5 + a, 0.5 + b, 0.5 + c) for a, b, c in jit])
    ch = make_arm(tmp_path, "rung1_x", [(0.9 + a, 0.9 + b, -20 + c) for a, b, c in jit])
    res = R.compare(inc, ch, "challenger", settlements=tmp_path / "s.tsv", clip=False)
    assert res["verdict"] == "RULE_NOT_APPLICABLE_ASK_PI"
    assert "uncertainty term = -20" in res["reason"] and "sanity band" in res["reason"]
    ch2 = make_arm(tmp_path, "rung1_y", [(0.5 + a, 0.5 + b, 5.0 + c) for a, b, c in jit])
    res2 = R.compare(inc, ch2, "challenger", settlements=tmp_path / "s.tsv", clip=False)
    assert res2["verdict"] == "RULE_NOT_APPLICABLE_ASK_PI" and "uncertainty term = 5" in res2["reason"]
    assert R.CLIP_GOAL_TERMS is False and R.GOAL_BAND == (-1.0, 1.5)


def test_f4_clip_mode(tmp_path, monkeypatch):
    jit = [(0.0, 0.0, 0.0), (0.01, 0.01, 0.01), (0.0, 0.01, 0.0)]
    inc = make_arm(tmp_path, "rung1_base", [(0.5 + a, 0.5 + b, 0.5 + c) for a, b, c in jit])
    ch = make_arm(tmp_path, "rung1_x", [(0.9 + a, 0.9 + b, 5.0 + c) for a, b, c in jit])
    res = R.decide(R.load_arm(inc), R.load_arm(ch), "none", clip=True)
    # clipped to (0.9, 0.9, 1.0): a clear win on every goal
    assert res["verdict"] == "CHALLENGER"
    assert res["per_goal_deltas"]["uncertainty"] == pytest.approx(1.0 - _mean_u(jit))
    monkeypatch.setattr(R, "CLIP_GOAL_TERMS", True)           # the one-line switch
    assert R.decide(R.load_arm(inc), R.load_arm(ch), "none")["verdict"] == "CHALLENGER"


def _mean_u(jit):
    return sum(0.5 + c for _, _, c in jit) / len(jit)


@pytest.mark.parametrize("clip", [False, True])
@pytest.mark.parametrize("den,key", [
    ({"uncertainty_marginal_pit_deviation": 1e-9,
      "correction_baseline_minus_floor_relative_min": 0.2}, "uncertainty denominator"),
    ({"uncertainty_marginal_pit_deviation": 0.1,
      "correction_baseline_minus_floor_relative_min": 1e-4}, "correction denominator"),
    ({}, "not recorded")])
def test_f4_tiny_denominator_escalates_in_both_modes(tmp_path, clip, den, key):
    inc = make_arm(tmp_path, "rung1_base", BASE)
    ch = make_arm(tmp_path, "rung1_x", shift(BASE, di=0.1), denominators=den)
    res = R.decide(R.load_arm(inc), R.load_arm(ch), "none", clip=clip)
    assert res["verdict"] == "RULE_NOT_APPLICABLE_ASK_PI" and key in res["reason"]


def test_f5_zero_seed_spread_refused(tmp_path, capsys):
    """Finding 5 repro: 3 identical seeds per arm, challenger imputation +1e-9, gave CHALLENGER."""
    inc = make_arm(tmp_path, "rung1_base", [(0.3, 0.4, 0.5)] * 3)
    ch = make_arm(tmp_path, "rung1_x", [(0.3 + 1e-9, 0.4, 0.5)] * 3)
    s = tmp_path / "s.tsv"
    rc = R.main(["compare", "--incumbent", str(inc), "--challenger", str(ch), "--simpler", "none",
                 "--settlements", str(s)])
    out = capsys.readouterr().out
    assert rc == 2 and "REFUSED" in out and "do not vary" in out and not s.exists()
    # one goal with zero spread in both arms is refused too (its veto bar would be 0)
    inc2 = make_arm(tmp_path, "rung1_b2", [(0.30, 0.40, 0.5), (0.31, 0.41, 0.5), (0.30, 0.40, 0.5)])
    ch2 = make_arm(tmp_path, "rung1_y", [(0.40, 0.40, 0.5), (0.41, 0.41, 0.5), (0.40, 0.40, 0.5)])
    with pytest.raises(R.Refused, match="uncertainty"):
        R.decide(R.load_arm(inc2), R.load_arm(ch2), "none")


def test_f5_done_condition_2_zero_spread_refused(tmp_path):
    """Finding 5 repro: a final design at baseline + 1e-12 with a spread of 0 gave PASS."""
    runs = tmp_path / "runs"
    best = (0.20, 0.15, 0.30)
    make_arm(runs, "final", [tuple(v + 1e-12 for v in best)] * 3)
    bl = tmp_path / "baselines"
    _baselines(bl, GOOD_BASELINES)
    st, lines = R.done_condition_2(runs / "final", bl)
    assert st == "REFUSED" and any("do not vary" in ln for ln in lines)


def test_f7_missing_or_undefined_is_not_fail(tmp_path):
    """Finding 7 repro: check_done(..., baselines=<missing dir>) returned FAIL."""
    runs = tmp_path / "runs"
    lad = _full_ladder(tmp_path, runs)
    make_arm(runs, "final", BASE)
    res = R.check_done(runs, lad, baselines=tmp_path / "missing")
    assert res["validation"][0] == "MISSING" and res["condition_2"] == "MISSING"
    # an undefined lead metric of the final design escalates, it does not fail
    make_arm(runs, "final_u", [(0.3, None, 0.5), (0.31, 0.4, 0.5), (0.3, 0.41, 0.51)])
    bl = tmp_path / "baselines"
    _baselines(bl, GOOD_BASELINES)
    res = R.check_done(runs, lad, final=runs / "final_u", baselines=bl)
    assert res["validation"][0] == "ESCALATE" and res["condition_2"] == "ESCALATE"
    # every baseline undefined on a goal escalates too
    bl2 = tmp_path / "bl2"
    _baselines(bl2, {"marginal_nb": (0.05, None, 0.0), "position_average": (0.2, None, 0.3),
                     "depth_only_scaling": (None, None, None)})
    st, lines = R.done_condition_2(runs / "final", bl2)
    assert st == "ESCALATE" and any("correction" in ln and "ESCALATE" in ln for ln in lines)
    # FAIL only on a defined margin
    _baselines(bl2, dict(GOOD_BASELINES, position_average=(0.20, 0.15, 0.49)))
    assert R.done_condition_2(runs / "final", bl2)[0] == "FAIL"


def test_f8_hand_written_verdicts_need_settlements_or_pi_reference(tmp_path):
    """Finding 8 repro: six ladder_add(..., verdict="CHALLENGER") with no settlements.tsv gave
    (True, []); the verdict "PI_" passed too."""
    runs = tmp_path / "runs"
    lad = tmp_path / "ladder.json"
    for r in range(1, 7):
        make_arm(runs, f"rung{r}_base", BASE)
        make_arm(runs, f"rung{r}_dna_off", BASE)
        R.ladder_add(lad, f"r{r}", f"rung{r}_base", f"rung{r}_dna_off", "challenger", True,
                     "CHALLENGER")
    ok, probs = R.done_condition_1(runs, lad)
    assert not ok and len(probs) == 6 and all("does not match" in p for p in probs)
    # a matching settlements row makes them settlements
    for r in range(1, 7):
        R.append_settlement(runs / "settlements.tsv", {"comparison": f"r{r}", "verdict": "CHALLENGER",
                                                       "time": "2026-11-11T00:00:00+00:00"})
    assert R.done_condition_1(runs, lad) == (True, [])
    # "PI_" and a PI verdict without its notepad reference are not settlements
    with pytest.raises(R.Refused):
        R.ladder_add(lad, "r1", "rung1_base", "rung1_dna_off", "challenger", True, "PI_")
    with pytest.raises(R.Refused, match="notepad"):
        R.ladder_add(lad, "r1", "rung1_base", "rung1_dna_off", "challenger", True, "PI_INCUMBENT")
    with pytest.raises(R.Refused):
        R.ladder_add(lad, "r1", "rung1_base", "rung1_dna_off", "challenger", True, "WIN")
    raw = json.loads(lad.read_text())
    raw["comparisons"][0]["verdict"] = "PI_"
    raw["comparisons"][1]["verdict"] = "PI_INCUMBENT"          # no notepad_ref
    lad.write_text(json.dumps(raw))
    ok, probs = R.done_condition_1(runs, lad)
    assert not ok and any(p.startswith("r1:") for p in probs) and any(p.startswith("r2:") for p in probs)
    assert R.main(["ladder-add", "--ladder", str(lad), "--name", "r1", "--incumbent", "rung1_base",
                   "--challenger", "rung1_dna_off", "--simpler", "challenger", "--d14", "true",
                   "--verdict", "PI_INCUMBENT", "--notepad-ref", "notepad#r1"]) == 0
    assert R.ladder_verdict(json.loads(lad.read_text())["comparisons"][-1], {}) == ("PI_INCUMBENT", "")


def test_f12_simpler_than_base_must_be_bool_or_null(tmp_path):
    """Finding 12 repro: a string "true" raised KeyError 'true'."""
    ch = make_arm(tmp_path, "rung1_x", BASE, simpler="true")
    with pytest.raises(R.Refused, match="must be true, false or null"):
        R.simpler_from_config(ch)


def test_f17_latest_settlement_is_newest_in_time(tmp_path):
    """Finding 17: the newest row wins, not the last file in sorted-path order."""
    runs = tmp_path / "runs"
    R.append_settlement(runs / "a" / "settlements.tsv",
                        {"comparison": "c", "verdict": "INCUMBENT", "time": "2026-11-02T00:00:00+00:00"})
    R.append_settlement(runs / "b" / "settlements.tsv",
                        {"comparison": "c", "verdict": "CHALLENGER", "time": "2026-11-01T00:00:00+00:00"})
    assert R.latest_settlements(runs) == {"c": "INCUMBENT"}
    inc = make_arm(runs, "rung1_base", BASE)
    ch = make_arm(runs, "rung1_x", shift(BASE, di=0.1))
    R.compare(inc, ch, "none", name="c", settlements=runs / "a" / "settlements.tsv")
    rows = _rows(runs / "a" / "settlements.tsv")
    assert rows[-1]["time"] > "2026-11-02" and R.latest_settlements(runs)["c"] == "CHALLENGER"


def test_ladder_simpler_explicit_off_the_rung_base(tmp_path):
    """Interface gap: simpler_than_base is relative to the challenger's own rung base."""
    a = make_arm(tmp_path, "rung2_base", BASE, simpler=True)
    b = make_arm(tmp_path, "rung3_base", shift(BASE, di=0.002), simpler=True)
    c = make_arm(tmp_path, "rung3_dna_off", shift(BASE, di=0.002), simpler=True)
    with pytest.raises(R.Refused, match="pass --simpler"):
        R.compare(a, b, None, settlements=tmp_path / "s.tsv")          # cross-rung
    with pytest.raises(R.Refused, match="pass --simpler"):
        R.compare(b.parent / "rung3_dna_off", b, None, settlements=tmp_path / "s.tsv")  # arm v arm
    assert R.compare(a, b, "incumbent", settlements=tmp_path / "s.tsv")["verdict"] == "INCUMBENT"
    assert R.compare(b, c, None, settlements=tmp_path / "s.tsv")["verdict"] == "CHALLENGER_AS_SIMPLER"
    with pytest.raises(R.Refused, match="rung<k>_<arm>"):
        R.ladder_add(tmp_path / "l.json", "x", "base", "rung1_x", "none", True)
    assert R.ladder_add(tmp_path / "l.json", "x", "final", "stage2_x", "none", False)["verdict"] == ""


def test_f2_copy_leak_spread_not_ok_escalates(tmp_path):
    inc = make_arm(tmp_path, "rung1_base", BASE)
    ch = make_arm(tmp_path, "rung1_x", shift(BASE, di=0.1), spread_ok=False)
    res = R.decide(R.load_arm(inc), R.load_arm(ch), "none")
    assert res["verdict"] == "RULE_NOT_APPLICABLE_ASK_PI" and "not measured reliably" in res["reason"]
    assert res["copy_leak"] == "incumbent=pass;challenger=unreliable"


def test_table_needs_scored_marker(tmp_path):
    """Finding 3: training's DONE alone is not a scored seed."""
    d = tmp_path / "rung1_t" / "seed0"
    d.mkdir(parents=True)
    (d / "DONE").write_text("x")
    assert R.seed_state(d) == "running"
    (d / "scores.json").write_text("{}")
    assert R.seed_state(d) == "running"
    (d / "SCORED").write_text("x")
    assert R.seed_state(d) == "done"
