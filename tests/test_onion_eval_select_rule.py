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


def write_seed(d: Path, imp, cor, unc, leak=True, sub="", simpler=None, done=True):
    d.mkdir(parents=True, exist_ok=True)
    tgt = d / sub if sub else d
    tgt.mkdir(parents=True, exist_ok=True)
    comp = None if None in (imp, cor, unc) else 0.5 * imp + 0.25 * cor + 0.25 * unc
    s = {"selection": {"space": "counts_log1p", "imputation_gap_closed": imp,
                       "correction_gap_closed": cor, "uncertainty_gap_closed": unc,
                       "composite": comp, "reason": "" if comp is not None else "undefined"},
         "copy_leak": {"copy_leak_pass": leak}}
    (tgt / "scores.json").write_text(json.dumps(s))
    if done:
        (d / "DONE").write_text("x")
    if simpler is not None or simpler is None:
        (d / "run.json").write_text(json.dumps({"config": {"simpler_than_base": simpler}}))


def make_arm(root: Path, name: str, goals, leak=True, simpler=None, sub=""):
    """goals: list over seeds of (imputation, correction, uncertainty)."""
    arm = root / name
    for k, g in enumerate(goals):
        write_seed(arm / f"seed{k}", *g, leak=leak if not isinstance(leak, list) else leak[k],
                   simpler=simpler, sub=sub)
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
    inc = make_arm(tmp_path, "rung1_base", [(0.30, 0.40, 0.5), (0.30, 0.50, 0.5), (0.30, 0.45, 0.5)])
    ch = make_arm(tmp_path, "rung1_x", [(0.50, 0.35, 0.5), (0.50, 0.45, 0.5), (0.50, 0.40, 0.5)])
    # correction drops 0.05 < 2 x spread 0.10; composite +0.0875 > 2 x composite spread 0.025 x 2
    assert run_compare(tmp_path, inc, ch, "incumbent")["verdict"] == "CHALLENGER"


def test_seed_spread_is_max_minus_min_larger_arm(tmp_path):
    inc = make_arm(tmp_path, "rung1_base", [(0.30, 0.4, 0.5), (0.30, 0.4, 0.5), (0.30, 0.4, 0.5)])
    ch = make_arm(tmp_path, "rung1_x", [(0.36, 0.4, 0.5), (0.40, 0.4, 0.5), (0.44, 0.4, 0.5)])
    res = R.decide(R.load_arm(inc), R.load_arm(ch), "incumbent")
    assert res["seed_spread"] == pytest.approx(0.5 * 0.08)          # challenger's range, not SD
    assert res["per_goal_spreads"]["imputation"] == pytest.approx(0.08)
    # delta = 0.05 composite, 2 x 0.04 = 0.08: within the bars, so the simpler incumbent stays
    assert res["verdict"] == "INCUMBENT" and "within" in res["reason"]
    ch2 = make_arm(tmp_path, "rung1_y", [(0.46, 0.4, 0.5), (0.50, 0.4, 0.5), (0.54, 0.4, 0.5)])
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
    assert R.check_rung(runs, 1, lad) == ["r1_tok"]
    with open(runs / "settlements.tsv", "w") as fh:
        fh.write("\t".join(R.SETTLEMENT_COLUMNS) + "\n")
        fh.write("\t".join(["r1_tok", "a", "b", "c", "", "", "", "", "", "INCUMBENT"]) + "\n")
    assert R.check_rung(runs, 1, lad) == []


def _full_ladder(tmp_path, runs, verdict="INCUMBENT", final_goals=None, base_goals=None):
    comps = []
    for r in range(1, 7):
        make_arm(runs, f"rung{r}_base", BASE)
        make_arm(runs, f"rung{r}_dna_off", BASE)
        comps.append({"name": f"r{r}_dna", "incumbent": f"rung{r}_base",
                      "challenger": f"rung{r}_dna_off", "simpler": "challenger",
                      "d14_condition_1": True, "verdict": verdict})
    comps.append({"name": "extra", "incumbent": "rung5_base", "challenger": "rung5_axis",
                  "simpler": "none", "d14_condition_1": False, "verdict": "TIE_ASK_PI"})
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
    assert res["condition_2"] == "PASS"
    rc = R.main(["check-done", "--runs", str(runs), "--ladder", str(lad), "--baselines", str(bl)])
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
    R.ladder_add(lad, "r1_dna", "rung1_base", "rung1_dna_off", "challenger", True, "PI_INCUMBENT")
    comps = json.loads(lad.read_text())["comparisons"]
    assert len(comps) == 1 and comps[0]["verdict"] == "PI_INCUMBENT"
