"""PI ruling 24: the blind tracks are still scored once after a validation failure, so `check-done`
reports the blind state and its margins whatever the validation state."""
from __future__ import annotations

from tests import test_onion_eval_select_rule as T

R = T.R
# position average's uncertainty 0.49: final 0.503 - 0.49 = 0.013 < 2 x seed spread 0.02 -> FAIL
FAILING_BASELINES = dict(T.GOOD_BASELINES, position_average=(0.20, 0.15, 0.49))


def _setup(tmp_path, val_baselines, blind_baselines=None):
    runs = tmp_path / "runs"
    lad = T._full_ladder(tmp_path, runs)
    T.make_arm(runs, "final", T.BASE)
    bl, bb = tmp_path / "baselines", tmp_path / "blind_baselines"
    T._baselines(bl, val_baselines)
    if blind_baselines is not None:
        for k in range(3):
            T.write_seed(runs / "final" / f"seed{k}", *T.BASE[k], sub="blind")
        T._baselines(bb, blind_baselines, sub="blind")
    return runs, lad, bl, bb


def test_r24_blind_state_after_validation_fail_pass(tmp_path, capsys):
    runs, lad, bl, bb = _setup(tmp_path, FAILING_BASELINES, T.GOOD_BASELINES)
    res = R.check_done(runs, lad, baselines=bl, blind_baselines=bb)
    assert res["validation"][0] == "FAIL"
    assert res["blind"][0] == "PASS"                # was MISSING ("waits for validation") before
    assert res["condition_2"] == "FAIL"             # PASS only when both splits pass
    for g in R.GOALS:
        assert any(ln.strip().startswith(f"{g}:") and "margin" in ln and "seed spread" in ln
                   for ln in res["blind"][1])
    rc = R.main(["check-done", "--runs", str(runs), "--ladder", str(lad), "--baselines", str(bl),
                 "--blind-baselines", str(bb)])
    out = capsys.readouterr().out.splitlines()
    assert rc == 1
    i_val = next(i for i, ln in enumerate(out) if ln.strip() == "validation: FAIL")
    i_bl = next(i for i, ln in enumerate(out) if ln.strip() == "blind: PASS")
    i_d14 = max(i for i, ln in enumerate(out) if ln.startswith("D14 condition"))
    assert i_d14 < i_val < i_bl
    assert sum("margin" in ln for ln in out[i_bl:]) == 3


def test_r24_blind_state_after_validation_fail_fail(tmp_path):
    runs, lad, bl, bb = _setup(tmp_path, FAILING_BASELINES, FAILING_BASELINES)
    res = R.check_done(runs, lad, baselines=bl, blind_baselines=bb)
    assert (res["validation"][0], res["blind"][0], res["condition_2"]) == ("FAIL", "FAIL", "FAIL")
    assert any("uncertainty" in ln and "does not beat" in ln for ln in res["blind"][1])


def test_r24_validation_fail_without_blind_pass(tmp_path, capsys):
    runs, lad, bl, _ = _setup(tmp_path, FAILING_BASELINES)
    res = R.check_done(runs, lad, baselines=bl)
    assert (res["validation"][0], res["blind"][0], res["condition_2"]) == ("FAIL", "MISSING", "FAIL")
    R.main(["check-done", "--runs", str(runs), "--ladder", str(lad), "--baselines", str(bl)])
    out = [ln.strip() for ln in capsys.readouterr().out.splitlines()]
    assert "D14 condition 2: FAIL" in out and "blind: MISSING" in out


def test_r24_validation_pass_states_unchanged(tmp_path):
    runs, lad, bl, bb = _setup(tmp_path, T.GOOD_BASELINES)
    res = R.check_done(runs, lad, baselines=bl)
    assert (res["validation"][0], res["blind"][0], res["condition_2"]) == ("PASS", "MISSING", "PENDING")
    runs, lad, bl, bb = _setup(tmp_path / "b", T.GOOD_BASELINES, FAILING_BASELINES)
    res = R.check_done(runs, lad, baselines=bl, blind_baselines=bb)
    assert (res["validation"][0], res["blind"][0], res["condition_2"]) == ("PASS", "FAIL", "FAIL")
    runs, lad, bl, bb = _setup(tmp_path / "c", T.GOOD_BASELINES, T.GOOD_BASELINES)
    res = R.check_done(runs, lad, baselines=bl, blind_baselines=bb)
    assert (res["validation"][0], res["blind"][0], res["condition_2"]) == ("PASS", "PASS", "PASS")
