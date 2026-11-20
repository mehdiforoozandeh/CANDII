"""Unit tests of tools/onion_eval/apply_rescoring_rules.py on synthetic arm-summary tables."""
import base64
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "onion_eval"))
import apply_rescoring_rules as A  # noqa: E402

KEYS = A.KEYS


def row(arm, n, mean=0.5, rng=0.02, **over):
    r = {"arm": arm, "n_seeds": str(n)}
    for k in KEYS:
        m, g = over.get(k, (mean, rng))
        r[f"{k}.mean"], r[f"{k}.range"], r[f"{k}.n_defined"] = str(m), str(g), str(n)
    return r


def refs():
    return {a: row(a, 3) for a in A.REF.values()}


def full(over=None):
    """Every arm complete and equal to the reference, with per-arm overrides."""
    t = refs()
    for g in A.GROUPS:
        for m in A.MODELS:
            arm = A.arm_of(g, m)
            t[arm] = row(arm, A.planned(g, m), **(over or {}).get(arm, {}))
    return t


def q_all(status="qualifies"):
    return [dict(rung=r, switch=s, status=status, run="", detail="")
            for r, s in (("rung3", "fixed_depth"), ("rung4", "average_input"), ("rung3", "impute_x4"))]


def test_classes():
    assert A.classify(0.05, 0.02) == "2x"
    assert A.classify(-0.03, 0.02) == "beyond"
    assert A.classify(0.02, 0.02) == "within"      # equal to the bar is within (strict >)
    assert A.classify(-0.04, 0.02) == "beyond"     # equal to twice the bar is beyond, not 2x


def test_bar_is_larger_range_of_arm_and_reference():
    t = refs()
    t["abl_scrambled_covariates_scrambled_covariates_convae"] = row(
        "abl_scrambled_covariates_scrambled_covariates_convae", 3, correction_only=(0.44, 0.05))
    c = A.cell(t, "scrambled", "convae", "correction_only")
    assert c["complete"] and abs(c["d"] + 0.06) < 1e-12 and c["bar"] == 0.05 and c["cls"] == "beyond"


def test_incomplete_arm_gets_no_verdict_and_names_the_seeds():
    t = full()
    t["abl_views_base_only_convae"] = row("abl_views_base_only_convae", 2)
    del t["rung4_views_base_only"]
    R, _ = A.apply_rules(t, q_all())
    r = R["2.2"]
    assert not r["decided"] and r["code"] == "not_decided"
    assert "conv autoencoder (2 of 3 seeds)" in r["verdict"] and "U-Net (0 of 3 seeds)" in r["verdict"]
    assert all("status" not in v for v in r["rows"].values())


def test_two_seed_arm_is_complete_by_plan():
    t = full()
    c = A.cell(t, "fixed_depth", "convae", "imputation_raw")
    assert c["planned"] == 2 and c["complete"]


def test_rule_21_strong_moderate_weak():
    def arms(ds):
        return {f"abl_scrambled_covariates_scrambled_covariates_{m}": {"correction_only": (0.5 + d, 0.02)}
                for m, d in zip(A.MODELS, ds)}
    R, _ = A.apply_rules(full(arms([-0.05, -0.05, 0.0])), q_all())
    assert R["2.1"]["code"] == "uses_covariates_strong" and "settled, strong" in R["2.1"]["rows"][8]["status"]
    R, _ = A.apply_rules(full(arms([-0.05, -0.03, 0.0])), q_all())
    assert R["2.1"]["code"] == "uses_covariates_moderate"
    R, _ = A.apply_rules(full(arms([-0.05, 0.0, 0.0])), q_all())
    assert R["2.1"]["code"] == "weak" and "run 2.8 gets priority" in R["2.1"]["rows"][8]["status"]


def test_rule_22_three_cases():
    def arms(ds, comp=0.0):
        return {A.arm_of("views_base", m): {"correction_only": (0.5 + d, 0.02), "composite_v3": (0.5 + comp, 0.02)}
                for m, d in zip(A.MODELS, ds)}
    assert A.apply_rules(full(arms([-0.1, -0.1, -0.1])), q_all())[0]["2.2"]["code"] == "all_views_strong"
    assert A.apply_rules(full(arms([0.0, 0.0, -0.1])), q_all())[0]["2.2"]["code"] == "no_benefit_pi_decides"
    assert A.apply_rules(full(arms([0.0, 0.0, 0.0], -0.03)), q_all())[0]["2.2"]["code"] == "all_views_provisional"


def test_rule_23_d13_gate():
    up = {"abl_two_step_two_step_native_unet": {"composite_v3": (0.55, 0.02)}}
    assert A.apply_rules(full(up), q_all())[0]["2.3"]["code"] == "enters_pool_pi_decides"
    up["abl_two_step_two_step_native_unet"]["uncertainty_raw"] = (0.4, 0.02)   # a goal down 2x blocks it
    assert A.apply_rules(full(up), q_all())[0]["2.3"]["code"] == "off_settled"


def test_qualify_rules_need_the_qualify_line():
    R, _ = A.apply_rules(full(), [], qualify_ran=False)
    for rid in ("2.4", "2.5", "2.6"):
        assert not R[rid]["decided"] and "qualify did not run" in R[rid]["verdict"]
    R, _ = A.apply_rules(full(), q_all("fails"))
    assert R["2.5"]["code"] == "unet_pool_removed" and R["2.6"]["code"] == "convattn_pool_removed"
    assert R["2.4"]["code"] == "unet_gain_not_held+convattn_removed"


def test_design_update_is_idempotent_and_restores_originals():
    text = ("---\nstatus: old\n---\n| 8 | cov | v | ev8 | weak | provisional (rule 2.1) |\n"
            "| 23 | views | all | ev23 | weak | provisional (rule 2.2) |\n\n## 2. Provisional rows\n")
    t = full({f"abl_scrambled_covariates_scrambled_covariates_{m}": {"correction_only": (0.4, 0.02)}
              for m in A.MODELS})
    del t["rung4_views_base_only"]
    R, _ = A.apply_rules(t, q_all())
    once = A.update_design(text, R, "2026-11-18 09:45 UTC", "20261118T1300Z")
    twice = A.update_design(once, R, "2026-11-18 09:45 UTC", "20261118T1300Z")
    assert once == twice
    r8 = [ln for ln in once.splitlines() if ln.startswith("| 8 |")][0]
    assert "settled, strong" in r8 and "<!--p256:" in r8
    r23 = [ln for ln in once.splitlines() if ln.startswith("| 23 |")][0]
    assert r23.endswith("| weak | provisional (rule 2.2) |") and "not decided" in r23
    orig = json.loads(base64.b64decode(re.search(r"<!--p256:([^-]+)-->", r8).group(1)))
    assert orig == {"confidence": "weak", "status": "provisional (rule 2.1)"}
    assert once.count("## 256-region rescoring: what it settled") == 1


def test_arm_list_matches_the_submit_script():
    sh = (ROOT / "slurm/onion_eval/rescoring_p256_nibi.sh").read_text()
    want = set(re.findall(r"\d:([a-z0-9_]+)", "\n".join(ln for ln in sh.splitlines() if ln.strip().startswith(
        tuple(g + ")" for g in A.GROUPS)))))
    have = {A.arm_of(g, m) for g in A.GROUPS for m in A.MODELS}
    assert have == want and len(have) == 21
    assert sum(A.planned(g, m) for g in A.GROUPS for m in A.MODELS) == 62


def test_decided_with_one_arm_incomplete():
    t = full({A.arm_of("views_base", m): {"correction_only": (0.4, 0.02)} for m in ("convae", "convattn")})
    del t["rung4_views_base_only"]
    r = A.apply_rules(t, q_all())[0]["2.2"]
    assert r["decided"] and r["code"] == "all_views_strong"
    assert "U-Net (0 of 3 seeds)" in r["verdict"] and "cannot change this verdict" in r["verdict"]
    assert r["rows"][23]["status"] == "settled (rule 2.2, 256 regions)"


def test_verdict_fixed_confidence_open():
    t = full({"abl_scrambled_covariates_scrambled_covariates_convae": {"correction_only": (0.4, 0.02)},
              "abl_scrambled_covariates_scrambled_covariates_convattn": {"correction_only": (0.47, 0.02)}})
    t["abl_scrambled_covariates_scrambled_covariates_unet"]["n_seeds"] = "2"
    r = A.apply_rules(t, q_all())[0]["2.1"]
    assert r["decided"] and r["branch"] == "uses"
    assert r["confidence"] == "at least moderate, final level waits for the U-Net"
    assert "settled, at least moderate" in r["rows"][8]["status"]


def test_open_arm_can_still_flip_the_branch():
    t = full({A.arm_of("views_base", "convae"): {"correction_only": (0.4, 0.02)}})   # conv + attention within
    del t["rung4_views_base_only"]
    r = A.apply_rules(t, q_all())[0]["2.2"]
    assert not r["decided"] and "U-Net (0 of 3 seeds)" in r["verdict"]
    assert all("status" not in v for v in r["rows"].values())
