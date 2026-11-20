"""Apply the pre-written rules 2.1 to 2.6 of the settled miniCANDI design to a 256-region arm summary.

The rules are in cruxvault/INSTRUCTIONS/peeling_the_onion/closeout/settled miniCANDI design.md,
section 2.0. They were written before any 256-region score. This script applies them as written;
it changes no threshold. For an arm row A, its reference row R and a key k:

    d   = A[k.mean] - R[k.mean]
    bar = max(A[k.range], R[k.range])          (the larger of the two seed ranges)
    class: "2x" if |d| > 2 bar, "beyond" if |d| > bar, else "within"

An arm with fewer seeds than planned (3; 2 for the conv autoencoder fixed-input-depth arm, PI
accepted) is "incomplete (k of n seeds)" and gets no verdict; a rule with an incomplete arm is "not
decided". Rules 2.4 (conv + attention), 2.5 (U-Net) and 2.6 (conv + attention) also need their line of
`tools/minicandi/mix_arms.py qualify`, which this script runs on the same table.

THE ONE COMMAND (from the worktree root; the foreman reruns it with each new table):

    /Users/mforooz/miniforge3/envs/candii/bin/python tools/onion_eval/apply_rescoring_rules.py \\
        <arm-summary tsv> [--as-of 2026-11-18T18:00Z]

It (1) writes verdicts.tsv, arms.tsv, verdicts.json, qualify.txt and RESCORING_READ.md into
cruxvault/results/rescoring_p256/read_<UTC stamp>/ of this worktree and of the foreman worktree,
(2) copies the table to cruxvault/results/rescoring_p256/arm_summary_p256_sv.tsv, (3) updates the
design table and the section "256-region rescoring: what it settled" of the settled-design document,
(4) points run_status.json (rescoring.read_dir) at the new read folder and writes the counts there,
(5) rebuilds the deck and the report and runs both verify steps with --strict. The builders find the
evidence through run_status.json: rescoring.read_dir names the read folder; they read its
verdicts.json and the copied table, and record both in their source manifests.

    python tools/onion_eval/apply_rescoring_rules.py <tsv> --no-build   # steps 1 to 4 only
    python tools/onion_eval/apply_rescoring_rules.py <tsv> --dry-run    # print the verdicts only

Stdlib only.
"""
from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
VAULT = ROOT / "cruxvault"
FOREMAN_VAULT = pathlib.Path(
    "/Users/mforooz/Desktop/research/libbrechteam@sfu/CANDII/.claude/worktrees/onion/cruxvault")
DES_REL = "INSTRUCTIONS/peeling_the_onion/closeout/settled miniCANDI design.md"
P16_REL = "results/ablations_closeout/arm_summary_p16_sv_2026-11-17.tsv"
OUT_REL = "results/rescoring_p256"
TABLE_REL = OUT_REL + "/arm_summary_p256_sv.tsv"
RSTAT_REL = "presentations/onion-closeout-2026-11-18/run_status.json"
DECK_DIR = "presentations/onion-closeout-2026-11-18"
STORY_DIR = "presentations/onion-closeout-2026-11-18-report"
DECK_PY = "/Users/mforooz/miniforge3/bin/python"          # has matplotlib and Pillow
CRUX = "/Users/mforooz/.claude/skills/crux/scaffold/crux.py"

MODELS = ("convae", "convattn", "unet")
MODEL = {"convae": "conv autoencoder", "convattn": "conv + attention", "unet": "U-Net"}
REF = {"convae": "rung2_base_heldout", "convattn": "rung3_base_heldout", "unet": "rung4_base_heldout"}
KEYS = ("composite_v3", "imputation_raw", "correction_v3_raw", "uncertainty_raw", "correction_only")
METRIC = {"composite_v3": "composite", "imputation_raw": "imputation",
          "correction_v3_raw": "correction on crop pairs", "uncertainty_raw": "calibration",
          "correction_only": "correction-only term"}

# The 7 arm groups of slurm/onion_eval/rescoring_p256_nibi.sh: group -> (label, {model: (arm, seeds)})
# (tests/test_onion_eval_apply_rescoring_rules.py holds this list equal to the submit script).
GROUPS = {
    "scrambled": ("scrambled covariates", "abl_scrambled_covariates_scrambled_covariates_{m}"),
    "views_base": ("base view only", "abl_views_base_only_{m}"),
    "two_step_native": ("two-step routing, native base depth", "abl_two_step_two_step_native_{m}"),
    "two_step_fixed": ("two-step routing, fixed base depth", "abl_two_step_two_step_fixed_{m}"),
    "fixed_depth": ("fixed input depth", "abl_depth_input_input_depth_fixed_{m}"),
    "average_input": ("average baseline input", "abl_average_input_average_input_{m}"),
    "impute_x4": ("imputation weight 4", "abl_losses_loss_impute_x4_{m}"),
}
SPECIAL_ARM = {("views_base", "unet"): "rung4_views_base_only"}
PLANNED = {("fixed_depth", "convae"): 2}     # seed 1 did not train; the PI accepted 2 seeds


def arm_of(g, m):
    return SPECIAL_ARM.get((g, m), GROUPS[g][1].format(m=m))


def planned(g, m):
    return PLANNED.get((g, m), 3)


# rule -> (title, arm groups, design rows, metric drawn in the figure per group)
RULES = {
    "2.1": ("covariates used for correction", ("scrambled",), (8,), "correction_only"),
    "2.2": ("training views", ("views_base",), (23,), "correction_only"),
    "2.3": ("two-step routing", ("two_step_native", "two_step_fixed"), (24,), "composite_v3"),
    "2.4": ("fixed input depth", ("fixed_depth",), (25,), "correction_v3_raw"),
    "2.5": ("average baseline input", ("average_input",), (13,), "correction_only"),
    "2.6": ("imputation loss weight 4", ("impute_x4",), (20,), "correction_v3_raw"),
}


# ------------------------------------------------------------------------------------------- table
def read_table(path) -> dict:
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["arm"]: r for r in csv.DictReader(fh, delimiter="\t")}


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if v != v else v


def classify(d, bar):
    if abs(d) > 2 * bar:
        return "2x"
    if abs(d) > bar:
        return "beyond"
    return "within"


def cell(table, g, m, key, p16=None):
    """One arm-model-key cell. complete is False when the arm or the reference has fewer seeds than
    planned, or fewer seeds with a defined value of this key."""
    arm, n_plan = arm_of(g, m), planned(g, m)
    a, r = table.get(arm), table.get(REF[m])
    n = int(a["n_seeds"]) if a else 0
    c = dict(group=g, model=m, arm=arm, key=key, n_seeds=n, planned=n_plan, complete=False, seeds_ok=False,
             arm_mean=None, ref_mean=None, d=None, bar=None, cls=f"incomplete ({min(n, n_plan)} of {n_plan} seeds)",
             d16=None, sign16="")
    if p16 is not None:
        e = _effect(p16, arm, REF[m], key)
        c["d16"] = e[0] if e else None
        c["cls16"] = classify(e[0], e[1]) if e else None
    if a is None or r is None or n < n_plan or int(r["n_seeds"]) < 3:
        if r is None or (r and int(r["n_seeds"]) < 3):
            c["cls"] = "incomplete (reference has fewer than 3 seeds)"
        return c
    c["seeds_ok"] = True
    nd = int(a.get(key + ".n_defined") or 0)
    if nd < n_plan:
        c["cls"] = f"incomplete ({nd} of {n_plan} seeds have a value)"
        return c
    e = _effect(table, arm, REF[m], key)
    if e is None:
        c["cls"] = "incomplete (no value)"
        return c
    c.update(complete=True, d=e[0], bar=e[1], cls=classify(e[0], e[1]), arm_mean=e[2], ref_mean=e[3])
    if c["d16"] is not None and c["d"] != 0 and c["d16"] != 0:
        c["sign16"] = "yes" if (c["d"] > 0) == (c["d16"] > 0) else "no"
    return c


def _effect(table, arm, ref, key):
    a, r = table.get(arm), table.get(ref)
    if a is None or r is None:
        return None
    am, ar, rm, rr = (_f(a.get(key + ".mean")), _f(a.get(key + ".range")),
                      _f(r.get(key + ".mean")), _f(r.get(key + ".range")))
    if None in (am, ar, rm, rr):
        return None
    return am - rm, max(ar, rr), am, rm


def down(c, level="beyond"):
    return c["complete"] and c["d"] < 0 and (c["cls"] == "2x" or (level == "beyond" and c["cls"] == "beyond"))


def up(c, level="beyond"):
    return c["complete"] and c["d"] > 0 and (c["cls"] == "2x" or (level == "beyond" and c["cls"] == "beyond"))


# -------------------------------------------------------------------------------------------- text
def num(x):
    return f"{x:+.3f}"


def rng(x):
    return f"{x:.3f}"


CLS_WORD = {"within": "within the seed range", "beyond": "beyond the seed range",
            "2x": "beyond twice the seed range"}


def ctext(c, with_model=True):
    who = MODEL[c["model"]] + " " if with_model else ""
    if not c["complete"]:
        return f"{who}not finished ({c['cls'].replace('incomplete (', '')}"
    p = 4 if f"{abs(c['d']):.3f}" == f"{c['bar']:.3f}" or f"{abs(c['d']):.3f}" == f"{2 * c['bar']:.3f}" else 3
    return f"{who}{c['d']:+.{p}f} (seed range {c['bar']:.{p}f}, {CLS_WORD[c['cls']]})"


def names(ms):
    ms = ["the " + MODEL[m] for m in ms]
    if not ms:
        return "no model"
    return ms[0] if len(ms) == 1 else ", ".join(ms[:-1]) + " and " + ms[-1]


def of_three(n):
    return "all three models" if n == 3 else f"{count_word(n)} of three models"


def count_word(n):
    return {0: "no", 1: "one", 2: "two", 3: "three"}.get(n, str(n))


# ------------------------------------------------------------------------------------------- rules
# Each rule is evaluated as a function of a few variables (a class level of a cell, or a qualify
# status). A variable whose arm is complete is fixed; an open one takes every value of its domain.
# The rule is decided when every assignment gives the same branch; the confidence clause may stay
# open ("at least <level>, final level waits for <model>"). Otherwise it is not decided.
import itertools

LEVELS = ("d2", "d1", "w", "u1", "u2")          # down 2x, down beyond, within, up beyond, up 2x
QSTAT = ("qualifies", "fails")
CONF_ORDER = ("moderate", "strong")


def lvl(c):
    if not c["complete"]:
        return None
    if c["cls"] == "within":
        return "w"
    return ("d" if c["d"] < 0 else "u") + ("2" if c["cls"] == "2x" else "1")


def _is_down(v):
    return v in ("d1", "d2")


def _vars(rid, C, Q, qr):
    """{name: (value or None, domain, model, group)} of the variables of a rule."""
    V = {}
    def cellvar(name, g, m, k, dom=LEVELS):
        V[name] = (lvl(C[(g, m, k)]), dom, m, g)
    def qvar(name, rung, switch, g, m):
        ln = Q.get((rung, switch))
        val = ln["status"] if ln and ln["status"] in QSTAT else None
        V[name] = (val, QSTAT, m, g)
    if rid == "2.1":
        for m in MODELS:
            cellvar(m, "scrambled", m, "correction_only")
    elif rid == "2.2":
        for m in MODELS:
            cellvar("co_" + m, "views_base", m, "correction_only")
            cellvar("comp_" + m, "views_base", m, "composite_v3")
    elif rid == "2.3":
        for g in ("two_step_native", "two_step_fixed"):
            for m in MODELS:
                cs = [C[(g, m, k)] for k in ("composite_v3", "imputation_raw", "correction_v3_raw", "uncertainty_raw")]
                val = None
                if all(c["complete"] for c in cs):
                    val = up(cs[0], "2x") and not any(down(c, "2x") for c in cs[1:])
                V[f"{g}_{m}"] = (val, (True, False), m, g)
    elif rid == "2.4":
        cellvar("ucor", "fixed_depth", "unet", "correction_v3_raw")
        cellvar("uimp", "fixed_depth", "unet", "imputation_raw")
        qvar("qca", "rung3", "fixed_depth", "fixed_depth", "convattn")
    elif rid == "2.5":
        qvar("qun", "rung4", "average_input", "average_input", "unet")
    elif rid == "2.6":
        qvar("qca", "rung3", "impute_x4", "impute_x4", "convattn")
    return V


def _outcome(rid, a):
    """(branch, confidence or None) of a full assignment."""
    if rid == "2.1":
        dn = sum(_is_down(a[m]) for m in MODELS)
        dn2 = sum(a[m] == "d2" for m in MODELS)
        return ("uses", "strong" if dn2 >= 2 else "moderate") if dn >= 2 else ("weak", None)
    if rid == "2.2":
        dn = sum(_is_down(a["co_" + m]) for m in MODELS)
        dn2 = sum(a["co_" + m] == "d2" for m in MODELS)
        wi = sum(a["co_" + m] == "w" for m in MODELS)
        if dn >= 2:
            return ("all_views", "strong" if dn2 >= 2 else "moderate")
        if wi >= 2 and not any(_is_down(a["comp_" + m]) for m in MODELS):
            return ("no_benefit", None)
        return ("provisional", None)
    if rid == "2.3":
        return ("enters" if any(v for k, v in a.items()) else "off", None)
    if rid == "2.4":
        u = ("tradeoff" if a["ucor"] in ("u1", "u2") and _is_down(a["uimp"]) else
             "not_held" if a["ucor"] == "w" else "other")
        return ((u, a["qca"]), None)
    if rid == "2.5":
        return (a["qun"], None)
    if rid == "2.6":
        return (a["qca"], None)


def evaluate(rid, V):
    """(decided, branch, confidence text, open models of the confidence clause)."""
    names_ = list(V)
    doms = [[V[n][0]] if V[n][0] is not None else list(V[n][1]) for n in names_]
    outs = {_outcome(rid, dict(zip(names_, vals))) for vals in itertools.product(*doms)}
    branches = {o[0] for o in outs}
    if len(branches) != 1:
        return False, None, None, []
    confs = {o[1] for o in outs}
    if len(confs) == 1:
        return True, branches.pop(), confs.pop(), []
    low = min(confs, key=CONF_ORDER.index)
    open_models = sorted({V[n][2] for n in names_ if V[n][0] is None}, key=MODELS.index)
    return True, branches.pop(), f"at least {low}, final level waits for {names(open_models)}", open_models


def _conf_word(conf):
    return conf.split(",")[0] if conf else conf


def apply_rules(table, qualify_lines, p16=None, qualify_ran=True):
    """{rule: result}. A result: title, groups, decided, waiting (list of text), pending (arms still
    running that cannot change a decided verdict), verdict, code, lines, rows, cells, figure_key."""
    C = {(g, m, k): cell(table, g, m, k, p16) for g in GROUPS for m in MODELS for k in KEYS}
    Q = {(ln["rung"], ln["switch"]): ln for ln in qualify_lines}
    out = {}
    for rid, (title, groups, rows, fkey) in RULES.items():
        incomplete = []
        for g in groups:
            miss = []
            for m in MODELS:
                cs = [C[(g, m, k)] for k in KEYS]
                if not cs[0]["seeds_ok"]:
                    miss.append((m, cs[0]["cls"].replace("incomplete (", "").rstrip(")")))
                elif not all(c["complete"] for c in cs):
                    miss.append((m, "no value for " + ", ".join(METRIC[c["key"]] for c in cs if not c["complete"])))
            if len(miss) == len(MODELS) and len({x for _, x in miss}) == 1:
                incomplete.append(f"{GROUPS[g][0]} on all three models ({miss[0][1]} each)")
            elif miss:
                parts = [f"the {MODEL[m]} ({x})" for m, x in miss]
                incomplete.append(f"{GROUPS[g][0]} on " + (parts[0] if len(parts) == 1 else
                                                           ", ".join(parts[:-1]) + " and " + parts[-1]))
        V = _vars(rid, C, Q, qualify_ran)
        decided, branch, conf, conf_open = evaluate(rid, V)
        res = dict(rule=rid, title=title, groups=list(groups), design_rows=list(rows), figure_key=fkey,
                   decided=decided, waiting=[], pending=[], code="not_decided", verdict="", lines=[], rows={},
                   qualify={}, branch=branch, confidence=conf,
                   cells=[C[(g, m, k)] for g in groups for m in MODELS for k in KEYS])
        for name, (val, dom, m, g) in V.items():
            if dom is QSTAT:
                rung = "rung3" if m == "convattn" else "rung4"
                sw = {"fixed_depth": "fixed_depth", "average_input": "average_input", "impute_x4": "impute_x4"}[g]
                ln = Q.get((rung, sw))
                res["qualify"][f"{rung} {sw}"] = ln["status"] if ln else ("not run" if not qualify_ran else "absent")
        if decided:
            res["pending"] = incomplete
        else:
            res["waiting"] = list(incomplete)
            for k, v in res["qualify"].items():
                if v in ("not run", "absent"):
                    res["waiting"].append(f"the `{k}` line of mix_arms.py qualify "
                                          f"({'qualify did not run' if v == 'not run' else 'line absent'})")
                elif v == "no_score" and not incomplete:
                    res["waiting"].append(f"the `{k}` line of mix_arms.py qualify (no 3-seed score)")
        globals()["_rule_" + rid.replace(".", "")](res, C, Q, qualify_ran)
        if not decided:
            res["code"] = "not_decided"
            res["verdict"] = "Not decided: it waits for " + "; ".join(res["waiting"]) + "."
        elif res["pending"]:
            res["pending_text"] = ("Still running: " + "; ".join(res["pending"]) + ". "
                                   + ("It cannot" if len(res["pending"]) == 1 else "They cannot")
                                   + " change this verdict.")
            res["verdict"] += " " + res["pending_text"]
        out[rid] = res
    return out, C


def _row_evidence(res, C, keys):
    """One evidence sentence over the rule's groups and the given keys, complete cells with numbers."""
    parts = []
    for g in res["groups"]:
        for k in keys:
            parts.append(f"{GROUPS[g][0]}, {METRIC[k]}: " +
                         "; ".join(ctext(C[(g, m, k)]) for m in MODELS))
    return ". ".join(parts)


def _known(cs, test):
    return [c["model"] for c in cs if c["complete"] and test(c)]


def _rule_21(res, C, Q, qr):
    cs = [C[("scrambled", m, "correction_only")] for m in MODELS]
    cr = [C[("scrambled", m, "correction_v3_raw")] for m in MODELS]
    res["lines"].append("Correction-only term, scrambled covariates minus base: " +
                        "; ".join(ctext(c) for c in cs) + ".")
    res["lines"].append("Correction on crop pairs: " + "; ".join(ctext(c) for c in cr) + ".")
    ev = _row_evidence(res, C, ("correction_only", "correction_v3_raw"))
    base_status = ('value settled by PI ruling; the field set is provisional (run 2.8); '
                   '"the models use covariates to correct" is ')
    if not res["decided"]:
        res["rows"][8] = dict(evidence=ev, wait=True)
        return
    dn, dn2 = _known(cs, down), _known(cs, lambda c: down(c, "2x"))
    crd = _known(cr, down)
    crd16 = [c["model"] for c in cr if c.get("cls16") in ("beyond", "2x") and (c["d16"] or 0) < 0]
    cr_open = [c["model"] for c in cr if not c["complete"]]
    rec = (f" Recorded separately: the correction on crop pairs is lower beyond the seed range on {names(crd)}"
           + (f" of the finished models ({names(cr_open)} not finished)" if cr_open else "")
           + f" (on the 16 regions: {names(crd16)}).")
    conf = res["confidence"]
    if res["branch"] == "uses":
        cw = _conf_word(conf)
        res["code"] = "uses_covariates_" + (conf if " " not in conf else "at_least_" + cw.split()[-1])
        res["verdict"] = (f"The models use covariates for same-view correction (settled, {conf}). Scrambled "
                          f"covariates lower the correction-only term beyond the seed range on {names(dn)}, "
                          f"and beyond twice the range on {names(dn2)}." + rec)
        res["rows"][8] = dict(evidence=ev, status=base_status + f"settled, {conf} (rule 2.1, 256 regions)",
                              confidence=f"{conf} (use of covariates for correction, 256 regions); weak (field set)",
                              short_status="value by PI ruling; use for correction settled",
                              short_conf=f"{cw} (use, 256 regions); weak (field set)")
    else:
        res["code"] = "weak"
        res["verdict"] = (f"The claim that the models use covariates for correction becomes weak: the "
                          f"correction-only term is lower beyond the seed range on fewer than two models ({names(dn)}). "
                          f"The covariates stay by the PI ruling, and the minimum covariate set (run 2.8) gets "
                          f"priority." + rec)
        res["rows"][8] = dict(evidence=ev, status=base_status + "weak (rule 2.1); run 2.8 gets priority",
                              confidence="weak (256 regions)", short_status="value by PI ruling; use weak",
                              short_conf="weak")


def _rule_22(res, C, Q, qr):
    cs = [C[("views_base", m, "correction_only")] for m in MODELS]
    cc = [C[("views_base", m, "composite_v3")] for m in MODELS]
    res["lines"].append("Correction-only term, base view only minus all views: " +
                        "; ".join(ctext(c) for c in cs) + ".")
    res["lines"].append("Composite: " + "; ".join(ctext(c) for c in cc) + ".")
    ev = _row_evidence(res, C, ("correction_only", "composite_v3"))
    if not res["decided"]:
        res["rows"][23] = dict(evidence=ev, wait=True)
        return
    dn, dn2 = _known(cs, down), _known(cs, lambda c: down(c, "2x"))
    wi = _known(cs, lambda c: c["cls"] == "within")
    conf = res["confidence"]
    if res["branch"] == "all_views":
        cw = _conf_word(conf)
        res["code"] = "all_views_" + (conf if " " not in conf else "at_least_" + cw.split()[-1])
        res["verdict"] = (f"All counterfactual views stay in training (settled, {conf}). Base view only lowers "
                          f"the correction-only term beyond the seed range on {names(dn)} and beyond twice the "
                          f"range on {names(dn2)}.")
        res["rows"][23] = dict(evidence=ev, status="settled (rule 2.2, 256 regions)",
                               confidence=f"{conf} (correction-only term on 256 regions)",
                               short_status="settled (256 regions)", short_conf=f"{cw} (256 regions)")
    elif res["branch"] == "no_benefit":
        res["code"] = "no_benefit_pi_decides"
        res["verdict"] = ("The views show no benefit on 256 regions: the correction-only term is within the "
                          f"seed range on {names(wi)} and no composite is lower beyond the range. "
                          "Base view only is the simpler arm; the PI decides.")
        res["rows"][23] = dict(evidence=ev, status="provisional: no benefit on 256 regions; the PI decides (rule 2.2)",
                               confidence="moderate (null, 256 regions)", short_status="the PI decides",
                               short_conf="moderate (null)")
    else:
        res["code"] = "all_views_provisional"
        res["verdict"] = "All views stay, provisional, weak: the outcome meets neither case of the rule."
        res["rows"][23] = dict(evidence=ev, status="provisional (rule 2.2: all views stay)", confidence="weak",
                               short_status="provisional", short_conf="weak")


def _rule_23(res, C, Q, qr):
    for g in res["groups"]:
        res["lines"].append(f"{GROUPS[g][0].capitalize()}, composite: " +
                            "; ".join(ctext(C[(g, m, 'composite_v3')]) for m in MODELS) + ".")
    uf = [C[("two_step_fixed", "unet", k)] for k in ("correction_v3_raw", "uncertainty_raw")]
    res["lines"].append("U-Net, fixed base depth: correction on crop pairs " + ctext(uf[0], False) +
                        "; calibration " + ctext(uf[1], False) + ".")
    ev = _row_evidence(res, C, ("composite_v3",)) + ". U-Net fixed base depth: correction on crop pairs " + \
        ctext(uf[0], False) + "; calibration " + ctext(uf[1], False)
    if not res["decided"]:
        res["rows"][24] = dict(evidence=ev, wait=True)
        return
    enters = []
    for g in res["groups"]:
        for m in MODELS:
            cs = [C[(g, m, k)] for k in ("composite_v3", "imputation_raw", "correction_v3_raw", "uncertainty_raw")]
            if all(c["complete"] for c in cs) and up(cs[0], "2x") and not any(down(c, "2x") for c in cs[1:]):
                enters.append((g, m))
    tr = ""
    if up(uf[0]):
        tr = (" The U-Net trade-off is recorded: with a fixed base depth, correction on crop pairs is up beyond the "
              f"seed range ({num(uf[0]['d'])}) and calibration moves {num(uf[1]['d'])}; off stays.")
    elif not uf[0]["complete"]:
        tr = " The U-Net trade-off check (fixed base depth) waits for its arm."
    if res["branch"] == "off":
        res["code"] = "off_settled"
        res["verdict"] = ("Two-step routing stays off (settled). No arm raises the composite by more than twice "
                          "the seed range with no goal lower by twice its range." + tr)
        res["rows"][24] = dict(evidence=ev, status="settled off (rule 2.3, 256 regions)", confidence="moderate",
                               short_status="settled off (256 regions)", short_conf="moderate")
    else:
        who = "; ".join(f"{GROUPS[g][0]} on the {MODEL[m]}" for g, m in enters)
        res["code"] = "enters_pool_pi_decides"
        res["verdict"] = (f"Two-step routing qualifies for the mix-and-match pool ({who}"
                          + ("; arms still running can add models" if res["pending"] else "")
                          + "): the composite rises by more than twice the seed range with no goal lower by twice "
                          "its range. The PI decides (a new arm, about 120 to 250 CPU-hours per model and 3 seeds)."
                          + tr)
        res["rows"][24] = dict(evidence=ev, status="provisional: two-step qualifies for the pool; the PI decides (rule 2.3)",
                               confidence="moderate", short_status="the PI decides (pool)", short_conf="moderate")


def _rule_24(res, C, Q, qr):
    for m in MODELS:
        res["lines"].append(f"{MODEL[m].capitalize() if m != 'unet' else 'U-Net'}: correction on crop pairs "
                            + ctext(C[('fixed_depth', m, 'correction_v3_raw')], False) + "; imputation "
                            + ctext(C[('fixed_depth', m, 'imputation_raw')], False) + ".")
    ev = _row_evidence(res, C, ("correction_v3_raw", "imputation_raw"))
    if not res["decided"]:
        res["rows"][25] = dict(evidence=ev, wait=True)
        return
    u_code, st = res["branch"]
    ucor, uimp = C[("fixed_depth", "unet", "correction_v3_raw")], C[("fixed_depth", "unet", "imputation_raw")]
    if u_code == "tradeoff":
        u = ("U-Net: the trade-off holds (correction on crop pairs up, imputation down, both beyond the seed "
             "range); the switch stays in the U-Net pool by PI ruling, and the native-depth score of run 2.7 "
             "decides whether it can be deployed.")
    elif u_code == "not_held":
        u = ("U-Net: correction on crop pairs is within the seed range, so the 16-region gain did not hold; "
             "tell the PI, who put the switch in the pool.")
    else:
        u = ("U-Net: the outcome is outside the two cases of the rule (correction on crop pairs "
             f"{CLS_WORD[ucor['cls']]}, {'up' if ucor['d'] > 0 else 'down'}; imputation {CLS_WORD[uimp['cls']]}); "
             "the PI reads it.")
    ca = ("Conv + attention: the qualify line keeps the switch in the conv + attention pool."
          if st == "qualifies" else
          "Conv + attention: the qualify line removes the switch from the conv + attention pool (remove it from "
          "POOL in tools/minicandi/mix_arms.py before the mix-and-match round).")
    res["code"] = {"tradeoff": "unet_tradeoff_holds", "not_held": "unet_gain_not_held", "other": "unet_other"}[u_code] \
        + ("+convattn_keeps" if st == "qualifies" else "+convattn_removed")
    res["verdict"] = (u + " " + ca + " The conv autoencoder (2 seeds) is reported only. "
                      "No fixed-depth value enters the recipe from these scores alone.")
    res["rows"][25] = dict(evidence=ev, status="provisional (rule 2.4 applied; run 2.7)",
                           confidence="weak", short_status="provisional (mix round)", short_conf="weak")


def _rule_25(res, C, Q, qr):
    for k in ("correction_only", "correction_v3_raw"):
        res["lines"].append(f"{METRIC[k].capitalize()}: " +
                            "; ".join(ctext(C[('average_input', m, k)]) for m in MODELS) + ".")
    ev = _row_evidence(res, C, ("correction_only", "correction_v3_raw"))
    if not res["decided"]:
        res["rows"][13] = dict(evidence=ev, wait=True)
        return
    ae = C[("average_input", "convae", "correction_only")]
    rec = (" The conv autoencoder correction-only term is up beyond the seed range again (" + num(ae["d"]) +
           "); it is recorded, and the value stays off on the conv models." if up(ae) else
           " The conv autoencoder check of the correction-only term waits for its arm; the value stays off on the "
           "conv models." if not ae["complete"] else
           " On the conv models the value stays off (no qualifying term).")
    if res["branch"] == "qualifies":
        res["code"] = "unet_pool_keeps"
        res["verdict"] = "The average baseline input stays in the U-Net pool (qualify line)." + rec
        res["rows"][13] = dict(evidence=ev, status="provisional (run 2.7; in the U-Net pool by rule 2.5)",
                               confidence="weak", short_status="provisional (U-Net pool)", short_conf="weak")
    else:
        res["code"] = "unet_pool_removed"
        res["verdict"] = ("The average baseline input stays off (settled). It leaves the U-Net pool (qualify line; "
                          "remove it from POOL in tools/minicandi/mix_arms.py)." + rec)
        res["rows"][13] = dict(evidence=ev, status="settled off (rule 2.5, 256 regions)", confidence="weak",
                               short_status="settled off (256 regions)", short_conf="weak")


def _rule_26(res, C, Q, qr):
    for k in ("correction_v3_raw", "imputation_raw"):
        res["lines"].append(f"{METRIC[k].capitalize()}: " +
                            "; ".join(ctext(C[('impute_x4', m, k)]) for m in MODELS) + ".")
    ev = _row_evidence(res, C, ("correction_v3_raw", "imputation_raw"))
    if not res["decided"]:
        res["rows"][20] = dict(evidence=ev, wait=True)
        return
    ae = C[("impute_x4", "convae", "correction_v3_raw")]
    rec = (" The cost on the conv autoencoder is confirmed: correction on crop pairs " + num(ae["d"]) +
           " (seed range " + rng(ae["bar"]) + "); weight 4 is not in its pool." if down(ae) else
           " The conv autoencoder check of the correction on crop pairs waits for its arm." if not ae["complete"] else
           " The conv autoencoder correction on crop pairs is not lower beyond the seed range on 256 regions.")
    if res["branch"] == "qualifies":
        res["code"] = "convattn_pool_keeps"
        res["verdict"] = ("Imputation weight 4 stays in the conv + attention pool (qualify line). The weights stay "
                          "1 / 1 / 1 unless run 2.7 picks weight 4." + rec)
        res["rows"][20] = dict(evidence=ev, status="provisional (run 2.7; weight 4 stays in the conv + attention pool, rule 2.6)",
                               confidence="weak", short_status="provisional (weight 4 in the pool)", short_conf="weak")
    else:
        res["code"] = "convattn_pool_removed"
        res["verdict"] = ("The imputation weight stays 1 (settled). Weight 4 leaves the conv + attention pool "
                          "(qualify line; remove it from POOL in tools/minicandi/mix_arms.py)." + rec)
        res["rows"][20] = dict(evidence=ev, status="imputation weight settled at 1 (rule 2.6, 256 regions); the other weights and terms untested",
                               confidence="weak", short_status="weight 1 settled; others untested", short_conf="weak")


# ----------------------------------------------------------------------------------------- qualify
def run_qualify(table_path):
    """(lines, stdout, exit code); exit None when qualify could not run."""
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    try:
        p = subprocess.run([sys.executable, str(ROOT / "tools/minicandi/mix_arms.py"), "qualify",
                            "--summary", str(table_path)], capture_output=True, text=True, env=env, cwd=ROOT)
    except OSError as e:
        return [], f"qualify did not run: {e}", None
    lines = []
    for ln in p.stdout.splitlines():
        c = ln.split("\t")
        if len(c) >= 4 and c[0].startswith("rung"):
            lines.append(dict(rung=c[0], switch=c[1], status=c[2], run=c[3], detail=c[4] if len(c) > 4 else ""))
    if p.returncode not in (0, 1, 3):
        return lines, p.stdout + p.stderr, None
    return lines, p.stdout, p.returncode


# ------------------------------------------------------------------------------------------ counts
def counts(table):
    planned_total = sum(planned(g, m) for g in GROUPS for m in MODELS)
    scored = sum(min(int(table[arm_of(g, m)]["n_seeds"]), planned(g, m)) if arm_of(g, m) in table else 0
                 for g in GROUPS for m in MODELS)
    arms_complete = sum(1 for g in GROUPS for m in MODELS
                        if cell(table, g, m, KEYS[0])["complete"])
    return dict(planned=planned_total, scored=scored, arms=len(GROUPS) * len(MODELS), arms_complete=arms_complete)


# ------------------------------------------------------------------------------------- generated text
def texts(R, cnt, rstat, as_of):
    """Every sentence the deck and the report show about the rescoring, from the verdicts."""
    dec = [r for r in R.values() if r["decided"]]
    und = [r for r in R.values() if not r["decided"]]
    nd, nr = len(dec), len(R)
    rs = rstat.get("rescoring", {})
    t = {}
    t["as_of"] = as_of
    t["banner"] = ("256-region rescoring: result" if nd == nr else
                   f"256-region rescoring: {nd} of {nr} rules decided")
    W = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
    t["h2"] = (f"All {W[nr]} pre-written rules are decided on all regions" if nd == nr else
               f"{W[nd].capitalize()} of the {W[nr]} pre-written rules {'is' if nd == 1 else 'are'} decided on all "
               f"regions; the other {W[nr - nd]} wait for arms that have not finished")
    t["status"] = (f"Scores as of {as_of}: {cnt['scored']} of {cnt['planned']} run-seeds scored on all regions; "
                   f"{cnt['arms_complete']} of {cnt['arms']} arms are complete. The scoring runs on Nibi since "
                   f"{rs.get('submitted_day', '?')}, {rs.get('submitted_utc', '?')} UTC. One scoring needs about "
                   f"{rs.get('predict_hours_per_round', '?')} hours there, about {rs.get('fir_rate_factor', '?')} times "
                   f"the Fir rate, because the jobs wait on input and output. U-Net prediction parts reach their "
                   f"{rs.get('unet_limit_h', '?')}-hour limit at {rs.get('unet_limit_reach_pct', '?')}% and are "
                   f"resubmitted with a {rs.get('unet_new_limit_h', '?')}-hour limit (the store resumes). The same "
                   f"jobs wait in the Fir queue (jobs {rs.get('fir_jobs', '?')}); the PI cancels them after the Nibi "
                   f"scores are on disk.")
    for rid, r in R.items():
        lab = f"Rule {rid}, {r['title']}"
        t[f"rule{rid}"] = r["verdict"] if r["decided"] else "Not decided. Waits for: " + "; ".join(r["waiting"]) + "."
        t[f"rule{rid}.head"] = lab
    # views (rule 2.2)
    r = R["2.2"]
    comp = [C for C in r["cells"] if C["key"] == "correction_only" and C["complete"]]
    if r["decided"]:
        t["views"] = ("On all regions, rule 2.2 is decided. " + r["verdict"])
    else:
        t["views"] = ("Screening tier: the correction-only term rests on the pair regions. "
                      + (("On all regions so far: " + "; ".join(ctext(C) for C in comp) + ". ") if comp else "")
                      + "Rule 2.2 waits for " + "; ".join(r["waiting"]) + ".")
    r = R["2.6"]
    t["loss"] = ("On all regions, rule 2.6 is decided. " + r["verdict"] if r["decided"] else
                 "The rescoring on all regions decides whether it stays in the pool (rule 2.6); it waits for "
                 + "; ".join(r["waiting"]) + ".")
    t["heat1"] = _heat_text([R[k] for k in ("2.1", "2.3", "2.4", "2.5")])
    t["heat2"] = _heat_text([R[k] for k in ("2.6", "2.2")])
    wait_rows = sorted(n for r in und for n in r["rows"])
    upd = "; ".join(f"{_pl('row', r['design_rows'])} by rule {r['rule']}: {_first(r)}" for r in dec)
    wait = (f"{_pl('Row', wait_rows)} {'waits' if len(wait_rows) == 1 else 'wait'} for unfinished arms "
            f"({_pl('rule', [r['rule'] for r in und])}).") if und else ""
    t["design"] = ((f"On all regions the rescoring updated {upd}. ") if dec else
                   "On all regions the rescoring has decided no rule yet. ") + wait
    t["eval_card"] = (f"Settled scores (all regions) of the decisive ablation arms: {cnt['scored']} of "
                      f"{cnt['planned']} run-seeds scored, {nd} of {nr} rules decided.")
    t["sum_views"] = ("On all regions, the views keep the correction-only term up (rule 2.2, settled)."
                      if R["2.2"]["decided"] and R["2.2"]["code"].startswith("all_views_") and
                      R["2.2"]["code"] != "all_views_provisional" else
                      "They keep the correction-only term up on the pair regions; on all regions rule 2.2 "
                      + ("is decided: " + R["2.2"]["verdict"] if R["2.2"]["decided"] else "is not decided yet."))
    t["sum_eval"] = (f"Correction terms need all regions; the rescoring of the decisive arms has scored "
                     f"{cnt['scored']} of {cnt['planned']} run-seeds and decided {nd} of {nr} rules.")
    t["sum_design"] = ((f"The rescoring updated {upd}. ") if dec else "") + wait + \
        " Many rows also wait for the mix-and-match round and the covariate set."
    # report
    t["abstract"] = (f"The pre-written check of 16 against 256 regions failed, so correction effects need the "
                     f"256-region rescoring of the decisive arms. At {as_of} it had scored {cnt['scored']} of "
                     f"{cnt['planned']} run-seeds and decided {nd} of {nr} pre-written rules"
                     + (": " + "; ".join(_tiny(r) for r in dec) if dec else "")
                     + (f". The other {nr - nd} rules wait for unfinished arms." if und else "."))
    t["crit3"] = (f"Metrics, selection rule and scorers are defined and pass their tests, and the replicate ladder "
                  f"ran on the real data. The 16-region screening tier failed its check, so correction terms need "
                  f"256 regions; the rescoring has scored {cnt['scored']} of {cnt['planned']} run-seeds and decided "
                  f"{nd} of {nr} rules.")
    t["crit4"] = (((f"The 256-region rescoring updated {upd}. ") if dec else "")
                  + "Many rows are provisional; the choice of architecture is the PI's.")
    t["discuss"] = (("The third reading is settled on 256 regions by rule 2.2." if R["2.2"]["decided"] and
                     R["2.2"]["code"].startswith("all_views_") and R["2.2"]["code"] != "all_views_provisional"
                     else "The third reading rests on 16-region scores until rule 2.2 of the rescoring is decided."))
    t["concl_eval"] = (f"the screening tier failed its check, so the correction terms need 256 regions; the "
                       f"rescoring has scored {cnt['scored']} of {cnt['planned']} run-seeds and decided {nd} of "
                       f"{nr} rules")
    t["concl_design"] = ("three architectures tie with a trade-off between imputation and correction, the count "
                         "input with a depth offset is settled"
                         + ((f", the 256-region rescoring updated {upd}") if dec else "")
                         + ((f", and {_pl('row', wait_rows)} {'stays' if len(wait_rows) == 1 else 'stay'} provisional "
                             "until their arms finish") if wait_rows else "") + ".")
    t["read"] = " ".join(
        (f"Rule {r['rule']} ({r['title']}). " + (r["verdict"] + " " + " ".join(r["lines"]) if r["decided"] else
                                                 "Not decided; it waits for " + "; ".join(r["waiting"]) + "."
                                                 + (" Complete arms so far: " + _complete_lines(r) if _complete_lines(r) else "")))
        for r in R.values())
    return t


def _first(r):
    f = r["verdict"].split(". ")[0].rstrip(".")
    return f[0].lower() + f[1:]


def _pl(word, ns):
    ns = [str(n) for n in ns]
    return (word + "s " if len(ns) > 1 else word + " ") + _nlist(ns)


def _heat_text(rs):
    dec = [r for r in rs if r["decided"]]
    und = [r for r in rs if not r["decided"]]
    out = [f"Rule {r['rule']} on all regions: {_first(r)}." for r in dec]
    if und:
        out.append(f"{_pl('Rule', [r['rule'] for r in und])} {'is' if len(und) == 1 else 'are'} not decided on all "
                   f"regions yet (arms not finished).")
    return " ".join(out)


def _complete_lines(r):
    by = {}
    for C in r["cells"]:
        if C["complete"] and C["key"] == r["figure_key"]:
            by.setdefault(C["group"], []).append(ctext(C))
    return " ".join(f"{GROUPS[g][0]}, {METRIC[r['figure_key']]}: " + "; ".join(v) + "." for g, v in by.items())


def _short(r):
    if r["decided"]:
        return f"Rule {r['rule']} ({r['title']}), all regions: {r['verdict']}"
    return f"Rule {r['rule']} ({r['title']}): not decided on all regions; it waits for {'; '.join(r['waiting'])}."


def _tiny(r):
    first = r["verdict"].split(". ")[0].rstrip(".")
    return f"rule {r['rule']}, {r['title']}: {first[0].lower() + first[1:]}"


def _nlist(ns):
    ns = [str(n) for n in ns]
    return ns[0] if len(ns) == 1 else ", ".join(ns[:-1]) + " and " + ns[-1] if ns else "none"


def _rlist(rs):
    return _nlist([r["rule"] for r in rs])


# --------------------------------------------------------------------------------------- outputs
def write_outputs(folder, R, C, cnt, qual, table_path, as_of, t, stamp):
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "verdicts.tsv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["rule", "title", "decided", "code", "verdict", "design_rows", "new_status", "new_confidence",
                    "waiting", "qualify_lines"])
        for r in R.values():
            rows = [n for n, v in r["rows"].items() if "status" in v]
            w.writerow([r["rule"], r["title"], "yes" if r["decided"] else "no", r["code"],
                        r["verdict"], ",".join(str(n) for n in r["design_rows"]),
                        " | ".join(f"row {n}: {r['rows'][n]['status']}" for n in rows),
                        " | ".join(f"row {n}: {r['rows'][n]['confidence']}" for n in rows),
                        "; ".join(r["waiting"]),
                        "; ".join(f"{k} {v}" for k, v in r["qualify"].items())])
    with open(folder / "arms.tsv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["rule", "arm_group", "model", "arm", "key", "n_seeds", "planned_seeds", "arm_mean", "ref_mean",
                    "d", "bar", "class", "d_16_regions", "sign_agrees_with_16"])
        rule_of = {g: rid for rid, v in RULES.items() for g in v[1]}
        for g in GROUPS:
            for m in MODELS:
                for k in KEYS:
                    c = C[(g, m, k)]
                    f = lambda x: "" if x is None else f"{x:.4f}"
                    w.writerow([rule_of[g], g, MODEL[m], c["arm"], k, c["n_seeds"], c["planned"],
                                f(c["arm_mean"]), f(c["ref_mean"]), f(c["d"]), f(c["bar"]), c["cls"],
                                f(c["d16"]), c["sign16"]])
    (folder / "qualify.txt").write_text(f"# mix_arms.py qualify --summary {table_path}\n# exit code: {qual[2]}\n"
                                        + qual[1], encoding="utf-8")
    payload = dict(stamp=stamp, as_of=as_of, table=str(table_path), counts=cnt,
                   qualify=dict(exit=qual[2], lines=qual[0]), rules=R, text=t,
                   keys=METRIC, models=MODEL, groups={g: v[0] for g, v in GROUPS.items()})
    (folder / "verdicts.json").write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (folder / "RESCORING_READ.md").write_text(read_md(R, C, cnt, qual, table_path, as_of, stamp), encoding="utf-8")


COMMAND = ("/Users/mforooz/miniforge3/envs/candii/bin/python tools/onion_eval/apply_rescoring_rules.py "
           "<arm-summary tsv> --as-of <YYYY-MM-DDTHH:MMZ of the table>")


def read_md(R, C, cnt, qual, table_path, as_of, stamp):
    L = [f"# 256-region rescoring: the read of {stamp}", "",
         "**Rerun** (from the worktree root; the foreman reruns it with each new arm-summary table):", "",
         "```bash", COMMAND, "```", "",
         "It writes this folder (and the same folder in the foreman worktree), copies the table to "
         f"`cruxvault/{TABLE_REL}`, updates the design table of `{DES_REL}`, points "
         f"`{RSTAT_REL}` at this folder, rebuilds the deck and the report, and runs both verify steps with "
         "`--strict`. Code: `tools/onion_eval/apply_rescoring_rules.py`.", "",
         f"**Table.** `{table_path}`, scores as of {as_of}. {cnt['scored']} of {cnt['planned']} run-seeds are "
         f"scored; {cnt['arms_complete']} of {cnt['arms']} arms are complete.", "",
         "**Method.** For each arm, model and key: d is the arm mean minus the reference mean, and the bar is the "
         "larger of the two seed ranges (max minus min over seeds). The class is \"2x\" when |d| is larger than "
         "twice the bar, \"beyond\" when |d| is larger than the bar, and \"within\" otherwise. An arm with fewer "
         "seeds than planned is \"incomplete (k of n seeds)\" and gets no verdict. A rule is decided only when "
         "all its arms are complete (and, for rules 2.4, 2.5 and 2.6, when its qualify line has a score). The "
         "rule texts are section 2.0 of the design document; no threshold is changed.", ""]
    for r in R.values():
        L += [f"## Rule {r['rule']}: {r['title']}", ""]
        if r["decided"]:
            L += [f"**Decided.** {r['verdict']}", ""]
            for n, v in r["rows"].items():
                if "status" in v:
                    L += [f"Design row {n}: status \"{v['status']}\"; confidence \"{v['confidence']}\".", ""]
        else:
            L += [f"**Not decided.** It waits for: {'; '.join(r['waiting'])}.", "",
                  f"Design row(s) {', '.join(str(n) for n in r['design_rows'])} stay as they are.", ""]
        L += ["Cells (arm minus reference; incomplete arms give no number):", ""]
        L += [f"- {ln}" for ln in r["lines"]]
        if r["qualify"]:
            L += [f"- Qualify line(s): " + "; ".join(f"`{k}` {v}" for k, v in r["qualify"].items()) + "."]
        L += [""]
    L += ["## Mix-and-match pool check (`mix_arms.py qualify`)", "",
          f"Exit code: {qual[2]} (0 PASS, 1 FAIL, 3 WAIT; WAIT is expected on this rescoring, because five "
          "single runs of the pool are not among the 62 run-seeds).", "", "```", qual[1].rstrip(), "```", "",
          "## Every cell", "", "See `arms.tsv` beside this file: one row per arm, model and key, with d, the bar, "
          "the class, the seeds, and the 16-region d with a column that says whether the sign agrees.", ""]
    return "\n".join(L)


# ---------------------------------------------------------------------------------- design document
_ROW_RE = re.compile(r"^\| (\d+) \|(.*)\|$")
_MARK_RE = re.compile(r" ?<!--p256:([A-Za-z0-9+/=]+)-->.*?<!--/p256-->")
SEC_START, SEC_END = "<!-- p256-section start (apply_rescoring_rules.py) -->", "<!-- p256-section end -->"


def update_design(text, R, as_of, stamp):
    """The design document with the rows of each rule updated, idempotently: each touched row keeps
    its original status and confidence in a base64 marker in its evidence cell, so a rerun restores the
    original and applies the new verdict."""
    out = []
    upd = {n: (r, v) for r in R.values() for n, v in r["rows"].items()}
    for line in text.split("\n"):
        m = _ROW_RE.match(line)
        if not m:
            out.append(line)
            continue
        cells = [c.strip() for c in m.group(2).split(" | ")]
        if len(cells) != 5:
            out.append(line)
            continue
        ev = cells[2]
        mk = _MARK_RE.search(ev)
        if mk:
            orig = json.loads(base64.b64decode(mk.group(1)).decode("utf-8"))
            cells[3], cells[4] = orig["confidence"], orig["status"]
            ev = _MARK_RE.sub("", ev)
        n = int(m.group(1))
        if n in upd:
            r, v = upd[n]
            orig = base64.b64encode(json.dumps({"confidence": cells[3], "status": cells[4]}).encode()).decode()
            if v.get("wait"):
                add = (f"256 regions (rescoring read {stamp}, scores as of {as_of}): rule {r['rule']} not decided; "
                       f"it waits for {'; '.join(r['waiting'])}. Cells so far: {v['evidence']}.")
            else:
                add = (f"256 regions (rescoring read {stamp}, scores as of {as_of}), rule {r['rule']}: "
                       f"{r['verdict']} {v['evidence']}.")
                cells[3], cells[4] = v["confidence"], v["status"]
            ev = ev + f" <!--p256:{orig}-->" + add.replace("|", "/") + "<!--/p256-->"
        cells[2] = ev
        out.append(f"| {n} | " + " | ".join(cells) + " |")
    text = "\n".join(out)
    sec = design_section(R, as_of, stamp)
    if SEC_START in text:
        text = re.sub(re.escape(SEC_START) + r".*?" + re.escape(SEC_END), lambda _: sec, text, flags=re.S)
    else:
        anchor = "## 2. Provisional rows"
        assert anchor in text, "design document: no section 2"
        text = text.replace(anchor, sec + "\n\n" + anchor, 1)
    nd = sum(1 for r in R.values() if r["decided"])
    text = re.sub(r"^status: .*$", f"status: from the 16-ablation evidence and the 256-region rescoring read of "
                  f"{stamp} ({nd} of {len(R)} rules decided, scores as of {as_of})", text, count=1, flags=re.M)
    return text


def design_section(R, as_of, stamp):
    day = stamp[:4] + "-" + stamp[4:6] + "-" + stamp[6:8]
    L = [SEC_START, f"## 256-region rescoring: what it settled ({day})", "",
         f"Generated by `tools/onion_eval/apply_rescoring_rules.py` from the arm summary of scores as of {as_of} "
         f"(read folder `results/rescoring_p256/read_{stamp}/`, git-ignored; the table is "
         f"`{TABLE_REL}`). It applies rules 2.1 to 2.6 of section 2.0 as written. A rule is decided only when "
         "all its arms have their planned seeds. Rerun: `" + COMMAND + "`.", "",
         "| rule | arms | verdict | design rows | new status |", "|---|---|---|---|---|"]
    for r in R.values():
        arms = "; ".join(GROUPS[g][0] for g in r["groups"])
        st = "; ".join(f"row {n}: {v['status']}" for n, v in r["rows"].items() if "status" in v) or \
            ("unchanged (waits for: " + "; ".join(r["waiting"]) + ")" if not r["decided"] else "unchanged")
        L.append(f"| rule {r['rule']} ({r['title']}) | {arms} | "
                 + (r["verdict"] if r["decided"] else "not decided").replace("|", "/") + " | "
                 + ", ".join(str(n) for n in r["design_rows"]) + " | " + st.replace("|", "/") + " |")
    L += ["", SEC_END]
    return "\n".join(L)


# -------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("table", help="arm-summary TSV of p256_sv scores (select_rule.py arm-summary)")
    ap.add_argument("--as-of", help="UTC time of the scores, e.g. 2026-11-18T09:45Z (default: file mtime)")
    ap.add_argument("--p16", default=str(VAULT / P16_REL), help="the 16-region arm summary")
    ap.add_argument("--foreman-vault", default=str(FOREMAN_VAULT))
    ap.add_argument("--stamp", help="read folder stamp (default: now, UTC, YYYYMMDDTHHMMZ)")
    ap.add_argument("--no-build", action="store_true", help="skip the deck and report builds and verify")
    ap.add_argument("--dry-run", action="store_true", help="print the verdicts; write nothing")
    a = ap.parse_args(argv)

    tp = pathlib.Path(a.table).resolve()
    table = read_table(tp)
    p16 = read_table(a.p16) if a.p16 and pathlib.Path(a.p16).is_file() else None
    as_of = a.as_of or dt.datetime.fromtimestamp(tp.stat().st_mtime, dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    as_of = as_of.replace("T", " ").replace("Z", " UTC")
    stamp = a.stamp or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%MZ")
    qual = run_qualify(tp)
    R, C = apply_rules(table, qual[0], p16, qualify_ran=qual[2] is not None)
    cnt = counts(table)
    rstat_p = VAULT / RSTAT_REL
    rstat = json.loads(rstat_p.read_text(encoding="utf-8"))
    t = texts(R, cnt, rstat, as_of)
    for r in R.values():
        print(f"rule {r['rule']}: {'DECIDED' if r['decided'] else 'not decided'}: {t['rule' + r['rule']]}")
    print(f"qualify exit {qual[2]}; scored {cnt['scored']} of {cnt['planned']} run-seeds")
    if a.dry_run:
        return 0

    rel = f"{OUT_REL}/read_{stamp}"
    folders = [VAULT / rel]
    fv = pathlib.Path(a.foreman_vault)
    if fv.is_dir() and fv.resolve() != VAULT.resolve():
        folders.append(fv / rel)
    for f in folders:
        write_outputs(f, R, C, cnt, qual, tp, as_of, t, stamp)
        print(f"wrote {f}")
    dst = VAULT / TABLE_REL
    if tp != dst.resolve():
        shutil.copyfile(tp, dst)
    des = VAULT / DES_REL
    des.write_text(update_design(des.read_text(encoding="utf-8"), R, as_of, stamp), encoding="utf-8")
    rstat["rescoring"].update(read_dir=rel, read_table=TABLE_REL, read_as_of=as_of, scored_p256=str(cnt["scored"]),
                              planned_p256=str(cnt["planned"]), arms_complete=str(cnt["arms_complete"]),
                              rules_decided=str(sum(1 for r in R.values() if r["decided"])), rules=str(len(R)))
    rstat_p.write_text(json.dumps(rstat, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"updated {des.name}, {rstat_p.name}; copied the table to {dst}")
    if a.no_build:
        return 0
    return build_and_verify()


def build_and_verify():
    def run(cmd, cwd):
        print("$ " + " ".join(cmd))
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        tail = (p.stdout + p.stderr).strip().splitlines()[-4:]
        print("\n".join("  " + x for x in tail))
        return p.returncode
    crux = [sys.executable, CRUX]
    bad = 0
    bad |= run([DECK_PY, f"{DECK_DIR}/build_deck.py"], VAULT) != 0
    bad |= run([sys.executable, f"{STORY_DIR}/build_story.py"], VAULT) != 0
    bad |= run(["latexmk", "-g", "-pdf", "-interaction=nonstopmode", "report.tex"], VAULT / STORY_DIR) != 0
    bad |= run(crux + ["deck", "--verify", f"{DECK_DIR}/index.html", "--strict"], VAULT) != 0
    bad |= run(crux + ["story", "--verify", f"{STORY_DIR}/report.tex", "--strict"], VAULT) != 0
    print("build and verify: " + ("FAILED" if bad else "all passed"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
