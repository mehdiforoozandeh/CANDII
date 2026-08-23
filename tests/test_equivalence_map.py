"""t22 — the cutover's completeness gate, held to a real key set.

`EVAL_PLAN.md` D15 makes the cutover hard: `eval.py` is **deleted**, gated on a published
key-by-key equivalence report. A report is only a gate if it is complete — a key that quietly
vanished between the two suites is precisely the thing a reader would never notice, and "we
checked" is not evidence.

So `tests/fixtures/eval_key_skeleton.json` carries the **key structure** of a real `candi.train`
run's `M1/M2/M3/S14` output — 651 leaf keys, every measured value stripped to `null`, because a
recorded number belongs in `cruxvault/` and never in `tests/`. This file asserts the rule table
in `tools/equivalence.py` claims all 651.

It stays useful after `eval.py` is gone: it is then the only record of what the old suite emitted,
and the only thing that can catch a rule being deleted along with the code it described.

THE FIXTURE IS FROZEN AND NOT MIGRATED. When the covariate keys were renamed off their `C1`..`C6`
codes to `covuse` / `covshare` / `depthdir` / `depthcounterfact` / `covspec` / `depthblind` /
`biokeep`, only the RIGHT-hand (bench) side of the rule table moved. The fixture holds the LEFT-hand
side — the h5-era `M1`/`M2`/`M3`/`S14` keys — and those spellings are what the archived jsons
actually contain, so editing them would make the fixture describe a file that never existed.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SKELETON = REPO / "tests" / "fixtures" / "eval_key_skeleton.json"


def _load_tool():
    """`tools/` is not a package, so the module is loaded by path rather than imported.

    It must be registered in `sys.modules` BEFORE `exec_module` runs. `equivalence.py` uses
    `from __future__ import annotations` on a `@dataclass`, so the decorator resolves its field
    types by looking the module up by name at class-creation time; an unregistered module resolves
    to `None` and `dataclasses` raises on `None.__dict__`.
    """
    spec = importlib.util.spec_from_file_location("equivalence", REPO / "tools" / "equivalence.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["equivalence"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tool():
    return _load_tool()


@pytest.fixture(scope="module")
def skeleton():
    return json.loads(SKELETON.read_text())


def test_the_skeleton_carries_keys_and_no_measured_values(skeleton) -> None:
    """A number in a test fixture is a number outside the vault. None may survive here."""
    def leaves(o):
        if isinstance(o, dict):
            for v in o.values():
                yield from leaves(v)
        elif isinstance(o, list):
            assert o == [], "list contents are not key structure"
        else:
            yield o
    body = {k: v for k, v in skeleton.items() if not k.startswith("_")}
    assert set(leaves(body)) == {None}
    assert "why_values_are_null" in skeleton["_provenance"]


def test_every_key_eval_py_emits_is_claimed_by_exactly_one_rule(tool, skeleton) -> None:
    """THE GATE. An unclaimed key means the report would be silently incomplete."""
    claimed, orphan = tool.cover(skeleton)
    assert not orphan, (
        f"{len(orphan)} key(s) claimed by no rule — the D15 report would be incomplete: "
        f"{orphan[:10]}")
    assert len(claimed) == 1411, (
        f"the fixture has {len(claimed)} leaf keys, not the recorded 1411. If eval.py's output "
        f"really changed, re-record the skeleton with `python tools/equivalence.py skeleton "
        f"<run.json> {SKELETON}` and say so; otherwise the flattener drifted.")


def test_the_report_refuses_to_render_while_a_key_is_unclaimed(tool, skeleton) -> None:
    """The gate must be enforced by the generator, not only by this test."""
    hobbled = [r for r in tool.RULES if "M3" not in r.old]
    original, tool.RULES = tool.RULES, hobbled
    try:
        with pytest.raises(SystemExit, match="claimed by no rule"):
            tool.report(skeleton, {})
    finally:
        tool.RULES = original


def test_every_verdict_carries_what_it_obliges(tool) -> None:
    """`dropped` is the only verdict with no counterpart, and every rule must say why."""
    for r in tool.RULES:
        assert (r.new is None) == (r.verdict == "dropped"), r.old
        assert len(r.why) > 20, f"{r.old}: a one-word reason is not a reason"


def test_the_denoise_half_mirrors_the_imputation_half(tool) -> None:
    """They are one measurement on two targets; a hand-written second copy would drift.

    `eval.py` is the cautionary case — `den_per_assay` and `imp_per_assay` were built by two
    near-identical code paths. Here the denoise rules are DERIVED, so the two cannot disagree.
    """
    heads = ("M1.imp_per_target.", "M1.imp_per_assay.", "M1.imp.")
    imp = {r.old: r for r in tool.RULES if r.old.startswith(heads)}
    den = {r.old: r for r in tool.RULES
           if r.old.startswith(("M1.den_per_target.", "M1.den_per_assay.", "M1.den."))}
    assert imp and len(den) == len(imp)
    for old, r in imp.items():
        twin = den[old.replace("M1.imp_per_target.", "M1.den_per_target.")
                      .replace("M1.imp_per_assay.", "M1.den_per_assay.")
                      .replace("M1.imp.", "M1.den.")]
        assert twin.verdict == r.verdict, old


#: The three derived denoise rules that fire on nothing, and why that is a FINDING rather than a
#: defect in the derivation.
#:
#: `eval.py` computes the oracle-scale split PER ASSAY on both halves — `den_per_assay.<A>` really
#: does carry `crps_oracle_scaled`, `scale_error` and `marg_crps`. What it never builds is the
#: denoising **macro** of them: `imp_macro_crps` ships beside `imp_macro_crps_oracle_scaled`,
#: `imp_macro_scale_error` and `imp_macro_marg_crps`, while `den_macro_crps` ships alone.
#:
#: That is the level at which the rule bites. `AGENTS.md` §7.2 says raw CRPS is never quoted without
#: its `oracle_scaled` / `scale_error` split, and the macro is the number anyone actually quotes —
#: so on the denoising half the one number a reader reaches for was the one with no split beside it.
#: `bench`'s `macro_denoise` carries all three.
DEN_MACROS_EVAL_PY_NEVER_COMPUTED = {
    "M1.den_macro_crps_oracle_scaled",
    "M1.den_macro_scale_error",
    "M1.den_macro_marg_crps",
}


def test_no_rule_is_dead_against_the_real_key_set(tool, skeleton) -> None:
    """A rule that claims nothing describes a key `eval.py` never emitted.

    Three do, and they are named above rather than deleted: they are the derived denoise twins of
    the imputation macro oracle-scale split, and their emptiness is the record that the old suite
    never macro'd it on the denoising half — the level at which the number is quoted.
    """
    claimed, _ = tool.cover(skeleton)
    fired = Counter(r.old for r in claimed.values())
    dead = {r.old for r in tool.RULES if not fired[r.old]}
    assert dead == DEN_MACROS_EVAL_PY_NEVER_COMPUTED, (
        f"unexpected dead rule(s) {sorted(dead - DEN_MACROS_EVAL_PY_NEVER_COMPUTED)}; "
        f"unexpectedly live {sorted(DEN_MACROS_EVAL_PY_NEVER_COMPUTED - dead)}")


def test_the_two_accepted_losses_stay_in_the_report(tool, skeleton) -> None:
    """A loss that was decided must keep reading differently from a loss nobody noticed.

    Both were put to the PI and accepted on 2026-08-21: the target-clustered bootstrap CIs, and the
    clamp telemetry that distinguishes "the model ignores depth" from "log2_mu saturated". Accepting
    them is not a reason to stop printing them — the report is the record that the trade was made
    deliberately, and a future reader who finds a `depthdir` near zero needs to know it means "no response"
    rather than "no sensitivity".
    """
    text = tool.report(skeleton, {})
    assert "Accepted losses" in text and "ruled on" in text
    assert "d_crps_clustered" in text and "clamp" in text
    assert "candi.stats.cluster_bootstrap_ci" in text, (
        "the report must say the statistic survived the cutover, or a reader will think it died "
        "with eval.py")


def test_the_headline_never_shows_a_crps_without_its_split(tool) -> None:
    """`AGENTS.md` §7.2 — raw CRPS is never quoted alone, and the headline is where it would be.

    Checked structurally rather than by eye: every headline row whose bench key ends in `.crps` must
    be followed by that same block's `crps_oracle_scaled` and `scale_error`.
    """
    rows = list(tool.HEADLINE)
    for i, row in enumerate(rows):
        new = row[2]                    # rows may carry a 4th element: a not-like-for-like note
        if not new.endswith(".crps"):
            continue
        block = new[: -len("crps")]
        following = {r[2] for r in rows[i + 1: i + 4]}
        assert f"{block}crps_oracle_scaled" in following, new
        assert f"{block}scale_error" in following, new


def test_the_noise_floor_is_quoted_with_the_numbers(tool, skeleton) -> None:
    """§7.2 again: the floor travels with every number, so it must be in the rendered text."""
    text = tool.report(skeleton, {})
    assert "0.09" in text and "0.1195" in text
    assert "12 held-out targets" in text
    assert "same checkpoint" in text.lower(), (
        "a reader must be told both columns are one checkpoint, or they will read a measurement "
        "difference as the model changing")


def test_the_fixture_carries_eval_pys_finest_level_and_not_only_its_coarser_ones(skeleton) -> None:
    """The staleness that already happened once, pinned so it cannot happen twice.

    The first fixture was recorded from a run made before this repo existed, whose `eval.py` had
    no `imp_per_target` / `den_per_target`. Everything downstream looked healthy: 651 keys, every
    one claimed, the gate green. 760 real keys -- 54% of the output, and the ONLY level that is a
    like-for-like counterpart of bench's `per_track` -- were simply not in the fixture to be
    unclaimed. A gate that can pass on a subset of the thing it gates is not a gate.
    """
    assert skeleton["M1"]["imp_per_target"], "the fixture predates eval.py's per-target level"
    assert skeleton["M1"]["den_per_target"], "the fixture predates eval.py's per-target level"
    a_track = next(iter(skeleton["M1"]["imp_per_target"]))
    assert a_track.count("|") == 2, (
        f"{a_track!r} is not an `input|target|assay` track key; bench's `per_track` uses that exact "
        f"string, which is what makes the two comparable track by track")


def test_the_per_assay_level_is_claimed_as_a_lost_level_not_as_a_lost_number(tool) -> None:
    """`imp_per_assay` has no bench counterpart, and the report must say WHICH kind of loss it is.

    The numbers under it are not gone -- every one is in `per_track` under a key naming the cell.
    What is gone is the pooling step. A reader who sees `dropped` without that distinction will go
    looking for a measurement that was never removed.
    """
    r = tool.match("M1.imp_per_assay.H3K4me3.crps")
    assert r is not None and r.verdict == "dropped"
    assert "LEVEL IS GONE" in r.why and "per_track" in r.why
    assert tool.match("M1.den_per_assay.H3K4me3.crps").verdict == "dropped"
    # ... while the per-TARGET form of the same key does have a counterpart
    t = tool.match("M1.imp_per_target.T_a|B_a|H3K4me3.crps")
    assert t is not None and t.verdict == "moved" and t.new == "per_track.*.count.crps"
