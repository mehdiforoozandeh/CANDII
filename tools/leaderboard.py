"""The rivals leaderboard compiler (plan/BENCHMARK_DESIGN.md, t58 → t82). Three verbs:

    python tools/leaderboard.py add <score.json> --board eic.19 --method Avocado \\
        --truth store --panel V_breadth --scope held-out \\
        --version 2026-09-05 --date 2026-09-05 --lineage rival \\
        --position-class transductive --cell-class retrained \\
        --scoring-sha <sha> --store-manifest-hash <hash>
    python tools/leaderboard.py build
    python tools/leaderboard.py check

`add` stamps one `candi.bench.external`-shaped score json into one row file under
`leaderboard/rows/<regime>/<truth>.<panel>.<scope>/<method>@<version>.json`. It computes nothing:
every number in a row is copied from the block the row's address names, gated on the way in — NaN
refused, provenance mandatory, the board's frozen eval-set hash enforced, and the companion rules
of the registry (count `crps` never without `crps_oracle_scaled` + `scale_error`; pval `crps`
never without `pit_ks` + `coverage_95`) applied as refusals rather than as conventions.

BENCHMARK_DESIGN §1 — a cell is addressed by method · regime · truth · panel · scope · metric, and
"if any field is unknown for a row, the row does not go in the ranked table". Here that is
structural rather than documented: the first five fields are the row's own path, `add` has no
default for any of them, and a row whose address does not resolve has nowhere on disk to live.

**The address is a lookup, never an aggregation.** `--scope` picks which aggregation of the one
scoring pass to read (the top level, or the `genome_wide` block `--held-out-chroms` produces) and
`--panel` picks one of the three numbers `harness.panel_macros` wrote inside it (§5.2). `--truth`
must agree with `provenance.truth.source`, so a store pass can never be relabelled as a challenge
one, and a challenge row carries the sha256 of the truth root's manifest.

Three kinds of row file, three directories:

  `leaderboard/rows/`   the ranked table — one directory per regime, one per address inside it
  `leaderboard/anchor/` the 2019 field (§6). No regime, because we never trained them, so these
                        rows are lifted out of the ranked table into a block underneath it
  `leaderboard/void/`   rows §3.3 retired. Read, shape-gated, and compiled **without their
                        numbers** — a void number under a new label would read as freshly
                        computed

`build` compiles registry + boards + rows into `_site/leaderboard.json` and copies the static site
beside it. It is a pure function of the repo tree: sorted keys, no timestamps, bit-identical reruns.

`check` re-runs every gate on the existing rows, rebuilds twice, and diffs the two builds bit-exact.
CI runs it before every deploy.

The composite (PRD §5.4): per in-composite metric, rows whose gap is under the metric's noise floor
are tied and share a rank interval; category sub-score = mean rank interval over the category's
ranked metrics; composite = unweighted mean of category intervals; the displayed rank is the spread
of best and worst achievable rank. A category is board-active when at least one row fully covers it
(every ranked metric present). A row missing any board-active composite category is incomplete, not
last: it gets no composite and is excluded from the headline ranking, and still ranks inside every
category it has numbers for (PI ruling 2026-08-27). The count arm ranks on its own sub-board.

Stdlib only, by rule.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = REPO / "leaderboard"
DEFAULT_OUT = REPO / "_site"

SLUG = re.compile(r"^[A-Za-z0-9._-]+$")
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TODO_HASH = "TODO-"

#: the container id of the anchor block (§6). It is not a regime and never ranks beside one.
ANCHOR = "anchor"

#: the address the site opens on: the ranked number of §4 and §5.2 under store truth.
CANONICAL_VIEW = "store.V_breadth.held-out"

#: provenance keys `add` copies into the row's flags-of-record block when the score json has them.
FLAG_KEYS = ("crps_estimator", "crps_k", "crps_seed", "allow_missing", "msevar",
             "signal_target_transform", "pval_pred_space", "pred_inversion", "seed",
             "aggregation", "n_experiments", "contributor_mode", "clip")

LINEAGES = ("candi", "rival", "baseline", "entrant")
POSITION_CLASSES = {"transductive": "position-transductive", "generalizing": "position-generalizing",
                    "unrecorded": "position class unrecorded"}
CELL_CLASSES = {"zero-shot": "zero-shot cell types", "retrained": "retrained per setting",
                "unrecorded": "cell-type class unrecorded"}

#: The board's panel vocabulary → the key `harness.panel_macros` writes for it. The two spellings
#: of the test panel are deliberate and both are load-bearing: the score json says `B` (the panel
#: letter `harness.panel_of` returns) and the board says `B_` (the biosample prefix a reader sees
#: in `B_bb`), so the map is the one place the two meet.
PANEL_JSON_KEY = {"V_breadth": "V_breadth", "V_matched": "V_matched", "B_": "B"}

#: the `scopes` value whose numbers live in the score json's `genome_wide` block
SCOPE_GENOME_WIDE = "genome-wide"

#: §5.2's middle number, and the one panel a single scoring pass can fail to populate.
#: `harness.panel_macros` MEASURES the matched assay set from the `B_` rows of the pass it is given
#: (`matched_to`), so a pass that scored only `V_` targets writes a `V_matched` block with an empty
#: `matched_to` and no experiments in it. That is not an 8-assay number of zero — it is no number,
#: and `add` refuses it rather than printing an empty aggregation under a panel heading. The fix is
#: `python -m candi.bench.external fill-panels --v <store.V_.json> --b <store.B_.json>`, which
#: re-measures `panels` over the union of the two passes' `per_track` and records `panels_from`.
PANEL_MATCHED = "V_matched"
PANEL_BREADTH = "V_breadth"
FILL_PANELS_CMD = "python -m candi.bench.external fill-panels --v <store.V_.json> --b <store.B_.json>"

#: §4's blanking rule, as method slugs. A method whose transferable parameters were fit at every
#: position has no genome-wide cell: the number would be a memorisation score, and "a blanked cell
#: is not computed". `add` refuses one rather than letting a badge carry it. The prose these three
#: names come from is `boards.json` `scopes["genome-wide"]["blanking_rule"]`, which the refusal
#: quotes, and `tests/test_leaderboard.py` holds the two in step.
GENOME_WIDE_BLANKED = ("Avocado", "ChromImpute", "Lavawizard")

#: §7's mandatory spread badge. A point-only method's prediction is wrapped in a Gaussian whose σ
#: is FITTED and flat per assay; a method that emitted its own per-bin spread is native. Which one
#: a row is, is a fact about the score json's `provenance.sigma_table`, never a judgement.
SIGMA_BADGE_NATIVE = "native heteroscedastic"
SIGMA_BADGE_FITTED = "fitted flat σ"

#: §7 — σ is fit on training-set residuals ONLY, never on `V_`, never on `B_`.
#: `competitors/sigma_pass.py` writes this prefix into `fitted_on`; a table without it was fit on
#: eval positions, which is a Rule 1 leak, and a row built on one is refused rather than badged.
SIGMA_FITTED_ON_PREFIX = "training-residuals:"

#: §4 — where the per-cell in-sample fraction would be read from. NOTHING WRITES IT TODAY: no key
#: of `candi.bench`'s or `candi.bench.external`'s score json holds it (checked 2026-09-02 across
#: `harness.run_bench` and `score_external`). `add` copies it the moment a producer starts writing
#: `provenance.in_sample_fraction`, and records an explicit `null` until then — so the row says the
#: fraction is unknown instead of implying it by an absent key.
IN_SAMPLE_KEY = "in_sample_fraction"


class GateError(SystemExit):
    """A refused row or a failed check. Message first, exit code 1."""

    def __init__(self, msg: str) -> None:
        super().__init__(f"leaderboard: {msg}")


def _refuse_nan(token: str) -> float:
    raise GateError(f"score json carries a non-finite literal ({token}); NaN never enters a row")


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=_refuse_nan)
    except FileNotFoundError:
        raise GateError(f"{path} does not exist")
    except json.JSONDecodeError as e:
        raise GateError(f"{path} is not valid JSON: {e}")


def dump_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, indent=1, ensure_ascii=False, allow_nan=False) + "\n"


# ---------------------------------------------------------------- registry ---

def load_registry(root: Path) -> Dict[str, Any]:
    reg = load_json(root / "registry.json")
    for field in ("metrics", "categories", "candi_self_comparison_bar"):
        if field not in reg:
            raise GateError(f"registry.json is missing `{field}`")
    for m in reg["metrics"]:
        for field in ("key", "arm", "category", "direction", "role", "floor", "decimals"):
            if field not in m:
                raise GateError(f"registry metric {m.get('key')} is missing `{field}`")
        if m["category"] not in reg["categories"]:
            raise GateError(f"registry metric {m['key']} names unknown category {m['category']}")
        if m["role"] == "ranked" and m["direction"] not in ("higher", "lower"):
            raise GateError(f"ranked metric {m['arm']}/{m['key']} needs a direction")
    return reg


def load_boards(root: Path) -> Dict[str, Any]:
    """boards.json — the regimes, plus the vocabularies §1 addresses a row by."""
    boards = load_json(root / "boards.json")
    for field in ("boards", "truths", "panels", "scopes", "markers", "anchor", "void",
                  "address"):
        if field not in boards:
            raise GateError(f"boards.json is missing `{field}`")
    for bid, b in boards["boards"].items():
        for field in ("label", "regime_id", "eval_set", "frozen", "caveats", "truths",
                      "noise_floor"):
            if field not in b:
                raise GateError(f"board `{bid}` is missing `{field}`")
        if "protocol" in b:
            raise GateError(f"board `{bid}` still carries `protocol` — P1/P2/P3 are retired "
                            "(BENCHMARK_DESIGN §9); the regime id and the `scope` field "
                            "replace them")
        for t in b["truths"]:
            if t not in boards["truths"]:
                raise GateError(f"board `{bid}` names unknown truth `{t}`")
        if "measured" not in b["noise_floor"]:
            raise GateError(f"board `{bid}` noise_floor block is missing `measured`")
    for bid, b in list(boards["boards"].items()) + [(ANCHOR, boards["anchor"])]:
        for field in ("store_manifest_hash", "regime_sha256"):
            if field not in b.get("frozen", {}):
                raise GateError(f"board `{bid}` frozen block is missing `{field}`")
    if ANCHOR in boards["boards"]:
        raise GateError("`anchor` is not a regime; it is the block underneath the ranked table")
    for method, marks in (boards.get("method_markers") or {}).items():
        for mk in marks:
            if mk not in boards["markers"]:
                raise GateError(f"method_markers[{method}] names unknown marker `{mk}`")
    return boards


def board_spec(boards: Mapping[str, Any], bid: str) -> Mapping[str, Any]:
    """The container `bid` names: a regime, or the anchor block."""
    if bid == ANCHOR:
        return boards["anchor"]
    if bid not in boards["boards"]:
        raise GateError(f"unknown board `{bid}`; boards.json has "
                        f"{sorted(boards['boards'])} plus `{ANCHOR}`")
    return boards["boards"][bid]


def view_key(truth: str, panel: str, scope: str) -> str:
    return f"{truth}.{panel}.{scope}"


def container_truths(spec: Mapping[str, Any], bid: str) -> List[str]:
    """Which truths a container offers.

    A regime lists `truths`. The anchor block names one `truth` — the 2019 measurement the field
    was ranked under — and may list `truths` instead when the same rescore is also read against
    store truth. Read rather than assumed, so a store-truth anchor row is refused while the
    committed anchor offers challenge truth alone, and lands the moment it offers both.
    """
    if bid != ANCHOR:
        return list(spec["truths"])
    listed = spec.get("truths")
    return list(listed) if listed else [spec["truth"]]


def view_keys(boards: Mapping[str, Any], bid: str) -> List[str]:
    """Every address a container offers — one view per (truth, panel, scope).

    Derived, never listed by hand: the site cannot show a number until the reader has picked
    all three, which is what makes §1's rule structural instead of documented.
    """
    spec = board_spec(boards, bid)
    truths = container_truths(spec, bid)
    panels = [spec["panel"]] if bid == ANCHOR else list(boards["panels"])
    scopes = [spec["scope"]] if bid == ANCHOR else list(boards["scopes"])
    return [view_key(t, p, s) for t in truths for p in panels for s in scopes]


def view_is_ranked(boards: Mapping[str, Any], bid: str, view: str) -> Tuple[bool, str]:
    """Does this address rank, and if not, why not?

    Three separate reasons, all from the design and none of them an error state:
    the anchor block carries no regime (§6), `V_ matched` and `genome-wide` are reported but
    never ranked (§5.2, §4), and no regime ranks at all until its noise floor is measured (§15).
    """
    _truth, panel, scope = view.split(".", 2)
    if bid == ANCHOR:
        return False, ("An anchor we did not run. These rows carry no regime, so they never "
                       "share a ranking denominator with ours (BENCHMARK_DESIGN §6).")
    if not boards["panels"][panel].get("ranked"):
        return False, boards["panels"][panel]["eli5"]
    if not boards["scopes"][scope].get("ranked"):
        return False, boards["scopes"][scope]["eli5"]
    floor = board_spec(boards, bid)["noise_floor"]
    if not floor.get("measured"):
        return False, floor.get("note", "the noise floor on this panel is not measured yet")
    return True, ""


def metric_slot(metric: Mapping[str, Any]) -> str:
    """Which sub-dict of row['metrics'] a registry metric lives in."""
    return metric["arm"] if metric["arm"] in ("pval", "count") else "diagnostics"


def metric_id(metric: Mapping[str, Any]) -> str:
    return f"{metric_slot(metric)}/{metric['key']}"


# ---------------------------------------------------------------- add ---

#: Registry diagnostic keys. `candi.bench.harness.c_block` writes nested dicts under these
#: names (and under the combined `depthblind_biokeep`); the stamper flattens to scalars.
DIAGNOSTIC_KEYS: Tuple[str, ...] = (
    "covuse", "covshare", "depthdir", "depthcounterfact", "covspec", "depthblind", "biokeep")

#: Nested C-block instrument → the field the instrument itself names as the headline.
#: `covariate.depthdir`: "`monotone_frac` … is the number to read".
#: `covariate.depthcounterfact` / EVAL.md §C: the quoted number is `frac_min_at_true`.
#: `covariate.covspec` already averages the per-aspect gaps into `mean_gap`.
#: Keys not in this map have no code-defined panel scalar (see flatten_c_block).
_C_HEADLINES: Dict[str, str] = {
    "depthdir": "monotone_frac",
    "depthcounterfact": "frac_min_at_true",
    "covspec": "mean_gap",
}


def _finite_scalar(val: Any) -> Optional[float]:
    """A real number, not a bool (bool is an int) and not NaN/Inf."""
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return None
    out = float(val)
    return out if math.isfinite(out) else None


def flatten_c_block(cblock: Mapping[str, Any]) -> Dict[str, float]:
    """Lift `harness.c_block`'s nested C json onto the seven registry diagnostic keys.

    Scalars already scalar pass through. Nested instruments take the headline the
    instrument names (`_C_HEADLINES`). `depthblind_biokeep` splits: `biokeep` is
    `covariate.biokeep`'s `bio_silhouette`; `depthblind` has no single headline and
    is omitted. `covuse` and `covshare` are per-covariate and have no code-defined
    panel aggregate, so a nested form is omitted rather than averaged.
    """
    out: Dict[str, float] = {}
    for key in DIAGNOSTIC_KEYS:
        raw = cblock.get(key)
        got = _finite_scalar(raw)
        if got is not None:
            out[key] = got
            continue
        headline = _C_HEADLINES.get(key)
        if headline and isinstance(raw, dict):
            got = _finite_scalar(raw.get(headline))
            if got is not None:
                out[key] = got
    combo = cblock.get("depthblind_biokeep")
    if isinstance(combo, dict) and "biokeep" not in out:
        got = _finite_scalar(combo.get("bio_silhouette"))
        if got is not None:
            out["biokeep"] = got
    return out


def address_macro(score: Mapping[str, Any], panel: str, scope: str) -> Dict[str, Any]:
    """The `macro`-shaped block ONE ADDRESS names (§1, §4, §5.2). A lookup, never an aggregation.

    The scope picks which aggregation of the single scoring pass to read — the top level, or the
    `genome_wide` block a pass given `--held-out-chroms` also wrote — and the panel picks one of
    the three numbers `harness.panel_macros` put inside it.

    A score json with no `panels` block is refused rather than read as though its `macro` were one
    panel: `macro` pools every scored track, so stamping it under `V_matched` or `B_` would print a
    22-assay number beneath an 8-assay label, which is the single mislabelling this board exists to
    prevent. Re-score it — `panels` costs no extra inference.
    """
    if panel not in PANEL_JSON_KEY:
        raise GateError(f"panel `{panel}` names no block in a score json; the addressable panels "
                        f"are {sorted(PANEL_JSON_KEY)}")
    block: Any = score
    if scope == SCOPE_GENOME_WIDE:
        block = score.get("genome_wide")
        if not isinstance(block, dict):
            raise GateError(
                "score json carries no `genome_wide` block, so it holds no genome-wide number to "
                "stamp. One pass yields two aggregations only when the scorer was given "
                "--held-out-chroms; without it §4's blanking rule means the genome-wide number was "
                "NOT COMPUTED, and `add` never computes one")
    panels = block.get("panels")
    if not isinstance(panels, dict):
        raise GateError(
            f"score json's {scope} block has no `panels` — it predates §5.2's three numbers. Its "
            "`macro` pools every scored track and is not any one panel, so there is nothing here "
            f"to stamp under `{panel}`. Re-score with a scorer that writes `panels`")
    key = PANEL_JSON_KEY[panel]
    macro: Dict[str, Any] = {}
    for arm in ("pval", "count"):
        arm_panels = panels.get(arm)
        if not isinstance(arm_panels, dict):
            raise GateError(f"score json's panels block has no `{arm}` arm")
        got = arm_panels.get(key)
        if not isinstance(got, dict):
            raise GateError(f"score json's panels[{arm}] has no `{key}` block; it carries "
                            f"{sorted(arm_panels)}")
        if panel == PANEL_MATCHED:
            refuse_an_unfilled_matched_panel(arm_panels, arm, scope)
        macro[arm] = got
    return macro


def refuse_an_unfilled_matched_panel(arm_panels: Mapping[str, Any], arm: str, scope: str) -> None:
    """`V_matched` from a pass that scored no `B_` targets is no number at all. See `PANEL_MATCHED`.

    The check is per arm and conditional on the same arm's `V_breadth`, because an arm the pass
    could not score at all (the count arm under challenge truth: no counts in the 2019 data) has an
    empty `V_breadth` too, and that is an absence the address already handles. What is refused is
    the one case where `V_` rows WERE scored and the matched subset came out empty anyway — which
    can only mean the `B_` rows the subset is measured from were in a different pass.
    """
    breadth = arm_panels.get(PANEL_BREADTH)
    if not isinstance(breadth, dict) or not int(breadth.get("n_experiments") or 0):
        return
    matched = arm_panels.get(PANEL_MATCHED) or {}
    if int(matched.get("n_experiments") or 0) and list(matched.get("matched_to") or []):
        return
    raise GateError(
        f"score json's {scope} panels[{arm}][{PANEL_MATCHED}] has no scored experiments "
        f"(matched_to={list(matched.get('matched_to') or [])!r}, "
        f"n_experiments={matched.get('n_experiments')!r}) while V_breadth scored "
        f"{breadth.get('n_experiments')}. The matched assay set is MEASURED from the pass's own "
        f"`B_` rows, and this pass had none, so there is no 8-assay number here to stamp — an "
        f"empty aggregation under a panel heading is exactly the mislabelling §5.2 exists to stop. "
        f"Fill it from the sibling `B_` pass first: {FILL_PANELS_CMD}")


def extract_metrics(score: Mapping[str, Any], registry: Mapping[str, Any],
                    allow_missing: bool, *, panel: str,
                    scope: str) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    """Copy registry metrics out of a bench-shaped score json, at one address.

    Returns (metrics, missing ids). Diagnostics come from the top-level `C` block on both scopes:
    `C` re-decodes perturbed prompts over the source's windows, so it is a property of the run and
    not of either aggregation — neither `harness.run_bench` nor `score_external` writes a
    genome-wide counterpart to divide it into.
    """
    if not isinstance(score.get("macro"), dict):
        raise GateError("score json has no `macro` block — not a candi.bench-shaped file")
    macro = address_macro(score, panel, scope)
    cblock = score.get("C") if isinstance(score.get("C"), dict) else {}
    diagnostics = flatten_c_block(cblock)
    out: Dict[str, Dict[str, float]] = {}
    missing: List[str] = []
    for m in registry["metrics"]:
        slot = metric_slot(m)
        if slot == "diagnostics":
            src = diagnostics
        else:
            src = macro.get(m["arm"], {}) if isinstance(macro.get(m["arm"]), dict) else {}
        val = src.get(m["key"])
        if val is None:
            # diagnostics are CANDI-lineage instruments; rivals lack them by construction and
            # their absence never needs --allow-missing.
            if slot != "diagnostics":
                missing.append(metric_id(m))
            continue
        if not isinstance(val, (int, float)) or isinstance(val, bool) or not math.isfinite(val):
            raise GateError(f"metric {metric_id(m)} is not a finite number: {val!r}")
        out.setdefault(slot, {})[m["key"]] = float(val)
    if missing and not allow_missing:
        raise GateError("score json lacks registry metrics "
                        f"{', '.join(missing)}; re-run with --allow-missing to record the gap")
    check_companions(out, registry)
    return out, sorted(missing)


def check_companions(metrics: Mapping[str, Mapping[str, float]],
                     registry: Mapping[str, Any]) -> None:
    """A metric never enters without the companions the registry declares for it."""
    for m in registry["metrics"]:
        comps = m.get("companions") or []
        if not comps:
            continue
        slot = metric_slot(m)
        have = metrics.get(slot, {})
        if m["key"] in have:
            lacked = [c for c in comps if c not in have]
            if lacked:
                raise GateError(f"{metric_id(m)} present without its companion(s) "
                                f"{', '.join(lacked)} — refused (registry companion rule)")


# The t54 `--placement-method` path is RETIRED (§6): anchor rows are per-entrant score jsons now.


def truth_manifest_hash(score: Mapping[str, Any], truth: str) -> Optional[str]:
    """Check the score json's own truth against the row's address, and pin the challenge one.

    `candi.bench.external` writes `provenance.truth` on BOTH paths, so an absent block can only be
    an older file, which had no `--truth-root` to read and therefore read the store. Returns the
    truth root's `manifest_sha256` for a challenge row and `None` for a store row: under challenge
    truth the store manifest pins the grid but not what the numbers were compared against, and a
    row that cannot say which build of the 2019 truth it read is not reproducible (§6).
    """
    prov = score.get("provenance") if isinstance(score.get("provenance"), dict) else {}
    block = prov.get("truth") if isinstance(prov.get("truth"), dict) else None
    source = "store" if block is None else block.get("source")
    if truth == "challenge":
        if source != "challenge":
            raise GateError(
                f"--truth challenge, but this score json was measured against `{source}` truth "
                "(provenance.truth.source). A challenge row comes from a pass run with "
                "--truth-root; relabelling a store pass would put two truths on one page (§6)")
        sha = (block or {}).get("manifest_sha256")
        if not sha:
            raise GateError("challenge-truth score json names no provenance.truth.manifest_sha256 "
                            "— without it the row cannot say WHICH build of the 2019 truth it was "
                            "measured against")
        return str(sha)
    if source != "store":
        raise GateError(
            f"--truth store, but this score json was measured against `{source}` truth "
            "(provenance.truth.source). Address it as that truth, or score it against the store")
    return None


def sigma_badge(score: Mapping[str, Any]) -> str:
    """§7's mandatory spread badge, and the Rule 1 refusal that comes with it.

    No σ table in the score's provenance means the method emitted its own per-bin spread. A table
    means a flat per-assay Gaussian was wrapped around a point prediction — which is fine, and
    disclosed — but only if it was fitted on TRAINING residuals. A table fitted on `V_` or `B_`
    read the panel it is scored on, so the row is refused instead of badged.
    """
    prov = score.get("provenance") if isinstance(score.get("provenance"), dict) else {}
    sigma = prov.get("sigma_table")
    if not isinstance(sigma, dict):
        return SIGMA_BADGE_NATIVE
    fitted_on = str(sigma.get("fitted_on") or "")
    if not fitted_on.startswith(SIGMA_FITTED_ON_PREFIX):
        raise GateError(
            f"σ table `fitted_on` is {fitted_on!r}, which does not start with "
            f"{SIGMA_FITTED_ON_PREFIX!r}. §7 fits σ on training-set residuals only — never on "
            "`V_`, never on `B_` — so this table read the panel the row is scored on and the row "
            "is refused rather than badged")
    return SIGMA_BADGE_FITTED


def in_sample_fraction(score: Mapping[str, Any]) -> Optional[float]:
    """§4's per-cell in-sample fraction, copied when a producer writes one. See `IN_SAMPLE_KEY`."""
    prov = score.get("provenance") if isinstance(score.get("provenance"), dict) else {}
    val = prov.get(IN_SAMPLE_KEY)
    if val is None:
        return None
    got = _finite_scalar(val)
    if got is None:
        raise GateError(f"provenance.{IN_SAMPLE_KEY} is not a finite number: {val!r}")
    return got


def address_ranks(boards: Mapping[str, Any], bid: str, panel: str, scope: str) -> bool:
    """Does the row's own address rank at all (§4, §5.2, §6)?

    The address alone: `V_matched` and `genome-wide` are reported and never ranked, and the anchor
    block shares no ranking denominator with a regime. The board's noise floor is a property of the
    board rather than of the row, so §15's "nothing ranks until the floor lands" stays where it
    can be lifted by one edit — `view_is_ranked`, at compile time.
    """
    if bid == ANCHOR:
        return False
    return bool(boards["panels"][panel].get("ranked")) and bool(boards["scopes"][scope].get("ranked"))


def method_markers(boards: Mapping[str, Any], bid: str) -> Mapping[str, Any]:
    """The markers a container's rows carry, by method.

    Every marker in the registry says what a method did while we trained it — an embedding
    substitution, no checkpoint to select, a selection key that is not CANDI's. None of that is
    true of an anchor row: those are 2019 submissions we never trained, and a shared slug
    (`Lavawizard` is both a retrained rival and a frozen entrant) must not carry our marker into
    the anchor block. So the anchor block takes no method markers at all.
    """
    if bid == ANCHOR:
        return {}
    return boards.get("method_markers") or {}


def refuse_a_blanked_cell(boards: Mapping[str, Any], method: str, scope: str) -> None:
    """§4 — a blanked genome-wide cell is not computed, so it is not stamped either."""
    if scope != SCOPE_GENOME_WIDE or method not in GENOME_WIDE_BLANKED:
        return
    rule = boards["scopes"][SCOPE_GENOME_WIDE].get("blanking_rule", "")
    raise GateError(f"`{method}` has no genome-wide cell on this board. {rule}")


def sha256_file(path: Path) -> str:
    """What a board's `frozen` hashes digest: a regime json, or the truth root's manifest.json."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_has_peak_head(score: Mapping[str, Any]) -> bool:
    """True when the score's macro carries `bernoulli_nll` — the stamp `loss_block` emits
    only if the producer supplied a real `peak_score` (`TrackRecord.has_peak_head`)."""
    macro = score.get("macro") if isinstance(score.get("macro"), dict) else {}
    for arm in ("pval", "count"):
        block = macro.get(arm)
        if isinstance(block, dict) and "bernoulli_nll" in block:
            return True
    return False


def score_path_of_record(path: Path) -> str:
    """Store vault evidence paths repo-relative, so any checkout resolves them."""
    s = str(path.resolve())
    marker = "/cruxvault/results/"
    return "cruxvault/results/" + s.split(marker, 1)[1] if marker in s else str(path)


def build_provenance(score: Mapping[str, Any], args: argparse.Namespace,
                     score_path: Path) -> Dict[str, Any]:
    prov = score.get("provenance")
    if not isinstance(prov, dict):
        raise GateError("score json has no `provenance` block")
    if "suite" not in prov:
        raise GateError("score provenance names no `suite` (scorer)")
    fir_path = args.fir_path
    if fir_path is None:
        sibling = score_path.parent / "FIR_PATH.txt"
        if sibling.exists():
            fir_path = sibling.read_text(encoding="utf-8").strip().splitlines()[0]
    if not fir_path:
        raise GateError("no FIR path: pass --fir-path or keep FIR_PATH.txt beside the score json")
    sigma = prov.get("sigma_table")
    sigma_id = None
    if isinstance(sigma, dict):
        sigma_id = {"method": sigma.get("method"), "fitted_on": sigma.get("fitted_on")}
        if not sigma_id["method"] or not sigma_id["fitted_on"]:
            raise GateError("score provenance sigma_table lacks method/fitted_on — no σ-table id")
    flags = {k: prov[k] for k in FLAG_KEYS if k in prov}
    out = {
        "scoring_sha": args.scoring_sha,
        "score_json": score_path_of_record(score_path),
        "fir_path": fir_path,
        "store_manifest_hash": args.store_manifest_hash,
        "regime": prov.get("regime"),
        "sigma_table": sigma_id,
        "has_peak_head": score_has_peak_head(score),
        "scorer": prov["suite"],
        "flags": flags,
        "artifacts_resolved_at_add": score_path.exists(),
    }
    # §5.2 — a `panels` block re-measured from a sibling pass says so, and the row says so too:
    # `V_matched` on such a row was measured from `B_` rows the SAME method scored in a different
    # pass, which a reader checking the number back to one score json needs to be told. Copied
    # verbatim (whatever `fill-panels` recorded), never summarised.
    if prov.get("panels_from") is not None:
        out["panels_from"] = prov["panels_from"]
    # method-level caveats recorded by the producer travel on the row (e.g. eDICE's
    # transductive + masking-rate caveats in its prediction manifest)
    manifest = prov.get("manifest") if isinstance(prov.get("manifest"), dict) else {}
    notes = {k: manifest[k] for k in ("notes", "caveat", "masking_caveat") if manifest.get(k)}
    if notes:
        out["method_notes"] = notes
    return out


def gate_address(boards: Mapping[str, Any], bid: str, truth: str, panel: str, scope: str) -> str:
    """§1's three reader-picked fields, against boards.json's own vocabulary. Returns the view key.

    Checked BEFORE any number is read, so an unknown truth, panel or scope is refused as the typo
    it is rather than as a missing block in the score json. Shared with `gate_row_address` so the
    stamper and the reader cannot drift apart on what an address is.
    """
    for field, value, vocab in (("truth", truth, "truths"), ("panel", panel, "panels"),
                                ("scope", scope, "scopes")):
        if value not in boards[vocab]:
            raise GateError(f"unknown {field} `{value}`; boards.json has "
                            f"{sorted(boards[vocab])}")
    offered = container_truths(board_spec(boards, bid), bid)
    if truth not in offered:
        raise GateError(f"truth `{truth}` is not offered by `{bid}` (it offers {sorted(offered)})")
    if bid == ANCHOR:
        # The anchor block sits at ONE address (§6): the 2019 field's own panel and scope. A row
        # anywhere else would rank beside our rows by living in the same directory tree.
        for field, value in (("panel", panel), ("scope", scope)):
            if value != boards["anchor"][field]:
                raise GateError(f"an anchor row is addressed {field} `{value}`, but the anchor "
                                f"block is `{boards['anchor'][field]}` only (§6)")
    return view_key(truth, panel, scope)


def gate_row_address(row: Mapping[str, Any], boards: Mapping[str, Any], bid: str,
                     registry: Mapping[str, Any]) -> str:
    """BENCHMARK_DESIGN §1 — resolve the row's address, or refuse the row.

    Returns the view key. Every field is checked against boards.json's own vocabulary, so a
    typo cannot invent a truth, a panel or a scope.
    """
    addr = row.get("address")
    if not isinstance(addr, dict):
        raise GateError(f"row {row.get('method')}@{row.get('version')} has no `address` block; "
                        "§1 gives no row a home without one")
    for field in ("regime", "truth", "panel", "scope"):
        if field not in addr:
            raise GateError(f"row {row['method']}@{row['version']} address lacks `{field}` — "
                            "§1: if any field is unknown, the row does not go in the table")
    if bid == ANCHOR:
        if addr["regime"] is not None:
            raise GateError(f"anchor row {row['method']}@{row['version']} names regime "
                            f"`{addr['regime']}` — we never trained these, so they carry none")
    else:
        want = boards["boards"][bid]["regime_id"]
        if addr["regime"] != want:
            raise GateError(f"row {row['method']}@{row['version']} says regime "
                            f"`{addr['regime']}` but lives under `{bid}` (`{want}`)")
    gate_address(boards, bid, addr["truth"], addr["panel"], addr["scope"])
    # §7 — the 2019 data has no counts and no peak calls, so those arms cannot be scored
    # against it. A row that carries them under challenge truth has two truths in it.
    if addr["truth"] == "challenge":
        allowed = set(boards["truths"]["challenge"]["arms"])
        for m in registry["metrics"]:
            block = row["metrics"].get(metric_slot(m), {})
            if m["key"] not in block:
                continue
            arm = "peak" if m["category"] == "peaks" else m["arm"]
            if arm not in allowed:
                raise GateError(
                    f"row {row['method']}@{row['version']} carries {arm}-arm metric "
                    f"`{m['key']}` under challenge truth — the 2019 data has no counts and no "
                    "peak calls, and two truths never share a row (§7)")
    for mk in row.get("markers", []):
        if mk not in boards["markers"]:
            raise GateError(f"row {row['method']}@{row['version']} carries unknown marker "
                            f"`{mk}`; boards.json has {sorted(boards['markers'])}")
    return view_key(addr["truth"], addr["panel"], addr["scope"])


def gate_row_against_board(row: Mapping[str, Any], board: Mapping[str, Any], bid: str) -> None:
    frozen = board["frozen"]
    for field in ("store_manifest_hash", "regime_sha256"):
        if str(frozen[field]).startswith(TODO_HASH):
            raise GateError(f"board `{bid}` eval-set hashes are not frozen yet; "
                            "stamp boards.json `frozen` before adding rows")
    prov = row["provenance"]
    if prov["store_manifest_hash"] != frozen["store_manifest_hash"]:
        raise GateError(f"row store manifest hash {prov['store_manifest_hash']} does not match "
                        f"board `{bid}` frozen hash {frozen['store_manifest_hash']} — "
                        "nobody quietly scores on easier data")
    if bid == ANCHOR and row["address"]["truth"] == "challenge":
        # The anchor block has no regime — we never trained these rows — so its second frozen hash
        # pins the TRUTH the rescore read instead: the sha256 of the truth root's manifest.json,
        # which every challenge score json carries as `provenance.truth.manifest_sha256`. A row
        # rescored against a different build of the 2019 truth is a different measurement.
        got = prov.get("truth_manifest_hash")
        if got != frozen["regime_sha256"]:
            raise GateError(
                f"anchor row truth manifest hash {got} does not match the anchor block's frozen "
                f"hash {frozen['regime_sha256']} — the anchor freezes the sha256 of the challenge "
                "truth root's manifest.json, because it has no regime to freeze")
    board_regime = board["eval_set"].get("regime", "")
    if board_regime.endswith(".json"):
        row_regime = prov.get("regime") or ""
        if Path(row_regime).name != Path(board_regime).name:
            raise GateError(f"row regime `{row_regime}` is not board `{bid}`'s "
                            f"regime `{board_regime}`")
    if row["metrics"].get("pval") and prov["flags"].get("pval_pred_space") != "-log10p":
        raise GateError("pval metrics extracted but provenance lacks pval_pred_space=-log10p — "
                        "the row predates the spaces contract and is not quotable (EVAL.md)")


def gate_row_shape(row: Mapping[str, Any], registry: Mapping[str, Any]) -> None:
    for field in ("board", "method", "version", "date", "lineage", "badges", "metrics",
                  "missing_metrics", "provenance"):  # `address` is gated separately
        if field not in row:
            raise GateError(f"row {row.get('method')}@{row.get('version')} is missing `{field}`")
    if not SLUG.match(row["method"]) or not SLUG.match(row["version"]):
        raise GateError(f"row method/version must match {SLUG.pattern}")
    if not DATE.match(row["date"]):
        raise GateError(f"row date `{row['date']}` is not YYYY-MM-DD")
    if row["lineage"] not in LINEAGES:
        raise GateError(f"row lineage `{row['lineage']}` is not one of {LINEAGES}")
    prov = row["provenance"]
    for field in ("scoring_sha", "score_json", "fir_path", "store_manifest_hash",
                  "regime", "scorer", "flags"):
        if not prov.get(field) and prov.get(field) != {}:
            raise GateError(f"row {row['method']}@{row['version']} provenance lacks `{field}` — "
                            "no provenance, no row (PRD §7.2)")
    metrics = row["metrics"]
    for slot, block in metrics.items():
        for key, val in block.items():
            if not isinstance(val, (int, float)) or isinstance(val, bool) \
                    or not math.isfinite(val):
                raise GateError(f"row {row['method']}@{row['version']} metric {slot}/{key} "
                                f"is not finite: {val!r}")
    check_companions(metrics, registry)


def cmd_add(args: argparse.Namespace) -> int:
    root = Path(args.root)
    registry = load_registry(root)
    boards = load_boards(root)
    board = board_spec(boards, args.board)
    score_path = Path(args.score)
    # §1 first: the address is resolved against boards.json before a single number is read, so a
    # field that does not exist is refused as a typo and not as a hole in the score json.
    gate_address(boards, args.board, args.truth, args.panel, args.scope)
    refuse_a_blanked_cell(boards, args.method, args.scope)
    score = load_json(score_path)
    truth_hash = truth_manifest_hash(score, args.truth)
    metrics, missing = extract_metrics(score, registry, args.allow_missing,
                                       panel=args.panel, scope=args.scope)
    provenance = build_provenance(score, args, score_path)
    if truth_hash is not None:
        provenance["truth_manifest_hash"] = truth_hash
    row: Dict[str, Any] = {
        "schema_version": 2,
        "board": args.board,
        "address": {
            # §1's four non-metric fields beyond the method. The anchor block carries no
            # regime, because we never trained those rows.
            "regime": None if args.board == ANCHOR else board["regime_id"],
            "truth": args.truth,
            "panel": args.panel,
            "scope": args.scope,
        },
        "markers": sorted(method_markers(boards, args.board).get(args.method, [])),
        "method": args.method,
        "version": args.version,
        "date": args.date,
        "lineage": args.lineage,
        "badges": {
            "position": POSITION_CLASSES[args.position_class],
            "cell_types": CELL_CLASSES[args.cell_class],
            # §7 — every distributional cell says which spread device produced it.
            "sigma": sigma_badge(score),
        },
        # §4, §5.2, §6 — whether this address ranks at all. The board's noise floor is applied on
        # top of it at compile time; this is the address's own answer.
        "ranked": address_ranks(boards, args.board, args.panel, args.scope),
        "in_sample_fraction": in_sample_fraction(score),
        "metrics": metrics,
        "missing_metrics": missing,
        "provenance": provenance,
    }
    gate_row_shape(row, registry)
    view = gate_row_address(row, boards, args.board, registry)
    gate_row_against_board(row, board, args.board)
    where = root / (ANCHOR if args.board == ANCHOR else f"rows/{args.board}")
    out = where / view / f"{args.method}@{args.version}.json"
    if out.exists() and not args.force:
        raise GateError(f"{out} already exists; pass --force to restamp")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump_json(row), encoding="utf-8")
    print(f"stamped {out.relative_to(root)}")
    return 0


# ---------------------------------------------------------------- ranking ---

def rank_spreads(values: Mapping[str, float], direction: str,
                 floor: Optional[float]) -> Dict[str, Tuple[int, int]]:
    """PRD §5.4.1 — rank spread per row id. Two rows whose gap is under the floor are tied.

    best = 1 + #{j beats i by more than the floor}; worst = N − #{i beats j by more than it}.
    No floor → floor 0: only exact ties share a rank.
    """
    f = floor or 0.0
    sign = -1.0 if direction == "higher" else 1.0
    ids = sorted(values)
    n = len(ids)
    out: Dict[str, Tuple[int, int]] = {}
    for i in ids:
        vi = sign * values[i]
        beat_me = sum(1 for j in ids if j != i and vi - sign * values[j] > f)
        i_beat = sum(1 for j in ids if j != i and sign * values[j] - vi > f)
        out[i] = (1 + beat_me, n - i_beat)
    return out


def interval_rank_spreads(intervals: Mapping[str, Tuple[float, float]]) -> Dict[str, Tuple[int, int]]:
    """Rank spread over [lo, hi] score intervals, lower is better.

    j is certainly better than i when hi_j < lo_i; both spreads count only certainties.
    """
    ids = sorted(intervals)
    n = len(ids)
    out: Dict[str, Tuple[int, int]] = {}
    for i in ids:
        lo_i, hi_i = intervals[i]
        surely_above = sum(1 for j in ids if j != i and intervals[j][1] < lo_i)
        surely_below = sum(1 for j in ids if j != i and hi_i < intervals[j][0])
        out[i] = (1 + surely_above, n - surely_below)
    return out


def compile_view(rows: Sequence[Mapping[str, Any]], registry: Mapping[str, Any],
                 view: str, ranked_here: bool = True,
                 unranked_reason: str = "") -> Dict[str, Any]:
    """One address: ranks, category sub-scores, composite spreads, sub-boards.

    `ranked_here` is a first-class state, not an error (§15). An address that does not rank —
    a reported-only panel or scope, the anchor block, or any regime whose noise floor is still
    unmeasured — still carries every number it has. It just carries no order over them, and
    says why. Nothing is silently dropped and nothing is silently ordered.
    """
    metrics_field = "metrics"
    ranked = [r for r in rows if r.get(metrics_field)]
    unranked = [r for r in rows if not r.get(metrics_field)]
    by_id = {f"{r['method']}@{r['version']}": r for r in ranked}

    def val(rid: str, m: Mapping[str, Any]) -> Optional[float]:
        return by_id[rid][metrics_field].get(metric_slot(m), {}).get(m["key"])

    ranked_metrics = [m for m in registry["metrics"] if m["role"] == "ranked"]
    metric_ranks: Dict[str, Dict[str, Tuple[int, int]]] = {}
    for m in ranked_metrics:
        have = {rid: val(rid, m) for rid in by_id}
        have = {rid: v for rid, v in have.items() if v is not None}
        if have:
            metric_ranks[metric_id(m)] = rank_spreads(have, m["direction"], m["floor"])

    # §5.2 / PI 2026-08-27 — a category is board-active when at least one row fully
    # covers it. A row that misses any active composite category is incomplete, not
    # last: no composite, no headline rank; it still ranks inside categories it covers.
    cats = registry["categories"]
    in_composite = [cid for cid, cat in sorted(cats.items()) if cat.get("in_composite")]

    def covers(rid: str, cid: str) -> bool:
        cat_metrics = [m for m in ranked_metrics if m["category"] == cid]
        return bool(cat_metrics) and all(val(rid, m) is not None for m in cat_metrics)

    composite_cats = [cid for cid in in_composite
                      if any(covers(rid, cid) for rid in by_id)]
    eligible = {rid for rid in by_id
                if all(covers(rid, cid) for cid in composite_cats)}

    out_rows = []
    intervals: Dict[str, Tuple[float, float]] = {}
    for rid, r in by_id.items():
        subscores: Dict[str, Tuple[float, float]] = {}
        for cid in in_composite:
            if not covers(rid, cid):
                continue
            spreads = [metric_ranks[metric_id(m)][rid]
                       for m in ranked_metrics if m["category"] == cid]
            if spreads:
                subscores[cid] = (sum(s[0] for s in spreads) / len(spreads),
                                  sum(s[1] for s in spreads) / len(spreads))
        composite = None
        partial = rid not in eligible
        if not partial and composite_cats:
            used = [subscores[cid] for cid in composite_cats]
            composite = (sum(s[0] for s in used) / len(used),
                         sum(s[1] for s in used) / len(used))
            intervals[rid] = composite
        missing_cats = [cid for cid in composite_cats if not covers(rid, cid)]
        out_rows.append({
            "id": rid,
            "method": r["method"],
            "version": r["version"],
            "date": r["date"],
            "lineage": r["lineage"],
            "badges": r["badges"],
            "metrics": r[metrics_field],
            "missing_metrics": r.get("missing_metrics", []),
            "provenance": r["provenance"],
            "address": r.get("address"),
            "markers": r.get("markers", []),
            # §4, §5.2 — the row's own rankability, and the fraction §4 wants printed beside a
            # genome-wide number (`null` until a producer writes one — see `IN_SAMPLE_KEY`).
            "ranked": bool(r.get("ranked")),
            "in_sample_fraction": r.get("in_sample_fraction"),
            "has_peak_head": bool((r.get("provenance") or {}).get("has_peak_head")),
            "verified": bool(r["provenance"].get("artifacts_resolved_at_add")),
            "metric_ranks": {mid: list(rk[rid]) for mid, rk in metric_ranks.items()
                             if rid in rk},
            "category_subscores": {c: list(s) for c, s in subscores.items()},
            "composite": list(composite) if composite else None,
            "partial_coverage": partial and bool(composite_cats),
            "missing_composite_categories": missing_cats,
        })
    overall = interval_rank_spreads(intervals) if intervals else {}
    for row in out_rows:
        row["rank"] = list(overall[row["id"]]) if row["id"] in overall else None
    out_rows.sort(key=lambda r: (r["composite"] is None,
                                 r["composite"] or [math.inf, math.inf],
                                 r["method"], r["version"]))

    # count-arm sub-board (§5.2): ranks only the methods that emit counts.
    count_m = next(m for m in registry["metrics"]
                   if m["arm"] == "count" and m["key"] == "crps")
    count_rows = [r for r in out_rows
                  if all(k in r["metrics"].get("count", {})
                         for k in ["crps", *count_m.get("companions", [])])]
    count_vals = {r["id"]: r["metrics"]["count"]["crps"] for r in count_rows}
    count_ranks = rank_spreads(count_vals, count_m["direction"], count_m["floor"]) \
        if count_vals else {}
    sub_count = {
        "note": cats["count_arm"]["note"],
        "rows": sorted(({"id": r["id"], "method": r["method"], "version": r["version"],
                         "metrics": {"count": r["metrics"]["count"]},
                         "rank": list(count_ranks[r["id"]])} for r in count_rows),
                       key=lambda r: (r["rank"], r["method"], r["version"])),
    }

    # CANDI-lineage sub-board (§5.3): covariate diagnostics, CANDI versions only.
    lineage_rows = [{"id": r["id"], "method": r["method"], "version": r["version"],
                     "date": r["date"], "metrics": {"diagnostics": r["metrics"]["diagnostics"]}}
                    for r in out_rows
                    if r["lineage"] == "candi" and r["metrics"].get("diagnostics")]
    sub_lineage = {"note": cats["covariate_diagnostics"]["note"],
                   "rows": sorted(lineage_rows, key=lambda r: (r["date"], r["id"]))}

    if not ranked_here:
        # Keep every number, drop every order. A rank computed without a resolution band
        # would be a claim we cannot make (§15); a hidden number would be evidence withheld.
        for row in out_rows:
            row["rank"] = None
            row["composite"] = None
            row["metric_ranks"] = {}
            row["category_subscores"] = {}
            row["partial_coverage"] = False
            row["missing_composite_categories"] = []
        out_rows.sort(key=lambda r: (r["method"], r["version"]))
        for r in sub_count["rows"]:
            r["rank"] = None
        sub_count["rows"].sort(key=lambda r: (r["method"], r["version"]))

    return {
        "categories_in_composite": composite_cats if ranked_here else [],
        "ranking": {"state": "ranked" if ranked_here else "unranked",
                    "reason": "" if ranked_here else unranked_reason},
        "rows": out_rows,
        "unscored": sorted(({"id": f"{r['method']}@{r['version']}", "method": r["method"],
                             "version": r["version"],
                             "note": f"no scores stamped at {view}"} for r in unranked),
                           key=lambda r: r["id"]),
        "sub_boards": {"count_arm": sub_count, "candi_lineage": sub_lineage},
    }


def read_rows(root: Path, boards: Mapping[str, Any], registry: Mapping[str, Any],
              bid: str) -> Dict[str, List[Dict[str, Any]]]:
    """Every stamped row of one container, bucketed by address."""
    board = board_spec(boards, bid)
    base = root / (ANCHOR if bid == ANCHOR else f"rows/{bid}")
    by_view: Dict[str, List[Dict[str, Any]]] = {v: [] for v in view_keys(boards, bid)}
    for path in sorted(base.glob("*/*.json")) if base.exists() else []:
        row = load_json(path)
        gate_row_shape(row, registry)
        view = gate_row_address(row, boards, bid, registry)
        gate_row_against_board(row, board, bid)
        if row["board"] != bid:
            raise GateError(f"{path} says board `{row['board']}` but lives under `{bid}`")
        if path.parent.name != view:
            raise GateError(f"{path} is addressed `{view}` but lives under "
                            f"`{path.parent.name}` — the path is the address")
        if view not in by_view:
            raise GateError(f"{path} is addressed `{view}`, which `{bid}` does not offer")
        by_view[view].append(row)
    return by_view


def compile_pending(board: Mapping[str, Any], bid: str, stamped: Sequence[str],
                    markers: Mapping[str, Any]) -> List[Dict[str, Any]]:
    out = []
    raw_pending = board.get("pending") or []
    if not isinstance(raw_pending, list):
        raise GateError(f"board `{bid}` pending must be a list")
    for item in raw_pending:
        if not isinstance(item, dict) or not item.get("method"):
            raise GateError(f"board `{bid}` pending entry needs a `method`")
        if not SLUG.match(item["method"]):
            raise GateError(f"board `{bid}` pending method `{item['method']}` is not a slug")
        if item["method"] in stamped:
            continue
        lineage = item.get("lineage") or "rival"
        if lineage not in LINEAGES:
            raise GateError(f"board `{bid}` pending `{item['method']}` lineage "
                            f"`{lineage}` is not one of {LINEAGES}")
        out.append({
            "method": item["method"],
            "version": item.get("version") or "",
            "lineage": lineage,
            "markers": sorted(markers.get(item["method"], [])),
            "note": item.get("note") or "results computing",
        })
    out.sort(key=lambda p: (p["method"], p["version"]))
    return out


def compile_void(root: Path, boards: Mapping[str, Any],
                 registry: Mapping[str, Any]) -> Dict[str, Any]:
    """§3.3 — what the board used to show, named but never numbered.

    The row files are read and shape-gated so they cannot rot, then everything except the
    identity is dropped. Printing a void score under a new label is the one failure this whole
    redesign exists to prevent, so the compiler removes the temptation rather than warning
    about it.
    """
    meta = boards["void"]
    rows = []
    base = root / "void"
    for path in sorted(base.glob("*/*.json")) if base.exists() else []:
        row = load_json(path)
        gate_row_shape(row, registry)
        former = path.parent.name
        if row["board"] != former:
            raise GateError(f"{path} says board `{row['board']}` but lives under `{former}`")
        rows.append({
            "method": row["method"],
            "version": row["version"],
            "date": row["date"],
            "lineage": row["lineage"],
            "former_board": former,
            "reason": meta.get("reason_by_board", {}).get(former, meta.get("eli5", "")),
        })
    rows.sort(key=lambda r: (r["former_board"], r["method"], r["version"]))
    return {"meta": meta, "rows": rows}


def compile_leaderboard(root: Path) -> Dict[str, Any]:
    registry = load_registry(root)
    boards = load_boards(root)
    out_boards: Dict[str, Any] = {}
    for bid in sorted(list(boards["boards"]) + [ANCHOR]):
        board = board_spec(boards, bid)
        by_view = read_rows(root, boards, registry, bid)
        views = {}
        for view, rows in sorted(by_view.items()):
            ranked_here, why = view_is_ranked(boards, bid, view)
            views[view] = compile_view(rows, registry, view, ranked_here, why)
        canonical = CANONICAL_VIEW if CANONICAL_VIEW in views else sorted(views)[0]
        climb: Dict[str, List[Dict[str, Any]]] = {}
        for r in views[canonical]["rows"]:
            climb.setdefault(r["method"], []).append(
                {"date": r["date"], "version": r["version"],
                 "composite": r["composite"], "lineage": r["lineage"]})
        for entries in climb.values():
            entries.sort(key=lambda e: (e["date"], e["version"]))
        stamped_methods = {r["method"] for rows in by_view.values() for r in rows}
        out_boards[bid] = {
            "meta": board,
            "kind": "anchor" if bid == ANCHOR else "regime",
            "canonical_view": canonical,
            "views": views,
            "climb": climb,
            "pending": compile_pending(board, bid, sorted(stamped_methods),
                                       method_markers(boards, bid)),
        }
    n_diag, scorers, lineages = 0, set(), set()
    for board in out_boards.values():
        view = board["views"][board["canonical_view"]]
        n_diag += len(view["sub_boards"]["candi_lineage"]["rows"])
        for r in view["rows"]:
            if r.get("provenance", {}).get("scorer"):
                scorers.add(r["provenance"]["scorer"])
            if r.get("lineage"):
                lineages.add(r["lineage"])
    return {
        "schema_version": 2,
        "registry": registry,
        "address": boards["address"],
        "truths": boards["truths"],
        "panels": boards["panels"],
        "panel_rules": boards.get("panel_rules", {}),
        "scopes": boards["scopes"],
        "markers": boards["markers"],
        "deferred_regimes": boards.get("deferred_regimes", []),
        "canonical_view": CANONICAL_VIEW,
        "boards": out_boards,
        "anchor_id": ANCHOR,
        "void": compile_void(root, boards, registry),
        "covariate_coverage": {
            "n_rows_with_diagnostics": n_diag,
            "scorers": sorted(scorers),
            "lineages": sorted(lineages),
        },
        "reproducibility": {
            "score_command": "python -m candi.bench.external --store <regime.json> "
                             "--pred <pred_root> --out <scores.json>",
            "stamp_command": "python tools/leaderboard.py add <scores.json> --board <b> "
                             "--method <m> --version <v> ...",
            "note": "Every row links its score json and FIR path; scoring stays a manual Fir "
                    "run and this build only compiles what was stamped (PRD §6, §9).",
        },
    }


SITE_FILES = ("index.html", "app.js", "style.css", "help.json")


def cmd_build(args: argparse.Namespace) -> int:
    root = Path(args.root)
    out = Path(args.out)
    compiled = compile_leaderboard(root)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / "leaderboard.json").write_text(dump_json(compiled), encoding="utf-8")
    for name in SITE_FILES:
        src = root / "site" / name
        if not src.exists():
            raise GateError(f"site source {src} is missing")
        (out / name).write_bytes(src.read_bytes())
    print(f"built {out} ({len(compiled['boards']) - 1} regimes + the anchor block, "
          f"{len(compiled['void']['rows'])} void rows)")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    root = Path(args.root)
    registry = load_registry(root)
    boards = load_boards(root)
    n_rows = 0
    for bid in sorted(list(boards["boards"]) + [ANCHOR]):
        board = board_spec(boards, bid)
        base = root / (ANCHOR if bid == ANCHOR else f"rows/{bid}")
        for path in sorted(base.glob("*/*.json")) if base.exists() else []:
            row = load_json(path)
            gate_row_shape(row, registry)
            gate_row_address(row, boards, bid, registry)
            gate_row_against_board(row, board, bid)
            # integrity: when the stamped score json is reachable here, its macro must still
            # say what the row says. Unreachable (CI has no cruxvault/results) is a skip.
            score_path = Path(row["provenance"]["score_json"])
            if not score_path.is_absolute():
                score_path = REPO / score_path
            if score_path.exists():
                # re-extracted at the ROW'S OWN ADDRESS: a row that silently started reading a
                # different panel or scope of the same file would still match otherwise.
                fresh, _ = extract_metrics(load_json(score_path), registry, allow_missing=True,
                                           panel=row["address"]["panel"],
                                           scope=row["address"]["scope"])
                stamped = {slot: block for slot, block in row["metrics"].items() if block}
                if fresh != stamped:
                    raise GateError(f"{path} no longer matches its score json {score_path}")
            n_rows += 1
    # void rows are never printed as numbers, but they are still gated so they cannot rot
    n_void = 0
    for path in sorted((root / "void").glob("*/*.json")) if (root / "void").exists() else []:
        gate_row_shape(load_json(path), registry)
        n_void += 1
    for name in SITE_FILES:
        src = root / "site" / name
        if not src.exists():
            raise GateError(f"site source {src} is missing")
        # the W3C namespace identifier (SVG createElementNS / favicon xmlns) is a name,
        # not a request; everything else that smells like a URL is refused.
        text = src.read_text(encoding="utf-8") \
            .replace("http://www.w3.org/", "").replace("http%3A%2F%2Fwww.w3.org%2F", "")
        if "http://" in text or "https://" in text:
            raise GateError(f"{src} references an external URL — the site makes no "
                            "external requests (PRD §8)")
    with tempfile.TemporaryDirectory() as tmp:
        a, b = Path(tmp) / "a", Path(tmp) / "b"
        for target in (a, b):
            ns = argparse.Namespace(root=root, out=target)
            cmd_build(ns)
        for name in ("leaderboard.json", *SITE_FILES):
            if (a / name).read_bytes() != (b / name).read_bytes():
                raise GateError(f"build is not deterministic: {name} differs between reruns")
    print(f"check passed: {n_rows} rows, {n_void} void, {len(boards['boards'])} regimes "
          "+ the anchor block, deterministic rebuild bit-exact")
    return 0


# ---------------------------------------------------------------- CLI ---

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="leaderboard.py", description=__doc__.splitlines()[0])
    p.add_argument("--root", default=str(DEFAULT_ROOT),
                   help="leaderboard directory (registry, boards, rows, site)")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="stamp one score json into one row")
    a.add_argument("score", help="candi.bench(.external)-shaped score json")
    a.add_argument("--board", required=True,
                   help=f"a regime id, or `{ANCHOR}` for a 2019-field row")
    # §1's address. No default on any of the three: an unknown field is a refused row.
    a.add_argument("--truth", required=True,
                   help="store | challenge — must agree with the score json's "
                        "provenance.truth.source")
    a.add_argument("--panel", required=True,
                   help="V_breadth | V_matched | B_ — read from the score json's `panels` block")
    a.add_argument("--scope", required=True,
                   help="held-out | genome-wide — genome-wide reads the `genome_wide` block")
    a.add_argument("--method", required=True)
    a.add_argument("--version", required=True)
    a.add_argument("--date", required=True, help="YYYY-MM-DD the scores landed")
    a.add_argument("--lineage", required=True, choices=LINEAGES)
    a.add_argument("--position-class", required=True, choices=sorted(POSITION_CLASSES))
    a.add_argument("--cell-class", required=True, choices=sorted(CELL_CLASSES))
    a.add_argument("--scoring-sha", required=True, help="git SHA of the scoring code")
    a.add_argument("--store-manifest-hash", required=True,
                   help="hash of the store manifest the run scored against")
    a.add_argument("--fir-path", default=None,
                   help="run directory on Fir; default reads FIR_PATH.txt beside the score json")
    a.add_argument("--allow-missing", action="store_true",
                   help="record absent registry metrics instead of refusing")
    a.add_argument("--force", action="store_true", help="restamp an existing row")
    a.set_defaults(func=cmd_add)

    b = sub.add_parser("build", help="compile rows into _site/")
    b.add_argument("--out", default=str(DEFAULT_OUT))
    b.set_defaults(func=cmd_build)

    c = sub.add_parser("check", help="re-run every gate and diff a double build bit-exact")
    c.set_defaults(func=cmd_check)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
