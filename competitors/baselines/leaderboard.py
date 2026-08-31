"""Fold the score files `candi.bench.external` writes into one leaderboard json (§5.5, §6.4).

    python -m competitors.baselines.leaderboard --protocol P1 \\
        --scores avg=avg.json marginal=marginal.json ... --out leaderboard.json

It computes NOTHING. Every number in the output is copied from a score file — this module groups
per-track rows by assay, takes the unweighted mean the way `harness.macro_mean` does, and enforces
the §6.4 reporting invariants on the way out.

THE FOUR INVARIANTS, AS CODE RATHER THAN AS A CONVENTION
--------------------------------------------------------
1. **Per-assay rows first, macro second.** `per_assay` is the first key of every method block;
   `macro` follows it.
2. **Broad and punctate are never pooled without their separate medians beside the pool.**
   `macro[arm]["<key>_broad_median"]` and `_punctate_median` sit next to every pooled key. The broad
   set is §6.4's: H3K27me3, H3K36me3, H3K9me3. Accessibility (DNase-seq, ATAC-seq) is a THIRD group
   and not a "mark" in either sense, so it gets `_accessibility_median` rather than being folded into
   punctate — pooling an accessibility assay into a histone median is the same error one level down.
3. **`auprc` never without `peak_base_rate`, `crps` never without `crps_oracle_scaled` and
   `scale_error`.** A score file that carries one without the other raises here, naming the track.
4. **A peak row that ranked by level is labelled.** `has_peak_head` travels from the score file's
   `provenance` into `per_assay[...]["peak_ranking"]` as either `"peak_head"` or
   `"coverage_ranking"`. §6.4 requires the label on the row itself.

The noise floors from `AGENTS.md` §7.2 are stamped into the header because the quoting rule in
`CLAUDE.md` says a number is not quotable without them, and a leaderboard is the one artifact that
will be quoted from most.

The §5.5 sanity anchors are checked, never tuned: `--check-anchors` exits non-zero and says which
one failed. A failed anchor is a defect in the generator or in the panel, and the plan's instruction
is to stop, not to adjust the baseline until it passes.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = ["BROAD", "ACCESSIBILITY", "NOISE_FLOORS", "assemble", "check_anchors",
           "build_parser", "main"]

#: §6.4, verbatim. Everything else that is a histone mark is punctate.
BROAD: Tuple[str, ...] = ("H3K27me3", "H3K36me3", "H3K9me3")

#: Not marks at all. Kept out of both medians and given their own.
ACCESSIBILITY: Tuple[str, ...] = ("DNase-seq", "ATAC-seq")

#: `AGENTS.md` §7.2. Quoted with every number this file emits, because the rule in `CLAUDE.md` is
#: that a CANDII number without its floor is not quotable. These are frozen pre-CANDII values and are
#: copied, never recomputed.
NOISE_FLOORS: Dict[str, float] = {
    "macro_crps_target_clustered": 0.09,
    "pooled_crps_seed_change": 0.1195,
    "macro_crps_seed_floor_bench": 0.0608,
}

#: §6.4 — the pairs that may not be separated, per arm. `arm -> key -> the keys that must ship
#: beside it`. `None` means "every arm".
#:
#: THE TWO CRPS RULES ARE DIFFERENT RULES, per the PI ruling of 2026-08-26 (plan amendment PR #27),
#: recorded in `PI_RULINGS` below.
#:
#: * **Count arm.** `crps` never without `crps_oracle_scaled` + `scale_error`. `oracle_scale` fits a
#:   per-assay scale to an NB and re-scores it, which splits capability from per-assay level.
#: * **Pval arm.** That split does not exist — `gauss_suite` has no oracle-scale counterpart, and
#:   demanding one would ask `candi.bench` for a key it does not compute. The Gaussian CRPS is
#:   instead quoted with `pit_ks` and `coverage_95`, the two keys that say whether the forecast
#:   distribution is calibrated at all. A CRPS with neither is a sharpness number with no calibration
#:   beside it, which is exactly the reading §6.4 exists to prevent.
_COMPANIONS: Dict[Optional[str], Dict[str, Tuple[str, ...]]] = {
    None: {"auprc": ("peak_base_rate",)},
    "count": {"crps": ("crps_oracle_scaled", "scale_error")},
    "pval": {"crps": ("pit_ks", "coverage_95")},
}

#: Columns on which a between-method GAP may not be presented as significant, because no noise floor
#: has been measured for them yet. Stamped onto the arm's macro block as `<key>_gap_not_quotable` so
#: the caveat travels on the row rather than in a caption someone drops.
#:
#: `CLAUDE.md`'s quoting rule is that a number needs its floor; `AGENTS.md` §7.2 supplies floors for
#: the COUNT arm's CRPS (target-clustered 0.09, seed change 0.1195, bench seed floor 0.0608) and for
#: nothing on the pval arm. Until the pval-arm Gaussian-CRPS floor is measured — its own task as of
#: 2026-08-26 — a pval CRPS is quotable as a value and a pval CRPS DIFFERENCE is not.
_GAP_NOT_QUOTABLE: Dict[str, Tuple[str, ...]] = {
    "pval": ("crps",),
}


def _group(assay: str) -> str:
    if assay in ACCESSIBILITY:
        return "accessibility"
    return "broad" if assay in BROAD else "punctate"


def _scalars(row: Mapping[str, Any]) -> Dict[str, float]:
    return {k: float(v) for k, v in row.items()
            if isinstance(v, (int, float, bool)) and not isinstance(v, str)}


def _check_companions(where: str, arm: str, keys: Sequence[str]) -> None:
    have = set(keys)
    for scope in (None, arm):
        for k, companions in _COMPANIONS.get(scope, {}).items():
            if k not in have:
                continue
            gone = [c for c in companions if c not in have]
            if gone:
                raise ValueError(
                    f"{where}: `{k}` is present and {gone} are not. RIVALS_PLAN.md §6.4 forbids "
                    f"quoting one without the other — a raw CRPS with no oracle_scaled/scale_error "
                    f"split conflates capability with per-assay level, and an AUPRC with no base "
                    f"rate is unreadable. Re-score; do not drop the companion key.")


def _mean(vals: Sequence[float]) -> float:
    return float(sum(vals) / len(vals))


def assemble(scores: Mapping[str, Path | str], *, protocol: str,
             notes: str = "") -> Dict[str, Any]:
    """`{method: score json path}` -> the leaderboard object. Nothing is recomputed."""
    methods: Dict[str, Any] = {}
    for name, path in scores.items():
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        prov = obj["provenance"]
        # ABSENT means the closed form was used. Its PRESENCE is the only thing that says a
        # count-arm `crps` in this file is an estimate, so it is read once here and carried onto
        # both the macro row and the method's provenance.
        estimator = prov.get("crps_estimator")
        by_assay: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
        n_tracks: Dict[str, Dict[str, int]] = {}
        peak_ranking: Dict[str, str] = {}
        for key, arms in obj["per_track"].items():
            for arm, row in arms.items():
                s = _scalars(row)
                _check_companions(f"{name}/{key}/{arm}", arm, sorted(s))
                assay = str(row["assay"])
                for k, v in s.items():
                    by_assay.setdefault(arm, {}).setdefault(assay, {}).setdefault(k, []).append(v)
                n_tracks.setdefault(arm, {})[assay] = n_tracks.get(arm, {}).get(assay, 0) + 1
                # `bernoulli_nll` is emitted only when the producer supplied a real `peak_score`
                # (`harness.loss_block` withholds it when `has_peak_head` is False), so its presence
                # IS the §6.4 label.
                peak_ranking[assay] = ("peak_head" if "bernoulli_nll" in s
                                       else "coverage_ranking")

        per_assay: Dict[str, Dict[str, Any]] = {}
        macro: Dict[str, Dict[str, Any]] = {}
        for arm, assays in by_assay.items():
            per_assay[arm] = {}
            for assay in sorted(assays):
                per_assay[arm][assay] = {
                    **{k: _mean(v) for k, v in sorted(assays[assay].items())},
                    "n_tracks": n_tracks[arm][assay],
                    "group": _group(assay),
                    # §6.4 — the label lives on the row, not in a caption someone may drop.
                    "peak_ranking": peak_ranking.get(assay, "coverage_ranking"),
                }
            pooled = obj["macro"].get(arm, {})
            _check_companions(f"{name}/macro/{arm}", arm, sorted(pooled))
            out = dict(pooled)
            keys = sorted({k for a in assays for k in assays[a]})
            for k in keys:
                for grp in ("broad", "punctate", "accessibility"):
                    vals = [_mean(assays[a][k]) for a in assays
                            if _group(a) == grp and k in assays[a]]
                    if vals:
                        out[f"{k}_{grp}_median"] = float(median(vals))
                        out[f"{k}_{grp}_n_assays"] = len(vals)
            # PI 2026-08-26 — the caveat rides on the row. A reader who diffs two methods on this
            # column has to meet the reason they may not, without going back to a caption.
            for k in _GAP_NOT_QUOTABLE.get(arm, ()):
                if k in out:
                    out[f"{k}_gap_not_quotable"] = (
                        f"no {arm}-arm noise floor measured for `{k}` yet (PI 2026-08-26): quote "
                        f"the value, never a between-method gap on it, until the floor lands.")
            # The count arm's CRPS is either exact or an estimate, and the row must say which.
            if arm == "count":
                if estimator:
                    out["crps_estimator"] = estimator
                    out["crps_k"] = prov.get("crps_k")
                    out["crps_seed"] = prov.get("crps_seed")
                else:
                    for k in _UNRELIABLE_AT_CLOSED_FORM:
                        if k in out:
                            out[f"{k}_unreliable"] = (
                                f"`{k}` is read off oracle_scale's n-grid, which runs up to 16x the "
                                f"fitted n and lands where the closed-form nb_crps is NaN. Do not "
                                f"quote it from a closed-form score file (t56 finding).")
            macro[arm] = out

        methods[name] = {
            # ORDER MATTERS HERE (§6.4): per-assay rows first, macro second.
            "per_assay": per_assay,
            "macro": macro,
            "provenance": {
                "method": prov["method"],
                "manifest": prov.get("manifest", {}),
                "pred_root": prov.get("pred_root"),
                "declared_tracks": prov.get("declared_tracks"),
                "missing_tracks": prov.get("missing_tracks", []),
                "n_scored_tracks": len(obj["tracks"]),
                "pred_inversion": prov.get("pred_inversion"),
                "point_only_tracks": prov.get("point_only_tracks", []),
                "crps_estimator": estimator or "closed_form",
                "crps_k": prov.get("crps_k"),
                "crps_seed": prov.get("crps_seed"),
                "sparse_assays": prov.get("manifest", {}).get("sparse_assays", []),
                "msevar": prov.get("msevar"),
            },
        }

    return {
        "generated": date.today().isoformat(),
        "protocol": protocol,
        "notes": notes,
        # Quoted with every number, per `CLAUDE.md` / `AGENTS.md` §7.2. Every floor here is a
        # COUNT-arm CRPS floor; the pval arm has none yet, which is what `_GAP_NOT_QUOTABLE` says.
        "noise_floors": dict(NOISE_FLOORS),
        "noise_floors_absent": {
            "pval_macro_crps": "not measured (separate task, opened 2026-08-26). A pval-arm "
                               "Gaussian CRPS is quotable as a value; a between-method GAP on it "
                               "is not, until this floor exists.",
        },
        "reporting": {
            "broad": list(BROAD),
            "accessibility": list(ACCESSIBILITY),
            "rule": "RIVALS_PLAN.md §6.4 — per-assay rows first, macro second; broad and punctate "
                    "medians always beside the pool; auprc with peak_base_rate; a row that ranked "
                    "peaks by predicted level is labelled coverage_ranking. CRPS companions are "
                    "PER ARM: count-arm crps with crps_oracle_scaled and scale_error, pval-arm "
                    "crps with pit_ks and coverage_95.",
            "pi_rulings": {k: dict(v) for k, v in PI_RULINGS.items()
                           if v.get("scope") == "reporting"},
            # One line a reader cannot miss when a table mixes the two. `assemble` never refuses a
            # mix — P2's `avg-arcsinh` has no count arm and was scored before the switch, so a mixed
            # table is a legitimate state — but it must be visible rather than inferable.
            "crps_estimator": {
                m: blk["provenance"]["crps_estimator"] for m, blk in methods.items()
            },
        },
        "methods": methods,
        "sanity": check_anchors(methods),
    }


#: PI rulings, carried by the code that writes the thing they rule on.
#:
#: A ruling lives HERE rather than being pasted into a generated json, because a leaderboard is
#: regenerated every time a method is added and a hand-edited verdict would vanish on the next run.
#:
#: `scope` says where the ruling is attached:
#:   * `"anchor"` — onto that §5.5 anchor's verdict block. A failing anchor with a ruling still
#:     reports `pass: False`; the ruling records what the PI concluded FROM the failure and never
#:     converts it into a success. Nothing in this file may set `pass` from a ruling.
#:   * `"reporting"` — onto the header's `reporting` block, because it governs table semantics
#:     rather than one result.
PI_RULINGS: Dict[str, Dict[str, str]] = {
    "avg_beats_marginal_near_universal": {
        "scope": "anchor",
        "date": "2026-08-25",
        "ruling": "ACCEPT as a real finding. The anchor failed at the near-universal reading for an "
                  "understood, mechanistic reason: all 7 losing tracks average over k=3 "
                  "contributors (winners: median 16), and a 3-cell cross-cell mean is a noisy "
                  "predictor. The PI ruled this a genuine property of the LOO average rather than "
                  "an artifact, and declined a post-hoc sparse-threshold change.",
        "rules_changed": "none — RIVALS_PLAN.md §5's sparse rule stays at n_eligible <= 2, which "
                         "flags nothing on this panel (min k = 3), and the 0.9 reading of "
                         "'near-universal' stands.",
    },
    "crps_companions_are_per_arm": {
        "scope": "reporting",
        "date": "2026-08-26",
        "ruling": "RIVALS_PLAN.md §6.4's 'CRPS always with oracle_scaled + scale_error' is "
                  "CLARIFIED as count-arm (NB) only. Pval-arm Gaussian CRPS rows are instead always "
                  "quoted with pit_ks and coverage_95 beside them. Until a pval-arm Gaussian-CRPS "
                  "noise floor is measured (a separate task as of 2026-08-26), no between-method "
                  "gap on that column may be presented as significant.",
        "rules_changed": "RIVALS_PLAN.md §6.4 amended by PR #27. No re-scoring: the clarification "
                         "changes what a table may say, not what candi.bench computed.",
        "enforced_by": "_COMPANIONS (count -> oracle_scaled + scale_error; pval -> pit_ks + "
                       "coverage_95) and _GAP_NOT_QUOTABLE (pval crps), which stamps "
                       "`crps_gap_not_quotable` onto the pval macro block.",
    },
    "sampled_crps_for_p2": {
        "scope": "reporting",
        "date": "2026-08-26",
        "ruling": "GO for the t56 fair-sampled NB-CRPS estimator on P2 at k=100, all four "
                  "count-arm methods. The rank-flip criterion is read on the MACRO ordering, which "
                  "never flips at any k or seed — recorded as the PI's reading. The strict "
                  "per-track reading failed only on a 0.000087 gap, 1000x below the noise floor.",
        "rules_changed": "none. P1 stays on the closed form; this is a P2 estimator choice, not a "
                         "change to what a CRPS means.",
        "enforced_by": "`--crps-approx 100 --crps-seed 0` on candi.bench.external. Every score "
                       "json it writes stamps provenance.crps_estimator/crps_k/crps_seed, and "
                       "`assemble` copies those onto the method block and onto "
                       "`reporting.crps_estimator` so a sampled CRPS can never be read as an exact "
                       "one — a leaderboard mixing the two says so per method.",
    },
}

#: Keys that are UNRELIABLE in any score json produced at the `n1e4` Poisson floor with the CLOSED
#: FORM, and must not be quoted from one. `oracle_scale` searches an n-grid (`n * 2**k` for k in
#: -4..4), so a track whose fitted `n` sits near the floor is evaluated at `n` up to 16x higher —
#: straight into the region where `candi.metrics.nb_crps` returns NaN. `crps` and
#: `crps_oracle_scaled` survive because their own arguments stay below the ceiling; the two keys
#: below are read off the n-grid itself and do not. Found by t56; the nb_crps-fix task owns the
#: repair. Stamped onto the count macro block so the caveat travels with the artifact.
_UNRELIABLE_AT_CLOSED_FORM: Tuple[str, ...] = ("n_star_log2", "crps_oracle_scaled_and_n")


def check_anchors(methods: Mapping[str, Any]) -> Dict[str, Any]:
    """The two §5.5 sanity anchors. Reported as verdicts; never used to adjust a baseline.

    1. The plain-mean pval baseline (`avg`) beats the per-assay marginal on macro mse — pval arm.
       If it does not, the average is not carrying cross-cell information and something upstream is
       wrong; the plan says stop.
    2. `beats_marginal` is near-universal for the moment-matched NB baseline (`avg`, count arm).
       `beats_marginal` is `bench.distributional.nb_suite`'s comparison against a marginal NB fitted
       on the truth itself, so a strictly richer per-bin predictor should clear it almost everywhere.
       "Near-universal" is read as >= 0.9 of the scored tracks.

    An anchor the PI has ruled on carries `pi_ruling` beside its verdict. The verdict itself is
    untouched by the ruling (see `PI_RULINGS`): anchor 2 reads `pass: False` on the P1 panel and is
    quoted that way, with the ruling explaining what was concluded from it.
    """
    out: Dict[str, Any] = {}
    avg, marg = methods.get("avg"), methods.get("marginal")
    if avg and marg:
        a = avg["macro"].get("pval", {}).get("mse")
        m = marg["macro"].get("pval", {}).get("mse")
        out["avg_beats_marginal_on_macro_pval_mse"] = {
            "avg_mse": a, "marginal_mse": m,
            "pass": None if a is None or m is None else bool(a < m),
        }
    if avg:
        frac = avg["macro"].get("count", {}).get("beats_marginal")
        out["avg_beats_marginal_near_universal"] = {
            "fraction_of_tracks": frac,
            "threshold": 0.9,
            "pass": None if frac is None else bool(frac >= 0.9),
        }
    for name, ruling in PI_RULINGS.items():
        if ruling.get("scope") == "anchor" and name in out:
            out[name]["pi_ruling"] = dict(ruling)
    out["all_pass"] = all(v.get("pass") is True for v in out.values() if isinstance(v, dict))
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m competitors.baselines.leaderboard",
        description="Fold candi.bench.external score files into one leaderboard json "
                    "(RIVALS_PLAN.md §5.5, §6.4).")
    p.add_argument("--scores", nargs="+", required=True, metavar="NAME=PATH",
                   help="one `method=scores.json` per entrant")
    p.add_argument("--protocol", required=True, choices=["P1", "P2", "P3"])
    p.add_argument("--out", required=True)
    p.add_argument("--notes", default="")
    p.add_argument("--check-anchors", action="store_true",
                   help="exit 1 when a §5.5 sanity anchor fails. STOP and report; do not tune.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    scores = dict(s.split("=", 1) for s in a.scores)
    board = assemble(scores, protocol=a.protocol, notes=a.notes)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(board, indent=2), encoding="utf-8")
    print(f"[leaderboard] {len(board['methods'])} method(s) -> {out}", flush=True)
    for k, v in board["sanity"].items():
        if isinstance(v, dict):
            print(f"[leaderboard] anchor {k}: {v}", flush=True)
    if a.check_anchors and not board["sanity"]["all_pass"]:
        print("[leaderboard] a §5.5 sanity anchor FAILED — stop and report (RIVALS_PLAN.md §5.5)",
              flush=True)
        return 1
    return 0


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(main())
