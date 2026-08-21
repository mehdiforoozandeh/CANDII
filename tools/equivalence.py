#!/usr/bin/env python3
"""t22 — the key-by-key equivalence report between `candi.eval` and `candi.bench`.

    python tools/equivalence.py map                        the rule table, and nothing else
    python tools/equivalence.py cover  <run.json>          every eval.py key accounted for?
    python tools/equivalence.py report <run.json> <bench.json> [--out report.md]

`EVAL_PLAN.md` D15 makes the cutover **hard**: `eval.py` is deleted, gated on a published
key-by-key report. That gate is only worth anything if the report is *complete* — a key that
quietly vanished between the two suites is exactly the thing a reader would never notice. So the
rule table below claims every leaf key `eval.py` emits, `cover` refuses to pass while one is
unclaimed, and the report is generated rather than written by hand.

The five verdicts, and what each obliges the writer to supply:

| verdict | meaning | must carry |
|---|---|---|
| `same` | same definition, same population | nothing; any delta is precision or seed |
| `moved` | same definition, **different population** | the mechanism, named |
| `replaced` | a different instrument, same question | why the old one was retired |
| `dropped` | no counterpart at all | why it is not being replaced |
| `new` | `bench` only | nothing; there is no old number to move |

The dominant mechanism is **position scope**, and it is worth stating plainly because it will
account for most of the report: `eval.py` scored a subsample — `--eval-budget 50000000` positions
drawn from batches that themselves covered a strided slice of chr21, since the eval loader advances
one `(T_, V_/B_)` pair per window batch. `candi.bench` scores **every 25 bp bin** (D2). So a key
tagged `moved` has not changed its formula; it has changed what it is a mean over. Nothing here
attributes a delta to a model change, because the model is the same checkpoint on both sides.
"""
from __future__ import annotations

import fnmatch
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCOPE = ("position scope: eval.py drew --eval-budget positions from a strided slice of the eval "
         "chromosome (one (T_, V_/B_) pair advances per window batch); bench scores every 25 bp "
         "bin of it (D2)")
POOL = ("pooling unit: eval.py's macro is a mean over ASSAYS; bench's is a mean over TRACKS "
        "(D4 — the per-track score is the primitive)")


@dataclass(frozen=True)
class Rule:
    """One claim about one family of `eval.py` keys. `old` is an fnmatch pattern over leaf paths."""
    old: str
    verdict: str                    # same | moved | replaced | dropped
    new: Optional[str]              # the bench key path, or None
    why: str

    def __post_init__(self) -> None:
        if self.verdict not in ("same", "moved", "replaced", "dropped"):
            raise ValueError(f"unknown verdict {self.verdict!r}")
        if (self.new is None) != (self.verdict == "dropped"):
            raise ValueError(f"{self.old}: a dropped key has no counterpart, and only a dropped one")


# ---------------------------------------------------------------------------
# M1 — health
# ---------------------------------------------------------------------------
M1_RULES: Tuple[Rule, ...] = (
    Rule("M1.imp_per_assay.*.crps", "moved", "per_track.*.count.crps",
         f"{SCOPE}. Also per-assay -> per-track: an assay's number pooled every cell that carried "
         f"it, which discards the cell, and the cell is the replication unit."),
    Rule("M1.imp_per_assay.*.crps_oracle_scaled", "moved", "per_track.*.count.crps_oracle_scaled",
         SCOPE),
    Rule("M1.imp_per_assay.*.scale_error", "moved", "per_track.*.count.scale_error", SCOPE),
    Rule("M1.imp_per_assay.*.marg_crps", "moved", "per_track.*.count.marg_crps", SCOPE),
    Rule("M1.imp_per_assay.*.marg_mu", "moved", "per_track.*.count.marg_mu", SCOPE),
    Rule("M1.imp_per_assay.*.marg_n", "moved", "per_track.*.count.marg_n", SCOPE),
    Rule("M1.imp_per_assay.*.beats_marginal", "moved", "per_track.*.count.beats_marginal", SCOPE),
    Rule("M1.imp_per_assay.*.beats_marginal_oracle_scaled", "moved",
         "per_track.*.count.beats_marginal_oracle_scaled", SCOPE),
    Rule("M1.imp_per_assay.*.c_star", "moved", "per_track.*.count.c_star", SCOPE),
    Rule("M1.imp_per_assay.*.n_star_log2", "moved", "per_track.*.count.n_star_log2", SCOPE),
    Rule("M1.imp_per_assay.*.crps_oracle_scaled_and_n", "moved",
         "per_track.*.count.crps_oracle_scaled_and_n", SCOPE),
    Rule("M1.imp_per_assay.*.n_points", "moved", "per_track.*.count.n_points",
         "bench scores every bin, so this rises to the chromosome's bin count"),
    Rule("M1.imp_per_assay.*.spearman_raw", "moved", "per_track.*.count.gwspear",
         f"renamed to the challenge's own spelling, same statistic. {SCOPE}"),
    Rule("M1.imp_per_assay.*.pearson_log1p", "replaced", "per_track.*.count.gwcorr",
         "gwcorr is Pearson on the RAW track, which is what score_metrics.py computes and what the "
         "published table reports. log1p Pearson is a different number and is not comparable to it; "
         "the E-block must match the organizers' code (D16), so the raw form wins."),
    Rule("M1.imp_per_assay.*.r2", "dropped", None,
         "r2 on raw counts is a monotone function of gwcorr under a fitted intercept and carried no "
         "decision the correlation did not; it was never quoted in AGENTS.md §7"),
    Rule("M1.imp_per_assay.*.mse_log", "replaced", "per_track.*.count.mse",
         "the E-block's `mse` is on the untransformed track, because normalize_dict is a no-op in "
         "score_metrics.py and the published numbers are untransformed (D16, quirk 5). mse_log and "
         "mse are not comparable and the report must not present them as a delta."),
    Rule("M1.imp_per_assay.*.median_mu", "dropped", None,
         "a diagnostic of the marginal fit, superseded by marg_mu / marg_n which are the fitted "
         "parameters themselves"),
    Rule("M1.imp_per_assay.*.median_target", "dropped", None, "as median_mu"),
    Rule("M1.imp_per_assay.*.marg_crps_legacy_median", "dropped", None,
         "the SUPERSEDED marginal: mu = median(target) + 1e-6. On 4 of 8 assays the median count is "
         "0, so the legacy bar was a near-zero constant and beating it meant nothing. It shipped "
         "beside the CRPS-optimal marginal only to show the gap; with the old suite gone there is "
         "nothing left for it to be a gap against."),
    Rule("M1.imp_per_assay.*.marg_mu_legacy_median", "dropped", None, "as marg_crps_legacy_median"),

    Rule("M1.den_per_assay.*", "moved", "per_track.*|denoise.count.*",
         "denoising is opt-in in bench (--kinds denoise) and its keys carry a fourth field. Same "
         f"per-assay -> per-track move as the imputation half. {SCOPE}"),

    Rule("M1.imp.crps", "moved", "macro.count.crps", f"{SCOPE}. {POOL}"),
    Rule("M1.imp.ece", "moved", "macro.count.ece", f"{SCOPE}. {POOL}"),
    Rule("M1.imp.calib_grid", "same", "per_track.*.count.calib_grid",
         "the 21-point non-randomised PIT grid is a constant, not a measurement"),
    Rule("M1.imp.calib_fbar", "moved", "per_track.*.count.calib_fbar",
         f"{SCOPE}; and it is now per track rather than pooled over the panel"),
    Rule("M1.imp.n_points", "moved", "macro.count.n_points", SCOPE),
    Rule("M1.imp.spearman_raw", "moved", "macro.count.gwspear", f"{SCOPE}. {POOL}"),
    Rule("M1.imp.pearson_log1p", "replaced", "macro.count.gwcorr", "as the per-assay form"),
    Rule("M1.imp.r2", "dropped", None, "as the per-assay form"),
    Rule("M1.den.*", "moved", "macro_denoise.count.*", "as den_per_assay"),

    Rule("M1.imp_macro_crps", "moved", "macro.count.crps", f"{SCOPE}. {POOL}"),
    Rule("M1.imp_macro_crps_oracle_scaled", "moved", "macro.count.crps_oracle_scaled",
         f"{SCOPE}. {POOL}"),
    Rule("M1.imp_macro_scale_error", "moved", "macro.count.scale_error", f"{SCOPE}. {POOL}"),
    Rule("M1.imp_macro_marg_crps", "moved", "macro.count.marg_crps", f"{SCOPE}. {POOL}"),
    Rule("M1.imp_macro_spearman_raw", "moved", "macro.count.gwspear", f"{SCOPE}. {POOL}"),
    Rule("M1.imp_macro_pearson_log1p", "replaced", "macro.count.gwcorr", "as the per-assay form"),
    Rule("M1.imp_beats_marginal_n", "moved", "macro.count.beats_marginal",
         "a COUNT of assays becomes a FRACTION of tracks, because the macro means over tracks and a "
         "bool means to its rate. Read it as a proportion, never as the old integer."),
    Rule("M1.imp_beats_marginal_oracle_scaled_n", "moved",
         "macro.count.beats_marginal_oracle_scaled", "as imp_beats_marginal_n"),
    Rule("M1.den_macro_*", "moved", "macro_denoise.count.*", "as den_per_assay"),
    Rule("M1.encoder_eff_rank_perpos", "moved", "C.C5_C6_invariance.encoder_eff_rank",
         "the effective rank moves into the C6 guard, where it belongs: it was never a health "
         "statistic on its own, it was the tripwire on M3's invariance claim (D13)"),
)

# ---------------------------------------------------------------------------
# M2 — the covariate counterfactuals
# ---------------------------------------------------------------------------
M2_RULES: Tuple[Rule, ...] = (
    Rule("M2.ablation.*.mean_d_crps", "moved", "C.C1_use.*.marginal_mean_d_crps",
         "the same marginal-substitution null, now reported beside the HRT conditional null it was "
         "always missing (D9) and with a randomization-test p-value rather than a bare effect size"),
    Rule("M2.ablation.*.uses_covariate", "moved", "C.C1_use.*.uses_covariate",
         "the verdict is now `marginal_p < 0.05` from an exact randomization test, not a threshold "
         "on an effect size"),
    Rule("M2.ablation.*.frac_true_better", "replaced", "C.C1_use.*.marginal_p",
         "the fraction of targets where the true value beat the substituted one is a sign test in "
         "disguise; the randomization-test p-value is the same evidence stated exactly"),
    Rule("M2.ablation.*.d_crps_clustered", "dropped", None,
         "the target-clustered bootstrap CI has no counterpart: C1's p-value comes from the "
         "randomization null itself, which needs no asymptotics and no cluster correction. THIS IS "
         "A REAL LOSS OF A CONFIDENCE INTERVAL and is the one entry in this table that should be "
         "argued rather than accepted — see the report's Open Items."),
    Rule("M2.ablation.*.mean_abs_d_mu", "dropped", None,
         "an unnormalised effect size in mu units, not comparable across covariates or assays; C2's "
         "Shapley share answers the question it was reaching for (D10)"),
    Rule("M2.ablation.*.mean_abs_d_eta", "dropped", None, "as mean_abs_d_mu"),
    Rule("M2.ablation.*.max_abs_d_eta", "dropped", None, "as mean_abs_d_mu"),
    Rule("M2.ablation.*.n_sentinel_skipped", "dropped", None,
         "bookkeeping for the cross-target substitution's sentinel guard; bench's resamplers draw "
         "from the observed covariate rows, so a sentinel is never manufactured"),
    Rule("M2.ablation.*.per_target", "dropped", None,
         "the per-target record backed the clustered CI; without the CI it has no consumer"),
    Rule("M2.ablation.*.covariate", "same", "C.C1_use.*", "the key name IS the covariate"),
    Rule("M2.ablation.*.row", "dropped", None, "the metadata row index is an implementation detail"),
    Rule("M2.ablation.*.mode", "dropped", None,
         "`cross_target` was the only null; bench names its two nulls in the key instead"),
    Rule("M2.ablation.*.n_targets", "moved", "C.n_units",
         "targets become units: one (window, assay) counterfactual slot each"),
    Rule("M2.ablation_within_batch.*.mean_d_crps", "same",
         "C.C1_use.*.within_batch_d_crps",
         "THE TRIPWIRE. Substitute the value already present; the difference is arithmetically "
         "zero. It must read exactly 0.0 in both suites, and a delta here is a bug in the probe."),
    Rule("M2.ablation_within_batch.*", "dropped", None,
         "the tripwire's remaining fields mirror `ablation`'s and are dropped for the same reasons"),

    Rule("M2.depth.median_total_slope", "replaced", "C.C3_direction.mean_dose_response_corr",
         "a single fitted slope cannot see a curve that wanders; C3 reports the correlation of "
         "predicted level against told depth AND monotone_frac, the step-by-step check"),
    Rule("M2.depth.total_slope_err", "dropped", None, "the standard error of a retired statistic"),
    Rule("M2.depth.total_slope_clamp_saturated", "dropped", None, "clamp telemetry; see below"),
    Rule("M2.depth.frac_targets_any_clamp", "dropped", None,
         "CLAMP TELEMETRY HAS NO COUNTERPART. `log2_mu` is clamped in the decoder head, and these "
         "keys said how often the sweep drove it into the clamp — which is how a depth response can "
         "look flat for a reason that is not the model ignoring depth. Flagged in Open Items."),
    Rule("M2.depth.median_frac_log2mu_at_clamp", "dropped", None, "as frac_targets_any_clamp"),
    Rule("M2.depth.max_frac_log2mu_at_clamp", "dropped", None, "as frac_targets_any_clamp"),
    Rule("M2.depth.p90_frac_log2mu_at_clamp", "dropped", None, "as frac_targets_any_clamp"),
    Rule("M2.depth.direction_clustered.*", "dropped", None,
         "as M2.ablation.*.d_crps_clustered — the clustered CI has no counterpart"),
    Rule("M2.depth.per_target", "dropped", None, "backed the clustered CI"),
    Rule("M2.depth.n_targets", "moved", "C.n_units", "as M2.ablation.*.n_targets"),
    Rule("M2.depth.covariate", "same", "C.C1_use.depth", "the key name IS the covariate"),

    Rule("M2.run_type.mean_responsiveness", "replaced", "C.C1_use.run_type.marginal_mean_d_crps",
         "responsiveness to a run_type flip becomes the covariate's effect size under the same "
         "null every other covariate is measured under, so the four are finally commensurable"),
    Rule("M2.run_type.model_unresponsive", "replaced", "C.C1_use.run_type.uses_covariate",
         "the same verdict with its sign flipped and a p-value behind it"),
    Rule("M2.run_type.*_clustered.*", "dropped", None,
         "as M2.ablation.*.d_crps_clustered — the clustered CIs have no counterpart"),
    Rule("M2.run_type.per_target", "dropped", None, "backed the clustered CIs"),
    Rule("M2.run_type.n_targets", "moved", "C.n_units", "as M2.ablation.*.n_targets"),
    Rule("M2.run_type.covariate", "same", "C.C1_use.run_type", "the key name IS the covariate"),
)

# ---------------------------------------------------------------------------
# M3 / S14
# ---------------------------------------------------------------------------
M3_RULES: Tuple[Rule, ...] = (
    Rule("M3.ratio", "replaced", "C.C5_C6_invariance.kbet_rejection_rate",
         "D12 — M3's within/between cosine-distance RATIO does not survive. It was read against an "
         "arbitrary 0.3 threshold, and a ratio of two distance means has no null distribution to "
         "test against. kBET, iLISI and batch ASW are the scIB instruments for exactly this "
         "question and each has a reachable floor and ceiling. RECORDED M3 NUMBERS ARE "
         "INCOMPARABLE TO THESE and the report must say so rather than tabulate a delta."),
    Rule("M3.within", "replaced", "C.C5_C6_invariance.batch_asw", "as M3.ratio"),
    Rule("M3.between", "replaced", "C.C5_C6_invariance.ilisi", "as M3.ratio"),
    Rule("M3.invariance_ok", "replaced", "C.C5_C6_invariance.invariance_ok",
         "same name, DIFFERENT RULE: the old one was `ratio < 0.3 and eff_rank > 1.0`; the new one "
         "requires kBET rejection < 0.25, batch ASW > 0.8 AND bio_silhouette > 0.25 — the last of "
         "which is the C6 guard a collapsed encoder fails (D13)"),
    Rule("M3.encoder_eff_rank_pooled", "moved", "C.C5_C6_invariance.encoder_eff_rank",
         "kept, but demoted: it is far too weak a guard on its own, since a latent with two "
         "directions clears `> 1.0` while being nearly collapsed"),
    Rule("M3.n_regions", "moved", "C.C5_n_latents", "regions become pooled latent vectors"),
    Rule("M3.n_between_pairs", "dropped", None, "the pair count of a retired statistic"),
)

S14_RULES: Tuple[Rule, ...] = (
    Rule("S14.frac_min_at_true", "moved", "C.C3_depth_counterfactual.frac_min_at_true",
         f"same definition, same calibration (0.25 constant-answer floor, ~0.73 ceiling). {SCOPE}"),
    Rule("S14.frac_beats_told1", "moved", "C.C3_depth_counterfactual.frac_beats_told1", SCOPE),
    Rule("S14.n_targets", "moved", "C.C3_depth_counterfactual.n_levels",
         "the unit is the DSF level being scored, not the target"),
    Rule("S14.per_target", "moved", "C.C3_depth_counterfactual.crps_at_true_level",
         "per-level CRPS rather than per-target"),
    Rule("S14.dsf_counterfactual_ok", "dropped", None,
         "a boolean over frac_min_at_true against a threshold that was never registered; C3 ships "
         "the number and its two calibrations instead of a verdict nobody agreed"),
)

RULES: Tuple[Rule, ...] = M1_RULES + M2_RULES + M3_RULES + S14_RULES

#: Blocks bench adds that eval.py never had. Listed so the report can state the gain, not only the
#: losses — a cutover report that only tabulates what went missing reads as a regression.
NEW_BLOCKS = {
    "E": "the nine ENCODE Imputation Challenge measures, bit-identical to the organizers' code "
         "(mse, gwcorr, gwspear, mseprom, msegene, mseenh, msevar, mse1obs, mse1imp). eval.py had "
         "NONE of them, so CANDI could not be placed against the published field at all.",
    "P": "the four post-hoc measures the challenge's own retrospective recommends instead of the "
         "nine: accuracy by observed and by imputed strength, precision/recall by cell-type "
         "specificity, and the two region-restricted correlations.",
    "D": "the Gaussian arm (gauss_crps, pit_ks), the C-index with its Monte-Carlo SE, and 95% "
         "coverage — none of which eval.py computed.",
    "B": "AUPRC, peak overlap and the correspondence curve against the MACS2 calls. AUROC is "
         "excluded deliberately (D14).",
    "C2": "Shapley effects — what FRACTION of the output each covariate owns. eval.py could say a "
          "covariate mattered; it could not say how much, or compare two of them.",
    "C4": "the covariate x aspect specificity matrix: does each covariate move what it should.",
    "pval arm": "every block above, run a second time against the -log10 p-value track (D1). "
                "eval.py scored the count head only.",
}


# ---------------------------------------------------------------------------
# mechanics
# ---------------------------------------------------------------------------

def flatten(obj: Any, path: str = "", out: Optional[Dict[str, Any]] = None,
            max_depth: int = 3) -> Dict[str, Any]:
    """Leaf paths of a nested result JSON. Lists are leaves — their length is the fact."""
    out = {} if out is None else out
    if isinstance(obj, dict):
        depth = path.count(".")
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            if isinstance(v, dict) and depth < max_depth:
                flatten(v, p, out, max_depth)
            else:
                out[p] = v
    else:
        out[path] = obj
    return out


def match(key: str) -> Optional[Rule]:
    """The FIRST rule whose pattern claims `key`.

    Order is significance: a specific rule is written above the catch-all it must beat
    (`ablation_within_batch.*.mean_d_crps` before `ablation_within_batch.*`).

    A rule also claims everything BELOW the path it names, which is why `d_crps_clustered` needs one
    line rather than fifteen — the clustered CI is one object with fifteen fields, and dropping it
    drops all fifteen for one reason. `*` matches across dots, so the patterns are paths, not
    globs over a single segment.
    """
    for r in RULES:
        if fnmatch.fnmatchcase(key, r.old) or fnmatch.fnmatchcase(key, r.old + ".*"):
            return r
    return None


def cover(run: Dict[str, Any]) -> Tuple[Dict[str, Rule], List[str]]:
    """Every eval.py leaf key -> its rule. The second element is what nothing claimed."""
    leaves = {}
    for blk in ("M1", "M2", "M3", "S14"):
        if blk in run:
            leaves.update(flatten({blk: run[blk]}))
    claimed, orphan = {}, []
    for k in sorted(leaves):
        r = match(k)
        (claimed.__setitem__(k, r) if r else orphan.append(k))
    return claimed, orphan


def _get(d: Dict[str, Any], dotted: str) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def report(run: Dict[str, Any], bench: Dict[str, Any]) -> str:
    claimed, orphan = cover(run)
    if orphan:
        raise SystemExit(
            f"{len(orphan)} eval.py key(s) are claimed by no rule, so the report would be "
            f"incomplete and the D15 gate cannot be met. Add a rule for each:\n  "
            + "\n  ".join(orphan[:40]))

    leaves = flatten({b: run[b] for b in ("M1", "M2", "M3", "S14") if b in run})
    by_verdict: Dict[str, List[Tuple[str, Rule, Any, Any]]] = {
        v: [] for v in ("same", "moved", "replaced", "dropped")}
    for k, r in claimed.items():
        new_val = None
        if r.new and "*" not in r.new:
            new_val = _get(bench, r.new)
        by_verdict[r.verdict].append((k, r, leaves.get(k), new_val))

    L: List[str] = []
    L.append("# t22 — `candi.eval` -> `candi.bench`, key by key\n")
    L.append(f"**Checkpoint** `{run.get('config', {}).get('tag', '?')}` — the SAME weights on both "
             f"sides. `train.py` writes the checkpoint (line 1300) and then calls `evaluate()` on "
             f"the model still in memory (line 1339), so nothing here is a model difference.\n")
    L.append(f"**{len(claimed)} eval.py leaf keys**, every one claimed by a rule. "
             + ", ".join(f"{v}: {len(by_verdict[v])}" for v in by_verdict) + "\n")
    L.append("## The one mechanism that explains most of this table\n")
    L.append(f"{SCOPE.capitalize()}.\n\nA key marked `moved` has the same formula and a different "
             f"population. Read every `moved` delta as a change of what the number is a mean over, "
             f"never as the model getting better or worse.\n")

    for v, title in (("dropped", "Dropped — no counterpart"),
                     ("replaced", "Replaced — a different instrument, the same question"),
                     ("moved", "Moved — same formula, different population"),
                     ("same", "Same — identical definition and population")):
        rows = by_verdict[v]
        if not rows:
            continue
        L.append(f"\n## {title} ({len(rows)})\n")
        L.append("| eval.py key | bench key | old | new | why |")
        L.append("|---|---|---|---|---|")
        seen = set()
        for k, r, old, new in rows:
            if r.old in seen:
                continue                       # one row per RULE, not per expanded key
            seen.add(r.old)
            fam = f"`{r.old}`" if "*" in r.old else f"`{k}`"
            L.append(f"| {fam} | {('`' + r.new + '`') if r.new else '—'} | "
                     f"{_fmt(old)} | {_fmt(new)} | {r.why} |")

    L.append("\n## What bench adds that eval.py never had\n")
    L.append("A cutover report that tabulates only what went missing reads as a regression. "
             "These have no old counterpart because there was nothing there.\n")
    for k, why in NEW_BLOCKS.items():
        L.append(f"- **{k}** — {why}")

    L.append("\n## Open items — the two entries that should be argued, not accepted\n")
    L.append("1. **The target-clustered bootstrap CIs are gone.** `M2.*.d_crps_clustered`, "
             "`direction_clustered`, `overall_clustered`, `single_clustered`, `paired_clustered`. "
             "C1's randomization test gives an exact p-value and needs no cluster correction, so "
             "nothing is *wrong* — but an interval is not a p-value, and `_cluster_bootstrap_ci` "
             "is still imported by `compare_arms.py` and `report_h74.py`. Whatever replaces it "
             "must live somewhere those two can reach.")
    L.append("2. **Clamp telemetry has no counterpart.** `log2_mu` is clamped in the decoder head, "
             "and `M2.depth.*_clamp*` reported how often the depth sweep drove it into the clamp. "
             "That is how a flat depth response can have a cause other than the model ignoring "
             "depth. C3 reports monotonicity and dose-response correlation and would read a "
             "clamp-saturated model as unresponsive without saying why.")
    return "\n".join(L)


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, list):
        return f"list[{len(v)}]"
    if isinstance(v, dict):
        return f"dict[{len(v)}]"
    return str(v)


def main(argv: List[str]) -> int:
    if not argv or argv[0] not in ("map", "cover", "report"):
        print(__doc__)
        return 2
    cmd = argv[0]
    if cmd == "map":
        for r in RULES:
            print(f"{r.verdict:9s} {r.old:52s} -> {r.new or '—'}")
        print(f"\n{len(RULES)} rules")
        return 0
    run = json.loads(Path(argv[1]).read_text())
    if cmd == "cover":
        claimed, orphan = cover(run)
        print(f"{len(claimed)} eval.py leaf keys claimed by {len(RULES)} rules")
        for v in ("same", "moved", "replaced", "dropped"):
            print(f"  {v:9s} {sum(1 for r in claimed.values() if r.verdict == v)}")
        if orphan:
            print(f"\nUNCLAIMED ({len(orphan)}) — the D15 gate is not met:")
            for k in orphan[:40]:
                print(f"  {k}")
            return 1
        print("\nevery key is accounted for")
        return 0
    bench = json.loads(Path(argv[2]).read_text())
    text = report(run, bench)
    if "--out" in argv:
        out = Path(argv[argv.index("--out") + 1])
        out.write_text(text)
        print(f"wrote {out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
