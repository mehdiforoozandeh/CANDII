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
account for most of the report. `eval.py` scored a subsample, and the subsampling had two stages
that are easy to conflate: the eval loader advances one `(T_, V_/B_)` pair per window batch, so a
track receives one window in every `n_pairs` and the pooled arrays are already a strided slice of
the chromosome; `--eval-budget` then cut *that* down further, but only if it was still larger.
**On a small panel the budget does not bind at all** and the whole difference is the striding —
the report computes the fraction from the run rather than assuming either stage did the work.
`candi.bench` scores **every 25 bp bin** (D2). So a key tagged `moved` has not changed its formula;
it has changed what it is a mean over. Nothing here attributes a delta to a model change, because
the model is the same checkpoint on both sides.
"""
from __future__ import annotations

import fnmatch
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCOPE = ("position scope: eval.py scored whatever its loader's batches happened to cover -- the "
         "eval loader advances one (T_, V_/B_) pair per window batch, so a track receives one "
         "window in every n_pairs -- and then subsampled THAT to --eval-budget positions if it "
         "was still larger; bench scores every 25 bp bin of the eval chromosome (D2)")
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
    Rule("M1.imp_per_target.*.crps", "moved", "per_track.*.count.crps",
         f"{SCOPE}. The track key is the same string on both sides "
         f"(`input|target|assay`), so this is one track's number against the same track's."),
    Rule("M1.imp_per_target.*.crps_oracle_scaled", "moved", "per_track.*.count.crps_oracle_scaled",
         SCOPE),
    Rule("M1.imp_per_target.*.scale_error", "moved", "per_track.*.count.scale_error", SCOPE),
    Rule("M1.imp_per_target.*.marg_crps", "moved", "per_track.*.count.marg_crps", SCOPE),
    Rule("M1.imp_per_target.*.marg_mu", "moved", "per_track.*.count.marg_mu", SCOPE),
    Rule("M1.imp_per_target.*.marg_n", "moved", "per_track.*.count.marg_n", SCOPE),
    Rule("M1.imp_per_target.*.beats_marginal", "moved", "per_track.*.count.beats_marginal", SCOPE),
    Rule("M1.imp_per_target.*.beats_marginal_oracle_scaled", "moved",
         "per_track.*.count.beats_marginal_oracle_scaled", SCOPE),
    Rule("M1.imp_per_target.*.c_star", "moved", "per_track.*.count.c_star", SCOPE),
    Rule("M1.imp_per_target.*.n_star_log2", "moved", "per_track.*.count.n_star_log2", SCOPE),
    Rule("M1.imp_per_target.*.crps_oracle_scaled_and_n", "moved",
         "per_track.*.count.crps_oracle_scaled_and_n", SCOPE),
    Rule("M1.imp_per_target.*.n_points", "moved", "per_track.*.count.n_points",
         "bench scores every bin, so this rises to the chromosome's bin count"),
    Rule("M1.imp_per_target.*.spearman_raw", "moved", "per_track.*.count.gwspear",
         f"renamed to the challenge's own spelling, same statistic. {SCOPE}"),
    Rule("M1.imp_per_target.*.pearson_log1p", "replaced", "per_track.*.count.gwcorr",
         "gwcorr is Pearson on the RAW track, which is what score_metrics.py computes and what the "
         "published table reports. log1p Pearson is a different number and is not comparable to it; "
         "the E-block must match the organizers' code (D16), so the raw form wins."),
    Rule("M1.imp_per_target.*.r2", "dropped", None,
         "r2 on raw counts is a monotone function of gwcorr under a fitted intercept and carried no "
         "decision the correlation did not; it was never quoted in AGENTS.md §7"),
    Rule("M1.imp_per_target.*.mse_log", "replaced", "per_track.*.count.mse",
         "the E-block's `mse` is on the untransformed track, because normalize_dict is a no-op in "
         "score_metrics.py and the published numbers are untransformed (D16, quirk 5). mse_log and "
         "mse are not comparable and the report must not present them as a delta."),
    Rule("M1.imp_per_target.*.median_mu", "dropped", None,
         "a diagnostic of the marginal fit, superseded by marg_mu / marg_n which are the fitted "
         "parameters themselves"),
    Rule("M1.imp_per_target.*.median_target", "dropped", None, "a diagnostic of the marginal fit, superseded by marg_mu / marg_n, as median_mu"),
    Rule("M1.imp_per_target.*.marg_crps_legacy_median", "dropped", None,
         "the SUPERSEDED marginal: mu = median(target) + 1e-6. On 4 of 8 assays the median count is "
         "0, so the legacy bar was a near-zero constant and beating it meant nothing. It shipped "
         "beside the CRPS-optimal marginal only to show the gap; with the old suite gone there is "
         "nothing left for it to be a gap against."),
    Rule("M1.imp_per_target.*.marg_mu_legacy_median", "dropped", None, "the superseded legacy-median marginal's fitted mu; it goes with marg_crps_legacy_median"),

    # The middle level. eval.py had THREE: per target, per assay, per panel. bench has two.
    Rule("M1.imp_per_assay.*", "dropped", None,
         "THE PER-ASSAY LEVEL IS GONE, not the numbers in it. eval.py averaged an assay's score "
         "over every cell that carried it, and that average discards the cell -- which is the "
         "replication unit (D4 makes the per-track score the primitive and the mean over TRACKS "
         "the headline). Every value this level averaged is in `per_track`, under a key that names "
         "the cell, so a reader who wants a per-assay mean can still take one and will be able to "
         "say what it is a mean over. The 20 field-level rules above apply unchanged at that level."),

    Rule("M1.imp.crps", "moved", "macro.count.crps", f"{SCOPE}. {POOL}"),
    Rule("M1.imp.ece", "moved", "macro.count.ece", f"{SCOPE}. {POOL}"),
    Rule("M1.imp.calib_grid", "same", "per_track.*.count.calib_grid",
         "the 21-point non-randomised PIT grid is a constant, not a measurement"),
    Rule("M1.imp.calib_fbar", "moved", "per_track.*.count.calib_fbar",
         f"{SCOPE}; and it is now per track rather than pooled over the panel"),
    Rule("M1.imp.n_points", "moved", "macro.count.n_points", SCOPE),
    Rule("M1.imp.spearman_raw", "moved", "macro.count.gwspear", f"{SCOPE}. {POOL}"),
    Rule("M1.imp.pearson_log1p", "replaced", "macro.count.gwcorr", "as the per-assay form of this key, for the reason given there"),
    Rule("M1.imp.r2", "dropped", None, "as the per-assay form of this key, for the reason given there"),

    Rule("M1.imp_macro_crps", "moved", "macro.count.crps", f"{SCOPE}. {POOL}"),
    Rule("M1.imp_macro_crps_oracle_scaled", "moved", "macro.count.crps_oracle_scaled",
         f"{SCOPE}. {POOL}"),
    Rule("M1.imp_macro_scale_error", "moved", "macro.count.scale_error", f"{SCOPE}. {POOL}"),
    Rule("M1.imp_macro_marg_crps", "moved", "macro.count.marg_crps", f"{SCOPE}. {POOL}"),
    Rule("M1.imp_macro_spearman_raw", "moved", "macro.count.gwspear", f"{SCOPE}. {POOL}"),
    Rule("M1.imp_macro_pearson_log1p", "replaced", "macro.count.gwcorr", "as the per-assay form of this key, for the reason given there"),
    Rule("M1.imp_beats_marginal_n", "moved", "macro.count.beats_marginal",
         "a COUNT of assays becomes a FRACTION of tracks, because the macro means over tracks and a "
         "bool means to its rate. Read it as a proportion, never as the old integer."),
    Rule("M1.imp_beats_marginal_oracle_scaled_n", "moved",
         "macro.count.beats_marginal_oracle_scaled", "a count of assays becomes a fraction of tracks, as imp_beats_marginal_n"),
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
    Rule("M2.ablation.*.mean_abs_d_eta", "dropped", None, "an unnormalised effect size, as mean_abs_d_mu; C2's Shapley share replaces it"),
    Rule("M2.ablation.*.max_abs_d_eta", "dropped", None, "an unnormalised effect size, as mean_abs_d_mu; C2's Shapley share replaces it"),
    Rule("M2.ablation.*.n_sentinel_skipped", "dropped", None,
         "bookkeeping for the cross-target substitution's sentinel guard; bench's resamplers draw "
         "from the observed covariate rows, so a sentinel is never manufactured"),
    Rule("M2.ablation.*.per_target", "dropped", None,
         "the per-target record backed the clustered CI; without the CI it has no consumer"),
    Rule("M2.ablation.*.covariate", "same", "C.C1_use.*", "the covariate name is the KEY in bench, not a value inside it"),
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
    Rule("M2.depth.total_slope_err", "dropped", None, "the standard error of a retired statistic; C3 reports monotonicity instead"),
    Rule("M2.depth.total_slope_clamp_saturated", "dropped", None, "clamp telemetry, as frac_targets_any_clamp — see the report's Open Items"),
    Rule("M2.depth.frac_targets_any_clamp", "dropped", None,
         "CLAMP TELEMETRY HAS NO COUNTERPART. `log2_mu` is clamped in the decoder head, and these "
         "keys said how often the sweep drove it into the clamp — which is how a depth response can "
         "look flat for a reason that is not the model ignoring depth. Flagged in Open Items."),
    Rule("M2.depth.median_frac_log2mu_at_clamp", "dropped", None, "clamp telemetry, as frac_targets_any_clamp — see the report's Open Items"),
    Rule("M2.depth.max_frac_log2mu_at_clamp", "dropped", None, "clamp telemetry, as frac_targets_any_clamp — see the report's Open Items"),
    Rule("M2.depth.p90_frac_log2mu_at_clamp", "dropped", None, "clamp telemetry, as frac_targets_any_clamp — see the report's Open Items"),
    Rule("M2.depth.direction_clustered.*", "dropped", None,
         "as M2.ablation.*.d_crps_clustered — the clustered CI has no counterpart"),
    Rule("M2.depth.per_target", "dropped", None, "the per-target record existed to back the clustered CI; without it there is no consumer"),
    Rule("M2.depth.n_targets", "moved", "C.n_units", "targets become units, as M2.ablation.*.n_targets"),
    Rule("M2.depth.covariate", "same", "C.C1_use.depth", "the covariate name is the KEY in bench, not a value inside it"),

    Rule("M2.run_type.mean_responsiveness", "replaced", "C.C1_use.run_type.marginal_mean_d_crps",
         "responsiveness to a run_type flip becomes the covariate's effect size under the same "
         "null every other covariate is measured under, so the four are finally commensurable"),
    Rule("M2.run_type.model_unresponsive", "replaced", "C.C1_use.run_type.uses_covariate",
         "the same verdict with its sign flipped and a p-value behind it"),
    Rule("M2.run_type.*_clustered.*", "dropped", None,
         "as M2.ablation.*.d_crps_clustered — the clustered CIs have no counterpart"),
    Rule("M2.run_type.per_target", "dropped", None, "the per-target record existed to back the clustered CIs; without them there is no consumer"),
    Rule("M2.run_type.n_targets", "moved", "C.n_units", "targets become units, as M2.ablation.*.n_targets"),
    Rule("M2.run_type.covariate", "same", "C.C1_use.run_type", "the covariate name is the KEY in bench, not a value inside it"),
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
    Rule("M3.within", "replaced", "C.C5_C6_invariance.batch_asw", "replaced together with M3.ratio; the cosine-distance family goes as one (D12)"),
    Rule("M3.between", "replaced", "C.C5_C6_invariance.ilisi", "replaced together with M3.ratio; the cosine-distance family goes as one (D12)"),
    Rule("M3.invariance_ok", "replaced", "C.C5_C6_invariance.invariance_ok",
         "same name, DIFFERENT RULE: the old one was `ratio < 0.3 and eff_rank > 1.0`; the new one "
         "requires kBET rejection < 0.25, batch ASW > 0.8 AND bio_silhouette > 0.25 — the last of "
         "which is the C6 guard a collapsed encoder fails (D13)"),
    Rule("M3.encoder_eff_rank_pooled", "moved", "C.C5_C6_invariance.encoder_eff_rank",
         "kept, but demoted: it is far too weak a guard on its own, since a latent with two "
         "directions clears `> 1.0` while being nearly collapsed"),
    Rule("M3.n_regions", "moved", "C.C5_n_latents", "regions become pooled latent vectors"),
    Rule("M3.n_between_pairs", "dropped", None, "the pair count of a retired statistic; nothing in C5 is built from cosine pairs"),
)

S14_RULES: Tuple[Rule, ...] = (
    # NOT `moved`. This rule used to claim "same definition, same calibration" and the run refuted
    # both halves of that in one line: eval.py 0.25 against bench 1.00, where 0.73 was supposed to
    # be the ceiling.
    Rule("S14.frac_min_at_true", "replaced", "C.C3_depth_counterfactual.frac_min_at_true",
         "THE UNIT CHANGED AND SO DID THE CALIBRATION. eval.py's denominator is the TARGET -- 12 "
         "held-out tracks, each asked whether its argmin over told-depths lands on the true one, "
         "and each scored on a FOREGROUND mask (the top 2% of the level-k realization). bench "
         "pools every C-block unit into one array per level and asks the question once per LEVEL, "
         "four times, on the whole track. Two consequences a reader must not miss. First, the "
         "denominator went from 12 to 4, so the new number moves in quarters. Second, the ~0.73 "
         "ceiling was a consequence of the foreground restriction, and with no foreground there is "
         "no such ceiling -- which is why bench can report 1.0, a value the old instrument could "
         "not produce. The `0.25` coincidence is a trap: in eval.py it is the deterministic value "
         "of always answering told-depth 1, in bench it is 1/4 levels. The two numbers may not be "
         "differenced, and this report shows no delta for them."),
    Rule("S14.frac_beats_told1", "replaced", "C.C3_depth_counterfactual.frac_beats_told1",
         "per LEVEL over four levels on the whole track, not per TARGET over twelve on a "
         "foreground mask -- as frac_min_at_true, and undifferenceable for the same reason"),
    Rule("S14.n_targets", "moved", "C.C3_depth_counterfactual.n_levels",
         "the unit is the DSF level being scored, not the target"),
    Rule("S14.per_target", "moved", "C.C3_depth_counterfactual.crps_at_true_level",
         "per-level CRPS rather than per-target"),
    Rule("S14.dsf_counterfactual_ok", "dropped", None,
         "a boolean over frac_min_at_true against a threshold that was never registered; C3 ships "
         "the number and its two calibrations instead of a verdict nobody agreed"),
)

def denoise_twin(r: Rule) -> Optional[Rule]:
    """The denoising rule implied by an imputation rule.

    Derived rather than written, because the two halves are the same measurement on a different
    target and a hand-written second copy would drift — which is exactly what `eval.py` did, where
    `den_per_assay` and `imp_per_assay` were assembled by two near-identical code paths.

    Denoising is opt-in in bench (`--kinds denoise`) and its track keys carry a fourth field, so
    every bench-side path is rewritten to the denoise track and the denoise macro.
    """
    old = r.old.replace("M1.imp_per_target.", "M1.den_per_target.") \
               .replace("M1.imp_per_assay.", "M1.den_per_assay.") \
               .replace("M1.imp.", "M1.den.").replace("M1.imp_macro_", "M1.den_macro_")
    if old == r.old:
        return None                      # not an imputation rule; nothing to mirror
    new = None if r.new is None else (
        r.new.replace("per_track.*.count.", "per_track.*|denoise.count.")
             .replace("macro.count.", "macro_denoise.count."))
    return Rule(old, r.verdict, new, r.why + " [the denoising half, derived from the imputation "
                                             "rule so the two cannot drift]")


#: Imputation rules first: an imputation key must never be claimed by a denoise pattern or the
#: other way round, and both families are disjoint by prefix, so order is safety rather than
#: significance here.
M1_RULES = M1_RULES + tuple(t for t in (denoise_twin(r) for r in M1_RULES) if t is not None)

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


#: `AGENTS.md` §7.2, quoted with every number by rule. The floor governs comparing two RUNS; both
#: sides of this report come from ONE checkpoint, so a delta here is a measurement difference and
#: not a model difference. It is quoted anyway, because it is the scale that says which of these
#: measurement differences would matter downstream: a key that moves by less than 0.09 will be
#: invisible under the noise any real arm-vs-arm comparison already carries, and one that moves by
#: more will not be.
NOISE_FLOOR = {
    "macro_crps_target_clustered": 0.09,
    "per_comparison_uncertainty": 0.13,
    "seed_change_moves_pooled_imp_crps": 0.1195,
    "seed_change_moves_spearman": 0.0562,
    "seed_change_moves_ece": 0.0354,
    "effective_replication": "12 held-out targets / 5 biosample pairs / 4 cell types",
}

#: The rows a reader actually looks at: one old path, one new path, one number each. Everything
#: else in the report is structure; this is the comparison.
#:
#: `crps` NEVER appears without `crps_oracle_scaled` and `scale_error` beside it (`AGENTS.md` §7.2),
#: which is why they are adjacent here and why the renderer refuses to emit one without the others.
HEADLINE: Tuple[Tuple[str, ...], ...] = (
    ("imputation macro CRPS", "M1.imp_macro_crps", "macro.count.crps"),
    ("  ... oracle-scaled", "M1.imp_macro_crps_oracle_scaled", "macro.count.crps_oracle_scaled"),
    ("  ... scale error", "M1.imp_macro_scale_error", "macro.count.scale_error"),
    ("  ... marginal bar", "M1.imp_macro_marg_crps", "macro.count.marg_crps"),
    ("imputation macro Spearman", "M1.imp_macro_spearman_raw", "macro.count.gwspear"),
    ("imputation pooled ECE", "M1.imp.ece", "macro.count.ece",
     "not like-for-like: eval.py pooled every position of every track into ONE calibration curve, "
     "so a deep track dominated it; bench calibrates each track and means the tracks (D4)."),
    ("denoise macro CRPS", "M1.den_macro_crps", "macro_denoise.count.crps"),
    ("  ... oracle-scaled", "", "macro_denoise.count.crps_oracle_scaled"),
    ("  ... scale error", "", "macro_denoise.count.scale_error"),
    ("denoise macro Spearman", "M1.den_macro_spearman_raw", "macro_denoise.count.gwspear"),
    ("S14 frac_min_at_true", "S14.frac_min_at_true",
     "C.C3_depth_counterfactual.frac_min_at_true",
     "NO DELTA: different denominators. eval.py asks the question once per TARGET (12) on a "
     "foreground mask; bench asks it once per LEVEL (4) on the whole track. The old instrument "
     "capped near 0.73 *because of* the foreground; the new one has no such cap."),
    ("S14 frac_beats_told1", "S14.frac_beats_told1",
     "C.C3_depth_counterfactual.frac_beats_told1",
     "NO DELTA: per level over four, not per target over twelve. See the row above."),
    ("C1 tripwire (must be 0.0)", "M2.ablation_within_batch.depth.mean_d_crps",
     "C.C1_use.depth.within_batch_d_crps"),
    ("scored positions", "M1.imp.n_points", "macro.count.n_points",
     "DIFFERENT FOOTINGS, so no delta is shown. eval.py's is the POOLED length of the concatenated "
     "arrays over every imputation track, after `--eval-budget` subsampling. bench's is the mean "
     "PER TRACK, and equals the eval chromosome entire. Divided out, the two give the position "
     "scope as a fraction -- see under the table."),
)

#: Emitted with no old counterpart, because there was none. The nine are the point of the whole
#: suite: without them CANDI cannot be placed against the published field at all.
HEADLINE_NEW: Tuple[Tuple[str, str], ...] = (
    ("EIC mse", "macro.count.mse"), ("EIC gwcorr", "macro.count.gwcorr"),
    ("EIC gwspear", "macro.count.gwspear"), ("EIC mseprom", "macro.count.mseprom"),
    ("EIC msegene", "macro.count.msegene"), ("EIC mseenh", "macro.count.mseenh"),
    ("EIC mse1obs", "macro.count.mse1obs"), ("EIC mse1imp", "macro.count.mse1imp"),
    ("pval arm CRPS", "macro.pval.crps"), ("pval arm gwcorr", "macro.pval.gwcorr"),
    ("pval arm mse", "macro.pval.mse"), ("C-index (count)", "macro.count.c_index"),
    ("  ... its Monte-Carlo SE", "macro.count.c_index_se"),
    ("95% coverage (count)", "macro.count.coverage_95"),
    ("AUPRC", "macro.count.auprc"), ("peak base rate", "macro.count.peak_base_rate"),
)


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
    cfg = run.get("config", {})
    L.append("**What this run is, and what it is not.** "
             f"{cfg.get('n_params', '?'):,} parameters "
             f"(d_model {cfg.get('d_model', '?')}, {cfg.get('n_transformer_layers', '?')} layers), "
             f"trained {cfg.get('epochs', '?')} epochs on {cfg.get('train_chroms', '?')} of "
             f"`{Path(str(cfg.get('h5', '?'))).name}` and scored on {cfg.get('eval_chroms', '?')}, "
             f"{cfg.get('num_assays', '?')} assays. That is the smoke-scale recipe, not CANDI's "
             "~2.35 M-parameter production architecture. **No number in this document is a "
             "statement about how well CANDI imputes.** The document's claim is narrower and does "
             "not depend on the model being good: for each key, where it went and by what "
             "mechanism. A wrong-sized model would break that claim only if a key were missing, "
             "and a missing key is exactly what the coverage gate refuses.\n"
             if cfg else "")
    # Keys AND rules. A per-assay panel of 8 assays x 20 fields x 2 halves is 320 keys and one
    # decision; quoting only the key count makes a tidy cutover read like a massacre.
    n_rules = {v: len({r.old for _, r, _, _ in by_verdict[v]}) for v in by_verdict}
    L.append(f"**{len(claimed)} eval.py leaf keys**, every one claimed by a rule, under "
             f"{sum(n_rules.values())} rules:\n")
    L.append("| verdict | rules | keys |")
    L.append("|---|---:|---:|")
    for v in by_verdict:
        L.append(f"| `{v}` | {n_rules[v]} | {len(by_verdict[v])} |")
    L.append("\nRead the RULES column. One rule is one decision; the key count is how many "
             "cells that decision covers, and a panel of 8 assays x 20 fields x 2 halves is 320 "
             "keys and a single call.\n")
    L.append("## The mechanism most keys moved under\n")
    # NOT str.capitalize(): it lowercases everything after the first character, which turns
    # "(T_, V_/B_)" into "(t_, v_/b_)" and "(D2)" into "(d2)".
    L.append(f"{SCOPE[0].upper()}{SCOPE[1:]}.\n\nA key marked `moved` has the same formula and a different "
             f"population. Read every `moved` delta as a change of what the number is a mean over, "
             f"never as the model getting better or worse.\n")
    L.append("This is what most KEYS moved under. It is **not** what moved the headline macro -- "
             "that turns out to be the pooling unit, and the two are separated with arithmetic "
             "under the headline table rather than left to the reader's assumption.\n")
    budget = cfg.get("eval_budget")
    pooled_n = _get(run, "M1.imp.n_points")
    if budget and pooled_n:
        if pooled_n < budget:
            L.append(f"**On this run `--eval-budget` did not bind.** It was {budget:,} against "
                     f"{pooled_n:,} pooled positions, so nothing was thrown away at that stage and "
                     f"the entire scope difference is the loader's striding. Read the ratio below "
                     f"as a property of how the eval loader batches, not of a budget flag.\n")
        else:
            L.append(f"**On this run `--eval-budget` did bind**: {budget:,} against "
                     f"{pooled_n:,} pooled positions, so both stages contributed.\n")

    L.append("\n## Headline — the same checkpoint, both suites\n")
    L.append("| | eval.py | bench | delta |")
    L.append("|---|---|---|---|")
    for row in HEADLINE:
        label, o, n = row[0], row[1], row[2]
        note = row[3] if len(row) > 3 else None
        ov = _get(run, o) if o else None
        nv = _get(bench, n)
        if note:
            d = note                          # a delta between two different footings is a lie
        else:
            d = (f"{nv - ov:+.6g}" if isinstance(ov, (int, float)) and isinstance(nv, (int, float))
                 and not isinstance(ov, bool) and not isinstance(nv, bool) else "—")
        L.append(f"| {label} | {_fmt(ov)} | {_fmt(nv)} | {_cell(d)} |")
    # The scope mechanism as a number, computed rather than asserted: the whole report leans on it.
    pooled = _get(run, "M1.imp.n_points")
    n_tracks = len(_get(run, "M1.imp_per_target") or {})
    per_track_new = _get(bench, "macro.count.n_points")
    if pooled and n_tracks and per_track_new:
        old_per_track = pooled / n_tracks
        L.append(f"\n**The position scope, divided out.** {pooled:,.0f} pooled positions over "
                 f"{n_tracks} imputation tracks is **{old_per_track:,.0f} per track**, against "
                 f"bench's **{per_track_new:,.0f}**. The old suite scored "
                 f"**{100 * old_per_track / per_track_new:.1f}%** of the eval chromosome. That "
                 f"single ratio is what most of the `moved` column is a consequence of.\n")

    L.append(f"\n**Read every delta against the noise floor** (`AGENTS.md` §7.2). "
             f"Target-clustered floor on macro CRPS is **{NOISE_FLOOR['macro_crps_target_clustered']}**; "
             f"per-comparison uncertainty **±{NOISE_FLOOR['per_comparison_uncertainty']}**; a "
             f"single SEED change moves pooled imputation CRPS by "
             f"**{NOISE_FLOOR['seed_change_moves_pooled_imp_crps']}**, Spearman by "
             f"{NOISE_FLOOR['seed_change_moves_spearman']}, ECE by "
             f"{NOISE_FLOOR['seed_change_moves_ece']}. Effective replication is "
             f"{NOISE_FLOOR['effective_replication']}.\n\nBoth columns are the SAME checkpoint, so "
             f"a delta here is a measurement difference and never a model difference. The floor is "
             f"quoted because it is the scale that decides which of these measurement differences "
             f"would survive contact with a real arm-vs-arm comparison.\n")

    # Track by track. The macro is a mean of these, and a mean hides whether the scope change
    # moved every track the same way or moved two of them a long way. Same key on both sides.
    per_new = _get(bench, "per_track") or {}
    for arm_label, old_block, suffix in (("imputation", "M1.imp_per_target", ""),
                                         ("denoising", "M1.den_per_target", "|denoise")):
      per_old = _get(run, old_block) or {}
      rows_t = []
      for key in sorted(per_old):
        # bench's denoise track keys carry a fourth field; eval.py's do not (t20 decision 2).
        o, n = per_old[key], (per_new.get(key + suffix) or {}).get("count") or {}
        if not n:
            continue
        rows_t.append((key, o, n))
      if rows_t:
        L.append(f"\n## Track by track — the {arm_label} arm ({len(rows_t)} tracks)\n")
        L.append("The macro is a mean of these. A mean cannot show whether the scope change "
                 "moved every track alike or moved two of them a long way, and that difference "
                 "decides whether the macro delta is a property of the measurement or of two "
                 "tracks. Same track key on both sides.\n")
        L.append("| track | CRPS old | CRPS new | d | oracle-scaled old | new | Spearman old | new |")
        L.append("|---|---|---|---|---|---|---|---|")
        for key, o, n in rows_t:
            def d2(a, b):
                return (f"{b - a:+.4g}" if isinstance(a, (int, float))
                        and isinstance(b, (int, float)) else "—")
            L.append(f"| {_cell('`' + key + '`')} | {_fmt(o.get('crps'))} | "
                     f"{_fmt(n.get('crps'))} | {d2(o.get('crps'), n.get('crps'))} | "
                     f"{_fmt(o.get('crps_oracle_scaled'))} | {_fmt(n.get('crps_oracle_scaled'))} | "
                     f"{_fmt(o.get('spearman_raw'))} | {_fmt(n.get('gwspear'))} |")
        deltas = [n.get("crps") - o.get("crps") for _, o, n in rows_t
                  if isinstance(o.get("crps"), (int, float))
                  and isinstance(n.get("crps"), (int, float))]
        if deltas:
            L.append(f"\nAcross {len(deltas)} tracks the CRPS shift runs "
                     f"{min(deltas):+.4g} to {max(deltas):+.4g}, median "
                     f"{statistics.median(deltas):+.4g}. Compare that SPREAD against the noise "
                     f"floor below, not the median: a scope change that moves every track the same "
                     f"way is a re-baselining, and one that moves two of them is a finding.\n")

    # THE decomposition. The macro delta has two candidate causes and they are separable from the
    # data already in hand: eval.py's own per-TRACK mean is the same pooling unit as bench's, so
    # differencing against it isolates position scope, and differencing it against eval.py's
    # per-ASSAY macro isolates the pooling unit. Without this the reader attributes the whole
    # headline delta to the mechanism named at the top of the report, and on this run that is
    # the wrong way round.
    for arm_label, old_per, old_assay, old_macro, new_macro in (
            ("imputation", "M1.imp_per_target", "M1.imp_per_assay",
             "M1.imp_macro_crps", "macro.count.crps"),
            ("denoising", "M1.den_per_target", "M1.den_per_assay",
             "M1.den_macro_crps", "macro_denoise.count.crps")):
        per = _get(run, old_per) or {}
        n_assays = len(_get(run, old_assay) or {})
        vals = [v["crps"] for v in per.values()
                if isinstance(v, dict) and isinstance(v.get("crps"), (int, float))]
        a_mean, n_mean = _get(run, old_macro), _get(bench, new_macro)
        if not vals or not isinstance(a_mean, (int, float)) or not isinstance(n_mean, (int, float)):
            continue
        t_mean = statistics.mean(vals)
        L.append(f"\n### Which mechanism actually moved the {arm_label} macro\n")
        L.append("`eval.py` records a per-track level of its own, so the headline delta splits "
                 "exactly rather than by argument. Its per-track mean uses bench's pooling unit on "
                 "eval.py's positions, which is the missing middle term.\n")
        L.append("| | CRPS | |")
        L.append("|---|---|---|")
        L.append(f"| eval.py, mean over {n_assays} ASSAYS | {a_mean:.5f} | the published headline |")
        L.append(f"| eval.py, mean over {len(vals)} TRACKS | {t_mean:.5f} | same positions, "
                 f"bench's pooling unit |")
        L.append(f"| bench, mean over {len(vals)} TRACKS | {n_mean:.5f} | the new headline |")
        L.append("")
        L.append(f"- **pooling unit** (assay-mean -> track-mean): **{t_mean - a_mean:+.5f}**")
        L.append(f"- **position scope** (subsample -> whole chromosome): "
                 f"**{n_mean - t_mean:+.5f}**")
        L.append(f"- total, as the headline reports it: {n_mean - a_mean:+.5f}\n")
        # How concentrated is the panel? That is what makes the two means different questions,
        # and it is a property of the run, not a sentence to be reused between arms.
        by_pair: Dict[str, int] = {}
        for k in per:
            by_pair[k.rsplit("|", 1)[0]] = by_pair.get(k.rsplit("|", 1)[0], 0) + 1
        top_pair, top_n = (max(by_pair.items(), key=lambda kv: kv[1]) if by_pair else ("", 0))
        seed_move = NOISE_FLOOR["seed_change_moves_pooled_imp_crps"]
        if abs(t_mean - a_mean) > abs(n_mean - t_mean):
            same_order = (abs(t_mean - a_mean) > seed_move / 3)
            L.append(f"**The pooling unit dominates; position scope does almost nothing.** "
                     f"Scoring the whole chromosome instead of the old slice moved "
                     f"this number by {abs(n_mean - t_mean):.5f}; changing what the mean is over "
                     f"moved it by {abs(t_mean - a_mean):.5f}"
                     + (f", which is of the same order as the {seed_move} a SEED change moves "
                        f"pooled imputation CRPS (`AGENTS.md` 7.2)" if same_order else
                        ", both of them far under any floor worth quoting")
                     + ". An assay-mean weights each assay once however many cells carried it; a "
                     f"track-mean weights each cell. Here {top_n} of {len(per)} {arm_label} tracks "
                     f"come from `{top_pair}`, so the two means are genuinely different questions "
                     f"-- and D4 settles which one is the headline.\n")
        else:
            L.append(f"Position scope dominates on this arm, as the mechanism note at the top "
                     f"predicts: {abs(n_mean - t_mean):.5f} against the pooling unit's "
                     f"{abs(t_mean - a_mean):.5f}.\n")

    L.append("\n## What has no old number, because there was none\n")
    L.append("| | bench |")
    L.append("|---|---|")
    for label, n in HEADLINE_NEW:
        L.append(f"| {label} | {_fmt(_get(bench, n))} |")
    L.append("")

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
        counts: Dict[str, int] = {}
        for _, r, _, _ in rows:
            counts[r.old] = counts.get(r.old, 0) + 1
        seen = set()
        for k, r, old, new in rows:
            if r.old in seen:
                continue                       # one row per RULE, not per expanded key
            seen.add(r.old)
            fam = f"`{r.old}`" if "*" in r.old else f"`{k}`"
            # A pattern ending in a BARE `*` claims a whole level, not one field, so the first key
            # it happens to claim carries no meaning: `M1.den_per_assay.*` would print the value of
            # `DNase-seq.beats_marginal`, which reads as the rule's number and is not one.
            if r.old.rsplit(".", 1)[-1] == "*":
                ov = f"{counts[r.old]} keys"
                nv = f"{counts[r.old]} keys" if r.new else "—"
            else:
                ov, nv = _fmt(old), _fmt(new)
            L.append(f"| {_cell(fam)} | {_cell(('`' + r.new + '`') if r.new else '—')} | "
                     f"{ov} | {nv} | {_cell(r.why)} |")

    L.append("\n## What bench adds that eval.py never had\n")
    L.append("A cutover report that tabulates only what went missing reads as a regression. "
             "These have no old counterpart because there was nothing there.\n")
    for k, why in NEW_BLOCKS.items():
        L.append(f"- **{k}** — {why}")

    # msevar is the one measure of the nine that needs an asset built from the corpus, so whether
    # the E-block is nine or eight is a property of the RUN, not of the suite. Read it off the
    # bench provenance rather than asserting nine and printing eight.
    mv = _get(bench, "provenance.msevar") or {}
    if mv.get("pool"):
        members = {k: v for k, v in mv.items() if k != "pool"}
        L.append(f"\n`msevar` IS present: pool `{mv['pool']}` (D7), "
                 f"{len(members)} assay pool(s) loaded. It weights squared error by the "
                 f"cross-biosample variance of that assay at that position, and the pool is built "
                 f"in pval space, so it may only ever weight the pval arm.\n")
    else:
        L.append("\n`msevar` is ABSENT from this run: no `--varpool` was given, so the E-block "
                 "here is **eight of the nine** measures. That is deliberate — the organizers' own "
                 "code returns a bare `0.0` with no variance vector, and a 0.0 is indistinguishable "
                 "from a perfect score in any table it lands in, so bench omits the key instead. "
                 "The pool is D7's and is built by `slurm/t18_varpool.sh` from the store's own "
                 "training biosamples.\n\n"
                 "Switching it on is not only a flag when the backend is a baked h5. A variance "
                 "pool is a vector on the CORPUS bin grid; an h5-backed run's grid is whatever the "
                 "bake tiled, which ends at the last whole window rather than at the end of the "
                 "chromosome. bench compares the two lengths and refuses on a mismatch rather than "
                 "weighting the wrong positions -- so a pool and a bake have to agree bin for bin, "
                 "and a bake that stops short needs that decision made before `msevar` can be "
                 "quoted from it.\n")

    L.append("\n## Accepted losses — put to the PI, and ruled on\n")
    L.append("Both were raised as capability the new suite does not carry, and both were accepted "
             "on 2026-08-21. They are recorded here rather than dropped from the report, because "
             "the difference between a loss that was decided and a loss nobody noticed is the "
             "whole reason this document exists.\n")
    L.append("1. **The target-clustered bootstrap CIs are gone.** `M2.*.d_crps_clustered`, "
             "`direction_clustered`, `overall_clustered`, `single_clustered`, `paired_clustered`. "
             "C1's randomization test gives an exact p-value and needs no cluster correction, so "
             "nothing here is wrong — but an interval is not a p-value, and the two are not "
             "interchangeable for an arm-vs-arm claim. **Accepted.** The statistic itself survives "
             "the cutover: it moved to `candi.stats.cluster_bootstrap_ci` ahead of it, so "
             "`compare_arms.py` and `report_h74.py` keep working after `eval.py` is deleted.")
    L.append("2. **Clamp telemetry has no counterpart.** `log2_mu` is clamped in the decoder head, "
             "and `M2.depth.*_clamp*` reported how often the depth sweep drove it into the clamp — "
             "which is how a flat depth response can have a cause other than the model ignoring "
             "depth. C3 reports monotonicity and dose-response correlation, and would read a "
             "clamp-saturated model as unresponsive without saying why. **Accepted**, with the "
             "consequence stated: a C3 near zero is 'no response', not 'no sensitivity'.")
    return "\n".join(L)


def _cell(text: str) -> str:
    """A markdown table cell. Bench's denoise track keys contain `|`, which would end the cell.

    `per_track.*|denoise.count.crps` inside backticks still splits the row in every renderer --
    code spans do not protect a pipe inside a table. It has to be escaped.
    """
    return str(text).replace("|", "\\|")


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


def strip_values(obj: Any) -> Any:
    """A run JSON with every measured value replaced by `null` and every list emptied.

    This is what `tests/fixtures/eval_key_skeleton.json` holds, and it is a COMMAND rather than a
    hand edit for a reason that already bit once: the fixture was first recorded from a run made
    before this repo existed, whose `eval.py` had no `imp_per_target` / `den_per_target` level.
    The coverage gate passed against it while 760 real keys -- 54% of the output -- were claimed by
    nothing at all. A fixture that can only be re-recorded by hand goes stale silently; one that is
    re-recorded by `python tools/equivalence.py skeleton <run.json> <out.json>` does not.

    Values are stripped because a number in `tests/` is a number outside the vault.
    """
    if isinstance(obj, dict):
        return {k: strip_values(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return []
    return None


def main(argv: List[str]) -> int:
    if not argv or argv[0] not in ("map", "cover", "report", "skeleton"):
        print(__doc__)
        return 2
    cmd = argv[0]
    if cmd == "map":
        for r in RULES:
            print(f"{r.verdict:9s} {r.old:52s} -> {r.new or '—'}")
        print(f"\n{len(RULES)} rules")
        return 0
    run = json.loads(Path(argv[1]).read_text())
    if cmd == "skeleton":
        body = {b: strip_values(run[b]) for b in ("M1", "M2", "M3", "S14") if b in run}
        body["_provenance"] = {
            "source": f"candi.train run {run.get('config', {}).get('tag', '?')}, "
                      f"{Path(run.get('config', {}).get('h5', '?')).name}, "
                      f"{run.get('config', {}).get('num_assays', '?')} assays",
            "what_this_is": "the KEY STRUCTURE of candi.eval's output and nothing else",
            "why_values_are_null": "every measured value is stripped. This fixture exists to hold "
                                   "tools/equivalence.py's rule table to a real key set; a recorded "
                                   "number belongs in cruxvault/, never in tests/, and AGENTS.md "
                                   "section 7 is frozen.",
            "recorded_by": "python tools/equivalence.py skeleton <run.json> <out.json>",
        }
        out = Path(argv[2])
        out.write_text(json.dumps(body, indent=1, sort_keys=True))
        print(f"wrote {out} ({len(flatten(body)) } leaf keys)")
        return 0
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
