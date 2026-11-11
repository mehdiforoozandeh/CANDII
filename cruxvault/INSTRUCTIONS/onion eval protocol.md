---
title: onion eval protocol
kind: background
parent: "[[peeling the onion]]"
siblings: ["[[onion caveats]]", "[[onion variation and objectives]]", "[[onion data plan]]", "[[miniCANDI design notes]]", "[[miniCANDI architecture ablations]]"]
created: 2026-11-10
status: background information; the headline metric set is the one PI decision in this file
---

# onion eval protocol

This file extends the "eval" section of [[peeling the onion]]. The first section is the headline metric set, which the PI chose on 2026-11-10. All other sections are background: methods, measurements and lessons that bear on the evaluation. Each entry was checked against its source on 2026-11-10. Source-path conventions are defined in [[onion caveats]].

## Index

- [headline-metric-set](#headline-metric-set) (PI decision)
- [max-error-decomposition-and-noise-floor](#max-error-decomposition-and-noise-floor)
- [crps-oracle-scaled-split](#crps-oracle-scaled-split)
- [nested-baseline-ladder](#nested-baseline-ladder)
- [r-spec-specificity-correlation](#r-spec-specificity-correlation)
- [resolution-question-prior-evidence](#resolution-question-prior-evidence)
- [resolution-study-open-points](#resolution-study-open-points)
- [calibration-battery](#calibration-battery)
- [predicted-vs-replicate-variance](#predicted-vs-replicate-variance)
- [region-strata-and-foreground-purity](#region-strata-and-foreground-purity)
- [value-strata](#value-strata)
- [broad-marks-top1-agreement](#broad-marks-top1-agreement)
- [identifiability-table](#identifiability-table)
- [cac-evaluation-test-sets](#cac-evaluation-test-sets)
- [cac-evaluation-baselines-and-controls](#cac-evaluation-baselines-and-controls)
- [cac-evaluation-gates](#cac-evaluation-gates)
- [cac-evaluation-twins](#cac-evaluation-twins)
- [truth-pipeline-swap](#truth-pipeline-swap)
- [score-on-normalized-and-unnormalized-truth](#score-on-normalized-and-unnormalized-truth)
- [inference-unit-and-seed-floors](#inference-unit-and-seed-floors)
- [metric-hygiene](#metric-hygiene)
- [peak-height-vs-shape](#peak-height-vs-shape)
- [crossed-variance-partition](#crossed-variance-partition)
- [within-locus-population-metric](#within-locus-population-metric)
- [magnitude-vs-shape-diagnostic](#magnitude-vs-shape-diagnostic)
- [instrument-hygiene](#instrument-hygiene)
- [fixed-random-window-scope](#fixed-random-window-scope)

---

## headline-metric-set

- **Status:** PI decision, 2026-11-10. These 7 metrics are the ones the PI monitors and the main manuscript reports. Every other metric in this file is background or supplementary.
- **Why a small set:** each extra metric adds a way to find a weakness. It also makes tracking, optimizing and interpreting harder.

| # | Goal | Metric | Definition | Implementation status (2026-11-10) |
|---|---|---|---|---|
| 1 | imputation | Recovered share of specific variance above the noise floor | 1 − (E_sp − N)/(V_sp − N), a ratio of target means. E_sp is the specific error, V_sp the specific truth variance, N the noise floor. | Max s06 `scripts/report2/aggregate2.py`; Max s07 `scripts/model/decomp_summary.py`. Both in asinh-p. Not in CANDII. |
| 2 | imputation | r_spec | corr(y − g(a), p − h(a)). See [r-spec-specificity-correlation](#r-spec-specificity-correlation). | Max s03 `scripts/metrics/metrics_lib.py::rspec`. Not in CANDII. |
| 3 | imputation | Count CRPS skill against the per-position average | CRPS(average) − CRPS(model), with each CRPS quoted as oracle_scaled + scale_error | The split exists in `CANDII: src/candi/bench/distributional.py::nb_suite`. No count baseline from the per-position average is scored yet. |
| 4 | CAC | Gap closed against depth-only scaling, on never-trained covariate pairs | (CRPS_depth_only − CRPS_model)/(CRPS_depth_only − CRPS_floor) | The raw gap is in `CANDII-worktree-t146: results/t150/REPORT.md`: 0.677 depth-only vs 0.527 learned, n = 2094 pairs. The floor term is not computed. |
| 5 | CAC | Depth-law error on never-trained pairs | Median of abs(log2 predicted total scale − log2 true depth ratio) | `CANDII-worktree-t146: results/h19/depth_pairs/analyze.py`. Earlier values: 0.19–0.55 never-trained vs 0.04–0.10 trained. |
| 6 | uncertainty | Non-randomized PIT deviation, pooled and worst stratum | Mean abs(F̄(u) − u) over a 21-point grid (Czado, Gneiting and Held 2009) | The per-track value exists in `CANDII: src/candi/metrics.py::calibration_pit_curve` (key `ece`). The pooled and worst-stratum versions are not implemented. |
| 7 | uncertainty | Predicted variance over replicate variance | Predicted NB variance divided by the variance between replicates, per stratum | Not implemented. The nearest prior work is Max s06 `summary2/vss.tsv` (Gaussian σ² vs sampling noise, in asinh-p). |

Every headline metric is quoted per assay as a macro average, with its seed band. Related: [crps-oracle-scaled-split](#crps-oracle-scaled-split), [nested-baseline-ladder](#nested-baseline-ladder).

## max-error-decomposition-and-noise-floor

Max's sprints 06 and 07 built an exact error budget. It is the closest existing tool to the "peeling" in [[peeling the onion]]. All of its numbers are in asinh(−log10 p) units of MACS2 p-values at 25 bp, on Max's Benchmark A v3: 282 held-out targets on chromosomes 6, 9, 13, 20, 21 and 22, from 40 ENCODE terms. See [[onion caveats#noise-floor-unit-rule]] before carrying any number into counts.

**Exact decomposition of one track.**
- The error is the MSE in units of the truth's variance, MSE/Var(x). It splits exactly as MSE/Var(x) = offset + scale + nonlinear + residual:
  - offset = (mean x − mean y)²/Var(x)
  - scale = (SD y − r·SD x)²/Var(x)
  - nonlinear = (1 − r²) − residual
  - residual = MSE(f(y))/Var(x)
- f(y) is an isotonic regression of the truth on the prediction, i.e. the best monotone map. It is a weighted pool-adjacent-violators fit on 1,600 quantile bins of y over quantiles 0–0.99, plus 400 bins over 0.99–1. It falls back to the best affine map if it does not improve on it.
- "Level" = offset + scale + nonlinear. Max reports that the identities hold to about 1e-14.

**Scale split.**
- The residual x − f(y) is standardized by SD(x). An orthonormal Haar transform then splits it into 14 detail levels. Level j compares the halves of blocks of 25·2^j bp, so the levels run from 50 bp to 409.6 kb. A coarse remainder makes 15 scales in all.
- The levels form four bands:

| Band | Scales | Levels |
|---|---|---|
| fine | 50–200 bp | 1–3 |
| peak | 0.4–3.2 kb | 4–7 |
| domain | 6.4–51.2 kb | 8–11 |
| broad | ≥ 102.4 kb | 12–14 plus the remainder |

**Shared and specific parts.**
- The model of the error is e_b = s + u_b. Here s is common to all held-out tracks of an assay, and u_b is track-specific with mean 0.
- At each scale, the shared part of track b is the mean Gram entry G[k,b,c] over tracks c from a different ENCODE term. This estimate is unbiased for the energy of s.
- specific_b = G[k,b,b] − shared_b.
- The naive squared mean residual would leak 1/B of each track's specific error into "shared". An assay with tracks from one term only gets no estimate.

**Four-way split of each component.**
- Round 2 applies the same split to the Gram matrices of the truth (V), the calibrated prediction (P) and the cross term (C).
- For each shared or specific band component:
  - explained = C²/P (0 if C ≤ 0, capped at V)
  - added = (P − C)²/P, i.e. error the prediction adds through misplaced variance
  - unexplainable = N, the sampling noise (specific parts only)
  - unexplained = V − explained − N
- So V = explained + unexplained + unexplainable, and E = unexplained + unexplainable + added.

**Noise floor.**
- floor = 1 − kR/(1 + (k − 1)R), the Spearman-Brown formula, per Haar level. Here R is the correlation between two equally noisy half-depth tracks, and k is the depth ratio. The per-level floors are weighted by the target's own variance spectrum.
- Sprint 07 found that the depth must be the pooled track's depth counted in DNA fragments. Sprint 06 had used one replicate's BAM record count. That is about half the pooled depth, and for paired-end data it is twice the fragment count.
- On six ENCODE experiments with real replicates, measured noise over predicted noise had a median of 1.05 in fragments and 1.42 in records.
- When the reads of Y are a subset of the reads of X, the floor is the conditional variance Var(Y | X) = E[(Y − D)²]/2. D is an independent draw from a pseudoreplicate half of the parent track. At 25 bp in counts, this floor is 0.30–0.85 of Var(Y).

**Headline numbers.**
- Best sprint 06 arm (design 3 + Flashzoi, 240k steps): MSE/Var = 0.686 = 0.115 level + 0.030 shared + 0.540 specific. The 95% interval is [0.637, 0.736]. The truth is 39% shared and 61% specific.
- Corrected floor: 0.282 over 282 targets. Of these, 180 come from replicate floors, 54 are transferred and 48 are class medians. This supersedes the sprint 06 value of 0.38.
- Replicate floor by assay: ATAC-seq 0.07, H3K4me3 0.11, H3K27ac 0.18, H3K27me3 0.35, H3K9me3 0.56.
- Four-way budget on 128 targets with a replicate floor: explained 0.444, unexplained 0.297, noise 0.259, added 0.011, level 0.100.
- The best model recovered about 0.21 of the recoverable specific variance. Under the old floor the figure was 0.30.

**CANDII copy.**
- `CANDII: tools/eic_decomp/errdecomp.py` is byte-identical to Max's module.
- `CANDII: tools/eic_decomp/aggregate.py` differs from Max's code in three ways:
  1. It groups the shared estimate by cell type, not by ENCODE term.
  2. It has no noise term.
  3. It runs on the EIC blind test.

- **Source:**
  - Max s06: `scripts/lib/errdecomp.py`, `scripts/v3/decompose_v3.py`, `scripts/res/multires.py`, `scripts/report2/aggregate2.py`, `output/summary/v3_budget_arms.tsv`, `report.html`, `report2.html`.
  - Max s07: `report_sections/05_floors.md`, `report_sections/02_model.md` §2.5, `notes/floors.md`, `output/floors/floors_final_per_target.tsv`, `scripts/covariates/knob_maps.py`.
- **Related:** [[onion caveats#noise-floor-unit-rule]], [[onion variation and objectives#read-sampling-noise-layer]].

## crps-oracle-scaled-split

> **Older-CANDI note:** this entry mentions results from an older CANDI implementation. miniCANDI is a first-principles redesign based on [[peeling the onion]]. Read these results as context, not as outcomes to reproduce.

- CANDII scores each count track's CRPS as crps = crps_oracle_scaled + scale_error.
- crps_oracle_scaled is the CRPS after the best multiplicative rescale µ·2^c of the predicted mean, fitted to that track. The search is coarse over c ∈ [−6, 6] in steps of 0.25, then refined in steps of 0.01, on up to 20,000 bins.
- scale_error is the part that one constant would remove.
- The oracle is fitted in-sample. So crps_oracle_scaled is an upper bound on capability, and scale_error can come out slightly negative.
- In the old CANDI four-arm comparison, this split cut the spread between arms from 0.7148 to 0.1133 macro CRPS, i.e. by 84%.
- The project rules require both terms whenever a CRPS is quoted.
- **Source:**
  - `CANDII: EVAL.md` §D, `src/candi/bench/distributional.py::oracle_scale`, `AGENTS.md` §7.2.
  - `CANDI: candi_kit/research/H48_SCORECARD.md`.

## nested-baseline-ladder

Max's sprint 03 notes and the CANDII bench together define a nested ladder of baselines, from weakest to strongest:
1. marginal NB, i.e. one fitted distribution per track
2. the per-position training average
3. a generic calibrated average, i.e. one per-assay curve learned on training tracks
4. the oracle-level average, i.e. the best smooth function of the average fitted on the truth, which is an upper bound
5. kNN averaging over similar biosamples, global (k = 5) or local
6. depth-only scaling of a source track
7. another experiment of the same ENCODE term
8. a true replicate
9. the noise ceiling, i.e. the square root of the replicate-to-replicate score

- **Source:**
  - Max s03: `notes/metrics.md`, `output/metrics/v2_level_fixed_bounds.tsv`.
  - `CANDII: src/candi/bench/distributional.py::marginal_nb`, `src/candi/reference.py`.
  - `CANDII-worktree-t146: results/t150/REPORT.md`.
  - Max s06: `output/summary2/related_vs_sampling.tsv`.

## r-spec-specificity-correlation

- Max's r_spec is the correlation between truth and prediction after a smooth function of the training average a has been removed from each: r_spec = corr(y − g(a), p − h(a)).
- g and h are least-squares piecewise-linear functions of a. They use about 400 quantile knots: 320 over probabilities 0–0.99 and 80 over 0.99–1. They are fitted on the bins being scored.
- A guard sets r_spec to 0 when SD(p − h(a)) ≤ 0.2% of SD(p).
- Any linear rescaling of the average scores exactly 0.
- On 212 tracks, four nonlinear rescalings of the average scored at most 0.0095 without the guard. So differences above about 0.01 are not fit artifacts.
- Max found that sprint 02's lead of CANDII over kNN in deviation correlation was mostly level. The specific parts were 0.096 and 0.103.
- In a depth-normalized unit, r_spec stayed at 0.11–0.15, while deviation correlation fell from 0.25 to 0.15.
- **Source:** Max s03: `scripts/metrics/metrics_lib.py::rspec`, `notes/metrics.md`, `output/metrics/v2_null_check.tsv`.

## resolution-question-prior-evidence

- **Onion section:** the "interesting question" on resolution.
- Max's sprint 06 measured the noise share of the truth's cell-type-specific variance at each Haar level, in asinh(−log10 p) units.
- In the fine band (200 bp or less), about 90% of the specific variance was sampling noise. The best model explained almost none of it (0.4%).
- The specific signal first exceeded the noise at these scales:

| Assay | Crossover scale |
|---|---|
| ATAC-seq | 400 bp |
| H3K4me1 | 3.2 kb |
| H3K4me3 | 6.4 kb |
| H3K27ac | 6.4 kb |
| H3K27me3 | 25.6 kb |
| H3K36me3 | 25.6 kb |
| H3K9me3 | 102.4 kb |

- These readings used the sprint 06 floor, which sprint 07 later found too high. The true crossovers are therefore probably at finer scales.
- Scoring at 200 bp lost 3–9% of the specific signal and removed 20–45% of the sampling noise.
- **Source:** Max s06: `output/summary2/perlevel.tsv`, `eue.tsv`, `res_tradeoff.tsv`.

## resolution-study-open-points

- No study of noise share by scale exists in count space.
- No study has split the source of variation, artifact or biology, by scale.
- In the CANDII testbed, thinning 25 bp count units independently made 1 kb sums too smooth. Real depth-arm halves matched whole-read thinning at a ratio of 1.002 for 1 kb shape, against 1.226 for unit thinning.
- Toneyan and colleagues (2022) showed that models scored at different bin sizes are not directly comparable, because binning smooths the coverage and changes the metrics.
- **Source:**
  - `CANDII: cruxvault/results/t142/DIGEST.md` (local).
  - Toneyan S, Tang Z, Koo PK. Nat Mach Intell 4:1088 (2022), doi:10.1038/s42256-022-00570-9.

## calibration-battery

- Max's sprint 07 models gave 90% interval coverage of 0.888–0.909 for every trained arm. So marginal coverage alone does not separate the arms.
- The CANDII bench already computes a count-arm PIT calibration error (`ece`) and interval coverage. The leaderboard rows do not carry them. On the board, the count arm shows only CRPS, its oracle-scaled and scale-error split, and NB NLL.
- Gneiting and Raftery (2007) set the frame: maximize sharpness subject to calibration, and score with proper scoring rules such as CRPS.
- **Source:**
  - Max s07: `output/model/arms.tsv`.
  - `CANDII: EVAL.md` §D, `leaderboard/rows/` (origin/main).
  - Gneiting T, Raftery AE. J Am Stat Assoc 102:359 (2007), doi:10.1198/016214506000001437.
- **Related:** [[onion variation and objectives#calibration-terms-and-pit]].

## predicted-vs-replicate-variance

- Foroozandeh Shahraki et al. (2024) found that the posteriors of chromatin-state annotation methods are very overconfident. Most positions get a posterior above 99%, yet two replicates of one cell type agree at only about 80% of bins.
- Their method, SAGAconf, calibrates confidence against measured replicate agreement instead of the model's own posterior.
- The same logic applies to a count model: its predicted NB variance can be checked against the variance between real replicates. No project result has done this check yet.
- **Source:**
  - Foroozandeh Shahraki M, Farahbod M, Libbrecht MW. Genome Res 34:469 (2024), doi:10.1101/gr.278343.123.
  - `CANDII: cruxvault/wiki/uncertainty-calibration.md`.
- **Related:** [headline-metric-set](#headline-metric-set) (metric 7).

## region-strata-and-foreground-purity

> **Older-CANDI note:** this entry mentions results from an older CANDI implementation. miniCANDI is a first-principles redesign based on [[peeling the onion]]. Read these results as context, not as outcomes to reproduce.

- In old CANDI, the foreground selector took the 98th percentile of each target, to select 2% of positions.
- On targets that were 36–67% zero, the threshold was 0. The mean selected fraction was 23.9%, and 21.6% of records selected every position. Those records gave 90.1% of the bootstrap sample, so the "foreground" tests were mostly background.
- Max's `TrackScorer` takes a dict of boolean region masks (`strata=`) and reports each metric per stratum. It can refit the level function inside each stratum or share one fit across strata.
- His "facultative" stratum is the top 10% of bins by cross-biosample SD.
- The onion's region list (promoters, enhancers, gene bodies, heterochromatin, foreground, background) can be passed as such masks.
- **Source:**
  - `CANDI: candi_kit/research/METADATA_AUDIT.md` (item S3).
  - Max s03: `scripts/metrics/metrics_lib.py` (`TrackScorer`, `facultative_mask`), `notes/metrics.md`.

## value-strata

- Max's sprint 06 split the error into strata of truth value.
- The top 10% of truth bins held about two thirds of the calibrated squared error for every predictor (0.66–0.74).
- At the strongest constitutive sites, the best single arm lost to the cross-cell-type average. In the top 0.1% of bins by training average, its per-bin calibrated squared error was 3.98, against 2.66 for the average.
- The units are squared error divided by the truth track's variance, in asinh(−log10 p).
- About 40% of positives are rare, and the models recovered only 6–10% of them.
- **Source:** Max s06: `output/summary/v3_value_strata.tsv`, `notes/01_handoff.md`.

## broad-marks-top1-agreement

- Max compared two half-depth pseudoreplicate halves of one 30M-read experiment.
- The top 1% of 25 bp bins in one half overlapped the other half's top 1% for:
  - H3K4me3: 85% of bins
  - DNase-seq: 76%
  - H3K27me3: 9%
  - H3K9me3: 4%
- So at 25 bp, the strongest bins of a broad mark are mostly sampling noise. Each figure comes from one track.
- **Source:** Max s06: `output/summary/floors_t112_class.tsv` (pair = pseudorep, kind = rank, val = 0.01).

## identifiability-table

> **Older-CANDI note:** this entry mentions results from an older CANDI implementation. miniCANDI is a first-principles redesign based on [[peeling the onion]]. Read these results as context, not as outcomes to reproduce.

- Old CANDI measured how much each covariate can say about the output beyond the input signal. It used probe models with a certified detection floor, on the EIC panel.
- No covariate cleared its own floor after correction. The best reading, read length on counts, was 1.17 times its floor before correction and 0.79 times after.
- On full EIC (363 non-control tracks), run type carries little information beyond assay and read length:

| Quantity | Bits |
|---|---|
| H(run type) | 0.756 |
| H(run type \| assay) | 0.551 |
| H(run type \| assay, read length) | 0.128 |

- Run type varies in only 6 of 83 (assay, read length) cells. On the old 5-biosample sandbox slice it carried 0.000 bits.
- Leek et al. (2010) state the general limit: when a batch variable is confounded with the outcome, no computational correction can separate them.
- This bears directly on what the new dataset must contain (see [[onion data plan#identifiability-pointer]]).
- **Source:**
  - `CANDI: cruxvault/h60_a_single_conditioning_dropout_trained_mo.md`, `h62_the_deadness_is_panel_specific_for_platf.md`, `h57_a_re_selected_biosample_panel_is_the_onl.md`.
  - `CANDII: cruxvault/results/t10/manifest.json` (local), recomputed 2026-11-10.
  - Leek JT et al. Nat Rev Genet 11:733 (2010), doi:10.1038/nrg2825.

## cac-evaluation-test-sets

- The CANDII CAC testbed trained on base-to-arm pairs. It also scored arm-to-arm pairs that it never trained on.
- In counts, the model's relative CRPS gain over a twin without covariates was:
  - 0.29 on trained pairs
  - 0.11–0.16 on never-trained pairs
- In −log10 p, the gain fell from 0.15 to 0.004–0.06.
- So a gain on trained pairs overstates what transfers to unseen covariate combinations. The testbed had 7 tracks and 3 seeds.
- Note: the finding text for this result says "scrambled twin", but its numbers are gains over the no-covariates twin.
- **Source:**
  - `CANDII: cruxvault/results/h19/report.md` §2 (local).
  - `CANDII: cruxvault/f8_the_chosen_designs_beat_the_scrambled_tw.md`, `f19_in_every_design_g_version_and_loss_train.md`.
- **Related:** [[onion caveats#two-step-cac-per-track-vs-shared]].

## cac-evaluation-baselines-and-controls

- In the final CAC testbed, the learned correction was compared with a depth-only baseline. That baseline scales counts by target depth over source depth and fits one NB size per pair.

| Pairs | Learned correction | Depth-only baseline |
|---|---|---|
| trained | 0.432 | 0.550 |
| never-trained | 0.527 | 0.677 |

- The margins were 11–66 times the across-seed SD.
- The same network without covariates tied depth-only on trained pairs (0.547 vs 0.550).
- Depth-only still beat the learned seed mean on 112 of 228 trained pairs.
- Max's sprint 08 added a no-change control: the half-versus-half comparison, run on a fresh random sample of the base's own reads. It showed that shared reads give a head start of 0.06–0.14 before any correction.
- **Source:**
  - `CANDII-worktree-t146: results/t150/REPORT.md`.
  - `CANDII: cruxvault/f33_margins_are_11_66x_the_seed_sd_on_every_.md`.
  - Max s08: `notes/00_handoff.md`.

## cac-evaluation-gates

- The CAC testbed used a depth law as a gate: a correction must scale the mean by the true depth ratio.
  - Median log2 error on trained pairs: 0.04–0.10.
  - Median log2 error on never-trained pairs: 0.19–0.55.
  - In the depth-offset run, the worst-case ratio error met the 0.10 bar in 0 of 72 cells.
- The swap check feeds the source covariates back as the target and expects the input returned. It failed in every chosen design: the median abs(log ratio) was about 0.2, against a bar of 0.1.
- A planted-effect positive control was specified but never run.
- All three runs of the covariate-swap probe failed their outcome-neutral controls, so their readings were void.
- **Source:**
  - `CANDII: cruxvault/f19_...`, `f17_large_clean_margins_on_the_432_runs_and_.md`, `results/h15/report.md` (local), `h1_conditioning_on_the_recorded_sequencing_.md`, `f22_all_three_runs_failed_their_outcome_neut.md`.
- **Related:** [[onion variation and objectives#steerability-vs-decodability]], [[miniCANDI architecture ablations]] (preprocessing #1 note).

## cac-evaluation-twins

- The CAC testbed compared each model with several twins:
  - a twin without covariates
  - a twin that reads covariate labels as plain ids
  - a twin that reads the input from a random bin of the same chromosome
- In the final run, the scrambled twin was scored on trained pairs with the labels it trained on, so that check could not fail. On never-trained pairs the twins were worse by 0.28–0.32 CRPS.
- Max's sprint 07 found that a random vector can act as a regularizer. A model trained with shuffled trans-regulator vectors gained +0.024 r_spec over the base model, more than the model trained with the real vectors. A wrong cell's vector helped as much as the right one.
- **Source:**
  - `CANDII: cruxvault/results/row2/GRID_SUMMARY.md` (local).
  - `CANDII-worktree-t146: results/t146/REPORT.md`.
  - Max s07: `notes/trans.md`.

## truth-pipeline-swap

- The CANDII benchmark scores the same 2019 challenge experiments under two truths: the organizers' 2019 processing and ENCODE4 v1.5.1 (2020). A standing that holds under both truths is not an artifact of one pipeline.
- Max's sprint 04 added more layers of the same experiments: single-end reprocessed tracks, and the quantile-normalized versions behind the corrected leaderboard.
- On 48 experiments, original and single-end truths correlated at only 0.37 (H3K9me3) to 0.84 (H3K4me3). This is the median arcsinh Pearson r at 25 bp.
- **Source:**
  - `CANDII: plan/BENCHMARK_DESIGN.md` §6.
  - Max s04: `notes/01_handoff.md`, `output/track_pairs.tsv`.

## score-on-normalized-and-unnormalized-truth

> **Older-CANDI note:** this entry mentions results from an older CANDI implementation. miniCANDI is a first-principles redesign based on [[peeling the onion]]. Read these results as context, not as outcomes to reproduce.

- The corrected EIC leaderboard scores against single-end reprocessed truth. That truth is then quantile-normalized: each track's peak and background bases are mapped to per-assay reference levels.
- Max's sprint 04 recovered this rule exactly. It matched at every base (3,031,042,417 bases) for 47 of 48 tracks.
- The normalization removes each track's level and scale. Two independent single-end rebuilds correlated at 0.83–0.90 before it and at 0.92–0.99 after it.
- CANDI with a per-assay affine floor (a·CANDI + b·Average + c) ranked 1st of 26 on every truth layer except the un-normalized single-end truth, where it ranked 6th.
- **Source:** Max s04: `output/candi_placement.tsv`, `notes/01_handoff.md`, `scripts/qnorm/apply_bp.py`; Max `notes/key_findings.md`.
- **Related:** [[onion data plan#exact-eic-truth]], [[miniCANDI architecture ablations]] (preprocessing #7 note).

## inference-unit-and-seed-floors

> **Older-CANDI note:** this entry mentions results from an older CANDI implementation. miniCANDI is a first-principles redesign based on [[peeling the onion]]. Read these results as context, not as outcomes to reproduce.

- Old CANDI found that confidence intervals widened by a median of 18 times when the bootstrap clustered on the 12 held-out targets instead of on positions. Under clustering, the run-type steering claims no longer showed a direction.
- In CANDII, one seed change moved the held-out macro count CRPS by:
  - V_breadth panel: 0.0139
  - V_matched panel: 0.0115
  - blind B panel: 0.2033. Here the two seeds also ranked the tracks differently.
- Max's sprint 07 found a seed gap of 0.001–0.011 r_spec (mean 0.007). It uses a paired cluster bootstrap over ENCODE terms.
- **Source:**
  - `CANDI: candi_kit/research/H48_REPORT.md` F4.
  - `CANDII: cruxvault/results/t86/SEED_FLOOR.md` §16.1 (local).
  - Max `notes/key_findings.md`; Max s07: `notes/trans.md`.
- **Related:** [[onion data plan#split-by-group]].

## metric-hygiene

> **Older-CANDI note:** this entry mentions results from an older CANDI implementation. miniCANDI is a first-principles redesign based on [[peeling the onion]]. Read these results as context, not as outcomes to reproduce.

- In old CANDI, one Spearman correlation over all held-out targets pooled assays of very different count scales. This made the gap between two arms about 3 times larger than the per-assay macro average: 0.132 against 0.040.
- Spearman correlation does not change under any strictly increasing map, so it cannot detect a monotone scale error.
- The CANDII bench omits AUROC by decision. Peak positives are rare, so the large negative count damps the false-positive rate. AUPRC is reported with its base rate.
- The benchmark design forbids subtracting the V_breadth score from the B score.
- **Source:**
  - `CANDI: cruxvault/h46_the_offset_off_imputation_gap_is_scale_m.md`.
  - `CANDII: EVAL.md` (D14), `plan/BENCHMARK_DESIGN.md` §5.3.

## peak-height-vs-shape

- The CANDII knob ranking compared each processing change with the pseudoreplicate floor, using two metrics:
  - **Peak height:** the abs(log2 fold change) of mean signal over the base's top-1% bins.
  - **Shape:** 1 − Spearman on 1 kb bins.
- Peak height cleared the floor in 27 of 27 count rows, at 33–763 times the floor.
- Shape cleared it in only 3 of 27: DNase at 3.75M reads, DNase paired-end, and a broad mark at antibody proxy f0.9.
- **Source:**
  - `CANDII: cruxvault/results/t139/knob_vs_floor.tsv`, `DIGEST.md` (local).
  - `CANDII: cruxvault/f24_peak_height_clears_the_floor_by_a_wide_m.md`.

## crossed-variance-partition

- Hoffman and Schadt (2016) fit one linear mixed model with crossed random effects, e.g. cell type, donor, lab and batch. They report the fraction of variance that each factor explains.
- Because all factors are fit jointly, the fractions do not depend on a peeling order. A sequential peeling, one source at a time, does depend on its order.
- Teng et al. (2021) fit a mixed model to 211 CTCF ChIP-seq samples from 90 cell types and three laboratories. Both the laboratory batch effect and the variability between biological replicates differed from one binding site to another. So a single genome-wide batch variance does not describe the noise at a given site.
- **Source:**
  - Hoffman GE, Schadt EE. BMC Bioinformatics 17:483 (2016), doi:10.1186/s12859-016-1323-z.
  - Teng M, Du D, Chen D, Irizarry RA. NAR Genom Bioinform 3:lqab098 (2021), doi:10.1093/nargab/lqab098.

## within-locus-population-metric

- Tu et al. (2026) scored sequence-to-function models across individuals at each locus, on 500 heritable genes in 859 individuals.
- The mean per-gene Pearson R for expression was:
  - AlphaGenome: 0.151
  - Borzoi: 0.058
  - cis-genotype elastic-net baseline: 0.670
- For chromatin accessibility in 100 lymphoblastoid lines, AlphaGenome came close to the heritability ceiling.
- Thus a model can rank loci well and still fail on variation across individuals at one locus.
- **Source:** Tu X, Spiro A, Chikina M, Mostafavi S. bioRxiv (2026), doi:10.64898/2026.02.01.702969.
- **Related:** [[onion caveats#population-paper-and-edice]].

## magnitude-vs-shape-diagnostic

- Borzoi splits its training loss into two parts:
  - a Poisson loss on the summed coverage of a window, i.e. its magnitude
  - a multinomial loss on the normalized coverage across positions, i.e. its shape
- Linder et al. (2025) note that independent Poisson counts per position are equivalent to a Poisson total followed by a multinomial allocation.
- In the CANDII knob ranking, processing changes moved peak height far more often than 1 kb shape (see [peak-height-vs-shape](#peak-height-vs-shape)). "Artifacts act on magnitude, and biology acts on shape" is a hypothesis, not a result.
- **Source:** Linder J, Srivastava D, Yuan H, Agarwal V, Kelley DR. Nat Genet 57:949 (2025), doi:10.1038/s41588-024-02053-6.

## instrument-hygiene

> **Older-CANDI note:** this entry mentions results from an older CANDI implementation. miniCANDI is a first-principles redesign based on [[peeling the onion]]. Read these results as context, not as outcomes to reproduce.

- In old CANDI, an audit found six broken measurement instruments in the evaluation code.
- The fixes were verified three ways: known-answer tests, a brute-force code check and real data.
- The verification pass then found three more bugs in its own tooling. One was a sign test that counted exact ties as negatives. It gave an arm whose deltas were all exactly zero the most significant p-value in the table (p = 0.0005).
- The report noted that the test suite had never been mutation-tested.
- **Source:** `CANDI: candi_kit/research/H48_REPORT.md` §1 and "Bugs found in the h48 tooling itself".

## fixed-random-window-scope

- A mid-training check of all held-out bins cost more than three epochs.
- Scoring only a 2.67% subset of windows made the check 16.1 times faster. About 206 s of each check did not depend on scope.
- The ENCODE Pilot Regions were the obvious subset. Their level sat +0.30 to +0.41 above full coverage, and the offset grew as training went on. They flipped the sign of two checkpoint comparisons.
- Fixed, seeded random windows of the same size were offset by only about −0.003. They reproduced the full checkpoint ordering in 194 of 200 draws.
- **Source:** `CANDII: EVAL.md` §"`--eval-regions`, the selection scope".
