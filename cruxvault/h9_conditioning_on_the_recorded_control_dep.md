---
id: h9
type: idea
schema: 2
title: Conditioning on the recorded control depth predicts a target track from a source track, beyond what a single value-axis map already does
parent: q3
status: idea
rule: all
measurement: "per-knob slice of the g × f ladder (t118): D_twin − D_g on the control depth base ↔ arm pairs, per mark class, against the no-covariates twin; 7 tracks x 3 seeds"
replicates: "TODO(PI) — source→target pairs per arm x seeds, on the counterfactual pairs t112 built. Whatever it is, it must clear the seed floor: one paired seed change on the q19 recipe moves pooled imputation CRPS by 0.0463 (macro 0.0327) under eval.py and macro CRPS by 0.0608 under bench, and the per-track floor is several times either macro (AGENTS.md §7.2 rule 2)."
verdict: 
metric: 
created: "2026-09-17T21:46:17"
updated: "2026-09-17T21:47:48"
null_approved: "2026-09-17T21:47:48"
null_hash: 8e4d67b544ab2035
---

# h9 — Conditioning on the recorded control depth predicts a target track from a source track, beyond what a single value-axis map already does

Parent:: [[q3_do_the_covariates_that_record_how_the_co]]

## ELI5

Knowing how deeply the background library was sequenced closes at least half of the p-value gap a plain rescale leaves. Untested; early readings show the best carry-over to new pairs.

## TL;DR

The pair is the base track and a counterfactual called against the same control thinned to q = 0.5 or 0.25 of its reads ([[t112_build_the_counterfactual_arms_for_t|the counterfactual-arms task]]). The counts are bit-identical to base, so the claim lives on the −log10 p axis. The bar is [[q1_do_the_recorded_experimental_covariates_|the parent question's locked protocol]] (gap-closed against one-warp ≥ 0.5 per mark class, gain above 2× seed |Δ|); no run has scored it. Exploratory readings: a generator g that reads q beats its no-covariates twin in p by the largest margin of any knob, on trained and never-trained pairs, in every configuration. The bar cannot yet be read, because the source track already beats the pseudoreplicate oracle.

## Expect

On this knob's base ↔ arm pairs, g(C, C′) beats its no-covariates twin (C and C′ re-scrambled at every step) by more than 2 × g's seed wobble, per mark class, in −log10 p, and the predicted counts stay equal to the source counts.

_Switched to the scrambled-covariate twin by PI ruling on 2026-10-31, replacing the one-warp → oracle bar cloned on 2026-09-17. The readings already taken from the g × f ladder predate this switch, so against this bar they are post hoc; a per-knob scoring of the existing ladder runs is the planned test._

## Null
Normalization: the pairs differ by one global monotone value-axis map, so a covariate-free rescale already closes the gap and the conditioning adds nothing.

## Problem Statement

A shallower control raises the variance of the local lambda estimate; the effect concentrates where the control is sparse. It is one of four control-and-caller knobs under [[q3_do_the_covariates_that_record_how_the_co|the control-and-caller question]]. The arm is ChIP only.

## Idea / Hypothesis

Conditioning on the recorded control depth predicts a target track from a source track, beyond what a single value-axis map already does

## Verifiables

- [ ] beatstwin [claim-directed]: on the control depth base ↔ arm pairs, D_twin − D_g > 2 × g's seed wobble (max pairwise |Δ| of D_g over 3 seeds), per mark class, in −log10 p; D is NB CRPS for counts and log-normal CRPS for −log10 p, always split into oracle_scaled + scale_error
      fails-if:: the control depth covariate adds nothing a scrambled-covariate twin of equal capacity cannot already do
- [ ] lawtest [claim-directed]: on never-trained control depth arm → arm pairs, where the knob has two or more arm levels, g beats the twin by more than 2 × seed wobble
      fails-if:: g keeps one map per trained (C, C′) pair and has no answer for a control depth combination it never saw
- [ ] shufflecollapse [outcome-neutral]: with C′ replaced at scoring by another arm's C′, g's advantage over the twin falls to within 2 × seed wobble
      fails-if:: g's f does not change when told the wrong target covariates, so any gain is not read from C′
- [ ] countsfixed [outcome-neutral]: the predicted counts equal the source counts, because a control depth change leaves the treatment pileup bit-identical
      fails-if:: f moves the counts, so the run measures something other than this knob

## Planned Intervention

No run has tested this claim: [[t103_build_the_in_vitro_covariate_testbed|the testbed build]] has not run on this arm, and [[t107_run_the_in_vitro_covariate_testbed_o|the testbed run]] covers only depth and run type. The readings come from runs that trained one generator g on all ten knobs at once: the g × f grid, rows 1 and 2 ([[t118_learn_the_covariate_conditioned_tra|the grid task]]; `plan/T118_COUNTERFACTUAL_F.md`), and the encoder-decoder rungs ([[t126_build_and_run_the_encoder_decoder_r|the encoder-decoder task]]; `plan/T126_ENCDEC_A.md`). Both score chr19 + chr21 over 3 seeds against a no-covariates twin.

## Run Links

- g × f grid rows 1 and 2: `nibi:/project/def-maxwl/mforooz/t118/{ladder,row2}/runs` (row-2 arrays 22855414, 22862752; knob extraction 24641277)
- Encoder-decoder: `nibi:/project/def-maxwl/mforooz/t126/{runs,D/runs}` (train 24646366, 24647762; law 24646367, 24647788, 24706070)
- Products: `nibi:/scratch/mforooz/t112_cf/products/<track>__ctldepth__{q0.5,q0.25}/` (scratch purge 2026-11-20)

## Artifacts

<!-- what the run produced. Keep files under results/h9/ and link at least the report:
     - [Report](results/h9/report.md)   - results/h9/curve.png -->
- [Per-knob readings for this knob family](results/q3/KNOB_READINGS.md) — the tables, the CRPS split and the caveats
- [Baseline rungs by arm](results/t118/rungs_v2.md)
- [Grid knob table, rows 1 and 2](results/t118/knob_generalization/knob_table.tsv) and its [figures](results/t118/knob_generalization/report.pdf)
- [Encoder-decoder knob table](results/t126/knob_table.tsv) and its [report](results/t126/report.pdf)
- [Grid deck](presentations/t118-grid/index.html) — published https://claude.ai/artifact/1mFz6C5jLaG1Ctc8JN572C
- [Encoder-decoder deck](presentations/t126-encdec/index.html) — published https://claude.ai/artifact/KP7pipaRbNbEStUaJR7jrZ
- [Counterfactual-arms manifest](results/t112/MANIFEST.tsv) and [summary](results/t112/COUNTERFACTUAL_ARMS.md) — published https://claude.ai/code/artifact/fb5ff45d-f155-4b01-ac50-db1179f4fd61

## Findings

Observed 2026-10-01 to 2026-10-21; not concluded. QuantileMatching, a per-pair monotone map, cuts noSolution's p CRPS by 8%. The source track already beats the pseudoreplicate oracle in p, so gap-closed cannot be read until the NB-draw oracle exists. Grid, gain over the twin, median of 16 configurations, bar 2× seed wobble: p, trained 0.62 (16/16 met, median wobble 0.003), never-trained 0.48 (16/16, wobble 0.006); count-identical pairs, trained 0.28 (14/16), never-trained 0.06 (4/16). Encoder-decoder, count-identical pairs: old design D 0.24 trained and 0.22 never-trained, learned-decoder rung 0.09–0.14, old design A −0.10 never-trained. No run measures the count-unchanged control directly. Tables: [knob readings](results/q3/KNOB_READINGS.md).
