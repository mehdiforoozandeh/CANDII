---
id: h10
type: idea
schema: 2
title: Conditioning on the recorded fragment extension predicts a target track from a source track, beyond what a single value-axis map already does
parent: q3
status: idea
rule: all
measurement: "per-knob slice of the g × f ladder (t118): D_twin − D_g on the fragment extension base ↔ arm pairs, per mark class, against the no-covariates twin; 7 tracks x 3 seeds"
replicates: "TODO(PI) — source→target pairs per arm x seeds, on the counterfactual pairs t112 built. Whatever it is, it must clear the seed floor: one paired seed change on the q19 recipe moves pooled imputation CRPS by 0.0463 (macro 0.0327) under eval.py and macro CRPS by 0.0608 under bench, and the per-track floor is several times either macro (AGENTS.md §7.2 rule 2)."
verdict: 
metric: 
created: "2026-09-17T21:46:17"
updated: "2026-10-31T18:16:47"
null_approved: "2026-09-17T21:47:48"
null_hash: 8e4d67b544ab2035
parked: true
---

# h10 — Conditioning on the recorded fragment extension predicts a target track from a source track, beyond what a single value-axis map already does

Parent:: [[q3_do_the_covariates_that_record_how_the_co]]

## ELI5

Knowing how far each read was stretched into a fragment closes at least half of the p-value gap a plain rescale leaves. Untested; on new pairs the model's p-values blow up.

## TL;DR

The pair is the base track and a counterfactual with reads extended to 0.5× or 2× the cross-correlation fragment length (DNase: the smoothing window) ([[t112_build_the_counterfactual_arms_for_t|the counterfactual-arms task]]). The counts are bit-identical to base, so the claim lives on the −log10 p axis. The checks (PI ruling 2026-10-31) ask g to beat its no-covariates twin by more than 2 × seed wobble per mark class in −log10 p, on base ↔ arm and never-trained arm → arm pairs, with predicted counts held at the source; no per-knob scoring has run yet. Exploratory readings: a per-pair monotone map already beats the pseudoreplicate oracle in p. A generator g that reads the extension beats its twin in p on trained pairs in 10 of 16 configurations; on the never-trained pair half the configurations explode.

## Expect

On this knob's base ↔ arm pairs, g(C, C′) beats its no-covariates twin (C and C′ re-scrambled at every step) by more than 2 × g's seed wobble, per mark class, in −log10 p, and the predicted counts stay equal to the source counts.

_Switched to the scrambled-covariate twin by PI ruling on 2026-10-31, replacing the one-warp → oracle bar cloned on 2026-09-17. The readings already taken from the g × f ladder predate this switch, so against this bar they are post hoc; a per-knob scoring of the existing ladder runs is the planned test._

## Null
Normalization: the pairs differ by one global monotone value-axis map, so a covariate-free rescale already closes the gap and the conditioning adds nothing.

## Problem Statement

Extension smooths the pileup; a longer extension merges neighbouring peaks and a shorter one sharpens them, a shape change no value-axis map captures. It is one of four control-and-caller knobs under [[q3_do_the_covariates_that_record_how_the_co|the control-and-caller question]], and the only one with DNase products.

## Idea / Hypothesis

Conditioning on the recorded fragment extension predicts a target track from a source track, beyond what a single value-axis map already does

## Verifiables

- [ ] beatstwin [claim-directed]: on the fragment extension base ↔ arm pairs, D_twin − D_g > 2 × g's seed wobble (max pairwise |Δ| of D_g over 3 seeds), per mark class, in −log10 p; D is NB CRPS for counts and log-normal CRPS for −log10 p, always split into oracle_scaled + scale_error
      fails-if:: the fragment extension covariate adds nothing a scrambled-covariate twin of equal capacity cannot already do
- [ ] lawtest [claim-directed]: on never-trained fragment extension arm → arm pairs, where the knob has two or more arm levels, g beats the twin by more than 2 × seed wobble
      fails-if:: g keeps one map per trained (C, C′) pair and has no answer for a fragment extension combination it never saw
- [ ] shufflecollapse [outcome-neutral]: with C′ replaced at scoring by another arm's C′, g's advantage over the twin falls to within 2 × seed wobble
      fails-if:: g's f does not change when told the wrong target covariates, so any gain is not read from C′
- [ ] countsfixed [outcome-neutral]: the predicted counts equal the source counts, because a fragment extension change leaves the treatment pileup bit-identical
      fails-if:: f moves the counts, so the run measures something other than this knob

## Planned Intervention

No run has tested this claim: [[t103_build_the_in_vitro_covariate_testbed|the testbed build]] has not run on this arm, and [[t107_run_the_in_vitro_covariate_testbed_o|the testbed run]] covers only depth and run type. The readings come from runs that trained one generator g on all ten knobs at once: the g × f grid, rows 1 and 2 ([[t118_learn_the_covariate_conditioned_tra|the grid task]]; `plan/T118_COUNTERFACTUAL_F.md`), and the encoder-decoder rungs ([[t126_build_and_run_the_encoder_decoder_r|the encoder-decoder task]]; `plan/T126_ENCDEC_A.md`). Both score chr19 + chr21 over 3 seeds against a no-covariates twin.

## Run Links

- g × f grid rows 1 and 2: `nibi:/project/def-maxwl/mforooz/t118/{ladder,row2}/runs` (row-2 arrays 22855414, 22862752; knob extraction 24641277)
- Encoder-decoder: `nibi:/project/def-maxwl/mforooz/t126/{runs,D/runs}` (train 24646366, 24647762; law 24646367, 24647788, 24706070)
- Products: `nibi:/scratch/mforooz/t112_cf/products/<track>__extsize__{k0.5,k2}/` (scratch purge 2026-11-20)

## Artifacts

<!-- what the run produced. Keep files under results/h10/ and link at least the report:
     - [Report](results/h10/report.md)   - results/h10/curve.png -->
- [Per-knob readings for this knob family](results/q3/KNOB_READINGS.md) — the tables, the CRPS split and the caveats
- [Baseline rungs by arm](results/t118/rungs_v2.md)
- [Grid knob table, rows 1 and 2](results/t118/knob_generalization/knob_table.tsv) and its [figures](results/t118/knob_generalization/report.pdf)
- [Encoder-decoder knob table](results/t126/knob_table.tsv) and its [report](results/t126/report.pdf)
- [Grid deck](presentations/t118-grid/index.html) — published https://claude.ai/artifact/1mFz6C5jLaG1Ctc8JN572C
- [Encoder-decoder deck](presentations/t126-encdec/index.html) — published https://claude.ai/artifact/KP7pipaRbNbEStUaJR7jrZ
- [Counterfactual-arms manifest](results/t112/MANIFEST.tsv) and [summary](results/t112/COUNTERFACTUAL_ARMS.md) — published https://claude.ai/code/artifact/fb5ff45d-f155-4b01-ac50-db1179f4fd61

## Findings

Observed 2026-10-01 to 2026-10-21; not concluded. In p, QuantileMatching, a per-pair monotone map, scores past the pseudoreplicate oracle (share 1.69), so that oracle is no ceiling. Grid, gain over the twin, median of 16 configurations, bar 2× seed wobble: p, trained 0.20 (10/16 met, median wobble 0.028; 4 cells explode); never-trained unusable, 8 of 16 cells explode; count-identical pairs, trained 0.36 (14/16), never-trained 0.28 (7/16). Encoder-decoder, count-identical pairs: old design D 0.24 trained and 0.22 never-trained, learned-decoder rung 0.10–0.14, old design A −0.06 never-trained. No run measures the count-unchanged control directly. Tables: [knob readings](results/q3/KNOB_READINGS.md).
