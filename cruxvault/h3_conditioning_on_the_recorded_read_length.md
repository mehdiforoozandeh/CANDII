---
id: h3
type: idea
schema: 2
title: Conditioning on the recorded read length predicts a target track from a source track, beyond what a single value-axis map already does
parent: q2
status: idea
rule: all
measurement: "per-knob slice of the g × f ladder (t118): D_twin − D_g on the read length base ↔ arm pairs, per mark class, against the no-covariates twin; 7 tracks x 3 seeds"
replicates: "TODO(PI) — source→target pairs per arm x seeds, on the counterfactual pairs t112 built. Whatever it is, it must clear the seed floor: one paired seed change on the q19 recipe moves pooled imputation CRPS by 0.0463 (macro 0.0327) under eval.py and macro CRPS by 0.0608 under bench, and the per-track floor is several times either macro (AGENTS.md §7.2 rule 2)."
verdict: 
metric: 
created: "2026-09-17T21:46:17"
updated: "2026-09-17T21:47:47"
null_approved: "2026-09-17T21:47:47"
null_hash: 8e4d67b544ab2035
---

# h3 — Conditioning on the recorded read length predicts a target track from a source track, beyond what a single value-axis map already does

Parent:: [[q2_do_the_covariates_that_record_which_read]]

## ELI5

Knowing that one run used shorter reads closes at least half of the gap a plain rescale leaves. Untested; early readings help on trained pairs only.

## TL;DR

The pair is a base ChIP track (101 bp reads) and its ENCODE re-run with reads cropped to 36 or 50 bp ([[t112_build_the_counterfactual_arms_for_t|the counterfactual-arms task]]). The checks (PI ruling 2026-10-31) ask g to beat its no-covariates twin by more than 2 × seed wobble per mark class, in counts and −log10 p separately, on base ↔ arm and never-trained arm → arm pairs; no per-knob scoring has run yet. Exploratory readings: a per-pair monotone map closes 0.46 of the count gap toward the oracle but 0.01 of the p gap. A generator g that reads the crop beats its no-covariates twin on trained pairs; the gain mostly fails on the never-trained 36 → 50 bp pair.

## Expect

On this knob's base ↔ arm pairs, g(C, C′) beats its no-covariates twin (C and C′ re-scrambled at every step) by more than 2 × g's seed wobble, per mark class, in counts and in −log10 p, separately.

_Switched to the scrambled-covariate twin by PI ruling on 2026-10-31, replacing the one-warp → oracle bar cloned on 2026-09-17. The readings already taken from the g × f ladder predate this switch, so against this bar they are post hoc; a per-knob scoring of the existing ladder runs is the planned test._

## Null
Normalization: the pairs differ by one global monotone value-axis map, so a covariate-free rescale already closes the gap and the conditioning adds nothing.

## Problem Statement

Read length changes mappability, duplicate marking and the fragment estimate together; whether the recorded length alone predicts the joint effect is open. It is one of four read-selection knobs under [[q2_do_the_covariates_that_record_which_read|the read-selection question]], with the design of [[h1_conditioning_on_the_recorded_sequencing_|the depth claim]]. The arm is ChIP only.

## Idea / Hypothesis

Conditioning on the recorded read length predicts a target track from a source track, beyond what a single value-axis map already does

## Verifiables

- [ ] beatstwin [claim-directed]: on the read length base ↔ arm pairs, D_twin − D_g > 2 × g's seed wobble (max pairwise |Δ| of D_g over 3 seeds), per mark class, in counts and in −log10 p, separately; D is NB CRPS for counts and log-normal CRPS for −log10 p, always split into oracle_scaled + scale_error
      fails-if:: the read length covariate adds nothing a scrambled-covariate twin of equal capacity cannot already do
- [ ] lawtest [claim-directed]: on never-trained read length arm → arm pairs, where the knob has two or more arm levels, g beats the twin by more than 2 × seed wobble
      fails-if:: g keeps one map per trained (C, C′) pair and has no answer for a read length combination it never saw
- [ ] shufflecollapse [outcome-neutral]: with C′ replaced at scoring by another arm's C′, g's advantage over the twin falls to within 2 × seed wobble
      fails-if:: g's f does not change when told the wrong target covariates, so any gain is not read from C′

## Planned Intervention

No run has tested this claim: [[t103_build_the_in_vitro_covariate_testbed|the testbed build]] has not run on this arm, and [[t107_run_the_in_vitro_covariate_testbed_o|the testbed run]] covers only depth and run type. The readings come from runs that trained one generator g on all ten knobs at once: the g × f grid, rows 1 and 2 ([[t118_learn_the_covariate_conditioned_tra|the grid task]]; `plan/T118_COUNTERFACTUAL_F.md`), and the encoder-decoder rungs ([[t126_build_and_run_the_encoder_decoder_r|the encoder-decoder task]]; `plan/T126_ENCDEC_A.md`). Both score chr19 + chr21 over 3 seeds against a no-covariates twin.

## Run Links

- g × f grid rows 1 and 2: `nibi:/project/def-maxwl/mforooz/t118/{ladder,row2}/runs` (row-2 arrays 22855414, 22862752; knob extraction 24641277)
- Encoder-decoder: `nibi:/project/def-maxwl/mforooz/t126/{runs,D/runs}` (train 24646366, 24647762; law 24646367, 24647788, 24706070)
- Products: `nibi:/scratch/mforooz/t112_cf/products/<track>__crop__{36,50}/` (scratch purge 2026-11-20)

## Artifacts

<!-- what the run produced. Keep files under results/h3/ and link at least the report:
     - [Report](results/h3/report.md)   - results/h3/curve.png -->
- [Per-knob readings for this knob family](results/q2/KNOB_READINGS.md) — the tables, the CRPS split and the caveats
- [Baseline rungs by arm](results/t118/rungs_v2.md)
- [Grid knob table, rows 1 and 2](results/t118/knob_generalization/knob_table.tsv) and its [figures](results/t118/knob_generalization/report.pdf)
- [Encoder-decoder knob table](results/t126/knob_table.tsv) and its [report](results/t126/report.pdf)
- [Grid deck](presentations/t118-grid/index.html) — published https://claude.ai/artifact/1mFz6C5jLaG1Ctc8JN572C
- [Encoder-decoder deck](presentations/t126-encdec/index.html) — published https://claude.ai/artifact/KP7pipaRbNbEStUaJR7jrZ
- [Counterfactual-arms manifest](results/t112/MANIFEST.tsv) and [summary](results/t112/COUNTERFACTUAL_ARMS.md) — published https://claude.ai/code/artifact/fb5ff45d-f155-4b01-ac50-db1179f4fd61

## Findings

Observed 2026-10-01 to 2026-10-21; not concluded. QuantileMatching, a per-pair monotone map, closes 0.46 (counts) and 0.01 (p) of the noSolution → oracle interval. Grid, gain over the twin, median of 16 configurations, bar 2× seed wobble: trained 0.18 in counts (16/16 met, median wobble 0.003) and 0.05 in p (6/16); never-trained 0.05 in counts (12/16) and −0.26 in p (1/16; 2 cells explode). Encoder-decoder, counts: old design D 0.20 trained, 0.19 never-trained; the learned-decoder rung 0.27–0.30 trained but −0.16 to −0.11 never-trained. Caveats: the twin is weaker than one-warp, and the oracle is a half-depth pseudoreplicate. Tables: [knob readings](results/q2/KNOB_READINGS.md).
