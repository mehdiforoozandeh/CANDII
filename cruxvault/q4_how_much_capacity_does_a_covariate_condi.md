---
id: q4
type: question
schema: 2
title: How expressive must the transformation f be — a per-bin affine map, a per-bin monotone curve, a kernel plus curve, or a conditioned CNN — for a generator g(C, C') to capture what the covariates do?
parent: q1
status: open
stale: false
created: "2026-09-17T21:46:17"
updated: "2026-10-02T12:29:01"
---

# q4 — How expressive must the transformation f be — a per-bin affine map, a per-bin monotone curve, a kernel plus curve, or a conditioned CNN — for a generator g(C, C') to capture what the covariates do?

Parent:: [[q1_do_the_recorded_experimental_covariates_]]

## ELI5

How simple can the translator be: a stretch, a bendable curve, a look at neighbouring bins, or a small network? So far the bendable curve does most of the work.

## TL;DR

How expressive must f be to use the covariates? The children are rungs A–D (affine, monotone curve, kernel plus curve, conditioned CNN), each judged against its scrambled-covariate twin and the rung below; a sequence-reading rung is parked. All four ran in two rows: g reads C, C′, or also the bin value. Exploratory readings: in counts the curve gives the main step and shape adds little. Some −log10 p pairs explode. No rung carries its gain to never-trained pairs.

## Question

How expressive must the transformation f be — a per-bin affine map, a per-bin monotone curve, a kernel plus curve, or a conditioned CNN — for a generator g(C, C') to capture what the covariates do?

## Protocol
Locked before any run: every rung is compared to its own scrambled-covariate twin and to the rung below, on the same pairs, split and scoring (plan/T118_COUNTERFACTUAL_F.md); rungs are read in order A, B, C, D.

## Answer so far

CRPS on all bins, chr19 + chr21, one g per track, mean of 3 seeds ± seed wobble (largest difference between two seeds).

- Counts, row 1: DNase A 3.06 ± 0.11, B 0.824 ± 0.004, C 0.818 ± 0.027, D 0.772 ± 0.036; narrow 0.479, 0.392, 0.379, 0.375. A's excess is scale error (1.70 of 3.06). Beats the rung below: B 9 of 12 cells, C 5, D 1. Beats its twin: A 3, B–D 11.
- −log10 p: A beats its twin in 12 of 12 cells and never explodes. B, C and D explode on a few sparse-to-full-depth pairs (CRPS > 20; D's DNase mean is 1.9 × 10⁴). In B and C the lowest knot's σ drifts; D's cause is not found ([[t123_diagnose_design_d_s_log10_p_explosi|the D diagnosis]]). The fix is open ([[t124_find_a_principled_fix_for_the_log10|the p-space fix]]); the PI rejected capping σ.
- Row 2 (g reads the bin value x): A2 DNase 0.849 ± 0.009, against 3.02 for a twin that reads x from a random bin. B–D change little, except C2 on DNase (0.730 ± 0.003). In p, row 2 explodes more, A2 too, against the spec.
- Never-trained arm → arm pairs: the relative count gain over the twin falls from 0.29 to 0.11 on same-knob pairs (median of 16 settings). More capacity does not fix it: per track, D keeps 0.23, A nothing. The depth law is unmet in every cell except one of B2. This points at g, not at f's form ([[q7_why_do_the_covariate_conditioned_g_f_des|the depth-law question]]).

Also open: [[t119_rebuild_the_two_dnase_mapq_arms_c12|the DNase MAPQ-arm rebuild]].

<!-- crux:ledger:start -->
**5 children** · ideas 0/5 done (supported 0, partial 0, refuted 0, inconclusive 0, invalid-run 0)

- `h12` [[h12_a_covariate_conditioned_affine_map_on_th|A generator g(C, C') that outputs a per-bin affine map on the log scale beats the same design trained with scrambled covariates, in count and −log10 p space]] — *running*
- `h13` [[h13_a_small_covariate_conditioned_convolutio|A generator g(C, C') that outputs a convolution kernel followed by a monotone curve beats its scrambled-covariate twin and the per-bin monotone-curve design]] — *running*
- `h14` [[h14_an_encoder_decoder_with_the_encoder_cond|A generator g(C, C') that modulates a small dilated convolutional network beats its scrambled-covariate twin and the kernel-and-curve design]] — *running*
- `h16` [[h16_a_generator_that_reads_the_dna_sequence_|A generator that reads the DNA sequence as well as the source and target covariates outputs a position-dependent transformation that beats the same design without sequence on the knobs that act through mappability and GC]] — *idea*
- `h17` [[h17_a_generator_g_c_c_that_outputs_a_per_bin|A generator g(C, C') that outputs a per-bin monotone curve on the log scale beats its scrambled-covariate twin and the per-bin affine design]] — *running*
<!-- crux:ledger:end -->
