---
id: q4
type: question
schema: 2
title: How expressive must the transformation f be — a per-bin affine map, a per-bin monotone curve, a kernel plus curve, or a conditioned CNN — for a generator g(C, C') to capture what the covariates do?
parent: q1
status: open
stale: false
created: "2026-09-17T21:46:17"
updated: "2026-09-25T12:00:00"
---

# q4 — How expressive must the transformation f be — a per-bin affine map, a per-bin monotone curve, a kernel plus curve, or a conditioned CNN — for a generator g(C, C') to capture what the covariates do?

Parent:: [[q1_do_the_recorded_experimental_covariates_]]

## ELI5
Once the settings are known, how simple can the translator be — a stretch, a bendable curve, a look at neighbouring bins, or a small network?

## TL;DR
The parent asks whether the recorded covariates carry the information; this one asks how expressive f must be to use it. Rewritten 2026-09-25 for the g(C, C') → f design (the one-warp map, the oracle and gap-closed are gone). Its children are the rungs A–D, each a form of f whose parameters a small g reads from [C, C'], each judged against its own scrambled-covariate twin and against the rung below; a fifth rung where g also reads DNA sequence is parked. Settled by the highest rung that still beats the rung below beyond seed wobble: if only A beats its twin, the covariates act as a rescale; if C or D is needed, they act on shape.

## Question

How expressive must the transformation f be — a per-bin affine map, a per-bin monotone curve, a kernel plus curve, or a conditioned CNN — for a generator g(C, C') to capture what the covariates do?

## Protocol
Locked before any run: every rung is compared to its own scrambled-covariate twin and to the rung below, on the same pairs, split and scoring (plan/T118_COUNTERFACTUAL_F.md); rungs are read in order A, B, C, D.

## Answer so far

_(interpretation — written by the PI/agent; auto-flagged stale when new evidence lands)_

<!-- crux:ledger:start -->
**5 children** · ideas 0/5 done (supported 0, partial 0, refuted 0, inconclusive 0, invalid-run 0)

- `h12` [[h12_a_covariate_conditioned_affine_map_on_th|A generator g(C, C') that outputs a per-bin affine map on the log scale beats the same design trained with scrambled covariates, in count and −log10 p space]] — *idea*
- `h13` [[h13_a_small_covariate_conditioned_convolutio|A generator g(C, C') that outputs a convolution kernel followed by a monotone curve beats its scrambled-covariate twin and the per-bin monotone-curve design]] — *idea*
- `h14` [[h14_an_encoder_decoder_with_the_encoder_cond|A generator g(C, C') that modulates a small dilated convolutional network beats its scrambled-covariate twin and the kernel-and-curve design]] — *idea*
- `h16` [[h16_a_generator_that_reads_the_dna_sequence_|A generator that reads the DNA sequence as well as the source and target covariates outputs a position-dependent transformation that beats the same design without sequence on the knobs that act through mappability and GC]] — *idea*
- `h17` [[h17_a_generator_g_c_c_that_outputs_a_per_bin|A generator g(C, C') that outputs a per-bin monotone curve on the log scale beats its scrambled-covariate twin and the per-bin affine design]] — *idea*
<!-- crux:ledger:end -->
