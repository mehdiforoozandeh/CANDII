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
Once the settings are known, how simple can the translator be — a stretch, a bendable curve, a look at neighbouring bins, or a small network?

## TL;DR
The parent asks whether the recorded covariates carry the information; this one asks how expressive f must be to use it. Rewritten 2026-09-25 for the g(C, C') → f design (the one-warp map, the oracle and gap-closed are gone). Its children are the rungs A–D, each a form of f whose parameters a small g reads from [C, C'], each judged against its own scrambled-covariate twin and against the rung below; a fifth rung where g also reads DNA sequence is parked. Settled by the highest rung that still beats the rung below beyond seed wobble: if only A beats its twin, the covariates act as a rescale; if C or D is needed, they act on shape.

## Question

How expressive must the transformation f be — a per-bin affine map, a per-bin monotone curve, a kernel plus curve, or a conditioned CNN — for a generator g(C, C') to capture what the covariates do?

## Protocol
Locked before any run: every rung is compared to its own scrambled-covariate twin and to the rung below, on the same pairs, split and scoring (plan/T118_COUNTERFACTUAL_F.md); rungs are read in order A, B, C, D.

## Answer so far

Drafted 2026-10-02; exploratory (outside the pre-registered ladder), not judged. In row 2, g also reads the bin's value x, so f changes per bin. Its shuffled-bin twin reads x from a random bin of the same chromosome. Numbers: `results/row2/GRID_SUMMARY.md`; ± is the seed wobble, the largest difference between two of 3 seeds.

- Counts: reading x fixes design A. One g per track, DNase CRPS as oracle-scaled + scale error (the part an ideal rescale removes): A 3.06 ± 0.11 = 1.36 + 1.7; A2 0.849 ± 0.009 = 0.8368 + 0.0121; twin 3.02 ± 0.0442 = 1.339 + 1.682. So the gain needs the bin's own value. B, C and D change little, except C2 on DNase.
- −log10 p: row 2, A2 included, has more (pair, seed) records with CRPS above 20 than row 1; the twin has no more than row 1. The spec predicted that A2 would likely remove them. Some row-2 means reach 10³–10⁶, with wobbles as large.
- Every row-2 design beats its no-covariates twin in counts in 10–12 of 12 cells. The depth law stays unmet (B2: 1 of 6 cells).

Open: a principled fix for the p-space explosion (the PI rejected capping σ and report-only); a hand check of the depth-law computation; why design D explodes without knots; [[t119_rebuild_the_two_dnase_mapq_arms_c12|the DNase MAPQ-arm rebuild]].

<!-- crux:ledger:start -->
**5 children** · ideas 0/5 done (supported 0, partial 0, refuted 0, inconclusive 0, invalid-run 0)

- `h12` [[h12_a_covariate_conditioned_affine_map_on_th|A generator g(C, C') that outputs a per-bin affine map on the log scale beats the same design trained with scrambled covariates, in count and −log10 p space]] — *running*
- `h13` [[h13_a_small_covariate_conditioned_convolutio|A generator g(C, C') that outputs a convolution kernel followed by a monotone curve beats its scrambled-covariate twin and the per-bin monotone-curve design]] — *running*
- `h14` [[h14_an_encoder_decoder_with_the_encoder_cond|A generator g(C, C') that modulates a small dilated convolutional network beats its scrambled-covariate twin and the kernel-and-curve design]] — *running*
- `h16` [[h16_a_generator_that_reads_the_dna_sequence_|A generator that reads the DNA sequence as well as the source and target covariates outputs a position-dependent transformation that beats the same design without sequence on the knobs that act through mappability and GC]] — *idea*
- `h17` [[h17_a_generator_g_c_c_that_outputs_a_per_bin|A generator g(C, C') that outputs a per-bin monotone curve on the log scale beats its scrambled-covariate twin and the per-bin affine design]] — *running*
<!-- crux:ledger:end -->
