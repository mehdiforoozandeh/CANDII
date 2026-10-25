---
id: h23
type: idea
schema: 2
title: On never-trained arm → arm depth pairs, the shared-weight encoder-decoder FiLM CNN meets the depth law at the trained-pair level
parent: q7
status: idea
rule: 
measurement: median |log2 error| of the predicted scale log2(Σμ/ΣX) against the log2 depth ratio, never-trained arm → arm depth pairs vs trained base ↔ arm pairs, chr19 + chr21, counts only; old one-pass FiLM CNN (design D) as the control
replicates: 7 tracks × 3 seeds × both g versions; the 7.5M pairs reported separately
verdict: 
metric: 
created: "2026-10-25T00:22:33"
updated: "2026-10-25T00:22:33"
builds_on: h22
---

# h23 — On never-trained arm → arm depth pairs, the shared-weight encoder-decoder FiLM CNN meets the depth law at the trained-pair level

Parent:: [[q7_why_do_the_covariate_conditioned_g_f_des]]

## ELI5

One small network cleans a track to a standard setting and then changes it to the target setting; the test is whether it gets the depth right for pairs of settings it never saw in training.

## TL;DR

One g reads two covariate vectors and sets the FiLM (per-layer scale and shift) of one dilated CNN f. The encoder runs f with FiLM g(c, C_clean) and outputs a per-bin latent z; the decoder runs the same f, same weights, with FiLM g(C_clean, c′) and outputs NB(loc, disp) for the target counts. C_clean is a learned covariate vector whose assay entries are copied from the source. Training uses base ↔ arm pairs only. The depth law says that for two runs that differ only in depth, log2(Σμ / ΣX) equals log2 of the depth ratio. The claim: on never-trained arm → arm depth pairs, the median |log2 error| of the predicted scale is within a tolerance δ of its trained-pair value on all 3 seeds; it is refuted if the gap exceeds δ on 2 or more seeds. The control is the one-pass FiLM CNN (design D) under the same training, which must stay above δ, or the check does not discriminate. δ is the PI's to set. The 7.5M pairs, held out of training, are reported separately.

## Null

<!-- spec 09: the BORING explanation — the cheapest way this result could be trivially
     true. One line, <=25 words, naming a family. Your checks must discriminate against it.
     The PI approves it before checks are written: `crux approve-null <id>`. -->
Selection: the never-trained arm → arm depth ratios lie inside the trained ratios, so interpolation reaches the trained-pair error without learning the law.

## Problem Statement

The composable encoder-decoder with an exact inverse did not close the depth-law miss; the PI then designed a symmetric form (2026-10-21) in which one g sets the FiLM of one CNN for both halves through a learned clean middle setting.

## Idea / Hypothesis

On never-trained arm → arm depth pairs, the shared-weight encoder-decoder FiLM CNN meets the depth law at the trained-pair level

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
- [ ] _(state a falsifiable, pre-registered check)_

## Planned Intervention

[[t128_build_and_run_phase_1_of_the_symmet|The symmetric encoder-decoder task]], phase 1: 8 settings (cross loss plus round-trip loss, deterministic, in f forms A and D; round-trip loss alone, deterministic and VAE; a latent-agreement term at λ = 0.1, 1, 10; cross plus round trip as a VAE) × 8 g's (7 per-track, 1 across-track) × 3 seeds, plus no-covariates twins for two settings: 240 runs, counts only. Design: `plan/T126_ENCDEC_D2.md`. Exploratory: no bar is set yet.

## Run Links

- Fir and Nibi, as recorded in `results/t128/FIR_PATH.txt` and `plan/T118_STATUS.md`; old D re-scored with the CRPS split: Nibi job 24964406.

## Artifacts

- [First-pass readings](results/t128/analyze_out.txt), [depth table](results/t128/depth_table.tsv)
- [Encoder-decoder readings so far](results/h22/report.md)

## Findings

Not closed. First pass (2026-10-24), median |log2 error| on never-trained depth pairs: the two complete cross-plus-round-trip settings 0.31–0.42 and 0.29–0.39, λ = 0.1 (11 of 24 runs) 0.32, against old D 0.29–0.34 and trained pairs 0.05–0.09. The settings trained on the round-trip loss alone miss trained pairs too (1.4–2.4), so the cross loss is needed. Chaining old D's two trained maps through the base gives 0.07–0.12. Four of eight settings are unfinished (λ = 1 and 10, VAE).
