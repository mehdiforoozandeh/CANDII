---
id: h22
type: idea
schema: 2
title: A composable g — one shared map from each side's covariates to a common latent, with an exact inverse — meets the depth law on never-trained depth pairs without arm → arm training
parent: q7
status: idea
rule:
measurement:
replicates:
verdict:
metric:
created: 2026-10-19T23:41:09
updated: 2026-10-19T23:41:09
---

# h22 — A composable g — one shared map from each side's covariates to a common latent, with an exact inverse — meets the depth law on never-trained depth pairs without arm → arm training

Parent:: [[q7_why_do_the_covariate_conditioned_g_f_des]]

## ELI5

If the model maps every run to a shared middle ground and back, it can connect any two runs, even a pair it never saw.

## TL;DR

One network φ maps one side's covariates c to (α(c), β(c)). The source is encoded as z = (log(1 + X) − α(c)) / β(c) and decoded as log μ = α(c′) + β(c′)·z, so an arm → arm pair composes two trained legs by construction. If composition is what is missing, this design brings never-trained depth pairs to the trained-pair error and the retrained old design A does not. First readings: it does not, nor do two learned-decoder variants.

## Null

<!-- spec 09: the BORING explanation — the cheapest way this result could be trivially
     true. One line, <=25 words, naming a family. Your checks must discriminate against it.
     The PI approves it before checks are written: `crux approve-null <id>`. -->
_(one line: the cheapest way this result could be trivially true — name a family from capacity, chance, leakage, selection, normalization, instrumentation)_

## Problem Statement

Old g reads [c, c′] together, so an arm → arm pair is an input it never saw. Chaining its two trained legs through the base meets the depth law (0.07–0.16 log2) where its direct answer does not ([[h19_g_fits_trained_base_arm_depth_pairs_but_|the trained versus never-trained hypothesis]]). Real data has few arm → arm examples, so the fix must come from g's structure.

## Idea / Hypothesis

A composable g — one shared map from each side's covariates to a common latent, with an exact inverse — meets the depth law on never-trained depth pairs without arm → arm training

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
- [ ] _(state a falsifiable, pre-registered check)_

## Planned Intervention

[[t126_build_and_run_the_encoder_decoder_r|The encoder-decoder task]] (`plan/T126_ENCDEC_A.md`). Rung A: φ with an exact inverse, β learned and β ≡ 1, against old A retrained. Rung D (PI, 2026-10-20): a FiLM CNN split into an encoder set by g_enc(c) and a decoder set by g_dec(c′), with λ·mse(Z, Z′), λ ∈ {0, 0.1, 1, 10}, against old D retrained; no exact inverse. Every arm: 8 g's (7 per-track, 1 across-track), 3 seeds, counts, a no-covariates twin, depth 7.5M held out; 384 runs.

## Run Links

- Rung A (Nibi): `/project/def-maxwl/mforooz/t126/runs/`, train 24646366, law 24646367.
- Rung D (Nibi): `/project/def-maxwl/mforooz/t126/D/runs/`, train 24647762, law 24647788, re-run 24706070 (tasks 73, 74, 174).
- Readings (Nibi): `/project/def-maxwl/mforooz/t126/readings/`, extract 24706111, re-check 24706606 (24706110 superseded).
- Symmetric follow-up (Fir and Nibi): `/project/def-maxwl/mforooz/t128/`; job ids in [results/t128/FIR_PATH.txt](results/t128/FIR_PATH.txt).

## Artifacts

<!-- what the run produced. Keep files under results/h22/ and link at least the report:
     - [Report](results/h22/report.md)   - results/h22/curve.png -->
- [Encoder-decoder readings report](results/t126/report.pdf)
- [Lab deck: encoder-decoder g](presentations/t126-encdec/DECK.md) — published https://claude.ai/artifact/KP7pipaRbNbEStUaJR7jrZ ([built page](presentations/t126-encdec/index.html))
- [Depth-law table by arm, g version and pair group](results/t126/depth_table.tsv)
- [Trained CRPS with the oracle_scaled / scale_error split](results/t126/fit_table.tsv)
- [Gain over the twin per knob](results/t126/knob_table.tsv)
- [Learned (a, b) on depth pairs](results/t126/ab_table.tsv)
- [Best step and runs at the step cap](results/t126/steps_table.tsv)
- [λ selection on chr22](results/t126/lambda_table.tsv)
- [Second-method re-check](results/t126/verify.txt)
- [Run locations](results/t126/FIR_PATH.txt)
- [Symmetric follow-up, first-pass readings](results/t128/analyze_out.txt)
- [Symmetric follow-up depth table](results/t128/depth_table.tsv)

## Findings

Exploratory readings, median |log2 error| on depth pairs ([report](results/h22/report.md)). Noise floor: old D's never-trained seed spread, 0.05 (per-track g) and 0.24 (across-track g).

- Encoder-decoder A misses trained pairs too (1.0–1.4, never-trained 1.9–2.0; old A 0.09–0.12, 0.38–0.45); its CRPS is mostly scale error (per-track 1.105: oracle_scaled 0.679, scale_error 0.426; old A 0.872: 0.589, 0.283).
- Encoder-decoder D keeps old D's trained fit (0.04–0.10; CRPS 0.448 against 0.445, scale_error 0.004 both, seed spread 0.004) but misses never-trained pairs as much (0.26–0.52 against 0.29–0.34). It falls below its twin on never-trained read-length, MAPQ and control-mixed-in pairs; 66 of 96 runs hit the step cap.
- First pass 2026-10-24 ([[t128_build_and_run_phase_1_of_the_symmet|the symmetric encoder-decoder task]], no exact inverse): never-trained 0.29–0.42, within noise of old D; four of eight settings are unfinished.
