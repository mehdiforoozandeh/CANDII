---
id: h29
type: idea
schema: 3
title: Adding a cross-assay imputation or masked-bin denoising objective to one shared count-space correction model changes how well the correction transfers to never-trained arm-to-arm pairs, compared with the same model trained on correction alone
parent: q12
rule: 
measurement: held-out CRPS on never-trained arm-to-arm pairs; depth-law error on never-trained depth pairs; chr19+chr21; shared CNN (design D) over 7 tracks
replicates: 4 arms (alone shared, per-track reference, cross-assay, denoising) x weights 0.25/1/4 for the two joint arms x 3 seeds = 24 runs
created: "2026-10-31T17:29:31"
updated: "2026-10-31T17:29:31"
---

# h29 — Adding a cross-assay imputation or masked-bin denoising objective to one shared count-space correction model changes how well the correction transfers to never-trained arm-to-arm pairs, compared with the same model trained on correction alone

Parent:: [[q12_when_covariate_correction_is_trained_joi]]

## ELI5

_(one sentence a science undergraduate follows — the correct terms, any beyond that glossed in 10 words or fewer)_

## TL;DR

_(one paragraph: what this claims, and what would settle it)_

## Expect

At imputation weight 1 (primary, one comparison per objective), the cross-assay or the masked-bin denoising arm moves held-out CRPS on never-trained arm-to-arm pairs (chr19 + chr21, all knobs) by more than 2x its seed wobble against shared correction alone (old shared D: CRPS 0.720, oracle_scaled 0.656, scale_error 0.064, spread 0.022, so bar ~0.044). Direction is a separate sub-claim each way (improves / degrades). The never-trained minus trained gap moves the same way. Weights 0.25 and 4 are a secondary dose-response reading. The depth law is a probe only (spread 0.24 is too wide for a 2x bar). Every CRPS is reported with its oracle_scaled / scale_error split.

_Pre-run: drafted by the pre-run reviewer on 2026-10-31; approved by the PI in chat ("i) yes") on 2026-10-31, before any run._

## Null

normalization — the auxiliary loss changes the correction's gradient scale; a matched-scale correction-alone arm reproduces the shift.

## Problem Statement

_(why this is worth testing)_

## Bears on

A change beyond seed noise means testbed negatives on model form do not carry over to joint training, so CANDI's objective design stays open; no change means the testbed lessons transfer

## Idea / Hypothesis

Adding a cross-assay imputation or masked-bin denoising objective to one shared count-space correction model changes how well the correction transfers to never-trained arm-to-arm pairs, compared with the same model trained on correction alone

## Verifiables

- [ ] [outcome-neutral] `driftctl` — the new shared correction-alone arm reproduces old shared D's never-trained CRPS 0.720 within 2x wobble. fails-if:: the kit drifted, so the run measured nothing.
- [ ] [outcome-neutral] `leakaudit` — no (source, target) covariate pair used in training equals a scored arm-to-arm pair; every imputation source is a base track. fails-if:: never-trained pairs were trained on.
- [ ] [outcome-neutral] early stopping on chr22 uses the correction loss only.
- [ ] [hypothesis] `notwrecked` — each joint arm beats its scrambled-covariate twin on trained pairs by more than 2x wobble.
- [ ] [hypothesis] `transfernotfit` — the never-trained CRPS shift exceeds the trained-pair shift by more than 2x wobble. fails-if:: the joint loss moves everything uniformly.
- [ ] [hypothesis] `matchedarm` — the joint arm differs from correction alone with matched loss scale by more than 2x wobble. fails-if:: the null holds. discriminates:: true
Runs: shared correction alone 3, matched-scale arm 3, 2 objectives x 3 weights x 3 seeds 18 = 24; per-track reference reused (fir:/project/def-maxwl/mforooz/t128p2/runs/D_<g>_counts_real_s{0,1,2}). Joint power ~41% under rule all.

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Artifacts

<!-- what the run produced. Keep files under results/h29/ and link at least the report:
     - [Report](results/h29/report.md)   - results/h29/curve.png -->
_(none yet)_
