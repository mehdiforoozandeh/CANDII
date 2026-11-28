---
id: h49
type: idea
schema: 3
title: The full-combination miniCANDI trained on 25 bp read-start counts, with its predictions converted to overlap counts, matches the same recipe trained on overlap counts on imputation, correction and calibration
parent: q75
rule: 
measurement: Headline (imputation_raw, correction_v3_raw, uncertainty_raw) and PR 65 deviance metrics on the eic_v1_2 16-region evaluation set, predictions converted by span kernel and NB moment match
replicates: 3 seeds per model; read-count reference reused from t370
created: "2026-11-24T23:49:58"
updated: "2026-11-24T23:49:58"
---

# h49 — The full-combination miniCANDI trained on 25 bp read-start counts, with its predictions converted to overlap counts, matches the same recipe trained on overlap counts on imputation, correction and calibration

Parent:: [[q75_can_one_minicandi_recipe_match_or_beat_t]]

## ELI5

_(one sentence a science undergraduate follows — the correct terms, any beyond that glossed in 10 words or fewer)_

## TL;DR

_(one paragraph: what this claims, and what would settle it)_

## Expect

Recorded before any result (t392 INSTRUCTIONS.md section 3): imputation ties within seed range or loses slightly because read starts are sparser; correction ties; calibration and deviance move a little in either direction.

## Null

<!-- optional (spec 19): the null hypothesis — the most parsimonious alternative explanation
     under which this result appears although the claim is false. One line, naming a family.
     A finding records whether it was ruled out; a null NOT ruled out caps confidence at weak. -->
_(optional: one line naming the null hypothesis — a family from capacity, chance, leakage, selection, normalization, instrumentation)_

## Problem Statement

_(why this is worth testing)_

## Bears on

A tie lets one read-start model serve our board and the EIC boards; a loss rules read-start training out as a recipe ingredient at this scale.

## Idea / Hypothesis

The full-combination miniCANDI trained on 25 bp read-start counts, with its predictions converted to overlap counts, matches the same recipe trained on overlap counts on imputation, correction and calibration

## Verifiables

<!-- optional (spec 19): checks that inform the closure judgment; they do not compute it. -->
_(optional: falsifiable checks, added when they can be defined — they inform the closure judgment, they do not compute it)_

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Artifacts

<!-- what the run produced. Keep files under results/h49/ and link at least the report:
     - [Report](results/h49/report.md)   - results/h49/curve.png -->
_(none yet)_

## Status notes (2026-11-25)

- The PI declined the 256-region scoring ("no") and put Phase 2, the transforms from predicted read starts to other signal formats, on hold ("hold for now"). Both were relayed by the foreman.
- The PI asked for a low-compute investigation of the causes of this negative result. Candidate causes are recorded as separate hypotheses under this question.
