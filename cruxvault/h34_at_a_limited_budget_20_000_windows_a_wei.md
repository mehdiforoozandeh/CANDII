---
id: h34
type: idea
schema: 3
title: At a limited budget (20,000 windows), a weighted stratified draw beats the uniform draw
parent: q14
rule: 
measurement: 
replicates: 3 seeds per arm; rung 4 base at 20,000 windows on eic_v1; arms peakfreq_w, peakfreq_u, ccre_like_w, variance_w, kcenter against blocks_minus_eval_w20k and terr_r1024_w20k
created: "2026-11-13T12:42:24"
updated: "2026-11-13T12:42:24"
---

# h34 — At a limited budget (20,000 windows), a weighted stratified draw beats the uniform draw

Parent:: [[q14_how_should_training_regions_be_selected_]]

## ELI5

_(one sentence a science undergraduate follows — the correct terms, any beyond that glossed in 10 words or fewer)_

## TL;DR

_(one paragraph: what this claims, and what would settle it)_

## Expect

a weighted peak-based arm gains 1 to 2 seed spreads on headline 1 of the narrow marks; no arm passes the full rule; the unweighted peak arm is worse than uniform on pooled PIT deviation and on the broad marks by more than 1 seed spread

## Null

<!-- optional (spec 19): the null hypothesis — the most parsimonious alternative explanation
     under which this result appears although the claim is false. One line, naming a family.
     A finding records whether it was ruled out; a null NOT ruled out caps confidence at weak. -->
_(optional: one line naming the null hypothesis — a family from capacity, chance, leakage, selection, normalization, instrumentation)_

## Problem Statement

_(why this is worth testing)_

## Bears on

A confirmed win would be a proposal to the PI for the prototype sampler; no win keeps the uniform draw.

## Idea / Hypothesis

At a limited budget (20,000 windows), a weighted stratified draw beats the uniform draw

## Verifiables

<!-- optional (spec 19): checks that inform the closure judgment; they do not compute it. -->
- [ ] Deciding metric: the composite, with two vetoes.
- [ ] Settling rule: an arm wins only with a composite gain above 2 seed spreads, no scorer stratum, region class or broad mark worse by more than 2 of its own seed spreads, and pooled PIT deviation not worse; the best winner is confirmed once at 80,000 windows; a confirmed win is a proposal to the PI, not a change.

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Artifacts

<!-- what the run produced. Keep files under results/h34/ and link at least the report:
     - [Report](results/h34/report.md)   - results/h34/curve.png -->
_(none yet)_
