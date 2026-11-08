---
id: h30
type: idea
schema: 3
title: Chaining one shared count-space correction model through the base transfers to never-trained arm-to-arm pairs across every count-moving knob, including cross-knob pairs, better than the direct route, and a pivot-consistency loss lets the direct route match the chained route
parent: q7
rule: 
measurement: held-out CRPS on never-trained same-knob and cross-knob arm-to-arm pairs, direct vs chained; depth law on depth pairs
replicates: chaining reuses the 3 shared correction-alone seeds; consistency arm 3 seeds
created: "2026-10-31T17:29:31"
updated: "2026-10-31T17:29:31"
---

# h30 — Chaining one shared count-space correction model through the base transfers to never-trained arm-to-arm pairs across every count-moving knob, including cross-knob pairs, better than the direct route, and a pivot-consistency loss lets the direct route match the chained route

Parent:: [[q7_why_do_the_covariate_conditioned_g_f_des]]

## ELI5

_(one sentence a science undergraduate follows — the correct terms, any beyond that glossed in 10 words or fewer)_

## TL;DR

_(one paragraph: what this claims, and what would settle it)_

## Expect

Over 6 seeds of the shared correction-alone model, routing through the base (inference chaining, leg 2 fed the leg-1 mean; 16-draw mixture as a second reading) lowers pooled held-out CRPS against the direct route by more than 2x seed spread in at least two thirds of never-trained (source knob, target knob) cells, cross-knob included, with the gain carried by oracle_scaled; chained depth-law error at most 0.16. The pivot-consistency arm (6 seeds) closes at least half of arm 1's direct-to-chained gap; this part is read only if arm 1's gap exceeds 2x spread. Prior reading (t128p3, old D): per-track chained 0.545 vs direct 0.662 (spread 0.008); shared chained 0.636 vs direct 0.667 (spread 0.064, within noise). Ceiling per cell: leg 2 fed the real base. Floor: the no-covariates twin.

_Pre-run: drafted by the pre-run reviewer on 2026-10-31; approved by the PI in chat on 2026-10-31 (6 seeds for the shared pivot model), before any run._

## Null

normalization — chaining's gain is total-depth re-anchoring at the base; oracle_scaled unchanged.

## Problem Statement

_(why this is worth testing)_

## Bears on

If chaining wins beyond depth, routing corrections through a base becomes a design input for mini-CANDI; if not, chaining only fixes the depth scale

## Idea / Hypothesis

Chaining one shared count-space correction model through the base transfers to never-trained arm-to-arm pairs across every count-moving knob, including cross-knob pairs, better than the direct route, and a pivot-consistency loss lets the direct route match the chained route

## Verifiables

- [ ] [outcome-neutral] `n1` — trained base-arm CRPS within old D's band (gap at most 0.006, t136 band); both routes scored on the same pairs, bins and CRPS estimator. fails-if:: the run measured nothing.
- [ ] [hypothesis] `v1` — in cells where routing wins, the oracle_scaled gain exceeds 2x spread in most of them. fails-if:: the gain is all scale_error. discriminates:: true
- [ ] [hypothesis] `v2` — the consistency arm's chained CRPS stays within 2x spread of arm 1's chained CRPS. fails-if:: the direct route matches only because the chained route got worse.
Runs: 3 extra shared correction-alone seeds (reusing h29's 3) + consistency arm 6 seeds. Joint power ~51%. Shared direct and chained share read noise between arm and arm', which favours direct, so a routed win is conservative.

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Artifacts

<!-- what the run produced. Keep files under results/h30/ and link at least the report:
     - [Report](results/h30/report.md)   - results/h30/curve.png -->
_(none yet)_
