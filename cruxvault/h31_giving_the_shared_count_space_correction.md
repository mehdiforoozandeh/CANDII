---
id: h31
type: idea
schema: 3
title: Giving the shared count-space correction model a mappability track, or one-hot DNA sequence, as extra input improves correction on read-length and MAPQ arms without harming the other knobs
parent: q11
rule: 
measurement: held-out CRPS per knob arm, against the shared correction-alone model
replicates: 2 input arms x 3 seeds = 6 runs
created: "2026-10-31T17:29:31"
updated: "2026-10-31T17:29:31"
---

# h31 — Giving the shared count-space correction model a mappability track, or one-hot DNA sequence, as extra input improves correction on read-length and MAPQ arms without harming the other knobs

Parent:: [[q11_where_along_the_genome_does_each_covaria]]

## ELI5

_(one sentence a science undergraduate follows — the correct terms, any beyond that glossed in 10 words or fewer)_

## TL;DR

_(one paragraph: what this claims, and what would settle it)_

## Expect

The mappability arm (per-bin 36-mer mappability as an extra input channel to the shared CNN) lowers held-out CRPS in the lowest-mappability decile (dec0, blacklist masked, strata fixed before the run) against the same model without it by more than 2x the comparator's seed SD, on at least 5 of 7 tracks for read-length crop 36 and at least 4 of 6 tracks for MAPQ 0 (DNase MAPQ arms broken); crop 50 read separately; dec0 CRPS on depth, dedup and fragment extension changes by less than wobble. Genome-wide CRPS is secondary. The one-hot DNA arm is a reading against both, capped at weak.

_Pre-run: drafted by the pre-run reviewer on 2026-10-31; approved by the PI in chat on 2026-10-31, before any run. Held-out chromosomes (19, 21) are disjoint from training (plan/T118_COUNTERFACTUAL_F.md:90)._

## Null

capacity — the extra input channel adds parameters; any per-bin channel, aligned or not, lowers CRPS.

## Problem Statement

_(why this is worth testing)_

## Bears on

If it helps, mini-CANDI needs a position-aware input before it adds read length as a covariate; if not, the mappability effect is either small or not learnable from 7 tracks

## Idea / Hypothesis

Giving the shared count-space correction model a mappability track, or one-hot DNA sequence, as extra input improves correction on read-length and MAPQ arms without harming the other knobs

## Verifiables

- [ ] [outcome-neutral] the comparator reproduces the shared correction-alone seeds within wobble; the depth-arm change is less than wobble; before training, the comparator's dec0 residual on MAPQ 0 is positive (t141 sign).
- [ ] [hypothesis] `shiftnull` — the mappability arm beats a 5 Mb circularly shifted mappability arm in dec0 on MAPQ 0 and crop 36 by more than 2x wobble. fails-if:: the shifted arm keeps most of the gain. discriminates:: true
- [ ] [hypothesis] `localise` — the dec0 gain exceeds the dec9 gain on MAPQ 0 and crop 36 by more than wobble. fails-if:: the gain is even across deciles.
Runs: mappability 3, shifted mappability 3, one-hot DNA 3 = 9. Joint power ~41%.


**Post-smoke amendments (PI rulings, 2026-10-31), made after the single-seed smoke test, before any full run.** (1) 36-mer mappability equals 1.0 on 36% of 1 kb blocks, so tied-rank deciles leave the top deciles empty; the high stratum for `localise` is the top non-empty decile (dec8, every fully mappable block), keyed `dectop`, in place of dec9. (2) Strata are genome-wide deciles of 1 kb mean mappability, blacklist dropped; the oracle scale is refitted per stratum. (3) The DNA encoder is Conv(4→16, k15), GELU, 25 bp average pool, Conv(16→4).
## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Artifacts

<!-- what the run produced. Keep files under results/h31/ and link at least the report:
     - [Report](results/h31/report.md)   - results/h31/curve.png -->
_(none yet)_
