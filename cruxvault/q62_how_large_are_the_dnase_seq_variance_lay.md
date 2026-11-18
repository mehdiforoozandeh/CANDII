---
id: q62
type: question
schema: 3
title: How large are the DNase-seq variance layers (read sampling, replicate, cell type, other experiment) when the 15,201-read track ENCSR000EKG is left out of the common depth of the DNase-seq pairs?
parent: q18
created: "2026-11-18T16:22:25"
updated: "2026-11-18T16:22:25"
---

# q62 — How large are the DNase-seq variance layers (read sampling, replicate, cell type, other experiment) when the 15,201-read track ENCSR000EKG is left out of the common depth of the DNase-seq pairs?

Parent:: [[q18_how_large_is_each_layer_of_the_variance_]]

## ELI5

_(one sentence a science undergraduate follows — the correct terms, any beyond that glossed in 10 words or fewer)_

## TL;DR

_(one paragraph: what this asks, and what would settle it)_

## Question

Open for the next session (ready queue entry 20): rerun the layers stage for DNase-seq without ENCSR000EKG, or with a minimum common depth, then the two-sided ladder report. Evidence of the defect: results/closeout_E4/ladder_two_sided_2026-11-18/COMPARE.md.

## Bears on

Gives the first interpretable DNase-seq layer values of the replicate ladder and the second-experiment layers; until this rerun, q18 has no answer for DNase-seq, because every DNase-seq pair was computed at 15,201 reads (the next shallowest side of a DNase-seq pair has 13,300,584 reads).

## Protocol

<!-- optional (engine 1.4 / spec 11): the pre-registered rules — endpoints, arms, thresholds,
     scope — fixed BEFORE any run. `crux deck` surfaces this as the pre-registration note. -->
_(optional: the pre-registered rules — endpoints, thresholds, scope — locked before any run)_

<!-- crux:ledger:start -->
_(no children yet)_
<!-- crux:ledger:end -->
