---
id: q95
type: question
schema: 3
title: Why does miniCANDI fail to follow the deduplication and single-end/paired-end covariates while it follows depth, and can a closed-form processing layer or more counterfactual views fix it?
parent: q80
created: "2026-11-27T16:24:08"
updated: "2026-11-27T16:24:08"
---

# q95 — Why does miniCANDI fail to follow the deduplication and single-end/paired-end covariates while it follows depth, and can a closed-form processing layer or more counterfactual views fix it?

Parent:: [[q80_does_the_final_minicandi_follow_a_change]]

## ELI5

_(one sentence a science undergraduate follows — the correct terms, any beyond that glossed in 10 words or fewer)_

## TL;DR

_(one paragraph: what this asks, and what would settle it)_

## Question

Why does miniCANDI fail to follow the deduplication and single-end/paired-end covariates while it follows depth, and can a closed-form processing layer or more counterfactual views fix it?

## Bears on

Explains the failing prompts and decides whether the EIC-processing correction claim can hold.

## Protocol

<!-- optional (engine 1.4 / spec 11): the pre-registered rules — endpoints, arms, thresholds,
     scope — fixed BEFORE any run. `crux deck` surfaces this as the pre-registration note. -->
_(optional: the pre-registered rules — endpoints, thresholds, scope — locked before any run)_

<!-- crux:ledger:start -->
**4 children** · open 2, leaning 2, settled 0, parked 0 · 1 finding(s) on this question

- `h58` [[h58_deduplication_is_a_per_position_cap_whos|Deduplication is a per-position cap whose expected effect is closed form in read-start space: kept starts per strand per 25 bp bin = 25(1-(1+lambda/(25a))^-a), with one fitted shape a per assay.]] — *leaning · moderate*
- `h59` [[h59_the_dedup_failure_is_a_missing_mechanism|The dedup failure is a missing mechanism, not missing data: correction of dedup does not rise with more dedup-off views, but rises with a closed-form cap layer.]] — *open*
- `h60` [[h60_the_se_pe_failure_is_first_a_data_gap_38|The SE/PE failure is first a data gap: 38 SE views covering 13 of 36 assays and 12 PE base experiments are too few.]] — *open*
- `h61` [[h61_our_log2_depth_counts_both_mates_of_pair|Our log2_depth counts both mates of paired-end reads, so PE depth is about 2x the depth MACS2 and our counts see.]] — *leaning · moderate*
- `f160` finding (weak) [[f160_se_pe_data_only_on_the_12_se_views_with_|SE/PE, data-only: on the 12 SE views with a PE parent, neither the…]]
<!-- crux:ledger:end -->
