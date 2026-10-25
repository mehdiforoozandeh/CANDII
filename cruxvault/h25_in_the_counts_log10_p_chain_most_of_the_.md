---
id: h25
type: idea
schema: 2
title: In the counts → −log10 p chain, most of the p error comes from f's predicted count distribution, not from the map h
parent: q6
status: idea
rule: 
measurement: CRPS on P in a 2 × 2 of {f's NB, true X′ given f's dispersion} × {bin formula, best learned h}, same pairs and seeds; share of the excess over the direct-p model removed by each swap
replicates: the part-2 pairs, designs D and D2, 3 f seeds
verdict: 
metric: 
created: "2026-10-25T00:22:33"
updated: "2026-10-25T00:58:40"
null_approved: "2026-10-25T00:58:40"
null_hash: 8f920311d2c7a411
---

# h25 — In the counts → −log10 p chain, most of the p error comes from f's predicted count distribution, not from the map h

Parent:: [[q6_how_closely_does_a_map_h_reproduce_the_l]]

## ELI5

The model first guesses read counts and then turns them into p-values, and most of its mistake is probably in the count guess, not in the conversion.

## TL;DR

A map h turns a count distribution and its matched control into a −log10 p track. With f's predicted NB as input, every h, from the bin formula to the learned MLPs, scores worse than the direct-p model (CRPS on P 0.178–0.276 against 0.143 ± 0.0016, the reference). The claim: f causes most of this excess. The test scores a 2 × 2 of {f's NB, the true X′ given f's dispersion} × {bin formula, best learned h} on the same pairs and seeds. It is supported if the swap to X′ removes at least 0.7 of the excess and the swap to the best h removes at most 0.2 (shares the PI's to confirm); it is refuted if the h swap removes more than the X′ swap. The existing "h alone" cells feed h a point X′, not an NB with f's dispersion, so they do not score this test.

## Null

<!-- spec 09: the BORING explanation — the cheapest way this result could be trivially
     true. One line, <=25 words, naming a family. Your checks must discriminate against it.
     The PI approves it before checks are written: `crux approve-null <id>`. -->
Instrumentation: −log10 p is a fixed function of counts and control, so h fed the true counts reproduces it by construction.

## Problem Statement

Fed f's predicted NB, every h from the bin-only formula to the learned MLPs scores worse on −log10 p than the earlier direct-p model, and the learned h does not beat the formula; fed the true counts, h fits closely.

## Idea / Hypothesis

In the counts → −log10 p chain, most of the p error comes from f's predicted count distribution, not from the map h

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
- [ ] _(state a falsifiable, pre-registered check)_

## Planned Intervention

Not built. It reuses the h runs of [[t125_build_and_run_the_depth_offset_coun|the depth-offset task]], part 2 (designs D and D2), and adds the X′-with-f's-dispersion input. Design: `plan/ETA_OFFSET_AND_H_DESIGN.md`.

## Run Links

_(none yet)_

## Artifacts

- [Chain against the capacity grid](results/t125/h_vs_oldgrid.tsv), [h alone](results/t125/h_alone_summary.tsv), [h in the chain](results/t125/h_chain_summary.tsv), [part-2 digest](results/t125/part2_digest.md)

## Findings

Not run. The motivating readings are in [[q6_how_closely_does_a_map_h_reproduce_the_l|the counts-to-p question]]: in the chain the bin formula is the best h (0.178), and the gap to the direct-p model is oracle-scaled, not scale error.
