---
id: h16
type: idea
schema: 2
title: A generator that reads the DNA sequence as well as the source and target covariates outputs a position-dependent transformation that beats the same design without sequence on the knobs that act through mappability and GC
parent: q4
status: idea
rule: 
measurement: same pairs, split, scoring and competitor as the A–D ladder (plan/T118_COUNTERFACTUAL_F.md); compared against option A and against its own scrambled-covariate twin with sequence kept; reported per knob, with the mappability- and GC-linked knobs (MAPQ, crop, dedup, pe) singled out
replicates: 
verdict: 
metric: 
created: "2026-09-25T01:19:46"
updated: "2026-09-25T01:19:46"
---

# h16 — A generator that reads the DNA sequence as well as the source and target covariates outputs a position-dependent transformation that beats the same design without sequence on the knobs that act through mappability and GC

Parent:: [[q4_how_much_capacity_does_a_covariate_condi]]

## ELI5

Some processing settings matter more in repetitive, hard-to-map DNA, so a model that can read the DNA may learn those settings better.

## TL;DR

Idea stage: not run, and no null or checks yet. The PI parked it on 2026-09-25 until the four sequence-blind designs of [[q4_how_much_capacity_does_a_covariate_condi|the capacity ladder]] had run; they have now run. Here g also reads the DNA sequence, so f can differ from bin to bin. It should beat the same design without sequence on the knobs whose effect depends on mappability or GC.

## Expect

_Not stated. The PI parked this design on 2026-09-25 before it ran: "let's park option E for later and focus of A,B,C,D for now"._

## Null

<!-- spec 09: the BORING explanation — the cheapest way this result could be trivially
     true. One line, <=25 words, naming a family. Your checks must discriminate against it.
     The PI approves it before checks are written: `crux approve-null <id>`. -->
_(one line: the cheapest way this result could be trivially true — name a family from capacity, chance, leakage, selection, normalization, instrumentation)_

## Problem Statement

Designs A–D apply one f at every position, so they cannot express knob effects that act through sequence: MAPQ and read length through mappability, dedup through PCR GC bias, run type most in repeats. When g also read each bin's own value (design A2), counts CRPS on DNase fell from 3.06 ± 0.11 to 0.849 ± 0.009, and a twin that read a random bin's value did not gain (3.02). So a position-dependent f helps when it reads the right local signal; sequence is one, and g does not see it now.

Risk: a sequence-reading g is large but sees few distinct (C, C') points, so it may learn each track's sequence pattern, not covariate effects; the scrambled-covariate twin that keeps the sequence input catches that. The sequence-blind designs already fail to carry their gain to never-trained arm → arm pairs ([[h19_g_fits_trained_base_arm_depth_pairs_but_|trained versus never-trained]]), and a larger g may make that worse.

Origin: the PI's own addition, "let's add a fifth option where g gets C, C', and the DNA sequence as input" (2026-09-25). The row-2 (A2) argument above is the agent's, not the PI's.

## Idea / Hypothesis

A generator that reads the DNA sequence as well as the source and target covariates outputs a position-dependent transformation that beats the same design without sequence on the knobs that act through mappability and GC

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
- [ ] _(state a falsifiable, pre-registered check)_

## Planned Intervention

Draft of 2026-09-25, not ruled (option E in `plan/T118_COUNTERFACTUAL_F.md`, parked): a dilated 1D CNN on one-hot DNA pooled to 25 bp bins, with Umap mappability (k = 24, 36, 50, 100), GC per bin and repeat masks as extra channels, and [C, C'] entering by feature-wise scale and shift. It pairs with design A made local (per-bin a, b and dispersion), so the comparison with A isolates what sequence adds. Alternatives on record: precomputed sequence tracks only, or features from a pretrained DNA model (Enformer, Borzoi, Nucleotide Transformer, Caduceus).

## Run Links

_(none yet)_

## Artifacts

- [Grid summary: the row-2 evidence for a per-bin f](results/row2/GRID_SUMMARY.md)

## Findings

Not run; no readings.
