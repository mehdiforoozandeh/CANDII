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

Parked until the four sequence-blind designs have run. Here g reads the DNA sequence as well as C and C', so the transformation f can differ from bin to bin. The claim is that this beats the same design without sequence on the knobs whose effect depends on mappability or GC (MAPQ, read length, dedup, run type). Settled by comparing it with the per-bin affine design and with its own scrambled-covariate twin that keeps the sequence; the checks and bars are not written yet.

## Null

<!-- spec 09: the BORING explanation — the cheapest way this result could be trivially
     true. One line, <=25 words, naming a family. Your checks must discriminate against it.
     The PI approves it before checks are written: `crux approve-null <id>`. -->
_(one line: the cheapest way this result could be trivially true — name a family from capacity, chance, leakage, selection, normalization, instrumentation)_

## Problem Statement

PARKED by the PI 2026-09-25, to revisit after the A–D ladder (plan/T118_COUNTERFACTUAL_F.md). Option E of the architecture ladder: g reads C, C' AND the DNA sequence, and outputs per-bin parameters of f; f(X) still predicts X'. Options A–D apply the same f at every position, so they cannot express knob effects that act through sequence: MAPQ and read length through mappability, dedup through PCR GC bias, run type most in repeats. Draft of how g reads sequence (not ruled): a dilated 1D CNN on one-hot DNA pooled to 25 bp bins, with Umap mappability (k = 24, 36, 50, 100), GC per bin and repeat masks as extra channels, [C, C'] entering by feature-wise scale and shift; paired with option A made local (per-bin a_i, b_i, dispersion) so E versus A isolates what sequence adds. Alternatives listed: precomputed sequence tracks only; pretrained DNA-model features (Enformer/Borzoi, Nucleotide Transformer, Caduceus). Risk: g here is large while it sees few distinct (C, C') points, so it may learn the track's sequence pattern rather than covariate effects; the scrambled-covariate twin keeps the same sequence input, which catches that.

## Idea / Hypothesis

A generator that reads the DNA sequence as well as the source and target covariates outputs a position-dependent transformation that beats the same design without sequence on the knobs that act through mappability and GC

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
- [ ] _(state a falsifiable, pre-registered check)_

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

_(none yet)_

## Artifacts

<!-- what the run produced. Keep files under results/h16/ and link at least the report:
     - [Report](results/h16/report.md)   - results/h16/curve.png -->
_(none yet)_

## Findings

_(written by the PI/agent when the case is closed)_
