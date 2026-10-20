---
id: h19
type: idea
schema: 2
title: g fits trained base ↔ arm depth pairs but does not carry the fit over to never-trained arm → arm depth pairs
parent: q7
status: idea
rule:
measurement:
replicates:
verdict:
metric:
created: 2026-10-19T23:41:09
updated: 2026-10-19T23:41:09
---

# h19 — g fits trained base ↔ arm depth pairs but does not carry the fit over to never-trained arm → arm depth pairs

Parent:: [[q7_why_do_the_covariate_conditioned_g_f_des]]

## ELI5

The model learned each depth change it was shown, but it gets depth changes it was never shown wrong.

## TL;DR

Training saw only base → arm and arm → base pairs; the law test uses arm → arm pairs, for example 15M → 7.5M. If the miss is this failure, trained depth pairs fit the law and never-trained ones miss it. A fix then targets coverage or composition, not capacity or the loss. Two related readings are recorded beside it: chaining g's two trained legs through the base, and the same gap on knobs other than depth.

## Null

<!-- spec 09: the BORING explanation — the cheapest way this result could be trivially
     true. One line, <=25 words, naming a family. Your checks must discriminate against it.
     The PI approves it before checks are written: `crux approve-null <id>`. -->
_(one line: the cheapest way this result could be trivially true — name a family from capacity, chance, leakage, selection, normalization, instrumentation)_

## Problem Statement

_(why this is worth testing)_

## Idea / Hypothesis

g fits trained base ↔ arm depth pairs but does not carry the fit over to never-trained arm → arm depth pairs

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
- [ ] _(state a falsifiable, pre-registered check)_

## Planned Intervention

Readings from the existing row 1 and row 2 runs. Fixes that would test it: [[t126_build_and_run_the_encoder_decoder_r|the encoder-decoder design A task]] (depth 7.5M held out of training) and training on arm → arm pairs (discussed, not decided).

## Run Links

_(none yet)_

## Artifacts

<!-- what the run produced. Keep files under results/h19/ and link at least the report:
     - [Report](results/h19/report.md)   - results/h19/curve.png -->
- [Report: depth law and every knob, trained against never-trained pairs](results/h19/report.md)
- results/h19/knob_generalization/report.pdf five figures: gain over the no-covariates twin by knob, design and g, and the depth law
- results/h19/knob_generalization/verify.txt headlines recomputed from raw records and from the aggregate
- results/h19/depth_pairs/out_raw.txt depth-law error by design, g version and pair kind, row 1
- results/h19/depth_pairs/out_raw_row2.txt the same, row 2
- results/h19/depth_pairs/compose_out.txt the error of g's two trained legs chained through the base

## Findings

Observed 2026-10-19, not concluded. Median |log2 error| of the predicted scale, real g: row 1 trained 0.055–0.095, never-trained 0.25–0.55; row 2 trained 0.043–0.077, never-trained 0.19–0.47; every design, both g versions. Never-trained predictions overshoot: design A predicts −1.23 for a true −1 and +2.41 for +2. Chaining the two trained legs through the base gives 0.07–0.16, lower than g's direct answer. The across-track g is no better than the per-track g. Relative CRPS gain over the no-covariates twin, median over 16 design × g configurations: counts 0.29 trained, 0.11 never-trained same knob, 0.16 two knobs; −log10 p 0.15, 0.06, 0.004. Design D is least bad, then C; A and A2 are worst.