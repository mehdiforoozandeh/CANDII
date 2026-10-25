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

Training saw only base → arm and arm → base pairs; the law test uses arm → arm pairs, for example 15M → 7.5M. If the miss is this failure, trained depth pairs fit the law and never-trained ones miss it, and a fix targets coverage or composition, not capacity or the loss. Readings so far match this pattern in every design, loss and g version tried. Chaining g's two trained legs through the base meets the law.

## Null

<!-- spec 09: the BORING explanation — the cheapest way this result could be trivially
     true. One line, <=25 words, naming a family. Your checks must discriminate against it.
     The PI approves it before checks are written: `crux approve-null <id>`. -->
_(one line: the cheapest way this result could be trivially true — name a family from capacity, chance, leakage, selection, normalization, instrumentation)_

## Problem Statement

The transformation grid of [[q4_how_much_capacity_does_a_covariate_condi|the capacity ladder]] misses the depth law, although the data obey it ([[q7_why_do_the_covariate_conditioned_g_f_des|the depth-law question]]). Before a fix is chosen, we must know whether the miss is everywhere or only on pair types that training never showed.

## Idea / Hypothesis

g fits trained base ↔ arm depth pairs but does not carry the fit over to never-trained arm → arm depth pairs

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
- [ ] _(state a falsifiable, pre-registered check)_

## Planned Intervention

Readings from existing runs: rows 1 and 2 of the grid (1,344 runs, 16 design × g configurations, 3 seeds, scored on chr19 + chr21); the loss head-to-head of [[t127_head_to_head_poisson_nll_mean_nb_nl|the loss task]]; the encoder-decoder runs of [[t126_build_and_run_the_encoder_decoder_r|the encoder-decoder task]] (depth 7.5M held out). The via-base reference of [[t128_build_and_run_phase_1_of_the_symmet|the symmetric encoder-decoder task]] runs old D's two trained maps in series, arm → base → arm, on the data (`plan/T126_ENCDEC_D2.md`).

## Run Links

- Rows 1 and 2 (Nibi): `/project/def-maxwl/mforooz/t118/ladder/runs`, `/project/def-maxwl/mforooz/t118/row2_xshuf/runs` ([meta](results/h19/knob_generalization/agg_r12_meta.txt)).
- Encoder-decoder runs (Nibi): `/project/def-maxwl/mforooz/t126/`; job ids in [results/t126/FIR_PATH.txt](results/t126/FIR_PATH.txt).
- Via base (Nibi): `/project/def-maxwl/mforooz/t128/viabase/`, job 24896767; readings in [results/t128/FIR_PATH.txt](results/t128/FIR_PATH.txt).

## Artifacts

<!-- what the run produced. Keep files under results/h19/ and link at least the report:
     - [Report](results/h19/report.md)   - results/h19/curve.png -->
- [Report: depth law and every knob, trained against never-trained pairs](results/h19/report.md)
- [Five figures: gain over the no-covariates twin by knob, design and g, and the depth law](results/h19/knob_generalization/report.pdf)
- [Headlines recomputed from raw records and from the aggregate](results/h19/knob_generalization/verify.txt)
- [Depth-law error by design, g version and pair kind, row 1](results/h19/depth_pairs/out_raw.txt)
- [The same, row 2](results/h19/depth_pairs/out_raw_row2.txt)
- [g's two trained legs chained through the base, from predicted scales](results/h19/depth_pairs/compose_out.txt)
- [Encoder-decoder depth table, with the held-out 7.5M group](results/t126/depth_table.tsv)
- [Via-base depth table (arm Dviabase)](results/t128/depth_table.tsv)
- [Loss head-to-head report](results/t127/report.md)

## Findings

Exploratory readings: median |log2 error| of the predicted scale on depth pairs.

- Rows 1 and 2, real g: trained 0.04–0.10, never-trained 0.19–0.55, in every design and g version. Never-trained predictions overshoot: design A predicts −1.23 for a true −1 and +2.41 for +2. The across-track g is no better.
- The gap holds for other knobs. Relative CRPS gain over the no-covariates twin, median over 16 configurations: counts 0.29 trained, 0.11 never-trained same knob, 0.16 two knobs.
- The gap survives a law-capable f (rung A0: trained 0.12, never-trained 0.38–0.46) and a Poisson-NLL mean (never-trained 0.28–0.38).
- A depth level unseen in training is easy on a trained pair type: base ↔ 7.5M gives 0.04–0.12 for old A, old D and encoder-decoder D.
- Chaining two trained legs meets the law. From predicted scales: 0.07–0.16 (rows 1 and 2). Running old D's two maps in series on the data: 0.12 per-track (seed spread 0.05) and 0.07 across-track (0.11), against direct 0.29 and 0.34; the per-track gap exceeds 2 × seed spread.
