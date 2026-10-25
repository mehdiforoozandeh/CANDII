---
id: h26
type: idea
schema: 2
title: A trained CANDI checkpoint, told a counterfactual arm's read length or run type in place of the base's, moves its predicted counts no closer to the arm's real track than when told the base's own covariates
parent: q8
status: idea
rule: 
measurement: "Covariate-swap probe: per track x arm, change in predicted log2 mean (told arm vs told base covariates) against the real arm/base log2 ratio, and NB NLL of the arm track under each prompt; CPU inference on a saved checkpoint, one chromosome"
replicates: 7 tracks x (2 read-length + 1 run-type arms); 2 seeds of checkpoint where they exist
verdict: 
metric: 
created: "2026-10-25T13:49:00"
updated: "2026-10-25T13:49:00"
---

# h26 — A trained CANDI checkpoint, told a counterfactual arm's read length or run type in place of the base's, moves its predicted counts no closer to the arm's real track than when told the base's own covariates

Parent:: [[q8_does_candi_s_training_which_never_pairs_]]

## ELI5

If we tell a trained CANDI model that a track was read with shorter reads or as paired reads, we expect its prediction not to change the way the real re-processed track changes.

## TL;DR

Claim: CANDI's output does not follow a change in read length or run type with biology fixed, because its training has no pair that shows such a change. The probe keeps the input fixed (the T_ assays of the cell type), swaps only the output covariates from the base's to an arm's, and compares the predicted change with the real base-to-arm change of [[t112_build_the_counterfactual_arms_for_t|the counterfactual arms]]. Supported if the predicted log2 change stays near 0, or does not track the real one, and the arm's NLL does not improve. Refuted if the predicted change tracks the real one beyond a permuted-covariate null. Depth arms are the positive control: the offset head must reproduce them.

## Null

<!-- spec 09: the BORING explanation — the cheapest way this result could be trivially
     true. One line, <=25 words, naming a family. Your checks must discriminate against it.
     The PI approves it before checks are written: `crux approve-null <id>`. -->
_(one line: the cheapest way this result could be trivially true — name a family from capacity, chance, leakage, selection, normalization, instrumentation)_

## Problem Statement

CANDI's training never shows one experiment next to a counterfactual of itself in read length or run type, so nothing in its objective identifies how these covariates change a track. The counterfactual arms give real tracks that differ from a base in exactly one knob, so the correction CANDI would have to make is known.

## Idea / Hypothesis

A trained CANDI checkpoint, told a counterfactual arm's read length or run type in place of the base's, moves its predicted counts no closer to the arm's real track than when told the base's own covariates

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
- [ ] _(state a falsifiable, pre-registered check)_

## Planned Intervention

CPU only, no training. Checkpoint: the selected CANDI checkpoint of [[t81_retrain_every_trainable_method_under|the retrain-every-method task]] (`/project/def-maxwl/mforooz/t81_checkpoints/t81_eic_19_s0.best.ckpt` and `_s1`; check the md5 against `results/t81/TRAIN_CANDI_EIC19.md` first, because the `/project` copy was once stale). Per track (C12M02, C19M16, C40M17, C40M18, C07M20, C19M22, C07M29): prompt with the cell type's T_ assays; decode the target column with y_meta set to (a) the base's covariates, (b) each arm's read length 36 / 50, (c) run type PE, (d) the depth arms 15M / 7.5M / 3.75M. Read per bin, on chr21: Δ = log2 mean(told arm) − log2 mean(told base) against log2(arm / base counts, +1); NB NLL of the arm track under (a) and under (b–d). Null: the same Δ with the arm's covariate values permuted across tracks. Optional second reading: `python -m candi.bench` with tier C (`covuse`, `covshare`) on the same checkpoint. The causal follow-up, which needs GPU: retrain CANDI with base→arm pairs added as input→target and repeat the probe.

## Run Links

_(none yet)_

## Artifacts

<!-- what the run produced. Keep files under results/h26/ and link at least the report:
     - [Report](results/h26/report.md)   - results/h26/curve.png -->
_(none yet)_

## Findings

Prior evidence, before any run; no checks are ticked. Pre-CANDII numbers are single seed, 12 held-out targets.

- Code: an unmasked assay's target is the same experiment, so read length and run type are equal on both sides; only depth varies, by independent binomial thinning (`src/candi/dataset.py:43-65, 430-446`; `src/candi/store/dataset.py:142`). A masked assay's covariates are fixed per (biosample, assay), and a batch holds one biosample.
- With the depth offset on, assay and run_type steering were functionally null (`AGENTS.md` §7.1, §7.2 item 3); at weight decay 0 the run_type direction score was 0.559, near chance (archive h47).
- With the offset off, run_type changed imputation in the right direction (archive h42), but run type, read length and depth are collinear on that panel.
- On the smoke-scale `t22_on_s0` checkpoint, `covuse` reports run_type unused ([equivalence report](results/t22/EQUIVALENCE.md)).
- No production CANDII checkpoint has a covariate block score: the t81 score jsons carry no `C` key (Fir, 2026-10-25).
