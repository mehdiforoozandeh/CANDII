---
id: f75
type: finding
schema: 3
title: "Weak on purpose: the direction is clear, but AlphaGenome 2 trained on these…"
parent: q20
confidence: weak
why: "Weak on purpose: the direction is clear, but AlphaGenome 2 trained on these tracks, so this result says nothing about imputation on unseen tracks and must not set the answer of this question."
evidence: results/t308/AG2_VS_BASELINES.md, results/t308/ag2_contrasts_val.tsv
created: "2026-11-14T19:19:13"
updated: "2026-11-14T19:19:13"
---

# f75 — Weak on purpose: the direction is clear, but AlphaGenome 2 trained on these…

Parent:: [[q20_which_eic_tracks_did_alphagenome_2_train]]

## Answer

Trained-on set, not imputation: on the 30 validation tracks in 20 cells that AlphaGenome 2 trained on (audit status observed_matched), its recalibrated r_spec (rspec_cal) was 0.614, against a pseudoreplicate-half noise floor of 0.585 [0.519, 0.639]; position average 0.122, marginal NB 0.069, closest-five oracle 0.154 (ag2_per_target_val.tsv). Position average minus AlphaGenome 2 = -0.492 [-0.578, -0.381] (cluster bootstrap over cells), ahead on 1 of 30 tracks (ag2_contrasts_val.tsv). ENCODE-peak AUPRC 0.809 against 0.533 for the position average. This measures AlphaGenome 2's fit to its own training tracks, so it is not evidence about imputation and does not answer the fair-track half of this question.
