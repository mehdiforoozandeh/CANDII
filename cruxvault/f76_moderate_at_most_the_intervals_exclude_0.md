---
id: f76
type: finding
schema: 3
title: "Moderate at most: the intervals exclude 0 and AlphaGenome 2 leads on 11 of…"
parent: q20
confidence: moderate
why: "Moderate at most: the intervals exclude 0 and AlphaGenome 2 leads on 11 of 11 tracks, but the set is 11 tracks in 9 cells and 9 of its 10 marks have one track, so it supports a direction against the training-free baselines, not a per-mark ranking."
evidence: results/t308/AG2_VS_BASELINES.md, results/t308/ag2_contrasts_val.tsv, results/t308/ag2_board_val.tsv
created: "2026-11-14T19:19:13"
updated: "2026-11-14T19:19:13"
---

# f76 — Moderate at most: the intervals exclude 0 and AlphaGenome 2 leads on 11 of…

Parent:: [[q20_which_eic_tracks_did_alphagenome_2_train]]

## Answer

Second half answered on the fair set: on the 11 validation tracks in 9 cells that neither AlphaGenome 2 nor our training set used (eic_v1, leave-one-assay-out headline regime), AlphaGenome 2 (matched read-type channel) had recalibrated r_spec (rspec_cal) 0.438, against position average 0.222, marginal NB 0.121, closest-five oracle 0.066, training-track mean signal 0 by definition, and a pseudoreplicate-half noise floor of 0.333 [0.276, 0.382] (ag2_per_target_val.tsv, ag2_board_val.tsv). Position average minus AlphaGenome 2 = -0.216 [-0.297, -0.123] (95% cluster bootstrap over cells, 2,000 resamples), position average ahead on 0 of 11 tracks; every baseline is behind with an interval excluding 0 on rspec_cal, pearson_cal and ENCODE-peak AUPRC (AlphaGenome 2 0.454 against position average 0.271). Spearman against the position average: -0.031 [-0.106, +0.064], no call. Not run: the control-free table, the retrieval baseline; no miniCANDI row yet; 33 training tracks are left out of the two training-track baselines for lack of a fragment length. Validation tracks only; no selection reads these numbers.
