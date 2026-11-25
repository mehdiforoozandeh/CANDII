---
id: f145
type: finding
schema: 3
title: The imputation gap and the deviance calibration gap exceed every seed range…
parent: q75
confidence: moderate
why: The imputation gap and the deviance calibration gap exceed every seed range (all read-start seeds below every reference seed), but the correction and headline calibration gaps fall within seed noise, part of the overconfidence is an expected moment-match conversion bias of unmeasured size, and 16 regions are a screen, not a confirmation (D3).
evidence: results/t392/SIDE_BY_SIDE.md
created: "2026-11-24T21:07:24"
updated: "2026-11-24T21:07:24"
---

# f145 — The imputation gap and the deviance calibration gap exceed every seed range…

Parent:: [[q75_can_one_minicandi_recipe_match_or_beat_t]]

## Answer

Training the full-combination recipe on 25 bp read-start counts (2 strands for ChIP, 1 channel for DNase and ATAC), then converting its predictions to overlap counts for scoring, does not improve on the same recipe trained on overlap counts: on 16 regions over 3 seeds it is lower on imputation (0.457 vs 0.500), correction (0.783 vs 0.816) and calibration (0.606 vs 0.685), and on the deviance imputation skill (0.375 vs 0.543), and its deviance calibration ratio is 1.31 vs 1.06 (overconfident). So read-start inputs are not a route to Pareto dominance at this scale.
