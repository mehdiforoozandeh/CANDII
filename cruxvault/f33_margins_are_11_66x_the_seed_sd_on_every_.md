---
id: f33
type: finding
schema: 3
title: Margins are 11-66x the seed SD on every pair kind, but the gap is…
parent: q1
confidence: moderate
why: Margins are 11-66x the seed SD on every pair kind, but the gap is concentrated in a subset of pairs (depth-only wins 112/228 trained pairs), part of it is denoising shallow sources rather than covariate information, and this is one simulated testbed, not real ENCODE tracks.
evidence: results/t150/REPORT.md
created: "2026-11-08T12:31:12"
updated: "2026-11-08T12:31:12"
---

# f33 — Margins are 11-66x the seed SD on every pair kind, but the gap is…

Parent:: [[q1_do_the_recorded_experimental_covariates_]]

## Answer

On the final CAC testbed, a correction that reads the recorded covariates beats plain depth scaling (counts x d'/d with a fitted NB size): mean CRPS 0.432 vs 0.550 on trained pairs, 0.419 vs 0.814 on held-out depth pairs, 0.527 vs 0.677 on never-trained pairs (seed SD 0.006-0.011). The same network without covariates only ties depth scaling (0.547 trained), so the covariates carry the gain. Depth scaling fixes scale error but not shape.
