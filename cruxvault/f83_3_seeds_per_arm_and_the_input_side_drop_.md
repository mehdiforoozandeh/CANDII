---
id: f83
type: finding
schema: 3
title: 3 seeds per arm and the input-side drop exceeds both seed ranges; 16 regions…
parent: q36
confidence: moderate
why: 3 seeds per arm and the input-side drop exceeds both seed ranges; 16 regions only, and the output-side runs were scored on Fir from Nibi checkpoints
evidence: results/views_ablation/VIEWS_ABLATION.md
created: "2026-11-15T17:12:22"
updated: "2026-11-15T17:12:22"
---

# f83 — 3 seeds per arm and the input-side drop exceeds both seed ranges; 16 regions…

Parent:: [[q36_does_hiding_depth_from_the_encoders_a_ha]]

## Answer

The loss from hiding depth is on the input side. Depth hidden on the output side only (3 seeds): composite 0.455 (range 0.013) against base 0.444 (0.030), gap closed 0.335 (0.011) against 0.298 (0.052). Depth hidden on the input side: composite 0.382 (0.009), specific variance 0.401 against 0.440. Both sides hidden with the input transform: composite 0.260 (0.019), uncertainty term -0.099.
