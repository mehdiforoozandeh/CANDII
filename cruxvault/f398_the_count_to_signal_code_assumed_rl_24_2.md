---
id: f398
type: finding
schema: 3
title: The count-to-signal code assumed (rl+24)/25 bins per read; the store rule…
parent: q90
confidence: moderate
evidence: results/t390/span_fix/SPAN_FIX.md
created: "2026-11-24T04:01:51"
updated: "2026-11-24T04:01:51"
---

# f398 — The count-to-signal code assumed (rl+24)/25 bins per read; the store rule…

Parent:: [[q90_can_a_count_space_model_like_minicandi_b]]

## Answer

The count-to-signal code assumed (rl+24)/25 bins per read; the store rule gives (rl+25)/25. Fixing it moved the count-route nMSE vs the exact MACS2 formula by +2.0% / -0.7% / -0.4% (C19M22 0.0172->0.0175, C40M17 0.255->0.253, C12M02 0.0212->0.0212) and changed no combined rank on either board. The count-route gap is not a scale error.
