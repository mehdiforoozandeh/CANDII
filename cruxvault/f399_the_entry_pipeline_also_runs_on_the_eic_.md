---
id: f399
type: finding
schema: 3
title: The entry pipeline also runs on the EIC validation board (42 of 45…
parent: q90
confidence: moderate
evidence: results/t390/validation_dryrun/VALIDATION_DRYRUN.md
created: "2026-11-24T04:02:13"
updated: "2026-11-24T04:02:13"
---

# f399 — The entry pipeline also runs on the EIC validation board (42 of 45…

Parent:: [[q90_can_a_count_space_model_like_minicandi_b]]

## Answer

The entry pipeline also runs on the EIC validation board (42 of 45 experiments; 3 never built into the store) on Fir and Nibi, placed against the published round-1 validation scores (MOESM6). The scorer matches MOESM6 on 8 of 9 measures to 1.3e-6; msevar does not reproduce, so it is reported but not ranked. On observed counts, the count route vs the store MACS2 signal had nMSE 0.367 / 0.136 / 0.0118 (C17M04, C12M32, C03M02), and those oracle tracks rank 2 / 1 / 1 of 11.
