---
id: f130
type: finding
schema: 3
title: The number is from the run log of 2026-11-17; the comparison files are on…
parent: q17
confidence: moderate
why: The number is from the run log of 2026-11-17; the comparison files are on Fir, which is in an outage, so the value was not re-read.
evidence: results/closeout_records/EIC_V1_2_BASELINE_EQUALITY.md
created: "2026-11-18T16:26:09"
updated: "2026-11-18T16:26:09"
---

# f130 — The number is from the run log of 2026-11-17; the comparison files are on…

Parent:: [[q17_what_is_the_dataset_and_what_did_it_cost]]

## Answer

The baselines scored on the table version with filled covariates (eic_v1_2) equal the eic_v1 baselines leaf by leaf (largest relative difference 3.3e-13 on values above 1e-3, no key added or lost): the new version changes only the covariates.
