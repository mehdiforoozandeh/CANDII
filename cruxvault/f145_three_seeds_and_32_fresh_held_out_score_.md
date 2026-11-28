---
id: f145
type: finding
schema: 3
title: Three seeds and 32 fresh held-out score regions, but the deployment gain is…
parent: q78
confidence: moderate
why: Three seeds and 32 fresh held-out score regions, but the deployment gain is smaller than the seed range
evidence: results/t370/recal_study/recal_study.md
created: "2026-11-25T02:09:32"
updated: "2026-11-25T02:09:32"
---

# f145 — Three seeds and 32 fresh held-out score regions, but the deployment gain is…

Parent:: [[q78_how_much_does_per_assay_dispersion_recal]]

## Answer

Small for imputation: PIT deviation 0.0200 with no fit, 0.0185 with a fit on other cells (-7.5%, inside the seed range 0.0076), 0.0150 with a same-cell fit (-25%, an upper bound no user reaches for an imputed track). For correction a user can fit on the same cell: 0.0089 to 0.0066 (-26%); 0.0078 on other cells. Every seed improves under every fit. Factors are unstable across seeds and 6 assays reach the Poisson floor.
