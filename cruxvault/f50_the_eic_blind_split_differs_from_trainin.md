---
id: f50
type: finding
schema: 3
title: The EIC blind split differs from training in its sequencing covariates, and…
parent: q17
confidence: moderate
evidence: results/t158/LEDGER_CHECK.md
created: "2026-11-14T02:19:21"
updated: "2026-11-14T02:19:21"
---

# f50 — The EIC blind split differs from training in its sequencing covariates, and…

Parent:: [[q17_what_is_the_dataset_and_what_did_it_cost]]

## Answer

The EIC blind split differs from training in its sequencing covariates, and validation does not: total variation distance train against blind 0.970 (platform; blind 51 of 51 HiSeq 4000), 0.922 (lab; 48 of 51 Snyder) and 0.903 (run type; blind all paired-end), still 0.947, 0.876 and 0.891 after matching the assay mix; train against validation 0.012 to 0.131. So validation scores cannot see the blind covariate shift.
