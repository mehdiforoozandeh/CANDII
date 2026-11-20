---
id: f46
type: finding
schema: 3
title: "Literature and code, before any selection experiment: every large sequence…"
parent: q14
confidence: weak
evidence: results/region_budget/REGION_RESEARCH.md
created: "2026-11-14T02:19:05"
updated: "2026-11-14T02:19:05"
---

# f46 — Literature and code, before any selection experiment: every large sequence…

Parent:: [[q14_how_should_training_regions_be_selected_]]

## Answer

Literature and code, before any selection experiment: every large sequence model checked (Basenji2, Enformer, Borzoi, AlphaGenome) and every imputer checked (ChromImpute, Avocado, eDICE, Sei) trains on the whole genome or on uniform random positions; peak-centred draws appear only in base-resolution TF and accessibility models (BPNet, ChromBPNet, DeepSEA). No published curve of accuracy against the number of training regions was found (not found, which is not shown absent). Our trainer draws a uniform locus and a uniform cell with no strata, and nothing keeps training windows out of the evaluation regions (they land there at the genome rate, about 3.4%; the evaluation set tests unseen tracks at seen loci). The research recommends the uniform draw as the default until a weighted, leak-free arm beats it by more than 2 seed spreads with no stratum veto; the selection experiment is pending.
