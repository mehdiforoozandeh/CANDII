---
id: f43
type: finding
schema: 3
title: "Variance estimators: a simulation audit of the layer estimators found 10…"
parent: q27
confidence: moderate
evidence: results/t305/LAYERS_REAL.md, results/t233/AUDIT.md, results/t234/FIXES.md
created: "2026-11-14T02:18:33"
updated: "2026-11-14T02:18:33"
---

# f43 — Variance estimators: a simulation audit of the layer estimators found 10…

Parent:: [[q27_are_the_onion_run_s_estimators_scorers_a]]

## Answer

Variance estimators: a simulation audit of the layer estimators found 10 defect rows (for example, the pseudoreplicate rung read the same file twice, layers.tsv stored twice the pair value, donor and experiment were crossed), all fixed with tests. On real data (eic_v1) the telescoping identity holds: of 2,775 assay x stratum x band groups, 390 hold exactly (largest difference 2.2e-19), 2,015 differ only because their rungs hold different cells (flagged), 370 are empty (mappability deciles d7 and d9), and 0 are unexplained.
