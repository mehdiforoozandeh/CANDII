---
id: f29
type: finding
schema: 3
title: Read from the data documents and the counterfactual-arm build record, not…
parent: q8
confidence: moderate
why: Read from the data documents and the counterfactual-arm build record, not measured by a run.
evidence: tasks/t112_build_the_counterfactual_arms_for_t.md
created: "2026-10-31T18:19:29"
updated: "2026-10-31T18:19:29"
---

# f29 — Read from the data documents and the counterfactual-arm build record, not…

Parent:: [[q8_does_candi_s_training_which_never_pairs_]]

## Answer

Background, not a test of the claim: the CANDI corpora hold processed signal only (363 EIC experiments, 3026 merged tracks, all built from released ENCODE bigWigs; no FASTQ or BAM recorded in DATA.md or STORE.md), so count thinning can make only depth pairs; every other covariate needs raw reads fetched on the cluster (BAM route ~1 h per product, FASTQ route median 13.4 h, per the counterfactual-arm build). On the 8-assay panel, assay plus read length fully determine run type (0.000 bits), so run type cannot be learned from that panel.
