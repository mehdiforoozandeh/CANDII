---
id: f121
type: finding
schema: 3
title: Read from the build report and the check logs of the version; no model has…
parent: q17
confidence: moderate
why: Read from the build report and the check logs of the version; no model has been compared across eic_v1 and eic_v1_2 in this record.
evidence: results/closeout_D1/CHANGES_eic_v1_2.md, results/closeout_D1/D1_RESULT.md
created: "2026-11-18T16:22:48"
updated: "2026-11-18T16:22:48"
---

# f121 — Read from the build report and the check logs of the version; no model has…

Parent:: [[q17_what_is_the_dataset_and_what_did_it_cost]]

## Answer

Table version eic_v1_2 (2026-11-17; eic_v1 untouched) fills lab, antibody, donor and platform on the 2,740 rows: lab, donor and platform 100% in every split; antibody 98.6% of training ChIP-seq target rows (1,739 of 1,763), 98.1% of validation (314 of 320), 100% of blind (72); 244 antibody cells and 24 donor cells are 'mixed' across replicates, 30 rows have no antibody in the catalog. Every vocabulary is built from training rows only, so an unseen value encodes to id 0 (unknown): validation rows with an unseen value are 18 (antibody, 6 accessions), 6 (donor) and 9 (platform); blind rows 48 of 72 (antibody, 4 accessions, mostly new lots). The earlier eic_v1_1 built its vocabularies from all splits, so unseen values got untrained embedding rows; it must not be used. A checkpoint is tied to its tables version.
