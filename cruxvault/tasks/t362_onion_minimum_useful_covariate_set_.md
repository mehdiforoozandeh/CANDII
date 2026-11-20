---
id: t362
type: task
title: Onion: minimum useful covariate set on the model that mix-and-match selects: leave-one-field-out, groups, removal curve
category: implementation
parent: t152
blocked_by: t220, t357
refs: q48
hypothesis_refs: 
status: open
created: 2026-11-18T16:21:32
updated: "2026-11-18T23:23:12"
---

# t362 — Onion: minimum useful covariate set on the model that mix-and-match selects: leave-one-field-out, groups, removal curve

Refs:: [[q48_which_minicandi_design_choices_move_impu\|q48]]

## Why

ready queue entry 14; PI decision 9 of 2026-11-17; a deliverable named by the PI

## Current scope (close-out reconciliation, 2026-11-18)

Ready queue entry 14 (`closeout/ready queue for scaling.md`); item file `closeout/closeout A12 minimum useful covariate set.md`. PI decision 9 of 2026-11-17: one model, the one that mix-and-match selects; leave-one-field-out, groups and the removal curve.
- Command (Fir): `CLUSTER=fir K=<kit> STEP=<ref|lofo|groups|curve> bash <kit>/tools/minicandi/configs/ablations/covariate_fields/submit_a12.sh` after the configs are regenerated for the mix-and-match recipe (`tools/minicandi/covariate_field_arms.py`).
- Cost: 3,008 (conv autoencoder), 3,298 (conv + attention) or 5,060 (U-Net) CPU-hours.
- Prerequisite: the recipe from t220; t357.
- Decision rule: item A12 file, section 7.4.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
