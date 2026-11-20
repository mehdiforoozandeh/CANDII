---
id: t357
type: task
title: Onion: eic_v1_2 references on 256 regions and the 256-region scores of its 9 base runs (conv autoencoder, conv + attention, U-Net)
category: implementation
parent: t152
blocked_by: None
refs: q56, q48
hypothesis_refs: 
status: open
created: 2026-11-18T16:21:21
updated: "2026-11-18T23:23:12"
---

# t357 — Onion: eic_v1_2 references on 256 regions and the 256-region scores of its 9 base runs (conv autoencoder, conv + attention, U-Net)

Refs:: [[q56_do_the_ablation_conclusions_drawn_from_t\|q56]], [[q48_which_minicandi_design_choices_move_impu\|q48]]

## Why

ready queue entry 6: every arm on eic_v1_2 scored on 256 regions needs these references and base scores

## Current scope (close-out reconciliation, 2026-11-18)

Ready queue entry 6 (`closeout/ready queue for scaling.md`).
- Commands (Fir): (1) `CLUSTER=fir K=<kit> STAGE=baselines bash <kit>/tools/minicandi/configs/ablations/mix/submit_mix.sh` (4 baselines + oracle); (2) after the 4 SCORED and `oracle_floor.json` exist, `CLUSTER=fir K=<kit> STEP=ref bash <kit>/tools/minicandi/configs/ablations/covariate_fields/submit_a12.sh`. Each first as a dry run.
- Cost: 95.1 + 322.2 = 417.3 CPU-hours.
- Prerequisite: Fir back from its outage; a fresh Fir kit. The base runs and the eic_v1_2 eval set are on Fir only.
- Decision rule: none. Blocks t220, t359, t360, t361, t362.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
