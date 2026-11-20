---
id: t220
type: task
title: Onion: mix-and-match of the single settings that helped: 256-region gate, then the round
category: implementation
parent: t152
blocked_by: t350, t357
refs: q48
hypothesis_refs: 
status: open
created: 2026-11-11T11:20:31
updated: "2026-11-18T23:23:12"
---

# t220 — Onion: mix-and-match of the single settings that helped: 256-region gate, then the round

Refs:: _(none)_

## Why

names the winning recipes for the evolution step (chunks MXp, MXh, MXa, MXb; after gate G3)

## Current scope (close-out reconciliation, 2026-11-18)

Ready queue entry 10 (`closeout/ready queue for scaling.md`); item file `closeout/closeout A13 mix-and-match combinations.md`.
- PI decision 8 of 2026-11-17: gate first, then the round for the switches that pass.
- Commands: gate on Fir, `CLUSTER=fir K=<fresh Fir kit> PRED_STORE_ROOT=/scratch/mforooz/onion/pred_store STAGE=gate bash <kit>/tools/minicandi/configs/ablations/mix/submit_mix.sh` (dry run, then `DRY_RUN=0`); then `select_rule.py arm-summary --sub p256_sv` and `mix_arms.py qualify --summary <tsv>` (exit 0 PASS, 1 FAIL, 3 WAIT); then the round with `STAGE=runs QUALIFY_SUMMARY=<tsv>`.
- Cost: gate 474.2 CPU-hours (135 jobs); round 5,389.9 on Nibi or 5,307.7 on Fir (1,134 jobs + 5 reference jobs). Recount if the gate removes a switch.
- Prerequisite: the 256-region rescoring (t350) and the eic_v1_2 references (t357). On Nibi, U-Net arms need `SCORE_PRED_TIME=12:00:00`.
- Decision rule: item A13 file, section 4.6. Fixed input depth is reported with the PI's deployment caveat.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
