---
id: t390
type: task
title: find and test a fair count-to-p-value route that puts miniCANDI on the EIC entrant board (exploratory, never merged)
category: implementation
parent: 
blocked_by: None
refs: q90
hypothesis_refs: 
status: open
created: 2026-11-22T13:58:48
updated: 2026-11-22T13:58:48
---

# t390 — find and test a fair count-to-p-value route that puts miniCANDI on the EIC entrant board (exploratory, never merged)

Refs:: [[q90_can_a_count_space_model_like_minicandi_b\|q90]]

## Why

answers whether miniCANDI can enter the EIC ranking fairly

## Plan

Every rung is an "entry" scored by the exact EIC scorer (`tools/onion_eval/eic_score.py`, nine measures, raw values) against the corrected truth (MOESM4, Zenodo 14736610) and the uncorrected single-end truth (Zenodo 14736608), and placed among the 25 published entrants. No model is involved: the ladder measures what each step from reads to the board costs.

Observed-data ladder, blind experiments (Nibi `/project/def-maxwl/mforooz/EIC_REPRO/003_pipeline/results/<exp>/`):
- L1 Jacob's single-end track, raw (before normalization): what the normalization step alone moves.
- L2 L1 normalized by the exact rule (Max sprint 04 `apply_bp.py`): sanity, must equal the truth.
- L3 our single-end pipeline p-value bigwig, raw: what reprocessing differences cost.
- L4 L3 normalized by the same rule with our own peaks.
- L5 MACS2 formula from 1 bp read starts of our BAMs and control: must equal L3 (formula exactness on the blind set; ledger R4).
- L6 formula from 25 bp overlap counts of the same reads (`signal_from_counts.py`, box spread, plug-in): the 25 bp conversion loss.
- L7 L6 normalized (peaks called on L6): the full count route on observed counts, the ceiling for miniCANDI's route.

Then: a fair map from the count route to the corrected scale using training and validation data only (D2); the cost of each output representation (overlap counts, fragment pileup, 1 bp starts; ledger R6); miniCANDI's own rows once blind predictions exist (gated: blind-row build t184 is the PI's call).

Probe first: L1-L7 on 3 tracks (one sharp mark, one broad mark, DNase), then the batch.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
