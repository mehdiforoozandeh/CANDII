---
id: t363
type: task
title: Onion: PI decision: which architecture goes to scaling (conv autoencoder, conv + attention, U-Net)
category: admin
parent: t152
blocked_by: None
refs: q48, q39
hypothesis_refs: 
status: open
created: 2026-11-18T16:21:32
updated: "2026-11-18T23:23:12"
---

# t363 — Onion: PI decision: which architecture goes to scaling (conv autoencoder, conv + attention, U-Net)

Refs:: [[q48_which_minicandi_design_choices_move_impu\|q48]], [[q39_do_the_context_rungs_2_to_6_convolution_\|q39]]

## Why

settled miniCANDI design, decisions for the next session (row 1)

## Current scope (close-out reconciliation, 2026-11-18)

Source: `closeout/settled miniCANDI design.md`, section "Decisions that the PI makes in the next session".
- Recommendation (agent's intent): carry the conv autoencoder and the U-Net into the first scaling step, and choose after the mix-and-match round (t220).
- Information: design row 1 (the three models tie on the composite on 256 regions; the U-Net is better on imputation, both conv models on correction); rows 3, 22, 24, 25.
- Would change it: the 256-region rescoring (t350) or the mix-and-match round moves one model beyond twice the seed range on the composite with no goal lower.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
