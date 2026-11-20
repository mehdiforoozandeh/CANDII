---
id: t256
type: task
title: Onion: RNA-seq input arms (track and expression vector) on the three models
category: implementation
parent: t152
blocked_by: None
refs: q48
hypothesis_refs: 
status: open
created: 2026-11-11T23:05:12
updated: "2026-11-18T23:23:12"
---

# t256 — Onion: RNA-seq input arms (track and expression vector) on the three models

Refs:: _(none)_

## Why

settles the stage-3 arms on the design and writes the RNA-seq side table (chunks S3a, S3b; after gate G3b, and after the extra-input chunk for a module-file design)

## Current scope (close-out reconciliation, 2026-11-18)

Ready queue entry 18 (`closeout/ready queue for scaling.md`); item file `closeout/closeout A5 RNA-seq input.md`. The other stage-3 arms ran elsewhere: loss arms in close-out item A1, the cell embedding is t360.
- Not built: no `rna_track` or `rna_vector` configs, no RNA-cell eval subset, no submit script. Tools exist: `tools/onion_data/rna_inputs.py`, `slurm/onion_data/rna_inputs.sh`, `tools/minicandi/stage3_arms.py`. RNA tracks are on Nibi; the PI allowed a Globus move to Fir.
- Cost when built: 2 arms x 3 models x 3 seeds = 18 runs; at most about 1,131 CPU-hours (training 164.4, scoring at most 644.4 + 322.2).
- Rule (PI ruling 27): RNA-seq input is an ablation only, never in the final EIC model.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
