---
id: t355
type: task
title: Onion: the PI cancels the Fir rescoring copies (jobs 4535347 to 4535923) after the Nibi scores are on disk
category: admin
parent: t152
blocked_by: t350
refs: q56
hypothesis_refs: 
status: open
created: 2026-11-18T09:48:04
updated: 2026-11-18T09:48:04
---

# t355 — Onion: the PI cancels the Fir rescoring copies (jobs 4535347 to 4535923) after the Nibi scores are on disk

Refs:: [[q56_do_the_ablation_conclusions_drawn_from_t\|q56]]

## Why

the queued Fir jobs duplicate the Nibi rescoring and would spend about 2,225 CPU-h again when Fir returns

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_

## PI ruling of 2026-11-19 01:40 UTC

The PI said: "you can cancel them once we have the scores on nibi". The agent cancels Fir jobs 4535347 to 4535923 after all 62 run-seeds are scored on Nibi, and not before. The ruling covers these job ids only.
