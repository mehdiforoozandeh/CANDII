---
id: t68
type: task
title: flatten the internal bench C-block (nested covariate dicts) into scalar registry keys so add can stamp a CANDI-lineage score
category: implementation
parent: t60
blocked_by: None
refs: 
hypothesis_refs: 
status: open
created: 2026-08-27T14:29:15
updated: 2026-08-27T14:29:15
---

# t68 — flatten the internal bench C-block (nested covariate dicts) into scalar registry keys so add can stamp a CANDI-lineage score

Refs:: _(none)_

## Why

add requires scalar metrics; c_block emits nested dicts (covuse per covariate, depthblind_biokeep combined) — a real internal-bench stamp is refused until the extract path flattens them

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
