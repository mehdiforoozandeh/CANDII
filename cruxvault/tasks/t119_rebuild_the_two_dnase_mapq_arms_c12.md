---
id: t119
type: task
title: rebuild the two DNase MAPQ arms (C12M02 mapq 0 and 10) with multimapping off, so the MAPQ cut applies; the t112 builds are byte-identical to the DNase base
category: data-acquisition
parent: 
blocked_by: None
refs: h15
hypothesis_refs: 
status: open
created: 2026-09-23T16:39:27
updated: 2026-09-23T16:39:27
---

# t119 — rebuild the two DNase MAPQ arms (C12M02 mapq 0 and 10) with multimapping off, so the MAPQ cut applies; the t112 builds are byte-identical to the DNase base

Refs:: [[h15_one_transformation_conditioned_on_the_so\|h15]]

## Why

the DNase MAPQ arms currently do not test the MAPQ knob

## Progress

- 2026-10-24: no rebuild is recorded in the vault, in `plan/T118_STATUS.md` or in any commit. The pseudoreplicate checks confirm the defect: the C12M02 MAPQ 0 and 10 arms have treatment reads byte-identical to the DNase base (`results/t117/CHECKS.md`). Every grid table leaves both arms out.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
