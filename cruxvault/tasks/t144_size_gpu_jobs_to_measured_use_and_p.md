---
id: t144
type: task
title: size GPU jobs to measured use and prefer the cluster with fair-share headroom
category: hpc-setup
parent: 
blocked_by: None
refs: 
hypothesis_refs: 
status: done
created: "2026-10-30T00:28:30"
updated: "2026-10-30T00:28:44"
---

# t144 — size GPU jobs to measured use and prefer the cluster with fair-share headroom

Refs:: _(none)_

## Why

2026-10-29 Fir queue diagnosis: def-maxwl_gpu fair-share exhausted, MAX_TRES billing on 4 CPUs, 12 h requested against an 86 min median run

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- [GPU job sizing rule](notes/gpu-job-sizing.md)

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
