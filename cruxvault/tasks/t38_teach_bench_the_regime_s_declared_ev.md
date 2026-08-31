---
id: t38
type: task
title: "teach bench the regime's declared eval_pairs: StoreSource imputes cross-cell as training does"
category: implementation
parent: 
blocked_by: None
refs: 
hypothesis_refs: 
status: done
created: "2026-08-22T17:18:18"
updated: "2026-08-22T17:18:35"
---

# t38 — teach bench the regime's declared eval_pairs: StoreSource imputes cross-cell as training does

Refs:: _(none)_

## Why

bench self-paired every cell and scored leave-one-out, a different task than training's declared T->V imputation; the monitor is meaningless without matching tasks

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- [Deliverable](results/t38/DELIVERABLE.md) — StoreSource reads regime.eval_pairs; cross_cell predicate; filtered target rule; one decode group per declared pair; 2f56cb1

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
