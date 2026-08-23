---
id: t35
type: task
title: "amend AGENTS.md 7.2's noise floor: it is 2-4x too large for this recipe"
category: admin
parent: 
blocked_by: None
refs: 
hypothesis_refs: 
status: done
created: "2026-08-21T15:03:12"
updated: "2026-08-22T17:18:11"
---

# t35 — amend AGENTS.md 7.2's noise floor: it is 2-4x too large for this recipe

Refs:: _(none)_

## Why

PI ruling 2026-08-21: amend in place with a dated note; measured 0.0327 (eval.py) and 0.0608 (bench) vs the frozen 0.1195

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- [The run behind the amendment](results/t22/SEED_FLOOR.md) — AGENTS.md §7.2 amended in place, dated 2026-08-22 (commit e5687cc): q19 seed floor 0.0463 pooled / 0.0327 macro (eval), 0.0608 macro (bench), jobs 55883172/55883173; ~0.09 clustered floor and ±0.13 left standing — no recorded replacement

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
