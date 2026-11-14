---
id: t184
type: task
title: Onion: EIC build, priority 4 (submit)
category: data-acquisition
parent: t152
blocked_by: t180
refs: 
hypothesis_refs: 
status: open
created: 2026-11-10T22:47:27
updated: 2026-11-10T22:47:27
---

# t184 — Onion: EIC build, priority 4 (submit)

Refs:: _(none)_

## Why

submits per-library and per-lane views and the blind experiments for eic_v2

## Decisions and runs

- Held by the foreman on 2026-11-13 (scheduling): group fair-share was low (CPU 0.14, GPU 0.23) and priority 4 feeds only blind-side inputs, which the ladder does not need; it waits until rung 1 has trained and scored. Source: run Log, INSTRUCTIONS/peeling_the_onion/run_state/plan.md.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
