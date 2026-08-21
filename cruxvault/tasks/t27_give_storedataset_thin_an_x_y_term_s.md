---
id: t27
type: task
title: give StoreDataset._thin an x/y term so the deterministic RNG stops making x and y identical at equal DSF
category: implementation
parent: 
blocked_by: None
refs: 
hypothesis_refs: 
status: open
created: 2026-08-21T01:03:23
updated: 2026-08-21T01:03:23
---

# t27 — give StoreDataset._thin an x/y term so the deterministic RNG stops making x and y identical at equal DSF

Refs:: _(none)_

## Why

t16 measured it: D22's counter-based eval RNG seeds each draw from (run_seed, biosample, assay, chrom, window_start, dsf_milli(dsf)) with NO x/y term, so whenever x_dsf == y_dsf the two _thin calls build the same generator and return the same draw. Measured 19/19 equal-DSF columns bit-identical at every level, not just at DSF 1 - so the store reproduces the bake's full 1-in-4 identity-copy leak under the eval RNG, not the 1-in-16 the task assumed. Latent today only because eval.py::build_eval_units pins dsf_sampling=off. PI-GATED: adding the term would move every deterministic eval number, so it is a decision, not a cleanup. Evidence: cruxvault/results/t16/REPORT.md, tools/dsf_leak.py, tests/test_dsf_leak.py.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
