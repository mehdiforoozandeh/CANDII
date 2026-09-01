---
id: t28
type: task
title: "teach eval.py::build_eval_units to take a StoreDataset, so a store-backed run can actually be scored"
category: implementation
parent: 
blocked_by: t14
refs: 
hypothesis_refs: 
status: done
created: "2026-08-21T01:03:39"
updated: "2026-08-21T15:35:51"
---

# t28 — teach eval.py::build_eval_units to take a StoreDataset, so a store-backed run can actually be scored

Refs:: _(none)_

## Why

t14 made StoreDataset emit the five imputation keys off a declared eval_pairs (D31), but eval.py still constructs its own CandiKitH5Dataset, so train.py --store still skips evaluate() and writes a run json with no M1/M2/M3/S14. StoreDataset already exposes the three names that factory reads (_eval_indices, _bios_candidates, _all_imp_biosamples), so the change is a dataset parameter threaded through evaluate -> _evaluate_fp32 -> build_eval_units. Kept separate because eval.py is the SCORED path: every recorded number in AGENTS.md section 7 came out of it, and it deserves its own before/after rather than riding along with a loader change. Note log_ref stays absent either way - it is the h74 reference table, which pins itself to an h5 fingerprint.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- [Merged ce212d0 (PR #16, squash)](results/t28/DELIVERABLE.md) — build_eval_units and quick_eval take a ready dataset, store guard narrowed to eval_pairs_declared(); 780 pass, golden 0 ULP. The old record named `09a0f3e`, which the squash left off `main`.

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
