---
id: t20
type: task
title: whole-chromosome bench harness and CLI, both pval and count arms
category: implementation
parent: t17
blocked_by: t19
refs: 
hypothesis_refs: 
status: done
created: "2026-08-19T23:52:14"
updated: "2026-08-20T01:25:48"
---

# t20 — whole-chromosome bench harness and CLI, both pval and count arms

Refs:: _(none)_

## Why

turns verified primitives into a scored checkpoint; full-track scope is what makes mse1obs and the partitions mean what they mean

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- [The whole-chromosome harness and the one CLI — commit 5d1e13d](results/t20/DELIVERABLE.md) — harness.py + cli.py close t20. The harness plans its OWN windows: neither training loader covers a chromosome (CandiKitH5Dataset advances one (T_, V_/B_) pair per window batch, so a track sees 1 window in n_pairs; Regime.windows drops mask-rejected tiles), so the datasets stay the authority on metadata and the harness owns which windows, in which order, for which pair — a full tile of every eval chromosome, every 25 bp bin scored exactly once, tail tile pulled back to end flush. Pair-outer/window-inner keeps one pair's targets in memory rather than the panel's; the only survivor is the bit-packed binarised peak call that pr_by_specificity needs across every cell type. Two backends: a baked h5 (T_/V_/B_ pairing, refuses a gapped eval tiling with the hole measured) and a CANDI_STORE (leave-one-assay-out by mask, one assay per pass — D16 makes store names opaque so there is no counterpart cell). Both arms, five blocks, one CLI; --eval-budget and --max-batches are refused BY NAME with the reason attached. bench still does not import candi.train, now held by a subprocess test on a cold interpreter. Anti-drift is pinned twice: the store backend calls _make_batch directly, and the h5 backend, which must re-implement assembly, is compared tensor-by-tensor to the loader's own batch — bit-identical across all sixteen. End-to-end anchor: a record whose prediction IS its target scores gwcorr=gwspear=1, mse=0 through the whole stack. Gates: 688 pass, 1 skipped, golden 0 ULP. OWED: the rank aggregation repeats CANDI's whole-track score across all ten competitor bootstraps, which biases AGAINST it under the second-best-bootstrap rule; scoring CANDI on the challenge's own position bootstraps needs its own task.

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
