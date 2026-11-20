---
id: t250
type: task
title: "Onion: Variance lost below 25 bp (run and report)"
category: implementation
parent: t152
blocked_by: t249, t242
refs: 
hypothesis_refs: 
status: done
created: "2026-11-11T23:05:11"
updated: "2026-11-18T09:48:19"
---

# t250 — Onion: Variance lost below 25 bp (run and report)

Refs:: _(none)_

## Why

answers how much variance is lost because 25 bp is too coarse

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- presentations/onion-closeout-2026-11-18-report/report.pdf

## Evidence

Nibi job 27693041 (kit CO9, 6 h 8 min) wrote `CO-A7/scale_variance.tsv` (2,208 rows). The evidence note `results/closeout_A7/variance_2026-11-18/VARIANCE_READ.md` is git-ignored and exists in the foreman worktree only; the tracked copy of the numbers is section 3.12 of the close-out report and deck slide 6. Scope: 43 pooled validation tracks, the 256 evaluation regions, 1 bp point-rule counts, each track at its own depth.

- Raw share of the 1 bp count variance that lies below 25 bp (law of total variance over blocks of 25 bp): broad marks 0.927 (19 tracks, assays 0.879 to 0.951), narrow marks 0.901 (22 tracks, 0.845 to 0.948). Pure read-sampling noise gives 1 - 1/25 = 0.960.
- Share below 25 bp that two biological replicates reproduce (cross-covariance, where read-sampling noise cancels): broad 0.0011 (15 pairs), narrow 0.0009 (14 pairs), i.e. about 0.1% of the 1 bp variance.
- Read halves (pseudoreplicates): all 138 counts-scale rows are below zero (-0.0058 to -0.0441). Every track has duplicate removal, so one strand of one base holds at most one read and the two halves exclude each other at 1 bp. On the control track the uniform-density prediction -m (b - 1) / b gives -0.0092, -0.0111, -0.0115 against the measured -0.0092, -0.0105, -0.0097 below 5, 25 and 125 bp. The read-half rows are not used.
- Limits: DNase-seq has one track and no replicate pair; 11 of the 17 assays with a replicate row rest on one pair; the 1 bp store is not verified end to end (verify job 27692994 was still running).
