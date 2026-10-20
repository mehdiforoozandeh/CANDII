---
id: t125
type: task
title: Build and run the depth-offset count model and the counts-to-p map h (exploratory)
category: implementation
parent: t118
blocked_by: None
refs: q5, q6
hypothesis_refs: 
status: open
created: 2026-10-08T16:58:24
updated: 2026-10-08T16:58:24
---

# t125 — Build and run the depth-offset count model and the counts-to-p map h (exploratory)

Refs:: [[q5_does_a_depth_offset_on_f_the_source_dept\|q5]], [[q6_how_closely_does_a_map_h_reproduce_the_l\|q6]]

## Why

Runs the depth-offset count model and the counts → p map h. The design is `plan/ETA_OFFSET_AND_H_DESIGN.md` on `exp/t118-counterfactual-f`. First step: verify the t112 DNase p recipe, the 25 bp p binning and the ChIP caller commands on Nibi.

## Progress

- 2026-10-11: the Nibi check of the p recipes and the 25 bp binning is done, and the PI settled the formula forks. Kit K9 shipped to Nibi; its verify job and the control-count cache ran.
- 2026-10-15: kit K10 (K9 plus a test-tolerance fix; the run code is identical) shipped. The Nibi smoke ran 11 count runs with their depth-law passes, 2 formula h scorings and 2 true-X′ h fits. The count runs, law passes and formula scorings had no CRPS > 20 record and no non-finite record.
- 2026-10-16: part 2's first half is done: 36 h fits on the true X′ and 38 alone scorings.
- 2026-10-17: part 1 is done. All 432 count runs are trained, scored and law-tested (432 of 432 present, 750 readings, 0 skipped). 216 runs trained on the GPU slice and 216 on CPU (PI-accepted). Evidence (gitignored, in the vault): [the grid summary](results/t125/GRID_SUMMARY_part1.md) holds every cell with its seed wobble; [the part-1 digest](results/t125/part1_digest.md) holds the drafted readings; [the full grid](results/t125/part1/grid.md), [the per-run device table](results/t125/part1/run_device.tsv) and the other `results/t125/part1/*.tsv` tables hold the rest; [FIR_PATH.txt](results/t125/FIR_PATH.txt) names the Nibi directories.
- 2026-10-17: kit K11 shipped, and part 2's second half is running on Nibi. The PI corrected its scope: the 8 real seed-0 f of one track (C19M16), one per design, with 96 h fits on f's NB (8 f × 12 h versions) and 8 chains. This replaces the earlier 768-fit and 16-f plans.
- 2026-10-17: the 96 h fits on f's NB (8 f × 12 h versions, one h seed each) are done on Nibi, 0 failed. 22 of 96 ran to the 4,000-step cap. 30 fits have 584 capped NB rows in all; the chain scoring has 0.
- 2026-10-17: 5 of 8 chains are done (designs C, D, A2, C2, D2). The 3 slow chains (A, B, B2) are split into 10 pair shards each on kit K12 (job 24313498), at most 3 of the 27 open pairs per shard. The sharding changes no number: chain C, re-assembled from its pair files on K12, equals the original.
- 2026-10-17: kit K13 shipped (Nibi tests 23 and 13 passed), and the part-2 aggregate ran on the 5 finished chains (job 24312235): 38 h alone, 5 of 8 chain f, 1,192 readings. The merged mixture at G = 512 is exact on C, A2 and C2; its per-bin max error is 8.6e-4 on D and 9.7e-4 on D2, below the 1e-3 bar. Evidence (gitignored, in the vault): [the part-2 digest](results/t125/part2_digest.md), [the combined grid summary](results/t125/GRID_SUMMARY.md) (part 2 marked "5 of 8 designs"), [the h grid](results/t125/part2/grid_h.md), [the repair table](results/t125/part2/h_repair.tsv), [the chain against the old grid](results/t125/part2/h_vs_oldgrid.tsv) and the other `results/t125/part2/*` files.
- 2026-10-17: the drafted part-2 reading is under [[q6_how_closely_does_a_map_h_reproduce_the_l|the question of how closely a map h reproduces the p track]]. It covers 5 of 8 designs and is redrafted when A, B and B2 land. Nothing is ticked, no verdict is recorded, and the task stays open.
- 2026-10-19: the PI stopped the run. Chain A finished; chains B and B2 timed out, their resubmission (job 24608447) and the 8-of-8 aggregate (24608467) were cancelled. The deck and the drafted reading under the p-track question stay at 5 of 8 designs. Record: `plan/T125_STATUS.md`, "Wrapped up 2026-10-19".
- Run record, job ids and PI decisions: `plan/T125_STATUS.md` on `exp/t118-counterfactual-f`.
- The drafted part-1 reading is under [[q5_does_a_depth_offset_on_f_the_source_dept|the depth-offset question]]. Nothing is ticked, no verdict is recorded, and the task stays open.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
