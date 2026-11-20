---
id: t351
type: task
title: "Onion: replicate ladder in the two-sided form, with the joint cluster bootstrap and the common-cell check"
category: implementation
parent: t152
blocked_by: None
refs: q59
hypothesis_refs: 
status: done
created: "2026-11-18T09:46:40"
updated: "2026-11-18T09:47:49"
---

# t351 — Onion: replicate ladder in the two-sided form, with the joint cluster bootstrap and the common-cell check

Refs:: [[q59_do_the_nested_replicate_ladder_and_the_c\|q59]]

## Why

gives the cell-type rung that the evaluation section may quote (PI decision 5 of 2026-11-17)

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- presentations/onion-closeout-2026-11-18-report/report.pdf

## Evidence

Built on the umbrella branch (merge a480b3b): `tools/onion_eval/ladder.py report --form two_sided` is the default; the one-sided form is an option and reproduces the Nibi report of job 27692829 byte for byte. The evidence note `results/closeout_E4/ladder_two_sided_2026-11-18/COMPARE.md` is git-ignored and exists in the foreman worktree only; the tracked copy of the numbers is the close-out report (section on the replicate ladder) and deck slide 12.

- Two-sided form: each cross-experiment pair subtracts 0.5 x (L(a) + L(b)), where L(x) is the mean biological-replicate pair value of the experiment of side x. Interval: 95% cluster bootstrap that resamples all cells jointly (coverage 281 of 300 synthetic data sets, 93.7%).
- Cell-type rung, all regions and bands: broad histone marks 0.00273 [0.0015, 0.00396] (355 pairs, 16 cells); narrow histone marks 0.003547 [0.00198, 0.00511] (445 pairs, 23 cells). One-sided values were 0.003597 and 0.004129; most of the drop comes from pairs whose b-side experiment has no replicate pair and leaves the matched set.
- Other-experiment rung: not estimable in any class, because no second experiment has a replicate pair of its own (pair list of 3,979 pairs).
- Common-cell check: 167 of 195 (class, stratum, band) groups pass; all 28 failures are DNase-seq strata.
- DNase-seq depth defect: `layers.py` sets the common depth of an assay to the shallowest side of any pair. ENCSR000EKG (cell C23) has 15,201 reads, so every DNase-seq pair value in the replicate ladder and in the second-experiment layers is computed at 15,201 reads. The next shallowest side of a DNase-seq pair has 13,300,584 reads (875 times more). The DNase-seq layer values are therefore not interpretable. The earlier reading "3.40" of the second-experiment DNase-seq increment was 3.40e-05.
