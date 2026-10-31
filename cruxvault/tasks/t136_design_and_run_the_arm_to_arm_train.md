---
id: t136
type: task
title: design and run the arm-to-arm training grid: g trained on base<->arm plus some arm->arm depth pairs, scored on held-out arm->arm pairs against the via-base chain (exploratory)
category: implementation
parent: t118
blocked_by: None
refs: h28, q7
hypothesis_refs: 
status: open
created: 2026-10-29T20:08:16
updated: 2026-10-29T20:08:16
---

# t136 — design and run the arm-to-arm training grid: g trained on base<->arm plus some arm->arm depth pairs, scored on held-out arm->arm pairs against the via-base chain (exploratory)

Refs:: [[h28_training_g_on_arm_to_arm_pairs_lets_a_di\|h28]], [[q7_why_do_the_covariate_conditioned_g_f_des\|q7]]

## Why

tells whether CANDI needs counterfactual augmentation or a base pivot (grid on Nibi, PI 2026-10-29)

## Decisions and runs

- PI 2026-10-29: the null (Leakage) approved, checks c1–c5 accepted at their recommended thresholds, design option A (no extra seeds), the across-track g dropped; see h28 and `plan/T136_ARM2ARM_GRID.md`.
- PI ruling 2026-10-29: run the grid on Nibi at 2 CPUs, 10000M, 7 h per run, not on Fir. Reason: the Fir def-maxwl_gpu fair-share was exhausted, Fir billed by 4 CPUs (MAX_TRES), and 12 h was requested against a median near 86 min.
- Foreman amendment A2 (2026-10-29): the Dref and Dviabase references stay on Fir; the readings extract the references on Fir and the grid on Nibi, and join them locally.
- Foreman amendment A5 (2026-10-29): smoke first, then the full grid only if Darm2's projected wall time fits 7 h.
- Kit Nibi `/project/def-maxwl/mforooz/t136/code/K1` (1f634e1). Smoke: train 25389086 (rows 9, 30) and law 25389100, COMPLETED; Darm1 11:55, Darm2 16:59 for 300 steps; projection at 12 000 steps Darm1 < 1 h, Darm2 < 1.5 h.
- Full grid, 2026-10-30 05:45Z: Darm1 train 25403683 + law 25403688; Darm2 train 25403711 + law 25403725. Output `/project/def-maxwl/mforooz/t136`, ledger `jobs.tsv` there. Never resubmit.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
