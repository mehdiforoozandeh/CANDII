---
id: t130
type: task
title: score the one-warp check set (blind, onewarp, model, oracle) on the counterfactual depth and run-type pairs
category: implementation
parent: t118
blocked_by: t137
refs: h1, h2
hypothesis_refs: 
status: open
created: 2026-10-25T00:22:02
updated: 2026-10-25T00:22:02
---

# t130 — score the one-warp check set (blind, onewarp, model, oracle) on the counterfactual depth and run-type pairs

Refs:: [[h1_conditioning_on_the_recorded_sequencing_\|h1]], [[h2_conditioning_on_the_recorded_run_type_pr\|h2]]

## Why

runs h1 and h2 on the counterfactual arms and g -> f, replacing the in-vitro testbed that was never built; scored once the resampling floor lands, against three baselines: the monotone scale map, the shuffled-covariate twin, and an oracle: the pseudoreplicate floor (PI ruling 2026-10-29), replaced for depth pairs by the exact nested-read oracle (PI ruling 2026-10-30)

## Decisions and runs

- PI ruling 2026-10-29: score blind against three baselines: the one-warp map (QuantileMatching), the no-covariates twin, and the pseudoreplicate floor as oracle. Every rung (A–D) and both g versions are reported; the model rung for any verdict is the PI's pick (h1 item vi, plan Decision v). Readings only, every threshold `TODO(PI)`.
- Foreman decision 2026-10-29 (G3 review, amendment A3): the floor of a pair is the same-track base product's depth curve read at the target's depth.
- Pass 1, 2026-10-30: extract job 25402234 on Nibi, kit `/project/def-maxwl/mforooz/t137/code/K2` (6743ab6), 576 row-1 ladder runs; second method 6 AGREE, 0 DISAGREE. Finding: on counts depth pairs the floor sat above one-warp (an inverted interval), because the depth arms share reads with the base.
- PI ruling 2026-10-30 (amendment A9): the oracle for depth pairs is the exact nested-read oracle: base → arm Binomial(base, r), a true ceiling; arm → base the exact per-bin Bayes posterior with an empirical prior from the training chromosomes, blacklist-free, a ceiling only for a per-bin predictor. The floor stays as a context column. Run type and −log10 p keep the floor as oracle.
- Nested oracle code merged at 8aa670b (oracle labels `binomial_ceiling` / `bayes_per_bin`). First submission 25411091 failed ×7 (numpy imported before the venv); fix da96b2d, kit `/project/def-maxwl/mforooz/t130/code/K2`; resubmitted as array 25412625 (tasks 0–6) and aggregate 25412626, both COMPLETED. Pass 2: 7 AGREE, 0 DISAGREE. Pass 1 is kept in `results/t130/pass1_floor_oracle/`.
- Pass 2 readings (nested-read oracle, counts, not judged): base → arm, rungs B–D close 0.77–0.92 of the one-warp → oracle gap in every mark class, gain > 2 × seed |Δ| in 18 of 18 cells; rung A ≤ 0 in DNase and narrow. arm → base (per-bin ceiling): C–D 1.1–3.7, B 0.23–0.86. Dispersion of the real arms 0.995–1.006 of binomial. h2 (paired-end, floor oracle) unchanged: the model beats one-warp > 2 × seed |Δ| in 16 of 16 narrow and broad cells; DNase mixed.
- Conclusion pending the PI: the model rung (h1 item vi) and the `TODO(PI)` thresholds decide it. This task stays open until then.
- Nibi output `/project/def-maxwl/mforooz/t130`. Digest: `results/t130/DIGEST.md`. Never resubmit these jobs.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
