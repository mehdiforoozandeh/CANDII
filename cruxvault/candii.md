---
id: root
type: project
title: CANDII
status: active
created: 2026-08-18T17:09:52
updated: 2026-08-18T17:09:52
---

# CANDII

Project root — the whole research program. Top-level questions point here via `Parent::`.

## Goal

A self-supervised epigenome imputer/denoiser over raw counts, emitting a Negative Binomial (n-hat, p-hat) per assay per 25 bp bin, conditioned on four experimental covariates on both the input and the output side so that it imputes and denoises zero-shot on cell types it has never seen.

## Programme

Two strands of work run under this root (as of 2026-10-24).

- **The benchmark.** A leaderboard scores CANDI and rival imputation methods on the EIC challenge tracks, with a distributional metric suite (`EVAL.md`, `plan/BENCHMARK_DESIGN.md`, `plan/LEADERBOARD_PRD.md`). This work lives in the taskhub, for example [[t81_retrain_every_trainable_method_under|the uniform retrain]] and [[t99_stamp_24_marginal_s_genome_wide_stor|the genome-wide marginal rows]]. No question node holds its results yet. The merge gate for experiment branches waits on it (`CLAUDE.md`).
- **The counterfactual testbed.** [[q1_do_the_recorded_experimental_covariates_|The covariate question]] asks whether the recorded covariates alone carry enough information to map one processing of a track onto another. Seven EIC tracks were reprocessed with the ENCODE pipelines into a base plus arms that change one of ten knobs. A small generator g reads the source and target covariates and outputs a transformation f of the source track. The readings so far are exploratory: g beats a scrambled-covariate twin on trained pairs, gains less on never-trained pairs, and misses the depth law.

## Navigate

See [[META]] for the live question tree and dashboard, and [[EXPERIMENTS]] for the flat experiment registry.
