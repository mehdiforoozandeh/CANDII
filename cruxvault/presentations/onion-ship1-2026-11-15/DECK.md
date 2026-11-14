# Onion Shipment One: the deck of shipment 1 of the peeling-the-onion run

Artifact: https://claude.ai/artifact/4RSUdsFwJC3nYcyHb9s2bU (published 2026-11-14 about 21:45 UTC, private, not shared)

Source: `index.html` beside this file, built by `build_deck.py` from `deck_src.html` (prezit paper arc). Shipment 1 holds the Methods (data, evaluation) and the model-free Results; one closing slide previews the first model score (screening score, one seed, no conclusion). The written form is the paper note `INSTRUCTIONS/peeling_the_onion/onion paper.md`. The deck is derived, not evidence: no finding cites it.

## Anchor

The vault has no single onion question above the others. This shipment answers four root-level questions of the vault: the dataset question (node q17), the variance-layer question (q18), the baseline question (q19, with its open child on the correction scale, q32) and the AlphaGenome 2 question (q20). Region selection (q21) appears as an interim result. No tree node was created.

## Where the numbers come from

`build_deck.py` reads every number from the files below and embeds it as DATA; the deck-sources manifest in `index.html` holds the sha256 of every file read (32 files). The speaker notes of each slide (press `n`) list that slide's files.

- Dataset: `results/t183/DATASET_EIC_V1.md`, `results/t309/dataset_table.tsv`.
- Evaluation: `results/t185/SUBMITTED.md`, `results/t283/REGIONS.md`, `results/t283/regions.json`, `results/t311/SUBMITTED.md`, `results/t309/sources.md` (composite weights); `presentations/onion-status-2026-11-14/chrom_sizes.json` (its sha256 is checked against `REGIONS.md`).
- Variance layers: `results/t305/layers_by_assay.tsv`, `layers_by_stratum.tsv`, `LAYERS_REAL.md` (the rerun with the sampling layer).
- Baselines: `results/t307/BASELINES.md`, `baselines_headline.tsv`, `correction_headroom.tsv`, `calibration_baselines.tsv`, `pit_curves.tsv`, `noise_floor_by_assay.tsv`, `validity_checks.tsv`, `scoring_time_memory.tsv`, `src/baselines/<name>/seed0/scores.json` (per-assay correction values).
- AlphaGenome 2: `results/ag2/AG2_FACTS.md`, `results/t273/AG2_LEAKAGE.md`, `results/t308/SUBMITTED.md`. The board file (`results/t308/AG2_VS_BASELINES.md` or `ag2_board_val.tsv`) did not exist at the build, so the slide shows the audit only.
- Region selection: `results/t292/STRATA_REAL.md`, `INSTRUCTIONS/peeling_the_onion/onion region budget and selection.md`.
- Preview: `results/metrics_screening/first_scores.md`.
- Schedule: `INSTRUCTIONS/peeling_the_onion/run_state/WHERE_WE_ARE.md`.
- `results/` is gitignored: the build reads this worktree's copies.

## Rebuild

From inside the vault (`cruxvault/`):

```bash
/Users/mforooz/miniforge3/envs/candii/bin/python presentations/onion-ship1-2026-11-15/build_deck.py
python3 ~/.claude/skills/crux/scaffold/crux.py deck --verify presentations/onion-ship1-2026-11-15/index.html --strict
python3 ~/.claude/skills/crux/scaffold/crux.py deck --lint presentations/onion-ship1-2026-11-15/index.html
```

Verification on 2026-11-14: verify (strict) passed, 32 source files unchanged; lint clean. A changed `WHERE_WE_ARE.md` makes verify report the deck stale; rebuild then.

## Arc (18 slides)

All headlines are drafted by Claude for the PI's review.

1. Title: ==The data, the evaluation, and what the data say before any model is scored.==
2. Methods (a): ==The frozen dataset holds 249 of 267 training and 42 of 45 validation tracks; 20 of the 21 absent ones are DNase-seq.==
3. Methods (b): ==Every track was rebuilt from its reads with the standard ENCODE pipeline of its assay, then split into processing views and read halves.==
4. Methods (c): ==256 regions of 16,384 bins hold every score; the screening tier runs the pair tests on 16 of them.==
5. Methods (d): ==Each goal has a lead metric on a scale from zero to one; the composite weighs imputation twice as much as each other goal.==
6. Results (a): ==At 25 bp, read sampling is the largest of the three layers in every assay; the replicate layer is small.==
7. Results (b): ==The cell-type layer is small per bin but leads at the broad scale, and it is largest in promoters for accessibility and the active marks.==
8. Results (c): ==Four baselines need no training, and the scorer gives the two reference baselines exactly zero and one.==
9. Results (d): ==The position average recovers 0.119 of the specific variance above the noise floor, and sampling noise is 58% to 85% of that variance.==
10. Results (e): ==Depth-only scaling falls behind the floor when the target is deeper than the source, and leads it when the target is shallower.==
11. Results (f): ==The position average is closer to calibration than the marginal negative binomial, and both predict at least the variance between read halves.==
12. Results (g): ==The position average is the reference a model must beat on all three goals.==
13. Open point: ==The position average scores gap closed 2.86 [1.85, 6.59] on a scale whose one is the floor, so the floor is not a lower bound.==
14. Results (h): ==AlphaGenome 2 trained on most of our validation tracks; the fair comparison uses only the tracks it never saw, kept apart from the rest.==
15. Results (i): ==At this window size, sampling training windows by peak classes is almost uniform sampling.==
16. Results (j): ==On the baselines, the screening set raised every correction CRPS by 0.029 to 0.033, inside the full-set intervals.==
17. Preview: ==The first scored model run, beside the baselines on the same screening set== (screening score, one seed, no conclusion).
18. Next: ==Shipment 2 adds the model results on screening scores and a first Discussion.==

## Disagreements between sources, and how the deck handles them

- The gap closed of 2.86 is the scorer's macro: a ratio of mean CRPS per assay, then a mean over the 11 assays (`tools/onion_eval/score.py`, `metrics.gap_closed`). The ratio of the macro CRPS values is 1.23, and the pair-mean ratio on the 102 headline pairs (`correction_headroom.tsv`) is 1.13. The open-point slide shows the per-assay numerators and denominators so that the reader sees why.
- The audit (`AG2_LEAKAGE.md`) and the Methods of the paper note say 12 unseen validation tracks in 10 cells and 32 trained-on tracks; the pack (`t308/SUBMITTED.md`) has 11 in 9 and 31, because two validation DNase-seq tracks are absent from `eic_v1`. The slide shows both and says why.
- `BASELINES.md` lists the screening correction CRPS shift as 0.029 to 0.036; computed over the four predictors' macro values of `baselines_headline.tsv` it is 0.029 to 0.033 (the 0.036 is the marginal NB's oracle-scaled part). The deck uses the computed range.
- `WHERE_WE_ARE.md` lists iteration 2 as the region check, budget curve, dataset update and CPU work on Fir; the plan's Amendment 10 puts the budget curve in later phases. The last slide names only the ablations and the first settled selections.
