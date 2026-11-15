# Onion Shipment Two: the deck of shipment 2 of the peeling-the-onion run

Artifact: pending (the foreman publishes it and adds the URL here and in `../ARTIFACTS.md`).

Source: `index.html` beside this file, built by `build_deck.py` from `deck_src.html` (prezit paper arc). Shipment 2 holds the model Results on screening scores and a first Discussion. The written form is the paper note `INSTRUCTIONS/peeling_the_onion/onion paper.md` (Model, Results 4 to 8, Discussion). The deck is derived, not evidence: no finding cites it.

## Nodes this deck covers

The vault has no single onion question above the others. The deck covers: the ladder question (q39, finding f78, open child q42), the wiring-check hypothesis (h40, finding f79, open question q43 under q23), the hard-offset question (q36, finding f81, open child q41) under the eta model-design question (q34), the selection hypothesis (h34, finding f80) and the AlphaGenome 2 question (q20, findings f75 and f76).

## Where the numbers come from

`build_deck.py` reads every number from the files below and embeds it as DATA; the deck-sources manifest in `index.html` holds the sha256 of every file read (19 files). The speaker notes of each slide (press `n`) list that slide's files.

- Score tables: `results/metrics_screening/scores_{ladder,rung1_arms,rung4_arms,selection_arms}.md` and `results/metrics_screening/sh2_refresh/` (the refreshed tool outputs of 2026-11-15T07:18Z, 58 scored runs: `ladder_tb2.tsv`, `rung1_arms_tb2.tsv`, `rung4_arms_tb2.tsv`, `selection_arms_tb2.tsv`, `table_runs.tsv` and `table_region.tsv` for the uncertainty term per seed, `runjson_rungs.tsv` for parameters and training FLOPs, `checks_tb2.tsv`, `scored_list.txt`, `run_time.txt`).
- Composite weights: `results/t309/sources.md`. The build recomputes every seed's composite from its three terms and stops if one differs by more than 2e-6.
- Oracle floor: `results/oracle_floor/ORACLE_FLOOR.md`.
- AlphaGenome 2 board: `results/t308/AG2_VS_BASELINES.md`.
- Eta view: `results/eta_space/depth_invariance.tsv`, `LAYERS_ETA.md`, `THEORY.md`.
- Schedule: `INSTRUCTIONS/peeling_the_onion/run_state/WHERE_WE_ARE.md`.
- `results/` is gitignored: the build reads this worktree's copies.

## Rebuild

From inside the vault (`cruxvault/`):

```bash
/Users/mforooz/miniforge3/bin/python presentations/onion-ship2-2026-11-15/build_deck.py
python3 ~/.claude/skills/crux/scaffold/crux.py deck --verify presentations/onion-ship2-2026-11-15/index.html --strict
python3 ~/.claude/skills/crux/scaffold/crux.py deck --lint presentations/onion-ship2-2026-11-15/index.html
```

Verification on 2026-11-15: verify (strict) passed, 19 source files unchanged; lint clean. Every slide was rendered at 1600 x 900 in headless Chrome and checked by eye. A changed `WHERE_WE_ARE.md` makes verify report the deck stale; rebuild then.

## Arc (14 slides)

All headlines are drafted by Claude for the PI's review.

1. Title: ==The model results on screening scores, and a first Discussion.==
2. Recap: ==One model imputes a missing assay, corrects a track to other processing settings, and states how sure it is.==
3. Methods: ==With the oracle floor as the top of the scale, every model lands between depth-only scaling and one.==
4. Results (a): ==Every architecture beats the position average; rungs two and three lead correction and rung four leads imputation.==
5. Results (b): ==Rung two has the highest composite, but its gain over rung one is below the two seed ranges that selection asks for.==
6. Results (c): ==No architecture leads both goals: rung four buys imputation with correction.==
7. Results (d): ==No wiring change moves the pointwise rung's composite beyond the seed range; scrambling the covariates costs imputation.==
8. Results (e): ==At rung four, scrambling the covariates raises gap closed on all pairs and lowers it on the clean pairs; hiding depth costs calibration.==
9. Results (f): ==Hiding depth from the network lowers the composite by 0.184, and most of the loss is calibration.==
10. Results (g): ==No selection rule beats uniform tiling; two lose on the composite, through the uncertainty term.==
11. Results (h): ==On the 11 fair tracks, AlphaGenome 2 beats every training-free baseline; our model has no row yet.==
12. Results (i): ==Above read sampling, the layers do not change with depth in eta space; on log counts they do.==
13. Discussion: ==Five first statements, each with its evidence and its limit.==
14. Next: ==17 runs of 8 arms are not scored yet, every score is a screening score, and our model has no AlphaGenome 2 row.==

## Choices and disagreements

- The seed range (maximum minus minimum) is the project's seed statistic (select_rule); the deck says "seed range", not "standard deviation", because no standard deviation is computed.
- The correction CRPS is left off every slide, because the tables do not carry its oracle-scaled and scale-error split (AGENTS.md section 7.2); gap closed is shown instead.
- The uncertainty term is not in the four score notes' tables; the deck reads it per seed from `table_runs.tsv` and `table_region.tsv` (the `select_rule.py table` output) and computes the paired differences in the page.
- The build of the cluster table script (`/scratch/mforooz/onion/tb2/build_tables_tb2.py`) lists the rung-4 arms by hand and leaves out the hard-offset arm. The refresh used a local copy with that arm added (`results/metrics_screening/sh2_refresh/build_tables_tb2.py`); the cluster copy is unchanged.
- The brief's hard-offset uncertainty term was -0.098; the seed mean computed from the three seed values (-0.182416, -0.141991, +0.028801) is -0.0985, which rounds to -0.099. The deck and the vault use -0.099.
