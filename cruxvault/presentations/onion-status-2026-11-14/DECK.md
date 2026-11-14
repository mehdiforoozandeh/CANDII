# Onion Run Status: status deck of the peeling-the-onion run, 2026-11-14

Artifact: https://claude.ai/artifact/FwKjytRr1UuhT3xhTpWDrL (published 2026-11-14, private)

Source: `index.html` beside this file, built by `build_deck.py` from `deck_src.html` (prezit paper arc, used as a status report). PI request 2026-11-14: one deck of the status so far: the story, what is done, what is left; what the 256 regions and the 16 regions are, why there are two sets, why not more regions, and what the iterative-deployment iterations are. No vault question covers a status report, so the deck has no anchor node; the study slug is `onion-status-2026-11-14`. Not evidence: no finding cites this deck.

## Where the numbers come from

Every number is read by `build_deck.py` from the files below and embedded as DATA; the deck-sources manifest in `index.html` holds the sha256 of every file read. The speaker notes of each slide list that slide's files.

- Dataset and evaluation set: `results/t183/DATASET_EIC_V1.md`, `results/t185/SUBMITTED.md`.
- Regions: `results/t283/REGIONS.md`, `results/t283/regions.json`; `chrom_sizes.json` beside this file is a copy of Nibi `/project/def-maxwl/mforooz/ONION/genome/chrom_sizes.json`, and the build checks that its sha256 equals `chrom_sizes_sha256` of `REGIONS.md`.
- Screening set with 16 pair regions: `results/t311/SUBMITTED.md`.
- Pre-registered pair-region check: `results/pair_region_check/PAIR_CHECK_PREREGISTRATION.md`; ruling 41 from `INSTRUCTIONS/peeling_the_onion/long-horizon notepad.md`; prior evidence from `results/region_budget/REGION_RESEARCH.md`.
- Cost of the two tiers: `results/predict_store_speed/FORWARD_SPEED.md` (window counts, measured speed, projected GPU hours), `results/scorer_memory/NOTES.md` (measured baseline scoring time), plan snapshot `INSTRUCTIONS/peeling_the_onion/run_state/plan.md` (store size and metric CPU hours: planning estimates).
- Variance layers: `results/t305/layers_by_assay.tsv`, `layers_by_stratum.tsv` (broad band, stratum all), `LAYERS_REAL.md` (the rerun with the sampling layer; the first run is not used).
- AlphaGenome 2: `results/t273/AG2_LEAKAGE.md`, `results/t308/SUBMITTED.md`, `results/t284/calls_summary.tsv`, `results/t276/MAX_REPRO.md`.
- Trainings and status: `results/t213/SUBMITTED.md`, `results/t291/SUBMITTED.md`, `results/trainings_iteration1/SUBMITTED.md`, `results/arms_d10_d15/SUBMITTED.md`, `INSTRUCTIONS/peeling_the_onion/run_state/WHERE_WE_ARE.md` (state at 2026-11-14 10:25 UTC, and the schedule).
- `results/` is gitignored: the build reads this worktree's copies.

## Rebuild

From inside the vault (`cruxvault/`):

```bash
/Users/mforooz/miniforge3/envs/candii/bin/python presentations/onion-status-2026-11-14/build_deck.py
python3 ~/.claude/skills/crux/scaffold/crux.py deck --verify presentations/onion-status-2026-11-14/index.html --strict
python3 ~/.claude/skills/crux/scaffold/crux.py deck --lint presentations/onion-status-2026-11-14/index.html
```

Verification on 2026-11-14: verify (strict) passed, 26 source files unchanged; lint clean.

## Arc (16 slides)

All headlines are drafted by Claude and marked for review.

1. Title: ==Peeling the onion: where the run stands on 2026-11-14.==
2. Story: ==We split the variance of an epigenomic signal into layers, then ask how much of each layer a small model can predict.==
3. Done (a): ==The dataset and the evaluation set are frozen.==
4. Done (b): ==At 25 bp, read sampling is the largest layer of every assay; at the broad scale the cell-type layer leads.==
5. Done (c): ==AlphaGenome 2 predictions are cached, the tracks it trained on are known, and our scorer matches Max's.==
6. Done (d): ==The first models are trained but not scored; the fast scorer splits a scoring into GPU forwards and CPU metrics.==
7. Regions (a): ==A region is one fixed stretch of the genome; 256 of them were drawn at random from 6,724 candidate blocks.==
8. Regions (b): ==Two kinds of test run on the regions: imputation items and pair tests.==
9. Regions (c, d): ==The 256 regions are the frozen set; the screening set runs pair tests on every 16th of them only.==
10. Regions (e): ==Pair tests are most of a scoring; on 16 regions a rung-1 scoring needs 10 times fewer GPU hours== (computed at render).
11. Regions (f): ==More regions wait for evidence; a pre-registered check decides whether 16 pair regions are enough, and it has not run yet.==
12. Iterations (a): ==Iterative deployment: few iterations, many parallel paths, and a paper that grows one section at a time.==
13. Iterations (b): ==Iteration 1 runs eleven paths at once and ships twice; iteration 2 holds only the work that needs its results.==
14. Now: ==Trainings, baseline scorings and the Globus copy run; the metric phase is the last piece of the fast scorer.==
15. Left: ==The queue, the unmerged metric phase and the lab's file limit are the risks to ship 2.==
16. Close: ==Two decks in the next day and a half, then iteration 2.==

## Disagreements between sources, and how the deck handles them

- `WHERE_WE_ARE.md` says the sampling layer is missing and the rerun is in work; `results/t305/LAYERS_REAL.md` holds the rerun with the sampling layer. The deck uses the note and marks the layers as done.
- `WHERE_WE_ARE.md` lists iteration 2 as "16 versus 256 region check, budget curve, dataset update, CPU work on Fir"; Amendment 10 of the plan puts the budget curve and the check on rung 1 in later phases and iteration 2 on the ablations of the best rungs. The deck follows Amendment 10 and says so in the notes of slide 13.
- `LAYERS_REAL.md` gives the broad-band cell-type share as "up to 0.90" in its main results and "0.60 to 0.91" in its band section. The deck computes the shares from `layers_by_stratum.tsv` (91% maximum among the assays shown).
- The plan text says 12 unseen validation tracks in 10 cells for AlphaGenome 2; the pack (`results/t308/SUBMITTED.md`) has 11 in 9, because two validation DNase-seq tracks are absent. Panel (a) of slide 5 shows the audit (45 tracks), panel (b) the pack.
- Rung-1 forward speed: 93 windows per second in `scorer_memory/NOTES.md` (serial legacy path), 79.6 in `FORWARD_SPEED.md` (end to end, the new path). The deck uses 79.6.
- The plan Log's clock times after the usage-limit resume differ from `results/t311/SUBMITTED.md` by about 25 minutes (the Log notes this). The deck quotes no Log time.
