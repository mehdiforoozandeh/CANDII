# Close-out of the counterfactual in-vitro covariate testbed: lab deck

Artifact: not published yet (the foreman publishes). Add the URL here when it exists.

Source: `index.html` beside this file, built by `build_deck.py` from `deck_src.html`. Anchor: [[q1_do_the_recorded_experimental_covariates_|q1]] and everything under it. Task: [[t143_close_out_deck_and_story_of_the_who|the close-out task]]. PI request 2026-10-30: one deck that marks the testbed done and states, for every original goal, fully met, partially met, or not met but answered.

The deck gives readings, not verdicts. The depth (h1), run-type (h2) and arm-to-arm (h28) conclusions wait for the PI's signature, and the deck says so on the slides.

## Where the numbers come from

- Result tables, read at build time: `results/t112/MANIFEST.tsv`, `results/t130/{blind_table,runs}.tsv`, `results/t136/{checks_readings,m5_depthlaw_table,m1_heldout_table,n_table}.tsv`, `results/t139/{knob_vs_floor,pairs}.tsv`, `results/t142/{floor_at_30M,compare_real_class,map_classes}.tsv`, `results/t142/t141_readthin/knob_classes.tsv`. `results/` is gitignored, so the build copies these files in first (see "Rebuild").
- Earlier readings, read by regex from the tracked nodes h1, h15, h26, q2, q3, q4, q5, q6, q7 and q8. If one of those sentences changes shape, the build stops and names it.
- The deck-sources manifest in `index.html` holds the sha256 of every file read; `crux deck --verify` re-reads them.

## Rebuild

From inside the vault (`cruxvault/`):

```bash
CLOSEOUT_SYNC_FROM=/Users/mforooz/Desktop/research/libbrechteam@sfu/CANDII/cruxvault/results \
  /Users/mforooz/miniforge3/envs/candii/bin/python presentations/testbed-closeout/build_deck.py
/Users/mforooz/miniforge3/envs/candii/bin/python ~/.claude/skills/crux/scaffold/crux.py deck --verify presentations/testbed-closeout/index.html --strict
/Users/mforooz/miniforge3/envs/candii/bin/python ~/.claude/skills/crux/scaffold/crux.py deck --lint presentations/testbed-closeout/index.html
```

Without `CLOSEOUT_SYNC_FROM`, the build reads the copies already under this vault's `results/`.

## Refreshing the arm-to-arm slides

Slides 12 and 13, the scorecard row "Training g on arm-to-arm pairs" and the "Open runs" limitation read `results/t136` at build time. At this build (2026-10-31) the grid is complete: both arms finished 21 of 21 runs, both controls (c2, c5) pass, and the rule reading is final (FAIL, on the CRPS check c4). The PI has not signed. Slide 13 adds the second arm's CRPS as a reading, not a check: it reaches the through-the-base CRPS. The steps below stay for any later re-run of the readings.

1. Re-run the arm-to-arm readings in the main checkout when Darm2 is complete (`tools/t136/readings/run_all.sh`, as G15 did), so `results/t136/*.tsv` are current.
2. Rebuild with `CLOSEOUT_SYNC_FROM` as above. The second-arm column, the rule line, the partial markers and the counts update from the tables.
3. Re-read the prose around the changed numbers. The headline of slide 12 is computed: it reads "meets the depth law ... but fails the CRPS check" only while the law checks pass and the CRPS check fails. The scorecard status for this row is typed (Partially met); change it in `deck_src.html` (`GOALS`, key `h28`) if the PI's reading changes it.

## Arc (23 slides)

- **Motivation (1–3).** The title and scope. Why a testbed: CANDI pairs an experiment only with a thinned copy of itself, so only depth changes. The question q1 and its 14 goals, coloured by final status.
- **Method (4–6).** The counterfactual arms (one base per track, one knob per arm) with the grid of knobs. The model: g reads (C, C′) and sets f; designs A–D. The references (one-warp, no-covariates twin, oracle), the pairs that are scored, the CRPS split and the significance criterion (gain above 2 × the seed range).
- **Results (7–18).**
  - (a) The covariates beat the twin on trained pairs; the gain falls on never-trained pairs.
  - (b) Depth: designs B–D close most of the one-warp-to-oracle gap with the exact thinning oracle.
  - (c) Run type: histone marks beat the one-warp; DNase is mixed; no oracle exists.
  - (d) The depth offset repairs design A's scale error; capacity in f does not fix never-trained pairs.
  - (e) The depth-law miss: four causes ruled out; the chain through the base meets the law.
  - (f), (g) Arm-to-arm training: the grid is complete and both controls pass; the law checks pass and the CRPS check fails, DNase-driven and scale error, so the rule reading is final (fail, not judged). The second arm, read and not checked, reaches the through-the-base CRPS.
  - (h) The map from counts to the p-value track is stable and limited by the count model.
  - (i) Every knob moves peak height above the floor; few move the shape above it.
  - (j) The resampling floor: whole-read thinning matches the real halves; signal level, not region, sets it.
  - (k) Where knobs act: depth uniformly; mapping quality and read length where mappability is low.
  - (l) The trained CANDI probe could not be read; it is parked.
- **Discussion (19–22).** Ruled in, ruled out, located. Limitations. The goals scorecard (the deck's second table). What carries to CANDI.
- **Backup (23).** Glossary.

## Goals scorecard at this build

| goal | status |
|---|---|
| The covariates beat the scrambled twin (main claim, h15) | partially met |
| Depth, beyond the one-warp (h1) | fully met in readings, not judged |
| Run type, beyond the one-warp (h2) | partially met, not judged |
| Read-side knobs (q2) | partially met |
| Control-side knobs (q3) | partially met |
| How expressive f must be (q4) | partially met |
| Does a depth offset help (q5) | fully met |
| A map from counts to the p-value track (q6) | fully met |
| Why the depth law is missed (q7) | partially met |
| Training g on arm-to-arm pairs (h28) | partially met, not judged |
| Does CANDI's training block the other covariates (q8) | not met, answered |
| Which knobs move a track the most (q9) | fully met |
| How much resampling alone moves a track (q10) | fully met |
| Where along the genome each knob acts (q11) | fully met |

Every number in this file's summaries is a copy of the deck at the time of writing; the deck itself reads the tables at build time.
