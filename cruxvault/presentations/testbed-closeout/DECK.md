# Close-out of the covariate artifact correction (CAC) testbed: lab deck

Artifact: not published yet (the foreman publishes). Add the URL here when it exists.

Source: `index.html` beside this file, built by `build_deck.py` from `deck_src.html`. Anchor: [[q1_do_the_recorded_experimental_covariates_|q1]] and everything under it. Tasks: [[t143_close_out_deck_and_story_of_the_who|the close-out task]] and [[t146_build_and_run_the_final_cac_testbed|the final CAC testbed experiments]]. PI requests: 2026-10-30 (one deck that marks the testbed done and states, for every goal, fully met, partially met, or not met but answered) and 2026-11-03 (cover the whole testbed, the final experiments included; one big deck, one idea per slide).

Each claim carries the PI's reading and confidence as recorded in its finding (f5, f6, f7, f30, f31, f32 are read from their frontmatter at build time). The goals' status words are typed in `deck_src.html` (`GOALS`); they summarise those readings and the PI can change them there.

## Where the numbers come from

- Result tables, read at build time: `results/t112/MANIFEST.tsv`, `results/t130/{blind_table,runs}.tsv`, `results/t136/{checks_readings,m5_depthlaw_table,m1_heldout_table,n_table}.tsv`, `results/t139/{knob_vs_floor,pairs}.tsv`, `results/t142/{floor_at_30M,compare_real_class,map_classes}.tsv`, `results/t142/t141_readthin/knob_classes.tsv`, and for the final experiments `results/t146/{analyze_out.txt,checks_h29,checks_h30_cells,checks_h30_pooled,checks_h31,checks_h31_per_track_seed,depthlaw,pooled,sacct_per_arm}.tsv`. `results/` is gitignored, so the build can copy these files in first (see "Rebuild").
- Earlier readings, read by regex from tracked nodes and findings: h1, h15, h26, q2–q8, f1–f8, f17, f19–f21, f30–f32, the power lines of h29–h31, `wiki/encode-imputation-challenge.md` (motivation) and `notes/mini_candi_design_inputs.md` (reprocessing cost, §7). If one of those sentences changes shape, the build stops and names it.
- The lowest-mappability-decile numbers of the mappability experiment are recomputed from `checks_h31_per_track_seed.tsv` (comparator seeds 0–2, per-seed mean over tracks); they reproduce `REPORT.md` and `analyze_out.txt` exactly.
- The deck-sources manifest in `index.html` holds the sha256 of every file read; `crux deck --verify` re-reads them.

## Rebuild

From inside the vault (`cruxvault/`):

```bash
CLOSEOUT_SYNC_FROM=/Users/mforooz/Desktop/research/libbrechteam@sfu/CANDII/cruxvault/results \
  /Users/mforooz/miniforge3/envs/candii/bin/python presentations/testbed-closeout/build_deck.py
python3 ~/.claude/skills/crux/scaffold/crux.py deck --verify presentations/testbed-closeout/index.html --strict
python3 ~/.claude/skills/crux/scaffold/crux.py deck --lint presentations/testbed-closeout/index.html
```

Without `CLOSEOUT_SYNC_FROM`, the build reads the copies already under this vault's `results/`. A file missing from the sync source is skipped at copy time and fails at its read.

## Arc (32 slides)

- **Motivation (1–4).** Title and scope. The ENCODE Imputation Challenge: an average baseline beat all but two of 23 entries, because of a single-end against paired-end duplicate-removal rule. Why a testbed: CANDI pairs an experiment only with a thinned copy of itself, and its corpora hold no reads. The question q1 and its 17 goals, coloured by status.
- **Method (5–7).** The counterfactual arms and the grid of knobs; the model (g reads (C, C′) and sets f; designs A–D); references, scored pairs, the CRPS split and the significance criterion.
- **Results, the lessons (8–19).** (a) covariates against the twin; (b) depth (PI: supports, B–D); (c) run type; (d) capacity and the depth offset; (e) the depth-law miss is a transfer failure, not capacity, width, loss, pseudocount or encoder-decoder; (f), (g) arm-to-arm training teaches the scale, not the distribution (PI: against); (h) predict counts, derive p by formula (PI: count space only); (i) knob ranking against the floor; (j) the resampling floor; (k) where knobs act; (l) the covariate-swap probe on trained CANDI (PI reading 2026-10-31: wrong direction; parked).
- **Method, the final experiments (20–21).** Three schematics on one shared count-space model; the pre-registration grid with the post-smoke repairs and power.
- **Results, the final experiments (22–24).** (m) joint objective: against, weak; (n) pivot through the base: against, moderate, with the agreement-loss reading marked exploratory; (o) mappability input: supports, weak, with the failed depth no-change control.
- **Discussion (25–30).** Ruled in and out; what goes into the mini-CANDI prototype; limitations (7 tracks, power 41–51%, the twin-check defect, the failed control); the goals scorecard over two slides; next-step options A–E with cost, unranked.
- **Backup (31–32).** Two glossary slides.

## Checks at this build (2026-11-03)

`crux deck --verify --strict`: 54 source files unchanged, every addressed number matches. `crux deck --lint`: clean. Rendered at 1280 × 720: no slide runs into the footer, no figure is empty.

Every number in this file is a copy of the deck at the time of writing; the deck itself reads the tables at build time.
