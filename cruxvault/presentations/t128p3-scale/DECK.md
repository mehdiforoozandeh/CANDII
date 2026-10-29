# A bigger g and a bigger f for old D through the base, phase 3: lab deck

Artifact: https://claude.ai/artifact/5JfcGnVRdH8NESQeSWp9tp

Source: `index.html` beside this file, built by `build_deck.py` from `deck_src.html`. Its numbers come from `cruxvault/results/t128p3/`, which is gitignored; the build reads the main checkout's copy by default (override with `T128P3_RESULTS=<dir>`). To rebuild, run these from the worktree root: `bash tools/t128p3/readings/run_all.sh`, then `python cruxvault/presentations/t128p3-scale/build_deck.py`. The design is in `plan/T126_ENCDEC_D2.md` § "Phase 3: PI decisions, 2026-10-27" and its "Design check" block; the run log is in `plan/T118_STATUS.md` § "t128 phase 3". The build fails if an arm code reaches the page.

The phase is exploratory: the deck gives readings, not verdicts.

## Arc (17 slides)

- **Motivation (slides 1–3).**
  - Phase 2: one g per track, old D through the base was the only route below old D direct on never-trained pairs.
  - The two arms, drawn: a bigger g (hidden width 64 → 256) and a bigger f (32 → 64 channels, so g's output grows from 256 to 512 FiLM values). Each grows one part only.
- **Method (slides 4–6).**
  - The grid of arms: sizes, parameter counts of g, the 15,000-step cap against the reference's 12,000, runs scored.
  - The two routes (direct, and through the base), the pairs that are scored, selection on chromosome 22 and scoring on chromosomes 19 and 21.
  - The CRPS split (shape and scale), the noise rule, and the pre-registered thresholds read from the tables. Seeds do not pair across arms.
- **Results (slides 7–14).** Each metric slide gives the definition, the best case and the worst case, and then the figure with its noise labels and thresholds.
  - Metric 1 through the base, pooled: one g per track, bigger f 0.530 and bigger g 0.534 against the reference's 0.545. Both gains stay under the pre-registered threshold of 0.0154; bigger f's gain of 0.0146 is just under it. One g across tracks, the thresholds are 0.128 and 0.258, and both arms are within noise.
  - Metric 1 per knob: no arm clears its threshold on any knob.
  - Metric 2, the generalisation gap: negative for every model, as in phase 2.
  - Metrics 3 and 4, trained pairs and the held-out 7.5M depth: the bigger models cost nothing.
  - Metric 5, the depth law (a sanity check): it holds on trained pairs, and the routes through the base stay closer on never-trained depth pairs.
  - Metric 9, steps: no best step above 12,000, so no run is step-confounded; median best steps 2,250–2,750.
  - The three routes through the base ranked by metric 1, with the direct reading beside, as a reading.
- **Discussion (slides 15–17).**
  - One line per metric.
  - What follows: no arm reads as an optimisation failure; by the spec's rule, no combined arm or second size follows yet. Three options for the PI: close the scale question, more seeds for bigger f, or tune the learning rate.
  - Four open points for the PI.

Every number above is a copy of the deck at the time of writing; the deck itself reads the tables at build time.
