# Symmetric encoder-decoder, phase 2: lab deck

Artifact: https://claude.ai/artifact/PGZNDxcAgy42NFuuVG5c1m (private). Version 1, published 2026-10-26, is built on all 96 runs and the 24 via-base passes.

Source: `index.html` beside this file, built by `build_deck.py` from `deck_src.html`. Its numbers come from `cruxvault/results/t128p2/`, which is gitignored; the build reads the main checkout's copy by default (override with `T128P2_RESULTS=<dir>`). To rebuild, run these from the worktree root: `bash tools/t128p2/readings/run_all.sh`, then `python cruxvault/presentations/t128p2-encdec/build_deck.py`. The design is in `plan/T126_ENCDEC_D2.md` § "Phase 2: PI decisions, 2026-10-25"; the run log is in `plan/T118_STATUS.md` § "t128 phase 2".

The phase is exploratory: the deck gives readings, not verdicts.

## Arc (22 slides)

- **Motivation (slides 1–3).**
  - Phase 1: the learned middle sat within noise of old D on never-trained pairs, and old D through the base was lower.
  - The PI's three paths, drawn as routes: a middle fixed at the base's setting, a middle anchored to the real base track, and three times the steps.
- **Method (slides 4–6).**
  - The grid of arms: learned, fixed and anchored middles, old D retrained at 12,000 steps, and old D through the base.
  - The pairs that are scored, selection on chromosome 22, and scoring on chromosomes 19 and 21.
  - The CRPS split (shape and scale) and the noise rule.
- **Results (slides 7–18).** Each metric slide gives the definition, the best case and the worst case, and then the figure with its noise labels.
  - Metric 1, never-trained CRPS: per track, old D through the base is lowest outside noise. The three middles are within noise of old D. Old D wins on same-knob pairs, and the middles are lower on two-knob pairs within noise. Across tracks, every gap is within noise.
  - Metric 2, the generalisation gap: it is negative for every arm. The cause is the make-up of the two pools, and the deck shows it. Knobs with one arm (deduplication, paired end) are only in the trained pool. Half of the trained depth pairs target the deep base. Matched by knob, the gap is near zero for the middles.
  - Metrics 3 and 4, trained pairs and the held-out 7.5M depth: the middles cost nothing.
  - Metric 5, the depth law (a sanity check): it holds on trained pairs. Only the chain through the base stays close on never-trained pairs.
  - Metric 6, hop 1: the fixed and anchored middles predict the base as well as old D's arm → base map.
  - Metric 7, middle consistency: only the anchored middle brings two arms' middles closer than their inputs.
  - Metric 8, the swap check: every middle passes it far from the identity. Old D's swap value is several times larger.
  - Metric 9, steps: 1 of 96 best steps is in the last 10 %. Early stopping ends the middles near 6000 steps and old D near 3500. More steps did not move metric 1.
  - The ranking by metric 1, with metric 2 beside it, as a reading.
- **Discussion (slides 19–22).**
  - One line per metric.
  - The conclusion on how to formulate g and f. With one g per track, the closer the middle is to the real base track, the lower the never-trained CRPS. Only the chain through the base clears the noise, and the gap left sits in the second hop.
  - The deferred scale questions (a bigger g, a bigger CNN, another activation than GELU), as options.
  - Five open points for the PI.
