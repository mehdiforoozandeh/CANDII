# Symmetric encoder-decoder, phase 1: lab deck

Artifact: https://claude.ai/artifact/62uowv4C2Dtzj4aLyDDqHA (private). Version 1, published 2026-10-25, is built on 222 of 240 runs; it is republished at the same URL when the last 18 Fir runs land.

Source: `index.html` beside this file, built by `build_deck.py` from `deck_src.html`. Its numbers come from `cruxvault/results/t128/`, which is gitignored and per worktree. To rebuild, run these from the worktree root: `bash tools/t128/readings/run_all.sh`, then `python cruxvault/presentations/t128-encdec-sym/build_deck.py`. The design is in `plan/T126_ENCDEC_D2.md`.

The phase is exploratory: the deck gives readings, not verdicts.

## Arc (22 slides)

- **Motivation (slides 1–4).** The t126 encoder-decoder carried nothing over to never-trained pairs, and its two halves were not alike. The PI's redesign: one g and one CNN for both halves, through a learned clean setting C_clean.
- **Method (slides 5–10).**
  - The model: f_e and f_d are the same CNN, with FiLM from g(c, C_clean) and from g(C_clean, c′).
  - C_clean form A (learned freely) against form D (learned, starting at the base, inside a hard box).
  - The losses: L1 cross, L2 round trip, L3 latent agreement, and the VAE.
  - The grid of 8 settings.
  - What is scored, and how each setting is selected on chr22.
  - The references (old D, and old D through the base) and the noise rule.
- **Results (slides 11–19).**
  - The depth law.
  - Never-trained CRPS, pooled and per knob, each with its oracle-scaled / scale-error split.
  - The gain over the no-covariates twin.
  - Collapse: Z stays close to x in every setting.
  - Latent agreement: only the L3 settings bring two tracks' latents closer than their inputs.
  - The swap check against the identity distance.
  - The λ choice.
  - Where C_clean went: it barely moves.
- **Discussion (slides 20–22).**
  - Ruled out: the round trip alone, which stays at the identity.
  - Ruled in: one shared CNN set by one g, with the cross loss.
  - Not shown: a gain over old D, or a latent free of x.
  - Five doubts.
  - The phase-2 options, marked as options for the PI, not recommendations.
