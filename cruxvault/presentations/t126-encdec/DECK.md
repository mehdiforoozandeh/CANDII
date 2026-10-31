# Encoder-decoder g: lab deck

Artifact: https://claude.ai/artifact/KP7pipaRbNbEStUaJR7jrZ (published 2026-10-21, private)

Source: `index.html` beside this file, built by `build_deck.py` from `deck_src.html`. Every number comes from `cruxvault/results/t126/`, which is gitignored and per-worktree. The visual readings report is `cruxvault/results/t126/report.pdf`. The design is in `plan/T126_ENCDEC_A.md`.

The study is exploratory, so the deck gives readings and no verdicts.

## Arc (16 slides)

- **Motivation.** In the earlier grid, every design fits the trained base ↔ arm pairs but misses the never-trained arm → arm pairs, for every knob. The diagnosis checked four suspects:
  - the depth-law check itself, which was ruled out;
  - the +1 in log(1 + count), which acts indirectly (old A learns b ≈ 1.7);
  - g extrapolating badly, which mostly holds;
  - the NB loss, which is still open.
- **Method.**
  - A shared φ(c) gives each track its own coordinates, so encoding and then decoding composes any arm → arm pair.
  - Rung A uses an exact inverse, with two arms: β learned and β = 1.
  - Rung D uses a learned decoder: a 2 + 2 layer CNN with a four-term loss and λ ∈ {0.1, 1, 10}.
  - The baselines are retrained with depth 7.5M held out of training. Every arm has a no-covariates twin.
  - The runs cover 7 per-track g's plus 1 across-track g, 3 seeds, counts only: 144 rung-A runs and 240 rung-D runs.
- **Results.**
  - (a) No encoder-decoder arm closes the never-trained depth gap. The held-out 7.5M depth is easy for every arm except encoder-decoder A.
  - (b) Encoder-decoder D gains more on trained pairs, but on the never-trained pairs of read length, MAPQ and control fraction it falls below its no-covariates twin.
  - (c) Encoder-decoder D keeps old D's fit on trained pairs. Encoder-decoder A loses fit, mostly as scale error.
  - (d) The exact inverse forces b(c → c′) · b(c′ → c) = 1. Old A needs a slope above 1 in both directions, which is its compensation for the +1.
- **Discussion.** Composition built into the structure is not enough on its own, in either form tried here. The doubts:
  - encoder-decoder D starts as a constant map, while old D starts at the identity;
  - 66 of 96 encoder-decoder D runs hit the 4000-step cap, so they may be under-trained;
  - the held-out 7.5M depth tests interpolation only;
  - for knobs other than depth, the model may be using each product's covariate vector as an id;
  - the differences between λ values are within the seed spread.
