# The g → f transformation grid: lab deck

Artifact: https://claude.ai/artifact/1mFz6C5jLaG1Ctc8JN572C (first published 2026-10-02, commit 12380d7; private until shared)

Source: `index.html` beside this file, built by `build_deck.py` from `deck_src.html` and `snippet.json` (a 10 kb real H3K27ac snippet, chr19:12,732,350–12,742,350, read from the t112 products on Nibi). Every number comes from `cruxvault/results/row2/` (`results_summary.tsv`, `checks_*.json`, `grid.md`), which is gitignored. Later edits: 3032e5a (PI review of slides 16, 19, 22) and 09d3dcb (a slide that compares one g per track with one g across tracks; 27 slides now). The PI shared the deck with the lab (`plan/T118_STATUS.md`, 2026-10-08).

The study is exploratory, so the deck gives readings and no verdicts.

## Arc

- **Motivation and background.** Recorded covariates C (how a track was made) and C′ (how the wanted track is made) should tell a model how to turn one track into the other. The data are the counterfactual arms: 7 tracks, each a base plus arms that change one processing knob.
- **Method.** A small g reads (C, C′) and outputs the parameters of f; f(X) predicts X′ as a distribution per bin. Row 1: g reads (C, C′). Row 2: g also reads the bin's own value. Designs A (affine), B (12-knot curve), C (kernel then curve), D (FiLM CNN). Twins: no-covariates, labels-as-ids, and (row 2) shuffled-bin.
- **Results.** Reading the bin's value fixes design A in counts; B–D change little. Row 2 has more exploding −log10 p pairs. The covariates beat the no-covariates twin almost everywhere. The depth law is unmet.
- **Discussion.** The p-space explosion needs a principled fix, the depth-law miss needs a cause, and design D's explosion needs a diagnosis.
