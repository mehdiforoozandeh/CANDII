# Deck: Poisson mean versus NB (loss head-to-head)

Published 2026-10-25: https://claude.ai/artifact/3QpVJzhgeZBwfSXu19y5Yj (private; share it from the page's Share menu).
Source: `index.html` in this folder, built by `build_deck.py` from `results/t127/` and `results/h19/depth_pairs/`.
Task: [[t127_head_to_head_poisson_nll_mean_nb_nl]]. Hypotheses: [[h21_fitting_the_mean_by_poisson_nll_and_the_]], [[h18_the_1_pseudocount_in_x_log_1_counts_make]], under [[q7_why_do_the_covariate_conditioned_g_f_des]].

## Arc (11 slides)

1. Title: does changing the training loss fix the depth law? 126 runs, exploratory.
2. Motivation: the model follows the depth law on trained base ↔ arm pairs and misses it on never-trained arm → arm pairs, in all 8 grid designs.
3. Motivation: the NB loss balances totals with per-bin weights n/(n + μ), so the totals can drift; the Poisson loss balances them exactly.
4. Method: the full NB loss against a Poisson loss on the mean plus an NB loss on the dispersion, which reads a frozen copy of μ.
5. Method: designs A, A0 (no +1) and D × the two losses × 7 tracks × 3 seeds, one g per track, counts only.
6. Finding 1: in design A the new loss removes the scale error (CRPS scale_error 0.253 → 0.005, wobble 0.013) and changes the oracle-scaled part little (0.581 → 0.531, wobble 0.012).
7. Finding 2: the DNase totals of design A (13.2–14.7× the observed total with the NB loss) return to 0.80–0.93×.
8. Finding 3: the depth law on never-trained pairs is not fixed: A 0.33 → 0.29, A0 0.46 → 0.38, D 0.27 → 0.28 median |log2 error|; the data obey the law to 0.004 log2.
9. Finding 4: in A0 and D the CRPS gain is about one seed wobble.
10. Finding 5: NB likelihood and PIT calibration do not change beyond the seed wobble.
11. Discussion (PI reading 2026-10-20): the loss is not the cause; the failure is in how g generalises. Next: a composable g, and training on arm → arm pairs.
