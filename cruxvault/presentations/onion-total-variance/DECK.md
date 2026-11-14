# Peeling the Variance: lab deck on the law of total variance

Artifact: https://claude.ai/artifact/H8ShBq8tJ1eMzwpjNpdU87 (published 2026-11-13, private)

Source: `index.html` beside this file, built by `build_deck.py` from `deck_src.html` (prezit paper arc). PI request 2026-11-13: "a /prezit for how we use the law of total variances to formulate our variance decomposition idea". No vault question covers the variance decomposition, so the deck has no anchor node; the study slug is `onion-total-variance`. The subject is the formal note [[onion total variance]].

Scope: the formulation and its checks. No real-data layer value appears: the variance-layer job on dataset version 1 finished on the cluster, but its outputs are not collected or checked. No model result appears: no model is trained.

## Where the numbers come from

- `sim_figures.json`: SIMULATED, written by `simulate_figures.py` beside this file. It calls the audit simulator `tools/onion_eval/total_variance.py` (unchanged) and `layers.py` on simulated tables, seed 0. Slides 3, 4, 5, 7 and 13 (telescoping card).
- `results/t233/AUDIT.md` and `results/t234/AUDIT_TVg_seed0.md`: the audit of the scorer against the identity, before and after the fixes (simulated). Slides 13, 14, 15.
- `INSTRUCTIONS/peeling_the_onion/onion total variance.md`: explained fraction in simulation (regex). Slide 16.
- `results/t183/DATASET_EIC_V1.md`, `results/t183/rung_pairs.tsv` (pair counts per rung only), `results/t185/SUBMITTED.md`, `INSTRUCTIONS/peeling_the_onion/long-horizon notepad.md` (second-experiment count, regex). Slide 18.
- `results/` is gitignored: the build reads this worktree's copies. The deck-sources manifest in `index.html` holds the sha256 of every file read.

## Rebuild

From inside the vault (`cruxvault/`):

```bash
/Users/mforooz/miniforge3/envs/candii/bin/python presentations/onion-total-variance/simulate_figures.py
/Users/mforooz/miniforge3/envs/candii/bin/python presentations/onion-total-variance/build_deck.py
python3 ~/.claude/skills/crux/scaffold/crux.py deck --verify presentations/onion-total-variance/index.html --strict
python3 ~/.claude/skills/crux/scaffold/crux.py deck --lint presentations/onion-total-variance/index.html
```

## Arc (18 slides)

Headlines drafted by Claude are marked.

1. Title: ==Where does the variance of an epigenomic signal come from? One identity, the law of total variance, splits it into layers.==
2. Motivation: ==A count in one bin varies for nested reasons, from biology down to read sampling== (onion figure; the three goals).
3. Method: ==Total variance = variance of the group means + mean variance inside a group== (simulated two-level example).
4. Method: ==Applied once per step of a chain of nested factors, the law splits the variance into eight terms== (simulated bar).
5. Method: ==Read sampling grows with depth, every other layer with depth squared, so pairs are compared at one depth==.
6. Method: ==Each rung of the replicate ladder is a pair of measurements that shares the chain down to one level== (grid).
7. Method: ==Half the mean squared difference of a pair measures every term below the level the pair shares== (simulated stair).
8. Method: ==A layer is the difference of two consecutive pair values; donor and experiment are siblings, not a chain==.
9. Method: ==Biology and lab work set the rate of a bin; covariates and read sampling turn the rate into a count== (graphical model).
10. Method: ==A calibrated predicted variance lies between the sampling term and the pair value of what the model knows==.
11. Method: ==The explained fraction of a layer is the share of that layer that the model's prediction difference captures==.
12. Method: ==The scorer computes every layer per cell at one depth, and every interval over cells==.
13. Results (a): ==On simulated counts the identity holds, and the scorer's layers add up exactly==.
14. Results (b): the audit flagged 10 of 24 checks; after the fixes 2 of 31 remain (headline computed at render).
15. Results (c): ==A perfectly calibrated cell-type model scores far above one at the halves, and about one only at the rung that shares what it knows==.
16. Results (d): ==The explained fraction credits each simulated model with the layers it knows, and nothing else==.
17. Discussion: ==Each estimate is unbiased only under six assumptions; crossed factors and the log scale break the simple story==.
18. Discussion: three of the seven rungs have no pairs yet on dataset version 1; the real-data layers are computed and await the collect (headline computed at render).
