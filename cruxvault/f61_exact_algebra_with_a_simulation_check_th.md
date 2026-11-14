---
id: f61
type: finding
schema: 3
title: Exact algebra with a simulation check that recovers the true layers;…
parent: q33
confidence: moderate
why: Exact algebra with a simulation check that recovers the true layers; moderate because the PI has not reviewed the note, the real-data and model parts are still being worked, and the additivity assumes multiplicative layers.
evidence: results/eta_space/THEORY.md, results/eta_space/eta_sim.json, results/eta_space/LITERATURE.md
created: "2026-11-14T16:37:01"
updated: "2026-11-14T16:37:01"
---

# f61 — Exact algebra with a simulation check that recovers the true layers;…

Parent:: [[q33_can_the_negative_binomial_be_written_wit]]

## Answer

Theory: yes for the variance decomposition. With Y | eta ~ Poisson(D r), r = exp(eta), the law of total variance gives exactly Var(Y) = D E[r] + D^2 Var(r): read sampling is the first term, linear in depth D, and every layer above sampling is D^2 times a depth-free Var(r). The layers are exactly additive in log(1 + theta), theta = squared coefficient of variation (CV) of the rate, for independent multiplicative layers; this equals Var(eta) for lognormal layers only (gamma layers differ by 5% at alpha 0.05, 59% at 0.5). The NB dispersion of a model given inputs X equals Var(r | X) / E[r | X]^2, the squared CV of the rate the inputs do not fix, so it is depth-free and in the same unit as the eta-space layers. Factorial and cross moments (E[Y(Y-1)]/D^2, E[Y Y']/(D D')) estimate the layers without a log, a pseudocount or thinning to a common depth. Simulation (2,000,000 bins, 5 seeds, truth replicate 0.05, cell type 0.5): the log1p sampling share fell from 0.979 to 0.057 as the mean count rose from 0.005 to 100 per bin, while the eta replicate layer stayed 0.049 to 0.051 and the cell-type layer 0.498 to 0.499 from mean count 0.05 to 100 (noise at 0.005). It does not help the per-bin noise: the estimates are weighted toward peak bins (weights r^2), so a layer whose eta variance differs between peaks and background is reported at its peak value unless stratified; and a share of variance needs a stated reference depth, because sampling is not an eta variance.
