---
id: f77
type: finding
schema: 3
title: "Weak on purpose: this qualifies one caveat of the earlier layers findings…"
parent: q18
confidence: weak
why: "Weak on purpose: this qualifies one caveat of the earlier layers findings from a simulation, not a new measurement of the layers, so it must not displace their answer to this question."
evidence: results/eta_space/THEORY.md
created: "2026-11-14T19:19:13"
updated: "2026-11-14T19:19:13"
---

# f77 — Weak on purpose: this qualifies one caveat of the earlier layers findings…

Parent:: [[q18_how_large_is_each_layer_of_the_variance_]]

## Answer

Correction to the depth caveat of f51 and of the layers note (LAYERS_REAL.md): on log1p counts the sampling share falls with depth, but the sampling layer itself is not monotone in depth. In the simulation of results/eta_space/THEORY.md section 4 (2,000,000 bins, 5 seeds, checked against the exact Poisson expectation), the log1p sampling variance rose from 0.0023 at 0.005 reads per bin to 0.205 at 5 reads per bin and fell to 0.032 at 100. The real data sit at about 0.02 to 0.05 reads per bin at half depth, where the sampling layer grows with depth. The layer values and shares of f51 to f53 are unchanged; only the sentence that the sampling layer shrinks with depth was wrong, and it is corrected in place in LAYERS_REAL.md and the paper.
