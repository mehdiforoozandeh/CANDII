---
id: f25
type: finding
schema: 3
title: The 15M floor passes an exact self-check and real depth-arm halves match it…
parent: q10
confidence: strong
why: "The 15M floor passes an exact self-check and real depth-arm halves match it (ratio 1.002); the 30M values are extrapolated on a concave curve. Settled by the PI's rule: no further work is planned on it (its tasks are done)."
evidence: results/t142/DIGEST.md, results/t138/DIGEST.md
created: "2026-10-31T12:58:41"
updated: "2026-10-31T12:58:41"
---

# f25 — The 15M floor passes an exact self-check and real depth-arm halves match it…

Parent:: [[q10_how_much_does_resampling_alone_move_a_tr]]

## Answer

Resampling alone moves 1 kb shape a lot and peak height almost not at all: the shape floor at 30M is about 0.43–0.53 in histone marks and 0.11 in DNase, while the peak-height floor is at most 0.02 log2. Signal level sets the noise, not region: level explains 0.18–0.38 of per-bin disagreement, and GC, mappability, repeats and peak context add at most 0.0003 R² at 15M. The PI's prior that noisy regions are non-uniform did not hold. For CANDI, a level-dependent spread such as the NB variance describes the floor.
