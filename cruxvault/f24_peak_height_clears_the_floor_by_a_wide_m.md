---
id: f24
type: finding
schema: 3
title: Peak height clears the floor by a wide margin; the shape ranking does not,…
parent: q9
confidence: moderate
why: Peak height clears the floor by a wide margin; the shape ranking does not, and the PI has not judged either.
evidence: results/t133/DIGEST.md, results/t139/DIGEST.md, results/t139/knob_vs_floor.tsv
created: "2026-10-31T12:58:41"
updated: "2026-10-31T12:58:41"
---

# f24 — Peak height clears the floor by a wide margin; the shape ranking does not,…

Parent:: [[q9_which_recorded_processing_covariates_mov]]

## Answer

In peak height, depth is the strongest knob (3.75M: 3.0 log2 in histone marks), then the antibody proxy (1.6–1.9) and read length 36 (1.3–2.0); the control and fragment-extension knobs move −log10 p shape least, and control depth is the weakest. In DNase, duplicate removal off and paired-end raise peaks most. On peaks the base and arm share, the peak-height ranking holds and every knob sits 33–763× above the resampling floor; shape clears the floor in only 3 of 27 rows, so the shape ranking is mostly noise.
