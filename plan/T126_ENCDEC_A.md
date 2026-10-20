# t126 — encoder-decoder rung A (designed with the PI, 2026-10-19)

Exploratory. Readings only: no bar, no ticks, no verdicts (PI 2026-10-19).

## Why

The t118 grid fits trained base↔arm pairs but does not carry over to never-trained arm→arm pairs.
On depth-only pairs the median |log2 error| is 0.04–0.09 on trained pairs and 0.18–0.55 on
never-trained pairs, in every rung and both g versions; the same gap shows for every testable knob
(`cruxvault/results/t118/knob_generalization/report.pdf`, main checkout). The depth-law computation
was audited and is correct. The +1 in log1p pushes rung A's fit to b ≈ 1.6–1.8, a ≈ ln r − 1.7.
Real data has few arm→arm examples, so the fix must come from g's structure, not more pairs.

## Design

A shared MLP φ maps ONE track's covariates c to that track's latent coordinates (α(c), β(c)), β > 0.

    x_i      = log(1 + X_i)
    z_i      = (x_i − α(c)) / β(c)                  encode: source → covariate-free latent
    log μ_i  = α(c') + β(c') · z_i                  decode: latent → target
    X'_i     ~ NB(μ_i, n = exp ψ(c'))               ψ: MLP on the target side only (PI)

Equivalent to rung A with b = β(c')/β(c), a = α(c') − b·α(c). By construction c' = c gives the
identity, and arm→arm composes through the latent. φ is one network applied to both sides.

## Arms (3), each with a no-covariates twin

1. **old A** — t118 rung A as is (g reads [c, c']), retrained under this run's training set.
2. **encdec-A, β learned.**
3. **encdec-A, β ≡ 1** — a pure shift, a = α(c') − α(c).

## Scope

- Counts only. Both g versions (per-track, across-track). 3 seeds.
  3 arms × 2 g × 3 seeds = 18 runs, plus 18 no-covariates twins = 36.
- **Held out of training: the depth 7.5M arm** of every track (PI). Its pairs are scored only.
  The other knobs are tested only on never-trained arm→arm pairs (product identity not ruled out
  for them; within a track every product's covariate vector is unique and can act as an id).
- Everything else as t118 row 1: training pairs base↔arm both directions, NB NLL, chromosome split
  (train all but chr19/21/22, validate chr22, score chr19+chr21), blacklist excluded from scoring,
  g = MLP 2×64.

## Readings

- Depth law (t118's computation) on trained, never-trained and held-out-7.5M pairs, per arm.
- Gain over the no-covariates twin on trained vs never-trained pairs, per knob (as the knob report).
- Learned (a, b) on depth pairs against (ln r, 1).

## Deliverables

Short visual report, then a prezit deck in the style of the t118 grid deck
(`cruxvault/presentations/t118-grid/`).
