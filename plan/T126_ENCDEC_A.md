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

## Rung D (PI scope extension, 2026-10-19/20)

Encoder-decoder rung D. PI decisions:

- **Encoder:** today's D trunk split in two. 2 dilated conv layers (dilations 1, 2; 32 channels;
  kernel 5), each FiLM-modulated by g_enc(c), then GELU. Input x = log1p(X); output latent Z,
  shape [B, 32, L].
- **Decoder:** 2 layers (dilations 4, 8), each FiLM-modulated by g_dec(c′), then GELU, then today's
  1×1 head giving (loc, disp). NO skip from x to the output: the source cannot bypass the latent.
  Total context stays 61 bins (same size and context as today's D). Zero-init / identity
  conventions from `form_d.py` where they still apply; any changed init is reported.
- **g_enc, g_dec:** separate MLPs, each 2×64 like t118's g, each reads one side's covariates only.
- **Loss** per base↔arm pair (X, c) → (X′, c′), Z = enc(X, c), Z′ = enc(X′, c′):
  NLL(X′ | dec(Z, c′)) + NLL(X | dec(Z′, c)) + NLL(X | dec(Z, c)) + NLL(X′ | dec(Z′, c′))
  + λ·mse(Z, Z′).
- **Arms**, each with a no-covariates twin, both g versions, 3 seeds, counts only, depth 7.5M held
  out of training: (1) old D retrained (t118 rung D, hold-out applied); (2) encdec-D, λ = 0;
  (3) encdec-D, λ > 0.
- **λ grid (PI, 2026-10-20): {0.1, 1, 10}.** All three are trained (both g versions, 3 seeds, each
  with a no-covariates twin) and all three are scored on chr19 + chr21, so the readings show every λ.
- **λ selection rule (PI):** one λ per g version by the t118 run-selection convention on chr22 —
  the validation NLL on the trained cross pairs (the run's best `val_nll`), mean over seeds; lowest
  wins.
- **Unit tests:** the self path dec(c) ∘ enc(c) is the code path the self terms train; the hold-out
  removes every 7.5M pair from training; an arm→arm prediction equals dec(enc(X1, c1), c2).

## PI confirmations (2026-10-20)

- Run count: 7 per-track g's + 1 across-track g, as t118 row 1 — 144 rung-A runs, 240 rung-D runs.
- Encdec-D starts as a constant map (no skip from x, so no identity init); old D starts at the
  identity. Accepted for this exploratory run; named as a doubt in the readings.
