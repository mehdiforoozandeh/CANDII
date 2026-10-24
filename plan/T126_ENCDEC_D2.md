# t126 phase 1 — symmetric encoder-decoder rung D (v2), designed with the PI 2026-10-21

Exploratory: readings only, no bar, no verdicts. It replaces the asymmetric 2+2 split of
`plan/T126_ENCDEC_A.md` § Rung D. The PI ruled that split wrong: the encoder and decoder must be
symmetric, and one g must set both.

## Model

- **One g**, as in t118 (MLP 2×64, reads [c_src, c_tgt]). It outputs **FiLM only** (γ, β per layer
  and channel, as old D). It does not output the CNN weights.
- **One CNN f**, old rung D exactly (`tools/t118/ladder/fforms/form_d.py`: 4 dilated conv layers, 32
  channels, kernel 5, dilations 1, 2, 4, 8, GELU, 1×1 head, skip loc = x_centre + head[0]).
  The encoder and the decoder are **the same f with the same weights**. Only the FiLM differs.
- **C_clean** is a learned "clean middle" covariate vector. The assay entries are copied from the
  source c, and only the processing knobs are learned.
- **Encoder f_e:** input x = log(1 + X); FiLM θ_e = g(c, C_clean); the head gives a Gaussian
  (μ_z, log σ_z) per bin.
- **Decoder f_d:** input Z; FiLM θ_d = g(C_clean, c′); the head gives NB (loc, disp) for X′.
- **Z:** 1 channel per bin, at the same 25 bp resolution (no pooling). Deterministic arms use
  Z = μ_z. VAE arms use Z = μ_z + σ_z·ε, with KL to N(0, 1) and a KL weight warmed up linearly
  from 0 to 1 over the first 1000 of 4000 steps.
- X and X′ are in NB space; Z is in Gaussian space. Two f's in series need a halo of 60 bins of
  x on each side of every training window.
- One CNN per g (its scope follows g's, as t118).

## C_clean forms

- **A (default):** learned freely; one C_clean per track (default scope).
- **D:** learned and initialised at the base's covariates, inside a hard box: each knob is a
  sigmoid onto the [min, max] seen in training (no penalty weight).

## Losses (per trained base↔arm pair (X, c) → (X′, c′); Z = f_e(X; c), Z′ = f_e(X′; c′))

- **L1 cross:** NLL(X′ | f_d(Z; c′)) + NLL(X | f_d(Z′; c)).
- **L2 round trip:** NLL(X | f_d(Z; c)) + NLL(X′ | f_d(Z′; c′)).
- **L3 latent agreement:** λ · mse(Z, Z′), with λ ∈ {0.1, 1, 10}, one λ chosen per g version on
  chr22 (lowest mean-over-seeds best validation NLL), all three scored.

## Phase 1 settings (8)

1. default: form A, L1 + L2, deterministic
2. form D, L1 + L2, deterministic
3. form A, L2 alone, deterministic
4. form A, L2 alone, VAE
5–7. form A, L1 + L2 + L3, deterministic, λ = 0.1, 1, 10
8. form A, L1 + L2, VAE

Each runs on the 7 per-track g's and the across-track g, with 3 seeds: 8 × 8 × 3 = 192 runs.
No-covariates twins are trained only for settings 1 and 3: 48 runs. **Total 240.**
Counts only. The depth 7.5M arm is held out of training (as t126). Training settings are as in t118
row 1 (lr 1e-3, batch 32, 4000 steps, best step on chr22). Scoring is on chr19 + chr21, with the
blacklist excluded.

## References (no new training)

- Old D from t126 (`/project/def-maxwl/mforooz/t126/D/`, arm D, 7.5M held out) and its twins.
- **Via the base:** old D's two trained maps chained, arm → base then base → arm, scored on
  never-trained pairs.

## Readings

1. Depth-law error (t118's computation): median |log2 error| on trained, held-out 7.5M and
   never-trained depth pairs, per arm, both g versions.
2. Never-trained CRPS per knob, against old D and the default, always with the oracle_scaled /
   scale_error split.
3. Gain over the no-covariates twin, trained versus never-trained, per knob, for settings 1 and 3.
4. Collapse: median |Z − x| and corr(Z, x) per arm, on the scoring chromosomes.
5. Latent agreement: for two arms of the same track, median |Z − Z′| against |x − x′|.
6. Swap check (c′ = c returns X), plotted **against** the identity distance per arm. The target is
   a model that passes the swap check while staying far from the identity.
- A gap counts only if it exceeds 2 × the larger seed spread (the largest pairwise |Δ| over the
  3 seeds). Gaps below that are shown and labelled "within noise".
- Selection uses chr22 only.

## Output

A prezit-style deck after phase 1 (PI: decks, not PDF reports), in the style of
`cruxvault/presentations/t126-encdec/`. The PI picks phase 2 from that deck. Phase 2 candidates
are the shared-scope C_clean, the full grid, and twins for the other arms.

## Doubts to carry

- Under form A, C_clean is not identifiable, so its learned value may mean nothing.
- With 1 channel and a skip that starts at the identity, the deterministic L2-only arm may stay
  at the identity. The VAE arm tests this.
