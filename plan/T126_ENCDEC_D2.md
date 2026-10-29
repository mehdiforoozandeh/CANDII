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

## Phase 2: PI decisions, 2026-10-25

The PI picked three paths from the phase-1 deck:
1. **Fixed real-base middle:** the encoder-decoder with C_clean frozen at the base's covariates.
2. **More steps:** the cap goes from 4000 to 12,000 steps, and the best step is still chosen on chr22.
3. **Anchored middle:** the encoder output is an NB prediction of the observed base track X_base,
   trained with NLL(X_base).

Deferred to a later phase: the scale questions, namely a bigger g, a bigger CNN, and other
activations (the CNN now uses GELU). The PI ruled that these wait until phase 2 defines how g and f
work together.

Run grid (PI, 2026-10-25): 4 arms × 8 g units × 3 seeds = 96 runs, all at 12,000 steps, no twins.
- a. Default: learnable C_clean (form A), L1 + L2.
- b. Fixed base: C_clean frozen at the base's covariates, L1 + L2.
- c. Anchored: C_clean fixed at the base; L1 + L2 + L_anchor at equal weights, where
  L_anchor = NLL(X_base | f_e(x_arm; g(c_arm, c_base))) with NB output. The decoder reads the encoder's
  predicted NB mean, in training as at test time.
- d. Old D, retrained at 12,000 steps, as the reference; via-base is scored from these maps.
Form D (box) is dropped, because its C_clean barely moves from the base.

Metrics (PI, 2026-10-25). Every CRPS has the oracle_scaled / scale_error split, and a gap counts only
above 2x the larger seed spread.
- Tier 1, the question: (1) never-trained arm->arm CRPS, pooled and per knob, against old D and
  via-base; (2) generalisation gap = never-trained CRPS - trained CRPS, same arm.
- Tier 2, no cost elsewhere: (3) trained-pair CRPS; (4) held-out 7.5M depth CRPS; (5) depth-law error
  on never-trained depth pairs (sanity check only).
- Tier 3, diagnostics: (6) hop-1 CRPS, the prediction of X_base against the real base (arms b, c, d;
  for b, the decoder's base prediction from the middle); (7) middle consistency, |Z - Z'| against
  |x - x'| for two arms of one track; (8) swap check against identity distance; (9) the share of runs
  whose best step is in the last 10% of 12,000 steps.
- The deck ranks the arms by (1) and shows (2) beside it, as a reading, not a gate.
- Final report: each metric is defined, with its best and worst cases explained, and then its
  results. The discussion and conclusion must draw on every metric and say what each one tells us.

## Phase 3: PI decisions, 2026-10-27

The PI read phase 2's winner as **the two-step via the base**: old D (rung D), trained on base ↔ arm
pairs, with each never-trained arm → arm pair predicted as arm → base → arm. It is the only model
that beat old D direct on never-trained pairs outside noise (phase 2, one g per track: 0.545
against 0.662). Phase 3 asks the deferred scale question on that model alone: does a bigger g, or a
bigger f, lower the never-trained CRPS of the two-step via the base?

Run grid (PI, 2026-10-27): 2 arms × 8 g units × 3 seeds = 48 runs. Each arm grows one part only.
- a. **Bigger g:** g = MLP 40 -> 256 -> 256 -> n_theta, ReLU (hidden width 64 -> 256; about 23 k ->
  about 142 k parameters). f as phase 2.
- b. **Bigger f:** f = rung D's trunk at 64 channels (was 32), same 4 layers, kernel 5, dilations
  1, 2, 4, 8, GELU; the receptive field stays 61 bins. About 16 k -> about 62 k parameters. FiLM
  carries one gamma and one beta per channel, so n_theta goes 256 -> 512 and only g's last layer
  grows; g's hidden width stays 64.
- No combined arm, and one size per axis (PI). A combined arm or a second size follows only if an
  axis helps.
- Training: rung D's loss and recipe as phase 2's arm d; cap 15,000 steps; the existing early stop
  (chr22 NLL every 250 steps, patience 4); the learning rate unchanged (a caveat, not tuned). The
  depth 7.5M arm of every track stays held out.
- **Reference:** phase 2's arm d (old D at 12,000 steps, the small model), read through the base. It
  is not rerun: no phase-2 old-D run reached its cap (best step median about 2,750, stop about 3,750),
  so the 12k and 15k caps act the same unless a bigger model trains longer. Metric 9 checks that.

Metrics. The phase-2 rules hold: every CRPS has the oracle_scaled / scale_error split, and a gap
counts only above 2x the larger seed spread of the two cells.
- Tier 1, the question: (1) never-trained arm -> arm CRPS **through the base**, pooled and per knob,
  each arm against the reference; also the direct never-trained CRPS of the same runs.
- Tier 2, no cost elsewhere: (2) generalisation gap; (3) trained-pair CRPS; (4) held-out 7.5M depth
  CRPS; (5) depth-law error on never-trained depth pairs (sanity check only).
- Tier 3: (9) the share of runs whose best step is in the last 10% of 15,000 steps; the best-step
  median against the reference's.
- The deck ranks bigger g, bigger f and the reference by (1), one g per track and one g across
  tracks, as a reading, not a gate.

Design check (crux-design, 2026-10-27: sound, with the fixes below; nothing in the PI's grid changes).
- *Pre-registered reading.* One g per track: a gain counts above 2x max(0.0077, the arm's own seed
  spread), at least about 0.015 (about 3% of 0.545). One g across tracks, pooled: the reference seed
  spread is 0.064 (seeds 0.603, 0.639, 0.667), driven by the dedup, extsize and pe knobs, so a gain
  counts only above about 0.128 (about 20%). This cell is reported but is read per knob. "Within
  noise" means "no gain above the stated threshold", never "no effect".
- *Optimisation failure.* If an arm's never-trained CRPS does not improve and its trained-pair CRPS
  also gets worse, read it as an optimisation failure (learning rate), not as "capacity does not help".
- *Arm b* is "bigger f, with the g head it needs": its g grows 23 k -> 40 k (output layer only), against
  142 k in arm a. Seeds do not pair across arms (a bigger model draws more initial weights).
- *Checks.* The via-base never-trained pair count equals the reference's (2,094 per seed). The via-base
  readout (tools/t128/viabase.py) rebuilds g and f at each run's recorded hidden width and channel
  count, not at the module constants. A run whose best step is above 12,000 is flagged as
  step-confounded and listed apart.

## Phase 3b: more seeds for the bigger f (PI decisions, 2026-10-29)

Phase 3's readings (deck https://claude.ai/artifact/5JfcGnVRdH8NESQeSWp9tp): through the base, one g
per track, bigger f gave 0.530 against the small reference's 0.545, a gain of 0.0146 just under the
pre-registered bar of 0.015; bigger g gave 0.534. The PI chose option B and ruled on the rest:
- (i) Option B: three more seeds per per-track g for the bigger f AND for the small reference.
- (ii) The bigger f's gain is not closed as "within noise"; the seeds decide it.
- (iii) Bigger g's high direct seeds (0.627, 0.82, 0.77 per track) are a caveat, not followed up.
- (iv) One g across tracks is dropped from the gain test.
- (v) The f size gets its own hypothesis under the depth-law question (h27).

Run grid: 2 arms × 7 per-track g × seeds 3, 4, 5 = 42 new runs.
- a. **Bigger f** (`Dbigf`): exactly phase 3's arm b (g hidden 64, f 64 channels, n_theta 512,
  15,000-step cap, early stop eval 250 / patience 4, lr unchanged, depth 7.5M held out).
- b. **Small reference** (`D`): exactly phase 2's arm d (g hidden 64, f 32 channels, n_theta 256,
  12,000-step cap, same early stop), so its six seeds pool with the three phase-2 seeds under one
  recipe. Run on the phase-3 code: its default sizes are bit-identical to phase 2 (pinned by
  tests/test_t128p3_sizes.py). The cap difference does not bind: no run of either arm reached a best
  step above 6,000 (phase 3 metric 9).
- Both arms are read directly and through the base, as phase 3. Seeds 0–2 are not rerun.

Reading (to be fixed by the design check before any compute): metric 1, never-trained arm → arm
CRPS through the base, one g per track, pooled over all six seeds per arm, with its split; plus the
same on the three new seeds alone, as a replication. Metrics 3, 4, 5 and 9 as phase 3, per-track only.

Reading (crux-design 2026-10-29; PI approved the same day). Per-seed value = the mean over the 7
per-track g; arms unpaired. Two bars, reported apart:
- Real: gap₆ > 4·SE, SE = √(sd_f²/6 + sd_ref²/6) (Welch; about 0.008 expected).
- Size: max(2·max(0.0077, 0.669·R₆,f, 0.669·R₆,ref), 0.015); R₆ = 6-seed range, 0.669 converts it
  to a 3-seed equivalent.
- Labels: gain (both pass) / gain real, below the size bar / within seed noise / loss (mirror). The
  gap is reported with its 95% interval. The range rule alone has about 9% power at 6 seeds and is
  not used as the gate.
- Replication, seeds 3–5 alone: gap₃ > 2·SE₃, one-sided.
- Controls: (a) kit drift — the reference's seeds 3–5 against its seeds 0–2, |Δ| ≤ 4·SE₃, else read
  the new seeds only; (b) readout — the phase-3 viabase.py rescoring one phase-2 D run reproduces its
  recorded via-base CRPS.
- The CRPS result is a reading under h27 only (PI); no separate hypothesis.
