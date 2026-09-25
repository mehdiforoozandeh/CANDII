---
id: h15
type: idea
schema: 2
title: A generator conditioned on the source and target covariates produces the transformation that maps a track onto its counterfactual arms, in both count and −log10 p space, beyond the same model trained with scrambled covariates
parent: q1
status: idea
rule: all
measurement: g(C, C') outputs f, and f(X) predicts X' as NB (n, p) per bin for counts and log-normal (μ, σ) per bin for −log10 p; one model per assay (7, the unit the checks judge); trained on base↔arm pairs of the t112 counterfactual corpus in both directions, validated on chr22, scored on chr19 + chr21 (blacklist removed) and on never-trained arm→arm pairs; D = NB CRPS / log-normal CRPS and Spearman, on all, non-zero and top-1% bins; competitor = the same model trained with C and C' scrambled across pairs; design record plan/T118_COUNTERFACTUAL_F.md
replicates: 7 tracks (1 DNase, 3 narrow, 3 broad marks) x base↔arm pairs in both directions (19 arm products per histone track, 9 for DNase; 246 pairs) x 3 seeds of each of the 7 per-assay f's
neutral_optout: "PI ruling 2026-09-23: no check voids the run; the swap check (C' = C returns X) gates the claim instead of acting as a control"
verdict: 
metric: 
created: "2026-09-23T14:40:11"
updated: "2026-09-25T00:36:55"
null_approved: "2026-09-25T00:36:55"
null_hash: ea3c8984092dbed3
---

# h15 — A generator conditioned on the source and target covariates produces the transformation that maps a track onto its counterfactual arms, in both count and −log10 p space, beyond the same model trained with scrambled covariates

Parent:: [[q1_do_the_recorded_experimental_covariates_]]

## ELI5

If you know exactly how two versions of the same experiment were processed, a model can work out how to turn one into the other — and not just by memorising each case.

## TL;DR

A generator g reads the full covariate vectors C (how X was made) and C' (how the wanted track is made) and outputs a transformation f; f(X) predicts X' as a distribution per bin. Pairs come from [[t112_build_the_counterfactual_arms_for_t|the counterfactual-arms task]]: 7 tracks, each a base plus arms that change one processing knob. The model trains on base ↔ arm and is scored on held-out chromosomes against its own twin trained with scrambled covariates, and on arm → arm pairs it never saw, which only a model that learned how C and C' relate can answer. Settled when f beats the twin beyond seed wobble and gets the arm → arm pairs right.

## Null
Normalization: the covariates are ignored — f does no better than the same model trained with C and C' scrambled across pairs.

## Problem Statement

CANDI's zero-shot claims rest on its covariate conditioning doing real work, and inside CANDI that cannot be isolated. Here each training pair differs in one recorded knob, so the effect of each covariate on signal magnitude and shape is known to exist. The open question is whether the covariate values alone carry enough to reproduce it, as a relation between C and C' rather than a table of per-pair maps (PI ruling 2026-09-25).

## Idea / Hypothesis

A generator conditioned on the source and target covariates produces the transformation that maps a track onto its counterfactual arms, in both count and −log10 p space, beyond the same model trained with scrambled covariates

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
<!-- Bars marked TODO(PI) are not set; CLAUDE.md forbids inventing a gate. -->
- [ ] beatstwin: D_twin − D_f > 2 x f's seed wobble (max pairwise |Δ| of D_f over 3 seeds), per mark class (DNase; narrow H3K27ac/H3K4me3/H3K4me1; broad H3K27me3/H3K36me3/H3K9me3), counts and p separately, CRPS all bins and top 1%, on chr19 + chr21. Further bar on the size of the gain: TODO(PI) — the earlier gap-closed ≥ 0.5 waits for a redefined oracle
      fails-if:: the covariates add nothing a scrambled-covariate twin of equal capacity cannot already do
      discriminates:: true
- [ ] lawtest: on never-trained arm → arm pairs within each track, scored on chr19 + chr21 and reported by knob combination, f beats the scrambled twin by more than 2 x seed wobble; bar: TODO(PI). For depth → depth pairs the predicted count scale must follow the depth ratio (tolerance TODO(PI))
      fails-if:: f keeps one map per trained (C, C') pair and has no answer for a combination it never saw
- [ ] shufflecollapse: at scoring, C' replaced by the C' of another arm of the same track whose target differs (drawn separately for counts and p); f's advantage over the twin must vanish; bar: TODO(PI) — previously gap-closed ≤ 0.1 against the void oracle
      fails-if:: f does not use C': its output does not change when told the wrong target covariates
- [ ] swapreturn: at scoring, C' set equal to C; f must return X; bar: TODO(PI) — previously gap-closed toward X ≥ 0.5 against the void oracle
      fails-if:: told the target is the source itself, f still transforms X, so its output is not steered by C'

## Planned Intervention

Full design, rulings and run record: `plan/T118_COUNTERFACTUAL_F.md`.

- Training pairs base ↔ arm, both directions; no arm → arm and no identity pairs in training. Train on all chromosomes but chr19, chr21, chr22; chr22 validates; chr19 + chr21 score. chrY, chrM dropped; blacklist out of scoring only.
- C and C' are the full knob vector plus the assay, one standardised concatenation; f never sees C' − C.
- Three versions: one per assay (judged), one per arm and one across all tracks (reported).
- p space is log-normal, measured 2026-09-24 (skew of x 2.9–17.6 vs log x −0.56–0.51; SD between pseudoreplicate halves ∝ level, slope 0.81–1.32).
- References, never pass/fail: noSolution (X' = X) and per-pair QuantileMatching, which reads no covariates. One-value rungs are Poisson (counts) and log-normal with one σ per arm (p), floored at 1e-3.
- Reported only: Spearman, CRPS on non-zero bins, the CRPS split, per-arm results, a point-output f.
- The pseudoreplicate oracle is not a floor (on Spearman it beats both references in only 1 of 142 count pairs and 18 of 242 p pairs), because a base and its arm share reads and the halves share none. It must be redefined before any gap-closed bar applies.
- Five claim-directed checks became four; under `all` at 80% power each that is 41% joint power, and bars may not be loosened to compensate.

## Run Links

_(none yet)_

## Artifacts

<!-- what the run produced. Keep files under results/h15/ and link at least the report:
     - [Report](results/h15/report.md)   - results/h15/curve.png -->
_(none yet)_

## Findings

_(written by the PI/agent when the case is closed)_
