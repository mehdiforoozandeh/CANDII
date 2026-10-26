---
id: h26
type: idea
schema: 2
title: A trained CANDI checkpoint, told a counterfactual arm's read length or run type in place of the base's, moves its predicted counts no closer to the arm's real track than when told the base's own covariates
parent: q8
status: idea
rule: all
measurement: "Covariate-swap probe: per track x arm, change in predicted log2 mean (told arm vs told base covariates) against the real arm/base log2 ratio, and NB NLL of the arm track under each prompt; CPU inference on a saved checkpoint, one chromosome"
replicates: 7 tracks x (2 read-length + 1 run-type arms); 2 seeds of checkpoint where they exist
verdict: 
metric: 
created: "2026-10-25T13:49:00"
updated: "2026-10-26T00:53:49"
null_approved: "2026-10-26T00:53:49"
null_hash: e7671d9e2bb33ffb
---

# h26 — A trained CANDI checkpoint, told a counterfactual arm's read length or run type in place of the base's, moves its predicted counts no closer to the arm's real track than when told the base's own covariates

Parent:: [[q8_does_candi_s_training_which_never_pairs_]]

## ELI5

If we tell a trained CANDI model that a track was read with shorter reads or as paired reads, we expect its prediction not to change the way the real re-processed track changes.

## TL;DR

Claim: CANDI's output does not follow a change in read length or run type with biology fixed, because its training has no pair that shows such a change. The probe keeps the input fixed (the T_ assays of the cell type), swaps only the output covariates from the base's to an arm's, and compares the predicted change with the real base-to-arm change of [[t112_build_the_counterfactual_arms_for_t|the counterfactual arms]]. Supported if the predicted log2 change stays near 0, or does not track the real one, and the arm's NLL does not improve. Refuted if the predicted change tracks the real one beyond a permuted-covariate null. Depth arms are the positive control: the offset head must reproduce them.

## Null

Instrumentation: per-bin chr21 arm/base log2 ratios are mostly sampling noise, so even a covariate-aware model could not track them.

<!-- Recommended by Claude via the crux-null agent; the PI delegated the choice on 2026-10-25. The PI approved it with `crux approve-null h26` on 2026-10-26. -->

## Problem Statement

CANDI's training never shows one experiment next to a counterfactual of itself in read length or run type, so nothing in its objective identifies how these covariates change a track. The counterfactual arms give real tracks that differ from a base in exactly one knob, so the correction CANDI would have to make is known.

## Idea / Hypothesis

A trained CANDI checkpoint, told a counterfactual arm's read length or run type in place of the base's, moves its predicted counts no closer to the arm's real track than when told the base's own covariates

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
<!-- Thresholds marked "recommended" were proposed by Claude under the PI's delegation (2026-10-25) and the PI approved them in chat on 2026-10-25. -->

**Probe definition, fixed before any inference.** Pairs = 7 tracks × {read length 36, read length 50, run type PE}. A pair whose arm value equals the base's native value is dropped and listed (no swap exists). Prompt (a) = the base's covariates; prompt (b) = the base's covariates with ONLY the knob changed, depth held at the base's value. Real ratio R = log2((arm+1)/(base+1)), median-centred per pair, so the global level (the arm's own mapped depth, which the depth offset explains) is out of the claim. Gate resolution for ratios: 1 kb pooled bins (PI-approved; 25 bp, 200 bp and 5 kb are reported, never gating). NLL is always computed at the native 25 bp NB. Checkpoint seeds s0 and s1 (t81, md5 checked against results/t81/TRAIN_CANDI_EIC19.md); a claim-directed check is met only if it is met on every seed that exists. Eligibility (`effectfloor`) is computed and frozen BEFORE any model inference.

**Outcome-neutral controls** must pass whatever the claim is; a failure makes the run invalid, never a refutation.

- [ ] [outcome-neutral] `swapwired` (manipulation check) — for every pair, assert on the decoder's metadata tensor that prompts (a) and (b) differ in exactly the intended field, encoded in the training data's units and inside the trained vocabulary (run-type index in the table; read length in the same scaling the dataset loader uses). Record the output-side η difference separately from the count difference.
      fails-if:: the read-length or run-type field is mis-encoded, lands in the wrong column, or is set on the input side, so a Δ of 0 is a wiring bug, not the model's behaviour
- [ ] [outcome-neutral] `depthcontrol` (positive control) — on the 15M / 7.5M / 3.75M arms of all 7 tracks: (1) the predicted global Δ (told depth arm − told base) equals log2(arm reads / base reads) within 0.01 log2; (2) the arm's NB NLL is lower under the depth-arm prompt than under the base prompt; (3) the real total log2(arm/base) is within 0.05 of log2 of the thinning fraction. The offset head makes (1) true by construction, so this tests the probe's plumbing and file pairing, not learning.
      fails-if:: y_meta does not reach the decoder, the decode reads the wrong column, or arm and base products are mispaired
- [ ] [outcome-neutral] `thinningblank` (negative control) — run the `effectfloor` statistic on the median-centred depth-thinning pairs. Binomial thinning has no per-bin pattern, so all 21 depth pairs must come out NOT eligible.
      fails-if:: the split-half apparatus finds a "reproducible arm effect" where none exists (for example from the +1 pseudocount compressing low-count bins the same way in both halves)
- [ ] [outcome-neutral] `effectfloor` (precondition against the null) — per pair, with the t117 halves: r_split = Spearman across 1 kb bins between log2((arm_pr1+1)/(base_pr1+1)) and log2((arm_pr2+1)/(base_pr2+1)), each median-centred and computed within deciles of base mean count, then averaged over deciles. A pair is ELIGIBLE when r_split exceeds the 99th percentile of a block-shift null (99 circular shifts of the pr2 ratio by ≥ 1 Mb on chr21) and r_split ≥ 0.2 (PI-approved). Met when at least one read-length pair AND at least one run-type pair are eligible; claim-directed checks score eligible pairs only.
      fails-if:: the null holds; the run then reads invalid for that knob

**Claim-directed checks** (eligible pairs only). Combination rule `all`, over `gapleft` and `swapcorr`. Joint power is about 64% (two checks at 80%), lower with the seed requirement. Thresholds may not be loosened to compensate.

- [ ] [hypothesis] `gapleft` — score the arm's pr2 half (all predictions shifted by the same log2 half-depth offset) under P_base (prompt a), P_arm (prompt b), and P_oracle = P_base × 2^(R̂), where R̂ is the median-centred, decile-adjusted 1 kb ratio estimated from the pr1 halves only (cross-fitted). G_oracle = NLL(P_base) − NLL(P_oracle); G_model = NLL(P_base) − NLL(P_arm); 95% CIs by a 100 kb block bootstrap on chr21. Met for a pair when the lower bound of G_oracle > 0 AND the upper bound of G_model / G_oracle ≤ 0.25 (PI-approved). Met overall when met on every eligible pair.
      fails-if:: the model, told the arm's knob, closes more than a quarter of a real, reachable gap
      discriminates:: true
- [ ] [hypothesis] `swapcorr` — Spearman between the median-centred model Δ (prompt b − prompt a, log2 mean, 1 kb) and the median-centred full-product real R. Met for a pair when ρ ≤ the 95th percentile of a block-shift null (99 circular shifts of Δ by ≥ 1 Mb). If max |Δ| < 1e-4 log2 over the chromosome, ρ is recorded as 0 and the pair is labelled "covariate unused". Met overall when met on every eligible pair.
      fails-if:: the model gets the direction and rank of the arm effect right but shrinks its size

**Reported, never gating.** Global level (predicted vs real track-wide log2(arm/base)); every gate statistic at 25 bp, 200 bp and 5 kb; optional `python -m candi.bench` tier C (`covuse`, `covshare`); CRPS only as `crps`, `crps_oracle_scaled` and `scale_error` together.

**Amendments after seeing data (PI rulings, 2026-10-26).** Post-hoc: made after the first run's numbers were seen. They bind rerun 1; the check texts above are kept as first pre-registered. Recorded verbatim:

> (ii) Rerun with two repairs, labelled as post-hoc amendments: (a) `effectfloor` (and `thinningblank`, which uses the same statistic) drops the decile stratification: r_split = Spearman across 1 kb bins of the median-centred half ratios, no deciles; the thinning-blank control is kept as the guard against pseudocount-driven false reproducibility; (b) `depthcontrol` checks the NLL direction only (part 2: the arm's NB NLL is lower under the depth-arm prompt than the base prompt) plus part 3 (real totals vs thinning fraction); part (1), the 0.01 log2 arithmetic identity, is dropped and its values reported only.
> (iii) Do NOT use the 2019 imputation challenge's blind-panel (B_) arrays in any form (not as prompts, not loaded). Rebuild prompts from non-blind arrays only; if a track's cell type then has no usable prompt, drop the track and list it. Record which tracks remain.
> (iv) Keep the existing 18 MB chr21 halves extract on Fir (no re-copy). Do not move any data through the laptop again: cluster-to-cluster data moves are Globus only; if you need data that exists only on Nibi, stop and say so on the return.
> Everything else stays as pre-registered (other thresholds, seeds s0/s1, combination rule `all` over gapleft and swapcorr, block-shift nulls; use 99 shifts everywhere this time, including the 25 bp reported-only null).

**Second post-hoc amendment (PI ruling, 2026-10-26), made after rerun 1's numbers were seen; binds rerun 2.** Recorded verbatim:

> Rerun with a residual reproducibility test: for each pair, fit the expected signal-level curve E[R | μ] and subtract it, then test split-half agreement of the residuals (Spearman across 1 kb bins, same block-shift null with 99 shifts, same r ≥ 0.2 floor). Define the curve without using the arm's own per-bin pattern: primary = a smooth fit (e.g. binned-median or LOESS on log μ, μ estimated from the OTHER half's base to avoid shared-noise circularity — e.g. curve for the pr1 ratio fitted against base_pr2 level, and vice versa; document the exact choice) and as a sanity check also the analytic curve with f = arm total / base total. Apply the same residual statistic to `effectfloor` and `thinningblank`; `thinningblank` must now come out NOT eligible on the depth pairs, or the run is invalid again. If eligible knob pairs exist, evaluate `gapleft` and `swapcorr` as pre-registered, with the oracle ratio R̂ = fitted curve + pr1 residual (cross-fitted), and record that change too. Everything else as rerun 1 (amended depthcontrol = parts 2+3; no blind-panel B_ arrays in any form — keep the tripwire; seeds s0/s1; 1 kb gate; combination rule `all`). First verify the mechanism directly: show on the thinning pairs that r_split of the raw ratio drops to near the null after subtracting the curve; report both numbers.

Exact choices, fixed before rerun 2 ran. At 1 kb, R1 = log2((A1+1)/(B1+1)) and R2 likewise for pr2. The curve for R1 is fitted against L2 = log2(B2+1), the pr2 base level, and the curve for R2 against L1. Primary curve: binned median, 50 equal-count bins of L by average rank (ties share a bin), median R per bin, linear interpolation between bin-median L values, flat beyond the ends. Analytic curve: E[R1] = log2((f1·μ+1)/(μ+1)) with f1 = ΣA1/ΣB1 and μ = B2·ΣB1/ΣB2 (likewise for R2). Residual e = R − curve. r_resid = Spearman(e1, e2); the null shifts e2. The gating statistic is the primary residual; the analytic residual and the raw r_split are reported. Also reported, never gating: the same residual with the curve fitted against the half's own base level and against the pooled base level, because a synthetic thinning check before the run showed the cross-half level can push r below 0. Oracle for `gapleft`: R̂ = c1(L2) + e1, which equals the uncentred pr1 ratio R1 by construction; this replaces rerun 1's decile-centred R̂.

## Planned Intervention

CPU only, no training. Checkpoint: the selected CANDI checkpoint of [[t81_retrain_every_trainable_method_under|the retrain-every-method task]] (`/project/def-maxwl/mforooz/t81_checkpoints/t81_eic_19_s0.best.ckpt` and `_s1`; check the md5 against `results/t81/TRAIN_CANDI_EIC19.md` first, because the `/project` copy was once stale). Per track (C12M02, C19M16, C40M17, C40M18, C07M20, C19M22, C07M29): prompt with the cell type's T_ assays; decode the target column with y_meta set to (a) the base's covariates, (b) each arm's read length 36 / 50, (c) run type PE, (d) the depth arms 15M / 7.5M / 3.75M. Read per bin, on chr21: Δ = log2 mean(told arm) − log2 mean(told base) against log2(arm / base counts, +1); NB NLL of the arm track under (a) and under (b–d). Null: the same Δ with the arm's covariate values permuted across tracks. Optional second reading: `python -m candi.bench` with tier C (`covuse`, `covshare`) on the same checkpoint. The causal follow-up, which needs GPU: retrain CANDI with base→arm pairs added as input→target and repeat the probe.

## Run Links

- 2026-10-25, chr21, CPU. Nibi 24997320 (halves extract); Fir 453552 (eligibility, frozen before inference), 454386_0-13 (inference, seeds 0 and 1), 457511 (scoring). Run root `fir:/project/def-maxwl/mforooz/h26`; small outputs in `results/h26/` (gitignored, main checkout) with `FIR_PATH.txt`. Code `tools/h26/`, `slurm/h26/`.
- 2026-10-26, rerun 1 (amendments of 2026-10-26), chr21, CPU. Fir 528375 (eligibility, written 01:06:09, before inference), 528686_0-13 (inference), 528687 (scoring). Run root `fir:/project/def-maxwl/mforooz/h26/rerun1`, reusing the first run's halves extract in place; small outputs in `results/h26/rerun1/` with `FIR_PATH.txt`. Code on branch `chunk/h26-rerun`.
- 2026-10-26, rerun 2 (second amendment of 2026-10-26), chr21, CPU. Fir 601843 (eligibility, written 13:10:33), 601844 (scoring). Run root `fir:/project/def-maxwl/mforooz/h26/rerun2`; predictions and halves read in place from rerun 1 and the first run (`infer.py` unchanged). Small outputs in `results/h26/rerun2/` with `FIR_PATH.txt`. Code on branch `chunk/h26-rerun2`.

## Artifacts

- [Rerun 2 report](results/h26/rerun2/REPORT.md)

## Findings

Rerun 2 of 2026-10-26, under the second amendment. Numbers from `results/h26/rerun2/eligibility.json` (`mechanism_check`) and `results.json`, 1 kb. Proposed ticks, not applied: `swapwired` met; `depthcontrol` unmet; `thinningblank` met by its letter only; `effectfloor` unmet; `gapleft` and `swapcorr` could not be evaluated. See [[pseudocount-log-ratios-carry-a-signal-level-curve]].

- Mechanism: on the 21 thinning pairs, raw r_split is -0.03 to 0.63 (15 eligible); after subtracting the curve it is -0.11 to -0.37, against a null p99 of 0.008-0.013. The curve explains the raw reproducibility, but the residual overshoots below the null instead of landing on it, most at 15M (raw near 0, residual about -0.36). The cross-half level fit induces this anti-correlation; own-level and pooled-level fits are negative too.
- `thinningblank`: 0 of 21 eligible, but only because the statistic is biased negative, as in the first run.
- `effectfloor`: 0 of 19. Read length 36 bp residual r 0.015-0.086 (above null p99, below the 0.2 floor); 50 bp and run type at or below 0.
- `swapwired` 80 of 80; `depthcontrol` part (2) 32 of 42, part (3) 21 of 21 (same predictions as rerun 1).

Rerun 1 of 2026-10-26, under the post-hoc amendments. Numbers from `results/h26/rerun1/eligibility.json` and `results.json`, 1 kb. Proposed ticks, not applied: `swapwired` met; `depthcontrol` unmet; `thinningblank` unmet; `effectfloor` unmet; `gapleft` and `swapcorr` could not be evaluated.

- Prompts are the `T_` cells only; no `B_` array was opened. All 7 tracks kept a prompt (HAP-1 and SJSA1 have one assay each). Predictions are bit-identical to the first run.
- `swapwired`: 80 of 80 rows.
- `depthcontrol`: part (2) holds on 32 of 42 rows (fails mostly at 15M); part (3) on 42 of 42. Part (1), reported: 10 of 42.
- `thinningblank`: 15 of 21 depth pairs eligible (7.5M r_split 0.34-0.52, 3.75M 0.48-0.63; null p99 0.32-0.40). Without deciles, the split-half statistic finds reproducibility where thinning has none.
- `effectfloor`: read length 8 of 12 pairs eligible (36 bp r_split 0.28-0.36, the same range as the thinning pairs); run type 0 of 7 (max 0.162).
- As scored on the 8 eligible pairs: the pr1 oracle worsens pr2 NLL on 38 of 38 rows, so `gapleft` has no reachable gap; `swapcorr` holds on 15 of 16 pair-seed rows (C07M20 36 bp, seed 0: ρ 0.315 > p95 0.225). With both controls failed, these are not read.

Run of 2026-10-25 (PI delegated; the null and the thresholds were approved after the run, on 2026-10-25 and 2026-10-26). Numbers from `results/h26/eligibility.json` and `results.json`. Proposed ticks, not applied: `swapwired` met; `depthcontrol` unmet; `thinningblank` met; `effectfloor` unmet; `gapleft` and `swapcorr` could not be evaluated (no eligible pair).

- `swapwired`: 80 of 80 prompt rows change one field in the target column only; read lengths 36 and 50 lie in the training range 20-101.
- `depthcontrol`: part (1) holds on 10 of 42 rows (worst 0.107 log2), because told depth also moves eta through FiLM; part (2) on 32 of 42; part (3) on 21 of 21.
- `effectfloor`: 0 of 19 knob pairs eligible; best 1 kb r_split 0.084. Decile stratification anti-correlates the two base halves and biases r_split down (depth pairs -0.15 to -0.46), so `thinningblank` passes for that reason too. The design needs repair before a rerun.
- Reported only: the model does use both covariates on the output side. Told 36 bp, chr21 level moves -0.22 to -0.89 log2 (real -1.05 to -1.08, mostly the bin-count rule). Told PE, it moves -0.19 to -0.98 for ChIP and +0.52 to +0.70 for DNase; the real tracks move -0.08 to +0.01. At 1 kb the pr1-fitted oracle worsens NLL on 38 of 38 rows.

Prior evidence, before the run. Pre-CANDII numbers are single seed, 12 held-out targets.

- Code: an unmasked assay's target is the same experiment, so read length and run type are equal on both sides; only depth varies, by independent binomial thinning (`src/candi/dataset.py:43-65, 430-446`; `src/candi/store/dataset.py:142`).
- With the depth offset on, run_type steering was functionally null (`AGENTS.md` §7.1, §7.2 item 3); at weight decay 0 its direction score was 0.559 (archive h47).
- With the offset off, run_type changed imputation in the right direction, but run type, read length and depth are collinear on that panel (archive h42).
