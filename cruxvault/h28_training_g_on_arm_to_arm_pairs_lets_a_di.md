---
id: h28
type: idea
schema: 2
title: Training g on arm-to-arm pairs lets a direct g -> f map meet the depth law on never-trained arm-to-arm pairs, without a pivot through the base
parent: q7
status: idea
rule: all
measurement: 
replicates: 
verdict: 
metric: 
created: "2026-10-29T20:07:43"
updated: "2026-10-29T20:53:44"
null_approved: "2026-10-29T20:53:44"
null_hash: 89756075876f7d61
---

# h28 — Training g on arm-to-arm pairs lets a direct g -> f map meet the depth law on never-trained arm-to-arm pairs, without a pivot through the base

Parent:: [[q7_why_do_the_covariate_conditioned_g_f_des]]

## ELI5

If the translator also practises on pairs where neither side is the original track, can it translate between two altered tracks directly, without a detour through the original?

## TL;DR

The direct g → f designs fit trained base ↔ arm depth pairs and miss the depth law on never-trained arm → arm pairs, while chaining two trained maps through the base meets it (0.07–0.12 against 0.29–0.34 direct). That fits two readings: g needs the base as a pivot, or g simply never saw an arm → arm pair. This claims the second: trained on some arm → arm pairs as well, a direct map meets the law on held-out arm → arm pairs. It decides what CANDI needs, since CANDI has a natural base only for depth (the full-depth track): if this holds, augmenting CANDI's corpus with counterfactual pairs is the route; if it fails, the pivot is. Settled by the median |log2 error| of the predicted scale on held-out arm → arm depth pairs against the via-base chain and the seed floor.

## Expect

Darm1 closes at least half the never-trained depth-law gap from the direct map to the pivot through the base (≤ 0.160), transfers the depth ratio, and closes at least half the held-out CRPS gap with a gain above 2 × seed spread.

_Pre-run: the five checks and their bars were committed on 2026-10-29, and the 42-run grid was submitted on 2026-10-30. Darm2 was read, not checked, by the PI's design._

## Null

<!-- spec 09: the null hypothesis — the most parsimonious alternative explanation under
     which this result appears although the claim is false. One line, <=25 words, naming a family. Your checks must discriminate against it.
     The PI approves it before checks are written: `crux approve-null <id>`. -->
Leakage: held-out arm→arm depth ratios also occur among trained arm→arm pairs, so g recalls seen ratios, not a pair-general depth law.

## Problem Statement

Arm of the PI's 2026-10-29 ruling on the unseen-pair gap: run beside the t135 pivot seeds, with the pivot as the default reading. Augmenting CANDI (by BAM or FASTQ re-derivation) is deferred until this and t135 say whether the architecture generalizes without it.

Why (PI, 2026-10-29): "in real candi data, we don't have a lot of counterfactuals for most covariates ... so in real candi data, the definition of 'base' is not very straightforward". If arm-to-arm training generalizes without the base, "it motivates augmenting real candi data".

## Idea / Hypothesis

Training g on arm-to-arm pairs lets a direct g -> f map meet the depth law on never-trained arm-to-arm pairs, without a pivot through the base

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
<!-- The PI accepted these five checks and their thresholds in chat on 2026-10-29 (plan/T136_ARM2ARM_GRID.md §6, "Checks as written"; design option A, no extra seeds). -->

**Definitions.** Every reading is per-track g, chr19 + chr21, pooled by the t118 rule (per seed: mean over a track's pairs, then mean over tracks; the reading is the mean over 3 seeds). "Spread" is the seed spread, max − min over seeds 0, 1, 2 of the per-seed value. `heldout_r2` = the 4 ordered depth pairs that touch 7.5M (ratio 2). `trained_r4` = 15M ↔ 3.75M (ratio 4). `trained_base` = base ↔ 15M and base ↔ 3.75M. The held-out arm → arm pairs = every law pair that touches 7.5M, all knobs. Dref = the existing phase-2 old D direct runs; Dviabase = Dref's two trained maps in series through the base. The readings recompute Dref and Dviabase on exactly the held-out pairs. Every CRPS is quoted with `crps_oracle_scaled_all` and `scale_error_all`.

**Outcome-neutral controls** must pass whatever the claim is; a failure makes the run invalid, never a refutation.

- [ ] [outcome-neutral] `basefit` — In Darm1 and in Darm2, the `trained_base` median |log2 error| is within 2 × the larger seed spread (the arm's or Dref's; Dref spread 0.010, so the band is at least ±0.020) of Dref's 0.052, and the trained base ↔ arm `crps_all` is within 2 × the larger seed spread (Dref spread 0.003, so at least ±0.006) of Dref's 0.445. A failure invalidates that arm's run.
      fails-if:: the extra arm → arm pairs crowd out the base pairs (Darm2 draws about 11 % base pairs), so the arm is not the same model plus arm → arm pairs
- [ ] [outcome-neutral] `runhealth` — All 42 runs (21 per arm) finish; no `config["train_pairs"]` row holds a 7.5M product; `len(config["extra_pairs"])` is 2 per track in Darm1, and 306 per histone track and 30 for DNase C12M02 in Darm2; at most 2 of 21 runs per arm have their best step in the last 10 % of the 12 000-step cap (best step > 10 800; Dref: 0 of 21).
      fails-if:: the harness trained the wrong pairs, leaked a 7.5M product into training, or cut runs before they converged

**Claim-directed checks**, read on Darm1 only and only when both controls pass. Combination rule `all` over `lawheldout`, `ratiotransfer` and `crpsheldout`. Three checks at 80 % power each give about 51 % joint power; the thresholds may not be loosened to compensate. Darm2 is read, not checked.

- [ ] [hypothesis] `lawheldout` — On `heldout_r2`, Darm1 closes at least 0.5 of the gap from Dref direct to Dviabase in median |log2 error| of the predicted scale: (Dref − Darm1) / (Dref − Dviabase) ≥ 0.5. With the phase-2 references (Dref 0.195, Dviabase 0.125) this means Darm1 ≤ 0.160. No noise clause (design option A: the gap 0.070 is below 2 × Dref's spread 0.061).
      fails-if:: arm → arm training on the ratio-4 pair leaves the ratio-2 held-out pairs where Dref left them
- [ ] [hypothesis] `ratiotransfer` — In Darm1, the `heldout_r2` median |log2 error| (ratio 2, never trained as an arm → arm ratio) is at most the `trained_r4` median |log2 error| (ratio 4, trained) + 2 × the larger of the two seed spreads.
      fails-if:: g recalls the one arm → arm ratio it trained on and does not carry the depth law to a ratio it has not seen on an arm → arm pair
      discriminates:: true
- [ ] [hypothesis] `crpsheldout` — On the held-out arm → arm pairs, Darm1's pooled `crps_all` closes at least 0.5 of the Dref → Dviabase interval, AND the gain Dref − Darm1 exceeds 2 × the larger seed spread (Darm1's or Dref's), AND the `crps_oracle_scaled_all` part closes at least 0.25 of its own Dref → Dviabase interval.
      fails-if:: the depth-law scale moves but the predicted distribution on unseen arm → arm pairs does not improve, or improves through `scale_error` alone

## Planned Intervention

[[t136_design_and_run_the_arm_to_arm_train|The arm-to-arm grid task]]: old D's phase-2 recipe (per-track g, depth 7.5M held out) in two new arms. Darm1 adds the two 15M ↔ 3.75M pairs (ratio 4) to training; Darm2 adds every arm → arm pair that does not touch 7.5M. 2 arms × 7 tracks × 3 seeds = 42 runs on Nibi (PI 2026-10-29; see Run Links). The score set is every pair that touches 7.5M; the depth pairs in it have ratio 2, which Darm1 never trains as an arm → arm ratio. The comparators are the existing phase-2 old D runs, direct and via the base, and the seed spread. Design and checks: `plan/T136_ARM2ARM_GRID.md`. The PI approved the null, the checks c1–c5 and design option A (no extra seeds) on 2026-10-29. Exploratory.

## Run Links

- **Cluster** (PI ruling 2026-10-29). The grid runs on Nibi at 2 CPUs, `--mem=10000M` and `--time=7:00:00` per run, not on Fir MIG slices. Reason (Fir queue diagnosis, 2026-10-29): the def-maxwl_gpu fair-share was exhausted; Fir billed each job by its 4 CPUs (MAX_TRES billing); and 12 h was requested against a measured median near 86 min. The Nibi queue was about 10× shorter. The Dref and Dviabase reference runs stay on Fir (`/project/def-maxwl/mforooz/t128p2/runs`, `/project/def-maxwl/mforooz/t128p2/viabase`) and are not rerun.
- **Kit**: Nibi `/project/def-maxwl/mforooz/t136/code/K1` (git 1f634e1). Output root Nibi `/project/def-maxwl/mforooz/t136`; job ledger `/project/def-maxwl/mforooz/t136/jobs.tsv`.
- **Smoke** (300 steps, 3 evaluations, task rows 9 = Darm1 and 30 = Darm2): train array 25389086 and law array 25389100, both COMPLETED. Wall time Darm1 11:55 (4.8 GB), Darm2 16:59 (3.9 GB). Projection at 12 000 steps and 49 evaluations: Darm1 < 1 h, Darm2 < 1.5 h with scoring, both inside 7 h. The law readback was correct: 7.5M held out, no 7.5M product in any training pair, law records carry the CRPS split.
- **Full grid** (submitted 2026-10-30 05:45Z): Darm1 train 25403683 with law 25403688; Darm2 train 25403711 with law 25403725. Never resubmit these.

## Artifacts

<!-- what the run produced. Keep files under results/h28/ and link at least the report:
     - [Report](results/h28/report.md)   - results/h28/curve.png -->
_(none yet)_

## Findings

_(written by the PI/agent when the case is closed)_
