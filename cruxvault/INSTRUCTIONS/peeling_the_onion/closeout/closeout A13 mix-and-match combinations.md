---
title: "A13: mix-and-match combinations"
parent: "[[closeout todo]]"
created: 2026-11-17
---

# A13: mix-and-match combinations

## 1. What the PI asked

- PI ruling of 2026-11-16 (`closeout todo.md`): "Mix-and-match combinations of winning settings are done in this work."
- PI ruling of 2026-11-17 on fixed input depth: test it as a mix-and-match arm; the caveat, verbatim: "requiring fixed input depth can be unrealistic in real world deployment and inference. just have this in mind".
- PI rulings of 2026-11-17: all covariate fields included (tables `eic_v1_2`); the Anscombe transform is dropped; no matched-compute runs.

Success criterion served: 4 (design settled). A12 (minimum covariate set) and M1 (settled design) depend on this item.

## 2. Question

Per architecture (conv autoencoder, conv + attention, U-Net): which combination of the single ablation choices that helped gives the best recipe on all 256 evaluation regions, on tables `eic_v1_2`, and do the gains of the single choices add up?

## 3. State today

- Single ablations scored on 16 regions (`p16_sv`), tables `eic_v1`, 3 seeds: [[ABLATIONS_READ]] and `results/ablations_closeout/arm_summary_p16_sv_2026-11-17.tsv`. The 16-region scores are not settled (the 16-versus-256 check failed; correction terms are scored on all 256 regions from now on).
- Base recipes on `eic_v1_2` are trained on Fir (`runs_fir/eic_v1_2_rung{2,3,4}_base_heldout`, 3 seeds, scored in `p16_sv` only). Their 256-region score does not exist.
- `eval/eic_v1_2/evalset.json` exists on Fir only; the 256-region reference terms of `eic_v1_2` (`eval/eic_v1_2/baselines`) exist on neither cluster.

## 4. Specification as worked (2026-11-17)

### 4.1 Which single choices qualify

Rule (foreman brief): a choice qualifies on a model when it is better than the base beyond the seed range on at least one of imputation, gap closed, calibration, and not worse beyond the seed range on any other of the three. "Beyond" means the difference of means exceeds the larger of the two seed ranges (max minus min over 3 seeds). I applied the rule to every `abl_*` row of the TSV against `rung{2,3,4}_base_heldout`; the full table is `results/closeout_MIX/qualify_single_choices.tsv`.

| model | choice | imputation | gap closed | calibration | qualifies |
|---|---|---|---|---|---|
| conv autoencoder | same-view pairs only | +0.031 (bar 0.028) | +0.012 | -0.055 | yes |
| conv + attention | concatenation injection | +0.002 | +0.050 (bar 0.031) | -0.053 | yes |
| conv + attention | fixed input depth 2^23 | -0.011 | +0.049 (bar 0.038) | +0.012 | yes |
| conv + attention | imputation loss weight 4 | +0.039 (bar 0.017) | +0.000 | +0.057 | yes |
| conv + attention | NB CRPS loss | -0.017 | +0.053 (bar 0.031) | -0.029 | yes |
| conv + attention | same-view pairs only | +0.032 (bar 0.021) | +0.010 | +0.003 | yes |
| U-Net | average baseline input | +0.034 (bar 0.023) | -0.006 | +0.001 | yes |
| U-Net | Pearson residual transform | +0.018 | +0.078 (bar 0.071) | +0.009 | yes |
| U-Net | fixed input depth 2^23 | -0.028 (worse, bar 0.023) | +0.168 (bar 0.071) | -0.020 | no by the rule; IN by the PI ruling of 2026-11-17 |
| U-Net | two-step, fixed reference depth | +0.019 | +0.129 (bar 0.071) | -0.165 (worse, bar 0.119) | no |

Not in the pool, with the reason: asinh and soft clip on the U-Net (composite +0.015 and +0.016 beyond, but no single term beyond: they fail the rule); only one count transform can be on at a time, so Pearson stands for the transforms. Two-step with a fixed reference depth fails the rule on calibration and doubles the FLOPs per window; its per-class reference depth is another agent's work (branch chunk/onion-AB2). The conv autoencoder with fixed input depth has 2 seeds and no qualifying term. The brief's lead "conv + attention imputation 0.387 to 0.426" for loss weight 4 matches the TSV (0.3868 to 0.4261).

**Qualification gate (added 2026-11-17 after review).** The table above uses 16-region scores. On 16 regions a correction difference below about 0.05 is not reliable (foreman's check), and three conv + attention switches qualify on gap closed alone at that limit: concatenation +0.050, fixed input depth +0.049, NB CRPS +0.053. So the pool is provisional. Before any submission, the rule is applied again to 256-region scores of the single-ablation runs: `python tools/minicandi/mix_arms.py qualify --summary <arm-summary tsv, --sub p256_sv>`. It exits 0 (PASS: the pool stands), 1 (FAIL: a switch no longer qualifies; rewrite `POOL` and the configs, then recount the cost), or 3 (WAIT: a switch has no 3-seed 256-region score). `submit_mix.sh STAGE=runs` refuses unless `QUALIFY_SUMMARY` passes the gate. The gate can only remove switches. U-Net fixed input depth stays in by the PI ruling.

| switch, model | 256-region score of the single run |
|---|---|
| fixed input depth, conv + attention and U-Net | queued on Fir (runs_fir/score_p256_decisive.sh, check 4) |
| imputation loss weight 4, conv + attention | queued on Fir (check 4) |
| average input, U-Net | queued on Fir (check 4) |
| same-view only, conv autoencoder and conv + attention | none: needs `STAGE=gate` |
| concatenation, conv + attention | none: needs `STAGE=gate` |
| NB CRPS, conv + attention | none: needs `STAGE=gate` |
| Pearson, U-Net | none: needs `STAGE=gate` |

The references `rung{2,3,4}_base_heldout` have `p256_sv` scores (check E1). `STAGE=gate` scores the 5 missing runs, 3 seeds each, on Fir (the checkpoints are there), score only: 3 x 30.0 + 3 x 3 x 25.4 + 3 x 51.85 = 90.0 + 228.6 + 155.5 = 474.1 CPU-hours.

### 4.2 Arms (configs `tools/minicandi/configs/ablations/mix/rung{2,3,4}/`, generator `tools/minicandi/mix_arms.py`)

Each switch is the exact key set of its single-ablation config; no two switches share a key. Each arm is the base recipe plus the union of its switches.

| model | switches | design | arms besides the reference |
|---|---|---|---|
| conv autoencoder | same-view only | full factorial | 1 |
| conv + attention | concatenation, fixed depth, loss weight 4, NB CRPS, same-view only | half fraction 2^(5-1), defining relation I = +ABCDE: every arm with 1, 3 or 5 switches on | 16 |
| U-Net | average input, Pearson, fixed depth | full factorial | 7 |

Why not the full factorial on conv + attention: it costs 31 arms, about 4,915 CPU-hours for that model alone, against 2,517 for the half fraction (section 4.8); the whole design would rise from about 5,390 to 7,788 CPU-hours. The half fraction is resolution V: the intercept, the 5 main effects and the 10 two-switch interactions are orthogonal (the test checks D'D = 16 I), so every main effect and every pairwise interaction is estimable, aliased only with three- and four-switch interactions. This half (odd switch counts) holds every single switch and the all-on arm; the reference (no switch) lies outside it and is run beside it. Arms with 2 or 4 switches are not run.

The reference arm, per model: the base recipe `configs/rung<r>/base_heldout.json` on `eic_v1_2` under the run name `eic_v1_2_rung<r>_base_heldout`. On Fir it exists (DONE), so only its 256-region scoring is new; on Nibi it is trained in the same submission. Every arm: 3 seeds (0, 1, 2), 80,000 windows, tables `eic_v1_2`, all 256 regions with the same-view pass, scorer sub-directory `p256_sv`, reference terms `eval/eic_v1_2/baselines`.

### 4.3 The deployment-realistic variant of fixed input depth

Every arm with fixed input depth is scored twice: (1) `p256_sv`, the inputs thinned to 2^23 at scoring as in training (the score that the single ablation reported); (2) `p256_sv_native_depth`, the same checkpoint scored from native-depth inputs (no thinning; C states the true depth), i.e. the deployment case where no depth cap can be imposed. Code: `ONION_SCORE_INPUT_DEPTH=native` (score.py, predict_store.py; `SCORE_INPUT_DEPTH=native` in submit_arms.sh). The decision rule reads score (2) for these arms; score (1) is reported beside it with the PI's caveat.

### 4.4 Prior leads (all `p16_sv`, tables `eic_v1`, from the TSV)

- U-Net base 0.432 / 0.507 / 0.610 (imputation / gap closed / calibration); average input 0.466 / 0.501 / 0.611; Pearson 0.450 / 0.585 / 0.619; fixed depth 0.404 / 0.675 / 0.590.
- Conv + attention base 0.387 / 0.673 / 0.652; loss weight 4 0.426 / 0.673 / 0.710; same-view only 0.419 / 0.683 / 0.656; concatenation 0.389 / 0.723 / 0.599; NB CRPS 0.369 / 0.726 / 0.624; fixed depth 0.376 / 0.722 / 0.665.
- Conv autoencoder base 0.382 / 0.718 / 0.631; same-view only 0.413 / 0.730 / 0.576.
- `eic_v1_2` base, `p16_sv`: conv autoencoder 0.370 / 0.682 / 0.599; conv + attention 0.386 / 0.671 / 0.627; U-Net 0.445 / 0.538 / 0.646.

### 4.5 Expectation, stated before any run

1. U-Net: average input and Pearson add on imputation (about +0.03 and +0.02) and do not interact beyond the seed range. Fixed depth keeps its gap-closed gain (+0.1 to +0.17) and its imputation loss (about -0.03) when combined.
2. U-Net, native-depth scoring of the fixed-depth arms: the gap-closed gain shrinks by at least half and imputation falls further, because the model never saw inputs deeper than 2^23 or a C depth above 23.
3. Conv + attention: loss weight 4 and same-view only both raise imputation and overlap (a negative interaction); concatenation and NB CRPS both raise gap closed and overlap the same way. No arm beats the best single switch on the composite beyond the seed range.
4. Conv autoencoder: same-view only raises imputation by about 0.03 again on 256 regions; the correction-only term falls (it lost 0.036 on 16 regions, within).
5. The 256-region gaps are smaller than the 16-region gaps for gap closed (E1: correction-only 0.03 to 0.046 higher on 16 regions).

### 4.6 Decision rule (picks the recipe per model)

1. Score every arm and the reference: imputation, gap closed, calibration, composite (version 3), and the correction-only term, each a mean over 3 seeds with its seed range. For an arm with fixed depth, use the native-depth score.
2. An arm is eligible when none of imputation, gap closed and calibration is lower than the reference beyond the seed range.
3. Among the eligible arms, take the arm with the highest composite (call it the leader). An arm with fewer switches is "in range" when the leader's composite minus its composite is not larger than the larger of the two composite seed ranges (the leader's and that arm's). The reference counts as an arm with 0 switches. If one or more arms with fewer switches are in range, the winner is the in-range arm with the fewest switches; a tie on the switch count goes to the higher composite. So the reference wins whenever it is in range of the leader (the simpler arm, D13).
4. The recipe is the reference when no eligible arm has a composite or one of the three terms higher than the reference beyond the seed range.
5. Conv + attention: fit the main effects and two-switch interactions on the 16 arms, using the native-depth score for every arm with fixed depth (the score of step 1); fit the thinned score as a second, reported model. If the fitted value of an untested arm (2 or 4 switches) exceeds the chosen arm's composite by more than its seed range, name it as a confirmation run; do not adopt it untested.
6. Flag (do not reject) an arm whose correction-only term is lower than the reference by more than twice the seed range; the PI decides.

### 4.7 Outcome -> conclusion

| outcome | conclusion |
|---|---|
| an eligible arm with 2+ switches wins (rule step 3) | the gains combine; that combination is the model's recipe for M1 |
| a single-switch arm wins, and the multi-switch arms are within or worse | the gains do not add; the recipe is that one switch |
| no arm passes step 4 | the recipe stays the base; the 16-region gains did not hold on 256 regions and `eic_v1_2` |
| a two-switch interaction is beyond the seed range | report the pair; it explains which gains overlap |
| fixed depth wins with thinned scoring but loses its gain under native-depth scoring | the gain needs the depth cap at inference; it is not deployable; the PI rules |
| fixed depth wins under native-depth scoring too | the gain comes from training on capped inputs; fixed-depth training is deployable |
| the winners differ across the 3 models | per-model recipes; M1 states which architecture goes forward |

### 4.8 Controls, traps, cost

Controls and traps:
- Scores of `eic_v1` and `eic_v1_2` differ in table hash; the scorer refuses to mix them. Every arm and its reference here use `eic_v1_2`, 256 regions, one kit and one cluster.
- The qualifying switches were chosen on the same 16 validation regions and cells; the 256-region rescore with the same seeds reduces but does not remove this selection bias (winner's curse). Expect smaller effects.
- Parameter counts: every arm builds within 10% of 1.0e6 (test). Concatenation removes about 15,000 parameters (-1.5%).
- Training time: average input arms take about 2 to 3 times the base wall time (sacct below); time limits are 8 h, 6 h or 4 h by switch.
- The average input file is `/scratch/mforooz/onion/aux/eic_v1_average.h5`, built for `eic_v1`; `eic_v1_2` changes only covariates. The dataset refuses an average that summed a row it may not average (RoleError); the Nibi check job below reads it with `eic_v1_2`.

Cost, in allocated CPU-hours (cores x wall time):
- Training per seed, mean of 3 seeds, Fir sacct of the existing runs (job ids from `runs_fir/<run>/seed<s>/train_*.out`): conv autoencoder base 14.84 (used for its arm), conv + attention `eic_v1_2` base 14.34 (the largest single-switch mean was 14.14), U-Net average input 30.56, U-Net fixed depth 17.94 (used for the U-Net arms without the average input), `eic_v1_2` bases 5.98 / 14.34 / 7.06.
- One 256-region same-view scoring of one run-seed, per model, measured on Fir (check E1: 8 predict jobs + 1 metrics job per run-seed, 3 seeds per run; `results/closeout_MIX/sacct_p256.txt`): conv autoencoder 90.1 / 3 = 30.0; conv + attention 76.3 / 3 = 25.4; U-Net 51.85, the mean of 53.4 (base, 160.3 / 3) and 50.3 (DNA-only skips, 150.9 / 3). The brief's figure of 23 is below all three.
- One arm = 3 seeds x training + 3 x scoring: conv autoencoder 3 x 14.84 + 3 x 30.0 = 134.5; conv + attention 3 x 14.34 + 3 x 25.4 = 119.2; U-Net 3 x 17.94 + 3 x 51.85 = 209.4; U-Net with average input 3 x 30.56 + 3 x 51.85 = 247.2.
- Training: references 3 x (5.98 + 14.34 + 7.06) = 82.1 (Nibi only); conv autoencoder 1 x 3 x 14.84 = 44.5; conv + attention 16 x 3 x 14.34 = 688.3; U-Net 4 x 3 x 30.56 + 3 x 3 x 17.94 = 528.2; total 1,343.2 on Nibi, 1,261.0 on Fir.
- Scoring: references 3 x (30.0 + 25.4 + 51.85) = 321.8; conv autoencoder 1 x 3 x 30.0 = 90.0; conv + attention 16 x 3 x 25.4 = 1,219.2; U-Net 7 x 3 x 51.85 = 1,088.8; total 2,719.8.
- Native-depth scoring: conv + attention 8 x 3 x 25.4 = 609.6; U-Net 4 x 3 x 51.85 = 622.2; total 1,231.8.
- Conv + attention, half against full factorial: half = 688.3 + 1,219.2 + 609.6 = 2,517.1; full = 31 x 3 x 14.34 + 31 x 3 x 25.4 + 16 x 3 x 25.4 = 1,333.6 + 2,362.2 + 1,219.2 = 4,915.0.
- 256-region reference terms of `eic_v1_2`: 4 baselines, 16 CPUs x (5,062 + 6,322 + 5,395 + 4,568) s of the `eic_v1` 256-region wall times = 94.9, plus the oracle reference (0.2): 95.1.
- Total: 1,343.2 + 2,719.8 + 1,231.8 + 95.1 = 5,389.9 CPU-hours on Nibi; 1,261.0 + 2,719.8 + 1,231.8 + 95.1 = 5,307.7 on Fir (Fir speeds; Nibi speed not measured). The full factorial would give 5,389.9 - 2,517.1 + 4,915.0 = 7,787.8. The gate stage adds 474.1 before it.
- Jobs (Nibi dry run): 81 training + 936 predict + 117 metrics = 1,134 jobs, plus 5 reference-term jobs.

## 5. Work steps

1. Specification (this section): done 2026-11-17.
2. Code and configs: `tools/minicandi/mix_arms.py`, 24 configs, the native-depth scoring switch, tests `tests/test_minicandi_mix_arms.py`: done 2026-11-17.
3. Qualification gate: `STAGE=gate` on Fir (5 runs x 3 seeds, score only), wait for check 4, then `select_rule.py arm-summary --sub p256_sv` and `mix_arms.py qualify --summary`. If it fails, rewrite the pool and recount.
4. Prerequisites (foreman): `eval/eic_v1_2/evalset.json` on the scoring cluster; a prediction-store root with room for the concurrent stores (about 55 GB and 1,700 files per run-seed).
5. Submit: `STAGE=baselines`, then `QUALIFY_SUMMARY=<tsv> STAGE=runs`, of `tools/minicandi/configs/ablations/mix/submit_mix.sh` (dry run by default).
6. Read: `select_rule.py arm-summary --sub p256_sv` and `--sub p256_sv_native_depth` (a comparison of a native-depth arm with a normal reference needs `--allow-mix input_depth_at_scoring`); apply section 4.6; fit the conv + attention effects; write `results/closeout_MIX/MIX_READ.md`; record findings.

## 6. Done when

Each model has a named recipe chosen by the rule of section 4.6 from 256-region scores on `eic_v1_2`, the fixed-depth arms carry both scores with the PI's caveat, the interactions of conv + attention are reported, and the findings are in the crux tree.
