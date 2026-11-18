---
title: close-out todo list (peeling the onion)
kind: the list the foreman agent must complete before the work is called done
parent: "[[peeling the onion]]"
created: 2026-11-16
---

# Close-out todo list

The PI selected these items on 2026-11-16, after an audit of the instruction files against the work done (`cruxvault/results/onion_instruction_audit/AUDIT.md`, 99 rows). The foreman agent must take every item to "done". Each item has its own file, linked below. The foreman updates the Status column here at each change.

## Success criteria (PI, 2026-11-16, verbatim)

"in this peeling the onion work, the success criteria is briefly to:
- have the new augmented EIC data + counterfactuals etc.
- have a training objective well defined
- have a well defined and tested set of evals according to peeling the onion goals
- have the minicandi design settled and tested and ready for scaling in the next session."

## Left for the follow-up work (PI, 2026-11-16)

- Scaling the miniCANDI prototype (data, context, model size).
- The number and selection of training regions, and the evaluation-region count.
- The new extended data set.

## How each item is worked (PI order)

- One Opus agent per item, started by the foreman when the item's turn comes. That agent may start its own subagents.
- Every agent checks the work of its immediate subagents. The foreman checks every item agent.
- Each item first states its question, the earlier leads, and how its result answers the question. Then it runs.

## Items

Status values: not started, spec, running, read, done, blocked (with the reason).

### Data (success criterion 1)

| Item | File | Depends on | Status |
|---|---|---|---|
| Fill the lab, antibody and donor covariates | [[closeout D1 lab antibody donor covariates]] | none | running: table version eic_v1_2 merged 2026-11-17 (train-only category lists; check_dataset PASS); its 4 baselines equal the eic_v1 baselines (largest relative difference 3.3e-13); the 9 base-recipe runs on eic_v1_2 scored on 16 regions (9 of 9); owed: 256-region references and 256-region scores of the 9 runs (95.1 + 322.2 = 417.3 CPU-hours; [[ready queue for scaling]] entry 6) |
| Second experiments: table, evaluation set, cross-lab pairs | [[closeout D2 second experiments and cross-lab pairs]] | D1 | read: add-on table eic_v1_2x (71 second experiments, 71 other-experiment pairs) merged 2026-11-17; layers (Nibi 27678103) and all 5 baselines completed 2026-11-18; PI decisions 3 and 4: lab reported with donor, the 17 DNase-seq rows dropped; DNase-seq layer values not interpretable (15,201-read track, [[ready queue for scaling]] entry 20); models next session (entry 13) |
| Data set update with DNase-seq in the headline | [[closeout D3 dataset update with DNase-seq]] | D1 | blocked: work-folder snapshot and candidate table (2,812 rows; DNase-seq correction-scorable) merged 2026-11-17, all 10 Nibi verify jobs pass; the 4 row rebuilds (about 5,000 CPU-hours) need writable Nibi scratch, and their FASTQs are downloaded again (removed from scratch 2026-11-17); the PI chooses the table version of the final model; [[ready queue for scaling]] entry 15 |
| Blind tracks, final model, blind pass, EIC board row | [[closeout D4 blind tracks final model EIC board]] | D1, D3, M1 | blocked: blind-only scope ruled 2026-11-17 (51 blind rows + 11 controls + 10 reruns, about 22,700 allocated CPU-hours); Fir port merged, then the PI moved the build to Nibi; needs writable Nibi scratch; the PI runs the submit command; dry run of 2026-11-17 20:12 UTC: 72 rows to submit, 0 skipped; the final model and the board row are not built ([[ready queue for scaling]] entries 16, 17) |

### Evaluation (success criterion 3)

| Item | File | Depends on | Status |
|---|---|---|---|
| The 16-versus-256 region check and settled scores | [[closeout E1 256-region check and settled scores]] | none | running: the check failed its rule, so correction terms come from all 256 regions; rescoring of 62 run-seeds on Nibi since 2026-11-17 23:30Z (PI decision 1); 30 of 62 scored at 15:55Z; rules 2.1 (covariates used for correction), 2.2 (all views) and 2.6 (imputation weight 4) decided, 2.3 to 2.5 wait; signs agree with the 16-region scores in 48 of 50 cells of complete arms; final pass at 18:00Z ([[ready queue for scaling]] entry 5) |
| Signal p-value and peaks from the predicted NB; AlphaGenome 2 board row; signal-space score | [[closeout E2 p-value and peaks from the predicted NB]] | none | running: merged 2026-11-17 (partial, 1 of 9 runs scored in signal space); box-spread signal formula ruled by the PI; its model-side form merged after two reviews (Log 13:09Z); box-spread rescoring of the 9 runs under both dependence bounds not submitted (Fir outage; [[ready queue for scaling]] entry 7); predicted-control route is a headline row (PI) |
| Report the metrics that are computed and never reported | [[closeout E3 report the computed metrics]] | none | done 2026-11-16 for the screening scores ([[computed metrics report]]); rerun on settled scores after E1 |
| Complete the replicate ladder and the crossed mixed model | [[closeout E4 complete the replicate ladder]] | D2 | done for the ladder: layers 27678107 and report 27692829 completed; two-sided form with joint cluster bootstrap built and merged 2026-11-18 (PI decision 5); cell-type rung broad 0.00273 [0.0015, 0.00396], narrow 0.003547 [0.00198, 0.00511]; no other-experiment rung estimable; mixed model not run (by rule); DNase-seq rerun open ([[ready queue for scaling]] entries 20, 21) |
| Scorer extensions: foreign covariates at scoring, artifact variance by platform, antigen and scale, per-stratum values, robust variance ratio | [[closeout E5 scorer extensions]] | D1 for lab and antigen | running: code merged 2026-11-17 after review; foreign-covariate scoring of the 9 base seeds scored (36 of 36, 4 modes), not yet read; by-lab and by-antigen tables possible on eic_v1_2, not yet run |

### Training objective (success criterion 2)

| Item | File | Depends on | Status |
|---|---|---|---|
| Write the settled training objective as one document | [[closeout O1 training objective document]] | A1 (its loss and two-step ablations) | running: document merged 2026-11-17 ([[onion training objective]]); its open choices wait for the 256-region rescoring, the mix-and-match round and the covariate set |

### Ablations and model design (success criterion 4)

| Item | File | Depends on | Status |
|---|---|---|---|
| The nine named ablations on three architectures | [[closeout A1 nine named ablations]] | none | scored and read 2026-11-17 ([[ABLATIONS_READ]], `results/ablations_closeout/`); findings recorded |
| Basic controls on all three models (scrambled covariates, mask mode, injection form) | [[closeout A2 basic controls on three models]] | none | scored and read 2026-11-17 ([[ABLATIONS_READ]], `results/ablations_closeout/`); findings recorded |
| Counterfactual views on the two conv models | [[closeout A3 counterfactual views on conv models]] | none | scored and read 2026-11-17 ([[ABLATIONS_READ]], `results/ablations_closeout/`); findings recorded |
| Read-derived QC metrics on the input side | [[closeout A4 QC metrics on the input side]] | QC table fill | read: QC fill (79 of 79 tars) and collect (Nibi 27678089) done 2026-11-17; fill rates NSC and RSC 89.0%, fragment length 87.8%, NRF and PBC 37.8%, FRiP 0%; arm accepted by the PI (decision 7), next session ([[ready queue for scaling]] entry 11) |
| RNA-seq input | [[closeout A5 RNA-seq input]] | RNA fetch | not started: no configs, no eval subset, no submit script ([[ready queue for scaling]] entry 18) |
| Learned cell-type embedding | [[closeout A6 learned cell-type embedding]] | none | spec: configs, leak audit and zero-shot control merged 2026-11-17; not submitted; recounted at the measured scoring cost: 728.6 CPU-hours (was 498 at 23 per scoring; [[ready queue for scaling]] entry 12) |
| Resolution 1 to 125 bp | [[closeout A7 resolution]] | 1 bp region store | read: 1 bp recount (412 views), tables (412 measurements, 73 pairs) and variance by bin size done 2026-11-18; below 25 bp lie 0.927 (broad) and 0.901 (narrow) of the 1 bp variance against 0.960 for noise alone, about 0.1% reproducible; verify job 27692994 still running at 16:55Z ([[ready queue for scaling]] entry 22); full arm design next session (PI decision 6, entry 9) |
| Count transforms | [[closeout A8 count transforms]] | none | scored and read 2026-11-17 ([[ABLATIONS_READ]], `results/ablations_closeout/`); findings recorded (Anscombe arm void: code defect; 9 runs) |
| Final tuning round: matched-compute runs | [[closeout A9 matched-compute runs]] | none | running: 12 trainings pending on Fir (jobs 4345024 to 4345035, 0 CPU-hours; Fir outage until about 2026-11-20); their scoring is [[ready queue for scaling]] entry 8 |
| Pointwise MLP under the corrected composite | [[closeout A10 pointwise MLP corrected score]] | GPU | not started: no script ([[ready queue for scaling]] entry 19) |
| DNA fusion: how the DNA features join the count features | [[closeout A11 DNA fusion]] | none | scored and read 2026-11-17 ([[ABLATIONS_READ]], `results/ablations_closeout/`); findings recorded |
| Mix-and-match combinations of the single choices that helped, per model | [[closeout A13 mix-and-match combinations]] | A1, A2, A3, A8, A11, D1, E1 | spec: merged 2026-11-17 after review: 24 arms + 3 references on eic_v1_2, 256 regions; fixed-depth arms also scored at native depth; pool provisional until a 256-region qualification gate passes (5 single runs, 474 CPU-hours); round about 5,390 CPU-hours on Nibi (5,308 on Fir); not submitted; PI budget approval open; [[ready queue for scaling]] entry 10 |
| Minimum useful covariate set: ranked list, exclusion curve, removable fields | [[closeout A12 minimum useful covariate set]] | D1, A13 | spec: per-field drop switch, ranking tool and 51 configs merged 2026-11-17; not submitted; needs the final recipe from A13, the eic_v1_2 256-region references and the PI's choice of scope (1,927 to 17,900 CPU-hours) (PI, 2026-11-17: last step of the session and a deliverable); [[ready queue for scaling]] entry 14 |
| Settle the miniCANDI design: one document, ready for scaling | [[closeout M1 settled miniCANDI design]] | A1 to A13, E1 | done: [[settled miniCANDI design]] merged 2026-11-17 (31 rows, rules 2.1 to 2.6); rows 8, 20, 23 updated by the 256-region read of 2026-11-18 15:54Z; rows 13, 24, 25 wait for the final pass; architecture and data version are next-session PI choices |

### Records and close-out

| Item | File | Depends on | Status |
|---|---|---|---|
| Bring the records up to date (rulings, compute tallies, paper note) | [[closeout R1 records up to date]] | none | done for this run: rulings to 75, compute tally, artifact index; crux findings, questions and tasks of 2026-11-18 recorded; status page and handoff rewritten 2026-11-18 |
| Close-out deck and written report | [[closeout R2 close-out deck and report]] | all | done, refreshed at the final pass: deck published version 5 (https://claude.ai/artifact/Kp1yqCgAUpCBDo19Q6oL9J), report `presentations/onion-closeout-2026-11-18-report/report.pdf`, both verify --strict passed; rebuilt by the 18:00Z final-pass command |

## PI rulings of 2026-11-16 (answers to the item files' open decisions)

- Blind-track rebuild: all 722 remaining rows.
- New data set version (after the covariate fill): retrain the 3 base recipes and the final model only.
- Count transforms: all 9 on all 3 models.
- The 16-versus-256 pair-region plan is amended to the corrected correction terms before it runs.
- RNA-seq tracks may move from Nibi to Fir by Globus.
- No matched-compute runs for ablation arms.
- Mix-and-match combinations of winning settings are done in this work.
- Two-step fixed reference depth: one per assay class.
- adaLN-zero covariate injection: an extra arm.
- DNA embeddings: Globus transfer Nibi to Fir started 2026-11-16.
- DNA fusion is a separate ablation on the 3 models (item A11).

## PI rulings of 2026-11-17 (after the read of the 16 ablations)

- Fixed input depth on the U-Net: test it as a mix-and-match arm. Keep in mind that a required fixed input depth can be unrealistic in real-world deployment and inference; state this caveat wherever the arm is reported.
- Anscombe count transform: drop the arm. The 9 runs are void (zero gradient at the zero-initialized output head); record the defect, no retrain.
- Conv autoencoder with fixed input depth: accept 2 seeds; no rerun of seed 1.
- Covariates: include every covariate field that we can fill (the corrected tables are the default). The last step of the session, and a deliverable, is the minimum useful covariate set (item A12).

## PI rulings of 2026-11-17 (after the archive, and the 24-hour order)

- (i) Remove `t112_cf`, `onion/fastq` and `EIC_REPRO` from Nibi scratch: yes. Done 19:45Z, 81 of 81 targets, with `t112_cf/pseudoreps`, `t112_cf/fastqarms/genome` and `EIC_REPRO/003` kept.
- (ii) The QC fill and the 1 bp recount run before `onion/work` leaves scratch: yes. The QC fill was submitted 19:58Z; for the recount probe and the recount the PI said "You submit it" (probe 27658461, 20:10Z).
- (iii) Remove today's superseded code snapshots on Nibi `/project`: yes. 14 kits packed to `onion_kit_archive`; kit CO8 is the only current one.
- (iv) The 256-region rescoring at about 2,225 CPU-hours: yes.
- PI condition, verbatim: "please make absolutely sure before you remove something we don't need it anymore or it's easy to regenerate or access from the archive".
- 24-hour order, verbatim: "you have 24 hours to wrap up the peeling the onion work. plan wisely to cover everything remaining in the next 24 hours." Deadline 2026-11-18T20:00Z ([[final 24 hours plan]]).

Every PI ruling and directive from 2026-11-15 20:05 UTC on is entered with the PI's words and its consequence in [[long-horizon notepad]], rulings 50 to 75.

## Open PI decisions

**Update 2026-11-18 (records item).** The PI answered decisions 1 (Nibi), 2 (gate first, then the round), 3 (one model with the removal curve), 4 (lab with donor; drop the 17 DNase-seq rows), 6 (full design, next session), 7 (accepted) and 8 (two-sided ladder with the common-cell check) on 2026-11-17 (`PI decisions 2026-11-17.md`). Decision 5 (table version of the final model) moved to the next session (ruling of 2026-11-17 21:00Z). Still open: 9 (cell-type embedding at 728.6 CPU-hours), 10 (matched compute), 11 (Nibi scratch quota), the architecture for scaling, and the cancel of the Fir rescoring copies after the Nibi scores are on disk (PI action, agreed 2026-11-17 22:25Z). The list below is the state of 2026-11-17 20:25Z.

From the foreman's Log up to 2026-11-17T20:25Z (records item, second pass). Each one names the entry of [[ready queue for scaling]] that waits for it.

Settled since the first pass: removal of the archived folders (ruling (i); `onion/work` waits for the QC fill and the recount, ruling (ii)); the corrected cost of the rescoring (ruling (iv)); the stale kits (ruling (iii)); the signal formula (box spread, ruling 71).

Still open:
1. Which cluster runs the 256-region rescoring. The Fir jobs 4535347 to 4535923 stay queued; a Nibi run duplicates them unless the PI cancels them (entry 5).
2. Budget of the mix-and-match gate (474 CPU-hours) and round (about 5,390 on Nibi, 5,308 on Fir) (Log 11:10Z; entry 10).
3. Scope of the minimum-covariate-set search (1,927 to 17,901 CPU-hours) (Log 11:20Z; entry 14).
4. Second experiments: report lab together with donor (recommended); drop the 17 unbuilt DNase-seq rows (recommended) (Log 11:30Z; entries 4, 13).
5. Table version of the final model: eic_v1_2, or the DNase-seq candidate, which retrains the 9 base runs (Log 11:19Z; entry 15).
6. Resolution design: reduced (about 405 CPU-hours) or full (about 2,340) (Log 12:47Z; entry 9).
7. QC-on-input arm: accept the changes to the "QC in C" design (presence flags, leak guards) and its 404 CPU-hours (Log 12:29Z; entry 11).
8. Replicate ladder: adopt the two-sided subtraction; replace the telescoping acceptance check by the common-unit check; the lab-layer report (Log 13:00Z; entry 3).
9. Learned cell-type embedding at the corrected 728.6 CPU-hours (entry 12).
10. Matched compute: confirm or reverse the per-rung narrowing of notepad L289 (item A9; entry 8).
11. Nibi scratch quota: the PI's figure is a soft quota of 20 TB; the measurement is 1,024 GiB soft and 20 TiB hard (notepad, corrected numbers, item 7). No resolution in the Log.
