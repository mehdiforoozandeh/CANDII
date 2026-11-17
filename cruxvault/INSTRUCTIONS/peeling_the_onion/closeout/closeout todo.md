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
| Fill the lab, antibody and donor covariates | [[closeout D1 lab antibody donor covariates]] | none | running: table version eic_v1_2 merged 2026-11-17 (train-only category lists; check_dataset PASS); its 4 baselines equal the eic_v1 baselines (largest relative difference 3.3e-13); the 9 base-recipe runs on eic_v1_2 scored on 16 regions (9 of 9); owed: 256-region references and 256-region scores of the 9 runs (about 320 CPU-hours) |
| Second experiments: table, evaluation set, cross-lab pairs | [[closeout D2 second experiments and cross-lab pairs]] | D1 | spec: add-on table eic_v1_2x (71 second experiments, 71 other-experiment pairs) built and checked on Nibi, merged 2026-11-17; scoring stages not submitted; 2 PI decisions open (report lab with donor; drop the 17 unbuilt DNase-seq rows) |
| Data set update with DNase-seq in the headline | [[closeout D3 dataset update with DNase-seq]] | D1 | blocked: work-folder snapshot and candidate table (2,812 rows; DNase-seq correction-scorable) merged 2026-11-17, all 10 Nibi verify jobs pass; the 4 row rebuilds (about 5,000 CPU-hours) need writable Nibi scratch; the PI chooses the table version of the final model |
| Blind tracks, final model, blind pass, EIC board row | [[closeout D4 blind tracks final model EIC board]] | D1, D3, M1 | blocked: blind-only scope ruled 2026-11-17 (51 blind rows + 11 controls + 10 reruns, about 22,700 allocated CPU-hours); Fir port merged, then the PI moved the build to Nibi; needs writable Nibi scratch; the PI runs the submit command |

### Evaluation (success criterion 3)

| Item | File | Depends on | Status |
|---|---|---|---|
| The 16-versus-256 region check and settled scores | [[closeout E1 256-region check and settled scores]] | none | read: the 16-versus-256 check failed its pre-written rule, so correction terms are scored on all 256 regions from now on; rescoring of 7 decisive arms (62 run-seeds, about 2,225 CPU-hours) blocked: queued on Fir in the outage, moved to Nibi by the PI, Nibi scratch refuses writes |
| Signal p-value and peaks from the predicted NB; AlphaGenome 2 board row; signal-space score | [[closeout E2 p-value and peaks from the predicted NB]] | none | running: merged 2026-11-17 (partial, 1 of 9 runs scored in signal space); box-spread signal formula ruled by the PI and merged for observed counts; its model-side form sent back by review; predicted-control route is a headline row (PI) |
| Report the metrics that are computed and never reported | [[closeout E3 report the computed metrics]] | none | done 2026-11-16 for the screening scores ([[computed metrics report]]); rerun on settled scores after E1 |
| Complete the replicate ladder and the crossed mixed model | [[closeout E4 complete the replicate ladder]] | D2 | spec (item agent started 2026-11-17 12:01 UTC) |
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
| Read-derived QC metrics on the input side | [[closeout A4 QC metrics on the input side]] | QC table fill | spec (item agent started 2026-11-17 12:01 UTC; the QC-fill pass reads onion/work on Nibi scratch, which is being archived) |
| RNA-seq input | [[closeout A5 RNA-seq input]] | RNA fetch | not started |
| Learned cell-type embedding | [[closeout A6 learned cell-type embedding]] | none | spec: configs, leak audit and zero-shot control merged 2026-11-17; not submitted (about 498 CPU-hours at the old cost of 23 CPU-hours per scoring; to be recounted) |
| Resolution 1 to 125 bp | [[closeout A7 resolution]] | 1 bp region store | spec (item agent started 2026-11-17 12:01 UTC; the recount pass reads onion/work on Nibi scratch, which is being archived) |
| Count transforms | [[closeout A8 count transforms]] | none | scored and read 2026-11-17 ([[ABLATIONS_READ]], `results/ablations_closeout/`); findings recorded (Anscombe arm void: code defect; 9 runs) |
| Final tuning round: matched-compute runs | [[closeout A9 matched-compute runs]] | none | running: 12 trainings pending on Fir (jobs 4345024 to 4345035, 0 CPU-hours; Fir outage until about 2026-11-20) |
| Pointwise MLP under the corrected composite | [[closeout A10 pointwise MLP corrected score]] | GPU | not started |
| DNA fusion: how the DNA features join the count features | [[closeout A11 DNA fusion]] | none | scored and read 2026-11-17 ([[ABLATIONS_READ]], `results/ablations_closeout/`); findings recorded |
| Mix-and-match combinations of the single choices that helped, per model | [[closeout A13 mix-and-match combinations]] | A1, A2, A3, A8, A11, D1, E1 | spec: merged 2026-11-17 after review: 24 arms + 3 references on eic_v1_2, 256 regions; fixed-depth arms also scored at native depth; pool provisional until a 256-region qualification gate passes (5 single runs, 474 CPU-hours); round about 5,390 CPU-hours on Nibi (5,308 on Fir); not submitted; PI budget approval open |
| Minimum useful covariate set: ranked list, exclusion curve, removable fields | [[closeout A12 minimum useful covariate set]] | D1, A13 | spec: per-field drop switch, ranking tool and 51 configs merged 2026-11-17; not submitted; needs the final recipe from A13, the eic_v1_2 256-region references and the PI's choice of scope (1,927 to 17,900 CPU-hours) (PI, 2026-11-17: last step of the session and a deliverable) |
| Settle the miniCANDI design: one document, ready for scaling | [[closeout M1 settled miniCANDI design]] | A1 to A13, E1 | not started |

### Records and close-out

| Item | File | Depends on | Status |
|---|---|---|---|
| Bring the records up to date (rulings, compute tallies, paper note) | [[closeout R1 records up to date]] | none | running: rulings 50 to 72, compute tally and corrected numbers entered in the notepad 2026-11-17; paper note partly updated; artifact index and crux validate not done |
| Close-out deck and written report | [[closeout R2 close-out deck and report]] | all | not started |

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

Every PI ruling and directive from 2026-11-15 20:05 UTC on (including the later rulings of 2026-11-17: rescoring and blind build to Nibi, nearline archive, box-spread signal formula, overnight order) is entered with the PI's words and its consequence in [[long-horizon notepad]], rulings 50 to 72.

## Open PI decisions

From the foreman's Log up to 2026-11-17T12:01Z (records item, 2026-11-17):
- Remove the four archived folders from Nibi scratch when 23 of 23 tars are verified (Log 12:20Z; night rule 14:00Z).
- Budget of the mix-and-match round (about 5,390 CPU-hours) and of its gate (474 CPU-hours) (Log 11:10Z).
- Scope of the minimum-covariate-set search (1,927 to 17,900 CPU-hours) (Log 11:20Z).
- Second experiments: report lab together with donor (recommended); drop the 17 unbuilt DNase-seq rows (recommended) (Log 11:30Z).
- Table version of the final model: eic_v1_2, or the DNase-seq candidate, which retrains the 9 base runs (Log 11:19Z).
- The corrected cost of the queued 256-region rescoring: about 2,225 CPU-hours, not about 1,450 (Log 11:04Z).
