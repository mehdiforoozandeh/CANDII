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
| Fill the lab, antibody and donor covariates | [[closeout D1 lab antibody donor covariates]] | none | not started |
| Second experiments: table, evaluation set, cross-lab pairs | [[closeout D2 second experiments and cross-lab pairs]] | D1 | not started |
| Data set update with DNase-seq in the headline | [[closeout D3 dataset update with DNase-seq]] | D1 | not started |
| Blind tracks, final model, blind pass, EIC board row | [[closeout D4 blind tracks final model EIC board]] | D1, D3, M1 | not started |

### Evaluation (success criterion 3)

| Item | File | Depends on | Status |
|---|---|---|---|
| The 16-versus-256 region check and settled scores | [[closeout E1 256-region check and settled scores]] | none | not started |
| Signal p-value and peaks from the predicted NB; AlphaGenome 2 board row; signal-space score | [[closeout E2 p-value and peaks from the predicted NB]] | none | not started |
| Report the metrics that are computed and never reported | [[closeout E3 report the computed metrics]] | none | not started |
| Complete the replicate ladder and the crossed mixed model | [[closeout E4 complete the replicate ladder]] | D2 | not started |

### Training objective (success criterion 2)

| Item | File | Depends on | Status |
|---|---|---|---|
| Write the settled training objective as one document | [[closeout O1 training objective document]] | A9 (losses), A4 (two-step) | not started |

### Ablations and model design (success criterion 4)

| Item | File | Depends on | Status |
|---|---|---|---|
| The nine named ablations on three architectures | [[closeout A1 nine named ablations]] | none | running (126 runs submitted 2026-11-16; 3 arms held or blocked) |
| Basic controls on all three models (scrambled covariates, mask mode, injection form) | [[closeout A2 basic controls on three models]] | none | not started |
| Counterfactual views on the two conv models | [[closeout A3 counterfactual views on conv models]] | none | not started |
| Read-derived QC metrics on the input side | [[closeout A4 QC metrics on the input side]] | QC table fill | not started |
| RNA-seq input | [[closeout A5 RNA-seq input]] | RNA fetch | not started |
| Learned cell-type embedding | [[closeout A6 learned cell-type embedding]] | none | not started |
| Resolution 1 to 125 bp | [[closeout A7 resolution]] | 1 bp region store | not started |
| Count transforms | [[closeout A8 count transforms]] | none | not started |
| Final tuning round: matched-compute runs | [[closeout A9 matched-compute runs]] | none | running (12 trainings pending on Fir) |
| Pointwise MLP under the corrected composite | [[closeout A10 pointwise MLP corrected score]] | GPU | not started |
| Settle the miniCANDI design: one document, ready for scaling | [[closeout M1 settled miniCANDI design]] | A1 to A10, E1 | not started |

### Records and close-out

| Item | File | Depends on | Status |
|---|---|---|---|
| Bring the records up to date (rulings, compute tallies, paper note) | [[closeout R1 records up to date]] | none | not started |
| Close-out deck and written report | [[closeout R2 close-out deck and report]] | all | not started |

## Open PI decisions

Listed in the foreman's chat block "What I need from you"; copied here when answered.
