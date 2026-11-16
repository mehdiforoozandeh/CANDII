---
title: "D3: dataset update with DNase-seq in the headline"
parent: "[[closeout todo]]"
created: 2026-11-16
---

# D3: dataset update with DNase-seq in the headline

## 1. What the PI asked

- `long-horizon notepad.md`:463 (ruling 38, PI's words): "freeze whatever we can get in the next 6h and then send out reliable watchdogs to wait for the remaining runs and as soon as they are finished you update the dataset with notes".
- `peeling the onion.md`:105: "we are updating our EIC dataset -- to train the new CANDI".
- `long-horizon decisions.md`:66 (D11b): "**Accessibility-only (second table).** Only DNase-seq or ATAC-seq is given, and the other assays are imputed. Cells with no accessibility track are left out of this regime."
- `long-horizon decisions.md`:70 (D8): the EIC model "trains only on the EIC training experiments".

Success criteria served: 1 (augmented EIC data) and 3 (evals). Audit rows: A11, A12, B14 (the accessibility-only table).

## 2. Question

Build the next tables version with every tail row that has verified, so that DNase-seq becomes imputation-scorable and the accessibility-only regime covers more validation targets. Rebuild the evaluation set on it, and say which comparisons move to the new version.

## 3. State today

- `eic_v1` (frozen 2026-11-13 14:30 UTC) holds 249 of 267 training and 42 of 45 validation tracks; DNase-seq is not scorable (1 of 3 validation tracks); 22 of 42 validation targets have no accessibility-only item (`long-horizon notepad.md`:470-473).
- Tail watch, Nibi `ONION/eic/tail/TAIL_STATUS.tsv` (checked 2026-11-16T14:49Z, read by this agent), 28 rows: 17 verified (rows 10, 37, 46, 52, 57, 63, 84, 124, 273, 337, 381, 388, 389, 390, 391, 394, 395). 11 not verified: row 56 (validation DNase-seq, C50; signal pass 27124285_56 PENDING, 7-day limit), and 10 training DNase-seq rows: 7 NODE_FAIL in the Nibi outage (113, 211, 216, 232, 384, 385, 387), 1 TIMEOUT (280), 2 FAILED (132 Java heap, 396 other).
- Validation DNase-seq: row 57 (C34) is now verified, so 2 of 3 validation tracks exist; the third (row 56) needs its signal pass.
- No update version is built: Nibi `tables/` holds only `eic_v1` (read 2026-11-16). Watch job 27483519 is PENDING.

## 4. Prior leads

- Scorable-set rule (notepad L51): an assay is imputation-scorable with at least 3 validation tracks from at least 3 distinct cells. Prediction: DNase-seq enters the headline only when row 56 verifies.
- `results/t305/LAYERS_REAL.md`: DNase-seq has the largest cell-type layer (23%, interval [0.0117, 0.0176]). Prediction: adding DNase-seq to the headline raises the weight of cell-type-specific signal in the imputation term.
- Plan Amendment 6 (`.orchestrate/plan.md` L754-840, chunk D6u): defines the update procedure, the `--superset-of` check and `versions.tsv`. Use it as written.

## 5. Work

1. Resubmit each unverified training tail row once, from kit KF4, with `ONION_TIME=7-00:00:00` and `ONION_MEM=192000M` (Amendment 6 B.5; notepad L514-515). The 7 NODE_FAIL rows failed on the outage, not on their own cause.
2. Wait for row 56's signal pass. If its counts are verified and only the signal is missing, record whether the count-space selection metric can use it (the signal is needed for signal-space scores only).
3. Run chunk D6u: build `eic_v1_<k>` with every verified row plus the D1 covariate columns (one version for both, so models retrain once), `check_dataset.py --superset-of` (relaxed for the D1 columns, see D1 risk 3), `CHANGES_<k>.md`, `versions.tsv`.
4. Rebuild the evaluation set on the new version (256 regions, same seed), and the 16-region screening set. Record both hashes.
5. Rebuild the baselines on the new evaluation set.
6. Submit the second-experiment rows whose DNase-seq partner is now verified (D2 step 1).
7. Write the notes: what entered, what is still absent with its cause, and the new scorable list.

## 6. How the result answers the question

Acceptance checks:
- `check_dataset.py --tables eic_v1_<k> --superset-of eic_v1` exits 0 (with only the D1 columns allowed to change).
- `versions.tsv` holds one line for `eic_v1_<k>` with its tables hash, eval-set hash and change note.
- The new `scorable.tsv` lists DNase-seq if row 56 verified; else the file states why not.
- The count of validation targets without an accessibility-only item is below 22, or the note explains why it is not.

## 7. Dependencies, cost, risks

- Depends on D1 (the covariate columns go into the same version). D2 and D4 depend on it.
- Cost: about 250 CPU-h + 19 GPU-slice-h + about 32 CPU-h per update (`.orchestrate/plan.md` L846, Amendment 6), plus the resubmitted DNase-seq rows (one row of 753 M reads ran over 1 day in `bwa sampe`; `results/t183/TAIL.md`).
- Risk 1: every model comparison is valid within one dataset version and one evaluation-set version only (notepad L465). The nine ablations (A1) and the final round (A9) run on `eic_v1`. The PI must say which comparisons are rerun on the new version (see M1).
- Risk 2: the slow DNase-seq paired-end alignment can need more than 3 days; row 56's signal pass has a 7-day limit.

## 8. Done when

`eic_v1_<k>` and its evaluation sets pass their checks, the change note lists every tail row as in or absent with its cause, and the scorable list states whether DNase-seq is in the headline.
