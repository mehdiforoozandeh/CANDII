---
title: "D4: blind tracks, final model, blind pass, EIC board row"
parent: "[[closeout todo]]"
created: 2026-11-16
---

# D4: blind tracks, final model, blind pass, EIC board row

## 1. What the PI asked

- `peeling the onion.md`:100: "We also want to show in the paper that we are addressing the issue that plagued the EIC paper." and "we want to be able to fairly compare CANDI's results with published EIC predictions and results."
- `long-horizon notepad.md`:270 (ruling 21): "it trains on training plus validation tracks, as challenge entrants did, with the design and budgets frozen on validation first. The training-only score is reported beside it."
- `long-horizon notepad.md`:335-336 (rulings 24, 25): "If the final design fails the done rule on validation, the blind EIC tracks are still scored, once." and "The entry is seed 0, named in the freeze file before any blind track is read. The 3-seed spread is reported beside it. Seeds are not averaged."
- `long-horizon decisions.md`:72 (D9): "The headline is scored with the challenge's own metrics, against the organizers' corrected truth (Zenodo record 14736610, top level, not `old/`)". "Count metrics against the un-normalized single-end reprocessed tracks are secondary."
- `long-horizon decisions.md`:105 (D14 condition 2): the final design beats the strongest training-free baseline on each goal "measured on validation, and then once on test."
- `long-horizon notepad.md`:443 (ruling 34): a second, labelled AlphaGenome 2 comparison scores blind tracks that AlphaGenome 2 trained on.

Success criteria served: 1 (EIC data) and 4 (design tested). Audit rows: A6 (blind part), C2, C3, C5 (blind part), B13, B23, F3.

## 2. Question

Build the blind-side data, freeze and train the final EIC model, run the single ledgered blind pass, and place miniCANDI on the EIC board beside the published entries.

## 3. State today

- No blind track is rebuilt: the 51 blind tracks are in build priority 4 (722 rows: blind experiments and their controls, their FASTQ-level arms, 54 per-library rows, 459 per-lane rows; `results/t170/build_plan.md` L112-113). Nibi `tables/` holds only `eic_v1`; 3 blind-only controls are absent (`onion paper.md`:27).
- Checked 2026-11-16 (`results/closeout_D4/BUILD.md`): 721 of the 722 rows are unbuilt (row 1074 is built). The 45 blind ChIP-seq rows need 11 control experiments (priority-4 rows 499 to 543), not 3: the 3 counts EIC control tracks. PI ruling 2026-11-16: build all 722 rows. Submit script: `slurm/onion_data/closeout_d4_submit.sh` (blind base rows and their controls first).
- Nibi `eic/board/` holds `verify_published.tsv` and `verify_published_measures.tsv`, no `final_board.tsv` (read 2026-11-16).
- No freeze file, no final model, no `BLIND_PASSES.tsv` entry (audit row C3).
- The scorer port reproduces the published board: all 25 ranks match (`results/t166/BOARD_REPRO.md`; notepad L340).

## 4. Prior leads

- `onion paper.md`:36: the blind set is 100% paired-end, 100% HiSeq 4000 and 94% from one lab; 73.1% of training tracks have reads of 36 bp or shorter. Prediction: the blind pass tests measurement shift that the designed views imitate only in part.
- `long-horizon notepad.md`:452-453: AlphaGenome 2 trained on 36 of 51 blind tracks; the fair blind set is 15 tracks in 11 cells, all broad marks. Prediction: the fair blind comparison speaks for H3K27me3 and H3K9me3 only.
- `results/final_round/FINAL_ROUND_EQUAL.md`: on validation (screening tier) every architecture's composite (0.335 to 0.533) exceeds the position average (0.197). No per-goal margin against 2 x the seed range exists (audit row F3).

## 5. Work

1. PI decision first: build only the blind subset of priority 4 (the 51 blind base rows and the 3 blind-only controls), or all 722 rows. E4 needs the per-library and per-lane rows; the board does not.
2. Build the chosen rows. PI ruling 2026-11-16/17: blind only (batch 1: 62 blind rows and the 10 outage reruns), on Fir (Nibi refuses every /scratch write since the soft-quota grace ran out; checked 2026-11-16T22:47Z). Fir script `slurm/onion_data/closeout_d4_submit_fir.sh` (kit COD4b) after the Globus copies of `results/closeout_D4b/FIR_PORT.md` section 3. Then collect, and build the blind tables and `evalset_blind.json`. Leak check: no blind row readable in training or validation mode.
3. Fix the platform id of blind rows (D1) before training, so C' of a blind target uses a training-seen vocabulary.
4. Write the freeze file: the M1 design, budgets, dataset and evaluation-set versions, seed 0 as the board entry.
5. Train the final model on training plus validation tracks, 3 seeds; train the training-only model beside it, 3 seeds. Fir CPU.
6. Check D14 condition 2 on validation: per goal, margin over the strongest training-free baseline against 2 x the seed range.
7. One blind pass, written to `BLIND_PASSES.tsv`: count metrics (secondary table, D9); signal from the predicted NB (E2) scored with the EIC scorer against the corrected truth; the AlphaGenome 2 fair and labelled blind comparisons (ruling 34).
8. Write `final_board.tsv` with the miniCANDI row (seed 0; spread beside it) among the 25 published entries.

## 6. How the result answers the question

Acceptance checks:
- `BLIND_PASSES.tsv` holds exactly one pass line, written after the freeze file's timestamp, or a second line with a stated reason.
- `final_board.tsv` holds 26 rows; the 25 published ranks equal `BOARD_REPRO.md`.

| outcome | conclusion |
|---|---|
| miniCANDI ranks above the published median on the corrected truth | the 4-axis formulation transfers to the blind measurement shift at prototype scale |
| below the median, but count metrics beat depth-only scaling | correction works in counts; the signal route or the measurement shift limits the board rank |
| D14 condition 2 fails on validation | the run reports and still runs the blind pass once (ruling 24) |

## 7. Dependencies, cost, risks

- Depends on D1, D3 and M1.
- Cost: all of priority 4 was projected at 14,920 CPU-h and 76.3 h wall at 50 rows (`.orchestrate/plan.md` L1935). That number is CPU time used. Allocated CPU-h, which fair-share charges, is about 83,400 for the 721 rows plus about 10,400 for the 10 outage reruns (range 74,800 to 110,800 in all; `results/closeout_D4/BUILD.md`). The blind subset is smaller; compute it from `build_list.tsv` reads x the measured CPU-h per million reads (0.198 ChIP-seq, 0.136 DNase-seq, 0.310 ATAC-seq; `results/t183/DATASET_EIC_V1.md` L78). Final models: 6 trainings, about 11 to 15 CPU-h each (`ablation_specs/1_control_input.md` section 7), plus one blind scoring.
- Risk 1: a second look at the blind set breaks the protocol. Every blind read goes through the ledger.
- Risk 2: the blind platform and lab are unseen in training even after D1; a poor blind result can come from the unseen ids, not from the model.

## 8. Done when

`BLIND_PASSES.tsv` holds the single blind pass of the frozen final model, and `final_board.tsv` holds the miniCANDI row beside the 25 published entries, with the training-only score and the 3-seed spread reported.
