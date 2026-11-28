---
title: miniCANDI on read-start counts (task t392)
kind: PI instruction for a long run
created: 2026-11-23
approved_by: Mehdi (PI), 2026-11-23, through the foreman session
foreman: Claude session local_b5c55e32-4e2a-4edc-b4b7-6f0134e8bc43 ("Open sessions on project")
branch: implementation/t392-read-start-minicandi (cut from origin/exp/t370-minicandi-recipe)
---

# miniCANDI on read-start counts

## 1. The problem

miniCANDI predicts **overlap counts**. The training store `eic_v1_2` adds +1 to every 25 bp bin that a read's span touches, i.e., bins floor(start/25) to floor(end/25) inclusive (`tools/t112/records.py:42`, `COUNT_RULE`). One read therefore adds to 2 to 5 neighboring bins.

Overlap counts cause three problems.

1. **They cannot convert to other signal formats.** A fragment pileup, i.e., the count of whole DNA fragments that cover each bin, needs each read's start and strand. The overlap rule discards both. The conversion from overlap counts to any other format is not invertible.
2. **The EIC p-value step expects a fragment pileup.** The pval EIC comparison (task t390, PR 64) measured the conversion loss against the exact MACS2 formula on 3 blind experiments. The evidence is `cruxvault/results/t390/postpile/POSTPILE.md` on branch `implementation/t390-eic-pvalue-entry`.

   | input to the p-value step | nMSE H3K4me3 | nMSE H3K27me3 | nMSE DNase |
   |---|---|---|---|
   | 25 bp overlap counts (today) | 0.0172 | 0.2550 | 0.0212 |
   | overlap counts spread over fragment length d | 0.0168 | 0.2314 | 0.0207 |
   | stranded read starts extended by d | 0.0028 | 0.0746 | n/a |
   | unstranded read starts spread over d | n/a | n/a | 0.0018 |
   | true fragment pileup | 0.0021 | 0.0616 | 0.00003 |

   No post-prediction spread of overlap counts closes the gap. Read starts recover 92 to 95% of it.
3. **The NB loss is a composite likelihood.** The loss adds up per-bin NB log-likelihoods, i.e., it treats the bins as independent. Overlap bins share reads, so that assumption is false. The per-bin marginals stay consistent, so today's per-bin metrics are valid. Joint uncertainty, sampled tracks and evidence weighting are wrong. Each read counts in the loss once per bin it touches.

## 2. The solution

Predict **25 bp read-start counts** with the same NB head.

- **ChIP:** stranded. Each ChIP track becomes two channels, plus-strand starts and minus-strand starts.
- **DNase:** unstranded. A read start is a cut site, so it is already the signal.

Each read lands in exactly one bin, so the per-bin independence assumption is close to true. Every other format is a fixed linear map of the read starts.

1. **Overlap counts** = read starts, convolved with a box of the read span (the `COUNT_RULE` above), per strand, then summed over strands.
2. **Fragment pileup** = each strand's read starts, shifted and extended by the fragment length d in that strand's direction, then summed.
3. **Signal p-value** = the MACS2 formula applied to the fragment pileup and its control (the PR 64 entry pipeline).

A linear map of predicted means is exact for the mean. For a distribution, a sum of NBs is approximated by one NB with the same mean and variance (moment matching): r = μ² / (var − μ), and a Poisson when var ≤ μ. With unequal p this is an approximation, not exact.

## 3. Phase 1 (approved 2026-11-23): side by side

**Goal.** Compare two versions of the full-combination miniCANDI on our current metrics.

- **Read-count model (reference).** The full combination: read-disjoint pairs, average input and Pearson-residual input. Config: `tools/minicandi/configs/recipe/combo/rung4/halves_mix_avg_pearson.json` (`windows_seen` 80000, `param_budget` 1e6, `train_halves` "pairs", `average_input` true, `count_transform` "pearson"). Its 3 seeds are already trained and scored on 16 and 256 regions in the t370 run. **Reuse those results; do not retrain it.**
- **Read-start model (new).** The same config, the same seeds (0, 1, 2), windows, budget and evaluation regions. The only change is the training target, which becomes read-start counts.

**Fair scoring.** Score both models against the same truth, today's overlap-count truth in `eic_v1_2`. Convert the read-start model's predictions to overlap counts with map (1) in section 2, means by convolution and the NB by moment matching. Put the conversion between `predict` and the metrics. Do not change the scorers. Also report the read-start model against read-start truth, as a sanity check only.

**Metrics.** First, the headline metrics: imputation, correction and calibration. Second, the deviance metrics from PR 65 (branch `exp/t371-deviance-evals`, doc `EVAL.md`), once PR 65 is merged into `exp/t370-minicandi-recipe`. Quote the seed range with every number (`AGENTS.md` §7.2).

**Work items.**

1. **Code.** Add read-start recounting with a strand channel (`tools/onion_data/recount.py` writes one unstranded array today). Carry the strand channel through the store, the input contract (the model has 36 fixed tracks) and the average-signal input (`average_h5`). Make `tools/minicandi/res_dataset.py` accept `average_h5`; today it refuses it. Write the conversion step and its tests. Tests come first, against this document (the crux-tests agent).
2. **Data.** Recall the read files from Nibi nearline. Then recount all 2,740 measurements of `eic_v1_2` in every view: base, crop, single-end, depth, MAPQ-0 and no-dedup. Build the read-start store.
   - **Gate:** convolving the *observed* read starts with map (1) must reproduce the `eic_v1_2` overlap counts **exactly**, bin for bin, on every measurement. A single differing bin stops the run.
3. **Train** the read-start model, 3 seeds.
4. **Score on 16 regions:** both models, headline and deviance, side by side.
5. **Stop and report.** Send the 16-region table to the foreman. The 256-region scoring (about 900 CPU-h with one shared prediction pass) runs only on the PI's word, which the foreman relays.

**Budget.** About 470 CPU-h through step 4: recount about 180, training about 163 (measured on the read-count model) and 16-region scoring about 127. Wall time is about 3 to 4 days, including 1 to 2 days of code. If the run passes 600 CPU-h before step 5, stop and report to the foreman.

**Deliverable.** `cruxvault/results/t392/side_by_side/SIDE_BY_SIDE.md`. It holds one table with one row per model and one column per metric, the seed ranges, one figure, and a one-paragraph reading. The folder follows the `AGENTS.md` §4.11 contract: `runall.sh`, `MANIFEST.tsv`, no file over 1 MB. Record the result as a crux finding with that file as evidence.

**What we expect** (the foreman's opinion, recorded before any result):

- Imputation ties within seed range, or the read-start model loses slightly, because read-start counts are sparser.
- Correction ties.
- Calibration and deviance move a little in either direction. There is no confident sign.

A tie is a good outcome. It means one read-start model can serve our board and the EIC boards.

## 4. Phase 2 (reserved; do not start)

Phase 2 starts only on the PI's word, relayed by the foreman, after Phase 1 is reported.

**Goal.** Starting from predicted read-start NB tracks, derive every signal format correctly: overlap counts, fragment pileup, fold enrichment and signal p-value.

Open questions for Phase 2:

1. How to estimate the fragment length d per experiment.
2. Whether moment matching is accurate enough, or the convolution of NBs must be computed exactly (by sampling or by the characteristic function).
3. How to propagate uncertainty into the p-value.
4. How to validate each transform. One way is against the observed formats, on held-out data and on the PR 64 entry pipeline, which already reproduces the EIC boards exactly.

## 5. How the run is managed

- **One voice to the PI.** The run reports only to the foreman session, local_b5c55e32-4e2a-4edc-b4b7-6f0134e8bc43, by SendMessage. It never addresses the PI directly. The foreman relays the PI's instructions, quoted in the PI's words.
- **Status.** At each milestone, send the foreman a table: work item, % done, what is left, blocker, ETA, cost so far in CPU-h.
- **Questions.** Phrase each question so it can be answered in one line, and give a recommendation. Keep working on anything that is not blocked while a question waits.
- **Orchestration.** One umbrella branch, `implementation/t392-read-start-minicandi`, with a draft PR into `exp/t370-minicandi-recipe`. Subagent worktrees are cut from the umbrella branch and merged back into it.
- **Merges.** Merges into `exp/t370-minicandi-recipe` or `main` belong to the PI, done through the foreman.
