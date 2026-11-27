# State

Rewritten, never appended. Cap: 5 KB.

## Goal now
Find and test a fair count-to-p-value route that puts miniCANDI on the EIC board (q90, task t390). Branch implementation/t390-eic-pvalue-entry, draft PR 64, never merged.

## Horizon
Exploratory, open. First measured result delivered 2026-11-22: ladder probe on 3 blind experiments.

## Known (sources in LOG.md and cruxvault/results/t390/ladder_probe/LADDER_PROBE.md)
- Organizers' side is exact: QN by Max's rule equals the MOESM4 truth on every base; the 1 bp MACS2 formula matches our pipeline track (nMSE <= 1.1e-4). 3 of 3.
- The truth's QN used the pseudoreplicate-overlap peak set. With it, our DNase route equals the truth exactly.
- 25 bp overlap counts cost nMSE 0.017 (H3K4me3), 0.255 (H3K27me3), 0.021 (DNase) against the exact formula.
- Our ChIP reprocessing differs from Jacob's before the p-value step (nMSE 0.013, 0.127). Likely cause: control reads (treatment matches; gap sits where control sets lambda). Not closable exactly without his control list (results/t390/chip_gap/CHIP_GAP.md).
- Observed-read count route + QN ranks 1-2 of 26 on MOESM4 measures: an oracle bound on conversion cost only.
- An entry must use the normalization of the truth it is scored against (QN rungs rank 14-26 on the SE truth).
- miniCANDI blind predictions do not exist: blind-row build t184 is a PI gate in the recipe run.

## In flight
- Validation dry run READY on Fir: `run_entry.sh deploy fir`, then `BOARD=validation run_entry.sh submit <pred> /scratch/mforooz/eicx/val/cov.tsv /scratch/mforooz/eicx/val/<run>`; waits on recipe-session validation predictions. Span fix DONE (SPAN_EXTRA 25).
- Entry pipeline DONE (536312a2): `run_entry.sh deploy` then `submit <pred> <cov.tsv> <out> C19M22 C40M17 C12M02`; 12/12 checks bit-exact. Waits on predict.py output from the recipe session.
- Probe 2 DONE (results/t390/ladder_probe2/PROBE2.md): fragment pileup nMSE 0.0021/0.062/0.00003; half-read peaks within 8.5% of truth peaks.

## Inherited from the recipe run (PI, 2026-11-27, via foreman)
Handoff: cruxvault/INSTRUCTIONS/t370_recipe/run_state/HANDOFF_CLOSEOUT.md on origin/exp/t370-minicandi-recipe (3951e62d). Never resubmit any job it lists.
- Avocado chain, Fir: genome 6291946_[14-22] PENDING (Priority) at 06:13 UTC, then predict 6291947, bigwig 6291948. On 6291948 done: score the Avocado bigwigs on the 45 validation experiments beside miniCANDI. Fallback if JobArrayTaskLimit with none running: scontrol update JobId=6291946 ArrayTaskThrottle=13 (PI-approved).
- Nibi blind retest: signal 28737212 (48) and 28737239 (3), all 51 RUNNING since 05:28. On success: Globus the rows to Fir, price the blind predict (validation cost 649.8 CPU-h for 43), send the submit to the foreman for the PI's yes. On failure: propose the crash-log-to-/project change via the foreman (PI's yes).
- Laptop watcher: scratchpad/handoff_watch.sh, every 15 min, exits on the first event.

## Next
- DONE 2026-11-25: all 45 validation experiments scored (commit 6714498a, cruxvault/results/t390/val_boards_final45/BOARDS.md). Depth cap (primary) 1 of 11, mean combined rank 3.22; averaged then capped 1 of 11, 3.20; raw 2 of 11, 4.31. Spend 66.73 of 80 CPU-h (sacct eicx_, 537 jobs). No job queued; watchers stopped.
- Record the finding on q90; apply the depth cap to the blind predictions when t184 delivers them.
0. t184 batch 1 submitted on Nibi 2026-11-23 08:01 UTC by the recipe session (PI-approved there): ChIP/ATAC 28629674-28629678, DNase 28629699-28629701; outputs /scratch/mforooz/onion/{work,measurements}; limits 3 d / 7 d. Never resubmit (not ours). Blind predictions follow the final architecture at original covariates; recipe session sends paths.
1. Read probe 2; record findings; choose the fair peak rule for the count route.
2. 48-experiment batch: held by the PI (D8).
3. Score the rungs against the MOESM2 (original PE) truth too, as D1 asks; probe 1 used the unnormalized SE truth instead. Max's pe cache: /scratch/maxwl/epi_imputation/s04/cache/pe.
4. Fair map for model outputs (training and validation data only, D2); needs blind predictions (t184).

## PI request
Ping the PI (push notification) when probe 2, MOESM2 or miniCANDI scores land. Foreman session local_b5c55e32-4e2a-4edc-b4b7-6f0134e8bc43 ("Open sessions on project", 2026-11-23): send it every PI-facing item and a 2-3 line status at each milestone. Overnight 2026-11-23 (relayed by foreman): compute by workers' judgment, report each job id to the foreman; 48-experiment batch and Jacob message stay held (D8); do not stop for questions, take the safe default and report it.

## Open asks to the PI
- None. D9 (PI yes, 2026-11-23): truth copied and verified (job 28706885, 51/51, SHA-256 match); a read-start-count arm is built in a new session, via the foreman.
