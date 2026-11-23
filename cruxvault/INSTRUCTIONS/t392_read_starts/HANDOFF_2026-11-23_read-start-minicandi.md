# Handoff: miniCANDI on read-start counts (task t392), long run

## Goal

Train the full-combination miniCANDI on 25 bp read-start counts (stranded for ChIP, unstranded for DNase). Score it side by side with the existing read-count version on the headline metrics and the deviance metrics, on 16 regions. Report the result to the foreman session. Phase 2 is reserved.

## Read these first, in order

1. `/Users/mforooz/Desktop/research/libbrechteam@sfu/CANDII/.claude/worktrees/readstart/cruxvault/INSTRUCTIONS/t392_read_starts/INSTRUCTIONS.md`. This is the PI's instruction: problem, solution, Phase 1 plan, budget, gates and deliverable.
2. `CLAUDE.md` and `AGENTS.md` §3, §4, §7 in the worktree. They give the repo rules, the invariants and the quoting rules.
3. `DATA.md` and `STORE.md`. They give the h5 schema, the input contract and the store layout that the strand channel must extend.
4. `.longrun/STATE.md` and `.longrun/DECISIONS.md` in the worktree. These belong to the t370 recipe run. Read them for context only; never edit them.
5. `cruxvault/results/t390/postpile/POSTPILE.md` on branch `origin/implementation/t390-eic-pvalue-entry`. It holds the evidence for why read starts matter.
6. `EVAL.md` on branch `origin/exp/t371-deviance-evals`. It defines the deviance metrics.

## Settled decisions (do not reopen)

- 2026-11-23, PI: Run the read-start version, staged. The run scores on 16 regions, then stops. 256 regions run only on the PI's word. Reason: about 470 CPU-h buys the answer to whether 256 is worth about 900 more.
- 2026-11-23, PI: Score both models against today's overlap-count truth. Read-start predictions are converted by box convolution, and the NB is moment-matched. Reason: the conversion from read starts to overlap counts is exact, and the reverse is impossible.
- 2026-11-23, PI: The read-count reference is the existing full combination (`tools/minicandi/configs/recipe/combo/rung4/halves_mix_avg_pearson.json`, seeds 0–2), reused, not retrained.
- 2026-11-23, PI: ChIP is stranded and DNase is unstranded. Reason: unstranded starts were worse than overlap counts on ChIP (nMSE 0.0614 vs 0.0172 on H3K4me3).
- 2026-11-23, PI: All communication goes through the foreman session, local_b5c55e32-4e2a-4edc-b4b7-6f0134e8bc43. The run never addresses the PI.
- 2026-11-23, foreman: The task id is t392. The crux CLI gave t371, which collides with the deviance task t371 on `exp/t371-deviance-evals`, so the task was renumbered by hand and `counter_t` was set to 392.

## Verify yourself before trusting

- The read-count reference's results (imputation 0.500, correction 0.818 with range 0.014, calibration 0.685 on 256 regions; 16-region numbers too). Check the t370 run's tables, `pareto_p256_sv.tsv` and `cruxvault/INSTRUCTIONS/t370_recipe/evolution/`, on `origin/exp/t370-minicandi-recipe`. Find where its 3 seeds' checkpoints and predictions live on Fir.
- The read files for the recount: 752 GiB in 6 nearline tars (onion-work, 1,104 GiB), on Nibi `/nearline/def-maxwl/mforooz/scratch_archive_2026-11-17/`. The index is at Nibi `/project/def-maxwl/mforooz/ONION/nearline_archive_2026-11-17/`, and testbed read files are on /project. These figures came from the recipe session; `ls` the paths and the index yourself.
- The point-rule region store (512 training and 256 evaluation regions, base views only) at Nibi `/project/def-maxwl/mforooz/ONION/CO-A7/res_store` (6.8G). It cannot serve correction, because it lacks the views. Use it only for tests.
- The cost figures in INSTRUCTIONS.md §3 (recount rate 9.2 CPU-h per 117 GiB from job 27662614; training 163 CPU-h for 3 seeds; 16-region scoring 127 CPU-h). They are the recipe session's measurements. Recheck them with `sacct`.
- Whether PR 65 (deviance scorer) has merged into `exp/t370-minicandi-recipe`. Check with `gh pr view 65 --repo mehdiforoozandeh/CANDII`. Merge `origin/exp/t370-minicandi-recipe` into the umbrella once it has.
- The average-input file named in the config, `/scratch/mforooz/onion/aux/eic_v1_average.h5`, and which cluster holds it.

## Never do

- Never address the PI. Send everything to the foreman (SendMessage to local_b5c55e32-4e2a-4edc-b4b7-6f0134e8bc43).
- Never merge into `exp/t370-minicandi-recipe` or `main`. Never push to `main`. Never force-push.
- Never edit `.longrun/` on this branch. Those files belong to the t370 run and conflict on every merge. Keep this run's memory under `cruxvault/INSTRUCTIONS/t392_read_starts/run_state/`. If the longrun skill insists on `.longrun/`, point it at that folder or ask the foreman.
- Never start Phase 2, and never score on 256 regions, without the PI's word relayed by the foreman.
- Never retrain the read-count reference.
- Never resubmit another session's jobs. That covers the recipe run's `rcp_` jobs, the deviance run's jobs (Fir 5395488–5395517, 5401671–5401702, 5443555–5443583), the t184 blind build on Nibi, and the pval run's `eicx_` jobs.
- Never delete nearline tars, /scratch or /project data.
- Never change the headline or deviance scorers. The conversion is a separate step.

## Traps

- **Task id collisions.** The crux counter on one branch does not see task files on unmerged branches. Before `crux task add`, list task ids across `git for-each-ref refs/remotes/origin` and set `counter_t` above the maximum.
- **Run-log conflicts.** The recipe session appends to `.longrun/LOG.md` and `term_counts.json` all the time. Any branch that touches them conflicts again and again. Do not touch them.
- **Worktree PYTHONPATH.** In any CANDII worktree, `export PYTHONPATH=$PWD/src`, or pytest tests the main checkout.
- **Fir torch imports.** More than 12 parallel torch imports off the shared venv fail. Cap arrays at `%12`.
- **sbatch comma split.** A comma-valued variable inside `--export=ALL,VAR=a,b` is truncated. Export it in the shell and pass `--export=ALL`.
- **Fir bad GPU nodes.** Exclude fc10713 and fc10715.
- **Fir tests.** The full pytest suite on a Fir kit fails 68 tests that also fail on the base (no `.git`, hashes recorded off Fir, launcher dry runs, environment). Judge a change by "no new failures versus the base".
- **Heavy compute.** Test suites, smokes and training run as cluster jobs, not on the laptop.
- **Concurrent sessions.** Other sessions share this repo. Scope every `git add`, and work only in this worktree and its subagent worktrees.

## State

- Worktree: `/Users/mforooz/Desktop/research/libbrechteam@sfu/CANDII/.claude/worktrees/readstart`.
- Branch: `implementation/t392-read-start-minicandi`, cut from `origin/exp/t370-minicandi-recipe` at e77c8b51, and pushed with a draft PR into `exp/t370-minicandi-recipe` (see the PR link in the foreman's paste message).
- Crux task: t392 (`cruxvault/tasks/t392_minicandi_on_read_start_counts_full.md`), refs q75 (the Pareto recipe question).
- Jobs: none submitted for t392.
- Other live work: PR 63 (recipe), PR 64 (pval EIC, do not merge), PR 65 (deviance; the foreman merges it into the recipe branch soon).

## Definition of done (Phase 1)

1. The tests for the recount, the store strand channel, `res_dataset` with `average_h5`, and the conversion step pass. They show no new failures versus the base.
2. The recount gate passes. Convolving observed read starts reproduces the `eic_v1_2` overlap counts with 0 differing bins on every measurement, and the result is logged.
3. The read-start model is trained on 3 seeds.
4. `cruxvault/results/t392/side_by_side/SIDE_BY_SIDE.md` exists with the 16-region table (both models; imputation, correction, calibration and the deviance metrics; seed ranges), one figure, and the `runall.sh` and `MANIFEST.tsv` contract.
5. A crux finding on q75 cites that file.

## Stop condition and what to report

Stop and send the foreman a report at any one of these:

- the 16-region table is done;
- the recount gate fails;
- the cost passes 600 CPU-h;
- a decision only the PI can make.

The report contains the side-by-side table, the CPU-h spent with job ids, any deviation from INSTRUCTIONS.md, and one-line questions for the PI, each with a recommendation. Then wait for the foreman's relay.
