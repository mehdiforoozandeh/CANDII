# t118 status — architecture ladder A–D (overnight run, started 2026-09-25)

Updated as the work goes. Design authority: `plan/T118_COUNTERFACTUAL_F.md`. Work order:
`plan/T118_HANDOFF.md`.

## Now

- **Waiting on the PI: Nibi tunnel down since 2026-09-25 11:23 (see Blocked on the PI).**


- Ground truth checked 2026-09-25: branch `exp/t118-counterfactual-f` in sync with origin, 0
  commits behind `origin/main`; Nibi reachable; 130 product dirs + `MANIFEST.tsv`, md5
  `599e2ca607961fe550b477558f894edf` (matches the handoff).
- Notebook: the four rungs and the main claim are now `running` (opt-out field added to the four
  rungs, citing the PI ruling of 2026-09-23). Commit 74ff2e2.
- Build plan written (git-excluded, `.orchestrate/plan.md` in this worktree). Waves: 0 = covariate
  table + ladder core (contracts, data reader, pair and task tables, synthetic products); 1 =
  training harness, the four f forms, scorer, figures/report, SLURM scripts (8 builders in
  parallel); 2 = end-to-end CPU smoke of all four rungs; 3 = pilot on Nibi, then all 576 runs.
- Wave 0 done and merged: covariate table (`tools/t118/covariates.tsv`, 130 products, source named
  for every base value) and the ladder core (pair counts verified on the real manifest: 38 per
  histone track, 14 DNase, 242 across tracks, 2 094 never-trained arm→arm pairs, 576 runs).
- Wave 1 done and merged (a0abc6f): harness, forms A–D, scorer, figures/report, SLURM. The harness
  and the scorer each passed an independent review, with fixes. 218 t118 tests pass.
- Wave 2 done (34e5705): every rung runs end to end on CPU on synthetic data (train → score →
  law test → aggregate → 9 figures → report); 226 t118 tests pass.
- Pilot passed on Nibi (12/12 tasks). **All 576 runs submitted** 2026-09-25 ~07:10 (train array
  22657297, law array 22657301). The six pilot runs are reused (their outputs sit in `runs/`).
- The cache build runs on Nibi now (it needs only merged code), so the pilot will not wait on it.
- Planner's estimate, to be checked by the pilot: per-track run ≈ 20 min on a MIG slice,
  across-track ≈ 35 min; law test 10–90 min on CPU per run; A and B done ≈ 3 h after submission
  if 40 slices run at once, all four ≈ 6–8 h.

## Per rung

| rung | built | pilot | submitted | finished | report |
|---|---|---|---|---|---|
| A — per-bin affine | no | no | no | no | no |
| B — per-bin monotone curve | no | no | no | no | no |
| C — kernel, then curve | no | no | no | no | no |
| D — conditioned CNN | no | no | no | no | no |

## Nibi jobs

| job | what | submitted | state |
|---|---|---|---|
| 22654779 | `t118L_cache`: memory-mapped cache, 130 products × 2 spaces | 2026-09-25 ~05:45 | completed, 260 files, 109 GB |
| 22655875 | early real-data check: design A, H3K27ac track, real g, seed 0, counts and p (code a0abc6f), output in `ladder/pilot0/` (never used by the full run) | 2026-09-25 ~06:10 | completed, both tasks exit 0 |
| 22656701 | pilot, train: design A, H3K27ac, 3 models × 2 spaces, seed 0 (code 34e5705) | 2026-09-25 ~06:50 | completed, 6/6 exit 0 |
| 22656702 | pilot, law test of the same 6 runs (starts per task after its train task) | 2026-09-25 ~06:50 | completed, 6/6 exit 0 |
| **22657297** | **full run, train + score: all 576 runs, array 0-575 %40, A → B → C → D (code 34e5705)** | 2026-09-25 ~07:10 | at 11:30 UTC: 544 done, 24 running, 8 failed (a shared-file race on /project, before training; fixed in 82cf8db and retried) |
| 22657301 | full run, law test (`aftercorr`) | 2026-09-25 ~07:10 | 17 done; the rest never started (SLURM held the whole array on the train array) and was cancelled, replaced by 22667088/90/91 |
| 22667088 | law test, re-submitted without a dependency for the 527 runs already trained (code 82cf8db) | 2026-09-25 ~11:50 | submitted |
| 22667089 | retry of the 8 train tasks that failed (92, 99, 209, 244, 280, 304, 340, 353) | 2026-09-25 ~11:50 | submitted |
| 22667090 | law test of those 8 (after each retry) | 2026-09-25 ~11:50 | submitted |
| 22667091 | law test of the 24 runs still training at 11:30 (starts when train array 22657297 ends) | 2026-09-25 ~11:50 | submitted |
| 22669098 | retry of law task 110 (transient CVMFS read error while building the venv) | 2026-09-25 ~12:10 | submitted |
| 22657365 | trial aggregation of rung A on the 6 pilot runs (tests figures and report on real data; overwritten by the real one) | 2026-09-25 ~07:15 | completed, 2 min, 7 GB; 9 figures + report + checks JSON |

## Output paths

- All Nibi outputs: `/project/def-maxwl/mforooz/t118/ladder/`

## Measured time per run

Early check (job 22655875, design A, one H3K27ac g, 38 training pairs, 1 MIG slice + 4 cores):

| run | venv | train | score | wall | peak host memory (sacct) |
|---|---|---|---|---|---|
| counts, real g, seed 0 | ~1 min | 361 s (2 000 steps, best at 1 000) | ~5 min | 12.2 min | 12.3 GB |
| −log10 p, real g, seed 0 | ~1 min | 561 s (3 750 steps, best at 2 750) | 70 s | 12.1 min | 13.5 GB |

Peak memory includes the memory-mapped cache pages, so the full run asks for 32 GB (command-line
`--mem`) instead of the script's 16 GB.

### What the early check showed (one seed, real g only — not a result, no twins yet)

Design A, H3K27ac track, mean CRPS over its 38 trained pairs on chr19 + chr21, blacklist removed;
references are the v2 baseline numbers for the same pairs:

| space | bins | real g, design A | noSolution | QuantileMatching |
|---|---|---|---|---|
| counts | all | 0.421 | 0.503 | 0.437 |
| counts | top 1% | 5.30 | 2.40 | 1.87 |
| −log10 p | all | 0.164 | 0.253 | 0.215 |
| −log10 p | top 1% | 3.27 | 2.70 | 2.31 |

On the top 1% of bins design A is worse than both references. The likely cause is design A's one
dispersion value for the whole track: fitted mostly on near-zero bins, it makes the predicted
distribution too wide at peaks (the references use a Poisson, which is narrow). Design B gives a
dispersion per level, which should fix this. I read it as a property of design A, not a bug, and did
not change anything. Seed wobble is not known yet, so none of these gaps can be judged.

## Choices I made

Science choices the plan does not settle; each is the most conservative option.

1. Fragment length is not a covariate. Only the extsize factor k is encoded; the actual fragment
   length (which also moves on the paired-end and MAPQ arms) is recorded in the table, unencoded.
2. Products with no control (DNase and the control-identity = none arms): control fraction,
   ratio k and control depth are 0, identity = none, has-control = 0 (the DNase rule applied to
   both).
3. Count-space pairs whose target counts equal the source counts (p-only arms as targets) are kept
   and flagged; the headline numbers include them, and a variant without them is reported beside.
4. The labels-as-ids twin's fixed permutation is a derangement (no pair keeps its own covariates).
5. Rung B's curve above the last knot continues with the last segment's slope (QuantileMatching
   stays flat there).
6. The across-track g gets the same step budget as a per-track g.
7. The depth law is computed in count space only.
8. The main claim's rung choice on chr22 uses CRPS on all bins (real g, mean over seeds, macro over
   tracks), per version of g and per space.
9. The law test scores all chr19 + chr21 bins, in a separate CPU job per run, regenerating
   predictions from the checkpoint.
10. Figures are drawn with matplotlib. The local `candii` env lacks it, so figures are tested with
    `/Users/mforooz/miniforge3/bin/python`; on Nibi the job venv has it. No environment changed.
11. A memory-mapped cache of all 130 products × 2 spaces (≈126 GB) goes under
    `/project/def-maxwl/mforooz/t118/ladder/cache/`. I will not delete it; that is the PI's call
    once the runs are scored.
12. At test time the no-covariates twin predicts with one θ for every query: the mean of its g's
    outputs over the training pairs. Otherwise g, which learned to ignore covariates, would be
    queried on covariate pairs it never saw (law test, shuffle, swap), and its answer there would
    be an extrapolation rather than "one average map". This is the conservative reading: it gives
    the twin its best average map, so "beats the twin" is not made easier.

## Failures and fixes

- 8 of 576 train tasks failed in their first 35 s: every task rewrote the shared `tasks.tsv`, and
  concurrent readers on /project got "Stale file handle". Fix (82cf8db, shell only; the Python code
  is unchanged from the run's 34e5705): each task reads its own copy and the shared one is never
  replaced. Retried as 22667089.
- The law-test array (`aftercorr` on the train array) never released a task while the train array
  was still running, and would never have run for the 8 failed tasks. Cancelled the pending part and
  re-submitted the law test directly for every trained run.

## Disagreements with the plan

- The design plan says the across-track g has "all 246 pairs". With both DNase MAPQ arms excluded
  (the handoff's rule) it is 242: 6 histone tracks × 38 + DNase 14. The code uses 242.
- **Swap check vs the input scale (for the PI; bar not changed).** f reads x = log(1 + counts), so
  the identity map predicts a mean of X + 1, not X; at X = 1 that alone gives |log ratio| = 0.69.
  The swap bar (median |log(predicted mean / X)| over bins with X > 0 below 0.1) can still be met
  by forms whose curve or head can bend at low counts (B, C, D), but design A (one line in log1p
  space) may miss it by construction in count space. Swap gates only the main claim. I report the
  number as measured.

## Blocked on the PI

- **Nibi tunnel down again (2026-09-25 11:23 local).** `ssh -o BatchMode=yes nibi` exits 255. Nibi work is
  stopped; I did not try to authenticate. Last known state (16:10 UTC): train 570/576 done, 6 running
  (the 8 retries included); law test A 105/144, B–D starting; one law task (110) failed on a
  transient CVMFS read error and was resubmitted (22669098). The jobs keep running. Needed: the PI
  runs `hpc up nibi`; then I aggregate each finished rung, copy the evidence down and link it.
