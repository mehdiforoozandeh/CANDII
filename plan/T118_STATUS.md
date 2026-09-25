# t118 status — architecture ladder A–D (overnight run, started 2026-09-25)

Updated as the work goes. Design authority: `plan/T118_COUNTERFACTUAL_F.md`. Work order:
`plan/T118_HANDOFF.md`.

## Now

- **Done: all 576 runs trained, scored and law-tested; all four designs aggregated on the final data**
  (jobs 22682907–10) and linked in the notebook: `cruxvault/results/h12` (A), `h17` (B), `h13` (C),
  `h14` (D), `h15` (main claim, drafted readings). Checks met: A 45/78, B 67/102, C 68/102, D 41/102.
- Running: an exploratory design X (the PI's idea: g also reads the bin value), 24 runs, array
  22683580, output `ladder/explore_x/` — outside the pre-registered ladder.
- Team overview page (design + pilot numbers): https://claude.ai/artifact/Gj3KNBCaK1efJpfTQ8KV7V



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
| A — per-bin affine | yes | yes | yes | yes, 144/144 + law | yes: `cruxvault/results/h12/report.md` |
| B — per-bin monotone curve | yes | — | yes | yes, 144/144 + law | yes: `cruxvault/results/h17/report.md` |
| C — kernel, then curve | yes | — | yes | yes, 144/144 + law | yes: `cruxvault/results/h13/report.md` |
| D — conditioned CNN | yes | — | yes | yes, 144/144 + law | yes: `cruxvault/results/h14/report.md` |

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
| 22676228 | retry of law tasks 433–440 (the same CVMFS read error) | 2026-09-25 ~14:30 | submitted |
| 22678427 | aggregation of design A (retry of 22678258, which hit the CVMFS error; node c537 excluded) | 2026-09-25 ~15:10 | completed, 2.6 min |
| 22678919 | aggregation of design B | 2026-09-25 ~15:40 | completed, 3 min |
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

## For the PI — design A, first full reading (drafted, nothing ticked)

The report holds every check with its value, bar and seed wobble: 45 of 78 checks meet their bar.
Shape of the result, in plain terms:

- **p space: the covariates help.** Against the no-covariates twin on the held-out chromosomes, all
  12 p-space readings meet the bar (every class, both versions of g, all bins and top 1%).
- **Counts: mostly not.** Only broad (both g's) and narrow (across-track g) all-bins readings meet
  the bar; top-1% readings are all unmet, and DNase is worse than the twin with a per-track g
  (−0.55 all bins). Design A's straight line on log(1 + counts) is the likely reason (see the pilot
  note).
- **Law test (never-trained pairs):** beats the labels-as-ids twin almost everywhere (often by
  orders of magnitude — that twin has no answer for unseen pairs); beats the no-covariates twin on
  all bins for narrow and broad, rarely on top 1%, and not for DNase with a per-track g.
- **Depth law: unmet in every class** (largest |predicted scale / depth ratio − 1| = 0.35 to 0.60
  against a bar of 0.10). Design A does not learn the depth scale to within 10%.
- **Rung choice for the main claim (chr22, provisional — B–D law tests are still running but
  validation does not depend on them):** C for the across-track g in both spaces; D per track in
  counts; A per track in p space, because B, C and D have runs that blow up there (next item).

### Design B against design A (drafted readings met, of 12 per cell unless noted)

| check | A counts | B counts | A p | B p |
|---|---|---|---|---|
| beats the no-covariates twin (held-out chromosomes) | 3 | 11 | 12 | 7 |
| law test: beats the no-covariates twin | 6 | 8 | 4 | 1 |
| law test: beats the labels-as-ids twin | 10 | 12 | 10 | 11 |
| beats the design below (B > A) | — | 9 | — | 8 |
| depth law (of 6; counts only) | 0 | 0 | — | — |

B's curve fixes count space, as expected (3 → 11 against the twin). In p space B does worse than A;
this matches the exploding upscaling pairs below, which hit B and D and barely A (A has one σ for
the whole track). Depth law still unmet (largest error 0.30–0.52). Design B: 67 of 102 checks met.

## All four designs (drafted readings met; 12 per cell unless noted)

| check | A counts | B counts | C counts | D counts | A p | B p | C p | D p |
|---|---|---|---|---|---|---|---|---|
| beats the no-covariates twin | 3 | 11 | 11 | 11 | 12 | 7 | 11 | 3 |
| law test vs the no-covariates twin | 6 | 8 | 12 | 9 | 4 | 1 | 4 | 1 |
| law test vs the labels-as-ids twin | 10 | 12 | 12 | 11 | 10 | 11 | 11 | 5 |
| beats the design below | — | 9 | 5 | 1 | — | 8 | 2 | 0 |
| depth law (of 6) | 0 | 0 | 0 | 0 | — | — | — | — |

Main claim (chr22 rule): C for the across-track g in both spaces, D per track in counts, A per track in
p. Shuffle met almost everywhere; swap unmet everywhere; depth law unmet everywhere. Details:
`cruxvault/results/h15/report.md`.

## A decision for the PI: a few p-space pairs explode the mean

In 32 of 576 run × split combinations, one or two pairs (of 14–242) score CRPS from ~25 to ~10⁷,
almost all in −log10 p space and mostly in designs B and D. They are all **upscaling** pairs: a
sparse source (depth 3.75M, 7.5M or 15M, or DNase with dedup off) to the full-depth base. Their
top-1% CRPS is ordinary (~4); the explosion is in the low bins. Reason: where the source is near
zero it tells little about the target, so the log-normal fit learns a large σ at low levels. That is
fine for the likelihood, but a log-normal's mean is median × exp(σ²/2), so its CRPS explodes. One
such pair then dominates the mean over pairs, and with it the seed wobble.

**Corrected diagnosis (checked 2026-09-25 with the PI).** In the worst pair (DNase depth 3.75M →
base, design B, across-track g) the lowest knot sits at x = log(1e-3), the floor, and its learned σ
is 10.9. But only 0.1% of source bins sit at the floor, and there the target's log-space SD is 1.8.
So the 10.9 is not what the data ask for: the lowest knot has almost no training data, its σ drifts,
and linear interpolation spreads the large σ over every bin between the floor and the median, where
exp(σ²/2) makes the CRPS explode. The PI rejected capping σ and reporting-only as not a fix. The
earlier idea of a censored log-normal is withdrawn (it rested on the wrong diagnosis). Open: D also
explodes and has no knots — to be checked. The PI proposed that g also read the bin value (design X
above); it is being tested.

## Failures and fixes

- 8 of 576 train tasks failed in their first 35 s: every task rewrote the shared `tasks.tsv`, and
  concurrent readers on /project got "Stale file handle". Fix (82cf8db, shell only; the Python code
  is unchanged from the run's 34e5705): each task reads its own copy and the shared one is never
  replaced. Retried as 22667089.
- Law and aggregation tasks failed 10 times on "Input/output error" reading the cluster's shared
  software filesystem (CVMFS) while building the venv (nodes c128, c166, c537) — a cluster fault, not
  ours; each was resubmitted.
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

- **Nibi tunnel down (2026-09-25 15:10 local), third time.** Nibi work stopped; I did not authenticate. Design X
  (array 22683580) keeps running; its results wait for `hpc up nibi`.
