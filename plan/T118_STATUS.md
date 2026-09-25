# t118 status — architecture ladder A–D and design X (as of 2026-09-25, evening)

Design authority: `plan/T118_COUNTERFACTUAL_F.md`. First work order: `plan/T118_HANDOFF.md`. Next
agent's work order: `plan/T118_HANDOFF_2.md`. Nothing here is ticked or a verdict; every reading is
a draft for the PI.

## State in one paragraph

All 576 pre-registered runs (4 designs × 3 models × 8 g's × 2 spaces × 3 seeds) are trained, scored
on chr19 + chr21, law-tested on the 2 094 never-trained arm → arm pairs, and aggregated. Each design
has a report, 9 figures and a checks file in `cruxvault/results/` (main checkout; gitignored), linked
from its hypothesis. The main-claim readings are drafted. Two problems stand out: a few p-space
pairs make the mean CRPS explode in designs B, C and D, and the depth law and swap checks are unmet
everywhere. An exploratory test of the PI's idea (g also reads the bin value; "design X") removed
the p-space explosions on the two tracks tested but did worse than B–D in DNase counts.

## Where everything is

| what | where |
|---|---|
| code | branch `exp/t118-counterfactual-f` (PR #48, draft): `tools/t118/ladder/`, `tools/t118/covariates.tsv`, `slurm/t118/ladder_*.sh`, `slurm/t118/explore_x*.sh`, `tests/test_t118_*.py` (238 tests) |
| code run on Nibi | kit `K3` = Python 34e5705 + SLURM 82cf8db (the 576 runs); kit `K4` = K3 + design X (5b2998d, CPU script) — under `/project/def-maxwl/mforooz/t118/ladder/code/` |
| runs | `nibi:/project/def-maxwl/mforooz/t118/ladder/runs/<rung>_<g>_<space>_<model>_s<seed>/` (ckpt, scores.json, law.json, figdata.npz) |
| aggregate | `nibi:/project/def-maxwl/mforooz/t118/ladder/agg/` (results.json, results_summary.tsv, checks_*.json, `<rung>/report.md` + figures) |
| design X runs | `nibi:/project/def-maxwl/mforooz/t118/ladder/explore_x/runs/X_*` |
| cache | `nibi:/project/def-maxwl/mforooz/t118/ladder/cache/` — 260 memory-mapped files, 109 GB. Kept; deleting it is the PI's call |
| evidence | main checkout `cruxvault/results/h12` (A), `h17` (B), `h13` (C), `h14` (D), `h15` (main claim), each with `FIR_PATH.txt`; also copied into the worktree's `cruxvault/results/` so `crux validate` resolves the links |
| team page | https://claude.ai/artifact/Gj3KNBCaK1efJpfTQ8KV7V (mirror: `cruxvault/results/t118/team_overview_2026-09-25.md`) |
| build plan and log | `.orchestrate/plan.md` in the worktree (git-excluded) |

## Results — the four designs (drafted readings met; 12 per cell unless noted)

Every gain is judged against 2 × the real g's seed wobble (3 seeds), per mark class (DNase; narrow;
broad), CRPS on all bins and on the top 1%, for both versions of g (one per track, one across tracks).

| check | A counts | B counts | C counts | D counts | A p | B p | C p | D p |
|---|---|---|---|---|---|---|---|---|
| beats the no-covariates twin (held-out chromosomes) | 3 | 11 | 11 | 11 | 12 | 7 | 11 | 3 |
| law test vs the no-covariates twin | 6 | 8 | 12 | 9 | 4 | 1 | 4 | 1 |
| law test vs the labels-as-ids twin | 10 | 12 | 12 | 11 | 10 | 11 | 11 | 5 |
| beats the design below | — | 9 | 5 | 1 | — | 8 | 2 | 0 |
| depth law (of 6; counts only) | 0 | 0 | 0 | 0 | — | — | — | — |

Checks met in total: A 45/78, B 67/102, C 68/102, D 41/102.

- **Design A** (one line in log space): p space is its strength (12/12 against the twin). In counts
  it fails, because one straight line on log(1 + counts) cannot represent "no change" at low counts
  and overshoots at peaks (on the p-only arms, a ≈ −1.6, b ≈ 1.9 instead of 0 and 1).
- **Design B** (monotone curve) fixes counts (3 → 11) but is worse than A in p space, which is where
  the exploding pairs hit.
- **Design C** (kernel, then curve) is the best in counts on the law test (12/12) and is the design
  the main claim judges for the across-track g.
- **Design D** (FiLM CNN) is strong in counts against the twin but collapses in p space (3/12),
  again because of exploding pairs.
- **Depth law**: unmet by every design in every class (predicted count scale misses the true depth
  ratio by 30–140% against a 10% bar). Not yet investigated: check the depth-law computation itself
  before reading this as a model failure.

## Results — the main claim (drafted; `cruxvault/results/h15/report.md`)

Chosen design by the pre-registered chr22 rule (lowest design within the best one's seed wobble):

| g | space | A | B | C | D | chosen |
|---|---|---|---|---|---|---|
| across tracks | counts | 0.491 | 0.395 | **0.353** | 0.360 | C |
| across tracks | −log10 p | 0.226 | 4.76 | **0.181** | 8 912 | C |
| per track | counts | 0.566 | 0.374 | 0.340 | **0.332** | D |
| per track | −log10 p | **0.224** | 0.229 | 0.338 | 386 | A |

Readings of the chosen design (met/total): shuffle 22/24 met (a wrong C′ removes the advantage);
swap 0/12 met (C′ = C does not return X to within a median |log ratio| of 0.1 — ≈0.19–0.22 for C in
counts; part of this is the log(1 + X) input scale); depth law 0/6 met. Under the rule `all`,
several checks are unmet for every chosen design.

## The p-space explosion (open; the PI wants a principled fix)

In 32 of 576 run × split combinations, one or two pairs (of 14–242) score CRPS from ~25 to ~10⁷,
almost all in −log10 p and in designs B, C and D. All are **upscaling** pairs (my label): a sparse
source (depth 3.75M, 7.5M, 15M, or DNase with dedup off) to the full-depth base. Their top-1% CRPS
is ordinary (~4); the explosion is in low bins, and one such pair dominates a run's mean and seed
wobble.

Diagnosis, checked with the PI: in the worst pair (DNase depth 3.75M → base, design B, across-track
g), the lowest knot sits at x = log(1e-3) (the floor) and its learned σ is 10.9. Only 0.1% of source
bins sit at the floor, and there the target's log-space SD is 1.8 — so 10.9 is not what the data ask
for. The lowest knot has almost no training data, its σ drifts, and linear interpolation spreads the
large σ over every bin between the floor and the median, where a log-normal's mean,
median × exp(σ²/2), makes CRPS explode. A is spared because it has one σ per pair.

Ruled out by the PI: capping σ (not principled); keeping the runs and only reporting (not a fix).
Withdrawn by me: a censored log-normal (it rested on a wrong diagnosis). Also discussed: a
log-Laplace would be worse (power-law tail; mean and CRPS infinite once its scale reaches 1). Still
unexplained: design D also explodes and has no knots.

## Design X — exploratory test of the PI's idea (not pre-registered)

The idea: g reads the bin value as well as (C, C′), so each bin value gets its own transformation.
As built: g outputs a 32-number pair vector from (C, C′); a shared MLP reads [bin value, pair vector]
and gives, per bin, loc = x + a and disp = d. So f is design A with the slope fixed at 1 and a
shift a and spread d that depend on the bin value — any function of x, not forced to be monotone.
It is a change to both g and f; it is fairly compared with B (both per-bin), not with C or D, which
see neighbouring bins. Tested: one g per track, H3K27ac and DNase, both spaces, real g and the
no-covariates twin, 3 seeds, held-out chromosomes only (no law test). Ran on CPU (16 cores) because
every 10 GB GPU-slice node was held by the scheduler for another job.

| track | space | A | B | C | D | X | exploding pairs A/B/C/D/X | X beats twin |
|---|---|---|---|---|---|---|---|---|
| H3K27ac | counts | 0.419 | 0.378 | 0.364 | 0.363 | 0.376 | 0/0/0/0/0 | met (0.085 > 0.002) |
| H3K27ac | p | 0.165 | 0.441 | 1.04 | 0.144 | 0.151 | 0/1/1/0/0 | met (0.036 > 0.004) |
| DNase | counts | 3.06 | 0.824 | 0.818 | 0.772 | 1.32 | 3/0/0/0/0 | unmet (0.30 < 0.42) |
| DNase | p | 0.752 | 0.541 | 0.526 | 18 500 | 0.659 | 0/0/0/3/0 | met (0.16 > 0.13) |

Mean CRPS over all bins, real g, mean over seeds; exploding = pairs with CRPS > 20 over 3 seeds. In p
space X had no exploding pair and beat its twin on both tracks. In counts it is close to B on
H3K27ac and clearly worse than B–D on DNase. Two tracks and sporadic explosions: suggestive, not
proof.

## Choices I made (science choices the plan did not settle; the most conservative option)

1. Fragment length is not a covariate; only the extsize factor k is encoded (fragment length is in
   the table, unencoded).
2. Products with no control (DNase, control-identity = none): control fraction, ratio k and control
   depth are 0, identity none, has-control 0.
3. Count-space pairs whose target counts equal the source are kept and flagged; headline includes
   them, a variant without them is reported beside.
4. The labels-as-ids twin's permutation is a derangement.
5. B's curve above the last knot continues with the last slope.
6. The across-track g gets the same step budget as a per-track g.
7. The depth law is computed in count space only.
8. The main claim's chr22 rung choice uses CRPS on all bins (real g, mean over seeds, macro over
   tracks).
9. The law test scores all chr19 + chr21 bins, in a CPU job per run, from the checkpoint.
10. Figures use matplotlib; tested locally with `/Users/mforooz/miniforge3/bin/python` (the
    `candii` env lacks it). No environment changed.
11. The 109 GB cache stays until the PI decides.
12. At test time the no-covariates twin predicts with one θ (its mean over the training pairs), so
    it is never asked to extrapolate onto covariate pairs it never saw.

## Disagreements with the plan

- The across-track g has 242 pairs, not the plan's "246" (both DNase MAPQ arms excluded).
- Swap vs input scale: with x = log(1 + counts), the identity predicts X + 1, so design A may miss
  the 0.1 swap bar by construction in counts. Bar unchanged; reported as measured.

## Failures and fixes (none changed a number)

- 8 train tasks failed in 35 s on an NFS "Stale file handle" (every task rewrote a shared
  `tasks.tsv`); fixed in 82cf8db (shell only) and retried.
- The law array chained with `aftercorr` never released a task while the train array ran; cancelled
  and resubmitted directly.
- 10 law/aggregation tasks failed on CVMFS "Input/output error" while building the venv (nodes
  c128, c166, c537); resubmitted with those nodes excluded.
- The Nibi tunnel dropped three times; Nibi work stopped each time until the PI ran `hpc up nibi`.
- Design X's GPU array (22683580) never started: all MIG nodes (g30–g37) were held (`mixed-`) for a
  higher-priority job. Cancelled; run on CPU (22688594), 4–51 min per run.

## Nibi jobs (all finished)

cache 22654779 · early check 22655875 · pilot 22656701 / 22656702 · train 22657297 (+ retry
22667089) · law 22657301 (cancelled part) → 22667088, 22667090, 22667091, 22669098, 22676228 ·
aggregation 22678427 (A), 22678919 (B), final 22682907 (C), 22682908 (D), 22682909 (A), 22682910
(B) · design X 22683580 (cancelled) → 22688594 (CPU).

## Measured cost

Per-track train + score ≈ 12 min on a 10 GB slice (cold cache; much faster warm); across-track law
test up to ~2 h on 8 CPU cores; aggregation 2–4 min; design X on 16 CPU cores 4–51 min per run.

## Waiting on the PI

- The principled fix for the p-space explosion (see above), and whether design X should become a
  pre-registered design.
- Ticks, verdicts (`crux close`), accepting the task, merging — none done.
