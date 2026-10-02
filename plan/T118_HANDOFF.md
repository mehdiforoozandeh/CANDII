# t118 handoff — build, run and report the architecture ladder A–D (overnight, 2026-09-25)

Written by the design session for the agent that does the work. **The design authority is
`plan/T118_COUNTERFACTUAL_F.md` on branch `exp/t118-counterfactual-f`** (draft PR #48). This file
tells you what to build and how to run it; where this file and the plan disagree, the plan wins and
you note the disagreement in the status file.

## 0. Your mandate (PI rulings 2026-09-25)

- **Build and run everything:** all 4 rungs × 3 models × 8 g's × 2 spaces × 3 seeds = **576 training
  runs**, plus scoring, figures and one report per rung. **Do not stop after the pilot** to ask for
  a budget. Submit it all even if it cannot finish by morning; the PI reads whatever rung is ready.
- **Permission:** the PI approved following the plan without asking at each step. That covers
  building, testing, committing, pushing to this branch, submitting SLURM jobs on Nibi, and moving
  the hypotheses to `running` in the notebook.
- **Still the PI's, never yours:** ticking checks, recording a verdict (`crux close`), accepting a
  task (`crux task accept`), approving a synthesis, merging to `main`. Draft these for the morning.
- **Science choices not covered by the plan: do not invent them.** Pick the most conservative
  option, record it in the status file under "Choices I made", and keep going. Routine engineering
  (optimizer, learning rate, window length, batch mix, file layout) is yours.
- Priority when time is short: **A and B complete over C and D.** Order the job queue that way.

## 1. Ground truth — verify each yourself before you rely on it

| what | where | check |
|---|---|---|
| design record, every ruling | `plan/T118_COUNTERFACTUAL_F.md` (this branch) | read it all first |
| worktree | `/Users/mforooz/Desktop/research/libbrechteam@sfu/CANDII/.claude/worktrees/counterfactual-f` | `git status`, `git fetch`, branch = `exp/t118-counterfactual-f`, in sync with origin |
| products (130) | Nibi `/project/def-maxwl/mforooz/t112_cf/products/<pid>/{counts25,pval25}.npz` + `MANIFEST.tsv` there | 130 product dirs + `MANIFEST.tsv`; manifest md5 `599e2ca607961fe550b477558f894edf` |
| npz layout | `tools/t112/bin25.py::write_npz`, docstring of `tools/t118/baseline_rungs.py` | one array per chr1..chr22, chrX; counts uint32, p float32; length floor(len/25) |
| arm definitions / knob defaults | `tools/t112/arms.py`, `tools/t112/bam_arm.py`, `tools/t112/fastq_arms.py`, `slurm/t112/*.sh` | source of the base values for the covariate table |
| baseline scorer (reuse, do not rewrite) | `tools/t118/baseline_rungs.py` (+ `tests/test_t118_baseline_rungs.py`) | Poisson / log-normal CRPS, subsets all/nonzero/top1, Spearman, split, blacklist |
| NB CRPS | `src/candi/bench/distributional.py::nb_crps_mean` | NB CRPS for NB (n, μ) outputs |
| reference numbers | Nibi `/project/def-maxwl/mforooz/t118/rungs_v2/` (246 pair JSONs + aggregate) | noSolution and QuantileMatching per pair; plan §4 has the mark-class table |
| blacklist | Nibi `/project/def-maxwl/mforooz/EIC_REPRO/002/scripts/hg38_blacklist_v2.bed` | sha256 starts `31c69342` |
| x-transformers wheel | Nibi `/scratch/mforooz/t112_cf/code/C15/wheels/x_transformers-2.11.23-py3-none-any.whl` | copy into `$KIT/wheels` of your code snapshot |
| the notebook | `cruxvault/` in the worktree | main claim `h15`; rungs `h12` (A), `h17` (B), `h13` (C), `h14` (D); parked `h16` (E, do not touch); task `t118` |

## 2. Hard rules

**Nibi and SLURM**
- One ssh session at a time: `ssh -o BatchMode=yes nibi …`. rsync takes ssh options only through
  `-e 'ssh -o BatchMode=yes'` (rsync's own `-o` is `--owner`); quote remote globs; `mkdir -p` the
  remote dir first; use `--no-g --no-p` into `/project`.
- Exit 69 or 255 means the tunnel is down. Stop Nibi work, write it in the status file, and wait —
  the PI runs `hpc up nibi`. **Never try to authenticate.** Keep building locally meanwhile.
- **No tool call over 240 s.** Poll long work with short checks.
- **Never resubmit anything named `t112_*`.** Name your jobs `t118L_*`.
- **Do not delete** the Nibi scratch leftovers `store_src`, `bamarms`, `ta/ctldepth_WRONG_N_20260917`.
- `/scratch/mforooz` is over quota (1826 of 1024 GiB): **write every output under
  `/project/def-maxwl/mforooz/t118/ladder/`** (≈370 GiB free on /project; your outputs are small —
  keep checkpoints small too).
- GPU spec is fixed by AGENTS.md invariant 13: every `#SBATCH --gres` line is
  `--gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1` (a 10 GB MIG slice). No other gres spec, ever.
  `--account=def-maxwl`.
- Job venv: the repo's pinned recipe, built per task in `$SLURM_TMPDIR` — `module load
  python/3.10.13`, `pip install --no-index -r requirements-fir.txt`, then `pip install --no-index
  --find-links $KIT/wheels -r requirements-pypi.txt`. Copy it from `slurm/t118/qm_rungs.sh`.
  `import candi` pulls torch, einops, einx, loguru, x_transformers eagerly.
- No `/usr/bin/time` on compute nodes; read peak memory from `sacct -o MaxRSS`.
- Pass inputs as **positional arguments**, never through a comma-valued `--export` (arrays truncate
  it). A task whose output already exists exits 0, so a resubmission redoes only what is missing.
- `sbatch --test-only` before every real submission.
- Cap concurrent tasks that import torch from a shared venv (seen on Fir: >12 parallel imports
  fail with partial-module errors). The per-task venv avoids the shared import, but cap arrays at
  `%40` anyway and watch the first wave.
- Do not chain `afterok` on a job that may already have finished (Slurm refuses it); verify
  finished work by its output files.

**Local**
- Python is `/Users/mforooz/miniforge3/envs/candii/bin/python`. In any worktree
  `export PYTHONPATH=$PWD/src`, or pytest tests the main checkout.
- The shell is zsh: unquoted `$VAR` does not word-split; use arrays. No `timeout` on macOS.
- No new dependency, package or environment change. torch, numpy, scipy are already there.
- `pytest tests/ -q` must stay green.

**Git**
- Work on `exp/t118-counterfactual-f`. **Cut every subagent worktree from this branch, never from
  `origin/main`** — origin/main lacks the plan, the tools and the notebook changes.
- Merge `origin/main` into the branch as you go (merge, never rebase). Commit and push often —
  work only you can see does not exist. End commit messages with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Before removing any worktree, check it for uncommitted work and rescue it.

**The notebook (crux)** — load the `crux` skill. Run the engine from inside `cruxvault/`:
`cd cruxvault && /Users/mforooz/miniforge3/envs/candii/bin/python ~/.claude/skills/crux/scaffold/crux.py <verb>`.
- Bookkeeping is silent. In anything the PI reads, never use node ids or crux process words; name
  a hypothesis by its content ("design A, the per-bin affine map").
- To move the rungs and the main claim to `running`: the engine wants an outcome-neutral check or
  an opt-out. The PI ruled on 2026-09-23 that no check voids the run; `h15` already carries
  `neutral_optout`. Add the same field, citing that ruling, to `h12`, `h13`, `h14`, `h17`, then
  `test --to running`. If the engine refuses for another reason, record why and carry on with the
  runs — the runs do not depend on the notebook state.
- `TASKHUB.md` is generated; after a merge, take the union of `cruxvault/tasks/` and regenerate
  (`import engine as E; E.refresh(Path("cruxvault"))`, scaffold on `sys.path`).
- Evidence goes to `cruxvault/results/<hid>/` in the **main checkout**
  (`/Users/mforooz/Desktop/research/libbrechteam@sfu/CANDII/cruxvault/results/`, gitignored),
  rsync'd down from Nibi: the report `.md`, figures, scored JSON, and `FIR_PATH.txt` naming the
  Nibi run directory. Link the report from each hypothesis's `## Artifacts`.

## 3. What to build — the spec in one place (the plan has the reasons)

**Pairs.** From `MANIFEST.tsv`: each non-base product gives base→arm and arm→base. **Exclude both
DNase MAPQ arms** (`C12M02`, arm `mapq`; byte-identical to the base) from training and scoring.
Count-space training keeps the p-only arms (ratio, ctlid, ctldepth, extsize; their counts equal the
base — verified from the manifest md5s). No arm→arm and no identity pairs in training. Pairs per
per-track g: 38 for each histone track, 14 for DNase (7 usable arms).

**Split.** Train on every chromosome except chr19, chr21, chr22; chr22 validates (early stopping,
run selection, rung selection); chr19 + chr21 score. chrY, chrM dropped. Blacklist bins removed
from scoring only.

**Covariates (build first — chunk C0).** One full vector per product: depth (log2 reads,
z-scored), run type (0/1), read length, dedup (0/1), MAPQ threshold (number), control read
fraction, MACS2 ratio k, control identity (**three categories: matched, other, none** — never the
accession), control depth, extsize k, assay (one-hot), plus a **"has control" flag**; DNase's
control entries are 0 with the flag 0. Continuous values z-scored over the training products of
the g in question; binary 0/1; categorical one-hot. The manifest records depth, read length, run
type and fragment length for every product, but the other knobs only on the arm that changes them;
**take the base values from the pipeline defaults recorded in the t112 tools and verify each one**
(the DNase pipeline has its own defaults). Write the table to `tools/t118/covariates.tsv` (tracked),
with a column naming the source of every base value, and a test that rebuilds it. g is never given
C' − C.

**g and f.** g is an MLP, 2 hidden layers of 64, input [C, C'], output f's parameters. One g per
space: the counts g's f reads count X only (x = log(1 + counts)) and outputs NB (mean, n) per bin;
the p g's f reads p X only (x = log(max(p, 1e-3))) and outputs log-normal (μ, σ) per bin.
- **A (h12)** per-bin affine: log mean (or μ) = a + b·xᵢ; one dispersion (NB n or σ). g → (a, b, dispersion).
- **B (h17)** per-bin monotone curve: 12 knots at fixed quantiles of the source x over the g's
  training set; g outputs positive steps between knots (always monotone) and a dispersion value at
  each knot, interpolated by level.
- **C (h13)** convolve x with a 33-bin (825 bp) kernel that g outputs, initialised to the identity,
  unconstrained; then B's curve.
- **D (h14)** 4 dilated conv layers × 32 channels, kernel 5, dilations 1, 2, 4, 8 (view 61 bins ≈
  1.5 kb); g outputs a per-channel scale and shift for each layer (FiLM); per-bin output head for
  the two distribution parameters.

**Two versions of g, both judged.** One g per track (7; the assay entry is constant inside) and
one g across all 7 tracks (all pairs, assay in C).

**Three models per (rung, version, space, seed).**
- the real g;
- **no-covariates twin**: (C, C') re-drawn across the training pairs at every step, so the
  covariates carry no information;
- **labels-as-ids twin**: one fixed permutation of (C, C') across the training pairs, kept for all
  of training (seed-dependent).
Same architecture, data, steps and seed for all three.

**Loss.** Negative log-likelihood: NB for counts, log-normal for p. p targets are floored at 1e-3
in the training loss only; scoring uses the real values. 3 seeds (0, 1, 2).

**Scoring** (reuse `baseline_rungs.py` conventions and closed forms; `nb_crps_mean` for NB):
distances D = NB CRPS (counts) / log-normal CRPS (p), and Spearman, on all, non-zero and top-1%
bins (top 1% by the real X'), mean over bins per pair, then per track, then macro over the tracks
of a mark class (DNase; narrow H3K27ac, H3K4me3, H3K4me1; broad H3K27me3, H3K36me3, H3K9me3). Also
the CRPS split, reported only.
- **Held-out chromosomes**: every trained pair, chr19 + chr21 (chr22 kept as `val`).
- **Law test**: every never-trained arm → arm pair within each track (for each version of g),
  reported by knob combination. Count-space pairs where both arms are p-only are identical; keep
  them, flag them.
- **Depth law**: on depth → depth pairs, the predicted total count scale vs the true depth ratio.
- **Shuffle**: C' replaced by the C' of another arm of the same track whose target differs in that
  space (seeded draw, separate per space).
- **Swap**: C' = C; median |log(predicted mean / X)| over bins with X > 0.
- **Seed wobble**: the largest pairwise |Δ| of D over the 3 seeds.

**Checks and bars** (all set; `rule: all`; written in each hypothesis):
- beat the twin: D_twin(no-covariates) − D > 2 × seed wobble, per mark class, per space, CRPS all
  bins and top 1%, both versions of g;
- law test: beats **each** twin by > 2 × seed wobble; depth scale within 10% of the depth ratio;
- beat the rung below (B > A, C > B, D > C): > 2 × the larger seed wobble;
- main claim (`h15`) only: shuffle — the advantage over the no-covariates twin falls to within 2 ×
  seed wobble; swap — median |log ratio| < 0.1;
- the main claim judges **the rung chosen on chr22**: the lowest rung within seed wobble of the
  best, then scored once on chr19 + chr21.
- noSolution and QuantileMatching are references only.
Compute every check's value against its bar into a checks JSON per rung. **Do not tick boxes.**

**Figures** (plan "Visualizations and report"; all nine): ladder plot; knob × rung heatmap
(relative gain over the no-covariates twin); law-test grid per track; depth-law plot; the learned
f (A: a, b vs knob value; B: g's curve per arm over QuantileMatching's per-pair curve; C: kernel
per knob); peak meta-profiles ±2 kb; track snippets over **10 kb loci** picked by the fixed rule
(per knob: the locus around the top-1% bin with the largest |X' − X|, one around a random top-1%
bin, one around a random background bin; chr19/chr21; seeded); calibration (PIT histograms,
secondary); checks card. One markdown report with PNGs per rung.

## 4. How to split the work (suggested — load `orchestrate` and `dispatch`)

Be the foreman: brief builders, read their diffs, merge, re-brief. Every brief names the ground
truth the builder must verify itself (section 1) — never pass a premise you have not re-checked.

| chunk | what | depends on | parallel with |
|---|---|---|---|
| C0 | covariate table + test | — | C1 skeleton, C3, C4 |
| C1 | shared harness: pair list (with exclusions), window sampler reading npz lazily, covariate encoder, g MLP, the f-form interface, both twins, NLL losses, early stopping on chr22, checkpoint, prediction of per-bin parameters on chr19/21/22, timing log | interface first, C0 to finish | C3, C4 |
| C2a–d | f-forms A, B, C, D against the C1 interface, each with unit tests (monotonicity for B, identity init for C, shapes, a planted-effect recovery test: a synthetic pair with a known depth ratio must be recovered by A) | C1 interface | each other |
| C3 | scorer: NB / log-normal CRPS with per-bin parameters, subsets, Spearman, law-test pair list, depth law, shuffle, swap, seed wobble, aggregation to mark class, checks JSON | — | C1, C2, C4 |
| C4 | figures 1–9 against a fixed results schema, developed on synthetic results | results schema from C3 | C1–C3 |
| C5 | SLURM: one array over the 576 runs (A and B first in index order), train → predict → score in one task or train then a CPU scoring array; the task table is a pure function of the manifest | C1–C3 | C4 |
| C6 | pilot: design A, one histone track, both twins, both spaces, seed 0 — on Nibi; measure time and memory per run; fix; **then submit everything** (no PI stop) | C5 | C4 |
| C7 | aggregate, figures, one report per rung, rsync evidence down, link from the notebook, draft the checks readings for the PI | runs | — |

Mechanical checks before the pilot: `pytest tests/ -q` green; a CPU smoke run of each rung on one
chromosome slice locally; `sbatch --test-only` on Nibi.

## 5. What must exist in the morning

1. **`plan/T118_STATUS.md`** (tracked, pushed; update it as you go, not only at the end):
   what is finished, running, pending or failed, per rung; Nibi job ids; output paths; measured
   time per run; "Choices I made"; anything that disagrees with the plan; anything blocked on the
   PI. Written in plain technical English, no crux ids.
2. For every rung that finished: `cruxvault/results/<hid>/report.md` + PNGs + checks JSON +
   `FIR_PATH.txt` (Nibi path), linked from the hypothesis's `## Artifacts`.
3. For every rung that finished: a drafted reading of each check (value vs bar, met or unmet) in
   the report — **not** ticked in the hypothesis.
4. The code, tests and SLURM scripts committed and pushed on `exp/t118-counterfactual-f`; PR #48
   description updated with a short progress summary.
5. Jobs still running are left running; the status file says when each is expected to finish.

## 6. Known traps

- The pseudoreplicate oracle is **dropped**; do not use it anywhere.
- `nb_suite` clips p at 1 − 1e-9; fine for NB outputs, but do not use it to push an NB to Poisson.
- Most count bins are 0; top 1% ties at the cut are broken by genomic order (as in the baseline).
- The labels-as-ids twin can match g on trained pairs by design; that is expected, not a bug.
- A 10 GB MIG slice limits memory: read windows from disk (npz members are per chromosome; load one
  chromosome's arrays for the products a g needs, or pre-extract to `.npy` once under
  `/project/def-maxwl/mforooz/t118/ladder/cache/` with memory-mapping).
- The across-track g needs all 130 products of one space (~65 GB unpacked): memory-map, never load
  all.
- Law test volume: ~2 100 arm→arm pairs per space per version; score them in CPU jobs from saved
  predictions if it does not fit the training task.
