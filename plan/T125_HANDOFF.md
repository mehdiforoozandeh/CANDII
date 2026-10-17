# t125 handoff: the depth-offset count model and the counts → p map h, end to end

Written 2026-10-08 by the session that designed t125 with the PI. Read this file first, then the
design note. You are the **orchestrator**. You run the whole of t125, from verification to a lab
deck. Builder and reader subagents do every piece of work, and you keep your own context for
briefing, review, merge and the PI.

## 1. The mission

Build, run, score and present the design in `plan/ETA_OFFSET_AND_H_DESIGN.md` (the design note).
It has two parts:

1. **The count model.** It is the t118 g → f grid with a depth offset. f reads
   eta_in = log(1 + X) − log d and outputs eta_out. The target counts are NB with
   log μ′ = log d′ + eta_out.
2. **The map h.** h takes the predicted NB for X′ and the control, and gives P = −log10 p. It is
   trained and scored against the real p track.

The run ends with three deliverables:

- the result grids and readings in `cruxvault/results/t125/`, with a `GRID_SUMMARY.md`;
- drafted "Answer so far" readings in the two vault questions (no verdicts);
- a prezit lab deck in the style of the t118 deck, published as an Artifact, with its link given
  to the PI.

## 2. Binding rules (from the PI; they do not change)

- **The phase is exploratory.** Run everything and deliver the grids. Do not tick checks, record
  verdicts, close nodes, accept tasks, approve syntheses or merge to main. Do not file new
  hypotheses unless the PI asks.
- **Never resubmit a finished job.** Every t118 job id is in `plan/T118_STATUS.md`. That includes
  the 576 row-1 arrays, the smoke array 22851946, the re-score job 22851953 and every row-2
  array. t125 writes to new output directories only.
- **If Nibi ssh exits 69 or 255,** stop all Nibi work. Note it in `plan/T125_STATUS.md` and wait
  for the PI to run `hpc up nibi`. Never authenticate.
- **Always pass `-o BatchMode=yes`** to ssh, and `-e 'ssh -o BatchMode=yes'` to rsync. Quote
  remote globs. Never run a bare `git fetch --all`: the `firmerge` remote goes to Fir.
- **Bring every science-affecting choice to the PI before you act on it:** a threshold, a data
  subset, an approximation, a change to a loss or to the scoring. Ask it as a multiple-choice
  question with a recommended option. The PI decides by MCQ forks and likes them.
- **Keep `plan/T125_STATUS.md` updated and pushed as you go.** Write every job id at
  submission.
- **The global rules in `~/.claude/CLAUDE.md` apply.** They cover ASD-STE100 style, ELI5 and
  TL;DR for long replies, the "What I need from you" block, and confirming before deletes,
  overwrites and pushes. The PI has authorized commits and pushes of t125 work to
  `exp/t118-counterfactual-f`.
- **Name vault nodes by their title, paraphrased, never by a bare id.** Do not use crux process
  words in chat (see the crux skill, "Voice").

## 3. Where things are

- **Repo:** `/Users/mforooz/Desktop/research/libbrechteam@sfu/CANDII`.
- **Work in the worktree** `.claude/worktrees/counterfactual-f`, on branch
  `exp/t118-counterfactual-f` (draft PR #48). t125 is a child of t118, so its code lands on that
  branch. It gets no branch or PR of its own.
- **Subagent worktrees** are cut from `exp/t118-counterfactual-f`, never from `origin/main`.
  Merge `origin/main` into the branch as you go (merge, not rebase).
- **Python:** `conda activate candii`, and `export PYTHONPATH=$PWD/src` in the worktree.
  Otherwise pytest tests the main checkout.
- **Design:** `plan/ETA_OFFSET_AND_H_DESIGN.md`, the source of truth for every fork the PI
  decided.
- **t118 history:** `plan/T118_COUNTERFACTUAL_F.md`, `plan/T118_ROW2_SPEC.md` and
  `plan/T118_STATUS.md`. The status file holds every job id, the measured costs and the
  row-2 readings.
- **t118 code:**
  - `tools/t118/ladder/` holds `fforms/form_{a,b,c,d}{,2}.py`, `base.py`, `model.py`, `train.py`,
    `pairs.py`, `score.py`, `aggregate.py`, `report.py`, `figures.py` and `smoke.py`.
  - `tools/t118/covariates.tsv` has one row for each arm.
  - `slurm/t118/` holds the scripts.
  - `tests/test_t118_*.py` holds 428 tests.
  - The t118 f has no depth offset: depth is only a z-scored input of g (`encoding.py`).
- **Nibi** (reach it with `ssh -o BatchMode=yes nibi`):
  - `/project/def-maxwl/mforooz/t112_cf/products`: the source products, counts and p, one set
    for each arm.
  - `/project/def-maxwl/mforooz/t118/ladder/{cache,runs,agg,code/K5,code/K6}`: the t118 cache,
    runs, aggregates and code kits.
  - `/project/def-maxwl/mforooz/t118/row2/` and `.../row2_xshuf/`: the row-2 outputs.
  - Put t125 under a new directory, `/project/def-maxwl/mforooz/t125/`. Ship code as a new kit
    (K7, …) with a `GIT_SHA`, as K5 and K6 were shipped.
- **Old-grid results** for comparison are in `cruxvault/results/row2/`: `results_summary.tsv`,
  `checks_*.json`, `grid.md` and `GRID_SUMMARY.md`. This directory is gitignored and present
  locally.
- **The vault:**
  - The questions are q5 (whether a depth offset on f changes how well g → f translates counts)
    and q6 (how closely a map h reproduces the −log10 p track as h's complexity and inputs grow).
    Both are under q1.
  - The task is t125, a child of t118.
  - Run crux from inside `cruxvault/` with `python ~/.claude/skills/crux/scaffold/crux.py <verb>`.
    Routine lint is `validate --check=tree,tasks`. 72 missing-output errors were there before
    t125 and are not yours.
- **Cockpit:** `crux serve` on :8787 serves this worktree's vault.
- **The t118 lab deck, which is your template:**
  - Source: `cruxvault/presentations/t118-grid/` (`deck_src.html`, `build_deck.py`,
    `snippet.json`, `index.html`).
  - Published at https://claude.ai/artifact/1mFz6C5jLaG1Ctc8JN572C (version 5).

## 4. Phases

Use the `orchestrate` skill. The first step is a plan file, `.orchestrate/plan_t125.md` in the
worktree. It is git-excluded. Do not reuse `.orchestrate/plan.md`, `plan_row2.md` or
`plan_xshuf.md`: they belong to t118. Its chunks must not overlap, and each chunk has acceptance criteria. Route every spawn
through the `dispatch` skill. Each brief names the ground truth that the agent must verify for
itself, and where it is.

### Phase 0: verify on Nibi (read-only)

1. **The DNase p recipe.** Find how the t112 DNase p-value tracks were made: which pipeline and
   caller, the commands, and what stands in for the control. DNase has no control. h's DNase
   path must follow that recipe.
2. **The 25 bp binning.** Find how the per-base p values became 25 bp bins in the t112
   products: mean, max, or another rule. Find the same for the counts track: read starts or
   coverage, and which strand handling.
3. **The ChIP caller commands.**
   - Find the local-λ windows, the genome background, how the control is scaled to the
     treatment, and the pileup extension (extsize or fragment length).
   - Find how the no-control arms (`ctlid none`) and the `ratio` arms are handled.
   - Read the actual logs or the WDL that ran. A doc is not enough.

Write the findings in `plan/T125_STATUS.md`. **If a finding makes the formula ambiguous, bring
it to the PI as an MCQ before Phase 2.**

### Phase 1: build part 1, the count model

- **Depth offset:** eta_in = log(1 + X) − log d, with d in millions of reads. The output is
  log μ′ = log d′ + eta_out.
- **Forms:** all eight t118 forms, both rows. g reads C and C′, with depth kept.
- **Spread rule:** each design's spread sees what its mean sees.
  - A and B set the spread by level.
  - C and D set it by level and context.
  - Row-1 A changes from one spread per pair to a spread by level.
  - Add the old-spread row-1 A arm.
- **Scoring:**
  - Keep CRPS with its oracle-scaled and scale-error parts.
  - Add the MSE on log(1 + counts) of the predicted mean.
  - Report all bins and the top 1% of bins by true target value.
  - Keep the depth-law reading on depth-only pairs.
- **Tests and smoke:** write tests first. Extend `smoke.py`, and run it locally.

### Phase 2: build part 2, the map h

- **Exact push-forward:** list k = 0, 1, … until the NB tail is below 1e-6. Compute the mean
  and CRPS exactly: the sorted one-pass form for a monotone h, and the pair sum for the
  Gaussian mixture.
- **Versions of h:**
  - the formula alone, bin-only and fragment-window;
  - learned affine h and MLP h at rung 1 and at rung 5;
  - the formula plus an affine or MLP correction at rung 5.
- **One shared h,** with the assay as an input.
- **Uncertainty:** push only; and push plus own noise (a Gaussian on log(1 + P)).
- **Training sources:**
  - the true X′;
  - f's NB, from seed 0 of each real f (64 models).
- **Losses:** listed in the design note.
- **Scoring:** score every learned h twice, on the true X′ and in the chain on f's NB.

**Not yet decided: the MLP size and the training settings of the learned h.** Bring them to the
PI as an MCQ. Recommend the smallest choice that matches g (2 × 64).

### Phase 3: smoke on Nibi

Run a few tasks for each part. Measure the time and memory, and set the array sizes from them.

### Phase 4: the count runs

There are 432 runs:

- 2 rows × 4 designs × 8 versions of g × 2 models (real and no-covariates twin) × 3 seeds = 384;
- plus 48 for the old-spread A arm.

### Phase 5: h on the true X′

These are small fits that run on CPU.

### Phase 6: h on f's NB, and the chain scores

This phase waits for Phase 4: it needs seed 0 of each real f.

### Phase 7: aggregate

Write `cruxvault/results/t125/GRID_SUMMARY.md` and the result tables.

- **Part 1:** compare each design with the same design in the old grid, and each model with its
  no-covariates twin. Separate a gain in the mean from a gain in the spread.
- **Part 2:**
  - the complexity × inputs grid for h;
  - the two uncertainty variants;
  - the repair reading: the h versions compared on f's NB;
  - the chain against the old grid's p results.
- **Rules for every reading:**
  - Quote it with its seed spread.
  - A gain counts only when it is larger than 2 × the larger seed wobble.
  - No correlation is a headline.

### Phase 8: the vault

- Draft the "Answer so far" sections of the two questions. Write them as readings, with no
  verdicts.
- Add a `## Progress` section to t125.
- Keep the evidence in `cruxvault/results/t125/`.
- Do not tick, close or accept anything.

### Phase 9: the lab deck

Use the `prezit` skill. The new deck goes in `cruxvault/presentations/t125-eta-h/`. Copy the
t118 deck's build pattern: `deck_src.html` with a `__DATA__` placeholder, plus `build_deck.py`.

The PI's deck style (see also the `lab-deck-style` memory):

- **Structure:** motivation, background, method, results, discussion, next steps.
- **Start from the basics.** The audience is lab members who do not know the design.
- **Use figures for everything.** The one exception is a grid concept, which is shown as a table.
- **Keep text low.** Label every panel, and divide multi-panel figures visibly.
- **Name the version of the model** (for example, one g per track) in the title or the legend.
- **Never use abbreviated labels.**
- **Keep the negatives.**

Check the layout in the browser pane at 1440 × 810. Then publish it as a new Artifact. Mirror
the link into `cruxvault/results/t125/DECK.md`, and give the PI the link.

## 5. Known pitfalls

- **Nibi nodes with a CVMFS "Input/output error":** t118 used `--exclude=c128,c166,c537,c371`.
  Check `plan/T118_STATUS.md` for the current list.
- **sbatch `--export`** cuts a value that contains commas. Export the variable in the shell and
  pass `--export=ALL`.
- **A dependency on an already-finished job is refused.** Chain only on live jobs.
- **The Bash tool shell is zsh.** An unquoted `$VAR` does not split on spaces, so use arrays.
  macOS has no `timeout`.
- **Another session can move HEAD in the main checkout.** Scope every `git add` to named files.
- **In the old grid, p-space explosions dominated the means.** Report how many (pair, seed)
  records have CRPS above 20, and never let one hide inside a mean.

## 6. When you are done

All of the following must be true:

- Every run has its scores, and every planned reading is in the grids.
- The vault holds drafted readings.
- The deck is published, and its link is given to the PI.
- `plan/T125_STATUS.md` is complete and pushed.

Then give the PI one short message. It holds the deck link, the three most important readings in
plain words, and a "What I need from you" block. That block lists the open calls: ticks,
verdicts, the merge, and cache deletion.
