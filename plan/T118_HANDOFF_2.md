# t118 handoff 2 — after the ladder run (2026-09-25)

Written by the agent that built and ran the A–D ladder, for the agent that continues. Read in this
order, in full: (1) this file; (2) `plan/T118_STATUS.md` — results, choices, failures, open problems;
(3) `plan/T118_COUNTERFACTUAL_F.md` — the design authority, every PI ruling; (4) the first work order
`plan/T118_HANDOFF.md` §2 (hard rules) — still binding. Where these disagree, the design plan wins.

## 1. Where things stand

- All 576 pre-registered runs are trained, scored, law-tested and aggregated. Reports, figures and
  checks for each design are in `cruxvault/results/{h12,h17,h13,h14}/` (A, B, C, D) and the main
  claim in `cruxvault/results/h15/` — main checkout, gitignored, rsync'd from Nibi. Each hypothesis's
  `## Artifacts` links its report. Nothing is ticked; no verdict is recorded; the task is not done.
- Headline, drafted: covariates help in p space with design A and in counts with B/C/D; the law test
  is strong in counts; the depth law and swap checks are unmet everywhere; a few p-space
  "upscaling" pairs explode the mean CRPS in B/C/D (diagnosis in the status file).
- Exploratory design X (the PI's idea: g reads the bin value too) was tested on 2 tracks: no p-space
  explosions, weaker than B–D in DNase counts. Not pre-registered.

## 2. Open work, in the order I would take it — confirm with the PI before spending compute

1. **The p-space explosion needs a principled fix (PI's explicit ask).** Ruled out by the PI:
   capping σ, and report-only. Known cause for B/C: the lowest knot sits at the floor
   x = log(1e-3), where ~0.1% of source bins lie; its σ is unconstrained and drifts (10.9 in the
   worst pair), and linear interpolation spreads it over the low bins, where a log-normal's mean
   (median × exp(σ²/2)) makes CRPS explode. First, find why **D** explodes too (no knots) — look at
   D's per-bin σ on the exploding pairs (`C12M02` DNase, upscaling pairs such as `depth__3.75M` →
   base, `dedup__off` → base). Then propose fixes to the PI in plain terms, with what each costs;
   candidates to evaluate, not to adopt on your own: knots only where training data exist; a σ
   that is a smooth function of level rather than free per knot; design X's form; training on the
   scoring rule. Do not pick one yourself — the loss and the scoring are PI rulings.
2. **Check the depth-law computation before believing it.** Every design misses by 30–140%.
   Confirm on one depth → depth pair by hand (predicted total count scale vs the true read-depth
   ratio from the manifest) that the check measures what the plan says. The code is
   `tools/t118/ladder/score.py` (depth-law records) and `aggregate.py` (the check).
3. **Design X, if the PI wants it continued.** It is outside the pre-registered ladder. Making it a
   rung needs a new hypothesis the PI approves (crux: propose, then `hypothesize`, null and checks
   through the crux-null / crux-verifiables agents). To run it at scale, `aggregate.py`,
   `figures.py` and `smoke.py` must accept rung X (they are limited to A–D); scripts:
   `slurm/t118/explore_x.sh` (GPU) and `slurm/t118/explore_x_cpu.sh` (CPU).
4. **Draft the PI's signature items** once 1–2 are settled: tick readings per hypothesis (drafts
   only, from each `checks_<rung>.json`), a findings paragraph per hypothesis, the task's `done`
   output. Never run `crux close`, `crux task accept`, `crux approve`, or merge to main.

## 3. Ground truth to verify yourself before relying on it

| what | where | check |
|---|---|---|
| branch | worktree `/Users/mforooz/Desktop/research/libbrechteam@sfu/CANDII/.claude/worktrees/counterfactual-f`, branch `exp/t118-counterfactual-f`, PR #48 | `git fetch`; in sync with origin; `git log -3` |
| tests | same worktree | `PYTHONPATH=$PWD/src /Users/mforooz/miniforge3/envs/candii/bin/python -m pytest tests/test_t118_*.py -q` → 238 pass (~4 min) |
| runs and aggregate | `nibi:/project/def-maxwl/mforooz/t118/ladder/{runs,agg,explore_x}` | 576 dirs with `LAW_DONE`; `agg/checks_{A,B,C,D,main}.json` |
| code on Nibi | `.../ladder/code/K3` (the 576 runs: Python 34e5705, SLURM 82cf8db), `K4` (+ design X) | `cat code/K3/GIT_SHA` |
| cache | `.../ladder/cache/`, 260 files, 109 GB | keep; deleting is the PI's call |
| evidence | main checkout `cruxvault/results/{h12,h17,h13,h14,h15}/` | report.md + figures + checks + `FIR_PATH.txt` |

## 4. Hard rules (short form — the full list is `plan/T118_HANDOFF.md` §2)

- One ssh at a time: `ssh -o BatchMode=yes nibi …`; rsync with `-e 'ssh -o BatchMode=yes'` and
  `-rt` (not `-a`: local permission errors). **Exit 69 or 255 = tunnel down: stop Nibi work, write
  it in the status file, wait for the PI to run `hpc up nibi`. Never authenticate.** It dropped
  three times on 2026-09-25.
- No tool call over 240 s; poll long work with a background loop (zsh: split with `read a b <<< "$o"`,
  never unquoted `$var`).
- GPU jobs: only `#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1`, `--account=def-maxwl`, job
  names `t118L_*`, never resubmit anything `t112_*`. If the 10 GB slice nodes show `mixed-` in
  `sinfo`, they are held for another job — use CPU (a CPU job carries no gres line).
- Outputs under `/project/def-maxwl/mforooz/t118/`; `/scratch/mforooz` is over quota.
- Exclude nodes `c128,c166,c537` (CVMFS read errors on 2026-09-25) with `--exclude`.
- Do not chain `--dependency=aftercorr` onto a big array (SLURM held the whole array); submit
  directly for finished runs, verify by output files.
- Aggregation jobs all write one `agg/results.json`: run them one after another
  (`--dependency=afterany:<previous>`), never in parallel.
- Local python `/Users/mforooz/miniforge3/envs/candii/bin/python`; `export PYTHONPATH=$PWD/src` in
  any worktree. No new dependency or environment change.
- Merge `origin/main` into the branch as you go; commit and push often; end commits with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Cut subagent worktrees from this branch.
- Keep `plan/T118_STATUS.md` updated and pushed. In anything the PI reads: plain technical English,
  no crux ids, name hypotheses by content. Every published artifact also gets an md copy with its URL
  under `cruxvault/results/`.

## 5. Left on disk (check before tidying; do not delete without the PI)

- Chunk worktrees under `/Users/mforooz/Desktop/research/libbrechteam@sfu/.orchestrate-wt/CANDII/t118-*`
  (branches `chunk/t118-*`), all merged into the working branch.
- `.orchestrate/plan.md` in the worktree: the build plan and its log.
- The 109 GB cache and the `pilot0/` and `explore_x/` directories on Nibi.
