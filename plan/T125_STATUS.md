# t125 status: the depth-offset count model and the counts → p map h

Orchestrator log for t125. Read `plan/T125_HANDOFF.md` first, then
`plan/ETA_OFFSET_AND_H_DESIGN.md`. The orchestrate plan is `.orchestrate/plan_t125.md` in this
worktree (git-excluded).

## State

| phase | state |
|---|---|
| 0 verify on Nibi | done 2026-10-11; formula forks with the PI |
| 1 build the count model | building (plan: 24 chunks, `.orchestrate/plan_t125.md`) |
| 2 build the map h | building |
| 3 smoke on Nibi | not started |
| 4 count runs (432) | not started |
| 5 h on the true X′ | not started |
| 6 h on f's NB, chain scores | not started |
| 7 aggregate | not started |
| 8 vault readings | not started |
| 9 lab deck | not started |

## Job ids

None submitted yet. Every id goes here at submission.

## Nibi access

- 2026-10-11: `ssh -o BatchMode=yes nibi` exited 255 (`Permission denied
  (publickey,keyboard-interactive,hostbased)`). All Nibi work stops until the PI runs
  `hpc up nibi`.
- 2026-10-11 01:26 EDT: `ssh -o BatchMode=yes nibi` returns `l2.nibi.sharcnet`. Nibi work resumes.

## Phase 0 findings

Read-only check on Nibi, 2026-10-11. Source: `/project/def-maxwl/mforooz/t112_cf/products/<pid>/provenance.json`
(exact commands), `covariates.json`, `MANIFEST.tsv`, the pipeline SIF scripts, and MACS2 2.2.4 source.

- **DNase P:** ENCODE atac-seq-pipeline v2.2.3 in DNase mode (`encode_task_macs2_signal_track_atac.py`),
  **not** dnase-seq-pipeline v3.0.0. `macs2 callpeak -t TA -f BED -g hs -p 0.01 --shift -75 --extsize 150
  --nomodel -B --SPMR --keep-dup all`, then `bdgcmp -m ppois -S lines(TA)/1e6`. No control: λ = max(λ_BG, λ_10k
  of the treatment's own reads), λ_BG = 150·N/2.7e9. No Tn5 shift. extsize arms change smooth-win to 75 or 300.
- **ChIP P:** chip-seq-pipeline v2.2.2. `callpeak -t TA -c CTL -f BED -g hs -p 0.01 --nomodel --shift 0
  --extsize FRAGLEN --keep-dup all -B --SPMR`, then `bdgcmp -m ppois -S lines(TA)/1e6`. λ_local = max(λ_BG,
  λ_d, λ_1k, λ_10k); control scaled to the smaller set (`--scale-to small`). FRAGLEN from cross-correlation.
  ctlid none: λ = max(λ_BG, λ_10k of treatment). ratio arms: `--ratio V`, V = k·30M/N_ctl. Two ctldepth q0.25
  arms (C19M16, C19M22) have treatment > control, so MACS2 scales the treatment down there.
- **ppois (tested with the SIF's MACS2):** P = −log10 P(X > floor(obs)) at λ; strict tail; no pseudocount;
  5-decimal rounding; float32 bigwig.
- **25 bp binning:** P = base-pair-weighted **mean** of per-base −log10 p over the bin (0 where no interval).
  Counts = **read-span overlap**: +1 to every bin the aligned read touches (no shift, no extension, no strand),
  chr1–22 + chrX, from the same nodup tagAlign (30M ChIP, 50M DNase). pe arms: each mate is a line.
- **Inputs for h:** `covariates.json` has depth, fraglen, control.reads, knob_value (ratio V or control depth),
  read_length, run_type. MACS2's N is all tagAlign lines (incl. chrM, contigs), not the sum of counts25.

## PI decisions

- 2026-10-11: learned MLP h = 2 × 64, with the same training settings as the t118 g (optimizer, learning rate, epochs, early stopping on chr22).
- 2026-10-11, formula h (from the phase 0 forks):
  - DNase follows the recipe that ran (ATAC pipeline v2.2.3 in DNase mode). Its conflict with the
    "dnase-seq-pipeline v3.0.0" rule is noted, not fixed.
  - Pileup ≈ count × fraglen / (read_length + 24). Bin-only uses the bin; the fragment-window variant
    uses the mean count over the bins one fragment wide.
  - λ = the full MACS2 max of windows (fragment, 1 kb, 10 kb, background) from the binned control, scaled
    as MACS2 does (to the smaller set; ratio V in ratio arms). No control: max(background, 10 kb of the
    treatment); in the chain that window uses f's predicted mean.
  - One Poisson tail per bin, with the caller's rules: strict tail, floor of the observed value,
    no pseudocount, N = all tagAlign lines.

- 2026-10-11, plan forks:
  - Law pass on depth-only pairs only (42 across tracks); `--law-scope all` stays available.
  - Repair training: 1 h seed, 768 fits (64 f × 12 h versions); seed spread quoted from the true-X′ h.
    Upgrade to 3 seeds only if the Nibi smoke shows one fit under 20 min on 8 cores (ask the PI then).
  - Own-noise loss on f's NB: expected Gaussian NLL, Σ_k NB(k)·NLL_k.
  - Own-noise scoring: exact mixture CRPS on log(1 + P); push-only also gets log-scale CRPS; push-only keeps
    native-scale CRPS; native MSE of the mean for both.
  - Control read length = the treatment base's read length; the Nibi smoke reports the formula-vs-real-P error.
    **Superseded the same day:** the crop arms cropped the control too, so every control's read length is
    measured from the tagAlign that ran (read-only Nibi step).

- 2026-10-11: formula + affine h = formula + an affine map of all rung-5 inputs (formula included), zero at init.

- 2026-10-11: the no-covariates twin keeps the depth offset. Real vs twin shows what the covariates add beyond the fixed d′/d scaling; it is not directly comparable with the old grid's twin gap.

- 2026-10-11: rung 5 of the learned h gains the treatment and control log read lengths (21 inputs).

## Log

- 2026-10-11: run started; Nibi down at start.
- 2026-10-11 01:30: Nibi up. Phase 0 reader and the planner dispatched. Chunk ids carry the prefix `t125-`.
- 2026-10-11: wave 0 merged (offset and task table, form A by level, eta model, h data side, formula h, exact push-forward, learned h). Control read lengths measured for 114 products (`tools/t125/hmap/control_read_length.tsv`). Wave 2 part 1 (eta training, eta scoring, SLURM part 1) building.
