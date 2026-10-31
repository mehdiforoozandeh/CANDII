---
id: h15
type: idea
schema: 2
title: A generator conditioned on the source and target covariates produces the transformation that maps a track onto its counterfactual arms, in both count and −log10 p space, beyond the same model trained with scrambled covariates
parent: q1
status: running
rule: all
measurement: "g(C, C') outputs f, and f(X) predicts X' as NB (n, p) per bin for counts and log-normal (μ, σ) per bin for −log10 p; g is what is trained, in two versions: one g per track (7 g's) and one g across all 7 tracks, both scored under the same checks (both versions decide pass/fail); trained on base↔arm pairs of the t112 counterfactual corpus in both directions, validated on chr22, scored on chr19 + chr21 (blacklist removed) and on never-trained arm→arm pairs; D = NB CRPS / log-normal CRPS and Spearman, on all, non-zero and top-1% bins; competitor = the same model trained with C and C' scrambled across pairs; design record plan/T118_COUNTERFACTUAL_F.md"
replicates: 7 tracks (1 DNase, 3 narrow, 3 broad marks) x base↔arm pairs in both directions (19 arm products per histone track, 9 for DNase; 246 pairs) x 3 seeds of each g (7 per-track g's and 1 across-track g)
neutral_optout: "PI ruling 2026-09-23: no check voids the run; the swap check (C' = C returns X) gates the claim instead of acting as a control"
verdict: 
metric: 
created: "2026-09-23T14:40:11"
updated: "2026-09-25T02:03:08"
null_approved: "2026-09-25T00:55:22"
null_hash: db64666c20881711
lock: a7e5a25e3939818d
locked: "2026-09-25T02:03:08"
lock_at: running
---

# h15 — A generator conditioned on the source and target covariates produces the transformation that maps a track onto its counterfactual arms, in both count and −log10 p space, beyond the same model trained with scrambled covariates

Parent:: [[q1_do_the_recorded_experimental_covariates_]]

## ELI5

If you know how two versions of one experiment were processed, a model can turn one into the other without memorising each case.

## TL;DR

A generator g reads the covariates C and C′ and outputs a transformation f; f(X) predicts X′ per bin. Pairs come from [[t112_build_the_counterfactual_arms_for_t|the counterfactual-arms task]]. The competitor is g trained with scrambled covariates. So far, not judged: g beats that twin on held-out chromosomes almost everywhere; the gain shrinks on never-trained arm → arm pairs. Every chosen design misses the swap check and the depth law.

## Expect

g(C, C′) → f beats the same model trained with scrambled covariates, per mark class, in counts and −log10 p separately, with no further bar on the size of the gain (PI 2026-09-25).

_Pre-run: locked 2026-09-25, before the row-1 training array was submitted._

## Null
Normalization: the covariates are ignored — g does no better than the same model trained with C and C' scrambled across pairs.

## Problem Statement

CANDI's zero-shot claims rest on covariate conditioning that cannot be isolated inside CANDI. Here each pair differs in one recorded knob; the question is whether the covariates alone reproduce its effect, as a relation between C and C′ (PI 2026-09-25). It replaces the one-warp design of [[h1_conditioning_on_the_recorded_sequencing_|the depth claim]] and [[h2_conditioning_on_the_recorded_run_type_pr|the run-type claim]].

Why this design (PI, 2026-09-25): one-warp "is not using C and C' at all", so beating it does not show that the covariates did any work. The PI wanted a model that learns how C and C′ change X into X′: "if C is depth and its 15M and if C' is 30M, then X' peak must be twice as big". So the competitor became the same model with the covariates scrambled, and the pseudoreplicate oracle was dropped once it proved not to be a floor ("let's forget about the oracle"). The PI also ruled that training runs both ways (base → arm and arm → base), so the swap check gates the claim.

## Idea / Hypothesis

A generator conditioned on the source and target covariates produces the transformation that maps a track onto its counterfactual arms, in both count and −log10 p space, beyond the same model trained with scrambled covariates

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
<!-- Bars set by the PI 2026-09-25 (defaults accepted: pass/fail is not the priority now). -->
- [ ] beatstwin: (twin = the no-covariates twin, (C, C') re-scrambled at every step) D_twin − D_g > 2 x g's seed wobble (max pairwise |Δ| of D_g over 3 seeds; D_g = the distance of the prediction f(X) made with the f that g outputs), per mark class (DNase; narrow H3K27ac/H3K4me3/H3K4me1; broad H3K27me3/H3K36me3/H3K9me3), counts and p separately, CRPS all bins and top 1%, on chr19 + chr21; no further bar on the size of the gain (PI 2026-09-25)
      fails-if:: the covariates add nothing a scrambled-covariate twin of equal capacity cannot already do
      discriminates:: true
- [ ] lawtest: (against both twins: no-covariates and labels-as-ids) on never-trained arm → arm pairs within each track, scored on chr19 + chr21 and reported by knob combination, g beats each twin by more than 2 x seed wobble. For depth → depth pairs the predicted count scale must follow the depth ratio (within 10% of the depth ratio; PI 2026-09-25)
      fails-if:: g keeps one map per trained (C, C') pair and has no answer for a combination it never saw
- [ ] shufflecollapse: at scoring, C' replaced by the C' of another arm of the same track whose target differs (drawn separately for counts and p); g's advantage over the no-covariates twin must fall to within 2 x seed wobble (PI 2026-09-25)
      fails-if:: g does not use C': the f it outputs does not change when told the wrong target covariates
- [ ] swapreturn: at scoring, C' set equal to C; the f that g outputs must return X: median |log(predicted mean / X)| over bins with X > 0 below 0.1 (PI 2026-09-25)
      fails-if:: told the target is the source itself, g's f still transforms X, so g is not steered by C'

## Planned Intervention

Run under [[t118_learn_the_covariate_conditioned_tra|the generator task]]; design in `plan/T118_COUNTERFACTUAL_F.md`, run record in `plan/T118_STATUS.md`.

- 7 tracks; base ↔ arm training pairs, both directions; chr22 validates; chr19 + chr21 score.
- f is one of the designs A–D of [[q4_how_much_capacity_does_a_covariate_condi|the capacity question]]; chr22 picks the judged design.
- g per track and g across tracks, 3 seeds each; NB for counts, log-normal for −log10 p.
- Twins: no-covariates and labels-as-ids. The pseudoreplicate oracle was dropped (PI 2026-09-25).

## Run Links

- Row 1 (the judged runs, 576): Nibi `/project/def-maxwl/mforooz/t118/ladder/runs`, kit K3; train 22657297 (retry 22667089), law 22667088/22667090/22667091/22669098/22676228, aggregation 22678427–22682910 (`plan/T118_STATUS.md`, "Nibi jobs").
- Row 2 (exploratory, 576) and the shuffled-bin twin (192): `/project/def-maxwl/mforooz/t118/row2/runs`, `/project/def-maxwl/mforooz/t118/row2_xshuf/runs`; aggregate `/project/def-maxwl/mforooz/t118/row2/agg` (jobs 23087020–23088588).
- Reference rungs: `/project/def-maxwl/mforooz/t118/rungs_v2/`.
- Never-trained knob readings: `/project/def-maxwl/mforooz/t118/knobgen/`, extraction job 24641277.

## Artifacts

- [Main claim: which design is judged, and its drafted readings](results/h15/report.md)
- [Checks of the chosen designs, with shuffle and swap](results/h15/checks_main.json)
- results/h15/results_summary.tsv every run's scores per track and class
- [Row 1, row 2 and the shuffled-bin twin: every cell and the drafted readings (exploratory 2026-10-01)](results/row2/GRID_SUMMARY.md)
- [The 2 × 4 grid with both metrics and the noSolution and QuantileMatching references](results/row2/grid.md)
- results/row2/FIR_PATH.txt Nibi paths of the row-2 runs, the twin runs and the aggregate
- [Trained versus never-trained pairs: depth law and every knob (2026-10-19)](results/h19/report.md) and its [figures](results/t118/knob_generalization/report.pdf)
- [Reference rungs noSolution and QuantileMatching](results/t118/rungs_v2.md)
- [Team overview, 2026-09-25](results/t118/team_overview_2026-09-25.md) — published https://claude.ai/artifact/Gj3KNBCaK1efJpfTQ8KV7V
- [Lab deck: the transformation grid](presentations/t118-grid/index.html) — published https://claude.ai/artifact/1mFz6C5jLaG1Ctc8JN572C; vault copy [DECK.md](presentations/t118-grid/DECK.md)

## Findings

Exploratory drafts (2026-09-25 to 10-19), not ticked. Bars are 2 × the real g's 3-seed wobble, per mark class.

- Chosen: C (across-track g), D (per-track, counts), A (per-track, p).
- Beats the no-covariates twin: 5–6 of 6 cells; shuffle 4–6 of 6.
- Swap unmet everywhere: median |log(mean / X)| 0.19–0.22 in counts, 0.46–0.55 in p (bar 0.1).
- Depth law unmet: worst-pair error 0.19–1.36 (bar 0.10). The real data obey the law ([audit](results/t122/AUDIT.md)).
- Law test on never-trained pairs against that twin: 4–6 of 6 in counts, 1–3 of 6 in p.
- Relative gain over that twin, median of 16 configurations, trained / never-trained one knob / two knobs: counts 0.29 / 0.11 / 0.16; p 0.15 / 0.06 / 0.004.
- Caveats: exploding p pairs make p means unreadable; the DNase MAPQ arms equal the base; no oracle-scaled / scale-error split exists here.
