---
id: q1
type: question
schema: 2
title: Do the recorded experimental covariates carry enough information to map one measurement of a track onto another measurement of the same underlying material?
parent: root
status: open
stale: false
created: "2026-09-01T02:16:11"
updated: "2026-09-01T02:16:11"
---

# q1 — Do the recorded experimental covariates carry enough information to map one measurement of a track onto another measurement of the same underlying material?

Parent:: [[candii]]

## ELI5

If two labs measure the same thing with different machine settings, can the settings alone
tell you how to turn one lab's numbers into the other's?

## TL;DR

CANDI is conditioned on four experimental covariates on both the input and the output side, and
everything it claims about zero-shot imputation rests on that conditioning doing real work. Inside
CANDI the question cannot be answered: the covariate pathway and the imputation trunk are
entangled, and the recorded depth response is a closed-form thinning identity rather than something
learned — `AGENTS.md` §7.2 rule 3 records a told-depth slope of exactly 1.0000 because
`log2_mu = (depth − depth_center) + eta`, and NB is closed under thinning with `n` preserved. This
question asks the same thing in isolation: a testbed with no imputation in it, given a source track
and both sides' covariates, asked for the target track. It is settled by whether covariate
conditioning closes a gap that covariate-free normalization leaves open.

## Question

Do the recorded experimental covariates carry enough information to map one measurement of a track onto another measurement of the same underlying material?

## Protocol

<!-- optional (engine 1.4 / spec 11): the rules locked BEFORE any run — endpoints, arms,
     thresholds, scope. `crux deck` surfaces this as the "rules locked up front" note. -->
Locked before any run under this question:

- The competitor is **not** the uninformed baseline. A covariate-free monotone map on the value
  axis is fit first, and every headline is measured against it.
- No child of this question may quote a genome-wide **correlation** as a headline metric. Depth and
  run type push the level in opposite directions and partly cancel, so a correlation over the whole
  genome yields a false null.
- The conditional entropy among the covariates is measured on the **panel actually used**, before
  any steering result from that panel is read. `AGENTS.md` §7.2 rule 5 records
  `H(run_type | assay_id, read_length) = 0.000 bits` on the shipped 8-assay panel (n = 26 `T_`
  records) against 0.551 bits on the full EIC panel, and `DATA.md` (§"The exposure covariates are
  collinear") makes the same requirement.
- Every number quotes its noise floor and the panel and recipe that floor was measured on
  (`AGENTS.md` §7.2 rule 2).

## Answer so far

_(interpretation — written by the PI/agent; auto-flagged stale when new evidence lands)_

<!-- crux:ledger:start -->
**2 children** · ideas 0/2 done (supported 0, partial 0, refuted 0, inconclusive 0, invalid-run 0)

- `h1` [[h1_conditioning_on_the_recorded_sequencing_|Conditioning on the recorded sequencing depth predicts a target track from a source track, beyond what a single value-axis map already does]] — *idea*
- `h2` [[h2_conditioning_on_the_recorded_run_type_pr|Conditioning on the recorded run type predicts a target track from a source track, beyond what a single value-axis map already does]] — *idea*
<!-- crux:ledger:end -->
