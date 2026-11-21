# Decisions in force

One row per decision. Superseded rows stay, with Status `superseded by <id>`. Cap: 10 KB.
Provenance: PI, agent proposal approved, agent (code-level or supplementary only).

| ID | Decision | Source (verbatim or file:line) | Provenance | Supersedes | Status |
|---|---|---|---|---|---|
| D1 | Win rule: Pareto dominance. A recipe wins when each of imputation, correction, calibration is at least the best base model's value (over conv AE, conv+attn, U-Net, same version) minus that base's seed range, and at least one term is beyond. | alacarte 2026-11-21: "Pareto dominance (Recommended)" | PI | composite + D13 as the selection rule for this run | in force |
| D2 | Search all three models (conv AE, conv+attn, U-Net). Skip-specific arms apply to the U-Net only. | alacarte: "All three models" | PI | | in force |
| D3 | Correction scored on 16 pair regions to screen, 256 regions to confirm finalists. No screen number is quoted as settled. | alacarte: "16 screen, 256 confirm" | PI | | in force |
| D4 | Fix-family arms: (a) FiLM(C,C')-conditioned count skips (U-Net); (b) pre/post CAC layers; (c) FiLM placement: encoder only, decoder only, bottleneck only, every block incl. attention. | alacarte: three families picked | PI | | in force |
| D5 | Pre/post CAC layer form: g reads only covariates and outputs parameters theta of a count map f_theta (testbed notation). Pre: g(C) -> f removes the C artifact from input counts. Post: g(C') -> f re-applies the C' artifact to the predicted NB. | PI: "g predicted parameters of a function f from covariates only ... remove covariate related artifact before the input and then add them back as we want after the output" | PI | agent's earlier wrong reading | in force |
| D6 | f capacity arms: monotone curve (default), conditioned CNN. | alacarte: "try both"; default "Monotone curve" | PI | | in force |
| D7 | Core-FiLM arms in the pre/post arm: on (default, candidate), off (diagnostic). | alacarte | PI | | in force |
| D8 | Two-step through the base view: retest on screen winners only. | alacarte: "Retest on winners" | PI | settled-design row 24 (off) is reopened for winners | in force |
| D9 | Control: build a leave-control-out evaluation first, then arms: separate control head, control loss weight. | alacarte: "Eval first, then heads" | PI | | in force |
| D10 | EIC comparison: validation vs retrained Avocado and the average during selection; one blind pass vs the 23 entrants at the end; signal p-value from the predicted NB, EIC scorer. | alacarte: "Validation + one blind" | PI | | in force |
| D11 | Data version eic_v1_2. | alacarte | PI | closes t364 for this run | in force |
| D12 | Mix-and-match switches fold into the screen as arms; minimum covariate set, QC input, cell embedding, resolution run after the recipe is chosen. | alacarte: "Fold mix-in, rest after" | PI | separate mix-and-match round (PI decision 8, 2026-11-17) | in force |
| D13 | Compute cap 6,000 allocated CPU-h for the run. | alacarte | PI | | superseded by D20 |
| D14 | Horizon 10 days from 2026-11-21; first measured screen result by 2026-11-24. | alacarte: "10 days, first in 3" | PI | | superseded by D31 |
| D15 | Headline: the three Pareto terms decide; Max's three (R2 on test tracks, with-vs-without covariates gain, real-vs-wrong covariates gain) are reported beside them. | alacarte: "Pareto terms + Max panel" | PI | | in force |
| D16 | Cluster: Fir first, Nibi failover; heavy compute never on the laptop. | global CLAUDE.md; memory heavy-compute-on-nibi | PI | | in force |
| D17 | Review and merge of PR 63 into main are the PI's. | Max notes:17-18; project CLAUDE.md | PI | | in force |
| D18 | Scope: only P0 and P1 rows are this run's goals; P2 and P3 are out. Priority 1: find the best miniCANDI recipe (every ablation needed to reach it is warranted). Priority 2: good ChIP-control prediction. | PI 2026-11-21: "none of p2 and p3 are within the scopre of this session ... the number one priorities is 1. fiding the best recipe ... priority 2 is the good control pred" | PI | | in force |
| D19 | Deeper-look analyses in scope (R12): the U-Net correction cause, the held-out-loss vs composite gap, the views that hurt the combined no-dedup + MAPQ-0 pair. | PI alacarte 2026-11-21, three picked | PI | | in force |
| D20 | 6,000 CPU-h is a checkpoint, not a ceiling: needed arms continue past it; report spend and what more buys at 6,000 and at each milestone. | PI 2026-11-21: "6,000 as checkpoint (Recommended)" | PI | D13 | in force |
| D21 | "Beyond" in D1 = difference of means larger than the larger of the two seed ranges (close-out convention). | planner default; settled design section 0 | agent | | in force |
| D22 | Held-out control cells for the control arms: every third of the 24 validation cells whose control is a training row (8 cells: C02 C10 C16 C20 C25 C31 C36 C46, cell-level rule). | planner default | agent | | in force |
| D23 | Max's "R2 on test tracks" = 1 - SSE/SST in log1p counts per validation target at 25 bp, mean over targets; validation stands in for test until a blind pass exists. | planner default | agent | | in force |
| D24 | 256-region prediction stays on CPU as in the references (no GPU slices). | planner default | agent | | in force |
| D25 | Run outputs live in cruxvault/INSTRUCTIONS/t370_recipe/ (analysis/, results/, run_state/, recipe.md), never in the protected miniCANDI_closeout/ folder. | longrun guard rule | agent | | in force |
| D26 | Post-recipe items of D12 (minimum covariate set, QC input, cell embedding, resolution) are out of this run (D18 scope). Blind pass not run in this run unless the PI starts the blind-row build (t184). | D18; cold plan review | agent | narrows D12 | in force |
| D27 | Date stop: on 2026-11-29 the recipe freezes on what is confirmed at 256 regions by then. | cold plan review | agent | | superseded by D31 |
| D28 | Pre/post CAC layers use smaller widths of the testbed classes (embedding d_cat 2 / d_cov 16 / hidden 32; G hidden 32; FormD 16 channels) so every arm stays within 1.10 x the 1.0e6 parameter budget; one curve side 15,784 params, one CNN side 23,234. | C2 builder; keeps arms parameter-matched | agent | | in force |
| D29 | Add arm "read-disjoint training pairs": training correction and same-view pairs read the source from the pr1 half and the target from the pr2 half (half depth in C and C', as in scoring), so source and target share no reads. All 3 models, 3 seeds, screened with the other arms; also combined with each winner. | Analyses C7, C8, C9 (2026-11-21): U-Net copies the input because training pairs share reads (copy gain 0.63-0.67 vs 0.03-0.09), while scoring uses disjoint halves; all 2,328 training rows have pr1/pr2 files (Fir). | agent proposal, under PI D18 ("every ablation needed is warranted") | | in force |
| D30 | Gate G0 passed on terms and scoring cost; training cost was 2.4 x fact 9, so the screen budget is the projection x 1.50 (5,209 CPU-h). The screen runs on kit RCP2 (RCP1 + C15-C17 + C29) and was submitted before the rerun test verdict (4888524). | probe.md, 2026-11-21 | agent, under PI "keep working and try not to be blocked by me" | | in force |
| D31 | Deadline: deliver the recipe within 24 h of 2026-11-21 18:30 UTC, i.e. by 2026-11-22 18:30 UTC. Supersedes the horizon of D14 and the freeze of D27. | PI chat 2026-11-21: "my deadline is not 2026-11-24. it's within 24 hours" | PI | supersedes D14, D27 | in force |
| D32 | Fix the curve pre/post layer NaN on the U-Net and rerun its 6 U-Net run-seeds (~235 CPU-h). | PI chat 2026-11-21: "i) yes" | PI | | in force |
| D33 | Run the training-only Avocado retrain (~86 GPU-h) for the EIC comparison. | PI chat 2026-11-21: "ii) yes" | PI | | in force |
| D34 | No blind-row build and no blind pass this run. | PI chat 2026-11-21: "iii) no" | PI | confirms D26 | in force |
