# Requirements ledger

One row per requirement in the PI's documents. Tier: P0 the run fails without it, P1 expected, P2 if time allows, P3 do not build.
Status: open, partial, done, dropped (dropped needs a PI decision id in Evidence). Done needs a measured result linked in Evidence.
Provenance: PI doc, PI, agent proposal approved.

| ID | Tier | Requirement (quoted) | Source | Provenance | Status | Evidence |
|---|---|---|---|---|---|---|
| R1 | P0 | "define and finalize the best recipe for minicandi one that's likely the best based on our objectives and data and problem formulation" | mini candi.md:41 | PI doc | done | D55: PI picked the final architecture 2026-11-23; f143 on q75; tasks/t370 Evidence |
| R2 | P0 | "the optimal architecture that can deliver all three: 1. best imputation 2. best correction 3. best uncertainty calibration" | mini candi.md:14-20 | PI doc | done | D54/D55: imp 0.503, cor 0.818, cal 0.717 vs bars 0.445, 0.675, 0.646 (pareto_p256_sv.tsv, 3 seeds) |
| R3 | P0 | "make a variant of the unet that gives us the best of both worlds ... breaking between the pareto of imputation quality and correction"; "Play with U-net architectures" | mini candi.md:8; Max notes 2026-11-19.md:2 | PI doc | done | D53/D54: correction 0.818 vs U-Net base 0.535 with imputation 0.503 vs 0.445 (pareto_p256_sv.tsv) |
| R4 | P0 | "improve all three together - not compromising the correction capability" | mini candi.md:25 | PI doc | done | D54: all three terms above the bars at 256 regions; correction not compromised |
| R5 | P1 | "having CAC (same f, g formulation) before the input to CANDI ... and after the output (post processing of predicted NB) -- that's just a suggested ablation"; "Perhaps by putting CAC upstream of U-net?" | mini candi.md:27; Max notes 2026-11-19.md:2 | PI doc | open | |
| R6 | P1 | "play with where we are injecting? count conv and decov towers? attention layers? all?" | mini candi.md:30 | PI doc | open | |
| R7 | P1 | "two step CAC through base was shown to be helpful. does that improve our performance?" | mini candi.md:32 | PI doc | open | |
| R8 | P0 | "have chip control as both input and output assay and also train candi to be able to predict chip controls ... i definitely want to see if we can get this variant working" | mini candi.md:35 | PI doc | done | D51: control CRPS skill 0.131 on held-out cells, weight 1 in the final (results/control/control_summary.tsv) |
| R9 | P1 | "compare minicandi against other imputation methods on our imputation metrics (+some of EIC imputation metrics)" | mini candi.md:37 | PI doc | open | |
| R10 | P1 | "process the predicted count data from miniCANDI to signal pvalues (which are comparable with the EIC)" | mini candi.md:37 | PI doc | open | |
| R11 | P1 | "make miniCANDI robust to this covariate shift and also be a good imputation method (+ caliberated uncertainty)" | mini candi.md:39 | PI doc | open | |
| R12 | P1 | "there are other things about the minicandi results that need some deeper investigation" | mini candi.md:6 | PI doc | open | PI 2026-11-21 named three analyses: U-Net correction cause, held-out-loss vs composite gap, views hurting the no-dedup + MAPQ-0 pair |
| R13 | P1 | "Headline metric: R2 on test tracks" | Max notes 2026-11-19.md:4 | PI doc | open | |
| R14 | P1 | "Difference in performance including covariates vs not" | Max notes 2026-11-19.md:5 | PI doc | open | |
| R15 | P1 | "Difference in performance giving as input the real covariates vs wrong covariates" | Max notes 2026-11-19.md:6 | PI doc | open | |
| R16 | P3 | "Assemble big data set including IHEC and any others. We will consult regarding train/dev/test split." | Max notes 2026-11-19.md:7 | PI doc | open | |
| R17 | P3 | "Look for signal units which are less variable across replicates?" | Max notes 2026-11-19.md:11 | PI doc | open | |
| R18 | P1 | "figure out the difference between the width of the layers of Sprint 10's onion and Mehdi onion" | Max notes 2026-11-19.md:13 | PI doc | done | cruxvault/INSTRUCTIONS/t370_recipe/analysis/layer_widths_sprint10_vs_ours.md: sprint-10 reference 4,622,554 params (4.4-4.7x ours), shared width 384 vs 60-120 / 104 / 25-100 |
| R19 | P0 | carry-over: "the base-model scores on eic_v1_2 at 256 regions on Fir (task t357)" -- the reference bar of the win rule; then "run STEP=ref of submit_a12.sh with kit CO10 (81 jobs)" | Max notes 2026-11-19.md:24-27 | agent handoff | done | Fir 2026-11-21: 9 of 9 runs_fir/eic_v1_2_rung{2,3,4}_base_heldout/seed*/p256_sv/SCORED exist |
| R20 | P0 | "Never resubmit a job that the Log of .orchestrate/plan.md lists"; "never resubmit a job that is still queued" | Max notes 2026-11-19.md:25-30 | agent handoff | open | |
| R21 | P1 | "Read first, in this order: ...HANDOFF_NEXT_SESSION.md ..., then closeout/settled miniCANDI design.md, then closeout/ready queue for scaling.md" | Max notes 2026-11-19.md:18-20 | agent handoff | partial | settled design read 2026-11-21 |
| R22 | P1 | "The main line of the next sprint is scaling miniCANDI (task t369). It waits on ... the architecture for scaling (t363)" -- this run's recipe is the input to t363; scaling itself is out of scope | Max notes 2026-11-19.md:27-29 | agent handoff | open | |
