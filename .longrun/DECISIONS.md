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
| D13 | Compute cap 6,000 allocated CPU-h for the run. | alacarte | PI | | in force |
| D14 | Horizon 10 days from 2026-11-21; first measured screen result by 2026-11-24. | alacarte: "10 days, first in 3" | PI | | in force |
| D15 | Headline: the three Pareto terms decide; Max's three (R2 on test tracks, with-vs-without covariates gain, real-vs-wrong covariates gain) are reported beside them. | alacarte: "Pareto terms + Max panel" | PI | | in force |
| D16 | Cluster: Fir first, Nibi failover; heavy compute never on the laptop. | global CLAUDE.md; memory heavy-compute-on-nibi | PI | | in force |
