# Decisions in force

One row per decision. Superseded rows stay, with Status `superseded by <id>`. Cap: 10 KB.
Provenance: PI, agent proposal approved, agent (code-level or supplementary only).

| ID | Decision | Source (verbatim or file:line) | Provenance | Supersedes | Status |
|---|---|---|---|---|---|
| D1 | Score in signal p-value space with the challenge's own metrics; headline truth = the organizers' corrected (SE reprocessed + quantile normalized) blind tracks, Zenodo 14736610 (the MOESM4 board); MOESM2 (original truth) reported beside it. | onion long-horizon decision D9; goal.md:29 | PI | | superseded by D7 (headline only) |
| D2 | Fair = the count-to-p-value route may use only what an entrant could use: training and validation tracks, public metadata, the challenge's processing recipe. No blind truth (raw or normalized) enters any fitted step. | PI 2026-11-22: "yes to both" | PI | | in force |
| D3 | Tested = the conversion is applied to observed counts of experiments with known p-value tracks, and its error is reported as a conversion floor beside miniCANDI's board scores. | goal.md:10 ("figured out exactly") | PI doc | | in force |
| D4 | No downloads and no large compute while the PI is away; small probes on data already on Fir/Nibi only. | global rule: large compute is the PI's | agent | | in force |
| D5 | The count route's peak rule is fixed before scoring: the ENCODE pipeline's naive-overlap rule on two random read halves, ported from its source. Blind-truth scores report the rule; they never select among rules. | auditor finding 2026-11-23 on D2 (selection on blind truth would break it) | agent, applies D2 | none | in force |
| D6 | Start the blind-row build (t184) so miniCANDI can predict the blind experiments. It belongs to the recipe run; this run hands the go-ahead to that session. | PI 2026-11-23: "yes to i and ii" (i = start the blind-row build) | PI | none | in force |
| D7 | Headline board for miniCANDI = MOESM2 (original truth, no QN, no reprocessing): it tests correction. MOESM4 is reported beside it. | PI 2026-11-23: "yes to i and ii" (ii = MOESM2 headline, MOESM4 beside) | PI | D1 headline truth | in force |
| D8 | Hold the 48-experiment conversion batch and the message to Jacob Schreiber. | PI 2026-11-23: "hold off on iii and iv" | PI | none | in force |
| D9 | Copy the 51 original-truth bigWigs to /project; a read-start-count miniCANDI arm is built in a new long-run session | "yes" (PI via foreman, 2026-11-23) | PI | none | in force |
| D10 | Validation board: raw predictions scored on the mean for all 42 experiments; the expected-p raw-vs-recal_wide pair only on the 8 first tracks (~30 CPU-h). Unscored experiments count as worst. | "yes" to option (a) (PI via foreman, 2026-11-24) | PI | none | in force |
| D11 | Fixes for the validation board: fit only on non-validation data; slope boards: in-sample labelled a preview, leave-one-experiment-out is the fair one (factor 1 where no other experiment); training-split forward pass not requested; depth-to-covariate rescale skipped. Budget 36 CPU-h. | foreman relaying PI, 2026-11-25 ("Go" on fix b; slope boards; 36 limit) | PI via foreman | none | in force |
| D12 | Blind-entry fix = 20 M read-depth cap on ChIP tracks (mu and depth_reads times 20 M / depth_reads when depth_reads > 20 M; DNase unchanged), fitted on 196 training ChIP tracks only. The carried board is depth-capped, raw beside it. Cap plus slope board skipped (does not fit the budget). | foreman 2026-11-25: "Make the 20 M depth cap the board you carry forward" | PI via foreman | D10 headline board | in force |
