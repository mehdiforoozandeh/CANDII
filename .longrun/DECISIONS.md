# Decisions in force

One row per decision. Superseded rows stay, with Status `superseded by <id>`. Cap: 10 KB.
Provenance: PI, agent proposal approved, agent (code-level or supplementary only).

| ID | Decision | Source (verbatim or file:line) | Provenance | Supersedes | Status |
|---|---|---|---|---|---|
| D1 | Covariate correction of what broke the EIC (dedup, SE/PE) is a main-text claim; dedup and other covariate results support it. Iterations 1-4 all run. | "this bigger claim definitely goes into the main text and the Ddupp or any other covariate-related stuff are somehow supporting this big claim" (chat 2026-11-27) | PI | | in force |
| D2 | No decision rule for read-start vs read-count is set in advance. Gather headlines, mechanism results and Max's slam-dunk test, then decide. | "we must first gather all of the information ... and only then decide" (chat 2026-11-27) | PI | | in force |
| D3 | Fair comparison = the factorial (input {start,count} x target {start,count}), 160,000 windows, 3 seeds, 256 regions. Under-training is a separate later question on the factorial winner. | "we should go with the factorial version, but undertraining is also an open question that we must address separately" (chat 2026-11-27) | PI | | in force |
| D4 | The correction headline gets dedup and SE pairs: same metric, wider pair set, breakdown per covariate. | "Add the pairs (Recommended)" (chat 2026-11-27) | PI | | in force |
| D5 | Max's unpushed sprint 19 notes are assumed not to exist. Use Max's repo sprints and his files on Fir/Nibi only. | "we assume they dont exist. just look at max's repo (different sprints) and their relevant files and data on fir/nibi" (chat 2026-11-27) | PI | | in force |
| D6 | Start iteration 1 under /longrun. | "i) yes. i want you to start the /longrun" (chat 2026-11-27) | PI | | in force |
| D7 | Branch exp/start-vs-count-dedup is cut from origin/exp/t370-minicandi-recipe (36a306d3), not origin/main: the read-start code, the final recipe and the q81 tree exist only there (PR 63 unmerged; 0 commits behind main). PR targets exp/t370 until PR 63 merges, then main. | CLAUDE.md "a branch starts at fresh origin/main"; global "cut from the branch you are working on" | agent | | in force |
| D8 | Leg B (read-count model in start space) uses Richardson-Lucy deconvolution with the per-measurement span kernel, scored strand-summed; not a finite difference. Reason: c(i)-c(i-1)=s(i)-s(i-w) is not invertible without noise blow-up and negatives. | BACKGROUND_AND_PLAN.md s1.5 | agent | | in force |
| D9 | Every model is scored in three spaces: overlap-count, strand-summed start, and 1 kb sums; each normalized to its own pseudoreplicate floor. | BACKGROUND_AND_PLAN.md s3 it.1 | agent | | in force |
| D10 | Job prefix sc_; heavy compute on Fir first (Nibi failover), per prior runs. | .longrun/config | agent | | in force |
| D11 | Tiers in GOALS.md set by the agent (ledger reader's recommendation); the PI may correct any row. | supervisor rule, ~/.claude/CLAUDE.md | agent | | in force |
| D12 | 256-region scoring for the factorial is in force: the PI chose the option that named it (D3). This supersedes the earlier decline, which covered only the t392 16-region run. | D3 option text "160,000 windows, 3 seeds, 256 regions" | PI (via D3) | t392 decline | in force |
