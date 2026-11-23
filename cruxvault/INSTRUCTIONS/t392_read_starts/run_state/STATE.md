# State (rewritten, never appended; cap 5 KB)

Goal now: Phase 1 of t392. Read-start miniCANDI, 16-region side by side vs read-count reference.
Horizon: 3-4 days from 2026-11-23. First measured result: probe gate on 4 measurements (chunk C7), day 1.
Worktree: .claude/worktrees/readstart, branch implementation/t392-read-start-minicandi, PR 66 (draft).
Plan: run_state/plan.md (15 chunks C1-C15, 8 waves). Planner found recount ~47 CPU-h (not 180); total 409-684.

In flight: plan review (cold reviewer). No jobs submitted. CPU-h so far: 0.

Next 3:
1. Act on plan review; launch wave 1 (C1 tests via crux-tests agent, C2 Nibi probe staging).
2. Wave 2 (C3 recount, C4 conversion, C5 plan/check tools).
3. Wave 3 (C6 dataset, C7 probe gate on Nibi).

Open asks to foreman (sent 2026-11-23):
(i) Gate wording: ANSWERED, D9.
(ii) Store location: Fir (D10); waiting on PI to start the transfer.
Decided unless objected: ATAC one channel (store keeps both strands).
Blocked: C11 (full build) on (i)+(ii); C14 deviance on PR 65 merge.
