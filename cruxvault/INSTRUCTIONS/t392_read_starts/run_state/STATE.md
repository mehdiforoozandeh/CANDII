# State (rewritten, never appended; cap 5 KB)

Goal now: Phase 1 of t392. Read-start miniCANDI, 16-region side by side vs read-count reference.
Horizon: 3-4 days from 2026-11-23 (est. table ~2026-11-26). First measured result: probe gate (C7b), day 1.
Worktree: .claude/worktrees/readstart, branch implementation/t392-read-start-minicandi, PR 66 (draft).
Plan: run_state/plan.md (24 chunks, waves 1-14; amended after review). Chunk worktrees: <repo parent>/.orchestrate-wt/CANDII/t392-<id>, branches chunk/t392-<id>.

In flight:
- Wave 1: C1 merged (64 tests). C2 retry 1 (Nibi login-node extract). Wave 2: C3, C4, C5 builders + C1 test review running.
- Transfer Nibi->Fir of 6 tars (1,185,827,906,560 B) to Fir /scratch/mforooz/rs392/tars/: running; watched by scratchpad watch_transfer.sh (local background, 10-min checks). Testbed transfer LANDED 20:48Z (18,813,836,293 B, 156 files), reported.
CPU-h so far: 0.

Next 3:
1. Review C1, C2 slips; merge.
2. Wave 2: C3 recount/gate/store writer, C4 conversion, C5 build plan + check tools.
3. Wave 3: C6 dataset, C7a probe recount on Nibi.

Open asks to foreman: none. All decisions D1-D14 answered.
Blocked: C11a (Fir build) on tars landing + C7b; C14 deviance on PR 65 merge (then merge origin/exp/t370 into umbrella, D8).
