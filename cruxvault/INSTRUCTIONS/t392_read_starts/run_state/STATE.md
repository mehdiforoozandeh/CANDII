# State (rewritten, never appended; cap 5 KB)

Goal now: Phase 1 of t392. Read-start miniCANDI, 16-region side by side vs read-count reference.
Horizon: table est. ~2026-11-26. First measured result: probe gate (C7b).
Worktree: .claude/worktrees/readstart, branch implementation/t392-read-start-minicandi, PR 66 (draft).
Plan: run_state/plan.md (24 chunks; Log at end is the run memory). Chunk worktrees: <repo parent>/.orchestrate-wt/CANDII/t392-<id>.

Merged: C1 (tests + fix round), C2 (probe files staged on Nibi), C3 (recount/gate/store), C4 (conversion), C5 (plan/check tools), C6 (read-start dataset).
In flight:
- C7a: kit RS1 pushed to Nibi, array rs392_probe (4 tasks) submitted by agent.
- Wave 4: C8 model, C9 adapter, C16 docs (agents).
- Tar transfer Nibi->Fir to /scratch/mforooz/rs392/tars/ (1,185,827,906,560 B), local watch scratchpad watch_transfer.sh (bg task, 10-min checks). Testbed transfer landed.
CPU-h so far: ~0 (job 28710841 3 s).

Next 3:
1. C7b: read probe gate (0 differing bins?), rate, memory; status table to foreman.
2. Merge wave 4; C10a end-to-end + full suite on Fir.
3. C11a Fir build once tars land (one array per tar + testbed, --testbed-root).

Open asks to foreman: read starts in and out (rec yes), sent 2026-11-23; proceeding with rec.
Blocked: C11a on tars; C14 deviance on PR 65 merge (then merge origin/exp/t370 into umbrella, D8).
