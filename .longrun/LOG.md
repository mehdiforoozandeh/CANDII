# Log

Append-only. Written only by scripts/log.sh. Older entries rotate to log/.
- 2026-11-27T23:23Z longrun initialized (protected=cruxvault/INSTRUCTIONS/start vs count + dedup, job_prefix=sc_)
- 2026-11-27T23:23Z kickoff: worktree startcount, branch exp/start-vs-count-dedup off origin/exp/t370 36a306d3; D1-D10 written; ledger reader running
- 2026-11-27T23:24Z ledger: 52 rows written (R1-R52); D11-D12; q94 q95 recorded
- 2026-11-27T23:27Z D13: no deadline, no cap (PI). STATE written. Next: orchestrate iteration 1.
- 2026-11-27T23:28Z D14 commit policy; iteration-1 planner spawned (fable)
- 2026-11-27T23:29Z glossary: blocked reply with d1, d12, d13, d14, d6, d7
- 2026-11-27T23:29Z glossary: blocked reply with prefix
- 2026-11-27T23:44Z iteration-1 plan: 16 chunks in 4 waves; cold plan review running
- 2026-11-27T23:53Z plan review done: 6 factual errors + unsound dedup floor; planner amending once
- 2026-11-27T23:59Z wave 1 spawned: 7 opus builders from dadf6b59
- 2026-11-28T00:07Z wave 1: C1 C2 C3 C4 C12a merged; Fir job 6726827 sc_untar pending
- 2026-11-28T00:10Z wave 2 spawned: C7a C8 C9 C10a C12b C13
- 2026-11-28T00:15Z glossary: blocked reply with stub
- 2026-11-28T00:23Z C8 C9 C10a C13 merged; D15 shared-weight dedup law primary; C12b retry 1 (OOM)
- 2026-11-28T00:24Z early 1 bp result: ENCSR956CTX br1 dedup-on equals distinct dedup-off positions exactly (77,217,569); probe 2 Fir 6729250
- 2026-11-28T00:25Z glossary: blocked reply with deduplicated
- 2026-11-28T00:35Z C7a merged; dump arrays 6731315-6731318 submitted (16.7 CPU-h projected); D16
- 2026-11-28T00:49Z 1 bp: dedup is per replicate (ENCSR956CTX both reps exact; pooled 2.6% repeats); 5 more jobs 6733480-84
- 2026-11-28T00:55Z wave 1 complete (C5 merged); f158 on h61 against; census: SE pairs 20 from 2 experiments
- 2026-11-28T01:11Z C10b merged; cap arrays 6737535 6737536
- 2026-11-28T02:30Z glossary: blocked reply with c10c, handback
- 2026-11-28T03:34Z C10c merged: start cap beats count law (1.35, CI 0.29-3.41) but neither explains dedup on 43 val pairs (cap gap -0.56); h58 mixed
