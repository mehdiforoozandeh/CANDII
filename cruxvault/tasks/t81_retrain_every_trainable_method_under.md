---
id: t81
type: task
title: retrain every trainable method under the uniform V_ selection rule
category: implementation
parent: 
blocked_by: t78, t79, t80
refs: 
hypothesis_refs: 
status: open
created: 2026-08-29T18:42:39
updated: 2026-08-29T18:42:39
---

# t81 — retrain every trainable method under the uniform V_ selection rule

Refs:: _(none)_

## Why

each method selects its checkpoint by its own convention today, so the V_ column is optimistic by different amounts for different methods

**Re-laned 2026-08-30, and it is no longer a child of t77.** t77's branch is
`implementation/t77-benchmark-design`, and CLAUDE.md's gate for that lane is that the PR must not
move a number — `pytest` green plus `tools/golden.py` bit-exact against the pre-change tree. This
task retrains every method, so it moves every number by construction. A retrain cannot land on an
implementation branch without either breaking that gate or forcing t77 to be relabelled, and t77 is
genuinely infrastructure: t78 (the DNase layer), t79 (the regimes), t80 (the eval stack) and t82
(the board) all hold golden at 0 ULP.

So t77 ships without this: it is de-parented and blocked on t78, t79 and t80 instead.

**RULED 2026-08-30 (PI): this is not a question or a hypothesis. It belongs in the taskhub only.**
It carries no `hypothesis_refs` and it is not going to acquire any. The lane question that was
raised here is therefore closed, and closed the simple way: CLAUDE.md gives the `exp` lane to "a
task the engine calls an *experiment* (any task carrying hypothesis refs)", and **this task is not
one**. It stays `implementation`, and it is not an experiment merely because it is expensive and
moves numbers. Nothing about the crux tree gates it — no null, no verifiables, no `crux close`.

**What that leaves is a real gate problem, and it is now the only one.** CLAUDE.md's
`implementation` gate is that the PR must not move a number — `pytest` green plus `tools/golden.py`
bit-exact — and a retrain moves every number by construction. Three ways out, and they are not
equally good:

1. **Flag-only.** CLAUDE.md already says work that changes no code runs from `main` and opens no
   PR, so no gate applies. Retraining under existing configs is exactly that. This is the honest
   reading for the CANDI and naive-baseline arms, which need no new code at all.
2. **Split the code from the run.** The launch scripts and any rival vendoring are ordinary
   `implementation` work that holds golden at 0 ULP and merges normally; the *run* that follows is
   flag-only. `t79`'s Avocado vendoring already worked this way — 865 passed, golden 0 ULP.
3. Relabel the task. Rejected: it is not an experiment, per the ruling above.

**What still has no answer is how a retrain's numbers get accepted**, which is a different question
from how a PR merges. The target-clustered noise floor on macro CRPS is ~0.09 and a seed change
alone moves pooled CRPS by 0.1195 — larger than several recorded between-arm gaps — and the floor
on the *new* panels is deferred and unmeasured (`plan/BENCHMARK_DESIGN.md` §15). Ranks are the
part that reproduces; scores are not (§5.3: 16 of 25 published ranks held exactly, resolution limit
~0.005 correlation units). So a board row from this task can go up **unranked** before the floor
lands, and §15 already allows that.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
