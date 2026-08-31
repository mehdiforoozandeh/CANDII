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

**Its lane is genuinely unresolved, and that needs the PI.** CLAUDE.md says the lane is `exp` for
"a task the engine calls an *experiment* (any task carrying hypothesis refs)". This task carries
**none**, and `experiment` is a reserved category the engine computes and refuses to accept by hand
(`engine.py:3264`). So it cannot be put in the `exp/` lane by editing a field — it would need
hypothesis refs, and deciding which hypotheses a retrain serves is a research judgment, not
bookkeeping. Left as `implementation` with this note rather than guessing.

**It cannot start yet, and not because of compute.** CLAUDE.md records the experiment-lane merge
gate as *"TODO, NOT YET DEFINED"*, and says plainly: do not invent a gate to unblock yourself —
raise it. It is raised. The gate is the PI's to define, and the note there is worth keeping in view:
the target-clustered noise floor on macro CRPS is ~0.09 and a seed change alone moves pooled CRPS
by 0.1195, which is larger than several recorded between-arm gaps. The noise floor on the *new*
panels is itself deferred and unmeasured (`plan/BENCHMARK_DESIGN.md` §15), so nothing this task
produces can carry a resolution band until that lands either.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
_(none yet)_

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
