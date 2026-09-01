---
id: t84
type: task
title: vendor Lavawizard — the other three rivals were already implemented
category: implementation
parent: 
blocked_by: None
refs: 
hypothesis_refs: 
status: done
created: "2026-08-31T00:52:17"
updated: "2026-08-31T14:55:26"
---

# t84 — vendor Lavawizard — the other three rivals were already implemented

Refs:: _(none)_

## Why

only Avocado is vendored; the other three pval-arm rivals of 7 cannot be retrained under the uniform V_ selection rule until their code is in the tree byte-identical with sha256s in PROVENANCE.md

**SCOPE WAS WRONG WHEN THIS WAS WRITTEN — 2026-08-31.** "Only Avocado is vendored" is true of
`implementation/t77-benchmark-design` and false of the repo. `implementation/t60-leaderboard-site-v2`
(main + 158, 0 behind, unmerged) already carries real implementations with SLURM scripts and tests:

| tree | files | python LOC | holds |
|---|---|---|---|
| `competitors/baselines/` | 5 | 1,151 | `generate.py`, `heads.py`, `leaderboard.py` — the five naive baselines |
| `competitors/chromimpute/` | 11 | 961 | `prepare`, `collect`, `fit_sigma`, `project_gtd` + 5 slurm scripts + a test |
| `competitors/edice/` | 14 | 1,710 | a full PyTorch eDICE (`edice_torch/{data,metrics,model,train}.py`), `eic_panel`, `run_eic`, `fit_sigma` + 3 slurm scripts + a test |
| `competitors/entrants/` | — | — | the 23 challenge submissions |
| `competitors/avocado/` | — | — | a second Avocado, beside this branch's `rivals/avocado/` |

So **only Lavawizard is genuinely absent.** For the other three this is not a vendoring task at all —
it is the branch reconciliation, which the PI approved on 2026-08-31 as: verify t60's gates (done —
1015 passed, `golden.py` 0 ULP), merge t60 to main, then rebase this branch onto it.

Retitling this to Lavawizard alone is a scope decision and is left to the PI.

## Output

<!-- required before `done`, and the engine checks it resolves. Either form:
     - [Deduped table](results/dedupe/table.tsv)   - [[wiki/candi-datasets]] -->
- [Lavawizard merged; the other three were already implemented](results/t84/DELIVERABLE.md)

## Evidence

_(experiments only: what this run showed, in prose. The structured fact is
`hypothesis_refs` in the frontmatter; this is the narrative beside it, and the
engine never parses it.)_
