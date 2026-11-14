---
title: crux recording rule
kind: run rule
parent: "[[peeling the onion]]"
created: 2026-11-14
source: PI directive of 2026-11-14 ("everything that you do ... I want all of them added to CRUX tree and its taskhub"); the crux skill (SKILL.md, references/taskhub.md, references/verbs.md)
---

# crux recording rule

==The crux vault is the one data structure of the onion run. The question tree holds what we want to know and what we learned. The taskhub holds the work. Every result of the run enters both, at the time it arrives. A file under `results/` or a line in the plan Log is not a record until a finding or a task output points to it.==

## When to record

==Record at each result: a returned chunk slip, a collected job, a merged chunk, a PI ruling, or a new question from the PI. The foreman does not wait for the end of an iteration. At each milestone (an iteration ships, a wave closes, or the PI asks), one recorder chunk records everything since the last recorder.==

## What to record

1. ==**A question** (`crux ask "<question>" -p <parent> --bears "<how its answer moves the parent's answer>"`) for each thing the PI or the plan wants to know. Read the tree first (`crux status --json`, `crux brief <id> --mode=situate --json`). Never open a second node for something the tree holds.==
2. ==**A hypothesis** (`crux hypothesize "<claim>" -p <qid> --expect "<result, with its size>" --bears "<...>"`) only when an expectation was stated BEFORE the result. If no expectation was stated before the result, record the result on a question. Never invent a prior expectation, and never loosen an expectation after the result.==
3. ==**A finding** (`crux finding <id> ... -e results/<folder>/<file>`) for each result. On a hypothesis it has a direction (supports, against, mixed, no answer yet), a confidence (strong, moderate, weak), one sentence of why (`-w`), and the comparison with the expectation (`--vs-expect`). On a question it has an answer (`-a`) and a confidence. Judge by the closure rules of the crux skill: a near miss lowers the confidence one step; a null that is not ruled out caps a supports finding at weak; a failed control caps any finding at weak; a finding from browsing data that nobody asked about is `--exploratory` and at most moderate. Use numbers exactly as the source file writes them.==
4. ==**Confidence and state.** The strongest finding sets a node's state: strong means settled, i.e. no further time or compute is planned on it. Use moderate or weak when the answer is partial or when more work on the node is planned, so that the node stays leaning.==
5. ==**A task** (`crux task add "<action>" -c <category> --parent <task> --blocked-by <ids or None> --ref <node> --why "<what it unblocks>"`) for each chunk at launch. A chunk that would open a question is a node, not a task.==
6. ==**A task done** (`crux task done <id> -o "[<label>](<vault path>)"`) when its work is finished. The output must resolve to a vault file: a note under `results/<folder>/`, a file under `INSTRUCTIONS/`, or a wikilink to a vault note. Never give a bare hash. A code-only chunk links its merge commit to the run-state snapshot of the plan (`INSTRUCTIONS/peeling_the_onion/run_state/plan.md`). A task that is partly done stays open, with one dated line under `## Decisions and runs` in its task file. `crux task done` appends; it never replaces an output.==
7. ==**Operational facts** (cluster behaviour, pipeline defects, costs) go on task outputs, or on the methods question of the run (the question on the estimators, scorers and training pipeline). They never go into a science hypothesis.==
8. ==**Never record a model-quality result before a model is scored on the evaluation set.** A pilot or smoke score is labelled and stays in its results note.==

## Who records

==One recorder per milestone. The recorder works on its own worktree, cut from the umbrella branch, so that no two agents write the generated files (`META.md`, `EXPERIMENTS.md`, `TASKHUB.md`) at the same time. The recorder uses the crux CLI for every write. It never hand-edits `META.md`, `EXPERIMENTS.md`, `TASKHUB.md`, `WIKI.md` or a ledger block. Builders of other chunks do not run crux write verbs, except the chunk that creates its own task ids (the plan's V chunks). The foreman merges the recorder's branch like any other chunk; when `TASKHUB.md` conflicts, the merge takes the union of `tasks/` and regenerates the generated files with `E.refresh`.==

## The recorder's checklist

1. ==Run `crux validate --check=tree,tasks` and save the output (the baseline).==
2. ==Read `crux status --json`, `crux task list`, and the nodes and expectations of the subtree that the run touches.==
3. ==Read the plan's `## Log` and `## PI rulings` since the last recorder, the notepad, and every new note under the umbrella worktree's `cruxvault/results/`.==
4. ==For each new result: verify it in its source file, then record the finding on the right node. Where the Log and a result note disagree, the result note wins; list the disagreement on the slip.==
5. ==For each new PI question: check the tree, then open the question with `--bears`. Put an item you are unsure of on the pending list (`crux pending add`), for the PI.==
6. ==For each finished chunk: mark its task done with a resolving output. For a chunk with no task id: add the task under its parent, then mark it as its state is.==
7. ==Results notes live under `cruxvault/results/`, which git ignores except converted folders (AGENTS.md section 4.11). If a cited note is missing from the recorder's worktree, copy the small note (under 1 MB) into the same folder so that the path resolves, and list the copy on the slip. Do not `git add` it unless the folder meets the runall contract.==
8. ==Run `crux validate --check=tree,tasks` again. The change must add no error that names a node or task the recorder touched.==
9. ==Commit only the vault paths that the crux CLI changed (`git add cruxvault`), and return the slip.==
