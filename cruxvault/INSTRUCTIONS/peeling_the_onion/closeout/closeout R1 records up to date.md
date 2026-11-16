---
title: "R1: bring the records up to date (rulings, compute tallies, paper note)"
parent: "[[closeout todo]]"
created: 2026-11-16
---

# R1: bring the records up to date (rulings, compute tallies, paper note)

## 1. What the PI asked

- `long-horizon prompt.md`:32: "if it's something deep in the code, or something that belongs in the supplementary material, you can decide yourself and then write it down in the notepad (cruxvault/INSTRUCTIONS/peeling_the_onion/long-horizon notepad)."
- `long-horizon decisions.md`:24 (D3): "The agent reports CPU-hours, GPU-hours and storage used at each milestone."
- `long-horizon notepad.md`:542 (PI directive 2026-11-14): "deliver the work as a research paper is written, part by part." Methods grow gradually; "Results, discussion and conclusions are added little by little."
- `long-horizon notepad.md`:548 (PI directive 2026-11-14): "I want all of them added to CRUX tree and its taskhub."

Success criteria served: all four, through the record. Audit rows: G3, G4, G5, G6, G8, G11.

## 2. Question

Make the notepad, the compute tally, the paper note, the artifact index and the crux tree agree with what the run did up to the close-out, so that R2 can be built from the records alone.

## 3. State today

- Notepad: ends at ruling 49 (2026-11-15). The PI directives and rulings of plan Log 2026-11-15T20:05Z, 2026-11-16T01:10Z, 01:50Z, 04:10Z, 08:30Z, 08:50Z, 21:10Z and 21:15Z are only in the Log (audit row G3).
- Compute tally: last entries in the notepad 2026-11-11 to 2026-11-13 (L272-273, L314-318, L474: 4,511 CPU-h used for the FASTQ build). Nothing for the ladder, the Fir work or ship 2 (audit row G4).
- Paper note `onion paper.md`: lacks the view ablation, the split-depth result, the tuning rounds, the final round and composite version 3. L54 still defines metric 4 with the half-against-half floor (superseded by ruling 47 and version 3). Discussion items 1 (L177) and 3 (L179) disagree with the scores (`COVERAGE.md`, "Where a doc disagrees"). "Later phases" (L185) is empty; there is no Conclusions section.
- Artifact index `presentations/ARTIFACTS.md`: lists ship 2 and the total-variance deck; lacks ship 1 (artifact 4RSUdsFwJC3nYcyHb9s2bU, per `run_state/WHERE_WE_ARE.md`), the status deck of 2026-11-14 and the report for Max (audit row G6).
- Crux tree: findings to f89 and questions to q47 (commit 449b0e0); `crux validate` not run (audit row G8).

## 4. Prior leads

- Each Log entry cited above holds the PI's words; copy them, do not paraphrase the ruling.
- `COVERAGE.md`, "Where a doc disagrees with the scores", items 1 to 4, gives the corrections for the paper note.
- Memory rule: every published artifact also gets an md under the vault with its URL.

## 5. Work

1. Notepad: one entry per PI ruling or directive from the Log since 2026-11-15T20:05Z, each with the PI's words, the date, and the effect; one entry per D13 outcome logged by the close-out items.
2. Compute and storage tally: `sacct` CPU-h (CPUTimeRAW) and GPU-h per cluster since 2026-11-13, grouped by work (dataset, scoring, ladder, tuning, final round, ablations, close-out items); `diskusage_report` on Nibi and Fir. Write the arithmetic. Repeat at the end of the close-out.
3. Paper note: update Methods (metric 4 with the oracle floor; composite version 3; correction-only term), Results (views, split depth, tuning, final round, ablations, close-out results as they arrive), Discussion (fix items 1 and 3), Later phases (the PI's follow-up list from `closeout todo.md`), and a Conclusions section.
4. `ARTIFACTS.md`: add ship 1, the status deck, the report for Max, and every close-out artifact.
5. Crux: run `crux validate --check=tree,tasks` from `cruxvault/`; fix each error; record every close-out finding as it arrives.
6. Vault close-out (plan chunk VC): each converted results folder has `runall.sh`, `MANIFEST.tsv` and its `.gitignore` line (`AGENTS.md` section 4.11).

## 6. How the result answers the question

Acceptance checks:
- `grep -c "^## " "long-horizon notepad.md"` rises by at least the number of Log rulings listed in section 3; each new entry quotes the Log line.
- The tally entry gives CPU-h, GPU-h and storage per cluster with the `sacct` command used.
- `grep -n "CRPS_floor" "onion paper.md"` finds no undeleted half-floor definition of metric 4; the paper has a "Conclusions" heading.
- `ARTIFACTS.md` lists every artifact URL found in the Log since 2026-11-14.
- `crux validate --check=tree,tasks` exits 0.

## 7. Dependencies, cost, risks

- Depends on nothing to start; it repeats as the other items finish.
- Cost: under 1 agent-day for the backlog, then minutes per result; `sacct` reads are read-only.
- Risk: a paper sentence written from memory. Every number comes from a results file or a finding.

## 8. Done when

The notepad holds every PI ruling of the run, the compute and storage tally covers the run to the close-out, the paper note has updated Methods, Results, Discussion, Later phases and Conclusions, `ARTIFACTS.md` is complete, and `crux validate` exits 0.
