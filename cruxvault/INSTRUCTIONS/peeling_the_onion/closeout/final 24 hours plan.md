---
title: final 24 hours plan (peeling the onion)
kind: plan, written by the foreman agent
parent: "[[closeout todo]]"
written: 2026-11-17T20:15Z
deadline: 2026-11-18T20:00Z
---

# Final 24 hours plan

The PI ruled on 2026-11-17: "you have 24 hours to wrap up the peeling the onion work. plan wisely to cover everything remaining in the next 24 hours." All times are UTC.

## Limits that set the plan

- Fir is in a planned outage until about 2026-11-20. No Fir job runs before the deadline. The 570 queued Fir jobs stay queued.
- Nibi `/scratch` is read-only (1,740 GiB, quota 1,024 GiB) until the pipeline work folders (`onion/work`, 1,105 GiB) leave. They leave after the QC fill and the 1 bp recount end.
- A training round of several thousand CPU-hours cannot train, score and be read in 24 hours with a margin. No new training round starts in this window.

## What "wrapped up" means against the four success criteria

| Criterion | State at the deadline | Not in this window |
|---|---|---|
| 1. Augmented EIC data with counterfactuals | Tables `eic_v1_2`, the second-experiment add-on, the candidate with DNase-seq, QC side tables, the 1 bp region store. Archive note. | Blind build (22,700 CPU-hours), 4 row rebuilds |
| 2. Training objective | Document merged. Open choices marked. | none |
| 3. Evaluation, defined and tested | All scorers merged and tested. The 16-versus-256 rule recorded. 256-region rescoring of the 7 decisive arms on Nibi, if scratch frees in time. | Rescoring that does not end in time |
| 4. Settled miniCANDI design, tested, ready for scaling | Settled-design document: each choice, its evidence, its confidence, and the choices that stay provisional with the run that settles each. | Mix-and-match round, covariate set, cell embedding, resolution arms, QC-on-input arm, RNA-seq input, pointwise MLP |

Every item in the right column goes into one "ready queue" document: the command, the cost, the prerequisite, the decision rule. The next session starts from that document.

## Timeline

| Hours | Cluster work (Nibi) | Agent work (no cluster) |
|---|---|---|
| 0 to 6 | QC fill (arrays 27658288, 27658295). Recount probe 27658461, then the recount. | Settled-design document, first version, from the 16-region evidence. Ready-queue document. |
| 6 to 8 | QC collect job. Check both passes. Remove `onion/work` from scratch (archived, PI yes). Scratch writable. | Independent review of the settled design. Records: rulings, artifacts list, tree validation. |
| 8 to 20 | 256-region rescoring of the 62 run-seeds of the 7 decisive arms (about 2,225 CPU-hours, approved). 1 bp tables from the recount store. | Close-out deck and close-out report, built on the settled design. |
| 20 to 23 | none | Read the 256-region scores against the pre-written rule. Update the settled design, the deck and the report. Number check. |
| 23 to 24 | none | Tests, final snapshot, status page, handoff. PR ready for the PI. |

## Stop rules

- Hour 8: if scratch is not writable, the rescoring does not start. The settled design then stands on the 16-region evidence, marked provisional, and the rescoring goes to the ready queue.
- Hour 20: every result not on disk is reported as "not finished". Jobs keep running; nothing is cancelled.
- No deletion beyond `onion/work`. No merge to main. No new large submission beyond the rescoring.

## Deliverables at the deadline

1. `closeout/settled miniCANDI design.md` (item M1).
2. `closeout/ready queue for scaling.md`: every unsent run with its command.
3. Close-out deck (slides) and close-out report (short paper), with every number traced.
4. Records: rulings, compute tally, tree and task hub up to date, status page, handoff note.
5. Branch tests pass; PR mehdiforoozandeh/CANDII#61 ready for the PI to review and merge.
