---
title: "R2: close-out deck and written report"
parent: "[[closeout todo]]"
created: 2026-11-16
---

# R2: close-out deck and written report

## 1. What the PI asked

- `long-horizon prompt.md`:28: "at the end of this, please create a /prezit and a detailed 12 page /storify reports with headline results and findings (+ methods + discussion)"
- `long-horizon decisions.md`:107 (D14): "If the ladder is complete and condition 2 fails, the run stops and reports. Then the run produces the /prezit deck and the 12-page /storify report that [[long-horizon prompt]] asks for."
- PI directive, plan Log 2026-11-16T01:10Z: "the final report must show whether and where the counterfactual views (EIC processed in different ways) help."
- The four success criteria (`closeout todo.md`, PI 2026-11-16): augmented EIC data and counterfactuals; a training objective well defined; a well defined and tested set of evals; the miniCANDI design settled, tested and ready for scaling.

Success criteria served: all four; this item reports them. Audit rows: G2, D28 (report part), F4.

## 2. Question

Deliver one deck (prezit) and one 12-page written report (storify) that state, for each success criterion, what was delivered, the headline results with seed ranges and tier, and what is left for the follow-up work.

## 3. State today

- No final deck or report exists (audit row G2). Earlier decks: ship 1 (artifact 4RSUdsFwJC3nYcyHb9s2bU), ship 2 (RHucrLM3AXXxVfzQtCqEDS, 14 slides), "Peeling the Variance" (H8ShBq8tJ1eMzwpjNpdU87); the 5-page report for Max (`presentations/onion-update-max-2026-11-15/report.pdf`), which lacks the view result.
- The source text is `onion paper.md` (Background, Methods, Results 1-8, Discussion), brought up to date by R1.

## 4. Prior leads

- PI preferences (memory): decks over PDF reports; figures over text, only grids as tables, labelled panels, basic explanations.
- Prior close-out pattern: "CAC Testbed Close-out (32 slides)", artifact YPwpootBTpn3pkY1nGVQGB (`presentations/ARTIFACTS.md`).

## 5. Work

1. Wait for R1 (records), and for the results of D1 to D4, E1 to E4, O1, A1 to A10 and M1. A section whose item is blocked says so with the reason.
2. Deck (`/prezit`), one part per success criterion:
   - Data: `eic_v1` and its update, counterfactual views, second experiments, blind build, covariates filled.
   - Training objective: the O1 document in one figure.
   - Evals: the metric set, the variance layers with the completed ladder (E4), the settled-tier check (E1), the AlphaGenome 2 and EIC boards (E2, D4), the metric tables (E3).
   - Design: the architecture comparison at equal windows and matched compute, every ablation with its D13 verdict, where the counterfactual views help, the M1 design and its scaling risks.
   - Follow-up work: the PI's list (scaling; training and evaluation region counts; the extended dataset).
   Run `crux deck --verify` (strict).
3. Report (`/storify`): 12 pages, sections Introduction (from the constitution), Methods, Results, Discussion, Conclusions; run `crux story --verify`.
4. Publish both as private artifacts; mirror each into `cruxvault/presentations/` with its URL and add both to `ARTIFACTS.md`.
5. Give the PI the links. Publishing anything public is the PI's action.

## 6. How the result answers the question

Acceptance checks:
- `crux deck --verify` and `crux story --verify` pass with 0 untraced numbers.
- The deck has one part per success criterion, each with a "met / partly met / not met" line and its evidence.
- The report is 12 pages in its PDF build.
- The views result appears with per-architecture and per-knob numbers (A3).
- Every model number carries its tier (screening or settled) and seed range; every CRPS-based number carries its noise floor (`AGENTS.md` section 7.2).

## 7. Dependencies, cost, risks

- Depends on all items.
- Cost: about 1 agent-day each for the deck and the report (audit row G2).
- Risk: an item still running at the deadline. Build from what exists, label the missing parts, and do not wait without the PI's word.

## 8. Done when

The close-out deck and the 12-page report are published as private artifacts, pass their verify checks, are mirrored in the vault and listed in `ARTIFACTS.md`, and the PI has the links.
