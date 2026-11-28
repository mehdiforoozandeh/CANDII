# State

## Goal now
Decide read-start vs read-count fairly, with the explanation (q94), and find why miniCANDI fails to follow dedup and SE/PE, and fix it (q95). Supports the main-text claim that CANDI corrects what broke the EIC (D1).

## Horizon
No deadline, no cap (D13). Target: iteration 1 measured result by 2026-11-30; later iterations 3-4 days each.

## In flight
- Iteration 1 wave 1: 7 builders (C1 deconv, C2 spaces, C3 covconv, C4 caps, C5 audits+golden, C6 dump tool, C12a 1bp tool + sc_untar job). Plan .orchestrate/plan.md (20 chunks, 5 waves).
- Branch exp/start-vs-count-dedup, draft PR 68 (base exp/t370-minicandi-recipe).

- Fir 6726827 sc_untar: 6 train experiments' tagAligns from tar 005 -> /scratch/mforooz/sc396/work (for the 1 bp dedup test).
- Merged: C1 C2 C3 C4 C12a. Waiting: C5 builder, C6 reviewer.

## Next
1. Planner writes the iteration 1 chunks (leg B deconvolution + 3 score spaces; covariance conversion; dedup cap test; SE/PE cap test; PE depth check).
2. Builders implement and probe one job each on Fir, then batch.
3. Record findings on h57, h58, h61; plan iteration 2 (factorial, 256 regions).

## Open asks to the PI
none
