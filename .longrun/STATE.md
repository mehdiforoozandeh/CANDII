# State

## Goal now
Decide read-start vs read-count fairly, with the explanation (q94), and find why miniCANDI fails to follow dedup and SE/PE, and fix it (q95). Supports the main-text claim that CANDI corrects what broke the EIC (D1).

## Horizon
No deadline, no cap (D13). Target: iteration 1 measured result by 2026-11-30; later iterations 3-4 days each.

## In flight
- Iteration 1 wave 1: 7 builders (C1 deconv, C2 spaces, C3 covconv, C4 caps, C5 audits+golden, C6 dump tool, C12a 1bp tool + sc_untar job). Plan .orchestrate/plan.md (20 chunks, 5 waves).
- Branch exp/start-vs-count-dedup, draft PR 68 (base exp/t370-minicandi-recipe).

- Fir 6726827 sc_untar: 6 train experiments' tagAligns from tar 005 -> /scratch/mforooz/sc396/work (for the 1 bp dedup test).
- Merged: C1 C2 C3 C4 C6 C12a (sc_untar 6726827 COMPLETED). C5 (audits, golden) still running.
- Wave 2 running: C7a dump submit, C8 scoring tool, C9 leg A tool, C10a cap test tools, C12b 1 bp run + unit rule, C13 kernel head (iteration 2 prep).

- Fir dump arrays: truth 6731315, model 6731316/6731317/6731318 (sc_dump; collect = C7b).
- Fir cap arrays: sc_dedup 6737535, sc_sepe 6737536 (collect = C10c, C11).
- 1 bp result: dedup is exact per replicate (f159).

## Next
1. Planner writes the iteration 1 chunks (leg B deconvolution + 3 score spaces; covariance conversion; dedup cap test; SE/PE cap test; PE depth check).
2. Builders implement and probe one job each on Fir, then batch.
3. Record findings on h57, h58, h61; plan iteration 2 (factorial, 256 regions).

## Open asks to the PI
none
