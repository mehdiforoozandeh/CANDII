# State

## Goal now
Decide read-start vs read-count fairly, with the explanation (q94), and find why miniCANDI fails to follow dedup and SE/PE, and fix it (q95). Supports the main-text claim that CANDI corrects what broke the EIC (D1).

## Horizon
No deadline, no cap (D13). Iteration 1 DONE 2026-11-28; iterations 2-4 take 3-4 days each.

## Iteration 1 result (cruxvault/results/t396/ITERATION1.md, merged edc427b6, golden bit-exact)
- Leg B: start-space test void (deconvolution gate failed, D17); read-count leads in count (0.543 vs 0.375) and 1 kb (0.394 vs 0.184) beyond seed range. h57 no answer yet (f163).
- Leg A: covariance conversion fixes read-start calibration (1.258 -> 1.032; ref 1.064); like-for-like inside seed range (f162).
- Dedup cap: start-space cap beats count law (1.35, CI 0.29-3.41) but explains no val pairs (-0.56); h58 mixed (f161).
- SE/PE: inconclusive (f160); EIC holds 20 SE pairs from 2 experiments. PE depth premise false (f158). 1 bp dedup exact per replicate (f159).

## In flight
- Nothing on the cluster. Milestone report to the PI sent 2026-11-28 with one strategic ask (SE/PE data).

## Next
1. Iteration 2 plan: the 2x2 factorial (C13 to-do list in ITERATION1.md), dump memory > 20000M, H3K36me3 as its own row.
2. Iteration 3: learned / per-library cap output layer in start space.
3. File a task for the read-start adapter per-batch stitch defect.

## Open asks to the PI
none (SE/PE source answered: D18; slam-dunk route answered: D17)
