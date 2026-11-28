# Requirements ledger

One row per requirement in the PI's documents. Tier: P0 the run fails without it, P1 expected, P2 if time allows, P3 do not build.
Status: open, partial, done, dropped (dropped needs a PI decision id in Evidence). Done needs a measured result linked in Evidence.
Provenance: PI doc, PI, agent proposal approved.

| ID | Tier | Requirement (quoted) | Source | Provenance | Status | Evidence |
|---|---|---|---|---|---|---|
| R1 | P0 | "investigate whether we are better off using the read count formulation or the read start formulation" | T:5 | PI doc | open | |
| R2 | P0 | "set up the background and where we are" on read start vs read count | T:5 | PI doc | done | BACKGROUND_AND_PLAN.md s1 (2026-11-27) |
| R3 | P0 | review the read-start data processed, its evaluation, and the comparison | T:5 | PI doc | done | BACKGROUND_AND_PLAN.md s1.1-1.4 |
| R4 | P0 | "plan for actually investigating this further" | T:5 | PI doc | done | BACKGROUND_AND_PLAN.md s3 |
| R5 | P1 | "pros and cons list for each one" | T:5 | PI doc | partial | BACKGROUND_AND_PLAN.md s2; update with results |
| R6 | P1 | use "the empirical results that we get to see" in the decision | T:5 | PI doc | open | |
| R7 | P0 | "compare them in a very fair setting" | T:5 | PI doc | open | |
| R8 | P0 | fairness: no advantage in "data availability" | T:5 | PI doc | open | |
| R9 | P0 | fairness: no advantage in "more compute" | T:5 | PI doc | open | |
| R10 | P0 | "find the actual scientific explanation" for which formulation does better | T:7 | PI doc | open | |
| R11 | P0 | leg A: "take the read start model, extend the reads ... evaluate that against the read counts" | T:3 | PI doc (Max) | partial | t392 + covariance redo: results/t396/legA_covariance/LEG_A.md (1.258 -> 1.032 vs ref 1.064); factorial redo in it.2 |
| R12 | P1 | leg A reading: worse start model = "more evidence that indeed the read start model is worse" | T:3 | PI doc (Max) | partial | LEG_A.md: like-for-like calibration inside seed range; count-space skill gap remains (LEG_B.md 0.543 vs 0.375) |
| R13 | P0 | leg B: "Take the read count model, use just the starts ... and evaluate" against read starts | T:3 | PI doc (Max) | partial | ran by deconvolution: results/t396/legB_spaces/LEG_B.md; start space void (gate c); factorial arm per D17 (PI confirmed) |
| R14 | P0 | slam dunk: "the read count model still does better when predicting read starts" | T:3 | PI doc (Max) | open | |
| R15 | P0 | the two predict "different things", so score both on a common target | T:3 | PI doc (Max) | open | |
| R16 | P1 | "check Sprint 19 in max's github repo for the analysis on SE/PE deduplication" | T:10 | PI doc | done | none on GitHub or Fir s19; see R17 |
| R17 | P0 | "we assume they dont exist. just look at max's repo (different sprints) and ... data on fir/nibi" | chat 2026-11-27 | PI | open | standing source constraint |
| R18 | P1 | "read and look at Max's GitHub repo, different sprints, their instructions, the findings" | T:30 | PI doc | done | BACKGROUND_AND_PLAN.md s1.6 |
| R19 | P1 | use the PI's information with Max's repo to inform the plan | T:30 | PI doc | done | BACKGROUND_AND_PLAN.md s3 |
| R20 | P0 | dedup and SE/PE "plagued the whole encode imputation challenge"; correct them | T:12 | PI doc | open | |
| R21 | P0 | main-text claim: CANDI corrects the covariate effects that broke the EIC | D1 | PI | open | |
| R22 | P1 | train/blind processed differently caused the mismatch; CANDI must handle it | T:12 | PI doc | open | blind rows wait on the Nibi build |
| R23 | P0 | "we are not equally able to correct for different covariates" (depth good, dedup poor) | T:16 | PI doc | open | |
| R24 | P0 | explain why dedup correction fails | T:16 | PI doc | open | |
| R25 | P0 | cause 1: "we don't have enough augmented data for some of the covariates" | T:22 | PI doc | open | |
| R26 | P1 | "whether we can actually provide and process new and more augmented data views" | T:22 | PI doc | open | |
| R27 | P2 | more views from the extended ENCODE / epiAtlas / ChIP-Atlas dataset | T:22 | PI doc | open | discuss only, no build |
| R28 | P0 | cause 2: "mechanical ways to make our model responsive to ... read deduplication" | T:24 | PI doc | open | |
| R29 | P0 | per covariate: "is there a known relationship that we can mathematically implement" | T:24 | PI doc | open | |
| R30 | P1 | the relationship in "read count data or read start data" | T:24-26 | PI doc | open | |
| R31 | P0 | cause 3: "losing some of the information ... in read count or overlap space" | T:28 | PI doc | open | |
| R32 | P0 | "providing more information to the model and preventing loss of information" in start space | T:28 | PI doc | open | |
| R33 | P0 | no decision rule before the data | D2 | PI | open | |
| R34 | P0 | gather headlines, mechanism results and the slam-dunk test, then decide | D2 | PI | open | |
| R35 | P0 | iterations 1-4 all run | D1 | PI | open | |
| R36 | P1 | the factorial, about 2,350 CPU-h | D3 | PI | open | |
| R37 | P2 | under-training: later, on the factorial winner | D3 | PI | open | |
| R38 | P3 | no 320,000-window arms in iteration 2 | D3 | PI | open | |
| R39 | P0 | correction headline gets dedup and SE pairs, breakdown per covariate | D4 | PI | partial | census: 458 dedup pairs (43 exp), 20 SE pairs (2 exp); results/t396/pair_census/ |
| R40 | P0 | start the longrun, iteration 1 onward | D6 | PI | done | 2026-11-27 kickoff |
| R41 | P1 | it.1: leg B deconvolution, three score spaces, pseudoreplicate floors | B s3 | agent proposal approved (D6) | partial | LEG_B.md: three spaces and floors done; deconvolution void (D17) |
| R42 | P1 | it.1: covariance term in the start-to-count conversion | B s3 | agent proposal approved (D6) | done | results/t396/legA_covariance/LEG_A.md: ratio 1.258 -> 1.032 vs ref 1.064; like-for-like inside seed range |
| R43 | P1 | it.1: closed-form dedup cap, start vs count space, 312 view pairs | B s3 | agent proposal approved (D6) | done | results/t396/dedup_cap/DEDUP_CAP.md: cap beats count law (diff 1.35, CI 0.29-3.41) but gap closed -0.56 on 43 val pairs; no law explains except DNase |
| R44 | P1 | it.1: SE/PE cap plus fragment kernel on PE/SE views | B s3 | agent proposal approved (D6) | partial | results/t396/sepe_cap/SEPE_CAP.md: gate d failed, inconclusive (f160); fragment kernel half not run; controlled pairs per D18 |
| R45 | P2 | it.1: does log2_depth count PE mates twice | B s3 | agent proposal approved (D6) | done | f158: counts and depth both count mates; results/t396/pe_depth/PE_DEPTH.md |
| R46 | P1 | it.2: 2x2 arms with kernel head | B s3, D3 | PI | open | |
| R47 | P0 | it.2 equal terms: tagAligns, windows, params, seeds, tuning sweep, 256 regions, three spaces | B s3, D3 | PI | open | |
| R48 | P1 | it.3: closed-form cap as an output layer; prompt test and dedup/SE pairs | B s3, D1 | PI | open | |
| R49 | P1 | it.4: retrain with 25/50/100% of dedup-off and SE views | B s3, D1 | PI | open | |
| R50 | P2 | record hypotheses in the notebook | B s4 | agent proposal approved | partial | q94, q95 recorded |
| R51 | P1 | each iteration 2-4 days, ends with a result | B s3 | agent proposal approved | open | |
| R52 | P1 | quote the noise floor with every number; CRPS only with its split | CLAUDE.md | project rule | open | standing |
