---
id: f39
type: finding
schema: 3
title: The AlphaGenome 2 side is sound. Our AG2 scorer reproduces Max's score_f1.py…
parent: q27
confidence: moderate
evidence: results/t276/MAX_REPRO.md, results/t267/API_SMOKE.md, results/t284/FETCH.md
created: "2026-11-14T02:18:32"
updated: "2026-11-14T02:18:32"
---

# f39 — The AlphaGenome 2 side is sound. Our AG2 scorer reproduces Max's score_f1.py…

Parent:: [[q27_are_the_onion_run_s_estimators_scorers_a]]

## Answer

The AlphaGenome 2 side is sound. Our AG2 scorer reproduces Max's score_f1.py (his repository at bde905a, run unchanged) on identical inputs: gate PASS, 12 of 12 targets and 48 of 48 comparisons within tolerances pinned before any difference was computed; largest absolute difference 4.9e-7 on the correlation metrics (rspec_cal 4.5e-7, pearson_cal 4.9e-7, spearman 4.6e-7) and 3.9e-6 on AUPRC (float32 against float64), with Max's rule of dropping positive tiles that are not whole valid tiles (1 to 1,473 per target, counted). API facts: one call returns ChIP-seq, DNase-seq and ATAC-seq; the ChIP-seq value per 128 bp bin is the sum of per-bp fold change (median over channels 99.669); about 1.0 s per call; the fetch on the 256 evaluation regions made 8,960 calls (35 conditions x 256 windows), all successful, 0 retries, no quota stop, 2,188 s; audit statuses agree with the API's imputation_mode labels on all 318 cached rows (0 leaks); no copy of the API key in the outputs. Max's own Benchmark F1 table did not exist yet, so the comparison is against his code run here.
