# t390 span_fix: SPAN_EXTRA 24 -> 25 in the count-to-signal converter

## The defect and the fix

`tools/onion_eval/signal_from_counts.py` converts 25 bp overlap counts to MACS2 pileup units with
`count * d / (rl + SPAN_EXTRA)`, i.e. it divides by the mean number of bins one read touches, times 25.
The constant was `BIN - 1 = 24`. The store count rule adds 1 to bins `start//25 .. end//25` inclusive,
with a half-open BED end `end = start + rl` (`tools/t112/bin25.py::counts_from_tagalign`;
`tools/t112/records.py` `COUNT_RULE`).

Derivation. Write rl = 25q + r with 0 <= r < 25, and let the start phase f = start mod 25 be uniform on
0..24. The read touches `(f + rl)//25 + 1` bins. `(f + r)//25` is 1 for r values of f and 0 otherwise, so its
mean is r/25. The mean bins touched is q + r/25 + 1 = rl/25 + 1 = (rl + 25)/25. A direct enumeration over
the 25 phases gave 2.44, 3.0, 4.0, 4.04, 5.04 for rl = 36, 50, 75, 76, 101, equal to rl/25 + 1.
The measured kappa (the mean bins per read on chr13, chr20, chr21, chr22) was 5.0401 / 5.0399 / 4.0200
(`../postpile/POSTPILE.md`). rl/25 + 1 gives 5.0401 / 5.0401 / 4.0201. The old constant gave
(101 + 24)/25 = 5.0 bins per read, so the pileup was (rl + 25)/(rl + 24) = 1.008 times too large. The scale factor changed by
(rl + 24)/(rl + 25) = 125.0015/126.0015 = 0.99206 for rl = 101.0015 bp (C19M22, `starts/*.json`).

The fix sets `SPAN_EXTRA = BIN` (commit 0f957f05). `tools/onion_eval/ag2_space.py` imports the constant,
so its normalized insertions, its pileup scale and its peak probabilities change by the same factor.
The separate copies in `tools/t125/hmap/caller.py` (`SPAN_EXTRA = BIN - 1`) and
`tools/t125/hmap/formula.py` (`SPAN_EXTRA = 24`, also used by `tools/t125/hmap/data.py:476`) belong to
another task and were not changed.

Tests. `tests/test_onion_eval_signal_from_counts.py` has a new test,
`test_span_extra_matches_the_bin25_rule`: it bins reads at all 25 start phases with
`counts_from_tagalign` and asserts that the mean bins per read equals (rl + SPAN_EXTRA)/25, for rl in
25, 36, 50, 75, 76, 100, 101. Known-answer tests that hard-coded 24 now use 25 (same file,
`test_onion_eval_nb_signal_checks.py`, `test_onion_eval_ag2_space.py`, `test_onion_eval_ag2_dump.py`).
`test_normalized_insertions_sum_to_1e8` had simulated a strict-overlap rule (`(start + rl - 1)//25`); it now
simulates the store rule. One Monte Carlo test (`test_box_d_peak_probability_is_on_the_pileup_scale`)
got tolerance 0.015 in place of 0.01: with the new scale, seed 6 gave a max error of 0.0133 at 3.9 SE,
with mean z = -0.06 (no bias). The darwin `scores.json` digests of `test_onion_eval_ag2_dump.py` were
re-pinned. Its suffstats and ledger digests did not change. The linux pins were already stale and still
need a Nibi run. All 194 tests in the 9 test files that touch the converter passed locally.

## The constant is the only change (C19M22)

| check | bins | differing | file |
|---|---|---|---|
| new code with SPAN_EXTRA patched back to 24 vs old rung L6 | 121,241,707 | 0 | `checks.tsv` `ident_span24_vs_L6old` |
| new code (`ladder_probe.py counts_route`) vs the entry pipeline's raw track | 121,241,707 | 0 | `ident_span25_vs_entryraw` |
| entry pipeline raw vs new L6 (`score_entry.py check --ref-l6`) | 121,241,707 | 0 | `entry check vs NEW rungs` |
| entry pipeline qn vs new L7 (ladder callpeaks + qnorm on the new L6) | 121,241,707 | 0 | same |
| peaks called on the new L6 by the ladder and by the entry pipeline | Jaccard 1.000000 | | `ident_peaks_vs_entry` |

The first row shows that the new code reproduces old rung L6 bit for bit when only the constant is set
back. So the old-vs-new difference is the constant alone. The -log10 p signal is not a pure multiple of the
old one: the pileup scales by 0.99206, but the score is `-log10 P(Poisson(lambda) > floor(pileup))`. The
floor and the lambda_bg term (d * T / G, which does not scale) make the change non-uniform. New L6 vs old
L6 differs in 73.1 M / 64.2 M / 37.7 M bins (C19M22 / C40M17 / C12M02), max abs 1.63 / 1.37 / 2.25.

`score_entry.py check` now takes `--ref-l6`, `--ref-l7`, `--ref-moesm2`, `--ref-moesm4` (default: the
probe-1 rungs and scores; `none` skips a score check). With the new references it passes (rows above).
Against the old rungs it fails by design. That failure is the exit 1 of the three entry test jobs.

## nMSE against the exact MACS2 formula (`nmse.tsv`)

nMSE = mean((A - B)^2) / var(B) on the scorer's grid (`ladder_probe.py compare`), 121,240,998 bins.
L5 is the exact 1 bp formula and L3 is the pipeline track. "Before" is `../ladder_probe/rung_compare.tsv`.

| experiment | L6 vs L5 before | after | L6 vs L3 before | after | L7 vs L4 before | after |
|---|---|---|---|---|---|---|
| C19M22 (H3K4me3) | 0.01716 | 0.01750 | 0.01708 | 0.01734 | 0.04375 | 0.04385 |
| C40M17 (H3K27me3) | 0.2550 | 0.2533 | 0.2552 | 0.2535 | 0.2825 | 0.2806 |
| C12M02 (DNase-seq) | 0.02124 | 0.02115 | 0.02116 | 0.02110 | 0.05478 | 0.05610 |

The fix moved nMSE by at most 2.0% of its value (C19M22 +2.0%, C40M17 -0.7%, C12M02 -0.4%), in both
directions. The count-route error is not a scale error: it comes from the 25 bp overlap counts and the
box spread, as `../postpile/POSTPILE.md` found. The noise floor of the formula itself is L5 vs L3 nMSE
6.1e-5 / 1.1e-4 / 1.4e-5 (`../ladder_probe/rung_compare.tsv`), far below these values.

## Scores and ranks (`placement.tsv`, `combined.tsv`)

mse among the 25 entrants (all23). Before: `../ladder_probe/placement.tsv` (qn, se) and
`../moesm2/placement.tsv` (MOESM2).

| truth | experiment | rung | mse before | after | rank before | after |
|---|---|---|---|---|---|---|
| qn | C19M22 | L6 | 13.843 | 14.004 | 1 | 2 |
| qn | C19M22 | L7 | 3.5938 | 3.5937 | 1 | 1 |
| qn | C40M17 | L6 | 0.11678 | 0.11695 | 1 | 1 |
| qn | C40M17 | L7 | 0.15778 | 0.15972 | 1 | 1 |
| qn | C12M02 | L6 | 191.07 | 191.66 | 10 | 10 |
| qn | C12M02 | L7 | 33.919 | 31.265 | 1 | 1 |
| se | C19M22 | L6 | 0.37297 | 0.37027 | 1 | 1 |
| se | C19M22 | L7 | 14.418 | 14.399 | 18 | 18 |
| se | C40M17 | L6 | 0.066474 | 0.066458 | 1 | 1 |
| se | C40M17 | L7 | 0.18905 | 0.19178 | 1 | 1 |
| se | C12M02 | L6 | 0.12811 | 0.12774 | 1 | 1 |
| se | C12M02 | L7 | 298.11 | 288.81 | 26 | 26 |
| MOESM2 | C19M22 | L6 | 893.92 | 895.24 | 19 | 19 |
| MOESM2 | C40M17 | L6 | 0.50613 | 0.50630 | 1 | 1 |
| MOESM2 | C12M02 | L6 | 552.48 | 553.06 | 21 | 21 |

The organizers' combined rank (entry added alone, bootstraps 0-9, of 26) did not change. MOESM2 raw (L6):
15 / 1 / 17.1 before and after. MOESM4 qn (L7): 1 / 1 / 1 before and after. Before:
`../entry_pipeline/board_moesm{2,4}.tsv`. One per-measure rank changed: qn mse of C19M22 L6 went from 1 to 2.

Why C12M02 qn moves most (mse 33.92 -> 31.26; 77.3 M bins differ, max abs 884). DNase-seq has no control,
so lambda = max(lambda_bg, local pileup). The pileup scales by 0.992 but lambda_bg does not, so -log10 p
falls near the peak threshold. The peaks called on the new L6 cover 78.80 M bp against 81.09 M bp before
(Jaccard 0.954; `peakcheck_L6new_vs_L6old`). QN is rank-based within the peak mask, so a 3% change of the
mask reassigns reference quantiles to many bins. For the ChIP experiments the control pileup scales with
the treatment, the peak sets move less (Jaccard 0.995 and 0.978), and qn moves less.

## Jobs (Nibi, `resources.tsv`; TotalCPU is not recorded, CPU-h = elapsed x 2 CPUs)

| job | id | state | elapsed |
|---|---|---|---|
| entry test C19M22 | 28760260 | FAILED (exit 1 of `check` vs old rungs, by design; all outputs written) | 13:14 |
| entry test C40M17 | 28761137 | FAILED (same) | 12:44 |
| entry test C12M02 | 28761138 | FAILED (same) | 10:52 |
| sf C19M22 / C40M17 / C12M02 | 28760263 / 28761140 / 28761141 | COMPLETED | 11:50 / 13:02 / 11:37 |
| sfid C19M22 (identity) | 28760266 | COMPLETED | 0:54 |
| sfid2 C19M22 (new L6/L7, check vs new) | 28767838 | COMPLETED | 5:45 |
| entry collect | 28760261, 28761139, 28767768 | CANCELLED; collect run on the login node (11 s) | |

Total 79.9 min elapsed x 2 CPUs = 2.67 CPU-h. Outputs: `/scratch/mforooz/eicx/spanfix/{entry,qc,scores,ident}`;
the old rungs under `/scratch/mforooz/eicx/ladder` were only read.
