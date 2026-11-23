# Post-prediction pileup probe (t390): can a 25 bp count be spread into a fragment pileup?

Written 2026-11-23 on branch `implementation/t390-eic-pvalue-entry`. This probe extends probe 2 (`../ladder_probe2/PROBE2.md`).
It uses the same 3 blind experiments and the same observed reads. No model, no quantile normalization (QN) and no truth scoring is involved.
Code: `tools/eic_entry/ladder_probe.py` subcommand `postpile_route`, `tools/eic_entry/run_probe.sh pp`, `tools/eic_entry/collect_postpile.py`.
`postpile.tsv` comes from `bash runall.sh`.

## Question

miniCANDI predicts 25 bp counts. Can a step after the prediction spread each 25 bp count over the fragment length d, build an approximate fragment pileup, and so recover most of the gain of the true fragment pileup (L6f) over the count route (L6)?

## What L6's box_d already does

L6 (`ladder_probe.py counts_route`, line 305) calls `signal_from_counts.signal_track(..., mode="plug_in", pileup="box_d")`.
For ChIP-seq, `signal_from_counts.py:536` spreads the 25 bp overlap counts of the treatment and the control with `_spread`. `_spread` calls `box_spread` (line 470), which convolves the counts with a centred box of d bp whose weights sum to 1 (`box_weights`, `box_spread` line 418).
Line 540 then multiplies the box mean by s = d / (rl + 24) (`SPAN_EXTRA = 24`, line 161) and scores floor(s * m) against MACS2's lambda (`plug_in_neglog10p`, line 287). DNase-seq uses the same spread at lines 523-524.

So L6 already is "25 bp overlap counts, spread over a centred box of d, as a pileup estimate". The proposed step for today's miniCANDI output equals L6 in its count representation. L6 differs from the variants below only in the conversion (integer floor and MACS2's own lambda in L6; Max's continuous "pf" conversion here). Variant A below measures the proposed step through the L6f conversion, so the answer does not depend on that difference.

## Variants

All variants use the L6f conversion unchanged: lambda = `lam_hat25` from the control's (DNase-seq: the treatment's) 25 bp overlap counts / kappa, and score = `mlog10_sf(pile25 + 1, lambda)` (`ladder_probe.py:793-794`, the same lines as `cmd_fragpile_route` 704-705). T is the tagAlign line count (line 762), not a sum of counts. Only `pile25`, the 25 bp pileup estimate, differs.

| variant | input (25 bp) | pile25 | code |
|---|---|---|---|
| exact | 1 bp read 5' ends | the true 25 bp mean of the 1 bp fragment pileup (this reproduces L6f) | `postpile_route` line 780 |
| A | overlap counts (what miniCANDI predicts) | counts / kappa, then the centred kernel | lines 774, 784 |
| D | overlap counts | counts / (rl / 25 + 1), then the centred kernel | lines 774, 784 |
| B | unstranded read-start counts | starts of both strands, then the centred kernel | lines 786-788 |
| C | stranded read-start counts | + starts extended d forward, - starts extended d backward | lines 789-791 |

Definitions:
- kappa is the measured mean number of 25 bp bins one read touches (sum of overlap counts / reads over chr13, chr20, chr21, chr22; `kappa_of`, line 662).
- The kernel (`bin_kernel`, line 714) is the expected 25 bp-mean coverage by one fragment whose anchor is uniform over the 25 bp of its bin. The centred kernel covers [x - d//2, x - d//2 + d). The + kernel covers [x, x + d). The - kernel covers [x - d, x), with x the BED end, as MACS2 and `pileup_from_positions` use. Each kernel sums to d / 25. A start in bin j adds kernel[k] to bin j + k (`apply_kernel`, line 728).

Readings that I chose:
1. "Divided by mean read length" (A): I divided by kappa, i.e. the measured reads-to-bins ratio, because only that turns overlap counts into reads per bin. kappa is 5.0401 / 5.0399 / 4.0200, and rl / 25 + 1 is 5.0401 / 5.0401 / 4.0201 (rl = 101.0, 101.0, 75.5 bp).
2. Variant D (the coordinator's add-on) divides by the expected bins per read, rl / 25 + 1. The store counts +1 to bins floor(start/25) through floor(end/25) inclusive (`tools/t112/records.py:42`, `COUNT_RULE`; `tools/t112/bin25.py:69-76`), so the expectation is rl / 25 + 1. This division does not cancel in the conversion. The background lambda is d * T / G with T from the line count (`lam_hat25`, line 647; T at line 762), the local lambda comes from the control / its own kappa, and `mlog10_sf` reads the absolute pileup. So D was run.
3. B centres the box on the read start bin (MACS2's symmetric-shift approximation for a model that predicts unstranded starts). It does not shift the box by d/2.
4. C for DNase-seq equals B. The pipeline extends both strands the same way (`--shift -75 --extsize 150`), so DNase-seq has no stranded rule. C was not run for C12M02.
5. The starts files on Nibi keep strand (`<chrom>_p` = + 5' ends, `<chrom>_m` = - BED ends). No tagAlign was re-read.

## Checks

| check | C19M22 | C40M17 | C12M02 |
|---|---|---|---|
| `exact` vs L6f, raw grid | 0 of 121,241,707 bins differ | 0 differ | 0 differ |

The new code with the true pileup reproduces L6f bit for bit. So the conversion is identical, and the variants differ from L6f only in `pile25`.

## Normalized MSE (`postpile.tsv`)

nMSE = mean((A - B)^2) / var(B), scorer grid (25 bp, blacklist bins deleted, 121,240,998 bins). The L6, L6f and L6-vs-L5 rows are from probes 1 and 2 (same `compare` files).

| row | what | vs L5: C19M22 | C40M17 | C12M02 | vs L6f: C19M22 | C40M17 | C12M02 |
|---|---|---|---|---|---|---|---|
| L6 | overlap counts, box_d, signal_from_counts conversion | 0.0172 | 0.2550 | 0.0212 | 0.0166 | 0.2647 | 0.0211 |
| A | overlap counts / kappa, centred spread | 0.0168 | 0.2314 | 0.0207 | 0.0151 | 0.1836 | 0.0206 |
| D | overlap counts / (rl/25 + 1), centred spread | 0.0168 | 0.2314 | 0.0206 | 0.0151 | 0.1836 | 0.0206 |
| B | unstranded starts, centred spread | 0.0614 | 0.6859 | 0.0018 | 0.0597 | 0.6148 | 0.0017 |
| C | stranded starts, directional spread | 0.0028 | 0.0746 | 0.0018 (= B) | 0.0007 | 0.0144 | 0.0017 (= B) |
| L6f | true 25 bp fragment pileup | 0.0021 | 0.0616 | 0.00003 | 0 | 0 | 0 |

Share of the L6-to-L6f gain (vs L5) that each variant recovers, (L6 - X) / (L6 - L6f):
- A: (0.01716 - 0.01677) / (0.01716 - 0.00208) = 3% (C19M22); (0.25500 - 0.23143) / (0.25500 - 0.06163) = 12% (C40M17); (0.02124 - 0.02065) / (0.02124 - 0.00003) = 3% (C12M02). D is the same to 4 digits.
- B: negative on ChIP-seq (worse than L6: 0.0614 and 0.6859); 92% on DNase-seq ((0.02124 - 0.00176) / 0.02121).
- C: (0.01716 - 0.00277) / 0.01508 = 95% (C19M22); (0.25500 - 0.07460) / 0.19337 = 93% (C40M17); DNase-seq = B, 92%.

## Cost and jobs (Nibi, 2 CPUs per job, `sacct`)

| job | id | elapsed | peak RSS |
|---|---|---|---|
| eicx_pp_C19M22 (probe, with the exact check) | 28647409 | 4:50 | 10.3 GiB |
| eicx_pp_C40M17 | 28647648 | 3:51 | 10.3 GiB |
| eicx_pp_C12M02 | 28647649 | 2:28 | 9.4 GiB |

Total: (290 + 231 + 148) s x 2 CPUs = 1,338 CPU-s = 0.37 CPU-h (budget 3 CPU-h). All three jobs COMPLETED.
Arrays: `/scratch/mforooz/eicx/ladder/postpile/<exp>/{exact,A,B,C,D}/<exp>.npy` (purge-exposed); comparisons `qc/<exp>_pp*_vs_{L5,L6f}.json`, `qc/<exp>_L6_vs_L6f.json`.

## What this means

1. **Spreading today's 25 bp overlap counts over d recovers almost none of the fragment-pileup gain.** A and D recover 3%, 12% and 3%. L6's box_d already does this spread, so the post-prediction step adds nothing new. An overlap count has already lost the strand and the read position, and no spread after the fact restores them.
2. **The division by reads per bin is immaterial.** kappa and rl / 25 + 1 agree to 4 digits, so D equals A. It does not cancel, but it changes nothing.
3. **Unstranded starts help only DNase-seq.** For DNase-seq the pipeline itself centres the fragment on the start, so B is the right model and recovers 92%. For ChIP-seq, a box centred on the start puts each fragment d/2 off its true place, and B is worse than L6 (0.061 vs 0.017; 0.686 vs 0.255).
4. **Stranded starts recover 93-95% of the gain on ChIP-seq.** C reaches nMSE 0.0028 and 0.0746 vs L5 (L6f: 0.0021, 0.0616). This is the bound for a model that predicts read-start counts by strand at 25 bp. The rest of the gap is the position of each start inside its 25 bp bin.
5. **For a count-space model, the change must be in what the model predicts, not in a step after it.** A model that predicts stranded 25 bp start counts (ChIP-seq) or unstranded start counts (DNase-seq) can reach close to L6f. A model that predicts overlap counts cannot, whatever the spread.

## Not done

- The two stranded tracks would be two model outputs. How their errors combine is not tested.
- `signal_from_counts.py:161` uses rl + 24 for the bins per read. The store rule gives rl + 25 (kappa x 25 = 126.0 for rl = 101.0). This is a 1% scale difference in L6's pileup; it was not changed here.
