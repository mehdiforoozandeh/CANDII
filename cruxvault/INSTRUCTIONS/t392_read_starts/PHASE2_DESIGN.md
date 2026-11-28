---
title: Phase 2 design: transforms from predicted read starts
task: t392 (read-start miniCANDI)
status: design only. No compute and no code until the PI starts Phase 2 (PI via foreman, 2026-11-24).
written: 2026-11-24
---

# Phase 2 design: from predicted read starts to the observed formats

## 1. Scope

Phase 1 trains miniCANDI to predict a Negative Binomial (NB) per 25 bp bin of read starts (that is, the 5' base of each read). ChIP-seq gets 2 strand channels. DNase-seq and ATAC-seq get 1 channel. Phase 1 converts that prediction to overlap counts only, for scoring.

Phase 2 adds 3 more transforms from the same prediction:

1. overlap counts (this exists, from Phase 1);
2. fragment pileup, i.e., the count of fragments that cover each position after each read is extended to fragment length d;
3. fold enrichment (FE), i.e., fragment pileup divided by the local background rate lambda;
4. signal p-value, i.e., -log10 P(Poisson(lambda) > pileup), the ENCODE/EIC track format.

Each transform must keep the NB uncertainty, so that a predicted track carries an uncertainty in every format.

## 2. Ground truth this design builds on

| fact | source |
|---|---|
| Read starts rebuild eic_v1_2 overlap counts exactly (0 differing bins, 12 of 12 probe files) | run_state/plan.md Log, 2026-11-24T01:40Z; Fir array 28735210 logs |
| Span kernel: `kernel[s, 8 + k]`, the mean reads of strand s per start in bin j that cover bin j + k, k = -8..8 | `tools/minicandi/read_start.py:15`, `STORE.md` (store attributes `kernel`, `n_reads_strand`) |
| Moment-matched NB of a weighted sum of start counts | `tools/minicandi/read_start.py::convert_nb` (line 75) |
| PR 64 count-to-p-value route: box spread of d bp, MACS2 lambda = max(lambda_bg, lambda_d, lambda_1k, lambda_10k) of the control scaled by T/C, then -log10 Poisson tail | `tools/onion_eval/signal_from_counts.py:213` and `tools/eic_entry/ladder_probe.py:286-312` on branch `implementation/t390-eic-pvalue-entry` |
| PR 64 exact 1 bp route from read 5' ends (rung L5) matches the ENCODE pipeline output to nMSE 6.1e-5 / 1.1e-4 / 1.4e-5 (C19M22 / C40M17 / C12M02) | `ladder_probe.py:193-284`; `cruxvault/results/t390/` on that branch |
| From observed reads, p-value at 25 bp vs the exact route L5 (nMSE, i.e., mean squared error over the variance of L5): overlap counts 0.0172 / 0.2550 / 0.0212; stranded starts 0.0028 / 0.0746 / 0.0018; unstranded starts fail on ChIP (0.0614 / 0.6859) | `cruxvault/results/t390/postpile/POSTPILE.md` on that branch |
| PR 64 uses a fixed d per experiment from the pipeline metadata (205, 200, 150); it has no d estimator | `tools/eic_entry/score_entry.py:82-84` on that branch |

The PR 64 numbers come from the observed reads of each experiment. They are oracle bounds, not model results. They show that stranded read starts are the right input for transforms 2 to 4. Overlap counts lose 3 to 12 times more.

## 3. Transform 1: overlap counts (exists)

Mean: the per-strand convolution of the start means with the span kernel, summed over strands. Uncertainty: the moment-matched NB of `convert_nb`. Phase 2 changes nothing here. The gate and the floor of Phase 1 are its validation.

## 4. Transform 2: fragment pileup

### 4.1 The map

A + strand start at base x covers [x, x + d). A − strand start at base x covers [x − d + 1, x + 1). At 25 bp, the expected coverage of bin j + k per start in bin j is the fragment kernel F[s, k]. It is the fraction of the d bases that fall in bin j + k, averaged over the start's position inside bin j. The pileup mean is

    pile[i] = sum over s, k of F[s, k] * mu_starts[s, i - k].

This is the same linear form as `convert_nb`, with F in place of the span kernel. The code reuse is direct. DNase and ATAC use a centred kernel of width d (PR 64 uses d = 150, `[x - 75, x + 75)`), which needs no strand.

### 4.2 Measure F; do not assume it

The start position inside a bin is not uniform near peaks. So F must be measured from the observed reads, the same way the span kernel was measured: extend each read to d at 1 bp, then count, per start bin, the bases that fall in each neighbour bin. This is one more output of the Phase 1 recount scan (`tools/onion_data/read_starts.py::scan`), with one more difference array. Its gate: the measured-F pileup from observed starts against the exact 1 bp pileup (PR 64 rung L5 method), binned to 25 bp. Its floor is the nMSE of that pair, as for overlap counts.

### 4.3 Estimate d per experiment

Three sources, in this order:

1. The pipeline's own value (`metadata.json` `--fraglen`), when the experiment has one. This is what the EIC truth tracks used.
2. Strand cross-correlation of the observed + and − starts, i.e., the lag that maximizes corr(plus[i], minus[i + lag]). Exclude the read-length phantom peak. This is the SPP/phantompeakqualtools method. At 25 bp the lag is quantized to 25 bp. A parabola through the peak and its 2 neighbours gives a sub-bin value. Validate it against method 1 where both exist.
3. For a predicted experiment with no reads (zero-shot imputation), apply method 2 to the predicted + and − start means. The model predicts both strands, so the shift between them carries d. As a fallback, take the median d of training experiments of the same assay.

Sensitivity: a d error of 25 bp changes the pileup kernel by one bin. Report the pileup nMSE at d ± 25 bp on the probe set, so that the d error budget is known.

### 4.4 Uncertainty: moment match or exact

The pileup of bin i is a weighted sum of about 2 × (d/25 + 1) start counts, i.e., about 18 terms at d = 200. Each term is an NB. The sum is not NB.

- **Moment match (default).** Match the mean and the variance with one NB, as `convert_nb` does: the excess variance over Poisson is v = sum (w·mu_sub)² / n_sub, and n = mu² / v, clamped to N_MIN..N_MAX. It is exact in the first 2 moments. It is sufficient for bin-wise mean and variance metrics, and for calibration of the central region.
- **Exact (only where the tail matters).** The p-value (section 6) is a tail quantity. Compute the exact PMF of the sum by convolution of the term PMFs (FFT, support truncated at the 1 − 1e-6 quantile), or by a saddlepoint approximation. Use it only to measure the moment-match error on a probe subset. Use it in production only if that error exceeds the threshold below.
- **Decision rule.** On the 16 regions, compare the moment-matched and the exact tail probabilities P(pileup > k) at k = the 0.99, 0.999 and 0.9999 quantiles. If the median relative error of -log10 of those tails is < 5%, moment match ships. Else exact ships for the p-value step only.

Neighbour bins share start terms, so their pileups are correlated. Bin-wise metrics need only the marginals. Region or peak metrics need the joint: section 6.3 handles it by sampling.

## 5. Transform 3: lambda and fold enrichment

MACS2 lambda (PR 64 `macs2_lambda`): lambda = max(lambda_bg, r·C_d, r·C_1k, r·C_10k). lambda_bg = d·T / G (G = 2.7e9). C_w is the control pileup averaged over a window of w bp. r = T / C, the treatment-to-control read ratio. DNase (no control) takes max(lambda_bg, lambda_10k of the treatment).

From read starts this is exact up to bin edges. All the window sums are linear in the control starts, and 1 kb and 10 kb are multiples of 25 bp. T and C are sums of the start means, so no read-length factor (`kappa`, `SPAN_EXTRA`) is needed. PR 64 needed those factors only because it started from overlap counts.

The control can be (1) the observed control starts, or (2) the predicted control (ChIP control is an output assay of the recipe). Run the observed-control path first; it isolates the treatment error. The predicted-control path adds the control error, and it is the only path for zero-shot imputation without a control experiment.

FE = pileup / lambda. Its uncertainty: the treatment NB divided by a lambda that is treated as fixed when the control is observed. With a predicted control, lambda is a max of sums with its own NB. Take it by sampling (section 6.3); a closed form for the max is not worth the effort.

## 6. Transform 4: signal p-value

### 6.1 Point value

g(k; lambda) = -log10 P(Poisson(lambda) > k). This is the PR 64 `plug_in_neglog10p` with ENCODE bdgcmp rounding (`ptrack25`, `ladder_probe.py:227`).

### 6.2 Two point estimates, and which one to score

- Plug-in: g(E[pileup]; lambda). This is what PR 64 scores.
- Expected: E[g(pileup; lambda)] = sum over k of P(pileup = k) · g(k; lambda), under the pileup NB. g is convex in the peak region, so the expected value is larger than the plug-in value (Jensen gap).

The truth track is g of a single observed pileup. The model that minimizes squared error against it predicts E[g], not g(E). So score E[g], and report the plug-in as well. The gap between the two is a direct measure of how much the uncertainty matters for the p-value.

### 6.3 Propagate the uncertainty

- Bin-wise: the distribution of g(pileup) follows from the pileup PMF through the monotone map g. Its quantiles are g of the pileup quantiles. This is exact and cheap: no sampling.
- Joint (peaks, region metrics, predicted control): Monte Carlo. Draw S = 100 samples of the start counts per bin from the start NBs. Run each sample through the deterministic chain of sections 4 to 6. Take the metric's spread over the samples. This handles the bin correlations and the max in lambda without approximation. Start samples are independent per bin by the model's own assumption. That assumption is part of what this check tests.
- Calibration check: the PIT, i.e., the predicted CDF at the observed value, of the observed p-value track under the predicted p-value distribution, on held-out experiments. A flat PIT histogram means calibrated uncertainty.

## 7. Validation of each transform

| transform | gate on observed reads (must hold) | floor on observed reads (report) | model check |
|---|---|---|---|
| overlap counts | 0 differing bins vs eic_v1_2 (done, Phase 1) | conversion floor (Phase 1) | Phase 1 headline metrics |
| fragment pileup | measured-F pileup from starts vs exact 1 bp pileup binned to 25 bp: within the floor; d from method 2 vs method 1 within 25 bp on the probe set | nMSE vs the exact 1 bp pileup | nMSE and calibration of the predicted pileup vs the observed pileup |
| lambda, FE | lambda from starts vs the PR 64 L5 lambda: bit-exact on window sums, up to bin edges | nMSE of FE vs L5 FE | same, predicted vs observed |
| signal p-value | rerun PR 64 postpile variant C (stranded starts) on our store and reproduce its nMSE 0.0028 / 0.0746 / 0.0018 against L5 | that nMSE, per experiment | plug-in and expected p-value vs the EIC truth (MOESM2 and qn), with PR 64's scorer unchanged; Jensen gap; PIT |

The PR 64 scorer is reused unchanged. Its own checks hold: 0 of 69 chromosome arrays differ in truth binning, and the scorer matches MOESM2 to relative error 1.9e-8 (`cruxvault/results/t390/moesm2/` on that branch). Phase 2 adds transforms only, never a scorer change.

## 8. Cost per step

Phase 1 measured: the recount at 0.16 s per MB of gzipped reads at 2 CPUs (probe, Nibi array 28735210). PR 64 measured: 0.73 to 1.3 CPU-h per entry track, 3.6 CPU-h per experiment for the full oracle ladder (`LADDER_PROBE.md:215-220` on that branch). The other figures below are estimates.

| step | what runs | estimate | arithmetic |
|---|---|---|---|
| F kernel and d by cross-correlation (observed) | one more output of the Phase 1 recount | about 5 CPU-h for 2,740 measurements | the recount is about 48 CPU-h; one more difference array adds about 10% |
| pileup, lambda, FE, plug-in p-value per predicted track | linear filters over 1.2e8 bins | < 0.05 CPU-h per track | about 18 taps × 1.2e8 bins ≈ 2e9 operations |
| expected p-value and bin-wise quantiles | sum over the NB support | about 0.1 CPU-h per track | 1.2e8 bins × about 50 support points |
| exact tail check | FFT convolution on 16 regions | about 2 CPU-h once | small regions; used once, for the decision rule |
| Monte Carlo, S = 100 | 100 × the deterministic chain on the scored regions | about 5 CPU-h per track on 16 regions | 100 × (pileup + lambda + p-value) |
| scoring against the EIC truth | PR 64 scorer, unchanged | 0.73 to 1.3 CPU-h per track | PR 64 resources.tsv |
| oracle ladder on observed reads (validation) | PR 64 L5 route on the probe experiments | 3.6 CPU-h per experiment | PR 64 measured |

The model's prediction pass is not in this table. It is the Phase 1 predict pass, which Phase 2 reuses.

A first Phase 2 pass on the 3 PR 64 experiments, with 3 seeds, costs about 3 × 3.6 (oracle) + 9 × (0.15 + 5 + 1.3) + 2 + 5 ≈ 76 CPU-h.

## 9. Order of work, when the PI starts Phase 2

1. Add the F kernel and the cross-correlation d to the recount (code plus tests; no new data pass if it rides with a rerun).
2. Gate the fragment pileup and the d estimate on observed reads (section 7 rows 2 and 4).
3. Run the moment-match vs exact decision on 16 regions.
4. Convert the Phase 1 predictions; score the plug-in and the expected p-value with the PR 64 scorer.
5. Monte Carlo and PIT for the calibration claim.

## 10. Choices decided by the PI (2026-11-24)

The PI decided these on 2026-11-24 ("yes"; relayed by the foreman). Phase 2 stays on hold until the Phase 1 result is reported.

1. Headline: the expected p-value E[g]. The plug-in p-value is reported beside it.
2. Fragment length d for zero-shot experiments: two arms, both reported. The default arm takes d from the cross-correlation of the predicted strands. The second arm takes the assay median. The pipeline value serves as a reference where it exists.
3. Zero-shot control: the recipe's predicted control. The observed-control path still runs first, where control reads exist.
