# Max's sprint 07 (2026-11-04 to 11-11): walkthrough

Deck: https://claude.ai/artifact/Cd4t3AQ9JdsisYvHRduuTc (local copy: `index.html`).
Source: github.com/mlibbrecht/01_2026-10-19_research, `projects/03_2026-10-20_epi_imputation/sprints/07_2026-11-04`.
Every number below comes from that sprint's `output/` tables or `report_sections/`.

Metric: r_spec is the prediction-truth correlation after the signal shared by all cell types is removed. Benchmark A v3 has 45 held-out biosamples, 283 tracks and 6 chromosomes. The seed gap averages 0.007.

1. **Recipe R2.** R2 adds RNA input and an equal-weight loss to R1. r_spec rose from 0.273 to 0.304, a gain of +0.031 [+0.022, +0.040] over two seeds. The loss is the larger ingredient. RNA helps once the loss counts the peaks (interaction +0.012 [+0.005, +0.019]). Source: `output/model/contrasts.tsv`, `interaction.tsv`.
2. **Noise floor.** The floor is 0.28 over all targets, not 0.38. On the 128 common targets it is 0.259 from ENCODE replicates, against 0.356 as sprint 06 reported. Sprint 06 used one replicate's depth, but the tracks pool both replicates. Our pseudoreplicate floors agree at the pooled depth for reads of 60 bp or longer, and overstate the noise by about 0.1 for 36 bp reads. Source: `output/floors/floors_headline.tsv`.
3. **Error budget.** R2's normalized error is 0.665: 0.110 calibration, 0.027 shared signal and 0.527 specific signal. Of the specific part, 0.282 is noise and 0.245 is not yet explained. R2 recovers 0.25 of the recoverable specific signal, against 0.21 for R1 and 0.07 for the training average. Source: `output/model/decomp_budget.tsv`.
4. **TF-expression vector through FiLM gave a null result.** Swapping the vector changed r_spec by +0.001 [−0.002, +0.003]. A constant vector (+0.011) and a shuffled vector (+0.024) reproduced the training gain, so the gain is regularization. Source: `output/trans/summary.md`, `contrasts.tsv`.
5. **Processing artifacts, on Mehdi's testbed.** A per-bin curve corrects the read-level settings to within 0.05 (depth is exact in counts). The peak-caller settings leave 3% to 41% of the variance unexplained. Source: `report_sections/04_covariates.md`.
6. **Variant effects on histone QTLs.** Spearman for the reference-panel model was 0.04–0.20. With reference dropout it was 0.26–0.28. The cell-state vector reached 0.38–0.41, and Borzoi 0.40–0.45. Source: `output/s2a/residual_scores.tsv`.
7. **Compute.** A 1g.10gb H100 slice trains a run in 10.4 h and matches CPU training (−0.005 [−0.011, +0.000]). Spend was about $335 of $2,000.

Asks for Mehdi (§12.5, §12.7): (1) the rival-comparison porting direction; (2) does paired-end duplicate marking keep a pair when either mate is unique? (3) do the single-end base and the paired-end arm come from the same library?
