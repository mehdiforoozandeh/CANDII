# BENCHMARK_DESIGN.md — the leaderboard's data regimes, panels, truths and metrics

**Status: live design doc, being written as it is decided.** Started 2026-08-29 from the
data audit in `cruxvault/results/t54/DATASET3_GAP.md` (artifact:
https://claude.ai/code/artifact/df80c2b6-e617-4e1e-ac0c-c14b7f2a5276). Supersedes the board
structure in `LEADERBOARD_PRD.md` §3–§4 and the protocol names in `RIVALS_PLAN.md` §2 wherever
the two disagree. Decisions below are PI-approved unless marked OPEN.

The goal this document serves: **a fair comparison between CANDI and every baseline and rival, on
the same data, with every number labelled by exactly how it was trained and scored.**

---

## 1. The address of a number

Every cell on the board is addressed by four fields plus the metric column. If any field is
unknown for a row, the row does not go in the ranked table.

| field | values |
|---|---|
| **method** | CANDI version · rival (`Avocado`, `ChromImpute`, `eDICE`, `Lavawizard`) · naive baseline (`marginal`, `avg`, `avg-arcsinh`, `knn1`, `knn5`) · anchor entrant (the 23 × 2019 submissions + `Average` + `Avocado_p0`) |
| **regime** | `eic.19→20,21,22` · `eic.pilot→20,21,22` · deferred: `eic.gw→20,21,22`, `merged.*` |
| **truth** | `store` (ENCODE4 2020) · `challenge` (2019 Synapse). `challenge` exists on EIC regimes only. |
| **panel** | `V_` (validation, 26 pairs) · `B_` (test, 12 cell types) |
| **scope** | `held-out` (chr20+21+22, the ranked number) · `genome-wide` (all 23, comparability only) |
| **metric** | arm (`count` / `pval` / `peak`) + key |

---

## 2. The two rules that define a regime

**Rule 1 — the data contract.** The scored track is never read by any method at any stage: not in
training, not in a correlation table, not in a σ-fit, not in per-position adaptation. This binds
every method identically and is the only fairness rule that is absolute.

**Rule 2 — the scope name.** A regime names the loci on which a method's *transferable*
parameters were fit. Per-position adaptation on the eval chromosomes — Avocado's genomic factors,
CANDI's encoder forward pass, ChromImpute's neighbour features — counts as **inference**, not
training, and is open to every method.

*Why Rule 2 is written this way, now confirmed from code.* Avocado's genomic factors are
per-chromosome `nn.Embedding` tables; `replace_genome` allocates a fresh table and re-initialises
it `uniform_(-0.05, 0.05)` (`competitors/avocado/vendor/avocado.py:90-97`), so on a chromosome it
never fit, Avocado returns its random init — not a degraded prediction, noise. Lavawizard raises
`KeyError` for an unknown chromosome (`dataset3.py:70-71`), and ChromImpute samples its 100,000
training locations inside the chromosomes it later predicts. Without Rule 2, three of five rivals
cannot enter a locus-holdout benchmark at all. Under Rule 1 the target track is never seen either
way.

---

## 3. The regimes

Two live, both evaluating on the same held-out chromosomes. Everything else is a placeholder.

| id | transferable parameters fit on | training bins | scored on |
|---|---|---|---|
| `eic.19→20,21,22` | chr19 | 2,344,704 (1.93 %) | chr20 + chr21 + chr22 |
| `eic.pilot→20,21,22` | the 44 ENCODE Pilot Regions, minus their overlap with the eval chromosomes | 1,022,368 (0.84 %) | chr20 + chr21 + chr22 |
| `eic.gw→20,21,22` | *deferred — placeholder* | | |
| `merged.19→…`, `merged.gw→…` | *deferred — placeholder* | | |

`eic.19` is primary; `eic.pilot` is the ablation. Same model, same eval, two training slices of
similar size, asking whether the choice of slice moves the result. Avocado's own supplement claims
it does not ("using the ENCODE Pilot Regions for the first step … does not lead to significantly
worse imputations than another subsample"), and nobody has checked that for a sequence-conditioned
model.

**Whole-genome training is deferred.** It is not needed to settle the current questions and the
compute is disproportionate. The placeholder stays so the axis is nameable later.

**The merged corpus** has no declared train/test split (D17 deferred it), the truth toggle (§6)
does not exist there, and the zero-shot question (§8) is parked against it.

### 3.1 The Pilot Regions, exactly

44 fixed regions from ENCODE's 2007 pilot phase: **29,955,196 bp** (~0.97 % of the genome) over 21
chromosomes — 14 hand-picked well-studied loci (`ENm*`, 14,955,138 bp) plus 30 randomly chosen
regions stratified by gene density and non-exonic conservation (`ENr*`, 15,000,058 bp). All 44 are
used, not the `ENr` half alone.

**4,395,988 bp of them sit on chr20/21/22 and are cut** under Rule 2, leaving 25,559,208 bp =
1,022,368 bins. Region sizes (≥500 kb for the `ENr` set) are far larger than CANDI's 768-bin
(19.2 kb) context, so window sampling is unaffected.

Two costs before this regime can run, both real work:

- **hg38 coordinates.** UCSC ships the `encodeRegions` track for **hg19 only** — no hg38 track
  exists, and the ENCODE portal has no annotation dataset for it. We own the liftover, and it
  needs its own provenance record and a check for regions that fail to lift cleanly.
- **A BED-restricted window sampler.** Regimes take chromosome lists today
  (`src/candi/store/regime.py`); training on scattered regions needs the loader to sample windows
  inside a BED. Avocado's per-position stage is unaffected — it already works per chromosome.

### 3.2 Avocado's joint fit moves to chr19

Our t50 retrain used chr20 for Avocado's shared-parameter stage. chr20 is now an eval chromosome,
and that stage fits transferable parameters, so it violates Rule 2. It moves to **chr19**, matching
CANDI's dev scope.

chr20 was never anyone's published recipe in the first place: the paper fits stage 1 on the ENCODE
Pilot Regions, and the challenge's `Avocado_p0` baseline used "the same model architecture and
training procedure described by Schreiber et al … applied as-is using the default settings and
hyperparameters." Under `eic.pilot`, Avocado's stage 1 returns to the Pilot Regions and becomes
faithful to the paper for the first time.

### 3.3 Consequence: every existing board row is void

The current "Full genome" board scored all 23 chromosomes, chr19 included — the chromosome CANDI
trained on. Those numbers are not portable to this design and are not carried forward.

---

## 4. Eval scope, and the two aggregations

**Scored scope: chr20 + chr21 + chr22.** 2,577,766 + 1,868,399 + 2,032,738 = **6,478,903 bins**,
5.34 % of the store's 121,241,684, split 40 / 29 / 31 % so no single chromosome owns the macro
average.

chrX was considered and dropped: at 6,241,635 bins it would have been 61.5 % of the panel, and
X-inactivation and copy number vary with the sex of each biosample, so a method that suits
female-derived lines could win the panel on that alone.

**Predictions are produced and scored genome-wide, once, and reported as two aggregations of the
same pass:**

| scope | what it is | what it is for |
|---|---|---|
| `held-out` | chr20+21+22 only | **the ranked number.** No method's transferable parameters were fit there, so nothing can be memorised |
| `genome-wide` | all 23 chromosomes | methodological comparability with a literature that scores this way, and 121 M scored bins instead of 6.5 M |

**The `genome-wide` cell is printed only where the method is out-of-sample somewhere** (APPROVED).
For a method whose transferable parameters were fit at every position — Avocado, ChromImpute and
Lavawizard, in both regimes — the number is a memorisation score, not an imputation score, and the
in-sample-fraction badge is not a strong enough warning to carry it. Those cells are **blank**, with
the reason on hover. Where the cell is printed it still carries the per-cell in-sample-fraction
badge: under `eic.19` CANDI is in-sample on 1 of 23 chromosomes, and that fraction is part of the
number's address. Without both rules the column becomes exactly the mislabelling this document
exists to delete.

**A blanked cell is not computed** (APPROVED 2026-08-29). Avocado, ChromImpute and Lavawizard are
predicted and scored on chr20+21+22 only, in both regimes. Their genome-wide number was never going
to be printed, so producing it buys nothing; it is not withheld evidence, it simply does not exist.
This is what takes the programme from 40 prediction runs to 30 and from 110 scoring passes to 95
(§12.3, §12.4). CANDI, eDICE and the naive baselines still run genome-wide, so the `genome-wide`
column still has entrants.

**Why both.** §11 records that the entire imputation lineage trains and evaluates at the same
positions, so a held-out-loci-only table has no methodological counterpart in the field; and a
genome-wide-only table carries the leak Schreiber's own pitfall paper warns about. Reporting both
from one pass costs one extra aggregation and no extra inference.

---

## 5. Panels — `V_` and `B_` are different sets, used differently

| | `V_` validation | `B_` test |
|---|---|---|
| composition | 26 `T_→V_` cell pairs — **45 experiments over 22 assays** | 12 `T_→B_` cell pairs — **51 experiments over 8 assays** |
| used during training | yes — monitoring, and **best-checkpoint selection** | never |
| used after training | scored on the best checkpoint → the reported validation result | scored once on the best checkpoint → the reported test result |
| on the board | a score slot | a score slot beside it, empty until the final run |

**Uniform selection.** Every trainable method selects its best checkpoint on `V_`, by the same
rule. Today they do not: Avocado picks on a 1-in-50 (position, track) entry mask
(`vendor/avocado.py:130-141`), and the others have their own conventions. This is a retrain for
every trainable method. The `V_` column is therefore optimistic by construction, identically for
everyone — which is exactly why `B_` exists.

**Methods with nothing to select are exempt, and say so** (APPROVED). ChromImpute and the kNN
baselines train once and produce one model; there is no checkpoint to pick and no knob tuned on
`V_`. They keep their `V_` score — it is a same-panel comparison and worth having — and the row
carries a **"no selection"** marker. The marker is not cosmetic: it says `V_` gave them none of the
optimistic lift every trainable method got, so their `V_`→`B_` behaviour is not comparable to
CANDI's without accounting for it.

**"Touched once" means predicted once** (APPROVED 2026-08-29). `B_` predictions are produced one
time, from the single checkpoint that method selected on `V_`, and *every* `B_` number on the
board — both aggregations of §4, the three-number split of §5.2, and both truths of §6 — is an
aggregation of that one prediction set. Re-scoring those stored predictions is free and allowed;
re-predicting `B_` is not, and needs a new checkpoint selection on `V_` to be legitimate.

**Rendering.** Side by side inside one regime row, never a toggle: a toggle invites treating them
as two views of one thing and erodes the touch-once discipline on `B_`.

### 5.1 Panel composition, measured (SETTLED)

Computed from `download_plan_eic.json`, the per-cell per-assay staging plan for all 89 biosamples:

| split | cells | experiments | distinct assays | assay mix |
|---|---|---|---|---|
| `T_` | 51 | 267 | 35 | DNase×34, H3K4me3×30, H3K27ac×22, H3K36me3×22, H3K4me1×18, H3K9me3×17, H3K27me3×16, … 28 more |
| `V_` | 26 | 45 | 22 | DNase×3, H4K20me1×3, H3K9ac×3, H3K9me3×3, H3K4me1×3, H3K36me3×3, H3K4me2×3, H3K27ac×3, H3K27me3×3, H3K79me2×3, H3K4me3×3, H2AFZ×2, **and 10 assays with n=1** |
| `B_` | 12 | 51 | 8 | H3K9me3×9, H3K27me3×9, H3K36me3×8, H3K27ac×7, H3K4me1×7, H3K4me3×5, ATAC×3, DNase×3 |

**The splits are disjoint on (cell, assay) — verified, zero overlaps.** No `T_X` shares an assay
with its `V_X` or `B_X` counterpart, for any of the 89 cells. This matters because
`StoreSource.targets` (`src/candi/bench/harness.py:540`) computes the panel as *assays the truth
cell has and the input cell does not*, while the challenge's blind set is a hand-declared list. The
disjointness makes those two definitions the same set by construction, so the computed panel is the
challenge's panel and no reconciliation is needed. Re-check this if the store is ever rebuilt.

The organizers chose `B_` to be the six core histone marks plus ATAC and DNase — the marks worth
imputing. `V_` instead reaches broadly across the assay panel, and **11 of its 22 assays hold a
single track**.

### 5.2 The three numbers (APPROVED)

`V_` and `B_` are different exams. The panel shape is a property of the dataset and applies
identically to CANDI, every rival and every baseline, so **ranking within a panel is fair and
unaffected**. What breaks is the `V_`→`B_` *delta* for one method — the comparison a reader makes
automatically when two columns sit side by side, and reads as a generalization gap.

So each trainable method reports **three** numbers per regime, not two:

| number | panel | job |
|---|---|---|
| `V_` (breadth) | all 45 experiments, 22 assays | ranked. The assay-breadth result. |
| `V_` (matched) | the 8 assays `B_` contains | **not ranked.** Exists only so the delta is readable. |
| `B_` | all 51 experiments, 8 assays | ranked. Touched once. |

Worked shape: `V_` 0.38 → `V_` matched 0.43 → `B_` 0.44 says the 0.38→0.43 step is the exam
changing and the 0.43→0.44 step is the real generalization gap. Without the middle number a reader
sees 0.38→0.44 and attributes all of it to the model.

Costs nothing: same predictions, a second aggregation over a subset of the same scored tracks.

**Also note** the per-assay macro is where the imbalance bites hardest — 11 singleton assays are
24 % of `V_` under per-track pooling but **50 %** of it under a per-assay macro. That is a
precision problem, not a fairness problem, and the noise floor (§15) has to be measured on the
`V_` breadth panel separately for that reason.

### 5.3 The ranking rule (APPROVED)

**One ranker everywhere: the challenge's own.** `src/candi/bench/ranking.py` implements it and is
the rule for every arm, every regime, and both truths — not only the 2019-truth anchor block.

1. Ten equally sized position bootstraps; all measures computed per team per bootstrap per experiment.
2. Within each (bootstrap, experiment) cell, scores → **ranks across methods** per measure, averaged
   over the measures.
3. A method's bootstrap score is `mean_e min(0.5, r_e)`, `r_e` = its rank ÷ number of methods.
4. Methods are ranked within each bootstrap; the **second-best** bootstrap rank decides the order.

Two known pathologies come with it and are **accepted, not fixed**: the `min(0.5, …)` cap makes
placing last on an experiment cost exactly what placing median costs, which favours an erratic
method over a consistent one; and step 4 is optimistic — it discards the eight worst bootstraps, so
it does not reward stability. These are not our design; they are what produced the published table,
and using anything else would make our order and the published order two incomparable quantities.

**A method missing an arm scores `MISSING_SCORE = 0.5`** — equal to the cap, i.e. treated as
below-median rather than disqualified. That is the challenge's own handling of an absent team and it
answers the "can a method missing an arm be ranked" question: yes, at the cap.

**Quote with every placement** (archive `h77`): a resolution limit of **~0.005 correlation units**,
and that **5 of 24** adjacent pairs invert on ≥3 of the ten chromosome subsets. A placement
separating two methods by less than that is not a placement. The order is reproducible; the scores
are not — 16 of 25 published ranks were held exactly on rescoring, with no method moving more than
two places.

**Reading rule.** Ranks are computed within a panel. Never subtract `V_` (breadth) from `B_` —
the matched number is the only legal subtraction, and it prints the panel caveat with it. **Rank
stability** between `V_` and `B_` is informative and is worth surfacing.

---

## 6. The truth toggle — same rows, second measurement

The 2019 challenge's 51 blind experiments **are** our `B_` test set, and its 45 validation
experiments are in our `V_` panel: the same ENCODE experiments, bridged 1:1 on accession
(`eic_bridge.csv`, 267 T + 45 V + 51 B = 363). The only difference is which pipeline produced the
truth track — the organizers' 2019 processing, or ENCODE4 v1.5.1 from Aug–Sep 2020.

So the challenge data is **not a separate board**. It is a switch on the page:

> **truth: store (ENCODE4 2020) · challenge (2019)**

**The story it tells.** If a method's standing holds under both truths, its result is not an
artifact of the pipeline that produced our truth. Nowhere else in this field does the same
experiment exist twice, processed independently by different people years apart. The target claim
is that CANDI is more robust to processing than the rivals are; we may not be able to show it yet,
and the column earns its place by being the only place that claim can be tested.

**One scorer, one grid, both truths.** For the toggle to measure the pipeline and nothing else,
only the truth may change. The Synapse bigwigs are therefore loaded into `candi.bench.external`
on the store's grid, with the store's blacklist policy and the same assay panel on both sides.
This view no longer reproduces the published 2019 numbers exactly, and that is accepted.

**Anchor entrants.** The 23 submissions plus `Average` and `Avocado_p0` are rescored through our
scorer on our grid, so they are readable beside our rows. They carry no regime — we never trained
them — so they are lifted out of the ranked table into a **separate anchor block underneath,
labelled as an anchor we did not run**. The vendored 001 scorer retires to a one-time recorded
proof that our path reproduces the published leaderboard; that proof lives in the vault, not on
the board.

**One extra figure: CANDI inside the 2019 field** (APPROVED 2026-08-29). Lifting the entrants out
of the ranked table means CANDI and the 2019 submissions never share a ranking denominator, so
"CANDI would have placed *N*th" cannot be read off the board. That number is worth having, so it is
computed once, separately: the challenge ranker over `B_` under challenge truth, with CANDI added to
the 25-entrant field. It is **a labelled figure, never a board row** — adding CANDI to the field
changes every entrant's rank denominator, which is the same reason §7 allows only one CANDI row. It
needs no new prediction and no new scoring pass, only one more run of the ranker over scores we
already have. It prints the non-independence count below with it.

**Known non-independence in the anchor block.** `CUImpute1`, `CUWA` and `ICU` submitted
byte-identical tracks for all 26 broad-mark experiments; `ICU`'s H3K4me1 *is* the organizers'
`Avocado_p0` baseline. The field has fewer distinct methods than rows — any "beats N methods"
claim must be counted, never read off the table.

---

## 7. Arms — who competes where

| arm | who appears | role |
|---|---|---|
| **pval** | everyone | **the head-to-head.** All rivals predict signal, so this is the only arm where CANDI meets the published field |
| **count** | CANDI + naive baselines | diagnostic: is CANDI's count modelling any good? No rival has a count head |
| **peak** | CANDI + naive baselines | diagnostic, same reason |

**eDICE carries a footnote marker** (APPROVED). It queries a `V_`/`B_` target using that target's
paired `T_` cell's learned embedding. Rule 1 permits it — the regime is a data contract, not a
parameter contract — but it is a different kind of access than the other methods get, so eDICE
stays in the ranked table with a symbol on the row that expands to exactly what the substitution
is. Comparable, and disclosed.

**One CANDI row per regime** (APPROVED). The board carries exactly one CANDI — the current best.
Version history lives in a separate figure, never as extra rows: the challenge ranker's step 2
ranks *across methods* within a cell, so every extra CANDI version changes the denominator and
shifts every rival's rank. CANDI-over-time is a real thing to show, and it is not a leaderboard.

Rivals are **not** carried into the count or peak arms. The AUPRC that Avocado, eDICE and
ChromImpute carry today is derived by ranking their predicted signal — a coverage ranking, not a
peak head — and it is dropped rather than badged.

**Under `truth: challenge`, the count and peak arms are greyed out.** The 2019 data has no counts
and no peak calls, so there is nothing to score against; scoring peaks against store truth while
the p-value arm uses challenge truth would put two truths in one row.

**The spread device, and its mandatory badge.** Point-only methods have no distribution, so their
prediction is wrapped in a Gaussian whose width σ is measured. **σ is fit on training-set
residuals only** — never on `V_`, never on `B_`. In-sample training residuals are accepted: they
run narrow, and the resulting overconfidence is the method's own calibration failure and should
show up as one. Every distributional cell carries a badge: `native heteroscedastic` versus
`fitted flat σ`. A flat bell cannot track a varying truth, so point-only methods lose on PIT and
coverage partly by construction, and the badge is what stops that being read as a modelling
result. (Scopes `t69`.)

Note the two σs are different things and must not be confused: the **marginal** σ is the spread of
the assay's signal itself and is the `marginal` naive baseline; the **residual** σ is the spread
of prediction-minus-truth and is what wraps a point prediction.

---

## 8. What the board does not test

Every eval pair is a mark the `V_`/`B_` cell has and its paired `T_` cell lacks, so the model
always sees other tracks from the same cell type. The board measures **missing-mark transfer
within a seen cell type**.

CANDI's zero-shot claim — imputation for a cell type never seen in training — is **not tested
here, and is deferred to the merged corpus**. Most rivals cannot enter such a test at all:
Avocado and eDICE need a learned embedding for the target cell, which by definition does not
exist. The zero-shot field will be CANDI, ChromImpute and the naive baselines.

---

## 9. Naming

Retired, because both now name things that changed meaning:

- **P1 / P2 / P3** encoded eval scope. Eval scope is now one thing for every regime, so P2 is
  false and P3 was never a protocol — it is a truth. Replaced by the regime ids of §3 plus the
  `scope` field of §4.
- **Dataset 2 / Dataset 3** are opaque codes that conflate the corpus with the pipeline.
  Replaced by `truth: store (ENCODE4 2020)` and `truth: challenge (2019)`.
- Board ids **`main` / `dev` / `entrants`** — `main` and `dev` become the two regime ids;
  `entrants` stops being a board and becomes the anchor block under challenge truth.

---

## 10. The DNase units defect, and the fix (APPROVED)

**What was found.** ENCODE ships no p-value track for DNase-seq. Verified per file against the
portal for all 363 EIC signal bigwigs: **316 ChIP-seq → `signal p-value`, 7 ATAC-seq → `signal
p-value`, 40 DNase-seq → `read-depth normalized signal`**, all released, bigWig, GRCh38, zero
anomalies. The old downloader (`EpiDenoise/data_utils.py:1432`) accepted exactly those two output
types and never `fold change over control`, and no experiment offers both — so the defect is
DNase-only and total, and no histone or ATAC track is contaminated.

Consequences that were live and unrecorded: `DATA.md` labels `signal_BW_res25/` as "−log10
p-value track", which is false for 40 of the panel's tracks; every method trained on the store's
p-value layer trained on mixed units; and any macro that pools assays pools a probability with a
scaled read count. 3 of the 45 scored `V_` targets are DNase.

**ATAC is the template, not ChIP.** ATAC-seq has no control experiment (`possible_controls` is
empty, like DNase; ChIP has one), and its `signal p-value` is produced by MACS2 v2.1.0 from
alignments alone in the Kundaje-lab ATAC signal step. ENCODE already computes exactly the
no-control p-value DNase needs — they simply never ran it on DNase.

**The fix, in four phases.**

**Phase 0 — stamp provenance.** Write `signal_bigwig_accession` and its `output_type` per track
into the store manifest, from `EpiDenoise/data/download_plan_eic.json` (which already carries
`signal_bigwig_accession` for all 363) joined with the portal sweep. Add a build assertion: a
track whose recorded `output_type` disagrees with the corpus's declared signal semantics fails the
build. This closes a real gap — the bigWig accession is recorded in no artifact that travels with
the data; `file_metadata.json` holds only the BAM and `navigation.json` only paths.

**Phase 1 — validate the method on ATAC (the gate).** Implement `pval_from_counts`: a
MACS2-equivalent no-control Poisson p-value at 25 bp from the store's DSF-1 counts, background =
max of genome-wide λ and local λ at MACS2's `slocal=1000` / `llocal=10000` defaults, with the
exact λ rule read off the MACS2 source rather than from memory. Run it on the 7 ATAC experiments
and compare against ENCODE's own ATAC p-value, already held binned in `signal_BW_res25`. Report
per-experiment Pearson, Spearman, MSE ratio, top-1 %-bin agreement. Pre-registered defaults, no
tuning.

**The pass/fail, pre-registered 2026-08-29 before any ATAC result was computed (APPROVED).** Over
the 7 ATAC experiments, all four must hold:

| statistic | bar |
|---|---|
| Pearson vs ENCODE's ATAC p-value | **median ≥ 0.90** |
| top-1 %-bin Jaccard | **median ≥ 0.60** |
| MSE ratio (ours / ENCODE's) | **median in [0.5, 2.0]** |
| worst single experiment, Pearson | **≥ 0.80** |

Anything short of all four is a **fail**, and a fail means the §10 fallback: re-download the 40
DNase BAMs to scratch and run MACS2 at base resolution. The bar is not moved after the numbers are
seen — that is the whole point of writing it here, in a commit that predates the run.

**Phase 2 — build the DNase layer.** Same code, 40 experiments, genome-wide. Second independent
check against the challenge's DNase p-value tracks, on `T_` experiments only so `V_`/`B_` stay
untouched under Rule 1. One shot, no tuning — tuning to match would make the truth toggle circular
for DNase and permanently exclude accessibility from the robustness story.

**Phase 3 — store layout, nothing deleted.** The read-depth normalized signal moves to its own
archived kind (`signal_rdns`) and stays queryable. The `pval` kind for DNase becomes the computed
layer, and only after Phases 1 and 2 pass. Each DNase track records both the archived source
accession and the derived layer's method, parameters and code sha.

**Phase 4 — the consequence.** Methods train jointly across assays, so a changed DNase target is a
full retrain, not a DNase-only patch.

**The risk that decides it.** Our counts are pre-binned at 25 bp with no read shift or extension,
while MACS2 for ATAC uses `--shift -75 --extsize 150` at base resolution. Phase 1 measures exactly
that gap. If ATAC agreement is poor, the fallback is re-downloading the 40 DNase BAMs to scratch
and running MACS2 properly — expensive, and worth knowing before committing.

---

## 11. What the field actually does — and why our departure is defensible

Audited 2026-08-29 across `cruxvault/raw/` primary sources and every rival implementation
available to us.

**Nobody in the imputation lineage holds out loci.** ChromImpute, PREDICTD, Avocado, eDICE, EpiMap
and the ENCODE Imputation Challenge itself all train and evaluate at the same genomic positions;
the held-out unit is always the cell-type × assay *experiment*. Schreiber states it plainly:
"Avocado is trained on the same genomic loci that it makes predictions for, whereas the challenge
participants had to make predictions for held-out chromosomes"
(`raw/schreiber-2020-encode3-compendium.xml`). Restricted scopes do exist — Pilot Regions, chr20,
chr12–22, chr21 — but they are memory or compute compromises, not leakage controls: "Avocado does
not fit a single model to the full genome because the genome latent factors could not fit in
memory."

**Transferable parameters are trained on 0.01–1 % of the genome, everywhere.**

| method | training loci for the transferable part | fraction |
|---|---|---|
| PREDICTD | cell-type/assay factors, "a randomly selected 0.01 % of the genome" | 0.01 % |
| ChromImpute | 100,000 randomly sampled 25 bp positions per predictor | 0.08 % |
| eDICE (upstream) | Roadmap chr21 only, 997,373 bins | 0.80 % |
| Avocado (paper) | ENCODE Pilot Regions, 44 regions / 29.96 Mb | 0.97 % |
| PREDICTD | genome factors, Pilot Regions + 2,640 ncHARs, 1,309,125 bins | 1.06 % |
| Avocado (our t50) | chr20 joint fit | 2.1 % |
| eDICE (our retrain) | chr19 | 1.9 % |
| Lavawizard | one model per chromosome, every bin — no cross-chromosome transfer | n/a |

So `eic.19` at 1.93 % and `eic.pilot` at 0.84 % are **squarely inside the field's convention**, and
whole-genome training would have been ~100× more than any published imputation method gives its
learned parameters — a second reason it is deferred rather than a compromise.

**Chromosome holdout is standard one door over.** Sei, GET, ChromBPNet, dHICA, EpiVerse, Coda and
AtacWorks all withhold named chromosomes; Enformer, Borzoi, AlphaGenome, EpiBERT and Corgi
withhold disjoint interval partitions.

**And the imputation convention has a named critic — the same group.** Schreiber's cross-cell-type
pitfall paper argues that "when the training and test sets contain the same genomic loci, the
resulting model may falsely appear to perform well by effectively memorizing the average activity
associated with each locus", and proposes exactly the hybrid cross-chromosome / cross-cell-type
split this document specifies. That is the citation the `held-out` scope is justified against; the
`genome-wide` scope of §4 is what keeps us legible to everyone who did it the old way.

---

## 12. Compute

The critical path is **G1 → retrain → predict → score**.

**The widened eval scope adds no training cost.** Regimes fix training scope, and the
position-parameterised rivals already fit all 23 chromosomes — a genome-wide scoring scope declines
a saving rather than adding a cost. What forces the retrain is §13, not §4.

Summary of the whole programme: **3 gates, 10 training runs + 5 σ refits, 30 prediction runs
(≈434 GB scratch), 95 scoring passes (≈4,360 CPU-h)**, plus Avocado's ≈40 GPU-h and CANDI's
unmeasured GPU cost. Detail below.

Two rulings of 2026-08-29 cut this from the 40 / 880 GB / 110 / 6,130 CPU-h first estimate: the
naive baselines collapse to one prediction and one score each (§12.2), and a method whose
`genome-wide` cell is blank is no longer predicted there (§4).

### 12.1 The gates — nothing trains until these pass

| # | work | cost | blocks |
|---|---|---|---|
| **G1** | **`t78` — the DNase p-value layer.** Phase 1 validates `pval_from_counts` against ENCODE's own ATAC p-value on the 7 ATAC experiments; Phase 2 builds the 40 DNase tracks genome-wide (§10) | CPU-only, hours. **Fallback risk:** poor ATAC agreement forces re-downloading 40 DNase BAMs to scratch and running MACS2 at base resolution | **every training run.** 34 of the 267 training tracks change units |
| **G2** | **`t79` — Pilot Regions to hg38.** UCSC ships `encodeRegions` in hg19 only. Lift the 44 regions, then write the BED-restricted window sampler (§3.1) | small, one-off | the `eic.pilot` regime only |
| **G3** | **CANDI's genome-wide inference timing.** No measurement exists. The only number on record is the store loader at 404 windows/s, which is a loader rate and not an inference rate | one chromosome, ≈1 GPU-h | the GPU budget for everything below. **Do this first** — it is the only measurement that could change the shape of this plan |

G1 is the real gate. Skipping it trains every method on 34 tracks of the wrong units.

### 12.2 Training runs — 10, plus 5 σ refits

| method | runs | why that count | arms |
|---|---|---|---|
| CANDI | 2 | `eic.19` and `eic.pilot` | pval + count + peak |
| Avocado | 2 | joint fit on chr19 / pilot, then 23 per-chromosome genome-factor fits each | pval |
| ChromImpute | 2 | one per regime | pval |
| eDICE | 2 | one per regime | pval |
| Lavawizard | 2 | one per regime | pval |
| `avg`, `avg-arcsinh`, `marginal`, `knn1`, `knn5` | **1 each** | no fitted position parameters, so the regime's training loci do not enter them — one fit serves both regimes | pval + count + peak |

**The naive baselines collapse to one of everything** (APPROVED 2026-08-29, closing costing 1 of
the old §12.5). Because their fit is regime-independent, so is their output: the two regimes would
produce byte-identical predictions. So each is fit once, predicted once and scored once, and the
one number is printed in both regime rows. First implementation of `avg` asserts this — it predicts
under both regimes once and checks the two outputs are identical — and the assertion, not the
argument, is what licenses the collapse for the other four.

Reference budgets on record: Avocado ≈20 GPU-h per model, ChromImpute 128 CPU-core-hours
genome-wide. CANDI's is G3.

**σ-tables refit for all 10 methods**, on training residuals (§7). Every existing σ was fit on `V_`
eval pairs and is void under Rule 1. Cheap, but nothing distributional scores until it is done.

### 12.3 Prediction runs — 30, once each

There are **15 method-regime units**, not 20: the 5 naive baselines contribute one unit each
instead of two (§12.2). Nine of the 15 predict genome-wide; six predict on chr20+21+22 only,
because their `genome-wide` cell is blank (§4).

| unit | count | scope | `V_` (45 tracks) | `B_` (51 tracks) |
|---|---|---|---|---|
| CANDI, eDICE — 2 methods × 2 regimes | 4 | genome-wide | 21.8 GB | 24.7 GB |
| `avg`, `avg-arcsinh`, `marginal`, `knn1`, `knn5` — 1 each | 5 | genome-wide | 21.8 GB | 24.7 GB |
| Avocado, ChromImpute, Lavawizard — 3 × 2 regimes | 6 | chr20+21+22 | 1.17 GB | 1.32 GB |
| | **15 units → 30 runs** | | **≈419 GB + 15 GB = ≈434 GB** | |

485 MB per track genome-wide (121,241,684 bins × 4 bytes); 25.9 MB per track on the 6,478,903
held-out bins. `B_` is **touched once**, at the very end. Scratch, never `/project`; deletable
after scoring.

### 12.4 Scoring runs — ≈6,100 CPU-h

One pass = 45 tracks genome-wide ≈ **50 CPU-h** on 4 cores, projected from a measured 5 min 09 s
for 20 chr21 tracks (`cruxvault/results/t51/PILOT_MEMO.md`).

| pass | count | CPU-h |
|---|---|---|
| `V_`, store truth | 15 | ≈466 |
| `B_`, store truth | 15 | ≈528 |
| `B_`, challenge truth — p-value arm only | 15 | ≈528 |
| the 25 anchor entrants, both truths | 50 | ≈2,835 |
| | **95** | **≈4,360** |

Each of the three 15-pass rows is 9 genome-wide passes (≈50 CPU-h on `V_`, ≈57 on `B_`) plus 6
held-out-only passes at 5.34 % of that (≈2.7 and ≈3.0), matching the 15 units of §12.3. The
CANDI-inside-the-2019-field figure (§6) adds **no** pass — it is one more run of the ranker over
the `B_` challenge-truth scores already counted here.

All CPU. Fir's CPU allocation is not the scarce resource. The two aggregations (`held-out`,
`genome-wide`) and the three `V_` numbers (§5.2) all come out of the same pass — no extra
inference.

**Check before assuming:** the 23 entrant bigwigs live on scratch, which purges at 60 days.

### 12.5 One costing left open

1. ~~**Do the naive baselines need one fit or two?**~~ **SETTLED 2026-08-29: one** — one fit, one
   prediction, one score, printed in both regime rows, with the identity assertion of §12.2.
2. **Does the truth toggle apply to `V_`, or only to `B_`?** The challenge staged 45–46 round-1
   validation tracks that were never scored, so `V_` under challenge truth is possible. It is
   **excluded** from the §12.4 table. Adding it is +20 passes, ≈1,000 CPU-h.

### 12.6 Storage, in full

485 MB per track genome-wide (121,241,684 bins × 4 bytes) against ≈1.2 GB per method-regime for the
three-chromosome subset — a 22 GB / 1.2 GB choice per `V_` method-regime, **now decided per method
by §4**: genome-wide for the nine units whose `genome-wide` cell is printed, three chromosomes for
the six whose cell is blank. Everything lives on scratch, never `/project`, and is deletable once
scored. **Scratch purges at 60 days**, which also
applies to the 23 entrant bigwigs already staged there.

---

## 13. What has to be redone

Accepted as the price of the corrections above.

1. **Every trainable method retrains** — CANDI, Avocado, ChromImpute, eDICE, Lavawizard, and the
   fitted baselines. Four independent reasons, any one of which alone would force it: the DNase
   target changes (§10), checkpoint selection moves to `V_` (§5), the eval chromosomes change to
   chr20+21+22 (§4), and Avocado's joint fit moves off chr20 (§3.2).
2. **Every `V_` row rescores** genome-wide, through one scorer, emitting both aggregations.
3. **`B_` is not touched** until each method's final, once-only run.
4. **The 23 entrant submissions rescore** through our scorer on our grid. Their bigwigs are on
   scratch, which purges at 60 days — check before assuming they are still there.
5. **σ-tables refit** on training residuals; every existing σ was fit on `V_` eval pairs and is
   void under Rule 1.
6. **`eic.pilot` is new work**, not a rerun: the hg38 liftover and the BED-restricted sampler
   (§3.1) must land before that regime can train anything.

---

## 14. Evidence behind this document

| claim | where it was checked |
|---|---|
| 363 experiments, 267 T / 45 V / 51 B, bridged 1:1 on ENCODE accession | `competitors/entrants/README.md` §3; 001 |
| the challenge scored genome-wide, ten bootstrap groups of 18–21 chromosomes; `score.py --chrom` defaults to `all` | the challenge repo's own `README.md` and `score.py:112-113`, pinned at `181b8023` |
| store vs challenge ChIP p-value: median Pearson 0.959 train / 0.852 blind; MSE ratio 1.005 / 1.435 | 001 Result 3, via `cruxvault/results/t46/max_notebooks_brief.md` |
| no ENCODE file reproduces Dataset 3 — 0 md5 matches over 363; earliest blind bigwig 2020-08-24 vs challenge close 2019-08-14 | 001 Result 7 |
| rescaling a store score into challenge space leaves 12–66 % per-experiment error | 005 Result 4 |
| training on store truth and scoring on challenge truth: 0.87–0.99 on punctate marks, 1.70 (H3K36me3) and 2.08 (H3K27me3) on broad | 005 Result 2/5 |
| DNase output types, all 40 experiments; ATAC and ChIP for contrast | ENCODE portal, per-file sweep of all 363 signal bigwigs, 2026-08-29 |
| ATAC p-value = MACS2 v2.1.0, alignments only, no control | ENCODE portal analysis step `kundaje-lab-atac-seq-signals-single-rep-step-v-1` |
| Pilot Regions: 44 regions, 29,955,196 bp, 21 chromosomes, 14 `ENm` / 30 `ENr`, hg19 only | UCSC `encodeRegions` track via `api.genome.ucsc.edu`, 2026-08-29 |
| Avocado returns its random init on an unfitted chromosome; Lavawizard raises | `competitors/avocado/vendor/avocado.py:90-97`; `competitors/lavawizard/dataset3.py:70-71` |
| per-method training loci and genome fractions across the literature | primary sources in `cruxvault/raw/`, audited 2026-08-29 |
| `V_`/`B_` panel composition — 45 exp / 22 assays vs 51 exp / 8 assays; splits disjoint on (cell, assay), 0 overlaps over 89 cells | `download_plan_eic.json` (Fir `EpiDenoise/data/`), recounted 2026-08-29 |
| the scored panel is *assays the truth cell has and the input cell does not* | `src/candi/bench/harness.py:526-543`; pairing declared by `tools/declare_eval_pairs.py` |
| eval-scope bin counts, scoring cost and storage | `cruxvault/results/t50/scores_avocado_P2.json` provenance; `cruxvault/results/t51/PILOT_MEMO.md` |

---

## 15. Status

Every design question raised in this pingpong is settled. What remains is execution.

### Deferred by ruling

- **The noise floor on the new eval panels** (2026-08-29). Two seeds of one method, same data;
  the spread between them is the resolution limit printed beside every rank. Must be measured
  separately for the `V_` breadth panel (22 assays, 11 of them singletons) and for the 8-assay
  panels, because they do not have the same resolution. Nothing is ranked with a resolution band
  until it is done; rows may go up unranked before then. `AGENTS.md` §7.2 records that a seed
  change alone moves pooled CRPS by 0.1195, so this is not a formality.
- **Whole-genome training** — a placeholder regime only (§3).
- **The `merged` corpus regimes** — placeholders. The zero-shot claim is tested there later, and
  only ChromImpute and the naive baselines can stand beside CANDI on it.

### Where this document lives

`plan/BENCHMARK_DESIGN.md`, tracked and committed, beside `LEADERBOARD_PRD.md` and
`RIVALS_PLAN.md` — which is where a reader looking for design prose will look.

### The work, as tasks

Execution order and every cost estimate are in §12. The critical path is
**G1 (`t78`) → retrain → predict → score**, and **G3 (CANDI's genome-wide inference timing) should
run first** because it is the only measurement that could change the plan's shape.


**`t77`** is the parent — *redesign the leaderboard's data regimes, panels and ranking so every
number has one address*. Its five children land on `t77`'s single branch. Per `CLAUDE.md`, a child
that wants its own PR was never a child.

| child | what it does |
|---|---|
| `t78` DNase p-value rebuild | recompute the `−log10 p` layer for all 40 DNase experiments from the BAMs on the ATAC template (§10); archive the old layer, do not delete it |
| `t79` regime rewrite | `eic.19→20,21,22` and `eic.pilot→20,21,22` as the two live regimes; Pilot Regions liftover to hg38; Avocado's joint fit moved to chr19 (§3) |
| `t80` eval-stack changes | the three-number `V_` aggregation (§5.2); the `held-out` / `genome-wide` split with the blanking rule (§4); the challenge ranker wired as the single ranker (§5.3) |
| `t81` retrains | every trainable method re-selected on `V_` under the uniform rule; every existing board row is void (§3.3) |
| `t82` board changes | apply the §9 naming across `boards.json`, `help.json`, `app.js`, `index.html` and the PRD; the anchor block; the truth toggle; the eDICE and "no selection" markers |

The §9 naming supersedes the earlier "ENCODE DCC MERGED / DEV-SET / EIC" rename request — the
regime names carry the training and eval loci, which those three labels did not.
