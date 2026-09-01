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
it `uniform_(-0.05, 0.05)` (`rivals/avocado/avocado.py:90-97`, vendored into this repo 2026-08-30), so on a chromosome it
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
| `eic.pilot→20,21,22` | the 44 ENCODE Pilot Regions, minus their overlap with the eval chromosomes | 1,023,489 (0.84 %) | chr20 + chr21 + chr22 |
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

Those are **hg19** figures, and the store is hg38. The liftover is done (below), so the numbers
that actually govern this regime are the hg38 ones:

| | hg19 (the source track) | hg38 (what the regime uses) |
|---|---|---|
| all 44 regions | 29,955,196 bp | 29,984,074 bp |
| cut on chr20/21/22 under Rule 2 | 4,395,988 bp | **4,395,877 bp** |
| training scope | 25,559,208 bp | **25,588,197 bp** |
| training bins at 25 bp | 1,022,368 | **1,023,489** |

The shift is +0.11 % of the training scope and changes nothing about the design. **Note the bin
count is now a containment rule, not a division:** 25,588,197 is not divisible by 25, so the
training bins are the 25 bp bins lying wholly inside a region, counted, not a quotient.

Region sizes (≥500 kb for the `ENr` set) are far larger than CANDI's 768-bin (19.2 kb) context, so
window sampling is unaffected — measured, not assumed: the sampler plans **1,294 fully-contained
768-bin windows = 993,792 bins = 97.10 %** of the contained budget. (This paragraph previously
quoted "1,328 = 99.6 %", which is a different quantity — the *region-anchored packing capacity*, how
many windows fit if each region's tiling restarts at its own first bin. D32 keeps D12's
`eligible_starts` and filters afterwards, so the tiling is anchored at chromosome bin 0 and 34 of
the 40 training regions lose a leading partial tile. **Ruled 2026-08-29: keep 1,294 and correct the
number.** Anchoring per region would buy 2.5 % more training windows at the price of `eic.pilot` no
longer sharing a window grid with every other regime — and 2.5 % is noise beside the 0.1195 that a
seed change alone moves. Both figures are pinned in
`test_the_pilot_regime_plans_its_training_windows` with the reason.)

Two costs before this regime can run:

- ~~**hg38 coordinates.**~~ **DONE 2026-08-29.** UCSC ships `encodeRegions` for hg19 only
  (`goldenPath/hg38/…/encodeRegions.txt.gz` returns 404; the hg19 URL returns 200), so we own the
  lift. UCSC `liftOver`, Kent tools 486, `hg19ToHg38.over.chain`, default options: **44 in, 44 out,
  0 unmapped, 0 split, 0 chromosome changes.** 41 of 44 regions move by ≤2 bp. Three do not:
  `ENm006` (chrX) grows +37,180 bp of real hg38 sequence, `ENm007` (chr19) loses 8,201 bp — the
  only region touching an ALT contig, `chr19_KI270938v1_alt`, which the store excludes by design —
  and `ENr333` (chr20) loses 119 bp and is cut anyway. Ensembl's independent GRCh37→GRCh38 mapper
  agrees on 16 of 16 endpoints across 8 regions and reproduces both the `ENm006` growth and the
  `ENm007` shrink, so those are assembly facts and not chain artefacts. Files:
  `configs/regions/encode_pilot_hg38.bed`, its hg19 source, and `configs/regions/PROVENANCE.md`.
  The BED keeps all 44 regions; the Rule 2 cut is declared by the regime's chromosome list, not
  baked into the file.
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

> **SETTLED 2026-08-29 — the code is right, this paragraph is the bug; but it exposed a real
> problem, below.** The store path scores **imputation**, not denoising. There is no declared pair
> direction on it at all: no shipped config carries `eval_pairs`, `tools/declare_eval_pairs.py`
> (cited in §14) does not exist, and `StoreSource.pairs` at `harness.py:483-484` is a hardcoded
> `[Pair(b, b) for b in self.ds.biosample_pool]` — **self-paired on the eval cell**. Because a `V_`
> or `B_` cell holds *only* blind assays, "the assays the input has" is the blind panel, so the two
> rules coincide. Checked on the real store: pair `V_DND-41→V_DND-41` returns `H3K27ac, H4K20me1`,
> exactly what `V_DND-41` holds, sharing nothing with `T_DND-41`'s nine. The target is masked per
> forward pass (`stream_tracks:653-657` applies `_apply_loo_mask`; `decode_groups` forces one assay
> per pass), so `targets` ignoring `kind` is correct — masking separates impute from denoise, not
> the panel. The rule sentence above describes `H5Source.targets:316-322`; the store's rule is
> "the eval biosample is the truth cell, and the panel is every assay it holds."
>
> **The real problem it exposed — FIXED 2026-08-29.** Self-pairing meant CANDI's prompt was **the
> eval cell's other blind assays**, never the paired `T_` cell's tracks. Two consequences, both
> verified from `manifest.json` with no project code: **16 of the 26 `V_` cells hold exactly one
> blind assay** (sizes `{1:16, 2:5, 3:1, 4:4}`), so leave-one-out emptied the encoder input and the
> track was dropped unscored — the `V_` panel could pose only **29** of the 45 experiments §5 and
> §5.2 promise; and every rival, and the 2019 challenge, prompt a blind experiment from *everything
> else known about that cell type*, so CANDI was sitting a handicapped and simply different exam.
>
> `StoreSource` now takes its pairs from the regime's **declared** `eval_pairs` (D31), input = the
> `T_` prompt cell, target = the truth cell, and `targets` reads the *target* — the same rule
> `H5Source` uses, so the two backends finally mean the same thing by a pair. `tools/declare_eval_pairs.py`
> now exists and writes the declaration; D16 forced its shape, so the `T_X`→`V_X` string surgery is
> an explicit **argument** and never a default, and asked to guess it refuses by name. Confirmed
> three ways on the real store: `V_` **45**, `B_` **51**, zero prompt-holds-target leaks, zero pair
> overlaps. Gates: 838 tests pass; `golden.py` 0 ULP against the pre-change tree, checked from a
> clean worktree.
>
> **Two things it leaves behind.** Keeping `_apply_loo_mask` means a target column already `MISSING`
> in the `T_` prompt becomes `CLOZE`, where the h5 path leaves it `MISSING` — `CLOZE` is what the
> masker writes at training time, so it matches training, but it is a real difference in the model's
> input for the same experiment and it makes h5 and store numbers non-comparable. And **no shipped
> regime config declares `eval_pairs` yet**, so every config still runs the self-paired exam —
> now loudly: `StoreSource` announces it at construction and `provenance()` records
> `"self_paired": true`, so a scored JSON says which exam produced it.

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
max of genome-wide λ and local λ, with the exact λ rule read off the MACS2 source rather than from
memory.

> **CORRECTION 2026-08-29, from the source.** This paragraph used to say "`slocal=1000` /
> `llocal=10000`". **Without a control, MACS2 uses `llocal` alone** — not `slocal`, and not the
> d-size window either. Read at tag `2.1.0.20140616`,
> `MACS2/cPeakDetect.pyx::PeakDetect.__call_peaks_wo_control`: `ctrl_d_s = [ self.lregion, ]`,
> with the comment "slocal and d-size local bias are not calculated!" directly above, and
> `--slocal`'s own help text reads "Invalid if there is no control data". The score is
> `-log10 P(X > k)`, a strictly-greater tail. Also: ENCODE's ATAC step ran MACS2 **2.1.1**, not the
> 2.1.0 named below — the no-control λ block is byte-identical between the two tags, checked by
> diff.

**`MSE ratio` here means `mean(ours²) / mean(theirs²)`**, a scale comparison — which is what the
symmetric factor-of-two band below is written for. It is *not* the repo's other "MSE ratio" (two
methods' error against a shared truth, as in §14); here there is no third truth, only our layer
against ENCODE's. Run it on the 7 ATAC experiments
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

**RESULT 2026-08-29 — PASS on all four**, genome-wide over all 7 ATAC experiments.

| statistic | bar | measured | |
|---|---|---|---|
| median Pearson | ≥ 0.90 | **0.9532** | pass |
| median top-1 % Jaccard | ≥ 0.60 | **0.6996** | pass |
| median MSE ratio | [0.5, 2.0] | **0.5078** | pass |
| worst single Pearson | ≥ 0.80 | **0.9085** | pass |

**The pass does not depend on which panel is used, and that is the check that matters.** Three of
the seven rows compare a **one-replicate** store BAM against a **three-replicate** ENCODE bigWig,
so ENCODE's values there are ~3× larger by depth alone. That depth gap is the whole of their
0.05–0.07 MSE ratios, and it is what drags the 7-experiment median down to 0.5078 — a mere 0.0078
above the floor. On the four like-for-like `T_` rows, where the bigWig derives from exactly the BAM
the store's counts were built from, the same four statistics read **0.9497 / 0.6075 / 0.8260 /
0.9325**: pass, with the marginal statistic no longer marginal.

**The three depth-mismatched rows stay in, unexcluded and unreweighted.** The bar was pre-registered
over all 7, and dropping rows after seeing them is precisely what pre-registration exists to forbid.
The `T_`-only figures are a disclosed sensitivity beside the result, never a substitute for it.

**Phase 2 is therefore authorised.** Phase 0 is already built: 363/363 tracks now carry
`signal_bigwig_accession` and `signal_output_type`, and a strict build raises
`SignalSemanticsConflict` on exactly 40 tracks — all DNase-seq, all `read-depth normalized signal`,
reproducing this section's audit precisely.

**Phase 2 — build the DNase layer.** Same code, 40 experiments, genome-wide. Second independent
check against the challenge's DNase p-value tracks, on `T_` experiments only so `V_`/`B_` stay
untouched under Rule 1. One shot, no tuning — tuning to match would make the truth toggle circular
for DNase and permanently exclude accessibility from the robustness story.

**PHASE 2 RESULT 2026-08-29 — the units defect is fixed; two experiments are not.** All 40 DNase
tracks built genome-wide with the gate-passing code unchanged, in the store's own uint16
arcsinh×2000 codec, `n_clipped = 0` on all 40, on scratch. The second check ran against the
challenge's 2019 DNase p-value on the **34 `T_` experiments only** — verified: the
`ours_vs_challenge` comparison exists on exactly those 34 rows and on no `V_` or `B_` row, so
Rule 1 held.

| median over the 34 `T_` | our computed layer | the store's RDNS today |
|---|---|---|
| Pearson | **0.8344** | 0.5406 |
| Spearman | 0.5647 | 0.5441 |
| MSE ratio | 1.6385 | 0.0004 |
| top-1 % Jaccard | 0.5649 | **0.6273** |
| mean ratio | **1.2454** | 0.0624 |

**What is decisively fixed is the scale** — mean ratio 0.0624 → 1.2454 — which is the defect §10
exists to correct. And the improvement is not only at the median: **our layer beats the RDNS track
it replaces on 26 of the 34**, with Pearson below 0.5 on **4** experiments against RDNS's **14**.

**What is not fixed is the ranking.** Spearman barely moves and top-1 % Jaccard gets *worse* than
the track it replaces. Local background changes which bins rank highest, so some of this is a real
methodological difference from the 2019 pipeline rather than an error.

**The honest ceiling.** There is no ENCODE DNase p-value to compare against — that absence is the
defect. So the same code was run on the 4 `T_` **ATAC** experiments against the *challenge's* ATAC
tracks: ours scores 0.7055 where ENCODE's own p-value scores **0.7833**, and ENCODE beats us on all
four. So a 2020 MACS2 run reaches only ~0.78 against a 2019 target, and much of the residual DNase
disagreement is pipeline vintage rather than our method — but our ATAC layer also sits ~0.08 below
ENCODE's own against that same target, which is the fairer read than quoting 0.8344 against 0.7833
across two different assays.

**DIAGNOSED 2026-08-29 — not a bug. The input is contaminated, and the ATAC gate could not have
caught it.** `pval_from_counts` is correct: the 1.8 × 10⁷ is the exact Poisson tail of the number
the store handed it, re-derived from `gammaln` with no `scipy` at 18,245,363.906 against the layer's
18,245,364.0. Two adjacent bins at `chr11:24,159,500–24,159,550` hold **9,762,321 and 9,762,619
counts** — a fifth of chr11's entire pileup in 50 bp, against a next-highest bin of 13,114. Reading
the BAM (`ENCFF257HEE`): 9,762,609 of those reads are **already flagged duplicate by Picard**, 13
are not, and 9,733,414 carry the identical 20-mer `AAGCAGAAGACGGCATACGA`, whose reverse complement
is an exact substring of the Illumina TruSeq adapter tail. It is 9.73 M untrimmed adapter
read-throughs parked on their best 20 bp match in hg38.

**Why Phase 1 could not have caught it.** ENCODE's DNase pipeline *marks* duplicates and keeps them
(37.4 % of that BAM); ENCODE's ATAC pipeline *removes* them — all four ATAC BAMs report
`duplicate_reads: 0`. `pval_from_counts` has no duplicate step because binned counts carry no read
start positions. **On ATAC that omission is a no-op, which is exactly why the gate passed at
0.9497. On DNase it is the whole failure.** The gate's "ATAC is the template" premise has a hole
precisely here, and no tightening of the Phase 1 bar would have found it.

`T_HAP-1` is the same cause with different geometry — 57.7 % duplicates spread over thousands of
towers; dropping its top 1,000 bins (0.0008 % of the track) moves it **0.2943 → 0.8116**. For
`T_K562`, two bins of 121,241,684 are **99.98 %** of the track's mean square. **`T_adrenal_gland`
and `T_upper_lobe_of_left_lung` are not this cause** — dropping top bins makes adrenal *worse* — and
remain undiagnosed. Checked separately: duplicate excess *mass* panel-wide does not predict
agreement (r = −0.24, and the most-contaminated third scores a higher median than the least), so it
is tower concentration that breaks a track, not duplication as such.

**The cheap fix was measured and is rejected.** Capping at the MACS2 `--keep-dup 1` ceiling moves
the two broken rows the right way but improves only 3 of 34 and halves the median, 0.8344 → 0.4630,
because it flattens real peaks along with artifacts.

**It reaches past the p-value layer.** Those 9.76 M reads are in the store's **counts**, which is
what §7's count arm scores and what CANDI's Negative Binomial models directly. Re-running MACS2
would repair `pval` and leave `counts` contaminated, so the two layers would disagree about what a
read is. And **25 of the 34 `T_` DNase tracks exceed the keep-dup-1 ceiling by more than 100×**
(median 336×), which says the tower pattern is common even where it happens not to hurt agreement.
Whether the 316 ChIP tracks carry it too has not been measured, and it decides whether this is a
DNase repair or a store-wide one.

**CENSUS 2026-08-30 — the damage is DNase-only, and ChIP is clean.** All 363 signal tracks
audited against the MACS2 `--keep-dup 1` ceiling `2 × (25 + L − 1)`:

| class | run type | n | tracks with bins over the ceiling | worst ratio | worst single-bin mass share |
|---|---|---|---|---|---|
| ChIP-seq | single-ended | 266 | **0** | 1.0 | 8.5 × 10⁻⁶ |
| ChIP-seq | paired-ended | 50 | 50 | 90.6 | 3.9 × 10⁻⁵ |
| ATAC-seq | paired-ended | 7 | 7 | 54.2 | 6.9 × 10⁻⁵ |
| DNase-seq | single / paired | 18 / 22 | 40 | **108,474** | **1.7 × 10⁻²** |

**266 single-ended ChIP tracks never break the bound across 32.25 billion bins.** The 50 ChIP
tracks that exceed it are *all* paired-ended, and every clean track is single-ended — the ceiling's
derivation is a single-end argument, so exceedance by a paired-end library is the bound's
limitation, not contamination. ENCODE ships the ATAC BAMs duplicate-free, and they read 13.1×–54.2×:
**that band is what correct data looks like.** ChIP tops out at 90.6× against DNase's 108,474×, and
the worst ChIP single-bin mass share sits *below* the known-clean ATAC maximum.

**Two kinds of tower, and they separate cleanly.** *Shared:* `chr1:629,700–634,775` is the top bin
of 36 tracks across 33 biosamples and is **inside hg38 blacklist v2**, as are the other large shared
loci — 78–100 % of every class's top-10 bins are blacklisted, so `mask.h5` is already 0 there.
(Note `min_valid_frac` is 0.9, so a window may still *contain* them.) *Private:* `T_K562`'s
`chr11:24,159,5xx` appears in **1 track of 363**, and none of K562's twelve other tracks shows
anything there — which rules out copy number, since that lifts every assay of a biosample together,
as it demonstrably does for `B_SJSA1` and `B_SJCRH30` at MDM2/CDK4.

**Six scored-truth tracks are affected, all DNase — but only three meaningfully.** Measured against
the known-clean paired-end ATAC band (≤ 0.050 of mass above the ceiling): `B_DND-41` **0.472**,
`V_OCI-LY7` **0.344** and `B_NCI-H929` **0.247** are genuinely elevated; `B_RWPE2` (0.019),
`V_adrenal_gland_embryonic` (0.005) and `V_vagina` (0.002) sit inside it. **None of the six has
tower geometry** — the worst is two orders below `T_K562`. **Zero of the 42 `V_` ChIP tracks is
affected.**

**CONTROL CENSUS 2026-08-30 — the 73 `chipseq-control` tracks, which the model reads as input.**
**49 of the 58 single-ended controls break the keep-dup-1 ceiling, up to 432×, against 0 of 266
single-ended ChIP signal tracks.** All 15 paired-ended controls exceed too (median 279×, max 664×),
outside the 13.1×–54.2× band duplicate-free ATAC occupies. The *median* control's largest-bin mass
share (6.68 × 10⁻⁵) exceeds the *most* concentrated known-clean ATAC track (6.85 × 10⁻⁵ — it ties).
Controls look like DNase, not like the ChIP tracks they control for; they stay two orders below
`T_K562`. Affected: 49 on the provable criterion — `T_` 28, `V_` 21, `B_` 0 (all 9 `B_` controls
are paired-ended).

**But every control tower is shared and blacklisted, without exception:** 730 of 730 control top-10
bins are blacklisted, all 36 clusters are `High Signal Region`, and the largest *non*-blacklisted
bin in the whole control set reaches 0.107× its own ceiling. **No control has a private per-library
stack.** Six loci are control-only and invisible to a signal-side census (`chr4:49.15 Mb`,
`chr2:89.83 Mb`, `chr17:21.99 Mb`, `chr4:49.71 Mb`, `chr21:7.26 Mb`, `chr13:18.21 Mb`).

**A third independent line on K562:** `T_K562`'s own control is clean — ceiling 122, max 115, ratio
**0.94**, zero bins over — while its DNase track reads 108,474×. After the diagnosis (against 39
other DNase experiments) and the signal census (against K562's own twelve other assays), this
settles that the stack is a DNase-library fault.

So the DNase-only ruling **narrows** rather than breaks: *private, non-blacklisted, per-library
towers* are DNase-only; *duplicate load above the ceiling* is not.

> **MEASURED AND CLOSED 2026-08-30 — the mechanism was not the one I guessed, but the concern was
> real.** `genome/mask.h5` is consumed by exactly one thing, `eligible_starts` (D12 window
> eligibility); nothing in the batch path zeroes bin values at masked positions. I inferred from
> that that towers reach the encoder through D12's 10 % invalid-bin budget. **They do not.**
> Blacklist intervals run 2 kb to 18.4 Mb wide, so 367 of the 373 blacklisted top-1 tiles are
> 768/768 invalid and D12 rejects them outright; **zero tracks land in the 1–76 invalid-bin band**,
> and across all 1,744 track × window-set pairs exactly one carries an above-ceiling value on an
> invalid bin. The shared blacklisted towers are genuinely excluded.
>
> **But a different, non-blacklisted population is ingested.** Under `eic.19`, 77 of 436 tracks put
> an over-ceiling bin inside a D12-eligible window; under `eic.pilot`, 74. **All 40 DNase tracks
> ingest, in both regimes**, on `mask == 1` bins outside the blacklist — the private per-library
> class. Worst under a live regime: `T_K562` DNase, **66,660 counts at `chr5:56,896,525`, 740.7×
> its ceiling**, in an `eic.pilot` training window; under `eic.19`, `T_HAP-1` at 41,214, 352.3×, in
> 1,182 of 2,848 training windows. The median `T_` DNase track is hit in **25.6 %** of `eic.19`
> training windows. (`T_HAP-1` reaches 104,516 on chr20–22, but it is a train biosample and no live
> regime reads it there — that is what a future `eic.gw` would hand the model.)
>
> **The two regimes are identical on the eval side** — D32 restricts the train split only, so both
> draw the same 6,749 eval windows and every `V_`/`B_` number is one number, verified in code across
> all 436 tracks. On the train side neither is clean: `eic.pilot` carries the larger extreme,
> `eic.19` the wider exposure and the only control ingestion.
>
> **Severity, which is what settles it.** Worst ingested value by group: DNase **740.7×**; control
> single-ended **6.97×**; control paired-ended 2.11×; ChIP/ATAC paired-ended 13.8× — inside the
> 13.1–54.2× band duplicate-free ATAC occupies, so not evidence of contamination; **ChIP
> single-ended 1.0×, never exceeding a bound that is a proof for single-end data.** So the entire
> serious problem is DNase, and the §10 rebuild fixes all 40 of them.
>
> **Residual after the rebuild, accepted and disclosed rather than repaired.** 20 single-ended
> control tracks provably ingest duplicate-inflated bins — but at **≤ 7×**, two orders below the
> thing being fixed, on model *input* rather than scored truth. Of the 23 `V_`/`B_` tracks that
> ingest, 6 are DNase (repaired), and the remaining ChIP and ATAC ones are paired-ended at ≤ 13.8×,
> which a duplicate-free reference also reaches. A second BAM campaign for controls is not worth
> that. If this is ever revisited, the number to beat is 6.97×.

**REBUILD 2026-08-30 — dedup and counts are done and verified; the p-value step is not.** All 40
BAMs re-fetched (246.3 GB; K562's reused from the diagnosis) and deduplicated with
`samtools view -F 1024`, which drops Picard's `0x400` flag and nothing else. Verified 40/40:
after-total equals `before − duplicates` on every file, 0 duplicate-flagged reads remain, and there
are 0 secondary and 0 supplementary reads anywhere, so `-F 1024` and ENCODE's own `-F 1796/1804`
remove the same reads here. Summed DNase depth 6.314 B → 4.410 B (−30.15 %). The counting rule is
proven against the store: run on the **raw** K562 BAM it reproduces the existing `counts` column on
23/23 chromosomes with **0 mismatching bins**.

| | max bin | ratio to ceiling | `-log10 p` max |
|---|---|---|---|
| `T_K562` before | 9,762,619 | 108,473× | 1.82 × 10⁷ |
| `T_K562` after | **90** | **1.0×** | **80.3** |
| `T_HAP-1` after | 6,677 | 56.6× | 7,607 |
| `B_DND-41` after | 8,069 | 43.1× (inside the clean-ATAC band) | — |

**All 18 single-ended DNase tracks now satisfy the keep-dup-1 bound exactly as the 266 clean
single-ended ChIP tracks do** (`frac_mass_over_ceiling` = 0 on all 18, ratio 0.75–1.01).

> **OPEN — dedup is right for paired-end reads and destructive for single-end.** Measured through
> the old 25 bp approximation as a one-variable control (same λ rule, same statistics, only the
> counts changed; `encode_rdns_vs_challenge`, where neither side moved, reproduces Phase 2 to
> 1e-12), median Pearson falls **0.8344 → 0.5974**. The split is diagnostic: **single-ended
> 0.7881 → 0.5250**, against **0.5482** for the keep-dup-1 cap §10 already rejected — the two agree
> per track to a median absolute difference of 0.032. **Picard keys a single-end duplicate on start
> position and strand alone**, so at a genuinely busy locus it marks distinct fragments as copies:
> `T_BE2C`'s MYCN amplicon — the diagnosis's own example of *real* signal — falls 0.9439 → 0.1929,
> top bin 116,402 → 123. On the 22 paired-ended libraries, where the key is both ends, dedup beats
> that cap on **every** row (+0.05 to +0.50) and mean ratio lands at **1.0002**.
>
> **Note what this exposes about the template.** §10 chose ATAC as the template because ENCODE's
> ATAC pipeline removes duplicates — but **all 7 ATAC experiments are paired-ended**, so the
> template's dedup step was never validated on single-end data, and ENCODE keeps duplicates for
> single-end DNase by choice rather than oversight.
>
> **Not yet settled, because the measurement above is the wrong instrument.** Base-resolution MACS2
> extends every tag to 150 bp before the Poisson test, so a pileup holding one tag per start still
> accumulates across 150 distinct starts — the per-bin saturation driving the single-end collapse is
> exactly what base resolution should undo, and §10 says *base resolution*, not *deduplicate*.
> **Ruled 2026-08-30: test it rather than assume it.** MACS2 runs on both the deduplicated and the
> original BAMs for the 18 single-ended tracks, so the question is answered by measurement.

**MACS2, approved 2026-08-30.** Not on Fir — no module, absent from `candi_venv`, no container.
Installed as `MACS2==2.2.9.1` from the Alliance offline wheelhouse into a **fresh venv on scratch**;
`candi_venv` untouched. ENCODE ran 2.1.1, which is Python-2.7-only and cannot run on Fir at all;
diffing `PeakDetect.__call_peaks_wo_control` between `v2.1.1.20160309` and `v2.2.9.1` leaves the λ
block byte-identical line for line, with the sole difference in peak-calling cutoff plumbing — the
one block we discard, since the p-value comes from `bdgcmp -m ppois`.

**MACS2, both arms, 2026-08-30 — the control arm won, and it identified the target.** 58 runs, 0
failures. MACS2 `2.2.9.1+computecanada` in an isolated scratch venv; store confirmed untouched.

| median over the 34 `T_` | Phase 2, 25 bp, dup-kept | dedup, 25 bp | MACS2 dedup | MACS2 original |
|---|---|---|---|---|
| Pearson | 0.8344 | 0.5974 | 0.6576 | — |
| — single-ended (n=17) | 0.7881 | 0.5250 | 0.5961 | **0.9915** |
| — paired-ended (n=17) | 0.8454 | 0.6691 | 0.6720 | **0.9765** |
| — all 34 `T_` | 0.8344 | 0.5974 | 0.6576 | **0.9851** |
| Spearman | 0.5648 | 0.5376 | 0.6939 | 0.7119 (se) |
| MSE ratio | 1.6386 | 0.0823 | 0.2212 | 0.6705 (se) |
| top-1 % Jaccard | 0.5649 | 0.5955 | **0.7865** | 0.8264 (se) |

**Base resolution does not rescue single-end dedup.** Holding dedup fixed, 25 bp → base lifts median
Pearson only 0.5250 → 0.5961 — it helps 17/17 rows and recovers about a sixth of what dedup removed.
The hypothesis was directionally right and quantitatively far short. It *does* fix the ranking
deficit Phase 2 flagged, across both run types: Jaccard 0.5649 → 0.7865, Spearman 0.5648 → 0.6939.

**The original arm reproduces the target.** Original beats deduplicated on 16 of 17 single-ended
rows — median Pearson **0.9915**, with seven rows above 0.997 and none but `T_K562` below 0.918.
A median of 0.9915 is not "a better layer"; it is the target's own generating process. **The reading
that fits: the challenge's 2019 DNase tracks were themselves made by MACS2 at base resolution from
duplicate-kept alignments.** That is a claim about the *target*, and it is the most useful thing
this programme has found.

`T_BE2C`'s MYCN amplicon settles the mechanism. In `chr2:15.70–16.40 Mb` the challenge target peaks
at **166,228.8** at `chr2:16,201,975`; the original arm gives **171,502.4 in the same bin**
(window r = 0.9986), while the dedup arm gives **104.2** — three orders short — and lands elsewhere.
On chr1, which carries no amplicon, the dedup arm scores `T_BE2C` at 0.8132 against 0.2050
genome-wide: the collapse is one locus, not a diffuse loss.

`T_K562` is the exact mirror and the sole inversion (0.0091 original vs 0.4271 dedup). In
`chr11:24.0–24.3 Mb` the challenge's own track maxes at **1.93**; the original arm faithfully
reproduces the **1.363 × 10⁷** adapter tower. **So the released ENCODE alignment is not the
alignment the 2019 pipeline used**, and dedup is the only arm that removes that tower.

> **OPEN — the circularity §10 warned about has arrived, by discovery rather than by tuning.**
> §10 says tuning to match the challenge "would make the truth toggle circular for DNase and
> permanently exclude accessibility from the robustness story." Nothing was tuned — a
> pre-specified recipe was run and the match was found. But the consequence for §6 is identical:
> if the store's DNase p-value is MACS2-on-original, then store truth ≈ challenge truth at r ≈ 0.99
> for DNase, and the truth toggle measures nothing there. Ruling needed.
>
**The paired-end control closes the confound, and strengthens the claim.** 22/22 runs. Every sign
matches the single-ended side at about a third the magnitude — the dedup penalty is −0.1101 paired
against −0.3265 single, which is exactly what deduplication keying on *both* ends predicts. The
discriminating point is the absolute level, not the gap: **on paired libraries dedup is faithful, so
if 0.9915 were merely single-end dedup destroying coverage, the deduplicated arm should have closed
most of the distance. It does not — 0.6720 against 0.9765.** Near-perfect agreement survives exactly
where dedup is not the confound. MACS2-original beats Phase 2 on **34/34**.

**New caveat, and it is real: the paired agreement is shape-only.** MSE ratio 0.6705 single against
**28.15** paired; mean ratio 0.80 against **2.98**. That offset is present in *every* arm including
Phase 2 (2.30×) and the store's own shipped signal (1.33×), so this run inherited it rather than
created it — but it means the generating-process claim is about **shape**, and `AGENTS.md` §7.2's
`oracle_scaled`/`scale_error` split is not optional for DNase. A fragment-versus-mate counting test
would settle the cause; it needs a third variant and has not been run.

**`T_HAP-1` is a second "released alignment ≠ 2019 alignment", one per run type.** Unlike `T_K562`,
the challenge's own track *does* carry signal at its tower (`chr17:2,700,525`, 5,972) — ours is
47.8× higher there and 120.6× at `chr4:31,466,150`. Two bins set its genome-wide number: in its own
window the original arm scores **0.9975**, and chr1 alone scores 0.6207 against 0.4776 genome-wide,
which a diffuse loss cannot do.

**`B_DND-41`'s tower survives deduplication** — top bin genome-wide in both arms, cut only 7×
(63,282 → 9,030). No single-ended track did that. It is a blacklisted `chr1:634,000` mappability
tower, so dedup is simply the wrong instrument for it. It is scored truth, so no agreement number
was computed and none exists.

**ADAPTER FILTER 2026-08-30 — the control passes, K562 improves but is not repaired, and the
`T_HAP-1` prediction is refuted.** Rule: a read is adapter-derived iff its whole sequence fits
inside a standard Illumina TruSeq adapter (either orientation, no overhang) at Hamming distance
`d ≤ NM`, the aligner's own edit distance for that read. No count threshold, no position rule, no
per-track parameter. BAM-level, no realignment — honest here because the defect is *full-length*
adapter reads carrying no genomic base. 9,902,610 reads removed of 6,319,734,741 (0.157 %), and
**28 of the 40 filtered BAMs are hard links to the originals**.

| median Pearson, 34 `T_` | MACS2 original | MACS2 adapter-filtered |
|---|---|---|
| all 34 | 0.9851 | **0.9851** |
| single-ended (17) | 0.9915 | **0.9915** |
| paired-ended (17) | 0.9765 | **0.9765** |

The only median that moves is MSE ratio over all 34, **3.4214 → 2.2235** — `T_K562` alone crossing
the middle. Only 8 of 34 rows differ beyond 1e-4.

- **`T_BE2C`'s MYCN amplicon survives untouched — the control passes.** Zero reads removed, peak
  171,502.4 in the target's own bin, window Pearson 0.9986, counts identical to the store's. The
  filter removes artifact and not biology.
- **`T_K562`: the adapter defect is fixed, the track is not.** The `chr11` tower is gone (counts
  9,762,619 → **339**, `-log10 p` 1.363 × 10⁷ → **300.8**), and Pearson moves 0.0091 → **0.4559**.
  It stops short because the filter *revealed a second private tower* at `chr5:56,896,525` — 66,660
  counts, 99.97 % duplicate-flagged, **the exact locus flagged above as ingesting into `eic.pilot`
  training windows**. Its 20-mer is an adapter–adapter *junction*, contained in no single adapter
  (min Hamming 6 of 20, against 0 for the chr11 one), so the rule cannot reach it. The extension
  that would — containment in the concatenation of two adapter strands — was named and deliberately
  **not** run, because naming a rule after seeing the number it would fix is tuning.
- **`T_HAP-1`: the ruling's prediction is refuted.** 126 reads removed of 243 M; both ratios
  unchanged to four decimals (47.8× and 120.6×). Its tower 20-mers are Hamming 9 and 8 from the
  nearest adapter — **not adapter at all**.
- **`B_DND-41`: unfixed, as predicted.** Zero reads removed. Its tower is mitochondrial sequence on
  the `chr1` NUMT, Hamming 21 of 36 from any adapter. It is scored truth, so no agreement number
  exists for it.

**Fragment-versus-mate, run once and left there.** `bamtobed` emits one tag per *record*, so a
paired fragment currently contributes two tags; `-f 64` halves the tag count exactly. Median paired
mean ratio **2.9862 → 1.6618** and MSE ratio 28.15 → 5.35, with Pearson flat (+0.0004) — but
**Jaccard falls 0.6527 → 0.5806 and Spearman 0.6282 → 0.6015**, and per track it overshoots below
1.0 on some rows while undershooting on others. It removes about two thirds of the excess and does
not reach 1.0. **Inconclusive; no second variant was run.**

**Correction to the promotion checklist, and it matters.** On the **adopted** arm the six scored
`V_`/`B_` depth changes are **0 % to 0.000189 %**, not the 7.9 %–30.6 % recorded earlier — that
range belongs to the deduplicated arm the ruling did **not** adopt.

### Rulings on the DNase layer (2026-08-30)

**Adopt the original arm; keep DNase in the truth toggle, with a caveat badge.** MACS2-on-original
is demonstrably the right pipeline and refusing the correct method to protect a comparison would be
backwards. §6's DNase toggle therefore prints with a badge stating that the two truths share a
generating process at r ≈ 0.99, so agreement there is not evidence of robustness.

**`T_K562`: trim adapters, then the original recipe.** Attack the actual defect — 9.73 M untrimmed
adapter read-throughs — rather than duplication in general, and apply it uniformly to all 40 tracks
so one assay keeps one treatment, which is what §10 exists to enforce. *(This ruling also expected
the filter to carry `T_HAP-1`. It did not — see the refutation above.)*

### The DNase layer is CLOSED (2026-08-30)

**Adopted: MACS2 at base resolution on adapter-filtered, duplicate-kept alignments.** Median Pearson
against the challenge **0.9851** over all 34 `T_` (0.9915 single-ended, 0.9765 paired-ended),
beating Phase 2 on 34/34. Three closing rulings:

1. **Stop at `T_K562` = 0.4559; do not chase the residual.** The junction rule that would reach its
   `chr5:56,896,525` tower was named only after seeing the number it would fix, and that is tuning
   whatever its biological justification. **Disclosed:** `T_K562` is the worst DNase row, and its
   chr5 tower — 66,660 counts, 740.7× its ceiling — continues to enter `eic.pilot` training windows.
2. **Accept `T_HAP-1` and `B_DND-41`, and disclose them.** Two instruments have now failed on both,
   and neither defect is one we introduced — they are properties of the released ENCODE alignments.
   `T_HAP-1` runs 47.8× and 120.6× above the target at two loci and is a training track;
   **`B_DND-41` is scored truth carrying a known mitochondrial-NUMT artifact for which no agreement
   number can ever exist.** Both ratios print beside every DNase number.
3. **Do not adopt fragment counting; disclose the scale offset.** It buys two thirds of the
   correction and pays in ranking quality (Jaccard 0.6527 → 0.5806) without reaching 1.0.
   `AGENTS.md` §7.2's `oracle_scaled` / `scale_error` split already separates shape from scale, so
   the paired ~3× offset is reportable without changing the layer every method trains on.

**What this means for §6.** The adopted layer agrees with challenge truth at r ≈ 0.99 in *shape*,
so the DNase truth toggle carries the badge ruled above — agreement there is not evidence of
robustness. The scale disagreement is real and is the one thing the toggle still measures for DNase.

**Phase 3 promoted 2026-08-30.** The rebuilt layers are in the live store. The sequence that was
run, its gates and the rollback are §25 and §26 of
`cruxvault/results/t78/G1_REBUILD_FROM_BAMS.md`; the checklist it came from is §19.

> **PHASE 3 PROMOTED 2026-08-30 — the cut-over ran, every gate passed on the first attempt.**
> The live store at `fir:/project/def-maxwl/mforooz/CANDI_STORE/eic` now carries the rebuilt
> `counts` and `pval` layers on 40 biosamples and the archived predecessor under the new
> `signal_rdns` kind. **The new `store_manifest_sha256` is
> `c9a95e4e424d94496be7197ad4aa3d08cd9d7d31144bcf72f53ca50505a2fd83`** — quote this, not
> `6c0e0c3e…`, for any run scored after this date. `manifest_equiv.py` reports the promoted
> manifest `EQUIVALENT` to the staged one and `candi.store verify` reports `OK`; the manifest's one
> metadata gap (`V_testis/chipseq-control/platform`) is pre-existing and unmoved.
>
> **The RDNS constraint holds, and was re-proved after the promotion, not only before it:** all 40
> archives still hash to the `pval.h5` they archive (`ARCHIVE STILL OK 40/40 POST-PROMOTION`). The
> archive was made by **renaming** the live file in place — no bytes moved and nothing was copied
> over it — so byte-identity is a property of the operation, not of a check taken afterwards.
>
> The promotion covers **40 of the corpus's 89 biosamples** — exactly the DNase-carrying ones (34
> `T_`, 3 `B_`, 3 `V_`). The other 49 are untouched: their signal layer was already
> `signal p-value`, so they end with no `signal_rdns.h5`, and both `build-manifest` and
> `verify_store` discover kinds per file, so that asymmetry is legal. All 363 track records declare
> `signal p-value`, so §10's strict units gate passes.
>
> Read back on chr21 bins 1,200,000–1,600,000 (30–40 Mb): `T_K562` `pval` max 7486.471 / mean
> 4.55361 against `signal_rdns` max 21.196 / mean 0.08033, correlating 0.9115 and 0.9435 with the
> counts they came from; `T_IMR-90` 6521.455 / 7.72938 against 20.807 / 0.06868, at 0.9610 and
> 0.9729. The 189 non-DNase tracks in those biosamples were bit-identical to the pre-promotion
> store over 52.4 × 10⁹ compared cells, the per-file `dtype` flips **0 of 89** on the adopted arm
> (the 11-biosample figure was the deduplicated arm's), and `depth` — covariate row 0 — moves on 12
> of 40 tracks, materially on `T_K562` alone (Δ log2 −0.0442).
>
> **Rollback is still open.** The pre-promotion `counts.h5` survives as `counts.h5.pre_t78p3` on all
> 40, each hashing to the original, beside `manifest.json.pre_t78p3` and `LIVE_SHA256.pre_t78p3.txt`.
> §25 Step 7 — deleting those backups — is **deliberately not run**, and does not run until the
> retrain has landed. Evidence: `cruxvault/results/t78/stage3/` and §25 of the memo above.

Correction to the earlier rebuild memo, on record: the depth change on the six scored `V_`/`B_`
tracks runs **7.9 %** (`V_adrenal_gland_embryonic`) to 30.6 %, not 1.6 % to 30.6 %.

Two items added to the promotion checklist: the layer must carry its MACS2 version, command and
wheel hash into the store; and the rebuilt counts' `duplicates_removed: False` attribute must be
renamed — it means "this tool did not do the removing" and reads as the opposite of the truth.

### The ruling (2026-08-30)

**Take §10's pre-registered fallback, for all 40 DNase experiments, rebuilding `counts` as well as
`pval`.** The root cause is that ENCODE's DNase pipeline keeps duplicates while its ATAC pipeline
removes them — so a DNase layer built from duplicate-kept counts was never faithful to the ATAC
template §10 chose, and this is the only route that makes it so. Deduplication needs read start
positions, which binned counts do not carry, so BAMs are required. A targeted repair of the five
damaged tracks was rejected: it would build one assay through two pipelines, which is the exact
inconsistency §10 exists to delete. Capping at the keep-dup-1 ceiling was measured and rejected
(helps 3 of 34, halves the median).

Three consequences on record. **Scored truth changes for 6 DNase `V_`/`B_` tracks** — legitimate
under Rule 1, which binds methods and not truths, but the DNase rows are then measured against a
truth we rebuilt rather than ENCODE's shipped file, and that must be disclosed wherever a DNase
number appears. **The Phase 2 layer built on 2026-08-29 is superseded** and does not go to Phase 3.
**Nothing is deleted:** the rebuild produces a new layer beside the existing one, per §10 Phase 3's
own rule.

**Superseded — kept for the record.** What follows was the reading before the diagnosis above: `T_K562` scores
Pearson **0.0038** against the challenge where RDNS scores 0.4938, with an MSE ratio of 2,746 and
`-log10 p` reaching 1.8 × 10⁷ — three orders past what `layout.py`'s codec was written against, on
20 bp reads. `T_HAP-1` regresses similarly, 0.2943 against RDNS's 0.5825. Both are **training**
tracks. A `-log10 p` of 10⁷ entering a joint loss would dominate it. Diagnosing this is bug-hunting,
which §10 permits; it is not tuning. **Phase 3 is held until it is understood.**

**Phase 3 — store layout, nothing deleted.** The read-depth normalized signal moves to its own
archived kind (`signal_rdns`) and stays queryable. The `pval` kind for DNase becomes the computed
layer, and only after Phases 1 and 2 pass. Each DNase track records both the archived source
accession and the derived layer's method, parameters and code sha.

> **Reaffirmed by the PI 2026-08-30, and it is a hard constraint on the promotion.** The original
> read-depth normalized signal files are **archived, never deleted and never overwritten.** They
> move to `signal_rdns` and remain readable; what changes is only which kind is the *default* for
> DNase — that becomes the computed −log10 p-value layer. A promotion that loses, truncates or
> writes over an RDNS file has failed, whatever else it got right, and the check that it did not is
> part of the promotion's own verification rather than an afterthought.

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
| ~~**G3**~~ | **DONE 2026-08-29 — the plan's shape does not change.** See §12.7 | job 57482367, 2 min 19 s | — |

G1 is the real gate. Skipping it trains every method on 34 tracks of the wrong units.

### 12.2 Training runs — 10, plus 5 σ refits

| method | runs | why that count | arms |
|---|---|---|---|
| CANDI | 2 | `eic.19` and `eic.pilot` | pval + count + peak |
| Avocado | 2 | joint fit on chr19 / pilot, then **3** per-chromosome genome-factor fits each | pval |
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

**Avocado fits 3 chromosomes of genomic factors, not 23** (CORRECTED 2026-08-30). Avocado's
genomic factors are per-position free parameters, so a chromosome it never fitted has no
representation and cannot be predicted at all — which is why the authors' own scheme refits them
per chromosome. But §4 blanks Avocado's `genome-wide` cell and rules that a blanked cell is **not
computed**, so it is only ever predicted on chr20+21+22. Fitting the other 20 chromosomes would
produce factors for positions nothing is ever scored at. The earlier "23" in this table predated
the blanking ruling and was never reconciled with it; it is about 20 fits per regime, 40 across
both. The joint fit's own chromosome (chr19 / the pilot regions) comes free with the shared run and
is not one of the three.

**σ-tables refit for all 10 methods**, on training residuals (§7). Every existing σ was fit on `V_`
eval pairs and is void under Rule 1. Cheap, but nothing distributional scores until it is done.

**MID-TRAINING EVAL IS NOW MORE EXPENSIVE THAN TRAINING (measured 2026-08-31).** A full-coverage
`V_` pass costs **5,446.6 s = 91 minutes**, not the ~13 minutes every earlier estimate here assumed.
The merged monitor scores all 12 tiered keys over **every 25 bp bin of chr20+21+22 on 45 tracks** —
`gwspear` alone is a rank sort over the whole scope. At `EVAL_EVERY=3` across 25 epochs that is 8–9
checks ≈ **12–14 GPU-h of evaluation**, against ~10–17 GPU-h of training for the same run.

The 12.9-minute figure that preceded it was a *different estimator*: `candi.eval.quick_eval`, which
sampled **16 windows per track** (~0.24 % of the scope) and was deleted from the tree in `2f56cb1`
when `candi/monitor.py` replaced it. Same collapse rule, same 45 tracks, 400× the coverage.

Two consequences, and neither is optional:

- **Size a retrain's walltime with the eval included**, or lengthen `EVAL_EVERY`. A 16 h band that
  fits the training does not fit the run.
- **`--early-stop-epochs` now watches a trustworthy number.** Its resolution is `EVAL_EVERY`, so
  the patience trades directly against 91 minutes a check. That trade is the reason to think about
  `EVAL_EVERY`, not the training cost.

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

485 MB per **array** genome-wide (121,241,684 bins × 4 bytes); 25.9 MB per array on the 6,478,903
held-out bins. **The table above counts one array per track and is therefore low — CANDI writes
five. See §12.6, corrected 2026-08-31.** `B_` is **touched once**, at the very end, and by the
same ruling `B_` lands on `/project`, not scratch; `V_` stays on scratch and is deletable after
scoring.

### 12.4 Scoring runs — ≈4,360 CPU-h

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

### 12.7 CANDI's inference cost, measured (G3, 2026-08-29)

Job 57482367 on Fir, whole chr21 (1,868,399 bins), one track end to end through the real
`StoreSource → stream_tracks → forward` path, harness unmodified. One H100 **MIG 1g.10gb** slice
per invariant 13, **fp32** (`bench/cli.py` wraps eval in `no_autocast`), `--batch-windows 4`
(the `candi.bench` default), 768-bin context, 35 assays, 2,353,661 params.

> **The weights are a random init.** No trained CANDI checkpoint exists on Fir — re-verified
> independently. This is a throughput measurement and carries no accuracy claim.

| | |
|---|---|
| rate | **185.6 windows/s = 142,541 bins/s** |
| split | forward 56 %, store loader 37 %, host copies 7 % — compute-bound, but only just |
| one genome-wide track | 851 s = **0.2363 GPU-h** |
| CANDI's 4 runs (`V_`/`B_` × 2 regimes) | 192 track sweeps = **45.4 GPU-h** (38.0 at `--batch-windows 16`) |
| the same, held-out only | **2.4 GPU-h** |
| peak memory | MaxRSS **4.2 GiB** against 32 GB requested — size prediction jobs at 8 GB, not 32 |

**Verdict: §4's `genome-wide` aggregation stays.** Its premium is **43 GPU-h, once** — the size of
Avocado's already-accepted ≈40 GPU-h, and 2.6 % of the programme against §12.4's ≈4,360 CPU-h.
These are MIG-slice hours, which is the unit invariant 13 makes us allocate in.

Three secondary findings. A perfect loader would buy 1.83×, so the loader is worth attention but is
not the wall. **The loader's cost is CPU work, not network I/O** — staging the store to node-local
disk moved the rate +3.4 %, so staging a prediction run buys nothing. And the rate is a property of
the model, not the cell: three `V_` biosamples spread 2 %.

**A CANDI pass is per-track, not per-eval-pair, on the store path.** `harness.py:621` returns
`[[a] for a in cols]` for a store source (the h5 path returns one group), consumed by the window
loop at `harness.py:653`. So the multiplier is 45 / 51 **tracks**, not 26 / 12 pairs. Confirmed
against the store's own availability, which gives exactly 26 `V_` → 45 and 12 `B_` → 51.

**The 404 windows/s previously on record is doubly incomparable** and should not be quoted again:
`cruxvault/results/t3/DELIVERABLE.md:34,64` gives it beside "18,209 non-overlapping **6144-bin**
windows" — a loader rate, on a window 8× the production context. bins/s is the portable unit.

### 12.8 A gap G3 found: there is no track writer — **WRONG, and corrected 2026-08-31**

> **This section's premise was false when it was written.** `src/candi/bench/dump.py` — 249 lines
> with 248 lines of tests — already wrote a CANDI checkpoint's predictions to disk in the
> `RIVALS_PLAN.md` §4.1 external contract, including `signal_mu` / `signal_sigma` / `peak_score`.
> It was on `implementation/t62-candi-rows`, and the survey below read only
> `implementation/t77-benchmark-design`. Merged onto the working branch 2026-08-31.
>
> What remains open is narrower and is `t83`'s real subject: does that writer already satisfy the
> claims below, or does it still need the multi-array HDF5 layout and the compression of
> `plan/T83_PREDICTION_WRITER.md` §3? **It is no longer a build from nothing, and it is no longer
> the thing gating the first prediction run.**
>
> Same failure as `t84`'s "only Avocado is vendored": a claim about the repo read off one branch.

The original text follows, for the record.


`stream_tracks` yields in-memory `TrackRecord`s straight into `score_track`. **The §12.3 prediction
run that persists 485 MB per track to scratch does not exist as code.**

This is not optional plumbing. Every "predict once, score many times" claim in this document rests
on it: the two aggregations of §4, the three `V_` numbers of §5.2, the two truths of §6, and above
all §5's ruling that `B_` is *predicted* once. Without a writer, each of those re-runs inference,
which for CANDI is 45.4 GPU-h a time and breaks the touch-once discipline outright.

So the writer is new work on the critical path, and it belongs to `t80`. It is cheap — CANDI's
whole share is 93 GB, minutes not hours — but it must land before the first prediction run.

### 12.5 One costing left open

1. ~~**Do the naive baselines need one fit or two?**~~ **SETTLED 2026-08-29: one** — one fit, one
   prediction, one score, printed in both regime rows, with the identity assertion of §12.2.
2. **Does the truth toggle apply to `V_`, or only to `B_`?** The challenge staged 45–46 round-1
   validation tracks that were never scored, so `V_` under challenge truth is possible. It is
   **excluded** from the §12.4 table. Adding it is +20 passes, ≈1,000 CPU-h.

### 12.6 Storage, in full

**CORRECTED 2026-08-31 (PI approved). The figures below are per ARRAY; a method writes as many
arrays as its heads have parameters.** 485 MB per array genome-wide (121,241,684 bins × 4 bytes).
The old text read that as 485 MB per *track*, which is one array — true of a point predictor and
false of every distributional one.

A `TrackRecord` (`harness.py:131-159`) carries the prediction arrays, not one number per bin:

| method kind | prediction arrays per track | genome-wide, raw |
|---|---|---|
| point (`ChromImpute`, the naive point baselines) | 1 — the value | 485 MB |
| **CANDI, 3 heads** | **5** — `mu`, `n` (NB), `signal_mu`, `signal_sigma` (Gaussian), `peak_score` | **2.37 GB** |
| a pval-arm rival with a σ-table | 2 — the value and its σ | 970 MB |

So CANDI's own share is **466 GB raw** across its four `V_`/`B_` × 2-regime units, against the
≈434 GB §12.3 gives for the whole programme. **§12.3's 434 GB total is therefore also low, and by
about the same factor** — it was summed from the same one-array-per-track assumption.

**Compression buys much less than first thought, and is not what makes this affordable.** An
earlier draft here quoted **2.69×** and put CANDI at ≈173 GB. That figure is wrong for
predictions, for a reason worth stating because it is easy to repeat:

2.69× was a blend of a sparse layer and a dense one — `counts` at 5.8×–19.7× pulling the average
up, `pval` at 1.3×–2.1× holding it down — measured on **truth** arrays, where the count layer
really is sparse integers. **A prediction has no sparse layer.** All five of CANDI's prediction
arrays are model outputs — `mu` and `n` off a softplus, `signal_mu`, `signal_sigma`, `peak_score`
likewise — and every one is a smooth full-mantissa float. The count arm's 15× never applies on the
prediction side at all.

| array kind | `np.savez` | `np.savez_compressed` |
|---|---|---|
| smooth float32 — **what a prediction is** | 1.00× | **1.27×** |
| sparse count-like — what truth counts are | 1.00× | 15.39× |

*Measured 2026-08-31 on 2 M-element float32 arrays, and separately reproduced.* `dump.py:102`
currently calls `np.savez`, which compresses **not at all** — so today CANDI's ≈466 GB is written
raw. Switching to `savez_compressed` (`t83`) brings it to **≈367 GB**.

> Still an estimate, on synthetic arrays. **The first real prediction run records its own ratio and
> this table is rewritten from it.** Full argument in `plan/T83_PREDICTION_WRITER.md` §4.2.

**What actually makes the plan affordable is that Fir has room** — 13 TiB free on `/project`,
17 TiB free on scratch, checked 2026-08-31. Compression is a convenience here, not the load-bearing
assumption it was written as.

**Where it lives — RULED 2026-08-31 (PI).** `V_` predictions go to **scratch** and are deletable
once scored. **`B_` predictions go to `/project`.** §5 rules that `B_` is predicted exactly once,
so the first prediction set is the only legitimate copy that will ever exist: any measure not
computed before a purge becomes unreachable, and re-predicting to recover it would break the
touch-once rule outright. **Scratch purges at 60 days** — which also applies to the 23 entrant
bigwigs already staged there, and to the one trained CANDI checkpoint (now copied to
`/project/def-maxwl/mforooz/t81_checkpoints/`).

---

## 13. What has to be redone

Accepted as the price of the corrections above.

1. **Every trainable method retrains** — CANDI, Avocado, ChromImpute, eDICE, Lavawizard, and the
   fitted baselines. Four independent reasons, any one of which alone would force it: the DNase
   target changes (§10), checkpoint selection moves to `V_` (§5), the eval chromosomes change to
   chr20+21+22 (§4), and Avocado's joint fit moves off chr20 (§3.2).
2. **Every `V_` row rescores** genome-wide, through one scorer, emitting both aggregations.
3. **`B_` is not touched** until each method's final, once-only run.
4. **The 23 entrant submissions rescore** through our scorer on our grid.

   **CHECKED 2026-08-31, and there is a deadline.** All 23 are present at
   `/scratch/mforooz/t54_submissions_round2` — **960 GB**, one directory per entrant. The oldest
   file is dated **2026-08-25**, so the 60-day purge clock expires around **2026-10-24**. Either
   the ≈2,835 CPU-h of anchor scoring (§12.4) completes before then, or the tree moves to
   `/project` first. `/project` had 13 TiB free at the time of the check, so it fits.

   The challenge **truth** is not at risk and needs no action: 363 bigwigs, 254 GB, already on
   `/project` at `DATA_EIC_SYNAPSE/` (`blind_truth`, `training_data`, `validation_data`),
   md5-verified against Nibi. It is the submissions, not the truth, that sit on a purge clock.
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
| the hg38 lift: 44/44 mapped, 0 unmapped/split, 29,984,074 bp, 25,588,197 training bp, 1,023,489 contained bins | `configs/regions/PROVENANCE.md`; recomputed from the shipped BED, 2026-08-29 |
| the lift cross-checked off UCSC: Ensembl GRCh37→GRCh38 agrees on 16/16 endpoints over 8 regions | `cruxvault/results/t79/G2_PILOT_HG38.md` |
| Avocado returns its random init on an unfitted chromosome; Lavawizard raises | `rivals/avocado/avocado.py:90-97` (vendored here); `competitors/lavawizard/dataset3.py:70-71` (not vendored) |
| per-method training loci and genome fractions across the literature | primary sources in `cruxvault/raw/`, audited 2026-08-29 |
| `V_`/`B_` panel composition — 45 exp / 22 assays vs 51 exp / 8 assays; splits disjoint on (cell, assay), 0 overlaps over 89 cells | `download_plan_eic.json` (Fir `EpiDenoise/data/`), recounted 2026-08-29 |
| the scored panel is *assays the truth cell has and the input cell does not* | **citation was wrong and the rule needs re-checking — see the note under §5.1.** `harness.py:526-543` is `StoreSource.counts_at_dsf`; the h5 rule is `H5Source.targets:316-322` and the store rule is `StoreSource.targets:486-488`. pairing declared by `tools/declare_eval_pairs.py` (**written 2026-08-29** — it did not exist when this row was first written, and the store path self-paired instead; see §5.1) |
| eval-scope bin counts, scoring cost and storage | `cruxvault/results/t50/scores_avocado_P2.json` provenance; `cruxvault/results/t51/PILOT_MEMO.md` |

---

## 15. Status

Every design question raised in this pingpong is settled. What remains is execution.

### Rulings of 2026-08-31 (PI)

Taken after the first CANDI retrain was launched, killed, and read.

- **`B_` is never READ during training, not merely never selected on.** "Never ever we use `B_` in
  training — `V_` is only for checkpoint selection and monitoring, not training." So `SELECT_ON=V`
  in `slurm/t81_train_candi.sh` is the compliant setting and is now the default; the open question
  that stood in that script is closed. `V_` is eval-only too: it selects and it monitors, and no
  gradient is ever taken on it.
- **Training stops on a stalled `V_`.** `--early-stop-epochs`, default 3, counted in **epochs**.
  Job 57620803_0 selected its best checkpoint at epoch 2 and then ran nine more GPU-hours while
  `V_imp_crps` rose 0.5604 → 0.5820 → 0.5864, with nothing in the loop able to end it. This also
  retires the "how many epochs?" question §12 left open: the answer is read off the curve, not set
  in advance.

  > **AMENDED 2026-08-31, and the amendment matters.** That 0.5604 → 0.5820 → 0.5864 curve was
  > measured by `candi.eval.quick_eval` on **16 windows per track** — 0.24 % of the eval scope, by
  > an estimator the merge has since deleted. Re-scoring the same epoch-2 checkpoint under the
  > merged full-coverage monitor gives **0.6252**, and that +0.0648 shift is **three times** the
  > 0.0216 margin by which epoch 2 beat epoch 5. **So the curve does not establish that the model
  > overfitted after epoch 2.** It may have; it cannot be shown from that data, and it never will
  > be, because `_keep_best` writes only on improvement and the epoch-5 and epoch-8 weights were
  > never saved. The early-stopping rule is still right — it now watches a full-coverage number —
  > but the evidence that motivated it was thinner than it looked.
- **§12.6's storage figures are per array, not per track**, and `B_` predictions live on
  `/project`. See §12.6.
- **The noise floor runs AFTER the retrains, not before.** `t86` competes with them for the same
  GPU, and it grew from one training run to two when the `eic.19` checkpoint proved unusable. Rows
  therefore go up **unranked** in the interim, which §15's deferral already allows.
- **`t82`'s two judgment calls are accepted as shipped.** The board stays locked — both regimes
  unfrozen, so `add` refuses every row until a retrain produces one — and `no-selection` marks all
  five naive baselines rather than only the two kNNs §5 names.
- **`t84`'s scope was wrong and is closed**; **`t87`** now owns the two same-named pairing tools
  the `main` merge collided.

### Rulings of 2026-09-01 (PI) — the selection rule, and two rivals

Taken after all four rivals gained a `V_` selection loop and each independently hit the same wall.

- **THE SELECTION KEY IS DELIBERATELY NOT UNIFORM.** CANDI selects on a **count-arm `crps`**
  (`monitor.py`'s `SELECTION_KEY`); Avocado, eDICE and Lavawizard select on **`pval:mse`**.
  §5's uniform rule therefore binds the **panel, the instrument, the scope and the cadence — but
  not the key**, and this is now the rule rather than a gap in it.

  The reason it cannot bind the key: §7 gives no rival a count head, and the pval `crps` needs a
  σ-table, which Rule 1 forbids fitting on `V_`. The third option — build the training-residual σ
  pass first so every method selects on `crps` — was rejected because that pass does not exist
  anywhere in the tree and would block all nine retrains behind it.

  **Which way the asymmetry cuts, stated so nobody has to re-derive it:** CANDI is tuned on an arm
  no rival can enter, while the head-to-head is the pval arm (§7). So CANDI is selected on a
  *different* objective from the one it is compared on, and the rivals are not. That **handicaps**
  CANDI rather than favouring it — the asymmetry is a legibility problem, not a fairness risk in
  our own direction. Every rival row carries it as a marker.

- **Lavawizard gets a transferable stage built** — a shared fit on `train_chroms` (or the BED under
  `eic.pilot`) producing the cell and assay factors, then per-chromosome position factors, which is
  Avocado's scheme. Without it its factors sit on chr20/21/22 and breach Rule 2 in **both** regimes,
  and its two regime rows are the same run twice. Accepted knowingly: this is **our two-stage
  variant of Lavawizard, not the published Lavawizard.** The original 2019 submission stays on the
  board unmodified as one of the 23 anchor entrants, so both readings are available to a reader.

- **ChromImpute's training sampler moves to `train_chroms`.** `GenerateTrainData` sampled its
  100,000 locations inside the chromosomes it predicts, and `Train` turns those into one predictor
  per (sample, mark) reused everywhere — transferable, so Rule 2's inference exemption does not
  reach it. Exempting it instead (amending §2's rule text to match its own rationale paragraph) was
  rejected: a transferable predictor that names no loci would weaken what a regime id means for
  every method. **This departs from the published recipe** and is recorded in §11 with the rest of
  the per-method departures. `Apply`'s neighbour features are untouched — §2 already names those as
  legitimate per-position inference.

  Rule 1 was checked and is **intact**, by a stronger argument than the audit's: the scored track is
  never written into the run directory at all, because only `biosamples.train` becomes
  `inputinfofile.txt` and that is the one file every jar command is handed.

### Deferred by ruling

- **The noise floor on the new eval panels** (2026-08-29). **Now `t86`, and cheaper than this
  entry assumes: seed 0's `eic.19` checkpoint already exists, so the floor costs ONE more training
  run, not two — and under `--early-stop-epochs` that is hours, not a day.** Two seeds of one
  method, same data;
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
| ~~`t81` retrains~~ | **de-parented 2026-08-30 — it does not ship with `t77`.** Every trainable method re-selected on `V_` under the uniform rule; every existing board row is void (§3.3). Blocked on `t78`, `t79` and `t80`. **RULED 2026-08-30 (PI): the retrains are not a question or a hypothesis — they live in the taskhub only.** So `t81` carries no `hypothesis_refs`, is not an experiment, and stays `implementation`; no null, no verifiables and no `crux close` gate it. What remains is that `implementation`'s merge gate forbids moving a number while a retrain moves every number — answered by `CLAUDE.md`'s own flag-only rule: work that changes no code runs from `main` and opens no PR, and any launch scripts split off as ordinary `implementation` work that holds golden at 0 ULP. Separately, how a retrain's numbers get *accepted* is still open, and §15's deferred noise floor is why: a row from `t81` may go up **unranked** until it lands |
| `t82` board changes | apply the §9 naming across `boards.json`, `help.json`, `app.js`, `index.html` and the PRD; the anchor block; the truth toggle; the eDICE and "no selection" markers |

The §9 naming supersedes the earlier "ENCODE DCC MERGED / DEV-SET / EIC" rename request — the
regime names carry the training and eval loci, which those three labels did not.
