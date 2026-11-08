# Mini-CANDI design inputs from the CAC testbed — DRAFT (2026-10-31)

Purpose: inputs for a first-principles, staged mini-CANDI prototype (few assays, few
covariates, few parameters, then scale). Each input names its evidence and confidence.
Confidence words follow the record: strong / moderate / weak; "exploratory" = not
pre-registered. Status: draft for PI correction. Not yet a deck.

Scope limit, PI ruling 2026-10-31: the testbed trained covariate correction ALONE. Its
negative results on model form do not carry over to joint training with imputation and
reconstruction. Data is a visible bottleneck; model form and objective stay open for CANDI
(question q12).

PI rulings 2026-10-31: (a) count space only. Mini-CANDI and all remaining CAC work predict counts;
no direct p-value head; the p-value track is derived from predicted counts by formula (2.4).
Control-side knobs (1.4) and the p-explosion diagnosis (5.2, experiment 6) drop out of scope.
(b) Routing through a base (2.5) and count-then-formula (2.4) accepted as design inputs; routing
stays a hypothesis for the prototype. (c) Costly reprocessing / augmentation (experiment 4,
section 7 FASTQ route) deferred until its return is clear. (d) Deck and short paper come after
the section-6 experiments are run.

## 1. Starting covariates

| # | Input | Evidence | Confidence |
|---|---|---|---|
| 1.1 | Start with depth. It moves peak height most (3.0 log2 at 3.75M, histone) and is the only knob with exact same-material pairs in the corpus. | f24, DATA.md | strong |
| 1.2 | Add run type second. It is the knob behind the EIC failure (dedup rule differs SE vs PE), and in DNase PE raises peaks most. The corpus has no SE/PE pair of one experiment; on the shipped 8-assay panel run type is fully set by assay + read length (0.000 bits), so it cannot be learned from that panel at all. | wiki/encode-imputation-challenge.md, f24, DATA.md | strong (EIC cause, paper); moderate (size in our data) |
| 1.3 | Read length third, but only with a position-aware input: crops act through mappability and repeats. | f24, f26 | strong |
| 1.4 | Control-side knobs (control scaling, identity, depth) carry the most transferable signal for the p-value track; relevant only if mini-CANDI predicts p. | f10, f11 | weak (readings only) |
| 1.5 | Defer MAPQ filter (acts through mappability, +0.45 log2 in the lowest decile) and fragment extension (p-track unstable on unseen pairs). | f26, f11 | strong / weak |

## 2. Model form

| # | Input | Evidence | Confidence |
|---|---|---|---|
| 2.1 | Keep depth as a learned covariate in the prototype, as a sanity probe (PI ruling 2026-10-31): the depth law is known exactly, so how well the network follows it measures how much correction it learns, cheaply. The closed-form offset (log mean = log depth + network output, depth hidden from the network) is parked for the publication version, once the correction is shown to work. Evidence the offset is cleaner: the model with no covariates nearly meets the law (0.054–0.135); the one that reads them bends it. | f17, PI 2026-10-31 | strong (evidence); design = PI ruling |
| 2.2 | A per-bin monotone curve captures most of the covariate effect on counts; affine fails on counts, CNN/kernel add little. Size the correction module small. | f12–f15, PI q4 "mostly answered" | weak (exploratory) |
| 2.3 | More capacity, width (4x g, 2x f), Poisson-mean loss, removing the +1 pseudocount, encoder-decoder and exact-inverse composable forms did NOT fix transfer to unseen covariate pairs — when the correction trains alone. | f1–f4, f20, f21, f23 | strong (alone only; see q12) |
| 2.4 | Predict counts (NB); derive the p-value track by formula. The fragment-window formula nearly matches the best learned map (0.0865 vs 0.0798 log CRPS). Train any count→p map on true counts. | f18 | strong (one track, one seed) |
| 2.5 | Route corrections through a base measurement: chaining two trained legs through the base meets the depth law (0.07–0.16) where direct unseen arm→arm misses (0.29–0.34). | f19, f23 | moderate |

## 3. Training pairs

| # | Input | Evidence | Confidence |
|---|---|---|---|
| 3.1 | Base↔arm pairs alone do not teach a rule that transfers to unseen arm→arm combinations. | f19, f23 | moderate |
| 3.2 | Direct arm→arm training teaches scale, not distribution (held-out CRPS gap closed −0.53; PI: not supported). Denser arm→arm gives a marginal edge (0.428 vs 0.435 through-base). | f5, f27 | moderate / weak (exploratory) |
| 3.3 | The corpus pairs an experiment only with thinned copies of itself; any non-depth correction needs new pairs. The testbed's 130 single-knob products (7 EIC tracks) are the only real ones today. | q8, t128 round-trip collapse, results/t112/MANIFEST.tsv | strong (structural) |
| 3.4 | Resampling noise is level-driven, not regional: signal level explains 0.18–0.38 of per-bin disagreement; GC, mappability, repeats add ≤0.0003 R². Peak height barely moves under resampling (≤0.02 log2); 1 kb shape moves a lot. | f25 | strong |

## 4. Gates the prototype must pass before scaling

| # | Gate | Why | Evidence |
|---|---|---|---|
| 4.1 | Scrambled-covariate twin: same model, covariates shuffled. | Separates "uses covariates" from "fits better". | f8, f12–f15 |
| 4.2 | Margin above 2× seed wobble. | Seed alone moves pooled CRPS 0.1195. | AGENTS.md §7.2 |
| 4.3 | Depth law on never-trained pairs (closed-form check). | The single test every design failed. | f17, f19 |
| 4.4 | Held-out covariate combinations, not only held-out cell types. | Trained-pair fit hides the transfer failure. | f8, f9 |
| 4.5 | Outcome-neutral controls (positive depth control, negative thinning blank). | Caught the covariate-swap probe as void. | f22 |
| 4.6 | Report peak height and shape separately against the resampling floor. | Shape rankings were mostly noise. | f24, f25 |

## 5. Open risks for joint training

| # | Risk | Evidence |
|---|---|---|
| 5.1 | Joint training with masked-assay imputation and reconstruction may change which task the gradients favour, so testbed negatives on model form may not hold. | PI 2026-10-31; q12 |
| 5.2 | The p-value track explodes in designs B–D on some cells; cause never diagnosed. | f16 (PI deferred) |
| 5.3 | Whether a trained CANDI uses read length / run type correctly is unknown: the covariate-swap probe is void. | f22, h26 |
| 5.4 | EIC blind cell types are poorly characterised; covariate correction does not address cross-cell-type generalisation. | wiki/encode-imputation-challenge.md |

## 6. Testbed experiments still worth running (ranked for mini-CANDI)

| Rank | Experiment | Design input it settles | Cost (rough) | Status |
|---|---|---|---|---|
| 1 | Joint-objective test: train g→f on several assays with a masked-assay imputation term added, and compare unseen-pair transfer (depth law, held-out CRPS) with correction trained alone. | 5.1 (q12): whether testbed negatives on model form hold under joint training | testbed scale, a few GPU-hours per arm | new |
| 2 (deferred) | Remove depth from g's input; keep the closed-form offset. Pre-registered bar: worst-pair |ratio−1| ≤ 0.10 on all seeds, refuted if > 0.5. | 2.1: the parked closed-form offset; deferred with it | rerun of the depth-offset grid, design D | h24, designed, never built |
| 3 | Built-in pivot: train every correction as source→base→target with one fixed reference covariate set, then score unseen arm→arm. | 2.5: whether the working chaining fix survives as a single trained model | testbed scale | new; chaining evidence f19/f23 |
| 4 | Run-type effect size on more tracks: real SE vs PE products beyond the 7 testbed tracks. | 1.2: how big run type is, per assay | CPU analysis if products exist | candidate source = EIC blind tracks reprocessed SE on Nibi; PI barred blind-panel arrays from h26, so needs PI ruling |
| 5 | Sequence- or mappability-aware generator for MAPQ and read-length crops. | 1.3, 1.5 | larger build | h16, parked 2026-09-25 |
| 6 | Diagnose p-track explosions in designs B–D. | 5.2 | small | deferred by PI; matters only if mini-CANDI predicts p directly (2.4 says derive p) |

Not ranked: rerun of the covariate-swap probe on trained CANDI (h26). It probes the old
architecture, which the redesign replaces; PI-gated.

## 7. Counterfactual pairs: what generation costs (from the testbed build)

Two cost routes, both measured on the 7-track build (130 products; results/t112/MANIFEST.md, tasks t111, t112, t117):

| Route | Knobs | Input needed | Cost recorded |
|---|---|---|---|
| BAM-level | depth thinning, antibody proxy (treatment/control mix), control scaling, control identity, control depth, fragment extension, pseudoreplicates | filtered BAM (+ control BAM) | ~1 h per product, SLURM array |
| FASTQ re-run (ENCODE pipeline) | run type SE↔PE, dedup on/off, read-length crop, MAPQ filter | FASTQ (PE FASTQ for SE↔PE) | median 13.4 h, max 30.6 h per product |

Constraints: depth and read length only go down. SE↔PE needs PE raw reads. Control knobs move
only the p-value track. Storage per 7-track build: ~1 TB FASTQ transient, ~1 TB BAM.

Corpus fact: the CANDI corpora (EIC 363 experiments / 89 biosamples; merged 3026 tracks / 361
biosamples) are signal-level only — built from released bigWigs. No FASTQ or BAM is recorded
for them. So, from the corpus as built, only depth pairs can be made (count thinning). Every
other knob needs raw reads fetched on the cluster first.

Open data items: MAPQ arms of the DNase track are byte-identical to base (t119, open). The
"41 of 48 EIC blind tracks reprocessed SE with BAMs" count lives in the EIC_REPRO tree on
the cluster; not verified from this repo.
