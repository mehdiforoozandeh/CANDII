# RIVALS_PLAN.md — baselines and competitor methods

Status: **decided** (PI, 2026-08-23). This file is the contract for the leaderboard program:
naive baselines for all three heads, retrained competitor methods, and the scoring routes.
It is written so an agent can pick up any task below, implement it, and verify it without
re-deriving the decisions. The decisions themselves are settled — re-open them with the PI,
not in a diff.

Provenance: PI decision session of 2026-08-23 (research briefing artifact "Rivals &
Baselines"); the viability research behind it drew on Max's experiments `001` and `005`
(Nibi, `/scratch/maxwl/epi_imputation/repo/experiments/`). Where this file cites one of
Max's numbers, the number lives in his notebooks, not here.

Vocabulary used throughout (defined once):

- **Dataset 2** — ENCODE's present-day processing of the EIC experiments: what
  `DATA_CANDI_EIC` and our stores hold (counts, `-log10 p` BW tracks, peaks).
- **Dataset 3** — the signal tracks the 2019 challenge itself distributed (Synapse
  `syn17083203`). The published leaderboard lives here. Max's 001 proved the two are
  different quantities and that scores do not translate between them.
- **Rival** — any non-CANDI method that produces prediction tracks.
- **Contributor** — a training biosample that carries the assay being averaged.

---

## 1. PRD — what we are building and why

**Goal.** A leaderboard that places CANDI against (a) naive baselines on all three heads
and (b) real competitor methods, under our own instruments, plus a one-shot placement
against the published EIC results. This is also the artifact that unblocks the
experiment-lane merge gate in `CLAUDE.md` ("blocked on an imputation-methods leaderboard
that does not exist yet").

**Non-goals.** No zero-shot (fully-unseen-cell-type) comparison in this round — that claim
is deferred to the MERGED round, whose split will hold out whole cell types from day one.
No dHICA (dropped: its 128 bp / 131 kb architecture is hardwired; the comparison would be
too soft to buy with a training run). No EPCOT (different task — window-level binary peak
classification needing same-sample accessibility — and no code license). Both stay as
related-work citations.

**Deliverables.**
1. `candi.bench` scores externally-produced prediction tracks (§4).
2. The naive baseline suite, emitted as external tracks and scored by the same entry (§5).
3. Retrained rivals on our EIC data: Avocado (Max's port), ChromImpute (their Java),
   eDICE (our PyTorch reimplementation), Lavawizard (spike → port) (§7).
4. All 23 EIC entrant submissions scored on Dataset-3 truth (§7.5).
5. A feasibility verdict on Enformer Celltyping (§7.6).
6. The challenge tracks staged on Fir, off purge-prone scratch (§8).

**The decision record** (each settled with the PI; the letter codes are used in task whys):

- **A1** Both claims, staged: internal leaderboard first, Dataset-3 placement at the end.
- **A2** Rivals train on EIC now; MERGED later, survivors only.
- **A3** Headline arm is **pval** (nine EIC measures + partition metrics). The count /
  CRPS tier is CANDI's showcase.
- **A4** Internal evaluation runs **both protocols, always**: declared pairs (P1) and a
  genome-wide pass (P2). V pairs during development; B pairs once, at the very end.
- **A4b** Zero-shot deferred to the MERGED round.
- **B1a** A rival's point track becomes a homoscedastic Gaussian; σ per method × assay,
  fitted on V-pair residuals (§6.1).
- **B1b** The count arm compares CANDI against count-native baselines only. No invented
  pval→count depths.
- **B2** Paper-default hyperparameters, one seed; a second seed only for methods that land
  within the noise floor of CANDI (§6.3).
- **B3** DNase-seq stays internal, is excluded from every Dataset-3 row; every table
  reports per-mark with broad and punctate separated; rivals enter the peak tier via
  signal-ranking AUPRC with the coverage-ranking caveat.
- **C** Avocado: vendor Max's port. ChromImpute: their Java, 20-pair pilot first. eDICE:
  PyTorch reimplementation validated on a Roadmap slice. Lavawizard: spike → port.
  Enformer Celltyping: go/no-go spike. Entrants: Dataset-3 side only.
- **D** Distributional baseline = moment-matched NB now; exact empirical-ensemble marginal
  as a follow-up; training-fitted per-assay marginal as the third tier. kNN at k ∈ {1, 5}.
- **E** Rival code lives in a top-level `competitors/` directory (outside `src/candi`,
  never imported by core). One prediction-track format, one bench entry point. The EIC
  round runs on **Fir**; challenge tracks move Nibi→Fir by Globus into `/project`.

---

## 2. The three protocols

**P1 — internal, declared pairs.** Exactly what `candi.bench --store` scores today: the
regime's `eval_pairs`, every 25 bp bin of the regime's `eval_chroms`. Development regime is
`configs/regime.eic_val.json` (51 training cells, 26 T→V pairs, eval chr21). The final
regime is `configs/regime.eic_test.json` (B pairs) and is run **once per method, at the
end** — the same B_ discipline `EVAL.md` already imposes on CANDI runs.

**P2 — internal, genome-wide.** The same pairs and truth, over every chromosome the store
carries, via the bench `--chroms` override. Rivals must therefore emit genome-wide
predictions (Avocado fits per-chromosome genomic factors; ChromImpute and eDICE predict
anywhere by construction). Note two things for readers of P2 numbers: CANDI's current
recipe trains on chr19 only, so for CANDI, P2 is almost entirely held-out sequence; and
position-parameterized rivals see every chromosome in training, so P2 structurally favors
them. That asymmetry is why P1 and P2 are both always reported (A4).

**P3 — placement, Dataset 3.** Once per finalist method: predictions scored directly
against the challenge's own blind-test tracks with Max's 001 vendored scorer (the nine
measures minus `msevar`, which 001 could not reconstruct), plus our partition metrics
ported to that path (§7.5). Never rescale a Dataset-2 score into Dataset-3 space — 005
measured 12–66 % per-experiment error for exactly that move. Broad marks and punctate
marks are reported separately; DNase-seq is excluded from every P3 row (B3; in Dataset 2
it is `read-depth normalized signal`, a different physical quantity).

---

## 3. Roles of the two sides of the repo

- `src/candi/bench/` grows **one** new capability: scoring external tracks (§4). Nothing
  else in core changes. Non-experiment gates apply: `pytest tests/ -q` green and
  `tools/golden.py check <ref.pt>` bit-exact.
- `competitors/` (new, top-level) holds everything else: one subdirectory per method, each
  with its own environment and README, plus the baseline generators. Nothing under
  `competitors/` is importable from `candi`, and `candi` never imports it. The only
  interface between the two worlds is the prediction-track format below.

---

## 4. Spec — the prediction-track contract and the bench external entry

### 4.1 On-disk format (the contract)

One directory per scored track:

```
<pred_root>/
  manifest.json
  <input_biosample>__<target_biosample>__<assay>/
    chr21.npz            # one npz per chromosome covered
    ...
```

- Directory name is the bench `track_key` with `|` replaced by `__` (filesystem-safe);
  `kind` is `impute` and is implied. Biosample names keep their `T_`/`V_`/`B_` prefixes.
- Each `chr*.npz` holds named arrays over the **absolute bin grid** of that chromosome —
  index `i` is the bin starting at `i * 25` bp, length exactly `source.n_bins(chrom)`
  (the store's `floor(chr_len / 25)` grid; the loader asserts the length and refuses a
  mismatch — no silent truncation or padding).
- Recognised arrays (all `float32`; supply what the method has, the scorer detects the
  rest):

  | array | meaning | enables |
  |---|---|---|
  | `signal_mu` | pval-arm point prediction, in `-log10 p` | pval arm: E, P blocks |
  | `signal_sigma` | pval-arm per-bin std, same space | pval arm: gauss_suite (CRPS etc.) |
  | `mu`, `n` | NB count prediction | count arm: E block + nb_suite |
  | `peak_score` | ranking score in [0, 1] | peak tier (B block) |

- External `signal_mu` is **always already in `-log10 p`** — external tracks carry no
  training-space transform, so the entry point scores them with
  `signal_target_transform="none"` and stamps `pred_inversion: "external"` in provenance.
- `manifest.json`: `{method, version, generated_by, date, arms, notes}` — copied verbatim
  into the result's provenance block so a score file is traceable to the code that made it.

### 4.2 The entry point

New module `src/candi/bench/external.py`, CLI:

```
python -m candi.bench.external --store <regime.json> --pred <pred_root> \
    --out <scores.json> [--chroms chr21,...] [--sigma-table <sigma.json>] [--varpool ...]
```

Implementation shape (reuse, do not re-derive):

1. Open the source with `bench.harness.open_source(store=...)` — pairs, truth and grid
   come from the regime exactly as they do for a model run.
2. For every declared pair × assay the prediction root covers, build a
   `bench.harness.TrackRecord`: truth (`counts`, `pval`, `peaks`) read from the source;
   prediction arrays from the npz. `peak_score` falls back to `signal_mu` when absent
   (the same coverage-ranking fallback, recorded the same way: `has_peak_head=False`).
3. `--sigma-table` (B1a, §6.1): a json `{method, fitted_on, sigma: {assay: value}}`. When
   a track has `signal_mu` but no `signal_sigma`, fill the constant σ from the table. No
   table and no sigma → the pval arm carries point blocks only (E, P) and no gauss_suite —
   absent keys, never NaN.
4. Score with `bench.harness.score_track`, aggregate with `macro_mean`, assemble the same
   `{provenance, tracks, per_track, macro, panel, ranking}` shape `run_bench` emits, with
   `provenance.method` from the manifest. A missing declared pair is listed in
   `provenance.missing_tracks` and the run **fails unless** `--allow-missing` is passed —
   the D2 lesson (silent partial panels) applies to external tracks too.

### 4.3 Done when

- Round-trip test: stream CANDI's own predictions to the external format, score both ways,
  every shared key equal to ~1e-6 (float re-order tolerance). This is the acceptance gate:
  the external path is the same instrument, proven, not asserted.
- A malformed root (wrong grid length, unknown biosample, missing declared pair) fails
  loudly with the offending track named.
- `pytest tests/ -q` green; `tools/golden.py check` bit-exact (this is a non-experiment
  lane: it must not move any existing number).

---

## 5. Spec — the naive baseline suite

All baselines are **generators**: `competitors/baselines/` writes prediction roots in the
§4 format; scoring is the external entry's job. All are computed from **training-split
biosamples only** (the regime's `biosamples.train`), and all obey the exclusion rule
below. The EIC challenge's own Average used training+validation — reproduce that variant
only on the P3 side, where comparability to the published `Average` rows is the point.

**Exclusion rule (leakage).** For a target track (pair T_X → V_X or B_X, assay a):
contributors are training cells carrying assay a, **minus every biosample sharing the
target's cell-type suffix** (T_X is excluded when predicting V_X — the bench impute dial
applies the same leave-one-out mask to CANDI's input, `harness._apply_loo_mask`). Emit
`n_contributors` per track; a track with 0 contributors is skipped and listed; tracks with
≤ 2 are flagged in the manifest (`sparse_assays`) and must carry that flag into any table.

### 5.1 Count head

Depth normalization follows `src/candi/reference.py` (which already implements the
depth-free leave-one-out mean for the h5 path — reuse its arithmetic, not necessarily its
storage): every contributor's counts are rescaled to `depth_center`,
`x_j = c_j * 2^(depth_center - d_j)`, where `d_j` is the contributor's log2 depth at DSF 1.

Per bin, with contributor values `x_1..x_k`:

- `m = mean(x_j)`, `v = var(x_j, ddof=1)` (v undefined at k = 1 → Poisson fallback below).
- Predicting for a target track at log2 depth `d_t`: scale `s = 2^(d_t - depth_center)`,
  point prediction `mu = s * m`, spread `V = s^2 * v`.
- **Moment-matched NB** (decision D): choose the NB with mean `mu` and variance `V`:
  `n = mu^2 / (V - mu)` when `V > mu`, else `n = 1e6` (the Poisson limit — the NB cannot
  be under-dispersed, so cross-cell agreement floors at Poisson noise). `p = n / (n + mu)`.
  Write `mu` and `n` arrays; the entry point derives nothing.

The variance choice is pre-registered here: the predicted spread is the **cross-cell
biological spread**, depth-scaled, floored at Poisson. It deliberately omits an extra
target-side counting-noise term; if that floor proves wrong, amend this paragraph first
and the code second.

### 5.2 pval head

- **Plain mean** (the EIC definition): arithmetic mean of `-log10 p` across contributors.
  This is the row that must exist for comparability.
- **arcsinh-mean variant**: mean taken in arcsinh space, `sinh` back. A separate method
  directory, labelled `avg-arcsinh` — never presented as the EIC baseline.
- **Distributional form**: `signal_sigma = std(contributors, ddof=1)` per bin — the
  cross-cell Gaussian marginal. This makes the pval baseline natively heteroscedastic; no
  sigma-table needed.

### 5.3 Peak head

- `peak_score = (# contributors with a peak at the bin) / k` — the fraction itself is the
  AUPRC ranking score (decision D). A `--threshold 0.5` majority variant may be emitted as
  a second root for discrete-decision checks; the fraction root is the baseline.

### 5.4 kNN and marginal tiers

- **kNN average**, k ∈ {1, 5}: per assay, rank contributors by Pearson correlation with
  the *input* cell's track of that assay computed on **training chromosomes only**
  (chr19 under the shipped regime), in arcsinh(`-log10 p`) space; predict the mean over
  the top k (same depth normalization for counts). k = 1 is the BestSingle baseline.
  Correlations use training data only — no eval-chromosome and no target-cell bins.
- **Per-assay marginal** (decision D, third tier): one NB per assay, fitted by moment
  matching over all bins of the **training cells' tracks on the training chromosomes** —
  no eval-chromosome and no target-cell data. One `(mu, n)` constant per assay, written as
  constant arrays. This is the training-fit analogue of bench's oracle `marg_crps` and the
  weakest distributional baseline.
- **Follow-up (separate task):** the exact empirical-ensemble marginal — score CRPS
  against the empirical CDF of the contributors directly. Requires a bench extension
  (ensemble CRPS) and is deliberately decoupled so the suite ships without it.

### 5.5 Done when

- `reference.py`'s L3 depth-free property holds for the generator (adapt
  `tests/test_reference.py::test_L3_reference_is_depth_free` to the store path).
- A synthetic-fixture test: 3 hand-built contributors → hand-computed `mu/n/sigma/
  fraction` per bin, exact match.
- The suite scores end-to-end through §4 on `regime.eic_val.json` and produces the first
  leaderboard json. Sanity anchors: the plain-mean pval baseline must beat the per-assay
  marginal on macro mse; `beats_marginal` for the moment-matched NB baseline should be
  near-universal (it is a strictly richer predictor).

---

## 6. Fairness rules, operationalized

### 6.1 σ-fit for point-only rivals (B1a)

Per method × assay: `sigma^2 = mean over V-pair tracks of (signal_mu - truth)^2`, pooled
over all bins of the P1 eval chromosomes, in `-log10 p` space. Fitted once, written as the
§4.2 sigma-table json with `fitted_on: "regime.eic_val eval_pairs"`. The **B-pair** run
uses the V-fitted table unchanged — that is what makes the quoted CRPS leak-free. V-pair
CRPS for these methods is in-sample for σ and every table that shows it says so.

### 6.2 What each method may see

Rivals train on Dataset-2 EIC signal (`signal_BW_res25`, i.e. `-log10 p`; arcsinh applied
by methods that specify it) for **training-split biosamples only** — the same 51 cells the
regime declares. No V_ or B_ track enters any rival's training, fine-tuning, correlation
table, or calibration. This is the single most checkable fairness property; every
`competitors/<method>/README` must state where its code enforces it.

### 6.3 Seeds and tuning (B2)

Paper-default hyperparameters, one seed, documented in the method README. A second seed
only when a method's macro gap to CANDI is within the applicable noise floor
(`AGENTS.md` §7.2: q19 seed floor, macro CRPS 0.0608 under bench; quote the floor of the
panel being compared). No tuning sweeps for anyone — parity by abstinence.

### 6.4 Reporting invariants (B3)

Every table: per-assay rows first, macro second; broad (H3K27me3, H3K36me3, H3K9me3) and
punctate marks never pooled together without their separate medians beside the pool;
`auprc` always with `peak_base_rate`; CRPS always with `crps_oracle_scaled` + `scale_error`;
a rival's peak-tier row always labelled coverage-ranking (no rival has a peak head).

---

## 7. Per-method playbooks

Each lives in `competitors/<name>/` with: `README.md` (provenance, environment, exact
training command, §6.2 statement), an env recipe, a `predict_*` script that writes §4
roots, and a SLURM script. Compute runs on Fir (`slurm-hpc` skill; account `def-maxwl`).

### 7.1 Avocado — vendor Max's port

- Source: `nibi:/scratch/maxwl/epi_imputation/repo/experiments/005_2026-07-29_avocado_cross_dataset/scripts/avocado.py`
  (plus its train/predict drivers). Vendor byte-identical first, record the md5 and the
  source path in the README, then adapt I/O to our store/npz — never silently edit the
  model. Coordinate with Max; his 005 notebook is the validation record (arm A: median MSE
  ratio 1.152 vs published `Avocado_p0`, per-experiment Pearson 0.999).
- Architecture (from 005): 32 cell factors, 256 assay factors, multi-scale genomic factors
  (25 @ 25 bp, 40 @ 250 bp, 45 @ 5 kbp), 2048–2048–1 ReLU head; arcsinh target, `sinh`
  back, clip at 0. Two-stage: joint fit on chr20, then per-chromosome genomic-factor fits
  with shared weights frozen — genome-wide prediction is 23 embarrassingly parallel array
  tasks. Budget: 005 used 120/60 epochs and measured that **half is enough** (held-out
  minimum near epoch 30); use 60/30. ~86 GPU-h at the old budget, so plan ~45.
- Train on our 51 training cells' pval tracks (arcsinh); predict all P1/P2 tracks; σ-table
  per §6.1.
- Done when: training curves reproduce 005's convergence shape (held-out entry-MSE
  plateau), P1 scores exist for all 26 pairs, and the README records the vendored md5s.

### 7.2 ChromImpute — their Java, as published

- Code: `github.com/jernst98/ChromImpute` (GPL-2.0, Java ≥1.6); the manual PDF is the
  authority for the seven commands (`Convert`, `ComputeGlobalDist`, `GenerateTrainData`,
  `Train`, `Apply`, `Eval`, `ExportToChromHMM`).
- Inputs: our pval tracks converted npz → bedgraph (25 bp bins; write a converter in
  `competitors/chromimpute/`). Training-split cells only in `inputinfofile`.
- **Pilot gate first**: 20 representative (cell, mark) pairs on chr21 + one genome-wide
  pair, wall-clock and RAM measured, before the full grid is submitted. The full P1 grid
  is one trained ensemble per declared (target cell, assay) — 26 pairs × their assays —
  each a SLURM array task.
- Prediction: `Apply` genome-wide per target; convert output back to §4 npz.
- Done when: the pilot memo (cost/pair) is in the task output, P1 scores exist, and a
  spot-check reproduces the manual's example-dataset numbers on their chr21 sample data
  (the no-retraining correctness check).

### 7.3 eDICE — PyTorch reimplementation

- Reference: Hawkins-Hooker et al., Nat Commun 2023; code `github.com/alex-hh/eDICE`
  (MIT, TF 2.11 — read, do not run). Model: per-bin factorized attention — cell embedding
  (256) and assay embedding refined by 4-head self-attention over the observed tracks at
  that bin, MLP decoder (2×2048, dropout 0.3), arcsinh target, 50 epochs, Adam 3e-4.
- **Validation gate before our data**: reproduce their packaged Roadmap demo (ships in
  their repo) with our reimplementation; land within their demo's reported metrics — the
  same discipline 005 used for Avocado. Record the comparison in the README.
- Then train on our 51 training cells. eDICE is transductive — a target needs a learned
  cell embedding, and V_/B_ cells have none. Handling, pre-registered: for a target
  V_X/B_X, use the **paired T_X cell's embedding** (same cell type; the pair is the
  regime's own declaration). This is the transductive analogue of the impute dial and must
  be stated in every eDICE row's caveat. σ-table per §6.1.
- Done when: Roadmap-demo validation recorded; P1 scores exist for all 26 pairs.

### 7.4 Lavawizard / Guacamole — spike, then port

- Their submission repo `github.com/ccchang0111/ENCODE_imputation_2019` (Keras, 2019,
  includes 23 model files). **Spike (time-boxed, 1 day):** does it install and predict
  today on Fir? Either way, the port target is the model description in the repo +
  challenge Table 1 (deep tensor factorization, arcsinh, average-signal feature).
- Anchor: their submitted Synapse tracks (§7.5) — the port, retrained on the challenge's
  own training data, should approach their published rows before it is retrained on ours.
- Done when: spike memo; port anchored; P1 scores exist.

### 7.5 The 23 entrant submissions — score, don't reimplement

- Download `syn17083203` `submissions_round2/` (needs a Synapse token with download scope;
  any registered account; `~/.synapse_pat` pattern from Max's 001) to Fir. Check size
  before pulling — budget ~1 TB scale; coordinate `/project` quota with Max.
- Score every submission against the Dataset-3 blind truth with the 001 vendored scorer
  (`fir_score.py`, `eic_metrics.py`, `fir_tracks.py` — copy byte-identical, md5s recorded,
  exactly as 005 did; that code reproduced published rows at 2e-5 worst-case).
- Port the bench P-block (partition metrics) onto that path: same partition definitions as
  `src/candi/bench/partitions.py`, applied to bigwig-loaded arrays. Validate by scoring
  the published Average baseline tracks and checking the P-block's own sanity properties.
- Output: one placement table, 23 entrants + Average + Avocado_p0 + our methods' P3 rows.
  These tracks never enter internal (Dataset-2) tables — 005's translation result is the
  reason and is cited wherever someone asks.
- Done when: the table exists with per-assay rows and the msevar-excluded caveat, and our
  reproduction of the published `Average` rows matches 001's (sanity re-run, not new work).

### 7.6 Enformer Celltyping — feasibility spike only

- Paper: Nat Commun 2024 (accessibility + DNA → histone marks in unseen cell types); code
  `github.com/neurogenomics/EnformerCelltyping`. Time-box: 2 days. Go/no-go gates, each a
  yes/no in the memo: (1) installs and predicts with shipped weights today; (2) retraining
  cost on our EIC estimated ≤ Avocado's budget order; (3) output grid reconcilable with
  25 bp without architecture surgery; (4) accepts hg38 inputs. All four yes → it gets a
  §7-style playbook of its own; any no → related-work citation, memo says which gate
  failed.

---

## 8. Data staging (the one move that touches other people's storage)

1. Globus Nibi→Fir (managed endpoints `alliancecan#nibi` → `alliancecan#fir-globus`, sync
   mode): `/scratch/maxwl/epi_imputation/data/synapse/` (254 GB: `blind_truth`,
   `train_plus_val`, `training_data`, `validation_data`) → a Fir `/project/def-maxwl/...`
   location agreed with Max (group quota is charged by group ownership — coordinate before
   moving, and `rsync` habits from the hpc-workspace skill apply: `--no-g --no-p`).
   Landing in `/project` retires the 60-day scratch-purge risk; tell Max either way.
2. Sync Max's 001 scorer + 005 Avocado artifacts (scripts are small — rsync; checkpoints
   via Globus if we want his trained weights rather than retraining).
3. Download `submissions_round2/` (§7.5) directly to Fir.
4. Record every landed path in the task output; nothing in this plan may point at Nibi
   scratch afterwards.

---

## 9. Task registry and verification

The taskhub carries one task per row below (ids assigned by the engine; titles match).
Verification column = what "done" must link.

| work | blocked by | verify by |
|---|---|---|
| stage challenge tracks + Max's artifacts on Fir | — | landed paths + sizes recorded; Max informed |
| bench external-track entry | — | §4.3 round-trip test + green suite + golden |
| naive baseline suite | external entry | §5.5 fixtures + first leaderboard json |
| Avocado retrain (Max's port) | external entry | §7.1 gates |
| ChromImpute pilot + full run | external entry | §7.2 gates |
| eDICE reimplementation + retrain | external entry | §7.3 gates (Roadmap demo first) |
| Lavawizard spike + port + retrain | external entry | §7.4 gates |
| entrant submissions scored on Dataset 3 | staging | §7.5 table + Average sanity match |
| Enformer Celltyping spike | — | §7.6 four-gate memo |
| ensemble-CRPS bench extension | baseline suite | scored empirical marginal ≈ NB compression sanity |

Branching follows `CLAUDE.md`: each row is a whole issue → its own branch → PR, pushed at
creation. The bench entry and the baseline suite are non-experiment lanes (tests + golden
gate). Retraining rows produce numbers and land under the leaderboard this plan itself
creates; until the baseline suite's json exists, no rival row is quotable.
