# eDICE — PyTorch reimplementation

`RIVALS_PLAN.md` §7.3. eDICE is a per-bin factorised-attention imputer: no positional parameters at
all, so one 6 M-parameter model predicts anywhere in the genome. That is what makes it worth
retraining against CANDI, and it is also why it needs the caveat in §"Transductive" below.

**Nothing here is quotable yet.** Rival numbers become quotable when the t49 leaderboard exists.
Until then this file records the method, not a result.

## Provenance

| what | where |
|---|---|
| paper | Hawkins-Hooker, Visonà, Narendra, Rojas-Carulla, Schölkopf, Schweikert. *Getting personal with epigenetics: towards individual-specific epigenomic imputation with machine learning.* Nat Commun 14:4750 (2023). |
| reference code | `github.com/alex-hh/eDICE` @ `f8cf4f9b8554bf5225c378d05fe7d397a452f23d` (2024-05-31), MIT, TensorFlow 2.11 |
| local clone | `fir:/project/def-maxwl/mforooz/rivals_src/eDICE` |
| Roadmap data | Edmond `doi:10.17617/3.VKEFB6`, file `roadmap_tracks_shuffled.h5` (4,125,156,368 B, md5 `e28b9c19b3843895742c9b34373dd0bf`), CC0 |
| local data | `fir:/project/def-maxwl/mforooz/rivals_src/edice_data/` |

**Read, never run** (§7.3). No TensorFlow was executed. The reference files this reimplementation
was written from, with md5s so a later reader can tell whether they changed:

```
971bcc5f40a64b86d30b120b571e75c7  edice/models/predictors.py
0f5c969f8d2540687207ea0ba9b7ee77  edice/models/embedders.py
0b225852c5a981236ab455082033d665  edice/models/layers.py
2b0ba793a24d2e3d055936b4d6db57c5  edice/models/attention.py
0ecc1e631112cedef301782a47259f56  edice/data_loaders/data_generators.py
e74bdd2805736f9beeed6841911087a9  edice/models/metrics.py
04c922710045ac09ff55fefa6b417d51  edice/utils/transforms.py
f8dc09bc50410a0850e4f6de5584d96c  scripts/train_roadmap.py
b6ef77a50838cce1aa812d52d8710e03  sample_data/roadmap/idmap.json
ce35c0c11c8c0dd31786d6ca3402a30e  sample_data/roadmap/predictd_splits.json
```

## Environment

Fir, `/project/def-maxwl/mforooz/candi_venv` (torch 2.6.0, h5py 3.12.0, numpy 1.26.4). No new
dependency: the reimplementation needs torch, numpy and h5py, all of which the CANDI environment
already carries. Nothing under `competitors/` is importable from `candi`; `run_eic.py` and
`fit_sigma.py` read `candi.store` and `candi.bench.harness` in the other direction, which is how the
emitted track set is guaranteed to be the set the scorer demands.

## The model

One example is one 25 bp bin. Within a bin the model sees a bag of observed (cell, assay, value)
triples and is asked for values at a set of (cell, assay) targets.

Two mirror towers. One treats **cells** as nodes and assays as their features; the other treats
**assays** as nodes and cells as their features. Each scatters the bin's observations into a dense
node × feature matrix, divides each node's row by that node's observation count, projects to 256
with a ReLU Dense, adds a learned per-node embedding, and refines with one 4-head cross-attention
block that queries only the nodes the targets name. The decoder is an MLP over the concatenated pair
(512 → 2048 → 2048 → 1, dropout 0.3). Target is `arcsinh(-log10 p)`; loss is MSE in that space;
Adam at 3e-4 for 50 epochs; 120 tracks masked per bin; batch 256 bins.

At the Roadmap settings this is **5,985,025 parameters** against the paper's "~ 6 M"
(Supplementary Table 1) — checked in `tests/test_edice_model.py::test_roadmap_parameter_count_matches_the_paper`.

### Where the paper and the code could be read differently

Every row here follows the **code**, and every row is marked `# FIDELITY:` at the line in
`edice_torch/model.py` that implements it.

| | the paper says | the code does | we follow | why it matters |
|---|---|---|---|---|
| layer norm | "transformer" | none — the Roadmap config pins `layer_norm_type=None` and selects `NoLNTransformerBlock` (`layers.py:107`) | code | a residual stack with nothing normalising it trains differently; adding LN would be a silent improvement on the thing we are measuring |
| attention mask | not discussed | `logits += mask * -1e9` (`attention.py:46`), not `-inf` | code | with `-inf`, a bin where every node is empty gives NaN; with `-1e9` it gives a finite uniform average. The Roadmap h5 contains such bins |
| node pooling | "conditioning on the observed tracks" | each node's feature row is DIVIDED by that node's observation count (`layers.py:100`) — a mean, not a sum | code | a cell with 25 assays and a cell with 1 would otherwise enter the tower at wildly different scales |
| initialisation | not stated | Keras defaults: `glorot_uniform` kernels, zero biases, `RandomUniform(-0.05, 0.05)` embeddings | code | `nn.Linear`'s kaiming default is ~1.7× wider on the decoder and `nn.Embedding`'s N(0,1) is 20× wider than Keras' — at 6 M parameters that is not a rounding difference |
| masked tracks per bin | "120 randomly selected tracks" | `n_targets=120` are the TARGETS; the remaining 912 are supports, re-drawn per bin per epoch | both, consistent | — |
| metric definition | "MSE", "Corr" | per-track state, averaged over tracks only in `result()`, after `sinh` inverts both truth and prediction (`base.py:317`) | code | mean-of-per-track-MSE ≠ pooled MSE, and raw ≠ arcsinh space. `edice_torch/metrics.py` reports **both** spaces so the gate comparison cannot be accused of picking the flattering one |
| attention scale | not stated | `1/sqrt(depth)` with `depth = d_model // heads = 64`, applied after the head split | code | — |
| which weights are scored | not stated | `train_roadmap.py:132` predicts from the **in-memory model after the last epoch**. Its `ModelCheckpoint(save_best_only, monitor="val_loss")` writes a best-on-val checkpoint that the prediction call then does not load | code — final epoch | scoring a best-on-val checkpoint instead would be model selection on a held-out panel, which is not what produced the published numbers |

One deliberate **departure**, in the driver rather than the model: `run_roadmap.py` watches the
TARGET panel each epoch, where the reference watches the `val` split. For the gate the supports are
`train ∪ val`, so the reference's monitor would be reporting error on tracks that are inside its own
training panel. Nothing selects on either curve — one seed, paper defaults, final-epoch weights — so
the curve is a diagnostic and the change cannot move the gate.

`edice_torch/model.py` also carries `SelfTransformerBlock`, unused at `n_attn_layers=1` but present
because the reference stacks `n_attn_layers - 1` of them before the cross block.

## Validation gate — **PASSED** (job `56704603`, 2026-08-25)

§7.3 requires the Roadmap reproduction to land before our EIC training starts. The target is the
eDICE row of **Supplementary Table 2**, "Performance metrics for the imputation of the 203 test
tracks on chromosome 21":

| | GW Corr | MSE Global | AUPRC MACS | AUROC MACS |
|---|---|---|---|---|
| AVG | 0.563 ± 0.02 | 0.159 ± 0.008 | 0.45 ± 0.029 | 0.931 ± 0.006 |
| Avocado | 0.668 ± 0.022 | 0.108 ± 0.005 | 0.593 ± 0.033 | 0.956 ± 0.005 |
| ChromImpute | 0.697 ± 0.019 | 0.119 ± 0.005 | 0.625 ± 0.032 | 0.967 ± 0.004 |
| PREDICTD | 0.669 ± 0.02 | 0.106 ± 0.005 | 0.592 ± 0.032 | 0.96 ± 0.005 |
| **eDICE (published)** | **0.735 ± 0.018** | **0.091 ± 0.005** | 0.651 ± 0.031 | 0.969 ± 0.004 |
| **eDICE (ours)** | **0.7459 ± 0.0089** | **0.0804 ± 0.0019** | n/a | n/a |

`± ` is the s.e.m. over the 203 test tracks — mean of PER-TRACK values, which is what the reference's
metric objects accumulate and what `metrics.summarise` returns. Ours is the **final-epoch** model on
all 997,373 bins, `train ∪ val` as supports, one seed (211), paper defaults throughout.

### The verdict

| | ours | published | difference |
|---|---|---|---|
| GW Corr | 0.7459 ± 0.0089 | 0.735 ± 0.018 | **+0.011 — 0.5 combined s.e.m.** |
| MSE Global | 0.0804 ± 0.0019 | 0.091 ± 0.005 | **−0.011 — 2.0 combined s.e.m.** |

Correlation reproduces within half a standard error. MSE is marginally *better* than published, by
about two combined standard errors. **The gate is passed**: a reimplementation that reproduces one
metric to within noise and beats the other slightly is a faithful one, not a broken one. The
plausible causes of the small MSE edge are the Keras-faithful initialisation, and that a single
published run carries its own seed noise — the paper reports no seed spread to compare against.

### Which space is Supplementary Table 2 in? — **arcsinh, both metrics**

The paper does not say, and its code says something the table contradicts: `base.py::get_raw_metrics`
`sinh`-inverts both truth and prediction, so the released metric objects report **raw** `-log10 p`.
Both runs answer it, and the full run answers it cleanly:

| | MSE | Corr |
|---|---|---|
| smoke (10 k bins), raw | 16.008 ± 4.797 | 0.7155 ± 0.0089 |
| smoke (10 k bins), arcsinh | 0.1050 ± 0.0032 | 0.6859 ± 0.0083 |
| **gate (997 k bins), raw** | **6.786 ± 1.377** | **0.7791 ± 0.0094** |
| **gate (997 k bins), arcsinh** | **0.0804 ± 0.0019** | **0.7459 ± 0.0089** |
| published | 0.091 ± 0.005 | 0.735 ± 0.018 |

Raw MSE is **75× the published value**; arcsinh is within 12 %. And where the smoke run's
correlation was misleading — raw 0.716 looked closer to 0.735 than arcsinh 0.686 — the gate reverses
it: arcsinh 0.746 sits half a s.e.m. from published while raw 0.779 sits 2.2 away. **Both** columns
of Supplementary Table 2 are in arcsinh space, so that table cannot have come from the metric
objects in the released code.

Two independent checks agree. Their AVG baseline at 0.159 raw MSE is impossible against `-log10 p`
tracks whose peaks reach the hundreds; and their Fg MSE of 1.041 is far too small for enriched bins
on that scale. **This is why `gate.json` records both spaces for both metrics** — had it stored only
the reading the released code implies, the gate would have appeared to fail by 75×.

**Two columns, not eight.** The Fg/Bg and MACS-classification columns of Supplementary Table 2 need
per-track MACS2 peak calls on Roadmap, and the Edmond deposit ships signal only. The gate is
therefore on the two genome-wide columns and this file says so rather than quietly reporting half a
table. That is enough to catch a wrong reimplementation: a model with the wrong masking, the wrong
pooling or the wrong initialisation does not land near 0.735/0.091 by accident.

**No TensorFlow reference run.** §7.3 says read their code, do not run it, so the comparison is
against the PUBLISHED numbers, not against a re-execution — the same discipline Max's 005 used for
Avocado (arm A vs the published `Avocado_p0` rows). The cost is that a discrepancy cannot be split
between "our reimplementation differs" and "their released code differs from their paper".

### Runs

| run | what | job | status |
|---|---|---|---|
| `sample` | the repo's packaged 10 k-bin chr21 sample, 20 epochs, `train` split. A **smoke** run: nothing is published for it to hit | `56703977` | **done**, rc=0, 10.2 s/epoch |
| `full` | `roadmap_tracks_shuffled.h5`, 997,373 bins, PredictD splits, supports `train ∪ val`, 50 epochs. **This is the gate** | `56704603` | **PASSED**, rc=0, 14:42:02, ~1007 s/epoch |

```bash
mkdir -p slurm-logs
MODE=sample sbatch competitors/edice/slurm/roadmap_gate.sh
MODE=full   sbatch --time=24:00:00 --mem=48G competitors/edice/slurm/roadmap_gate.sh
```

Each writes `gate.json` (published vs ours, both spaces, per-epoch history), `test_preds.npz`,
`train_end.pt` (the scored weights) and `last_epoch.pt` (crash insurance, overwritten in place, no
best-so-far logic) under `/project/def-maxwl/mforooz/rivals_src/edice_runs/<mode>/`.

**Cost.** The smoke run did 40 batches in 10.2 s = 0.255 s/batch on one `1g.10gb` MIG slice,
projecting 3,896 batches/epoch × 50 epochs to ≈ 14–15 h. The gate then took **14:42:02** at
~1007 s/epoch — the projection was good to within 2 %. Peak RSS 9.6 GB against a 48 GB ask. This is
why the per-epoch diagnostic pass runs on 50 k bins rather than all 997,373: at full width it would
have added roughly four hours for a curve nothing selects on.

## Fairness (§6.2) — where the 51-cell rule is enforced

**`eic_panel.py::training_panel`**, and nowhere else. It builds the panel from
`regime["biosamples"]["train"]` — the 51 `T_` cells of `configs/regime.eic_val.json` — and opens no
other biosample. Every consumer takes its columns from that one object: the training matrix, the
support panel at prediction time, and the σ fit. `assert_no_eval_leakage(panel)` re-checks the
property and is called at the top of `train`, `predict` and `panel`, so it is a test rather than a
promise.

Two further properties, measured on the real store (`run_eic.py panel`, 2026-08-25):

* the training panel is **267 pval tracks over 51 cells** (1–25 assays per cell);
* the declared target set is **45 tracks over 26 pairs**, and the
  **support/target overlap is 0** — the assay being imputed is by construction absent from the
  prompting `T_` cell (`harness.StoreSource.targets`: an assay the target cell has and the input
  cell does not), so it cannot be copied off the support panel.

No `V_` or `B_` track enters training, the support panel, fine-tuning, or the σ table. The σ table
is fitted on V-pair residuals and the B-pair run reuses it unchanged (§6.1), which makes B-pair CRPS
leak-free and V-pair CRPS **in-sample for σ** — every table showing the latter must say so.

## Transductive (§7.3) — the caveat that travels with every row

eDICE learns one embedding per cell id. A `V_X`/`B_X` cell has none: it was never in training, by
construction. **Pre-registered handling:** a target on `V_X` is queried with the paired `T_X` cell's
embedding, the regime's own declaration that they are the same cell type. This is the transductive
analogue of the impute dial — the model is told which cell type to answer for, not which experiment.

`eic_panel.py::requested_tracks` is where the substitution happens; the caveat string is copied
verbatim into every prediction root's `manifest.json`, so it reaches `provenance` and cannot be lost
between the run and the table.

## Seeds and tuning (§6.3)

Paper defaults, one seed (`211`, the reference's own). No sweeps. A second seed only if the macro
gap to CANDI falls inside the applicable noise floor — `AGENTS.md` §7.2, quoted with the panel being
compared.

### Masking rate on our EIC — **decided (PI, 2026-08-25): `--n-targets 31`**

"Paper defaults" (§6.3) does not settle how many tracks to mask per bin here, and the two readings
are far apart:

* eDICE masked **120 of Roadmap's 1032** tracks per bin — **11.6 %**;
* our EIC panel holds **267** tracks, so the same *absolute* 120 masks **45 %**, while the same
  *rate* is **31**.

**The PI chose the rate: 31.** Rationale: it preserves the training-signal proportion of the
published setup. The absolute 120 would mask 45 % of every bin, making our variant structurally
harder than the one the paper reports — so a weak eDICE row would then be evidence about our
masking choice, not about eDICE.

This is a pre-registered departure from the paper's literal number, and it is **carried as a caveat
wherever an eDICE row appears**, next to the transductive caveat:

> eDICE masks 31 of 267 tracks per bin — the paper's 11.6 % rate, not its absolute 120, which would
> mask 45 % of our smaller panel.

`--n-targets` deliberately stays **required** in both `run_eic.py train` and `slurm/eic_train.sh`.
Now that the value is settled the flag could carry a default, but keeping it explicit puts the
number on the launch line and into the SLURM log, where the record of what was actually run lives.

A second, smaller one: `run_eic.py train --train-chroms` defaults to the regime's `train_chroms`
(chr19), matching CANDI's own recipe. eDICE's own recipe trains on every bin it has. The choice is
not a leakage question — eDICE has no positional parameters, so P1 and P2 are equally out-of-sample
for it in the only sense that exists for this architecture — but it is a budget question, and it is
recorded here rather than assumed.

## Arms this method fills

`signal_mu` only, in `-log10 p`, clipped at 0. eDICE has **no count head** — B1b forbids inventing a
read depth to manufacture one — and **no peak head**, so `candi.bench.external` records the
coverage-ranking fallback and every peak-tier row is labelled as such (B3). The forecast
distribution arrives from the §6.1 σ-table, not from the model.

## Checkpoint selection on `V_` (§5) — built 2026-08-31, PI ruling

`run_eic.py train` used to run `--epochs` and keep the last state. It now selects, by the rule
`plan/BENCHMARK_DESIGN.md` §5 asks of every trainable method:

- it derives a **`V_`-only regime** from the one it was given (26 of the 38 declared pairs; the 12
  `B_` pairs are dropped before any track is opened), the same derivation
  `slurm/t81_train_candi.sh` performs for CANDI. `Selector.__init__` asserts no `B_` target is
  reachable — the PI's 2026-08-31 ruling is that `B_` is never *read*, not merely never ranked;
- every `--eval-every` epochs it writes that panel to a §4.1 prediction root and scores it with
  **`candi.bench.external.score_external`** — the same instrument that produces a board row.
  eDICE's own `edice_torch/metrics.py` is *not* used: it is far cheaper, and selecting on it would
  make the rule non-uniform, which is the only thing §5 asks for;
- `model.best.pt` is written **the moment the metric improves**, so a walltime kill still leaves a
  validly selected checkpoint. `model.selected.pt` (weights + vocabulary, the scorable object) is
  written at the end; `model.pt` is always the last epoch and is **not** what gets scored;
- `--early-stop-epochs` counts **epochs**, so its resolution is `--eval-every`.

**The selection key is not CANDI's.** CANDI selects on count-arm `crps`. eDICE emits `signal_mu`
and no `signal_sigma`, so `score_external` records a point-only track and computes no
distributional key at all — and B1b forbids inventing a read depth to give it a count arm. So the
panel, the instrument, the scope and the cadence are uniform, and the **key is not**: eDICE selects
on macro pval `mse` (`--select-metric`, which offers only keys the instrument can produce for a
point track). **This is a real gap in §5's "same rule" and it is for the PI, not for this file.**

**It costs more than training.** One check is ~38 min of GPU prediction plus ~73 min of CPU
scoring, ≈1.9 h; 17 checks at `EVAL_EVERY=3` is 31.8 h against 8.8 h of `eic.19` training. The
arithmetic, its two measured anchors and the sized walltime are in `slurm/eic_train.sh`'s header.

## Training loci under a `regions` BED (D32) — built 2026-08-31

`configs/regime.eic_pilot.json` restricts training to the 44 ENCODE Pilot Regions lifted to hg38.
eDICE honours it now; `slurm/eic_train.sh` used to refuse the regime and no longer does.

D32's rule is containment: a training window counts only if it lies **wholly** inside a region, on
a tiling anchored at chromosome bin 0 and never re-anchored per region. **eDICE's window is one
25 bp bin** — it carries no positional parameters and no context — so containment here is per bin,
and the scope is the regime's own declared figure: **40 regions, 25,588,197 bp, 1,023,489 bins**,
printed by the launcher before the run starts.

CANDI's 768-bin context cannot fit inside the leading or trailing partial tile of 34 of those 40
regions, so it plans 993,792 bins, 97.10 % of the same scope (§3.1). The two methods therefore see
the same **regions** and not the identical bin set. Narrowing eDICE to CANDI's 1,294 windows was
rejected: the 2.9 % is an artefact of CANDI's context length, and D32 states a containment rule,
not a window grid. Reads walk region by region so a slab stays contiguous — a scattered fancy
index over chr1 would have spanned ~10 M bins and needed ~10 GB to materialise.

### What P1 and P2 scoring confirmed (jobs `56904726`, `56914129`)

Both protocols are scored — A4 requires both, always. The §4.1 contract held end to end in each:

| | P1 (chr21) | P2 (genome-wide) |
|---|---|---|
| chromosomes / bins | 1 / 1,868,399 | **23 / ~124 M** |
| `missing_tracks` | `[]` | `[]` — nothing dropped across 23 chromosomes |
| `declared_tracks` / scored | 45 / 45 | 45 / 45 |
| provenance | `pred_inversion: external`, `signal_target_transform: none`, `pval_pred_space: -log10p` | same |
| blocks | `E`, `P`, `D`, `B` | same — `D` only because the σ-table supplied a distribution |
| **`macro.count`** | **`{}`, 0 entries — ABSENT, not NaN-filled** | **`{}`** |
| per-track arms | `["pval"]` | `["pval"]` — no count arm fabricated anywhere |
| `msevar` | absent, with its own note rather than a silent `0.0` | same |
| non-finite values | none | none |

**The check that matters most is the σ-table's `fitted_on` string.** In *both* files it reads
`regime.eic_val.json eval_pairs, chroms ['chr21']` — so P2 reused the P1 V-pair table **unchanged**
rather than refitting on genome-wide residuals. That is §6.1 as a fact on disk rather than a
promise: refitting per protocol would have made the CRPS column mean two different things in two
rows of the same table, and nothing in the output would have said so.

`pit_ks` and `coverage_95` are present in both, satisfying the PI's ruling below.

### B3's CRPS companions — **ruled (PI, 2026-08-26; plan amendment PR #27)**

Raised from this run: §6.4 required "CRPS always with `crps_oracle_scaled` + `scale_error`", but
those keys come **only from `nb_suite`**, the count arm. `EVAL.md` §D lists the pval arm's
`gauss_suite` as `crps`, `pit_ks`, `coverage_95`, `c_index` + `c_index_se`, `n_points` — no split, by
design. Every point-only rival fills the pval arm alone, so no rival row could ever have carried it.

**The ruling: the CRPS split is count-arm only; a pval-arm Gaussian CRPS is quoted with `pit_ks` and
`coverage_95` instead.** Further, **no pval-arm CRPS gap is significant until that arm's own noise
floor is measured** — a separate task is filed for it, so an eDICE-vs-CANDI pval CRPS difference is
not interpretable yet, whatever its size.

Nothing in this method's outputs changes: `pit_ks` and `coverage_95` are already present in every
macro and per-track block the P1 run produced.

## EIC runs — all compute for t52 is done

| stage | job | result |
|---|---|---|
| train, `N_TARGETS=31` | `56847505` | COMPLETED rc=0, 09:15:19, ~660 s/epoch, loss 0.11852 → 0.09568 |
| P1 score | `56903103` | FAILED — `TrackView.pval()` missing `start` in `fit_sigma.py` |
| P1 score (retry) | `56904726` | COMPLETED rc=0, 08:51 — resumed the finished predict |
| P2 score | `56906350` | TIMEOUT at 12:00:23 — predict finished all 23 chroms, wall hit in the bench pass |
| P2 score (follow-up) | `56914129` | COMPLETED rc=0, 09:00:33 — skipped the 6.1 h predict, scored genome-wide |

**B pairs are untouched.** Per §P1/A4 the `eic_test` regime runs once, at the very end, per
method. Nothing in this branch has opened a `B_` biosample.

The P2 timeout is why `eic_score.sh` skips a complete prediction root. `56906350` finished all 23
chromosomes and wrote its manifest before the wall arrived mid-scoring; `56914129` was queued with
`--dependency=afternotok` and a 24 h wall, found the root complete, and went straight to the bench
pass. Nothing was recomputed — but the margin was one chromosome, since a *partial* root is redone
rather than resumed. If P2 ever needs re-running, per-chromosome resume with length validation is
the fix, not a longer wall.

## Commands

```bash
# unit tests (41) -- NOT part of the core `pytest tests/ -q` gate. The §5 and D32 tests build a
# real CANDI_STORE with the repo's own writer, so they need `src` on the path as well.
cd competitors/edice && PYTHONPATH=.:../../src pytest tests/ -q

# the panel, without touching a GPU
PYTHONPATH=.:../../src python run_eic.py panel --regime ../../configs/regime.eic_19.json

# EIC, on Fir. §12.2: eDICE is a pval-arm method and runs ONCE PER REGIME.
mkdir -p slurm-logs
N_TARGETS=31 REGIME=$REPO/configs/regime.eic_19.json    sbatch competitors/edice/slurm/eic_train.sh
N_TARGETS=31 REGIME=$REPO/configs/regime.eic_pilot.json sbatch competitors/edice/slurm/eic_train.sh
N_TARGETS=31 SCOPE=heldout    sbatch --time=03:00:00 competitors/edice/slurm/eic_score.sh
N_TARGETS=31 SCOPE=genomewide sbatch --time=12:00:00 --mem=96G competitors/edice/slurm/eic_score.sh
```

`eic_train.sh` carries its own `--time=60:00:00`, sized for 50 epochs plus 17 `V_` selection checks
— read its header before overriding it, because the eval is 78 % of the band.

`N_TARGETS=120` with `--time=48:00:00` runs the absolute reading instead; nothing else changes, and
the manifest's `masking_caveat` describes whichever reading was actually run. Both `eic_train.sh`
and `run_eic.py train` **refuse to start without the flag**, and name the decided value.

`eic_score.sh` chains predict → σ-fit → `candi.bench.external`. `PROTOCOL=p2` sets `--chroms all`,
which resolves to every chromosome the store carries — measured: **23 chromosomes, 124 M bins**,
chr1 alone 9,958,256. The σ-table is fitted **once, on P1**, and P2 reuses it: refitting σ on a
genome-wide pass would quietly change what the CRPS column means between two rows of one table.

The EIC wall is **projected, not measured** — scaling the gate's 0.255 s/batch over chr19's 9,159
batches/epoch gives ≈ 10 h at 31 targets and ≈ 30 h at 120, since cost is dominated by
`batch × n_targets` through the decoder.
