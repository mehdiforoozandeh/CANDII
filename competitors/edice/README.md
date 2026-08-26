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

### What P1 scoring confirmed (job `56904726`)

The first time our own instrument read eDICE output, the §4.1 contract held end to end:

| | |
|---|---|
| `missing_tracks` | `[]` — all 45 declared tracks present, nothing silently dropped |
| `declared_tracks` | 45, matching the panel dry-run exactly |
| provenance | `pred_inversion: external`, `signal_target_transform: none`, `pval_pred_space: -log10p` |
| blocks | `E`, `P`, `D`, `B` — `D` present only because the σ-table supplied a distribution |
| **`macro.count`** | **`{}`, 0 entries. The count arm is ABSENT, not NaN-filled** |
| per-track arms | `["pval"]` — no count arm fabricated anywhere |
| `msevar` | absent, carrying its own note rather than a silent `0.0` |
| σ-table | 22 assays over 45 tracks, `fitted_on: regime.eic_val.json eval_pairs, chroms ['chr21']` |

No non-finite value anywhere in the macro block.

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

## Commands

```bash
# unit tests (21) -- NOT part of the core `pytest tests/ -q` gate
cd competitors/edice && PYTHONPATH=. pytest tests/ -q

# the panel, without touching a GPU
PYTHONPATH=.:../../src python run_eic.py panel --regime ../../configs/regime.eic_val.json

# EIC, on Fir. BLOCKED until the gate is recorded AND the PI has called --n-targets.
mkdir -p slurm-logs
N_TARGETS=31 sbatch --time=24:00:00 competitors/edice/slurm/eic_train.sh   # DECIDED, PI 2026-08-25
N_TARGETS=31 PROTOCOL=p1 sbatch --time=03:00:00 competitors/edice/slurm/eic_score.sh
N_TARGETS=31 PROTOCOL=p2 sbatch --time=12:00:00 --mem=96G competitors/edice/slurm/eic_score.sh
```

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
