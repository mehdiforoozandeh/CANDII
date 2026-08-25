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

`edice_torch/model.py` also carries `SelfTransformerBlock`, unused at `n_attn_layers=1` but present
because the reference stacks `n_attn_layers - 1` of them before the cross block.

## Validation gate — status: **PENDING**

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
| **eDICE (ours)** | *pending* | *pending* | n/a | n/a |

`± ` is the s.e.m. over the 203 test tracks, which is what our `metrics.summarise` returns.

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
| `sample` | the repo's packaged 10 k-bin chr21 sample, 20 epochs, `train` split. A **smoke** run: nothing is published for it to hit | `56703977` | submitted 2026-08-25 |
| `full` | `roadmap_tracks_shuffled.h5`, 997,373 bins, PredictD splits, supports `train ∪ val`, 50 epochs. **This is the gate** | *pending the smoke run* | — |

```bash
mkdir -p slurm-logs
MODE=sample sbatch competitors/edice/slurm/roadmap_gate.sh
MODE=full   sbatch --time=24:00:00 --mem=48G competitors/edice/slurm/roadmap_gate.sh
```

Each writes `gate.json` (published vs ours, both spaces, per-epoch history), `test_preds.npz` and
`model.pt` under `/project/def-maxwl/mforooz/rivals_src/edice_runs/<mode>/`.

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

### The one open decision

`--n-targets` on the EIC path has **no default and the run refuses to start without it**, because
"paper defaults" is genuinely ambiguous here and the two readings are far apart:

* eDICE masked **120 of Roadmap's 1032** tracks per bin — 11.6 %;
* our EIC panel holds **267** tracks, so the same *absolute* 120 masks **45 %**, while the same
  *rate* is **31**.

That is a decision about the training signal, not a knob to tune, so it is pre-registered rather
than defaulted. **PI call needed before EIC training starts.**

A second, smaller one: `run_eic.py train --train-chroms` defaults to the regime's `train_chroms`
(chr19), matching CANDI's own recipe. eDICE's own recipe trains on every bin it has. The choice is
not a leakage question — eDICE has no positional parameters, so P1 and P2 are equally out-of-sample
for it in the only sense that exists for this architecture — but it is a budget question, and it is
recorded here rather than assumed.

## Arms this method fills

`signal_mu` only, in `-log10 p`, clipped at 0. eDICE has **no count head** — B1b forbids inventing a
read depth to manufacture one — and **no peak head**, so `candi.bench.external` records the
coverage-ranking fallback (`has_peak_head=False`) and every peak-tier row is labelled as such (B3).
The forecast distribution arrives from the §6.1 σ-table, not from the model.

## Commands

```bash
# unit tests (21) -- NOT part of the core `pytest tests/ -q` gate
cd competitors/edice && PYTHONPATH=. pytest tests/ -q

# the panel, without touching a GPU
PYTHONPATH=.:../../src python run_eic.py panel --regime ../../configs/regime.eic_val.json

# EIC: train -> predict -> sigma -> score
python run_eic.py train   --regime ../../configs/regime.eic_val.json --out runs/eic \
                          --n-targets <PRE-REGISTERED>
python run_eic.py predict --regime ../../configs/regime.eic_val.json \
                          --model runs/eic/model.pt --out preds/edice_p1
python fit_sigma.py       --regime ../../configs/regime.eic_val.json \
                          --pred preds/edice_p1 --out preds/edice_sigma.json
python -m candi.bench.external --store ../../configs/regime.eic_val.json \
    --pred preds/edice_p1 --out scores/edice_p1.json --sigma-table preds/edice_sigma.json
```

P2 is the same `predict` call with `--chroms` naming every chromosome the store carries, scored
through the same entry with the bench's own `--chroms` override.
