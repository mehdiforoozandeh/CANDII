# The 23 EIC entrant submissions, scored on Dataset-3 truth

Plan contract: `plan/RIVALS_PLAN.md` §7.5, under protocols §2/P3. **Score, do not reimplement** —
these are the tracks the 23 teams actually submitted to the 2019 ENCODE Imputation Challenge, and
the point is to place our methods beside published rows that were produced from exactly these files.

Nothing here is importable from `candi`, and `candi` never imports it (plan §3).

---

## 1. The vendored scorer

Five files copied **byte-identical** from Max's experiment 001 and never edited. Adaptation lives in
separate driver files beside them; a change to anything under `vendor/` invalidates the reproduction
argument below and must not happen.

| file | md5 | what it is |
|---|---|---|
| `eic_metrics.py` | `2647f20e946f21cd823f62ef29b19009` | the nine challenge measures, the ten bootstrap chromosome groups, blacklist deletion |
| `fir_tracks.py` | `af394d6e35870ca61466dd43436055fa` | `load_official_bigwig` — `ceil(len/25)` binning, exactly as `bw_to_npy.py` |
| `fir_score.py` | `569f4d9223604f7aef5a9f95b03fe816` | one (prediction, truth) pair → one CSV row per bootstrap |
| `fir_build_average_official.py` | `566067338b862b069765f7f7fd7d7e6f` | rebuilds the challenge's Average baseline from its own tracks |
| `fir_synapse_download.py` | `9d79fd79e4b6dddf84292652e4c3fe1d` | resumable, size- and md5-verified Synapse pull |

**Verified three ways** on 2026-08-25, all agreeing:

1. the live files in 001's `scripts/` on Fir,
2. the entries in the staged manifest `MAX_EPI_IMPUTATION/experiments/MD5SUMS.nibi.txt`,
3. 005's own independently vendored copies under `005_.../output/inputs/vendor_001/`.

Re-check at any time with `md5sum -c` against `vendor/MD5SUMS` (which stores the same digests in
`md5 -r` order). The digests also travel inside every result: `score_entrant.py` stamps
`scorer_vendor_md5` into each per-experiment json, so a score always names the scorer that made it.

Source: `/project/def-maxwl/mforooz/MAX_EPI_IMPUTATION/experiments/001_2026-07-27_eic_leaderboard_repro/scripts/`

### Why this code and not the organizers' own

001's transcription reproduces the **published** leaderboard to a worst relative difference of
2.0 × 10⁻⁵ (Avocado) and 2.4 × 10⁻⁵ (Average) over 8 measures × 51 experiments × 10 bootstraps;
median ~3 × 10⁻¹⁰. Worst case is `gwspear`, from rank tie-breaking. It is deliberately faithful to
behaviour that is arguably wrong — `normalize_dict` is a no-op in the official code, so every
published MSE is on the raw scale — because the goal is to land on the published numbers, not on
better ones.

### `msevar` is excluded, everywhere

The ninth measure does not reproduce. No variance vector 001 tried got closer than median ratio 0.19
(training experiments) or 0.61 (train+val+blind pooled); the published vector is dominated by a few
chr19 bins whose source the notebook could not identify. 001 and 005 both exclude it
(`--var-source none`) and so does everything here. **Eight of nine measures are reported**, and that
caveat is printed on every table this directory emits rather than kept in a footnote.

---

## 2. The P-block port

`pblock_bigwig.py` is a numpy-only transcription of `src/candi/bench/partitions.py`.

It is a port and not a wrapper for one reason: the Fir scorer environment is numpy + scipy +
pyBigWig with **no torch**, and `candi.bench.partitions` imports through `candi`. Two copies of a
definition is a maintenance hazard, and `tests/test_pblock_port.py` is the whole mitigation — it
holds the port and the store-path module to **exact equality** (not `approx`) on identical synthetic
input, across every function and the whole suite. If they ever disagree, the store-path module is
right and the port is the bug; it is the copy.

```bash
cd competitors/entrants && PYTHONPATH=../../src:. pytest tests/ -q     # 15 passed
```

The test is deliberately **not** under `tests/`: that is the candi suite and must stay free of any
dependency on `competitors/`. It covers every function plus the whole suite, and two of its cases
guard the fixture rather than the code — the synthetic track must actually occupy the underflow and
overflow strength bins, or the parity claim would be resting on the 35 interior bins alone.

### Verified on real tracks, 2026-08-25

`slurm/driver_parity.slurm` ran genome-wide (121,241,707 bins, chr1–22 + chrX) on three assays:

| target | stand-in prediction | assay | nine-measure cross-check | P-block |
|---|---|---|---|---|
| C05M18 | C06M18 | H3K36me3 (broad) | exact, 8 × 10 bootstraps | `acc_obs` 0.5662 |
| C19M22 | C28M22 | H3K4me3 (punctate) | exact, 8 × 10 bootstraps | `acc_obs` 0.8320, **promoter branch fired** |
| C12M01 | C31M01 | ATAC-seq | exact, 8 × 10 bootstraps | `acc_obs` 0.7546 |

The H3K4me3 run exercised `prom_corr_h3k4me3` over 58,121 promoter windows: 56,662 scored and 1,459
undefined — the constant windows landing in `n_undefined` rather than being scored 0, which is the
behaviour the port is supposed to inherit. All three emitted 37 occupied strength bins (35 interior
plus underflow and overflow), `truth_binarisation: "signal>=2"`, `pblock_blacklist_deleted: false`
beside `nine_measures_blacklist_deleted: true`, and the five vendored md5s.

Cost: ~13 min per experiment genome-wide, of which the P-block is 13–16 s. `CHECK=1` roughly doubles
it (21–26 min), since the cross-check runs a second full scorer.

### Three deviations, forced by what Dataset 3 distributes

Each is stamped into the emitted json, not left to a reader's memory.

1. **No peak calls exist.** The store path binarises truth with MACS2 peak membership and prediction
   with `Y >= 2`, on purpose — the truth has real calls and the prediction does not. The challenge
   distributed signal tracks only. Both sides therefore binarise at `>= 2` here, which makes
   `acc_by_*_strength` a self-consistency measure on a threshold rather than a comparison against
   called peaks. Recorded as `truth_binarisation: "signal>=2"`. A Dataset-3 P-block row is **not**
   comparable to a store-path P-block row.
2. **The blacklist is not applied to the P-block.** The official scorer *deletes* blacklisted bins,
   shifting every downstream index. The P-block is positional — its promoter windows are bin
   coordinates — so on a deleted grid it would read displaced loci. `src/candi/bench/annotations.py`
   refuses the deletion on the store path for exactly this reason. The nine measures keep the
   deletion (it is what produced the published numbers); the P-block takes the register instead.
   Recorded as `pblock_blacklist_deleted: false`. Both grids come from **one** bigwig load, so they
   cannot drift.
3. **`peak_shape_corr_dnase` never fires**, since DNase is excluded and its windows need the peak
   calls point 1 says do not exist.

The annotation is the same bytes on both paths: `gencode.v29.genes.gtf.bed.gz`,
md5 `3c2897b51371ecc2eeba4f4cb4db295e`, identical between `src/candi/bench/assets/` and the challenge
repo's `annot/hg38/`. Same annotation, not merely the same annotation version.

---

## 3. Data and paths on Fir

Dataset-3 tracks **never come to the laptop**.

| what | where |
|---|---|
| blind truth (51), training (268), validation (46) | `/project/def-maxwl/mforooz/DATA_EIC_SYNAPSE/` — 254 GB, md5-verified against Nibi |
| 001's scorer, bridge, annotations, recorded scores | `/project/def-maxwl/mforooz/MAX_EPI_IMPUTATION/experiments/001_2026-07-27_eic_leaderboard_repro/` |
| the ID bridge | `…/001_…/output/bridge/eic_bridge.csv` — 267 T + 45 V + 51 B = 363 |
| challenge scoring repo @ `181b8023` | `…/001_…/output/inputs/imputation_challenge/` |
| our checkout | `/project/def-maxwl/mforooz/CANDII_t54` |
| our outputs | `/project/def-maxwl/mforooz/CANDII_t54_work/` |
| entrant submissions (when pulled) | `/project/def-maxwl/mforooz/DATA_EIC_SYNAPSE/submissions_round2/` |

**IDs bridge on ENCODE experiment accession, never on name munging.** Our `aliases.json` uses
3-digit assay codes (`M022` = H3K4me3) and the challenge uses 2-digit (`M22`); joining on the
accession matches 1:1 on all 363 experiments. `eic_bridge.csv` is that join and is the authority
here — do not rebuild it by string surgery.

Scored chromosomes are **chr1–chr22 + chrX**. A project note claiming autosomes only is wrong.
The official binning is `ceil(chrom_len / 25)`, one bin longer than our store's `floor`.

Environment: `env_fir.sh` builds a throwaway venv in `$SLURM_TMPDIR` (numpy 1.26.4, scipy 1.15.1,
pyBigWig from the local wheelhouse; ~60 s). It is per-job rather than shared, the pattern 005 adopted
after Fir's instability episode, and it deliberately keeps pyBigWig out of the project's `candi_venv`.

---

## 4. The Average gate — prove the path before scoring anyone

The chain, and what each link buys:

```
published `Average` rows (MOESM2, team_id 100)
  <- 001's our_average_tv.csv     [001 vs published: median rel 2.7e-8, max 2.6e-7, 100% within 1e-6]
  <- our re-run on Fir            [gate_average.py measures this link]
```

"Training set" for the round-2 baseline means **train + validation**, not train alone. 001 settled it
via the ATAC-seq exception: ATAC has zero validation experiments and is the only assay matching
published on training-only. The train-only variant misses (frac within 1 % = 0.18); trainval matches
exactly. Hence `VARIANT=trainval`.

```bash
cd /project/def-maxwl/mforooz/CANDII_t54/competitors/entrants && mkdir -p slurm_logs

# 1. rebuild the Average from the staged train+val tracks (23 chromosomes)
sbatch --array=0-22 --export=ALL,VARIANT=trainval slurm/build_average.slurm

# 2. score it against blind truth with the VENDORED scorer, unmodified (51 experiments)
sbatch --array=0-50 slurm/gate_score.slurm

# 3. compare to 001's recorded numbers
python3 gate_average.py \
    --ours      /project/def-maxwl/mforooz/CANDII_t54_work/gate_scores \
    --reference /project/def-maxwl/mforooz/MAX_EPI_IMPUTATION/experiments/001_2026-07-27_eic_leaderboard_repro/output/scores/our_average_tv.csv \
    --out-json  /project/def-maxwl/mforooz/CANDII_t54_work/average_gate.json
```

Step 2 runs `vendor/fir_score.py` as a CLI, so a match is evidence about the staged data and the way
we drive the scorer, with our own code contributing nothing but the argument list.

Verdicts: `PASS` at ≤ 1e-9 (bit-level — we are re-running the same code on the same data, so exact is
the expectation); `PASS_WITH_DRIFT` up to 2.4e-5, which is 001's own worst case against the published
table and means find the source before scoring entrants; `FAIL` above that; `INCOMPLETE` if any
reference row has no counterpart — the gate never passes on a partial panel.

### Result, 2026-08-25: **PASS at exactly 0.0**

510 (experiment, bootstrap) rows × 8 measures, 0 missing and 0 extra, worst relative difference
**0.000e+00** — bit-identical to 001 on every measure. Recorded in
`CANDII_t54_work/average_gate.json`.

So the full chain holds: the staged Dataset-3 tracks reproduce the Average that 001 built, the
vendored scorer reproduces the scores 001 recorded, and 001 in turn matched the *published* Average
rows to median 2.7e-8 / max 2.6e-7. **An entrant score produced by this path inherits that
verification.**

Build inputs, for the record: 23 assays × 23 chromosomes = 529 npz, no assay missing a source track,
287 contributor tracks summed over the 23 assays needed for the blind/validation targets, drawn from
the 312 train+validation bigwigs linked into the farm.

The comparison logic was itself checked three ways before use: the reference against itself gives
`PASS` at 0.0; 001's *train-only* Average against its trainval reference gives `FAIL` at 2.157e-01,
which is the discriminating comparison that settled the variant question; and a truncated input gives
`INCOMPLETE`.

---

## 5. Downloading the submissions

`syn17083203/submissions_round2/` = **`syn21976211`**. Surveyed 2026-08-25:
**1030.6 GB, 1172 files, 23 teams** (`fetch_submissions.py --survey` talks to the API only and
transfers nothing; the breakdown is in `CANDII_t54_work/submissions_survey.json`).

`/project def-maxwl` had 16 TiB of 28 TiB used at survey time, so it fits — but the quota is
**group-charged**, so the pull is coordinated with Max and needs a go-ahead. `fetch_submissions.py`
refuses any tree over `--max-gb 500` without `--yes-i-checked`, which makes forgetting to report the
size an error rather than a surprise on someone else's storage.

Note `UIOWA_Michaelson` has **50** files where every other team has 51 — check which experiment is
absent before scoring that team, and record it rather than letting a partial panel through.

```bash
# survey again (transfers nothing)
sbatch --export=ALL,SURVEY=1 slurm/fetch_submissions.slurm

# once approved — eight tasks shard the tree by file, no racing, resumable
sbatch --array=0-7 --export=ALL,YES=1 slurm/fetch_submissions.slurm
```

Token: a Synapse personal access token with view + download scope at `~/.synapse_pat`, mode 600
(the driver refuses a group- or world-readable file and never prints the token). Everything under
`syn17083203` is otherwise unrestricted — no data-access committee.

`syn21519009` (Lavawizard weights) belongs to a **different task**; do not pull it here.

---

## 6. Scoring an entrant, and the table

```bash
sbatch --array=0-50 --export=ALL,LABEL=Guacamole,PREDDIR=/project/def-maxwl/mforooz/DATA_EIC_SYNAPSE/submissions_round2/Guacamole \
    slurm/score_entrant.slurm
```

All 23 teams, once the tracks are down — one array per team, so a team that fails is resubmitted
alone:

```bash
SUB=/project/def-maxwl/mforooz/DATA_EIC_SYNAPSE/submissions_round2
for team in "$SUB"/*/; do
    t=$(basename "$team")
    sbatch --array=0-50 --export=ALL,LABEL="$t",PREDDIR="$team" slurm/score_entrant.slurm
done
```

`score_entrant.py` loads truth and prediction **once** and emits both blocks — a whole-genome track
is ~124 M bins, so a second load would roughly double a 23 × 51 grid. Measured cost of the P-block
half is ~20 s and ~5 GB peak genome-wide, so it is nearly free beside the bigwig read. `CHECK=1`
additionally re-runs `vendor/fir_score.py` on the same inputs and requires an **exact** match on all
eight measures; that doubles wall clock, so it is for the gate and spot-checks, not the full grid.

Standing proof that the adapter did not move a number, needing no submissions at all:

```bash
sbatch --array=0-2 slurm/driver_parity.slurm
```

It presents a different cell's blind track under the target's filename — a real, non-degenerate
prediction from files already on disk — and requires exact agreement with the vendored scorer on one
broad mark, one H3K4me3 (the only assay where the P-block's promoter branch fires) and one ATAC.

DNase-seq experiments exit 0 without scoring (decision B3). That drops 3 of the 51 blind-test
experiments, leaving **48**: 26 broad (H3K27me3 ×9, H3K9me3 ×9, H3K36me3 ×8) and 22 punctate
(H3K27ac ×7, H3K4me1 ×7, H3K4me3 ×5, ATAC-seq ×3).

```bash
python3 placement_table.py \
    --scores Average=…/Average/*.csv Avocado_p0=…/Avocado_p0/*.csv Guacamole=…/Guacamole/*.csv … \
    --pblock Guacamole=…/Guacamole/*.json … \
    --out-md placement.md --out-json placement.json
```

Aggregation, stated once: bootstraps averaged within an experiment → experiments combined by
**median** within an assay (what 001 and 005 both report, and what survives the heavy tail the
blacklist deletion leaves on weak marks) → macro is the mean over assay medians, so each assay weighs
equally rather than each experiment.

`placement_table.py` enforces the reporting invariants rather than trusting the writer: per-assay
rows first, broad and punctate macros printed **before** any pooled row, DNase dropped at load, and
all six caveats appended to every table it emits.

**These tracks never enter an internal (Dataset-2) table.** 005 measured 12–66 % per-experiment error
for rescaling a Dataset-2 score into Dataset-3 space (best H3K4me3 12 %, worst H3K4me1 66 %; DNase's
factor is ~1327, not a translation factor at all). Score directly against Dataset-3 truth; never
translate.
