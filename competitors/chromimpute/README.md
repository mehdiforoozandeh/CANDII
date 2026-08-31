# ChromImpute on our EIC data

`RIVALS_PLAN.md` §7.2. ChromImpute is run **as published** — Jason Ernst's jar, his seven commands,
his defaults. Nothing in this directory models anything. `prepare.py` turns CANDI_STORE `pval`
tracks into the three files `Convert` wants, `collect.py` turns `Apply`'s wigs back into the §4.1
prediction root, and `slurm/` drives the six stages. Any number this method produces comes out of
the jar.

| | |
|---|---|
| upstream | `github.com/jernst98/ChromImpute`, GPL-2.0, **v1.0.5** (released 2024-01-01) |
| jar | vendored on Fir at `~/scratch/t51_chromimpute/tool/ChromImpute.jar` — the repo's own prebuilt jar, not rebuilt from source |
| java | `module load java/21.0.1` on Fir; the jar targets 1.7 |
| authority | `ChromImpute_manual.pdf` in that repo, for all seven command signatures |
| hyperparameters | paper defaults throughout (§6.3: one seed, no sweeps) — see *What we do not pass* |

## Where fairness is enforced (§6.2)

**`prepare.training_tracks`, and nowhere else.** It walks `regime["biosamples"]["train"]`, refuses
any name without the `T_` prefix, and keeps only assays in `regime["assays"]`. Its output is
`inputinfofile.txt`, and `inputinfofile.txt` is the *entire* compendium ChromImpute can see: all
five data commands (`Convert`, `ComputeGlobalDist`, `GenerateTrainData`, `Train`, `Apply`) take it
as a required parameter and none of them reads a track that is not listed in it. No `V_` or `B_`
biosample can therefore enter training, the correlation table, or a feature vector.
`test_training_tracks_admit_training_cells_only` and `test_training_tracks_refuse_a_non_training_prefix`
hold that.

Two consequences worth stating rather than discovering later:

- The **imputation target is a `T_` cell**. A declared pair `(T_X, V_X)` is one cell type twice:
  `T_X` carries the assays a method may condition on and `V_X` carries the held-out truth. So
  `Apply` is asked for `(sample = T_X, mark = A)` where `A` is an assay `T_X` does not have, and
  the answer is scored against `V_X`'s `A`. `prepare.impute_targets` builds that set as
  `assays(V_X) − assays(T_X)`.
- ChromImpute is **position-transductive by construction**. Its trees are trained at genomic
  positions where the target mark was observed *in other samples*, and those positions include the
  evaluation chromosome. That is the published method, not a choice of ours, and it is why P1 and
  P2 are both reported (§2). It leaks no held-out *experiment*: the cells supplying those positions
  are training-split cells.

`chipseq-control` appears in the store manifest but not in the regime's assay list, and the same
filter drops it. It is a control column, not an assay.

## The grid

`regime.eic_val.json`: 51 training cells → **267 training tracks** over **35 marks** in the
compendium; 26 declared pairs → **45 (cell, assay) imputation targets** over **22 distinct marks**.
The stage costs scale differently — `Convert` per training track, `ComputeGlobalDist` per
compendium mark, `GenerateTrainData` per *target* mark, `Train` and `Apply` per target — which is
why `prepare.pilot_subset` spreads the pilot over marks rather than over cells.

Two of those scalings were learned the hard way and are worth stating:

- **`ComputeGlobalDist` must run for all 35 compendium marks, not the 22 target marks.**
  `GenerateTrainData`'s `loadDistInfo` opens `DISTANCEDIR/<sample>_<mark>.txt` for every
  `(sample, mark)` in `inputinfofile` before it reaches the target, so one missing mark fails every
  target in two seconds.
- **`GenerateTrainData` and `Apply` each hold one open gzip reader per compendium track** — 267 of
  them at once. An unthrottled SLURM array puts thousands of concurrent handles on Lustre and draws
  transient `FileNotFoundException ... (Cannot send after transport endpoint shutdown)` opens that
  read like missing files. `slurm/submit.sh` caps those stages with `--array=...%10`
  (`CI_THROTTLE`).

## Two grid traps, both handled

**The chromosome length we declare is a multiple of 25.** `Convert` writes `(len − 1) / 25 + 1`
bins. Our store's grid is `floor(len / 25)`, and `bench.external` refuses any prediction whose
length is not exactly that. So `chrominfo.txt` declares `n_bins × 25`, not the true chromosome
length, and the partial tail bin never exists — the same bin the manual's `wigToBigWig -clip` note
is about. `test_chrominfo_length_makes_convert_emit_exactly_n_bins` is the check.

**Everything ChromImpute writes is rounded to two decimals.** Both the `Convert` writer and the
`Apply` writer format through a `NumberFormat` with `setMaximumFractionDigits(2)`
(`ChromImpute.java`). So the method's `-log10 p` predictions arrive quantized to 0.01, and there is
no flag to change it. That is a property of the rival, recorded here so nobody reads it as a bug in
our converters. It is also why `prepare.py` writes bedgraph at four decimals: `Convert` averages the
input over the bases of each bin, our input is already constant across each 25 bp bin, so the
average is the bin value and anything past the second decimal is invisible downstream anyway.

A third, smaller one: our sample and mark names both contain `_`, and `Apply`'s default output name
is `impute_<sample>_<mark>.wig`, which cannot be split back. We always pass `-o` and name the file
ourselves (`collect.apply_output_name`). ChromImpute never parses these names — it only
concatenates them — so the separator is ours to pick.

## The pipeline

```bash
STORE=/project/def-maxwl/mforooz/CANDI_STORE/eic
REPO=/project/def-maxwl/mforooz/CANDII_t51
RUN=~/scratch/t51_chromimpute/pilot_chr21
JAR=~/scratch/t51_chromimpute/tool/ChromImpute.jar

# 0. the three files + the run-length bedgraphs        (python, array over --shard i/N)
PYTHONPATH=$REPO/src python prepare.py --store $STORE \
    --regime $REPO/configs/regime.eic_val.json --out $RUN/input --chroms chr21 --pilot 20

# 1..5, one SLURM array per stage, chained with --dependency=afterok
java -mx6000M  -jar $JAR Convert          -c chr21 -m $MARK $RUN/input/signal $II $CI $CONV
java -mx12000M -jar $JAR ComputeGlobalDist -m $MARK          $CONV $II $CI $DIST
java -mx20000M -jar $JAR GenerateTrainData -c chr21          $CONV $DIST $II $CI $TRAINDATA $MARK
java -mx20000M -jar $JAR Train                               $TRAINDATA $II $PRED $SAMPLE $MARK
java -mx8000M  -jar $JAR Apply -c chr21 -o impute.$SAMPLE.$MARK.wig \
                                                             $CONV $DIST $PRED $II $CI $IMP $SAMPLE $MARK

# 6. back to the §4.1 contract
PYTHONPATH=$REPO/src python collect.py --store $STORE --targets $RUN/input/targets_pilot.tsv \
    --impute-dir $RUN/OUTPUTIMPUTEDIR --pred-root $RUN/pred --chroms chr21 --jar $JAR
```

`II` is `$RUN/input/inputinfofile.txt`, `CI` is `$RUN/input/chrominfo.txt`. `slurm/submit.sh`
does all of it; `slurm/stage.sh` is the one array script every stage runs through.

**`unset JAVA_TOOL_OPTIONS`** before any `java` call on Fir. The `java` module exports `-Xmx2g`,
which silently overrides the `-mx` on the command line and turns a large stage into an
`OutOfMemoryError`. `slurm/stage.sh` does it.

## What we do not pass

Every ChromImpute option is left at its default, per §6.3 (paper defaults, one seed, no sweeps).
Named here so the omissions are deliberate rather than forgotten: `-b numbags` (1), `-a
mintotalensemble` (0), `-f numsamples` (100 000 training locations), `-k maxknn` (10), `-n knnwindow`
(20 bins), `-w windownarrow windowwide` (20 / 400 bins), `-i incrementnarrow incrementwide` (1 / 20
bins), `-m minnumpoints` (20). `-dnamethyl` is not used — we have no methylation. The only flags we
do pass select what to parallelize over (`-c`, `-m`, `-l`) or name an output file (`-o`).

## Arms

`pval` only. ChromImpute predicts a point in `-log10 p`: no spread, so `signal_sigma` comes later
from the §6.1 σ-table and not from the method; no count prediction, because B1b forbids inventing a
read depth; no peak score, so the bench falls back to coverage ranking and records
`has_peak_head=False` (§6.4 requires every ChromImpute peak row be labelled that way).

## Correctness check (§7.2, "the no-retraining check")

The vendor ships an example dataset — chr21 Roadmap data for eight marks, **with pre-trained
predictors** — at `pubs.broadinstitute.org/jernst/EXAMPLE.zip`. Running `Apply` against those
predictors exercises `Apply`, `ComputeGlobalDist`'s output format and the converted-wig reader
without retraining anything, so any disagreement is ours:

```bash
java -mx4000M -jar $JAR Apply EXAMPLE/CONVERTEDDATADIR EXAMPLE/DISTANCEDIR EXAMPLE/PREDICTORDIR \
    EXAMPLE/tier1_samplemarktable.txt EXAMPLE/hg19sizes_chr21.txt EXAMPLE/OUTPUTDATA E034 H3K9ac
java -jar $JAR Eval EXAMPLE/CONVERTEDDATADIR <observed E034 H3K9ac file> \
    EXAMPLE/OUTPUTDATA impute_E034_H3K9ac.wig.gz EXAMPLE/hg19sizes_chr21.txt
```

The manual publishes no expected numbers for it, so the recorded result is the `Eval` line itself,
in the pilot memo, alongside the jar's `Version` string.

## Tests

```bash
PYTHONPATH=$PWD/src pytest competitors/chromimpute/tests -q
```

Deliberately outside `tests/`: that suite is `candi`'s, and §3 keeps the two sides apart. The
non-experiment gate (`pytest tests/ -q`, `tools/golden.py check`) is unaffected by anything here —
this task adds no line under `src/candi`.
