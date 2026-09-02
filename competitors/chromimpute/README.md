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
  positions where the target mark was observed *in other samples*. That is the published method,
  not a choice of ours. It leaks no held-out *experiment* — the cells supplying those positions
  are training-split cells, so `BENCHMARK_DESIGN.md` Rule 1 is intact — but the positions
  themselves are the regime's business, and **which chromosomes they come from is decided by the
  chrominfo `GenerateTrainData` is handed**, not by the method. We hand it the regime's training
  loci, in every regime. See *The training grid* below, which is also where the departure from the
  published recipe is recorded.

`chipseq-control` appears in the store manifest but not in the regime's assay list, and the same
filter drops it. It is a control column, not an assay.

## The training grid

**A departure from published practice, on the record.** ChromImpute as published samples its
100,000 training locations inside the chromosomes it later predicts; the ENCODE Imputation
Challenge and the rest of the imputation lineage do the same (§11). We do not. `BENCHMARK_DESIGN.md`
§2 Rule 2 names the loci a method's **transferable** parameters may be fit on, `Train` turns the
sampled instances into one predictor per (sample, mark) that `Apply` reuses at every position it
predicts — so the predictor is transferable and Rule 2 reaches it. **The sampler is pointed at the
regime's `train_chroms`.** Numbers this directory produces are therefore *our* ChromImpute under
this benchmark's rule, not the paper's, and a row carrying them says so.

What does **not** move: `Apply`'s neighbour features still work on the eval chromosomes. §2 Rule 2
names per-position adaptation on the scored chromosomes as inference, open to every method, and
lists ChromImpute's neighbour features by name.

**Two chrominfo files, because one file cannot say two things.** `GenerateTrainData` spreads its
100,000 locations over everything the chrominfo it is handed declares — verified by running it
against two grids that differ only in which chromosomes they declare, with the same converted
directory: the training instances came back byte-identical, so the converted files of an undeclared
chromosome are never read. So `prepare.py` writes `chrominfo.train.txt` (the training loci) beside
`chrominfo.txt` (the chromosomes `Apply` predicts), and `stage.sh` hands the first to
`ComputeGlobalDist` and `GenerateTrainData` and the second to `Apply`. There is no fallback from one
to the other: a missing `chrominfo.train.txt` makes every stage but `prepare` refuse, because the
fallback *was* the bug.

Fitting on one chromosome set and applying to another is something the jar does without complaint —
predictor files are named `classifier_<sample>_<mark>_<i>_<j>.txt.gz`, with no chromosome in the
name, and a `Train`-on-chrT / `Apply`-on-chrE smoke run end to end returns ordinary predictions.

**The sampling density changes, and the row should say so.**

| grid | training bins | 100,000 locations are |
|---|---|---|
| ChromImpute as published (genome-wide) | 121,241,684 | **0.08 %** — the §11 convention |
| `eic.19`, our training grid (chr19) | 2,344,704 | **4.3 %**, ~53× the convention |
| `eic.pilot`, our training grid (40 Pilot Regions) | 1,023,489 | **9.8 %**, ~122× the convention |

The published 0.08 % *is* 100,000 locations over the whole genome — we change the scope, never `-f`
— so all three rows run the same jar at the same default and differ only in how much genome the
sample is spread over. That is a difference between the two regime rows which is not the regime's
scored slice, and it is larger for `eic.pilot` than for `eic.19`.

(An earlier draft of this file put `eic.19` at ~2.6 %. That number was 100,000 over chr21+chr22 and
had dropped chr20; it also described the *old* behaviour, sampling on the eval chromosomes, which
would have been 1.5 %. Both are superseded by the table above.)

### What containment means under a `regions` regime (D32)

`eic.pilot` narrows the training loci further, to the 44 ENCODE Pilot Regions
(`BENCHMARK_DESIGN.md` §3.1), and the jar has no flag for a BED: the smallest scope any of its
commands understands is a chromosome out of `chrominfo`. So we declare the scope as a grid instead
of asking the sampler to respect one.

D32's rule is written for CANDI's 768-bin windows — a locus counts only if the *whole* window lies
inside one region, on the chromosome's own bin grid. ChromImpute samples 100,000 single 25 bp
locations, so the same rule at a window of one bin
is: bin `i` is a legal training location iff `[i*25, (i+1)*25)` lies inside one region, i.e. bins
`ceil(start/25)` up to `end//25`. It is a containment **count**, not `bp // 25` — the hg38 regions
do not begin or end on the grid — and the indices are chromosome bin indices, so the grid stays
anchored at chromosome bin 0 and is never re-anchored at a region start, which is what §3.1 ruled.
On `eic.pilot` that is **40 regions, 1,023,489 bins**, the same two numbers §3.1 pins for CANDI.
`test_containment_is_counted_the_same_way_the_window_sampler_counts_it` holds the two to one rule
by checking `region_scope` against `RegionSet.bin_spans`.

**Each region becomes one declared chromosome**, one line of `chrominfo.train.txt` at
`n_contained_bins * 25`, with a bedgraph per region per training track. Two consequences are the
reason for the shape:

- **One task per mark, no `-c`.** The whole region grid is a fortieth of a real chromosome, so
  splitting it buys nothing. `-c` would be safe — it **divides** the sample rather than repeating
  it per declared chromosome, measured on a two-chromosome grid at `-f 400`: unsplit gave 400
  instances, the two `-c` tasks gave 193 + 207 — but 40 array tasks per mark for minutes of work is
  not worth the file count. `submit.sh` splits only a training grid of more than one *chromosome*.
- **No feature window spans two regions.** 400 bins is the widest neighbour window and every
  region is ≥500 kb, so keeping the regions apart costs edge truncation on ~2 % of the scope and
  buys never averaging two loci megabases apart. Concatenating them into one pseudo-chromosome
  would have done the opposite.

Without a `regions` block the training grid is the regime's `train_chroms` whole — `chr19` under
`eic.19`, one declared chromosome — and everything above about the two files still holds.

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
REPO=/project/def-maxwl/mforooz/CANDII_t78_code
RUN=~/scratch/t51_chromimpute/eic_19
JAR=~/scratch/t51_chromimpute/tool/ChromImpute.jar

# 0. the four files + the run-length bedgraphs         (python, array over --shard i/N)
PYTHONPATH=$REPO/src python prepare.py --store $STORE \
    --regime $REPO/configs/regime.eic_19.json --out $RUN/input --chroms chr20,chr21,chr22 --pilot 20

# 1..5, one SLURM array per stage, chained with --dependency=afterok. Note WHICH chrominfo each
# command gets: the two stages that fit transferable parameters get $CI_TRAIN, Apply gets $CI.
for C in chr20 chr21 chr22; do
java -mx6000M  -jar $JAR Convert          -c $C -m $MARK $RUN/input/signal $II $CI       $CONV
done
java -mx6000M  -jar $JAR Convert       -c chr19 -m $MARK $RUN/input/signal $II $CI_TRAIN $CONV
java -mx12000M -jar $JAR ComputeGlobalDist -m $MARK       $CONV $II $CI_TRAIN $DIST
java -mx20000M -jar $JAR GenerateTrainData                $CONV $DIST $II $CI_TRAIN $TRAINDATA $MARK
java -mx20000M -jar $JAR Train                            $TRAINDATA $II $PRED $SAMPLE $MARK
for C in chr20 chr21 chr22; do
java -mx8000M  -jar $JAR Apply -c $C -o impute.$SAMPLE.$MARK.wig \
                                                          $CONV $DIST $PRED $II $CI $IMP $SAMPLE $MARK
done

# 6. back to the §4.1 contract
PYTHONPATH=$REPO/src python collect.py --store $STORE --targets $RUN/input/targets_pilot.tsv \
    --impute-dir $RUN/OUTPUTIMPUTEDIR --pred-root $RUN/pred --chroms chr20,chr21,chr22 --jar $JAR
```

`II` is `$RUN/input/inputinfofile.txt`, `CI` is `$RUN/input/chrominfo.txt` (the apply grid) and
`CI_TRAIN` is `$RUN/input/chrominfo.train.txt` (the training grid). `slurm/submit.sh` does all of
it; `slurm/stage.sh` is the one array script every stage runs through.

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

`pval` only. ChromImpute predicts a point in `-log10 p`: no spread, so `signal_sigma` would have to
come from the §6.1 σ-table and not from the method; no count prediction, because B1b forbids
inventing a read depth; no peak score, so the bench falls back to coverage ranking and records
`has_peak_head=False` (§6.4 requires every ChromImpute peak row be labelled that way).

**σ comes from the training-residual pass, and from nowhere in this tree (D2).** The retired fitter
here pooled the bench's per-track `mse` — already a mean squared residual against `V_` truth — and
returned its square root; its own table said `IN-SAMPLE for the V panel`, and §12.2 voids every such
σ under Rule 1. It is deleted. The replacement is shared with every other point-only method:

```bash
python tools/sigma_training_regime.py --regime configs/regime.eic_19.json \
    --n-cells 12 --seed 890217 --out $CI_RUN/regime.eic_19.sigma.json
python -m competitors.sigma_pass --regime $CI_RUN/regime.eic_19.sigma.json \
    --pred $CI_RUN/train_pred --out $CI_RUN/sigma.json --method ChromImpute
```

It needs ChromImpute's predictions for the sampled **training** tracks, not the `V_` panel's. A table
is usable exactly when its `fitted_on` starts with `training-residuals:`; given one, the run scores
the D block too. Given none, it scores the E and P blocks and writes no `gauss_suite` — absent keys,
not NaN.

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
