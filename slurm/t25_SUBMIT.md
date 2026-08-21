# t25 — the submit sequence

**Do not start any of this before `t24` is merged to `main` and `$KIT` is synced.** A half-updated
kit produces a corpus in two codecs, and the `transform` root attr tells you what a file *is*, not
what it was supposed to be. `slurm/t25_rebuild_pval.sh` prints the kit's short SHA in every task log
so a mixed run is at least diagnosable after the fact.

```bash
hpc up fir                      # in Terminal — an agent cannot answer the Duo prompt
hpc push CANDII fir             # $KIT = /home/mforooz/projects/def-maxwl/mforooz/CANDII
```

## 0. Lists and log directory, once

```bash
STORE=/project/def-maxwl/mforooz/CANDI_STORE
mkdir -p $STORE/t25/logs
cp $STORE/t12/eic_biosamples.txt    $STORE/t25/
cp $STORE/t12/merged_biosamples.txt $STORE/t25/
wc -l $STORE/t25/*_biosamples.txt        # expect 89 and 361
```

The lists are t12's verbatim — the same biosamples, the same order, so an array index means the same
thing in both runs and the two logs can be diffed.

## 1. The rebuild (jobs A–C)

`%15` is not decoration: the same Lustre source tree serves everything else on the account.

```bash
STORE=/project/def-maxwl/mforooz/CANDI_STORE
KIT=/home/mforooz/projects/def-maxwl/mforooz/CANDII

# A — EIC, 89 tasks, ~2.1 node-hours
sbatch --array=0-88%15 --export=ALL,CORPUS=eic,SRC=/project/6014832/mforooz/DATA_CANDI_EIC \
       $KIT/slurm/t25_rebuild_pval.sh

# B — MERGED, 361 tasks, ~8.4 node-hours
sbatch --array=0-360%15 --export=ALL,CORPUS=merged,SRC=/project/6014832/mforooz/DATA_CANDI_MERGED \
       $KIT/slurm/t25_rebuild_pval.sh

# C — the 5-biosample slice, so it does not stay on the old codec
sbatch --array=0-4 --export=ALL,CORPUS=eic_slice,SRC=/project/6014832/mforooz/DATA_CANDI_EIC \
       $KIT/slurm/t25_rebuild_pval.sh
```

Budget, from t12's measured actuals on the identical workload: 450 tasks, mean 84 s, max 319 s, **0
failures**, 10.57 node-hours. Expect the same plus ~7% larger writes. Wall clock at `%15` is
**30–60 min**. `--overwrite` replaces one file at a time, so the transient disk cost is one file and
the steady-state growth is ~+20 GB on 289 GB; `/project` has 12 TiB free.

Before moving on, confirm every task exited 0:

```bash
grep -c 'rc=0' $STORE/t25/logs/pval_<ARRAY>_*.out
grep -L 'rc=0' $STORE/t25/logs/pval_<ARRAY>_*.out    # must print nothing
```

## 2. The manifests (job D)

Required, not optional: `build-manifest` is what republishes `pval_scale` / `pval_transform` per
track, and the manifest is where a consumer looks to answer "what are the units".

```bash
CSV_ARGS="--metadata-csv <the same flags t12 used>"   # see cruxvault/results/t12/
sbatch --export=ALL,CORPUS=eic,SRC=/project/6014832/mforooz/DATA_CANDI_EIC,CSV_ARGS="$CSV_ARGS" \
       $KIT/slurm/t25_manifest.sh
sbatch --export=ALL,CORPUS=merged,SRC=/project/6014832/mforooz/DATA_CANDI_MERGED,CSV_ARGS="$CSV_ARGS" \
       $KIT/slurm/t25_manifest.sh
```

## 3. The gate (job E)

`tools/pval_codec_scan.py` **is** `PVAL_CODEC_PLAN.md` §6. It exits 0 only when every file is at
schema 2 / arcsinh / 2000, every `pval_clip_frac` reads exactly 0.0 (D28), the manifest names the
same codec the files do, and the spot round trips land inside `eps * hypot(1, x)`.

```bash
for C in eic merged eic_slice; do
  python $KIT/tools/pval_codec_scan.py --corpus-root $STORE/$C \
      --json $STORE/t25/${C}_codec_scan.json || echo "FAILED: $C"
done

# the two tracks the defect was worst on, round-tripped against the source
python $KIT/tools/pval_codec_scan.py --corpus-root $STORE/eic \
    --roundtrip --source-root /project/6014832/mforooz/DATA_CANDI_EIC \
    --spot B_SJCRH30/H3K4me3 --spot T_DND-41/ATAC-seq \
    --json $STORE/t25/eic_roundtrip.json
```

`B_SJCRH30/H3K4me3` is the worst-clipped track in the corpus (0.371% of its bins) and ATAC-seq is
the assay where all seven tracks clipped. `n_above_old_ceiling` in the report is the count of bins
the old codec flattened to 655.35; it must be > 0 for those spots, or the round trip is not testing
what it claims to.

**The corpus is mixed until this is green, and a mixed corpus is safe to READ (D27) and not safe to
TRAIN on** — precision differs between biosamples. Do not start a run against the store until E
passes.

## 4. Bring the evidence home

```bash
mkdir -p cruxvault/results/t25
rsync -avz -e 'ssh -o BatchMode=yes' \
  fir:/project/def-maxwl/mforooz/CANDI_STORE/t25/'*_codec_scan.json' \
  fir:/project/def-maxwl/mforooz/CANDI_STORE/t25/'*_roundtrip.json' \
  cruxvault/results/t25/
echo "/project/def-maxwl/mforooz/CANDI_STORE/t25" > cruxvault/results/t25/FIR_PATH.txt
```

Checkpoints and logs stay on the cluster; only the small evidence comes down.
