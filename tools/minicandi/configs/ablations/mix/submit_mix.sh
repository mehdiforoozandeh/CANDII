#!/bin/bash
# Close-out item A13, mix-and-match combinations: train and score the combination arms of
# configs/ablations/mix/rung{2,3,4} and their reference arms on table version eic_v1_2, scored on
# ALL 256 evaluation regions (eval/eic_v1_2/evalset.json) into <run>/seed<s>/p256_sv, with the
# same-view pass; every arm with fixed input depth is scored a second time from native-depth inputs
# into <run>/seed<s>/p256_sv_native_depth (the deployment case). Run on a login node of the cluster.
# DRY RUN BY DEFAULT: it prints the sbatch lines and submits nothing.
#
#   CLUSTER=nibi PRED_STORE_ROOT=<dir> STAGE=baselines bash submit_mix.sh             # dry run
#   CLUSTER=nibi PRED_STORE_ROOT=<dir> STAGE=runs bash submit_mix.sh                  # dry run
#   CLUSTER=nibi PRED_STORE_ROOT=<dir> STAGE=runs DRY_RUN=0 bash submit_mix.sh        # submits
#
# Required: CLUSTER (fir | nibi); PRED_STORE_ROOT (STAGE=runs): the prediction-store root. This
#   script picks no location: on Nibi /scratch refuses writes (2026-11-17), and /project has few
#   free file slots. One 256-region same-view store is about 55 GB and 1,700 files per run-seed
#   (removed by its metrics job); count the concurrent run-seeds before choosing.
# Optional: K (kit; default onion_kit/COMIX); RUNS_ROOT (default runs_fir on Fir, runs on Nibi);
#   RUNGS (default "2 3 4": conv autoencoder, conv + attention, U-Net); NATIVE (default 1: also the
#   native-depth scoring of the fixed-depth arms; 0 skips it).
#
# STAGE=baselines: the 256-region reference terms of eic_v1_2 that every metrics job reads
#   (REFS_FROM = eval/eic_v1_2/baselines): marginal_nb, position_average, depth_only_scaling,
#   pseudoreplicate_floor (score.sh baseline, 16 CPUs, 128000M, 04:00:00; their eic_v1 256-region
#   wall times were 4,568 to 6,322 s) and the oracle reference (oracle_floor.py compute, 2 CPUs).
#   A baseline whose SCORED exists is skipped. Needs eval/eic_v1_2/evalset.json on the cluster.
# STAGE=runs: refuses (exit 3, also in a dry run it prints REFUSED) until the 4 baselines are SCORED
#   and the oracle reference exists. Then, per rung and seed 0 1 2:
#   (1) the reference arm, the base recipe configs/rung<r>/base_heldout.json under the run name
#       eic_v1_2_rung<r>_base_heldout. On Fir that run exists (DONE, Fir 2026-11-17), so only its
#       scoring is submitted; on Nibi it is trained too.
#   (2) every arm of mix/rung<r>/arms.tsv as run mix_<arch>_<arm>.
#   (3) NATIVE=1: every arm whose native_depth_scoring is yes, SCORE_INPUT_DEPTH=native, score only,
#       after its training job (submit_arms.sh finds it in squeue by the run's job name).
#   submit_arms.sh is idempotent: a SCORED score dir, or a live job of the same name, is skipped.
# Training time limit: 08:00:00 with the average input (single arm up to 3.21 h, Fir sacct),
#   06:00:00 with fixed input depth or the NB CRPS loss (up to 2.00 h and 1.07 h), else 04:00:00.
set -u
module load python/3.10.13 >/dev/null 2>&1; unset EBPYTHONPREFIXES
O=/project/def-maxwl/mforooz/ONION
CL="${CLUSTER:?set CLUSTER=fir or CLUSTER=nibi}"
case "$CL" in
  fir) RUNS="${RUNS_ROOT:-$O/runs_fir}"
       export ONION_EXCLUDE=fc10713,fc10715,fc30601,fc30568,fc30605,fc30564,fc30654,fc30435,fc30562,fc30355,fc30570,fc30455,fc30659,fc30565,fc30613,fc30649 ;;
  nibi) RUNS="${RUNS_ROOT:-$O/runs}"; unset ONION_EXCLUDE ;;
  *) echo "CLUSTER must be fir or nibi" >&2; exit 2 ;;
esac
K="${K:-/project/def-maxwl/mforooz/onion_kit/COMIX}"
V=eic_v1_2
T=$O/tables/$V
EV=$O/eval/$V/evalset.json
B=$O/eval/$V/baselines
MIX=$K/tools/minicandi/configs/ablations/mix
export DRY_RUN="${DRY_RUN:-1}"
STAGE="${STAGE:?set STAGE=baselines or STAGE=runs}"
RUNGS="${RUNGS:-2 3 4}"
NATIVE="${NATIVE:-1}"
declare -A ARCH=([2]=convae [3]=convattn [4]=unet)
[ -d "$MIX/rung2" ] || { echo "kit $K has no mix configs" >&2; exit 2; }
for p in "$T/measurements.tsv" "$EV"; do
  [ -e "$p" ] || { echo "missing $p (on $CL)" >&2; exit 2; }
done
BASELINES="marginal_nb position_average depth_only_scaling pseudoreplicate_floor"

case $STAGE in
baselines)
  n=0
  for name in $BASELINES; do
    out=$B/$name/seed0
    [ -e "$out/SCORED" ] && { echo "SCORED $out"; continue; }
    line=(sbatch --parsable --account=def-maxwl --job-name=onion_score_${V}_p256_$name --time=04:00:00
          --cpus-per-task=16 --mem=128000M --gres=none --output=$O/logs/%x_%j.out
          $K/slurm/onion_eval/score.sh $K $EV $out baseline $name)
    if [ "$DRY_RUN" = 1 ]; then echo "DRY ${line[*]}"; else mkdir -p "$out"; echo "$("${line[@]}") $name"; fi
    n=$((n+1))
  done
  OF=$B/oracle_floor/seed0
  if [ -e "$OF/oracle_floor.json" ]; then echo "EXISTS $OF/oracle_floor.json"; else
    BODY="set -eu; module load python/3.10.13; virtualenv --no-download \$SLURM_TMPDIR/venv; \
source \$SLURM_TMPDIR/venv/bin/activate; pip install --no-index --upgrade pip; \
pip install --no-index -r $K/requirements-fir.txt; \
pip install --no-index --find-links $K/wheels -r $K/requirements-pypi.txt; cd $K; \
export PYTHONPATH=$K/src:$K/tools PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2; \
python $K/tools/onion_eval/oracle_floor.py compute --evalset $EV --out $OF --tables $T"
    line=(sbatch --parsable --account=def-maxwl --job-name=onion_oracle_${V}_p256 --time=02:00:00
          --cpus-per-task=2 --mem=16000M --gres=none --output=$O/logs/%x_%j.out --wrap "$BODY")
    if [ "$DRY_RUN" = 1 ]; then echo "DRY $(printf '%q ' "${line[@]}")"; else mkdir -p "$OF"; echo "$("${line[@]}") oracle_floor"; fi
    n=$((n+1))
  fi
  echo "baseline jobs submitted (or printed): $n; DRY_RUN=$DRY_RUN" ;;
runs)
  PS="${PRED_STORE_ROOT:?set PRED_STORE_ROOT explicitly: this script picks no store location}"
  miss=""
  for name in $BASELINES; do [ -e "$B/$name/seed0/SCORED" ] || miss="$miss $name"; done
  [ -e "$B/oracle_floor/seed0/oracle_floor.json" ] || miss="$miss oracle_floor"
  if [ -n "$miss" ]; then
    echo "REFUSED: 256-region reference terms not ready on $V ($B):$miss; run STAGE=baselines first" >&2
    [ "$DRY_RUN" = 1 ] || exit 3
  fi
  export ONION_STAGE=1 CLUSTER=$CL NICE="${NICE:-100}" SEEDS="0 1 2" PRED_STORE_ROOT="$PS"
  export SCORE_SAME_VIEW=1 SCORE_PHASES=1 SCORE_SHARDS=8 SCORE_PRED_DEVICE=cpu SCORE_PRED_CPUS=8 \
    SCORE_PRED_MEM=32000 SCORE_INFER_BATCH=8 SCORE_PRED_TIME=06:00:00 SCORE_CPUS=16 SCORE_MEM=64000 \
    SCORE_TIME=06:00:00 REFS_FROM=$B KEEP_PRED_STORE=0
  SUB() { bash $K/slurm/minicandi/submit_arms.sh $K $T $EV $RUNS "$@" 2>&1 | grep -v "^$"; }
  for r in $RUNGS; do
    a=${ARCH[$r]}
    echo "## rung $r ($a): reference arm"
    TRAIN_TIME=04:00:00 SCORE_SUBDIR=p256_sv SUB $r "${V}_rung${r}_base_heldout=$K/tools/minicandi/configs/rung$r/base_heldout.json"
    echo "## rung $r ($a): mix arms"
    while IFS=$'\t' read -r arm run sw _ _ nd _; do
      [ "$arm" = arm ] && continue
      c=$MIX/rung$r/$arm.json
      case ",$sw," in
        *,average_input,*) tt=08:00:00 ;;
        *,fixed_depth,*|*,nb_crps,*) tt=06:00:00 ;;
        *) tt=04:00:00 ;;
      esac
      m=$(python3 -c "import json,os,sys;d=json.load(open(sys.argv[1]));print(' '.join(d[k] for k in ('average_h5',) if d.get(k) and not os.path.exists(d[k])))" "$c")
      if [ -n "$m" ]; then echo "BLOCKED $run: missing $m"; continue; fi
      TRAIN_TIME=$tt SCORE_SUBDIR=p256_sv SUB $r "$run=$c"
      if [ "$NATIVE" = 1 ] && [ "$nd" = yes ]; then
        SCORE_INPUT_DEPTH=native SCORE_SUBDIR=p256_sv_native_depth SUB $r "$run=$c"
      fi
    done < "$MIX/rung$r/arms.tsv"
  done
  echo "DRY_RUN=$DRY_RUN; cluster $CL; runs root $RUNS; store root $PS" ;;
*) echo "STAGE must be baselines or runs" >&2; exit 2 ;;
esac
