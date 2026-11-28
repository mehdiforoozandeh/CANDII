#!/bin/bash
# depth_sub.sh <train|val> <track ...>: one diag_depth.py job per track on Fir (8G, 2 CPUs, 30 min).
O=/scratch/mforooz/eicx/val/depth_rule; C=/scratch/mforooz/eicx/entry/code
case $1 in train) COV=/scratch/mforooz/eicx/val/train_cov.tsv; TR=/scratch/mforooz/eicx/val/train_truth ;;
           val) COV=/scratch/mforooz/eicx/val/cov.tsv; TR=/project/def-maxwl/mforooz/DATA_EIC_SYNAPSE/validation_data ;; esac
SET=$1; shift
ENV='source /etc/profile >/dev/null 2>&1; export EIC_CLUSTER=fir; module load StdEnv/2023 python/3.11 scipy-stack; export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2; virtualenv -q --no-download --system-site-packages $SLURM_TMPDIR/venv; source $SLURM_TMPDIR/venv/bin/activate; pip install -q --no-index h5py'
mkdir -p $O/logs $O/$SET
for T in "$@"; do
  J=$(sbatch --parsable --account=def-maxwl --gres=none --job-name=eicx_entry_depth_$T --cpus-per-task=2 --mem=8G --time=0:30:00 \
    --output=$O/logs/%x_%j.out --wrap="$ENV; set -euo pipefail; python $C/tools/eic_entry/diag_depth.py --cov $COV --track $T --truth $TR --out $O/$SET")
  echo "$SET $T $J" | tee -a $O/JOBS.txt
done
