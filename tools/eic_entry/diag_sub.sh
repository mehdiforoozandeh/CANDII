#!/bin/bash
# diag_sub.sh <track ...>: one diag_scale.py job per track on Fir (16G, 4 CPUs, 1 h).
O=/scratch/mforooz/eicx/val/rcp_combo_w2x_unet_s0_raw
P=/project/def-maxwl/mforooz/ONION/runs_rcp/eic_val/rcp_combo_w2x_unet_s0
C=/scratch/mforooz/eicx/entry/code
ENV='source /etc/profile >/dev/null 2>&1; export EIC_CLUSTER=fir; module load StdEnv/2023 python/3.11 scipy-stack; export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4; virtualenv -q --no-download --system-site-packages $SLURM_TMPDIR/venv; source $SLURM_TMPDIR/venv/bin/activate; pip install -q --no-index h5py'
mkdir -p $O/logs
for T in "$@"; do
  J=$(sbatch --parsable --account=def-maxwl --gres=none --job-name=eicx_entry_diag_$T --cpus-per-task=4 --mem=16G --time=1:00:00 \
    --output=$O/logs/%x_%j.out --wrap="$ENV; set -euo pipefail; python $C/tools/eic_entry/diag_scale.py --pred $P --cov $O/covariates.tsv --track $T --out $O")
  echo "$T $J" | tee -a $O/DIAG_JOBS.txt
done
