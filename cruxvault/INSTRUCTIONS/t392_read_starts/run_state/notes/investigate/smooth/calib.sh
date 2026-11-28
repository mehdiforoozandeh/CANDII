#!/bin/bash
#SBATCH --account=def-maxwl
#SBATCH --gres=none
#SBATCH --exclude=fc10713,fc10715
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000M
#SBATCH --output=/project/def-maxwl/mforooz/t392/investigate/smooth/logs/%x_%j.out
set -euo pipefail
D=/project/def-maxwl/mforooz/t392/investigate/smooth
set +u; module load python/3.10.13; set -u
export OMP_NUM_THREADS=1 PYTHONNOUSERSITE=1
virtualenv --no-download "$SLURM_TMPDIR/venv" >/dev/null
source "$SLURM_TMPDIR/venv/bin/activate"
pip install --no-index -q numpy scipy h5py pandas
python $D/smooth_summary.py calib --pred-store /scratch/mforooz/rs392/pred_store/runs/rs_halves_mix_avg_pearson_unet/same_view/seed0/p16_sv \
  --resid $D/out/live_H3K36me3/resid.jsonl $D/out/live_H3K4me3/resid.jsonl $D/out/live_H3K27ac/resid.jsonl \
  --out $D/out/calib_seed0.jsonl --nb-table /project/def-maxwl/mforooz/t392/investigate/conv_sparsity/out/nb_table.npy
python $D/smooth_summary.py calib-table --calib $D/out/calib_seed0.jsonl --out $D/out/calib_seed0_table.json
