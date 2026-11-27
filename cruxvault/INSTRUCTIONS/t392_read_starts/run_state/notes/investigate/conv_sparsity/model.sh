#!/bin/bash
#SBATCH --account=def-maxwl
#SBATCH --gres=none
#SBATCH --exclude=fc10713,fc10715
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000M
#SBATCH --output=/project/def-maxwl/mforooz/t392/investigate/conv_sparsity/logs/%x_%j.out
set -euo pipefail
D=/project/def-maxwl/mforooz/t392/investigate/conv_sparsity
set +u; module load python/3.10.13; set -u
export OMP_NUM_THREADS=1 PYTHONNOUSERSITE=1
virtualenv --no-download "$SLURM_TMPDIR/venv" >/dev/null
source "$SLURM_TMPDIR/venv/bin/activate"
pip install --no-index -q numpy scipy h5py pandas
python -c "import numpy, scipy, h5py, pandas; print(numpy.__version__, scipy.__version__, h5py.__version__)"
python $D/conv_sparsity.py model --pred-store /scratch/mforooz/rs392/pred_store/runs/rs_halves_mix_avg_pearson_unet/same_view/seed$SEED/p16_sv --out $D/out/model_seed$SEED.jsonl --nb-table $D/out/nb_table.npy
