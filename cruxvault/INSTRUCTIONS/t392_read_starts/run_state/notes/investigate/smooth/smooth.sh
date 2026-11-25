#!/bin/bash
#SBATCH --account=def-maxwl
#SBATCH --gres=none
#SBATCH --exclude=fc10713,fc10715
#SBATCH --output=/project/def-maxwl/mforooz/t392/investigate/smooth/logs/%x_%j.out
set -euo pipefail
D=/project/def-maxwl/mforooz/t392/investigate/smooth
KIT=/project/def-maxwl/mforooz/onion_kit/RS7
RUN=/project/def-maxwl/mforooz/t392/runs/rs_halves_mix_avg_pearson_unet/seed0
set +u; module load python/3.10.13; set -u
virtualenv --no-download "$SLURM_TMPDIR/venv" >/dev/null
source "$SLURM_TMPDIR/venv/bin/activate"
pip install --no-index -q --upgrade pip
pip install --no-index -q -r "$KIT/requirements-fir.txt"
pip install --no-index -q --find-links "$KIT/wheels" -r "$KIT/requirements-pypi.txt"
cd "$KIT"
source "$KIT/slurm/_kit_pin.sh"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$KIT/src:$KIT:$KIT/tools:$KIT/tests"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python $D/smooth_skill.py --run $RUN --evalset /project/def-maxwl/mforooz/ONION/eval/eic_v1_2/evalset_p16.json \
  --assays $ASSAYS --out $D/out/$OUTNAME --regions $REG ${MAXI:+--max-items $MAXI} --threads $SLURM_CPUS_PER_TASK \
  --pred-store /scratch/mforooz/rs392/pred_store/runs/rs_halves_mix_avg_pearson_unet/same_view/seed0/p16_sv ${FROMSTORE:+--from-store}
