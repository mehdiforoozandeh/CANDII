#!/bin/bash
#SBATCH --job-name=h26_halves
#SBATCH --account=def-maxwl
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000M
#SBATCH --time=1:00:00
#SBATCH --output=/project/def-maxwl/mforooz/h26/logs/%x_%j.out
# h26 step 1, on Nibi: chr21 counts of the t117 halves the probe needs, into one small npz.
set -euo pipefail
H26=/project/def-maxwl/mforooz/h26
module load python/3.11 scipy-stack
python3 "$H26/code/tools/h26/extract_halves.py" "$H26/halves_chr21.npz"
