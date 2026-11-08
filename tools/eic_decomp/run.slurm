#!/bin/bash
# EIC error decomposition (t149): one array task per line of TASKS (tab-separated: <slug> <assay>).
# Reads Max's sprint 04 inputs (group-readable since 2026-11-07); writes $OUT.
#SBATCH --job-name=eic_decomp
#SBATCH --account=def-maxwl_cpu
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --output=/scratch/mforooz/eic_decomp/logs/decomp_%A_%a.out
set -uo pipefail
S04=/scratch/maxwl/epi_imputation/s04
HERE=/scratch/mforooz/eic_decomp/code
OUT=/scratch/mforooz/eic_decomp/out
source $S04/code/score/env.sh
IFS=$'\t' read -r SLUG ASSAY <<< "$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASKS")"
python $HERE/decompose.py --joblist $HERE/joblist.tsv --slug "$SLUG" --assay "$ASSAY" \
	--code $S04/code --cache $S04/cache --avg-bw /project/def-maxwl/mforooz/EIC_REPRO/inputs/average_test \
	--out $OUT
