#!/bin/bash
# t138: download the regional-map annotations on a Nibi compute node (PI approved, Decision (iii)).
# Runs tools/t138/annot.sh: UCSC hg38.fa.gz, UCSC rmsk.txt.gz, the Umap k36 bedGraph (or the
# MAPPABILITY_MISSING marker), hg38.chrom.sizes and ANNOT.tsv. bash + curl + gzip; no venv.
#
# NO --export; positional arguments only. The annot dir must be t138 or under it.
#   sbatch --parsable --output=/project/def-maxwl/mforooz/t138/logs/%x_%j.out \
#       $KIT/slurm/t138/annot.sh $KIT /project/def-maxwl/mforooz/t138/annot
#SBATCH --account=def-maxwl
#SBATCH --job-name=t138_annot
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=none
#SBATCH --exclude=c128,c166,c537,c371,c329

set -euo pipefail
USAGE="usage: t138/annot.sh <kit_dir> <annot_dir>"
KIT="${1:?$USAGE}"
ANN="${2:?$USAGE}"
case "$ANN" in
  */t138|*/t138/*) ;;
  *) echo "annot dir $ANN is not t138 or under it; t138 writes only there" >&2; exit 2 ;;
esac
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "bash $KIT/tools/t138/annot.sh $ANN"
  exit 0
fi
echo "=== t138 annot job ${SLURM_JOB_ID:-local} host=$(hostname) $(date -u)"
echo "kit git sha: $(cat "$KIT/GIT_SHA" 2>/dev/null || echo unknown)"
bash "$KIT/tools/t138/annot.sh" "$ANN"
echo "=== done $(date -u)"
