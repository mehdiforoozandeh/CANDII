#!/bin/bash
# t25 step 3 — PVAL_CODEC_PLAN.md section 6, as a job. Exits nonzero if the corpus did not
# actually get the codec it was supposed to get.
#SBATCH --account=def-maxwl
#SBATCH --job-name=t25_gate
#SBATCH --output=/project/def-maxwl/mforooz/CANDI_STORE/t25/logs/gate_%j.out
#SBATCH --error=/project/def-maxwl/mforooz/CANDI_STORE/t25/logs/gate_%j.err
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
set -uo pipefail
STORE=/project/def-maxwl/mforooz/CANDI_STORE
KIT="${KIT:-/home/mforooz/projects/def-maxwl/mforooz/CANDII_t25}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1; unset PYTHONPATH || true
source /project/6014832/mforooz/EpiDenoise/candi_venv/bin/activate
cd "$KIT"; export PYTHONPATH="$KIT/src"
echo "[t25] gate host=$(hostname) kit=$(git -C "$KIT" rev-parse --short HEAD 2>/dev/null) $(date -Is)"
rc=0
for C in eic merged eic_slice; do
  echo "======================= $C ======================="
  python "$KIT/tools/pval_codec_scan.py" --corpus-root "$STORE/$C" \
      --json "$STORE/t25/${C}_codec_scan.json" > /dev/null || rc=1
done
echo "================= round trip, EIC ================="
# B_SJCRH30/H3K4me3 is the worst-clipped track in the corpus (0.371%); ATAC-seq is the assay
# where all seven tracks clipped, and those seven live in B_DND-41, B_NCI-H929, B_RWPE2,
# T_adrenal_gland, T_heart_left_ventricle, T_omental_fat_pad and T_testis. The first submission
# read `T_DND-41`, which carries no ATAC-seq at all, so the gate failed on a FileNotFoundError
# against the source rather than on anything about the corpus. Name a biosample from that list.
# n_above_old_ceiling must be > 0 or the round trip is not
# testing what it claims to.
python "$KIT/tools/pval_codec_scan.py" --corpus-root "$STORE/eic" \
    --roundtrip --source-root /project/6014832/mforooz/DATA_CANDI_EIC \
    --spot B_SJCRH30/H3K4me3 --spot B_DND-41/ATAC-seq \
    --json "$STORE/t25/eic_roundtrip.json" > /dev/null || rc=1
echo "[t25] GATE rc=$rc $(date -Is)"
exit $rc
