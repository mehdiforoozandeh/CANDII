#!/bin/bash
#SBATCH --job-name=candii_peak_smoke
#SBATCH --account=def-maxwl
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=0:50:00
#SBATCH --output=/scratch/mforooz/candii_merge/peak_smoke_%j.out
#SBATCH --error=/scratch/mforooz/candii_merge/peak_smoke_%j.err
set -uo pipefail

REPO=/project/6014832/mforooz/CANDII_merge
H5=/scratch/mforooz/candi_kit/gatec.h5
OUT=/scratch/mforooz/candii_merge/runs
mkdir -p "$OUT"

source /project/6014832/mforooz/EpiDenoise/candi_venv/bin/activate
cd "$REPO"
export PYTHONPATH=src
export WANDB_MODE=disabled

# The dead-GPU footgun (issue #3) is exactly what would make this smoke meaningless: a CPU fallback
# would run, print finite losses, and prove nothing about bf16. Fail loudly instead.
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
  echo "[smoke] FATAL: no CUDA on this node — refusing to run a GPU check on CPU"; exit 2; }
nvidia-smi -L

rc=0

echo; echo "################ 1. peak-head bf16 checks ################"
python tools/peak_check.py || rc=1

run () {  # name, extra flags...
  local name="$1"; shift
  echo; echo "################ $name ################"
  python -m candi.train --h5 "$H5" --out-dir "$OUT" --tag "$name" \
      --epochs 1 --steps-per-epoch 20 --batch-size 2 --lr 5e-4 --seed 0 \
      --eval-batch-size 2 --eval-max-batches 1 --eval-budget 5000 \
      --n-boot 20 --m3-regions 1 --eval-every 0 "$@" 2>&1 | tail -25
  local s=${PIPESTATUS[0]}
  [ "$s" -eq 0 ] || { echo "[smoke] $name EXITED $s"; rc=1; }
}

run default_fp32                                            # must behave as before
run allheads_fp32 --heads count,signal,peak --precision fp32
run allheads_bf16 --heads count,signal,peak --precision bf16
run peakonly_bf16 --heads count,peak         --precision bf16

echo; echo "################ 2. verdict on the written run JSONs ################"
python - <<'PY'
import glob, json, math, sys
bad = 0
for f in sorted(glob.glob("/scratch/mforooz/candii_merge/runs/*.json")):
    d = json.load(open(f))
    losses = d.get("train_losses") or []
    finite = all(isinstance(x, (int, float)) and math.isfinite(x) for x in losses)
    m1 = ((d.get("M1") or {}).get("imp") or {}).get("crps")
    cfg = d.get("config") or {}
    print("  %-46s steps=%3d finite=%s  heads=%-22s precision=%-5s  imp_crps=%s"
          % (f.rsplit("/", 1)[-1], len(losses), finite,
             ",".join((cfg.get("arch") or {}).get("heads", [])) or "?",
             cfg.get("precision", "?"),
             ("%.4f" % m1) if isinstance(m1, float) and math.isfinite(m1) else m1))
    if not finite or not losses:
        print("      ^^ NON-FINITE OR EMPTY LOSS CURVE"); bad += 1
sys.exit(1 if bad else 0)
PY
[ $? -eq 0 ] || rc=1

echo; echo "[smoke] rc=$rc"
exit $rc
