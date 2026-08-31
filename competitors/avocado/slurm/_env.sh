# Sourced by every t50 job. Paths first, so a job that runs somewhere else fails at the source
# line rather than three stages later inside python.
#
# WS is /scratch on purpose: the binned matrices are ~129 GB and the per-chromosome checkpoints
# another ~50 GB, and none of it is a result. What comes back to /project is small -- the training
# logs, the scores, the sigma table. /scratch purges after 60 days; anything still wanted then must
# be re-derived, which is why the commands are all in the README rather than in anybody's history.
set -uo pipefail

REPO="${REPO:-/project/def-maxwl/mforooz/CANDII_t50}"
VENV="${VENV:-/project/def-maxwl/mforooz/candi_venv}"
WS="${WS:-/scratch/$USER/t50_avocado}"
REGIME="${REGIME:-$REPO/configs/regime.eic_val.json}"
AVO="$REPO/competitors/avocado"

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLBACKEND=Agg

[ -d "$VENV" ] || { echo "[env] no venv at $VENV" >&2; exit 1; }
source "$VENV/bin/activate"
[ -d "$AVO" ] || { echo "[env] no competitors/avocado at $AVO" >&2; exit 1; }
mkdir -p "$WS/binned" "$WS/ckpt" "$WS/pred" "$WS/logs"
