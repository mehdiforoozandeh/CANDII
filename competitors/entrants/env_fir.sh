#!/bin/bash
# The scorer environment on Fir: numpy, scipy, pyBigWig, and nothing else.
#
# Built fresh in $SLURM_TMPDIR per job rather than shared, which is the pattern experiment 005
# adopted after Fir's connection-instability episode: a per-job venv costs 30-120 s and cannot be
# left half-upgraded by another job. It also keeps `pyBigWig` out of the project's own `candi_venv`
# -- the scorer has no business changing the environment CANDI trains in.
#
# No torch on purpose. `pblock_bigwig.py` is a numpy-only port of `candi.bench.partitions` precisely
# so this environment is enough; if something here ever needs `import candi`, the port has been
# bypassed and the parity test no longer covers what is running.
#
# Usage:  source env_fir.sh
set -euo pipefail

module --force purge
module load StdEnv/2023 python/3.11 scipy-stack/2025a

VENV="${SLURM_TMPDIR:-/tmp/$USER}/entrant_scorer_venv"
if [ ! -d "$VENV" ]; then
    python -m venv --system-site-packages "$VENV"
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    pip install --no-index --quiet --upgrade pip
    pip install --no-index --quiet pyBigWig
else
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
fi

python - <<'PY'
import numpy, scipy, pyBigWig
print(f"[env] numpy {numpy.__version__} scipy {scipy.__version__} pyBigWig ok", flush=True)
PY
