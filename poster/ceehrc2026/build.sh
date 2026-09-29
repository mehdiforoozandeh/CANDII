#!/usr/bin/env bash
# Rebuild the poster: draw every panel, then typeset. Run from anywhere.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-/Users/mforooz/miniforge3/envs/candi-local/bin/python}   # needs matplotlib, numpy, pandas
"$PY" split_ga.py
"$PY" make_eic_panels.py
latexmk -pdf -interaction=nonstopmode -halt-on-error poster.tex >/dev/null
echo "wrote poster.pdf"
