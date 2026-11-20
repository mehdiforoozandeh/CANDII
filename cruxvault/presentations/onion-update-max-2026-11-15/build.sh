#!/bin/bash
# Rebuild report.pdf: figures from the vault result files, then LaTeX.
set -euo pipefail
cd "$(dirname "$0")"
/Users/mforooz/miniforge3/bin/python figs/make_figs.py
latexmk -pdf -interaction=nonstopmode -halt-on-error report.tex >/dev/null
latexmk -c >/dev/null
echo "built $(pwd)/report.pdf"
