#!/usr/bin/env bash
# Rebuild the posters: draw every panel, then typeset the three PDFs, then write
# the two web pages. Run from anywhere.
#   poster.tex            the comprehensive 44 x 82 in draft
#   poster_landscape.tex  44 x 34 in
#   poster_portrait.tex   34 x 44 in
#   html/poster_{landscape,portrait}.html  the same two layouts as web pages
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-/Users/mforooz/miniforge3/envs/candi-local/bin/python}   # needs matplotlib, numpy, pandas
"$PY" split_ga.py
"$PY" make_eic_panels.py
for tex in poster poster_landscape poster_portrait qr; do
  latexmk -pdf -interaction=nonstopmode -halt-on-error "$tex.tex" >/dev/null
done
pdftocairo -svg qr.pdf panels/qr.svg
"$PY" make_html.py
grep -h "^POSTER:" poster_landscape.log poster_portrait.log
echo "wrote poster.pdf poster_landscape.pdf poster_portrait.pdf html/*.html"
