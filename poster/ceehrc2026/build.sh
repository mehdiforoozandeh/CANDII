#!/usr/bin/env bash
# Rebuild the poster: draw every panel, then typeset the PDF, then write the web
# page. Run from anywhere.
#   poster_landscape.tex          44 x 34 in
#   html/poster_landscape.html    the same layout as a web page
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-/Users/mforooz/miniforge3/envs/candi-local/bin/python}   # needs matplotlib, numpy, pandas
"$PY" split_ga.py
"$PY" make_eic_panels.py
for tex in poster_landscape qr; do
  latexmk -pdf -interaction=nonstopmode -halt-on-error "$tex.tex" >/dev/null
done
pdftocairo -svg qr.pdf panels/qr.svg
"$PY" make_html.py
grep -h "^POSTER:" poster_landscape.log
echo "wrote poster_landscape.pdf html/poster_landscape.html"
