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
# The poster is fixed at 44 x 36 in (3168 x 2592 pt), one page (PI, 2026-10-04).
info=$(pdfinfo poster_landscape.pdf)
grep -q "^Page size: *3168 x 2592 pts" <<<"$info" || { echo "FAIL: poster_landscape.pdf is not 44 x 36 in"; exit 1; }
grep -q "^Pages: *1$" <<<"$info" || { echo "FAIL: poster_landscape.pdf is not one page"; exit 1; }
"$PY" make_html.py
grep -h "^POSTER:" poster_landscape.log
echo "wrote poster_landscape.pdf html/poster_landscape.html"
