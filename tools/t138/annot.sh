#!/bin/bash
# t138: download the public annotations the regional map needs (PI approved, Decision (iii)).
#
#   bash tools/t138/annot.sh <annot_dir>
#
# Runs inside slurm/t138/annot.sh on a Nibi compute node (internet access). Writes into <annot_dir>:
#   hg38.fa.gz            UCSC hg38 genome fasta (~940 MB)
#   rmsk.txt.gz           UCSC RepeatMasker table (~160 MB)
#   k36.umap.bedgraph.gz  Umap k36 multi-read mappability (fallback name k36.umap.bed.gz); if
#                         neither URL exists, the file MAPPABILITY_MISSING is written instead and
#                         the map runs on three features
#   hg38.chrom.sizes      chromosome lengths computed from hg38.fa.gz (awk, no samtools)
#   ANNOT.tsv             file, url, bytes, sha256 of every downloaded file
# A file already present is not downloaded again. Needs bash, curl, gzip, awk, sha256sum only.
set -euo pipefail
ANN="${1:?usage: annot.sh <annot_dir>}"
mkdir -p "$ANN"
UCSC=https://hgdownload.soe.ucsc.edu/goldenPath/hg38
UMAP=https://bismap.hoffmanlab.org/raw/hg38

fetch() {  # fetch <url> <dest>; returns curl's status, leaves no partial file
  local url="$1" dest="$2"
  if [[ -s "$dest" ]]; then echo "have $dest"; return 0; fi
  echo "get $url"
  if curl -fL --retry 3 -o "$dest.part" "$url"; then
    mv "$dest.part" "$dest"
  else
    local rc=$?
    rm -f "$dest.part"
    return $rc
  fi
}

fetch "$UCSC/bigZips/hg38.fa.gz" "$ANN/hg38.fa.gz"
fetch "$UCSC/database/rmsk.txt.gz" "$ANN/rmsk.txt.gz"
ROWS=("hg38.fa.gz	$UCSC/bigZips/hg38.fa.gz" "rmsk.txt.gz	$UCSC/database/rmsk.txt.gz")

rm -f "$ANN/MAPPABILITY_MISSING"
if fetch "$UMAP/k36.umap.bedgraph.gz" "$ANN/k36.umap.bedgraph.gz"; then
  ROWS+=("k36.umap.bedgraph.gz	$UMAP/k36.umap.bedgraph.gz")
elif fetch "$UMAP/k36.umap.bed.gz" "$ANN/k36.umap.bed.gz"; then
  ROWS+=("k36.umap.bed.gz	$UMAP/k36.umap.bed.gz")
else
  echo "neither $UMAP/k36.umap.bedgraph.gz nor $UMAP/k36.umap.bed.gz could be fetched" \
    > "$ANN/MAPPABILITY_MISSING"
  echo "MAPPABILITY_MISSING written"
fi

if [[ ! -s "$ANN/hg38.chrom.sizes" ]]; then
  gzip -dc "$ANN/hg38.fa.gz" | awk '
    /^>/ { if (n != "") print n "\t" l; n = substr($1, 2); l = 0; next }
    { sub(/\r$/, ""); l += length($0) }
    END { if (n != "") print n "\t" l }' > "$ANN/hg38.chrom.sizes.part"
  mv "$ANN/hg38.chrom.sizes.part" "$ANN/hg38.chrom.sizes"
fi

{
  printf 'file\turl\tbytes\tsha256\n'
  for row in "${ROWS[@]}"; do
    f="${row%%	*}"; u="${row#*	}"
    printf '%s\t%s\t%s\t%s\n' "$f" "$u" "$(stat -c %s "$ANN/$f")" \
      "$(sha256sum "$ANN/$f" | cut -d' ' -f1)"
  done
} > "$ANN/ANNOT.tsv"
cat "$ANN/ANNOT.tsv"
