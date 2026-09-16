#!/bin/bash
#SBATCH --job-name=ci_gwprobe
#SBATCH --account=def-maxwl
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1
#SBATCH --mem=16000M
#SBATCH --time=6:00:00
#
# The genome-wide half of the old §7.2 pilot gate: one track, all 23 chromosomes, through `prepare`
# and `Convert`, timed and sized.
#
# HISTORICAL AS OF 2026-08-31. §4 blanks ChromImpute's `genome-wide` cell and rules that a blanked
# cell is not computed, so no genome-wide ChromImpute run is planned any more. Kept because it is
# the only measurement of `Convert`'s cost per chromosome, which still sizes the eval-scope run.
#
# Why one *track* and not the "one genome-wide pair" §7.2 asks for: `Apply -c chrom` needs the whole
# compendium converted for that chromosome, because it opens a reader per (mark, cell) in
# `inputinfofile`. So a single genome-wide pair costs the same `Convert` as all 45 of them — the
# entry price for P2 is 267 tracks × 23 chromosomes of `Convert`, and one track measures it exactly.
# Everything downstream of `Convert` is then priced off the chr21 pilot times the bin ratio, which
# is 64.9 (121 241 684 genome bins / 1 868 399 chr21 bins).
set -euo pipefail

REPO=${CI_REPO:-/project/def-maxwl/mforooz/CANDII_main}
STORE=${CI_STORE:-/project/def-maxwl/mforooz/CANDI_STORE/eic}
PY=${CI_PY:-/project/def-maxwl/mforooz/EpiDenoise/candi_venv/bin/python}
REGIME=${CI_REGIME:-$REPO/configs/regime.eic_19.json}
JAR=${CI_JAR:-/project/def-maxwl/mforooz/tools/ChromImpute.jar}
RUN=${CI_RUN:-/scratch/mforooz/t81_rivals/ChromImpute/gw_probe}
SAMPLE=${CI_SAMPLE:-T_K562}
MARK=${CI_MARK:-H3K4me3}

module load java/21.0.1
unset JAVA_TOOL_OPTIONS
mkdir -p "$RUN/CONVERTEDDIR"

START=$SECONDS
PYTHONPATH=$REPO/src $PY "$REPO/competitors/chromimpute/prepare.py" --store "$STORE" \
    --regime "$REGIME" --out "$RUN/input" --chroms all \
    --only-sample "$SAMPLE" --only-mark "$MARK" --force
T_PREP=$((SECONDS - START))
echo "prepare(1 track, 23 chroms): ${T_PREP}s"
du -sb "$RUN/input/signal" | awk '{print "bedgraph bytes:", $1}'

START=$SECONDS
for C in $(cut -f1 "$RUN/input/chrominfo.txt"); do
  java -mx8000M -jar "$JAR" Convert -c "$C" -l "$SAMPLE" -m "$MARK" \
      "$RUN/input/signal" "$RUN/input/inputinfofile.txt" "$RUN/input/chrominfo.txt" \
      "$RUN/CONVERTEDDIR"
done
T_CONV=$((SECONDS - START))
echo "Convert(1 track, 23 chroms): ${T_CONV}s"
du -sb "$RUN/CONVERTEDDIR" | awk '{print "converted bytes:", $1}'

printf 'prepare_s\t%s\nconvert_s\t%s\nsample\t%s\nmark\t%s\n' \
    "$T_PREP" "$T_CONV" "$SAMPLE" "$MARK" > "$RUN/timing.tsv"
