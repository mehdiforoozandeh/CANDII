#!/bin/bash
#SBATCH --job-name=ci_example
#SBATCH --account=def-maxwl
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1
#SBATCH --mem=8000M
#SBATCH --time=2:00:00
#
# §7.2's no-retraining correctness check, on the vendor's own example dataset.
#
# EXAMPLE.zip (pubs.broadinstitute.org/jernst/EXAMPLE.zip, 986 MB) ships chr21 Roadmap `-log10 p`
# tracks for eight marks across the tier-1 samples, the distance rankings, and the *pre-trained*
# predictors for one target: E034 H3K9ac. Running `Apply` against those predictors exercises the
# converted-wig reader, the distance files, the predictor format and the wig writer without
# retraining anything, so anything that disagrees is on our side of the line.
#
# What this check cannot do: compare numbers. The manual publishes no expected output for the
# example, and E034 H3K9ac is genuinely unobserved in the shipped compendium, so there is no truth
# track to `Eval` against. The recorded result is therefore the run itself — that the command in
# the quick-start completes in the documented ~20 minutes, and that what it writes is a well-formed
# fixed-step wig of exactly ceil(chrom_len / 25) bins that `collect.read_wig` parses.
set -euo pipefail

E=${CI_EXAMPLE:-$HOME/scratch/t51_chromimpute/EXAMPLE}
JAR=${CI_JAR:-$HOME/scratch/t51_chromimpute/tool/ChromImpute.jar}
OUT=$E/OUTPUTDATA

module load java/21.0.1
unset JAVA_TOOL_OPTIONS          # the module's -Xmx2g would override -mx4000M below
mkdir -p "$OUT"

java -jar "$JAR" Version
START=$SECONDS
java -mx4000M -jar "$JAR" Apply \
    "$E/CONVERTEDDATADIR" "$E/DISTANCEDIR" "$E/PREDICTORDIR" \
    "$E/tier1_samplemarktable.txt" "$E/hg19sizes_chr21.txt" "$OUT" E034 H3K9ac
echo "Apply wall: $((SECONDS - START))s"

python3 - "$OUT/chr21_impute_E034_H3K9ac.wig.gz" "$(cut -f2 "$E/hg19sizes_chr21.txt")" <<'PY'
import gzip, sys
path, length = sys.argv[1], int(sys.argv[2])
head, vals = [], []
with gzip.open(path, "rt") as fh:
    for line in fh:
        if line[:12].lower().startswith(("track", "browser", "fixedstep", "#")):
            head.append(line.rstrip()); continue
        vals.append(float(line))
want = (length - 1) // 25 + 1
print("headers:", head)
print(f"bins: {len(vals)}  expected ceil(len/25): {want}  match: {len(vals) == want}")
print(f"min {min(vals)}  max {max(vals)}  mean {sum(vals)/len(vals):.4f}  "
      f"nonzero {sum(v > 0 for v in vals)}")
PY
