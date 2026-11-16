#!/bin/bash
# Close-out D4, Fir port (PI ruling 2026-11-16/17: blind only = batch 1 of closeout_d4_submit.sh).
# Builds, on Fir, the 51 blind base rows of the EIC build list and the 11 blind control experiments
# they need (rows 496-557), and optionally the 10 DNase-seq rows that failed with NODE_FAIL in the
# Nibi outage of 2026-11-15. Dry run is the default. Runs on a Fir login node:
#
#   ssh -o BatchMode=yes fir 'bash -s -- blind reruns'           < slurm/onion_data/closeout_d4_submit_fir.sh
#   ssh -o BatchMode=yes fir 'DRY_RUN=0 bash -s -- blind reruns' < slurm/onion_data/closeout_d4_submit_fir.sh
#
# Arguments: the parts to handle, "blind", "reruns" or both.
#   blind   62 rows: 59 ChIP-seq/ATAC-seq rows (incl. the 11 controls) in ONE call (3 days; the
#           signal pass waits for the controls), and the 3 blind DNase-seq rows (7 days)
#   reruns  EIC rows 113, 211, 216, 232, 384, 385, 387 and second-experiment rows 2049, 2075, 2079
#           (7 days; 384, 385, 387, 2079 at 192000M). Their Nibi state is input.json and
#           encode_metadata.json in work/<experiment>/base only (no ALIGN_DONE, no base measurement),
#           so the Fir run starts from zero, as a Nibi rerun would. Do not also rerun them on Nibi.
#
# The code is the Nibi build code byte for byte: every call is `build_row.sh submit` of the Fir kit
# COD4b, and the script refuses unless the kit's build files have the sha256 of kits KF4/KX4
# (git 0d578dc; KX4 = 2f79ec4 has the same build files). The kit paths are hard-coded to the same
# absolute paths on both clusters (/project/def-maxwl -> /project/6014832 on both), so the outputs
# land at the same paths as on Nibi: /scratch/mforooz/onion/{fastq,work,measurements}.
# The Caper/Cromwell execution tree lives in the job's $SLURM_TMPDIR (node-local, 7.2 TB per node
# on Fir) and dies with the job; only harvested keepers reach /scratch (about 100 files per row).
#
# Prerequisite: the Globus copies Nibi -> Fir listed in cruxvault/results/closeout_D4b/FIR_PORT.md.
# A dry run lists every missing prerequisite and refuses; it is safe to run at any time.
# Refuses: off Fir; kit COD4b missing or its build files differ from KF4; any prerequisite missing or
# with the wrong hash (pipeline images, build lists, genome sizes); /scratch above MAX_FILES_K
# thousand files or MAX_GIB GiB (diskusage_report).
#
# Env: DRY_RUN (default 1), CONC (build tasks per array, default 50), SCONC (stage tasks per array,
#      default 2), MAX_FILES_K (default 970: the run adds at most about 9.1K files to /scratch, the
#      limit is 1,000K), MAX_GIB (default 15360 = 15 TiB: the run adds at most about 3.3 TiB, the
#      limit is 19 TiB), FORCE=1 skips the two /scratch guards only.
set -uo pipefail
DRY_RUN="${DRY_RUN:-1}"; CONC="${CONC:-50}"; SCONC="${SCONC:-2}"; FORCE="${FORCE:-0}"
MAX_FILES_K="${MAX_FILES_K:-970}"; MAX_GIB="${MAX_GIB:-15360}"
PARTS="${*:-}"
[ -n "$PARTS" ] || { echo "usage: closeout_d4_submit_fir.sh blind|reruns [blind|reruns]"; exit 2; }
for p in $PARTS; do case "$p" in blind|reruns) ;; *) echo "unknown part '$p' (blind, reruns)"; exit 2;; esac; done

O=/project/def-maxwl/mforooz/ONION
KIT=/project/def-maxwl/mforooz/onion_kit/COD4b
EIC=/project/def-maxwl/mforooz/EIC_REPRO/003_pipeline
B=$O/eic/build_list.tsv; BX=$O/eic/other_experiment_build_list.tsv
M=/scratch/mforooz/onion/measurements; W=/scratch/mforooz/onion/work
B_SHA=33c077f464138c57cbe31827857061b1524a22ed571d00a7ab8b7de81ddc0b83
BX_SHA=267443cdf86c4f6aea619da6bbe06f77d810381c5eec532b5326f8bca7ea87bc

# --- one line per submit call: part|list|priority|time|mem|rows|label (rows and limits as Nibi)
CALLS=(
"blind|B|4|3-00:00:00||496-500,502-520,522-555,557|blind ChIP-seq and ATAC-seq base rows + their 11 controls (one call: the signal pass waits for the controls)"
"blind|B|4|7-00:00:00||501,521,556|blind DNase-seq base rows (paired-end)"
"reruns|B|2|7-00:00:00||113,211,216,232|outage rerun: EIC training DNase-seq (rule B.5: 7 days)"
"reruns|B|2|7-00:00:00|192000M|384,385,387|outage rerun: EIC training DNase-seq (as their last run: 192000M)"
"reruns|BX|5|7-00:00:00||2049,2075|outage rerun: second-experiment DNase-seq"
"reruns|BX|5|7-00:00:00|192000M|2079|outage rerun: second-experiment DNase-seq, 1,115 M reads"
)

# --- build code of kits KF4/KX4 (sha256 of git 0d578dc; checked equal on Nibi KF4 2026-11-16)
PINS=(
"7926706614e5879bc7b9e81ddcf15b6e2fdba411bb1e7d9fe0dd52ae03132784 tools/onion_data/fastq_uniform.py"
"aff4099adf92bf13174893f36eef098ddb475779073d6b39ffe94bb5c1e32186 tools/onion_data/fastq_dnase.py"
"5ed429eb6c96eff7c66361ee8c852953bbd735c5d2cd8b6a0786d133a183ecea tools/onion_data/align_views.py"
"4247358d1918f49d4b46d12764278e787696827edc9b418b52010c6ffe6a485a tools/onion_data/build_list.py"
"99d1210cd119a115d931f9672610cba045f7f89393f88e90a93be19f31acad09 slurm/onion_data/build_row.sh"
"84cd159b9494eabb2c4a2b860bcab75b3ad33cdd01cb8b15be058488ee966f5b slurm/onion_data/stage_fastq.sh"
"d33f0f4011ed5a56341972b8c36cbe121160188298c51f5e78089d7462b2fe17 slurm/_kit_pin.sh"
"dbaadfa090d86de215b42ee997d6507c268d75ee4e7d998ca51cbb74e42b4d8f tools/t112/bam_arm.py"
"bceba4cf41c1f73410557f9e41068a11d9ecf2e354421e71e9f9ec32a58b10c3 tools/t112/arms.py"
"48682ede46c657f9c0acd02717eb6a5b4795f4b4e4a62c5c26d0a82e1e79a883 tools/t112/records.py"
"ea960f8a9329cef1dc9c66065dd243a89ec3fb282ea102d74ff7add21464177a tools/t112/bin25.py"
"38c80ad43e122b07f81a88d3e95becee2a8fe3ddaa133f0cd4d5e91a571f31ce tools/t117/pseudoreps.py"
"8a38f4fd741afcf90c6b09e314509a9adcd44b6034bd9f43f944b4083549abc0 requirements-fir.txt"
"49f3e748436304205421ff7b3d16127f44b44cc80e903f42805b4f0aa83636ab requirements-pypi.txt"
)

# --- prerequisites copied from Nibi (Globus), at the same paths: kind|path|expected
#     kind: f = file exists, d = directory not empty, md5 / sha256 = file with that hash
REQS=(
"f|$EIC/runner/env.sh|"
"f|$EIC/caper_venv/bin/caper|"
"f|$EIC/chip-seq-pipeline2/chip.wdl|"
"f|$EIC/atac-seq-pipeline/atac.wdl|"
"md5|$EIC/sif/chip-seq-pipeline_v2.2.2.sif|f6e408f7e77bafde4556883f33552191"
"md5|$EIC/sif/atac-seq-pipeline_v2.2.3.sif|04d9fa482d67cf633845653750f58cef"
"f|$HOME/.caper/cromwell_jar/cromwell-82.jar|"
"f|$HOME/.caper/womtool_jar/womtool-82.jar|"
"f|$O/pipelines/hg38/hg38.local.tsv|"
"f|$O/pipelines/hg38/8c692fba4640609720272154ab0faa30/ENCFF643CGH.tar.gz|"
"sha256|$O/pipelines/dnase/dnase-seq-pipeline_v3.0.0.sif|2cd73eb14798f90a3372eafdd5549768c0ae705da9c1a3d054cc0e6921680d1d"
"f|$O/pipelines/dnase/dnase-seq-pipeline_v3.0.0.tar.gz|"
"f|/scratch/mforooz/onion/pipelines/dnase/refs/GRCh38.fa.bwt|"
"f|/scratch/mforooz/onion/pipelines/dnase/refs/GRCh38.fa.fai|"
"f|/scratch/mforooz/onion/pipelines/dnase/refs/nuclear_chroms.txt|"
"f|/scratch/mforooz/EIC_REPRO/003/refcache/c52f52c7bfa357f55a39b1de7e4d0b0c/GRCh38_EBV.chrom.sizes.tsv|"
"sha256|$O/genome/chrom_sizes.json|5c2409a37a1cb7b0d69735711c0ff616742ea7667dac4f0211a8dae94a9cd3e2"
"f|$O/catalog/encode/files.tsv|"
"sha256|$B|$B_SHA"
"sha256|$BX|$BX_SHA"
)

die() { echo "REFUSED: $*"; exit 3; }
case "$(hostname -f 2>/dev/null)" in *.fir.*) ;; *) die "run this on a Fir login node";; esac

echo "=== closeout D4 Fir build $(date -u +%FT%TZ) parts: $PARTS  DRY_RUN=$DRY_RUN CONC=$CONC SCONC=$SCONC"
bad=0
[ -f "$KIT/.kit_stamp" ] && echo "kit: $(cat "$KIT/.kit_stamp")" || { echo "MISSING kit $KIT"; bad=1; }
for p in "${PINS[@]}"; do
  want="${p%% *}"; f="${p#* }"
  got=$(sha256sum "$KIT/$f" 2>/dev/null | cut -d' ' -f1)
  [ "$got" = "$want" ] || { echo "KIT DIFFERS from KF4: $f (${got:-missing})"; bad=1; }
done
for r in "${REQS[@]}"; do
  IFS='|' read -r kind path want <<< "$r"
  case "$kind" in
    f) [ -s "$path" ] || { echo "MISSING $path"; bad=1; } ;;
    md5|sha256)
      if [ ! -s "$path" ]; then echo "MISSING $path"; bad=1
      else got=$(${kind}sum "$path" | cut -d' ' -f1)
           [ "$got" = "$want" ] || { echo "WRONG $kind $path: $got != $want"; bad=1; }; fi ;;
  esac
done
[ "$bad" = 0 ] || die "prerequisites missing or changed (above); see cruxvault/results/closeout_D4b/FIR_PORT.md"
echo "prerequisites: all present, hashes match"

# --- /scratch guards (diskusage_report: "/scratch (user ...)  4647GiB/  19TiB   934K/1000K")
DU=$(diskusage_report 2>/dev/null \
     | awk '/\/scratch \(user/ {for (i = 4; i <= NF; i++) if ($i ~ /[0-9]+(\.[0-9]+)?[KMGT]iB\/?$/) {print $i, $NF; exit}}')
USED_GIB=$(echo "$DU" | awk '{v=$1; sub(/\/$/,"",v); n=v+0; if (v ~ /TiB/) n*=1024; else if (v ~ /MiB/) n/=1024; printf "%d", n}')
FILES_K=$(echo "$DU" | awk '{split($2,a,"/"); v=a[1]; n=v+0; if (v ~ /M$/) n*=1000; else if (v !~ /K$/) n/=1000; printf "%d", n}')
echo "/scratch: ${USED_GIB:-?} GiB, ${FILES_K:-?}K files (refuse above $MAX_GIB GiB or ${MAX_FILES_K}K files)"
if [ "$FORCE" != 1 ]; then
  [ -n "$USED_GIB" ] && [ -n "$FILES_K" ] || die "cannot read diskusage_report (FORCE=1 to override)"
  [ "$FILES_K" -le "$MAX_FILES_K" ] || die "/scratch holds ${FILES_K}K files; the run adds up to about 9.1K (limit 1,000K)"
  [ "$USED_GIB" -le "$MAX_GIB" ] || die "/scratch holds $USED_GIB GiB; the run adds up to about 3.3 TiB (limit 19 TiB)"
fi

ACTIVE=$(squeue -u "$USER" -h -r -n onion_build,onion_signal,onion_stage -o '%K' 2>/dev/null \
         | grep -E '^[0-9]+$' | sort -un | paste -sd, -)
echo "queued or running onion_* array tasks on Fir: ${ACTIVE:-none}"

LOG=/dev/null
if [ "$DRY_RUN" = 0 ]; then
  LOG=$O/logs/closeout_d4_fir_submit_$(date -u +%Y%m%dT%H%M%SZ).log
  echo "submit record: $LOG"
fi
for c in "${CALLS[@]}"; do
  IFS='|' read -r PART LIST PRI TIME MEM ROWS LABEL <<< "$c"
  case " $PARTS " in *" $PART "*) ;; *) continue;; esac
  L=$B; [ "$LIST" = BX ] && L=$BX
  echo "--- $PART: $LABEL"
  echo "    kit COD4b, list $(basename "$L"), priority $PRI, time $TIME, mem ${MEM:-64000M (header)}, rows $ROWS"
  if [ "$DRY_RUN" != 0 ]; then
    python3 "$KIT/tools/onion_data/build_list.py" pending --build-list "$L" --priority "$PRI" --rows "$ROWS" \
      --out "$M" --work "$W" --files "$O/catalog/encode/files.tsv" ${ACTIVE:+--active "$ACTIVE"} 2>&1 >/dev/null \
      | awk '{n[$1]++} /^skip/{print "    " $0} END{printf "    pending check: %d submit, %d skip\n", n["submit"], n["skip"]}'
    env ONION_SBATCH_TEST=1 ONION_TIME="$TIME" ${MEM:+ONION_MEM=$MEM} \
      bash "$KIT/slurm/onion_data/build_row.sh" submit "$KIT" "$L" "$PRI" "$CONC" "$SCONC" "$ROWS" 2>/dev/null \
      | sed 's/^/    /'
  else
    { echo "--- $PART: $LABEL ($(date -u +%FT%TZ))"
      env ONION_TIME="$TIME" ${MEM:+ONION_MEM=$MEM} \
        bash "$KIT/slurm/onion_data/build_row.sh" submit "$KIT" "$L" "$PRI" "$CONC" "$SCONC" "$ROWS" 2>&1; } \
      | tee -a "$LOG" | grep -v '^skip\|^submit row'
  fi
done
echo "=== done. Dry run: $([ "$DRY_RUN" = 0 ] && echo no || echo 'yes (nothing submitted)')"
