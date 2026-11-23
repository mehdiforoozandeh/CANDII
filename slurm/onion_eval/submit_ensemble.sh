#!/bin/bash
# Task t370 (miniCANDI recipe run), chunk ens: seed ensemble and NB dispersion recalibration of
# finished recipe runs, scored with the options of submit_recipe.sh (no training). Run on a Fir
# login node. DRY RUN BY DEFAULT: prints the sbatch lines; DRY_RUN=0 submits.
#
#   K=<kit dir> RUNS="<run name> ..." SUB=p16|p256 [VARIANTS="ens3 ens3_recal seed0_recal"]
#     [PRED_TIME= FIT_TIME= COMB_TIME= COMB_MEM= MET_TIME=] bash submit_ensemble.sh
#
# Per run (dir $O/runs_rcp/<run>/seed{0,1,2}, never written):
#   predict  slurm/onion_eval/predict.sh of each seed a variant needs, into the new store
#            $PS/<run>/<SD>/seed<s> (SD = p16_sv or p256_sv; p16: 4 shards, eval set
#            evalset_p16.json; p256: 8 shards, evalset.json); the predict options of
#            submit_recipe.sh (cpu, 8 CPUs, 32000M, batch 8, same view, staged). A complete store
#            is not predicted again.
#   fit      once per run (both tiers): ensemble.sh fit on 32 calibration regions outside the
#            regions and fit_regions of evalset_p16.json and evalset.json, items of
#            evalset_p16.json -> $O/runs_rcp_ens/<run>/recal/factors.json (skipped when present).
#   combine  per variant: ens3 = seeds 0, 1, 2 moment-matched; ens3_recal = the same with the
#            "ensemble" factors; seed0_recal = seed 0 with the "seed0" factors -> the store
#            $PS/<run>/<variant>_<SD>.
#   metrics  score.sh metric phase (16 CPUs, 64000M, REFS_FROM of the tier, same view) with
#            --run <run>/seed0 -> $O/runs_rcp_ens/<run>/<variant>_<SD>/scores.json; score.sh removes
#            the combined store after SCORED (its basename is the score dir's).
# A variant with SCORED is skipped. Job lines go to $O/runs_rcp_ens/SUBMITTED.tsv.
set -u
set -o pipefail
O="${O:-/project/def-maxwl/mforooz/ONION}"
K="${K:?set K=<kit dir>}"
V=eic_v1_2
SUBT="${SUB:?set SUB=p16 or p256}"
RUNS="${RUNS:?set RUNS=<run names>}"
VARIANTS="${VARIANTS:-ens3 ens3_recal seed0_recal}"
DRY="${DRY_RUN:-1}"
PS="${PRED_STORE_ROOT:-/scratch/mforooz/onion/pred_store}/runs_rcp_ens"
OUT="$O/runs_rcp_ens"
EV16=$O/eval/$V/evalset_p16.json
EV256=$O/eval/$V/evalset.json
case $SUBT in
  p16) EV=$EV16; B=$O/eval/$V/baselines_p16; NSH=4; SD=p16_sv ;;
  p256) EV=$EV256; B=$O/eval/$V/baselines; NSH=8; SD=p256_sv ;;
  *) echo "SUB must be p16 or p256" >&2; exit 2 ;;
esac
# measured on Fir 2026-11-22 (rcp_halves_mix_pearson_unet, p16): predict shard 19-25 min at 8 CPUs,
# 32.8 GB peak; fit 11.4 min at 4 CPUs; combine of one member 3.1 min, 8.2 GB peak
P256=0; [ "$SUBT" = p256 ] && P256=1
PRED_TIME="${PRED_TIME:-$([ $P256 = 1 ] && echo 04:00:00 || echo 01:00:00)}"; FIT_TIME="${FIT_TIME:-00:45:00}"
COMB_TIME="${COMB_TIME:-$([ $P256 = 1 ] && echo 03:00:00 || echo 00:45:00)}"; COMB_MEM="${COMB_MEM:-16000}"
MET_TIME="${MET_TIME:-$([ "$SUBT" = p256 ] && echo 06:00:00 || echo 03:00:00)}"
for p in "$K/slurm/onion_eval/predict.sh" "$K/slurm/onion_eval/score.sh" "$K/slurm/onion_eval/ensemble.sh" \
         "$K/tools/onion_eval/ensemble_store.py" "$EV" "$EV16" "$EV256" "$B"; do
  [ -e "$p" ] || { echo "missing $p" >&2; exit 2; }
done
EXC="$(sed -n 's/^ *export ONION_EXCLUDE=\([^ ]*\).*/\1/p' "$K/tools/minicandi/configs/base_on_version.sh" | head -n 1)"
[ -n "$EXC" ] || { echo "no ONION_EXCLUDE list in configs/base_on_version.sh" >&2; exit 2; }
COMMON=(--account=def-maxwl --parsable "--exclude=$EXC" --nice=100 --gres=none)
unset ONION_FOREIGN_COVARIATES ONION_FOREIGN_SEED ONION_SCORE_INPUT_DEPTH ONION_KEEP_PRED_STORE
export ONION_STAGE=1 ONION_SCORE_SAME_VIEW=1

store_complete() {
  python3 - "$1" <<'PYEOF'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
try:
    n = int(json.loads((d / "STORE.json").read_text())["n_shards"])
except Exception:
    sys.exit(1)
sys.exit(0 if all((d / f"PREDICTED_{i}of{n}.json").is_file() for i in range(n)) else 1)
PYEOF
}
sub() {   # sub <label> <sbatch args...>: prints the job id (or a placeholder in a dry run)
  local label="$1"; shift
  if [ "$DRY" = 1 ]; then echo "DRY $label: sbatch $*" >&2; echo "$label"; return; fi
  local j; j="$(sbatch "$@")"; j="${j%%;*}"
  [[ "$j" =~ ^[0-9]+$ ]] || { echo "sbatch ($label) gave no job id: $j" >&2; exit 1; }
  echo "$j"
}
log_tsv() {
  [ "$DRY" = 1 ] && return
  mkdir -p "$OUT"
  [ -f "$OUT/SUBMITTED.tsv" ] || printf 'run\ttier\tstep\tjob\twhat\tutc\n' > "$OUT/SUBMITTED.tsv"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$SUBT" "$2" "$3" "$4" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/SUBMITTED.tsv"
}
dep() { [ -z "$1" ] || echo "--dependency=afterok:$1 --kill-on-invalid-dep=yes"; }

for RUN in $RUNS; do
  case "$RUN" in rcp_*) ;; *) echo "refused: run name $RUN does not start with rcp_" >&2; exit 2 ;; esac
  RD="$O/runs_rcp/$RUN"
  RO="$OUT/$RUN"
  need=""
  for v in $VARIANTS; do
    case $v in ens3|ens3_recal) need="$need 0 1 2" ;; seed0_recal) need="$need 0" ;;
      *) echo "unknown variant $v" >&2; exit 2 ;; esac
  done
  SEEDS_NEEDED="$(echo $need | tr ' ' '\n' | sort -u | tr '\n' ' ')"
  for s in $SEEDS_NEEDED; do [ -f "$RD/seed$s/ckpt.pt" ] || { echo "no $RD/seed$s/ckpt.pt" >&2; exit 2; }; done
  [ "$DRY" = 1 ] || mkdir -p "$RO/logs"
  declare -A PIDS=()
  for s in $SEEDS_NEEDED; do
    ST="$PS/$RUN/$SD/seed$s"
    if store_complete "$ST"; then echo "note $RUN seed$s: store $ST complete"; PIDS[$s]=""; continue; fi
    ids=()
    for ((i = 0; i < NSH; i++)); do
      j="$(export ONION_SHARD=$i/$NSH ONION_INFER_BATCH=8 ONION_DEVICE=cpu; sub "predict_${RUN}_s${s}_$i" \
          "${COMMON[@]}" "--job-name=ens_${RUN}_s${s}_pred" "--time=$PRED_TIME" --cpus-per-task=8 \
          --mem=32000M "--output=$RO/logs/predict_${SD}_s${s}_${i}of${NSH}_%j.out" --export=ALL \
          "$K/slurm/onion_eval/predict.sh" "$K" "$RD/seed$s" "$EV" "$ST")" || exit 1
      ids+=("$j"); echo "$j $RUN seed$s predict $i/$NSH -> $ST"; log_tsv "$RUN" predict "$j" "seed$s $i/$NSH $ST"
    done
    PIDS[$s]="$(IFS=:; echo "${ids[*]}")"
  done
  FID=""
  case " $VARIANTS " in *_recal*)
    if [ -f "$RO/recal/factors.json" ]; then echo "note $RUN: factors.json exists"
    else
      FID="$(sub "fit_$RUN" "${COMMON[@]}" "--job-name=ens_${RUN}_fit" "--time=$FIT_TIME" --cpus-per-task=4 \
          --mem=16000M "--output=$RO/logs/fit_%j.out" --export=ALL "$K/slurm/onion_eval/ensemble.sh" "$K" fit \
          --runs "$RD/seed0" "$RD/seed1" "$RD/seed2" --evalset "$EV16" --exclude-evalset "$EV256" \
          --out "$RO/recal")" || exit 1
      echo "$FID $RUN fit -> $RO/recal/factors.json"; log_tsv "$RUN" fit "$FID" "$RO/recal/factors.json"
    fi ;;
  esac
  for v in $VARIANTS; do
    SO="$RO/${v}_$SD"
    CS="$PS/$RUN/${v}_$SD"
    if [ -f "$SO/SCORED" ]; then echo "skip $RUN $v: $SO/SCORED exists"; continue; fi
    case $v in
      ens3) MEM=("$PS/$RUN/$SD/seed0" "$PS/$RUN/$SD/seed1" "$PS/$RUN/$SD/seed2"); FA=(); W="${PIDS[0]}:${PIDS[1]}:${PIDS[2]}" ;;
      ens3_recal) MEM=("$PS/$RUN/$SD/seed0" "$PS/$RUN/$SD/seed1" "$PS/$RUN/$SD/seed2")
                  FA=(--factors "$RO/recal/factors.json" --factor-set ensemble); W="${PIDS[0]}:${PIDS[1]}:${PIDS[2]}:$FID" ;;
      seed0_recal) MEM=("$PS/$RUN/$SD/seed0"); FA=(--factors "$RO/recal/factors.json" --factor-set seed0)
                   W="${PIDS[0]}:$FID" ;;
    esac
    W="$(echo "$W" | tr -s ':' | sed 's/^://; s/:$//')"
    [ "$DRY" = 1 ] || mkdir -p "$SO"
    if [ -f "$CS/COMBINED.json" ]; then CID=""; echo "note $RUN $v: $CS combined"
    else
      CID="$(sub "combine_${RUN}_$v" "${COMMON[@]}" "--job-name=ens_${RUN}_${v}_comb" "--time=$COMB_TIME" \
          --cpus-per-task=4 "--mem=${COMB_MEM}M" "--output=$SO/combine_%j.out" $(dep "$W") --export=ALL \
          "$K/slurm/onion_eval/ensemble.sh" "$K" combine --members "${MEM[@]}" --out "$CS" \
          ${FA[@]+"${FA[@]}"} --record "$SO/ENSEMBLE.json")" || exit 1
      echo "$CID $RUN $v combine -> $CS"; log_tsv "$RUN" combine "$CID" "$v $CS"
    fi
    MID="$(export ONION_SCORE_PHASE=metrics "ONION_PRED_STORE=$CS" "ONION_REFS_FROM=$B"; sub "metrics_${RUN}_$v" \
        "${COMMON[@]}" "--job-name=ens_${RUN}_${v}_met" "--time=$MET_TIME" --cpus-per-task=16 --mem=64000M \
        "--output=$SO/metrics_%j.out" $(dep "$CID") --export=ALL \
        "$K/slurm/onion_eval/score.sh" "$K" "$EV" "$SO" run "$RD/seed0")" || exit 1
    echo "$MID $RUN $v metrics -> $SO/scores.json"; log_tsv "$RUN" metrics "$MID" "$v $SO"
  done
  unset PIDS
done
