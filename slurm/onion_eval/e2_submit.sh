#!/bin/bash
# E2 (close-out of peeling the onion): signal p-value and peaks from the predicted NB of the three
# base recipes, 3 seeds each, and their rows on the AlphaGenome 2 (AG2) board, with the box_d signal
# pileup (PI ruling 2026-11-17, "Adopt the box spread"). Run on the laptop; every job is submitted
# over ssh. DRY RUN BY DEFAULT: DRY_RUN=0 submits.
#
#   bash slurm/onion_eval/e2_submit.sh fir             # step 1 (Fir: the 9 model dumps)
#   bash slurm/onion_eval/e2_submit.sh nibi_baselines  # step 1b (Nibi, independent of step 1)
#   (foreman: Globus Fir $O/e2/tracks_box -> Nibi $O/e2/tracks_box, the 9 dump dirs)
#   bash slurm/onion_eval/e2_submit.sh nibi_board      # step 2 (Nibi, after 1, 1b and Globus)
#
# Every signal of the board must carry one pileup form (ag2_score.py refuses a mix), so the
# count-derived reference methods are rebuilt with box_d too (step 1b); the t308 dumps (inbin) are
# not used. The inbin scores of conv autoencoder seed 0 (e2/scores/rung2_base_heldout/seed0, job
# 4448330) stay as they are; the box_d run goes to new directories (*_box).
#
# Models (Fir runs_fir): conv autoencoder = rung2_base_heldout, conv + attention =
# rung3_base_heldout, U-Net = rung4_base_heldout; seeds 0, 1, 2.
#
# Step 1, Fir, kit $KIT_ID:
#   9 score jobs: score.py legacy path (one process, model on CPU, 2 threads) on
#     e2/evalset_e2_targets_only.json (evalset_p16.json without correction or replicate pairs;
#     sha256 22c300ab...), --signal-pileup box_d, --lambda-source predicted_control (signal_pc.npy,
#     ctl_pred.npy), --fraglen-table e2/eic_v1_fraglen.tsv (sha256 4f2efdb6...),
#     --dump-signal e2/tracks_$SFX/<run>/seed<k> -> e2/scores_$SFX/<run>/seed<k>/{scores.json,SCORED}.
#     Cost: 9 x 2 CPUs x about 9.5 h (timing job 4448330: targets 32,517 s) = about 171 CPU-h.
#   1 job (afterok on the 9): nb_signal_checks.py lambda over the 18 regime dirs -> e2/checks_$SFX/lambda
#   1 job (afterok on the 9): nb_signal_checks.py table over the 9 scores.json -> e2/checks_$SFX/table
# Step 1b, Nibi, kit $KIT_ID:
#   2 baseline dumps (score.sh baseline position_average | marginal_nb, --signal-pileup box_d,
#     --fraglen-table, targets-only eval set e2/evalset_e2_targets_only.json, sha256 89aef389...)
#     -> ag2/tracks/eic_v1/baselines_box/<b>, scores e2/baselines_$SFX/<b>/seed0. 16 CPUs x up to 4 h
#     each (t308 with all pairs: 2 h 43 min and 1 h 51 min) = at most 128 CPU-h, likely about 40.
#   1 job (afterok position_average): ag2_baselines.py run --signal-pileup box_d -> the
#     training_mean_signal, closest5_oracle and pseudoreplicate_half dirs under baselines_box
#     (4 CPUs, t308: 21 min).
# Step 2, Nibi, kit $KIT_ID (needs the 9 dumps under Nibi $O/e2/tracks_box and step 1b):
#   2 jobs: ag2_score.py score, tables headline and control_free -> ag2/scores/eic_v1/e2_box/<table>
# Never resubmits a job whose output exists or whose job name is queued or running (squeue -u $USER
# -n <name>); writes only under e2/, ag2/tracks/eic_v1/baselines_box/ and ag2/scores/eic_v1/e2_box/.
set -euo pipefail
WHERE_ARG="${1:?usage: e2_submit.sh fir|nibi_baselines|nibi_board}"
case "$WHERE_ARG" in fir) WHERE=fir ;; nibi_baselines|nibi_board) WHERE=nibi ;;
  *) echo "usage: e2_submit.sh fir|nibi_baselines|nibi_board" >&2; exit 2 ;; esac
DRY="${DRY_RUN:-1}"
#: E2 review: the bin dependence of the box_d expectation, one value per submission
#: (comonotone = default, the upper variance bound; independent = the lower bound). Run the
#: script once per value; the outputs go to *_box (comonotone) or *_box_ind (independent).
DEP_FORM="${SIGNAL_BOX_DEPENDENCE:-comonotone}"
case "$DEP_FORM" in comonotone) SFX=box ;; independent) SFX=box_ind ;;
  *) echo "SIGNAL_BOX_DEPENDENCE must be comonotone or independent" >&2; exit 2 ;; esac
KIT_ID="${KIT_ID:-COE2f}"
O=/project/def-maxwl/mforooz/ONION
K=/project/def-maxwl/mforooz/onion_kit/$KIT_ID
EV=$O/e2/evalset_e2_targets_only.json      # one file per cluster, same 256 regions
FL_FIR=$O/e2/eic_v1_fraglen.tsv
FL_NIBI=$O/tables/eic_v1_fraglen.tsv
A=$O/ag2; BB=$A/tracks/eic_v1/baselines_$SFX
ARCH=(conv_ae:rung2_base_heldout conv_attn:rung3_base_heldout unet:rung4_base_heldout)
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15)
# score job shape: 2 CPUs, 20 GB, 14 h. Timing job 4448330 (rung2 seed0, with pairs): 11 h 44 min,
# targets 32,517 s (9.0 h), pairs 9,294 s, RSS 12.3 GB under a 12 GB limit
SCORE_TIME="${SCORE_TIME:-14:00:00}"
SCORE_MEM="${SCORE_MEM:-20000M}"

remote() {   # run one command on the cluster, or print it
  if [ "$DRY" = 1 ]; then echo "[dry-run $WHERE] $1"; else "${SSH[@]}" "$WHERE" "$1"; fi
}
# guard: 0 (skip) when <file> exists or a job named <name> is queued or running on the cluster.
# A dry run checks too (read-only ssh), so its printout shows what a live run would submit.
done_or_queued() {   # done_or_queued <file> <job name>
  "${SSH[@]}" "$WHERE" "test -f $1 || [ -n \"\$(squeue -h -u \$USER -n $2)\" ]" 2>/dev/null
}
submit() {   # submit <sbatch command> -> prints the job id (or the dry-run line)
  if [ "$DRY" = 1 ]; then remote "$1"; else "${SSH[@]}" "$WHERE" "$1" | tail -1; fi
}

case "$WHERE_ARG" in
fir)
  remote "test -d $K && test -f $FL_FIR && test -f $EV"
  SB="module load python/3.10.13 >/dev/null 2>&1; sbatch --parsable --account=def-maxwl --gres=none --exclude=fc10713,fc10715"
  DEPS=()
  LDUMPS=""; TRUNS=""
  for a in "${ARCH[@]}"; do
    name="${a%%:*}"; run="${a#*:}"
    for s in 0 1 2; do
      out=$O/e2/scores_$SFX/$run/seed$s
      dump=$O/e2/tracks_$SFX/$run/seed$s
      LDUMPS+=" --dump ${name}_s${s}_heldin=$dump/leave_one_assay_out --dump ${name}_s${s}_hidden=$dump/accessibility_only"
      TRUNS+=" --run ${name}_s${s}=$out/scores.json"
      if done_or_queued "$out/SCORED" "e2${SFX}_score_${run}_s$s"; then
        echo "skip $run seed$s: $out/SCORED exists or e2${SFX}_score_${run}_s$s is queued or running"; continue
      fi
      jid="$(submit "mkdir -p $out $dump && $SB --job-name=e2${SFX}_score_${run}_s$s --time=$SCORE_TIME --cpus-per-task=2 --mem=$SCORE_MEM --output=$O/e2/logs/%x_%j.out $K/slurm/onion_eval/score.sh $K $EV $out run $O/runs_fir/$run/seed$s --dump-signal $dump --fraglen-table $FL_FIR --signal-pileup box_d --signal-box-dependence $DEP_FORM --lambda-source predicted_control --device cpu")"
      if [ "$DRY" = 1 ]; then echo "$jid"; else echo "score $run seed$s: job $jid"; DEPS+=("$jid"); fi
    done
  done
  DEP=""; [ ${#DEPS[@]} -eq 0 ] || DEP="--dependency=afterok:$(IFS=:; echo "${DEPS[*]}")"
  if done_or_queued "$O/e2/checks_$SFX/lambda/lambda_check.json" e2${SFX}_lambda; then echo "skip e2${SFX}_lambda"; else
    submit "$SB $DEP --job-name=e2${SFX}_lambda --time=04:00:00 --cpus-per-task=1 --mem=16000M --output=$O/e2/logs/%x_%j.out $K/slurm/onion_kit/run.sh $K tools/onion_eval/nb_signal_checks.py lambda --evalset $EV --fraglen-table $FL_FIR --out $O/e2/checks_$SFX/lambda$LDUMPS"; fi
  if done_or_queued "$O/e2/checks_$SFX/table/signal_space_macro.tsv" e2${SFX}_table; then echo "skip e2${SFX}_table"; else
    submit "$SB $DEP --job-name=e2${SFX}_table --time=00:30:00 --cpus-per-task=1 --mem=4000M --output=$O/e2/logs/%x_%j.out $K/slurm/onion_kit/run.sh $K tools/onion_eval/nb_signal_checks.py table --out $O/e2/checks_$SFX/table$TRUNS"; fi
  ;;
nibi_baselines)
  remote "test -d $K && test -f $FL_NIBI && test -f $EV"
  SB="module load python/3.10.13 >/dev/null 2>&1; sbatch --parsable --account=def-maxwl --gres=none"
  PA=""
  for b in position_average marginal_nb; do
    out=$O/e2/baselines_$SFX/$b/seed0
    if done_or_queued "$out/SCORED" "e2${SFX}_dump_$b"; then echo "skip e2${SFX}_dump_$b"; continue; fi
    jid="$(submit "mkdir -p $out && $SB --job-name=e2${SFX}_dump_$b --time=04:00:00 --cpus-per-task=16 --mem=64000M --output=$O/e2/logs/%x_%j.out $K/slurm/onion_eval/score.sh $K $EV $out baseline $b --dump-signal $BB/$b --fraglen-table $FL_NIBI --signal-pileup box_d --signal-box-dependence $DEP_FORM")"
    if [ "$DRY" = 1 ]; then echo "$jid"; else echo "dump $b: job $jid"; fi
    if [ "$b" = position_average ]; then PA="$jid"; fi
  done
  DEP=""; [ "$DRY" = 1 ] || [ -z "$PA" ] || DEP="--dependency=afterok:$PA"
  if done_or_queued "$BB/training_mean_signal/leave_one_assay_out/tracks.json" e2${SFX}_ag2_baselines; then echo "skip e2${SFX}_ag2_baselines"; else
    submit "$SB $DEP --job-name=e2${SFX}_ag2_baselines --time=12:00:00 --cpus-per-task=4 --mem=32000M --output=$O/e2/logs/%x_%j.out $K/slurm/onion_kit/run.sh $K tools/onion_eval/ag2_baselines.py run --tables $O/tables/eic_v1 --evalset $EV --truth-from $BB/position_average --split val --out $BB --fraglen-table $FL_NIBI --signal-pileup box_d"; fi
  ;;
nibi_board)
  M=""
  for a in "${ARCH[@]}"; do
    name="${a%%:*}"; run="${a#*:}"
    M+=" --method minicandi_$name=$O/e2/tracks_$SFX/$run/seed0,$O/e2/tracks_$SFX/$run/seed1,$O/e2/tracks_$SFX/$run/seed2"
    for s in 0 1 2; do remote "test -f $O/e2/tracks_$SFX/$run/seed$s/leave_one_assay_out/tracks.json"; done
  done
  for b in position_average marginal_nb training_mean_signal closest5_oracle pseudoreplicate_half; do
    M+=" --method $b=$BB/$b"
  done
  for table in headline control_free; do
    if done_or_queued "$A/scores/eic_v1/e2_$SFX/$table/SCORED" "e2${SFX}_ag2_$table"; then
      echo "skip e2${SFX}_ag2_$table: SCORED exists or job queued"; continue
    fi
    submit "module load python/3.10.13 >/dev/null 2>&1; sbatch --parsable --account=def-maxwl --gres=none --job-name=e2${SFX}_ag2_$table --time=06:00:00 --cpus-per-task=4 --mem=32000M --output=$O/e2/logs/%x_%j.out $K/slurm/onion_kit/run.sh $K tools/onion_eval/ag2_score.py score --evalset $O/eval/eic_v1/evalset.json --audit $A/audit/ag2_eic_audit.tsv --tables $O/tables/eic_v1 --truth-from $BB/position_average$M --ag2 $A/pack/eic_v1 --peaks $A/peaks --split val --table $table --allow-unscored --out $A/scores/eic_v1/e2_$SFX/$table"
  done
  ;;
esac
