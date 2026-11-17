#!/bin/bash
# E2 (close-out of peeling the onion): signal p-value and peaks from the predicted NB of the three
# base recipes, 3 seeds each, and their rows on the AlphaGenome 2 (AG2) board. Run on the laptop;
# every job is submitted over ssh. DRY RUN BY DEFAULT: DRY_RUN=0 submits.
#
#   bash slurm/onion_eval/e2_submit.sh fir     # step 1
#   (foreman: Globus Fir $O/e2/tracks -> Nibi $O/e2/tracks, the 9 dump dirs)
#   bash slurm/onion_eval/e2_submit.sh nibi    # step 2
#
# Models (Fir runs_fir): conv autoencoder = rung2_base_heldout, conv + attention =
# rung3_base_heldout, U-Net = rung4_base_heldout; seeds 0, 1, 2.
#
# Step 1, Fir, kit $KIT_ID:
#   9 score jobs: score.py legacy path (one process, model on CPU, 2 threads) on
#     e2/evalset_e2_targets_only.json: evalset_p16.json (the 256 regions of the AG2 comparison) with
#     no correction or replicate pair (sha256 22c300ab...; written once by E2; the pair jobs are not
#     needed here and would double the cost),
#     --lambda-source predicted_control (every ChIP-seq target also gets the model's predicted
#     control: signal_pc.npy, ctl_pred.npy), --fraglen-table e2/eic_v1_fraglen.tsv (sha256
#     4f2efdb6..., the side table of t308), --dump-signal e2/tracks/<run>/seed<k>.
#     -> e2/scores/<run>/seed<k>/{scores.json,SCORED}. A job whose SCORED exists is not submitted.
#   1 job (afterok on the 9): nb_signal_checks.py lambda over the 18 regime dirs -> e2/checks/lambda
#   1 job (afterok on the 9): nb_signal_checks.py table over the 9 scores.json -> e2/checks/table
# Step 2, Nibi, kit $KIT_ID (needs the 9 dumps under Nibi $O/e2/tracks):
#   2 jobs: ag2_score.py score, tables headline and control_free, with the t308 methods and
#     minicandi_conv_ae, minicandi_conv_attn, minicandi_unet (3 seeds each)
#     -> ag2/scores/eic_v1/e2/{headline,control_free}/ag2_board_val.tsv, per_target.tsv, ...
# Never resubmits a job whose output exists or whose job name is queued or running (squeue -u $USER
# -n <name>); writes only under e2/ and ag2/scores/eic_v1/e2/.
set -euo pipefail
WHERE="${1:?usage: e2_submit.sh fir|nibi}"
DRY="${DRY_RUN:-1}"
KIT_ID="${KIT_ID:-COE2b}"
O=/project/def-maxwl/mforooz/ONION
K=/project/def-maxwl/mforooz/onion_kit/$KIT_ID
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

case "$WHERE" in
fir)
  EV=$O/e2/evalset_e2_targets_only.json
  remote "test -d $K && test -f $O/e2/eic_v1_fraglen.tsv && test -f $EV"
  SB="sbatch --parsable --account=def-maxwl --gres=none --exclude=fc10713,fc10715"
  DEPS=()
  LDUMPS=""; TRUNS=""
  for a in "${ARCH[@]}"; do
    name="${a%%:*}"; run="${a#*:}"
    for s in 0 1 2; do
      out=$O/e2/scores/$run/seed$s
      dump=$O/e2/tracks/$run/seed$s
      LDUMPS+=" --dump ${name}_s${s}_heldin=$dump/leave_one_assay_out --dump ${name}_s${s}_hidden=$dump/accessibility_only"
      TRUNS+=" --run ${name}_s${s}=$out/scores.json"
      if done_or_queued "$out/SCORED" "e2_score_${run}_s$s"; then
        echo "skip $run seed$s: $out/SCORED exists or e2_score_${run}_s$s is queued or running"; continue
      fi
      cmd="mkdir -p $out $dump && module load python/3.10.13 >/dev/null 2>&1; $SB --job-name=e2_score_${run}_s$s --time=$SCORE_TIME --cpus-per-task=2 --mem=$SCORE_MEM --output=$O/e2/logs/%x_%j.out $K/slurm/onion_eval/score.sh $K $EV $out run $O/runs_fir/$run/seed$s --dump-signal $dump --fraglen-table $O/e2/eic_v1_fraglen.tsv --lambda-source predicted_control --device cpu"
      if [ "$DRY" = 1 ]; then remote "$cmd"; else
        jid="$("${SSH[@]}" fir "$cmd" | tail -1)"; echo "score $run seed$s: job $jid"; DEPS+=("$jid"); fi
    done
  done
  DEP=""; [ ${#DEPS[@]} -eq 0 ] || DEP="--dependency=afterok:$(IFS=:; echo "${DEPS[*]}")"
  if done_or_queued "$O/e2/checks/lambda/lambda_check.json" e2_lambda; then echo "skip e2_lambda: output exists or job queued"; else
  remote "$SB $DEP --job-name=e2_lambda --time=04:00:00 --cpus-per-task=1 --mem=16000M --output=$O/e2/logs/%x_%j.out $K/slurm/onion_kit/run.sh $K tools/onion_eval/nb_signal_checks.py lambda --evalset $EV --fraglen-table $O/e2/eic_v1_fraglen.tsv --out $O/e2/checks/lambda$LDUMPS"; fi
  if done_or_queued "$O/e2/checks/table/signal_space_macro.tsv" e2_table; then echo "skip e2_table: output exists or job queued"; else
  remote "$SB $DEP --job-name=e2_table --time=00:30:00 --cpus-per-task=1 --mem=4000M --output=$O/e2/logs/%x_%j.out $K/slurm/onion_kit/run.sh $K tools/onion_eval/nb_signal_checks.py table --out $O/e2/checks/table$TRUNS"; fi
  ;;
nibi)
  A=$O/ag2; B=$A/tracks/eic_v1/baselines_fl
  M=""
  for a in "${ARCH[@]}"; do
    name="${a%%:*}"; run="${a#*:}"
    M+=" --method minicandi_$name=$O/e2/tracks/$run/seed0,$O/e2/tracks/$run/seed1,$O/e2/tracks/$run/seed2"
    for s in 0 1 2; do remote "test -f $O/e2/tracks/$run/seed$s/leave_one_assay_out/tracks.json"; done
  done
  for b in position_average marginal_nb training_mean_signal closest5_oracle pseudoreplicate_half; do
    M+=" --method $b=$B/$b"
  done
  for table in headline control_free; do
    if done_or_queued "$A/scores/eic_v1/e2/$table/SCORED" "e2_ag2_$table"; then
      echo "skip e2_ag2_$table: SCORED exists or job queued"; continue
    fi
    remote "module load python/3.10.13 >/dev/null 2>&1; sbatch --parsable --account=def-maxwl --gres=none --job-name=e2_ag2_$table --time=06:00:00 --cpus-per-task=4 --mem=32000M --output=$O/logs/%x_%j.out $K/slurm/onion_kit/run.sh $K tools/onion_eval/ag2_score.py score --evalset $O/eval/eic_v1/evalset.json --audit $A/audit/ag2_eic_audit.tsv --tables $O/tables/eic_v1 --truth-from $B/position_average$M --ag2 $A/pack/eic_v1 --peaks $A/peaks --split val --table $table --allow-unscored --out $A/scores/eic_v1/e2/$table"
  done
  ;;
*) echo "usage: e2_submit.sh fir|nibi" >&2; exit 2 ;;
esac
