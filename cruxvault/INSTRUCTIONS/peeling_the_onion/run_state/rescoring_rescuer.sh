#!/bin/bash
# every 15 min: a run-seed whose predict shard timed out, that is not SCORED and has no live job, is resubmitted with a 12 h limit
W="/Users/mforooz/Desktop/research/libbrechteam@sfu/CANDII/.claude/worktrees/onion"
S="ssh -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 nibi"
while :; do
  list=$($S 'O=/project/def-maxwl/mforooz/ONION/runs_fir; live=$(squeue -u $USER -h -o %j | sort -u); sacct -u $USER -S now-14hours -X -n -o jobname%80,state | awk "\$2==\"TIMEOUT\" && \$1~/^mc_/{print \$1}" | sort -u | while read n; do a=${n#mc_}; s=${a##*_s}; a=${a%_s*}; [ -f $O/$a/seed$s/p256_sv/SCORED ] && continue; echo "$live" | grep -qx "$n" && continue; echo "$a $s"; done' 2>/dev/null | grep -v -i 'multifactor\|authentification\|^=$\|^ *$')
  n=0
  while read -r a s; do
    [ -n "$a" ] || continue
    case "$a" in abl_scrambled*) g=scrambled;; *views_base_only*) g=views_base;; *two_step_native*) g=two_step_native;; *two_step_fixed*) g=two_step_fixed;; *input_depth_fixed*) g=fixed_depth;; *average_input*) g=average_input;; *impute_x4*) g=impute_x4;; *) continue;; esac
    $S "KIT_ID=CO9 ARMS=$g ONLY='$a' SEEDS='$s' SCORE_PRED_TIME=12:00:00 HOLD=0 DRY_RUN=0 bash -s" < "$W/slurm/onion_eval/rescoring_p256_nibi.sh" 2>/dev/null | grep "^submitted" | sed "s/^/rescued $a seed $s: /"
    n=$((n+1))
  done <<< "$list"
  sc=$($S 'ls -d /project/def-maxwl/mforooz/ONION/runs_fir/abl_*/seed*/p256_sv/SCORED /project/def-maxwl/mforooz/ONION/runs_fir/rung4_views_base_only/seed*/p256_sv/SCORED 2>/dev/null | wc -l' 2>/dev/null | tail -1)
  echo "$(date -u +%H:%M) rescued=$n scored=$sc"
  [ "$sc" = 62 ] && break
  /bin/sleep 900
done
