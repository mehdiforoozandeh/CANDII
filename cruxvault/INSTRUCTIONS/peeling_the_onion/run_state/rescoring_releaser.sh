#!/bin/bash
# release held rescoring run-seeds on Nibi in priority order, at most CAP active at one time
CAP=${CAP:-6}
ORDER=""
for a in abl_scrambled_covariates_scrambled_covariates_convae abl_scrambled_covariates_scrambled_covariates_convattn abl_scrambled_covariates_scrambled_covariates_unet \
  abl_views_base_only_convae abl_views_base_only_convattn rung4_views_base_only \
  abl_depth_input_input_depth_fixed_convae abl_depth_input_input_depth_fixed_convattn abl_depth_input_input_depth_fixed_unet \
  abl_average_input_average_input_convae abl_average_input_average_input_convattn abl_average_input_average_input_unet \
  abl_losses_loss_impute_x4_convae abl_losses_loss_impute_x4_convattn abl_losses_loss_impute_x4_unet \
  abl_two_step_two_step_native_convae abl_two_step_two_step_native_convattn abl_two_step_two_step_native_unet \
  abl_two_step_two_step_fixed_convae abl_two_step_two_step_fixed_convattn abl_two_step_two_step_fixed_unet; do
  for s in 0 1 2; do ORDER="$ORDER mc_${a}_s$s"; done; done
while :; do
  out=$(ssh -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 nibi "CAP=$CAP ORDER='$ORDER' bash -s" <<'R' 2>/dev/null | grep -v -i 'multifactor\|authentification\|^=$\|^ *$'
q=$(squeue -u $USER -h -o "%j %r" | grep "^mc_")
active=$(echo "$q" | awk '$2!="JobHeldUser"{print $1}' | sort -u | grep -c .)
held=$(echo "$q" | awk '$2=="JobHeldUser"{print $1}' | sort -u)
rel=""
for n in $ORDER; do
  [ "$active" -ge "$CAP" ] && break
  echo "$held" | grep -qx "$n" || continue
  ids=$(squeue -u $USER -h -n "$n" -o %i | paste -sd, -)
  [ -n "$ids" ] && scontrol release $ids && { active=$((active+1)); rel="$rel $n"; }
done
scored=$(ls -d /project/def-maxwl/mforooz/ONION/runs_fir/abl_*/seed*/p256_sv/SCORED /project/def-maxwl/mforooz/ONION/runs_fir/rung4_views_base_only/seed*/p256_sv/SCORED 2>/dev/null | wc -l)
nheld=$(squeue -u $USER -h -o "%j %r" | grep "^mc_" | awk '$2=="JobHeldUser"{print $1}' | sort -u | grep -c .)
use=$(diskusage_report 2>/dev/null | grep scratch | grep -o "[0-9]*GiB/" | head -1)
echo "active=$active held=$nheld scored=$scored scratch=$use released:$rel"
R
)
  echo "$(date -u +%H:%M) $out"
  case "$out" in *"active=0 held=0 "*) break;; esac
  /bin/sleep 600
done
