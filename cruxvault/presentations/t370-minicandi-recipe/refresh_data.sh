#!/bin/zsh
# Pull the deck's source tables from Fir into data/ (small files only). Then: python3 build_deck.py
set -e
HERE=${0:A:h}; O=/project/def-maxwl/mforooz/ONION; K=/project/def-maxwl/mforooz/onion_kit/RCP6
ssh -o BatchMode=yes fir "cd $O && source /project/def-maxwl/mforooz/candi_venv/bin/activate 2>/dev/null; \
  for s in p16_sv p256_sv; do \
    python3 $K/tools/onion_eval/select_rule.py arm-summary --runs $O/runs_rcp --prefix rcp_ --sub \$s > runs_rcp/_sel/deck_arms_\$s.tsv 2>/dev/null; \
    python3 $K/tools/onion_eval/select_rule.py arm-summary --runs $O/runs_fir --prefix eic_v1_2_rung --sub \$s > runs_rcp/_sel/deck_base_\$s.tsv 2>/dev/null; \
  done; python3 runs_rcp/_sel/cg.py > /dev/null" >/dev/null 2>&1
rsync -q -e 'ssh -o BatchMode=yes' "fir:$O/runs_rcp/_sel/{deck_arms_p16_sv.tsv,deck_arms_p256_sv.tsv,deck_base_p16_sv.tsv,deck_base_p256_sv.tsv,copy_gain_p16_sv.tsv,ens_p256_sv.tsv}" $HERE/data/
rsync -q -e 'ssh -o BatchMode=yes' "fir:$O/runs_rcp/eic_val/table_unet_vs_average.tsv" $HERE/data/ 2>/dev/null || true
rsync -q -e 'ssh -o BatchMode=yes' "fir:$O/runs_fir/ARM_SUMMARY_p16_sv_2026-11-17.txt" $HERE/data/eic_v1_arm_summary_p16_sv.tsv
cp $HERE/../../INSTRUCTIONS/t370_recipe/evolution/scores_ladder.md $HERE/data/
ls -la $HERE/data
cp $HERE/../../../../onion/cruxvault/results/ladder_tuning/ROUND2_READ.md $HERE/data/ 2>/dev/null || true
ROOT=${HERE:h:h:h}
for s in p16_sv p256_sv; do
  python3 $ROOT/tools/onion_eval/pareto_select.py --summary $HERE/data/deck_base_$s.tsv --summary $HERE/data/deck_arms_$s.tsv --arm-prefix rcp_ --min-seeds 2 --out $HERE/data/pareto_$s.tsv >/dev/null 2>&1 || true
done
echo refreshed
