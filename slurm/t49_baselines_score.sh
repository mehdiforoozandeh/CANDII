#!/bin/bash
# t49 — score a set of baseline prediction roots. ONE ARRAY TASK PER METHOD, P1 or P2.
#
#   # P1: chr21 out of whatever roots hold it
#   PRED=.../preds SCORES=.../scores CHROMS=chr21 VARPOOL=/scratch/$USER/candi_kit/varpool \
#       sbatch --array=0-4 --time=3:00:00 --mem=24G --cpus-per-task=2 slurm/t49_baselines_score.sh
#   # P2: genome-wide, after the generation array
#   sbatch --array=0-4 --dependency=afterok:<gen_array_jobid> slurm/t49_baselines_score.sh
#
# ONE TASK PER METHOD BECAUSE SCORING IS THE EXPENSIVE HALF, and by a lot. Measured on Fir, chr21,
# 45 declared tracks: generation is ~4.5 minutes for all five methods; `bench.external` is ~2.4
# minutes PER TRACK, so ~110 minutes per method. Scoring five methods inside one job is nine hours
# of serial work for something that is five independent runs. The first shape of this task did
# exactly that and would have hit its walltime.
#
# For P2 the dependency must be `afterok` on the WHOLE generation array: `bench.external` scores a
# track over the CONCATENATION of every chromosome (the top-1 % thresholds of mse1obs/mse1imp are
# taken over all of them at once), so a root missing one chromosome cannot be scored at all. It is
# also why P2 cannot be split by chromosome to make it cheaper: a per-chromosome top-1 % is a
# different metric, not a cheaper estimate of the same one.
#
# P2 COST, MEASURED RATHER THAN GUESSED — and it is the reason `--time` is not defaulted for it.
# chr21 is 1.87 M bins and the whole store is 121 M, so P2 is 65x per track. Measured on chr21:
# ~2.4 min/track for a method carrying the count arm, so P2 is ~2.6 h/track and ~117 h for the 45
# declared tracks. `avg-arcsinh` is pval-only — no NB CRPS and no `nbinom.ppf` over every bin — and
# comes in near 10 h. The 24 h default below is right for chr21 and FAR too short for P2; the count
# arm needs the b5 bin. Submit P2 with an explicit `--time`, e.g.
#
#   sbatch --array=0-4 --time=5-18:00:00 --mem=96G --cpus-per-task=2 slurm/t49_baselines_score.sh
#
# and mind the standing reservations: a 7-day request is refused outright with
# `ReqNodeNotAvail, Reserved for maintenance` whenever a maintenance reservation falls inside the
# window, so size the walltime to land BEFORE the next one rather than asking for the bin maximum.
#
# CRPS_APPROX=100 CRPS_SEED=0 SWITCHES THE COUNT ARM TO t56's FAIR-SAMPLED ESTIMATOR, which the PI
# approved for P2 on 2026-08-26. That is what makes P2 affordable: ~54 h/method at k=100 against
# ~117 h exact, and it stays finite at Poisson floors where the closed form is NaN. It needs a
# checkout carrying the t56 code — `bench.external` rejects the flag otherwise, which is the right
# failure, because a silent fall-back to the closed form would produce a 117 h job labelled as a
# 54 h one. Every score json it writes carries provenance.crps_estimator/crps_k/crps_seed, and a
# count-arm CRPS from such a json is an ESTIMATE and is quoted as one.
#
# The leaderboard is NOT assembled here: it needs every method's score file, so run
# `python -m competitors.baselines.leaderboard --protocol P1|P2 --scores ...` once the array is
# done. That step is json folding and takes seconds.
#
# WHY --gres ON A CPU-ONLY JOB: see slurm/bake.sh.
#SBATCH --account=def-maxwl
#SBATCH --job-name=candi_t49_score
#SBATCH --output=slurm-logs/t49_score_%A_%a.out
#SBATCH --error=slurm-logs/t49_score_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4

set -uo pipefail

KIT="${KIT:-/project/def-maxwl/$USER/CANDII_t49}"
VENV="${VENV:-/project/def-maxwl/$USER/candi_venv}"
REGIME="${REGIME:-configs/regime.eic_val.json}"
PRED="${PRED:-/project/def-maxwl/$USER/t49_baselines/p2/preds}"
SCORES="${SCORES:-/project/def-maxwl/$USER/t49_baselines/p2/scores}"
VARPOOL="${VARPOOL:-}"
# t56's fair-sampled NB-CRPS estimator (PI-approved for P2 at k=100, 2026-08-26). Empty = the
# closed form. Only reachable when the checkout carries the t56 code; on a checkout without it
# `bench.external` rejects the flag, which is the failure we want rather than a silent exact run.
CRPS_APPROX="${CRPS_APPROX:-}"
CRPS_SEED="${CRPS_SEED:-0}"
CHROMS="${CHROMS:-chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr8,chr9,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22,chrX}"

METHOD_LIST=(avg avg-arcsinh knn1 knn5 marginal)
METHOD="${METHOD_LIST[${SLURM_ARRAY_TASK_ID:-0}]}"

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1; unset PYTHONPATH || true
export MPLBACKEND=Agg
module load StdEnv/2023 python/3.10.13 >/dev/null 2>&1
source "$VENV/bin/activate" || { echo "[error] no venv at $VENV" >&2; exit 1; }
cd "$KIT"
source "$KIT/slurm/_kit_pin.sh"
export PYTHONPATH="$KIT/src:$KIT"

case "$REGIME" in *eic_test*) echo "[t49] REFUSING: $REGIME is the B-pair regime (A4)"; exit 2;; esac
echo "[t49-score] host=$(hostname) commit=$(git rev-parse --short HEAD) method=$METHOD"

mkdir -p "$SCORES"
python -m candi.bench.external \
    --store "$REGIME" --pred "$PRED/$METHOD" --out "$SCORES/$METHOD.json" \
    --chroms "$CHROMS" ${VARPOOL:+--varpool "$VARPOOL"} \
    ${CRPS_APPROX:+--crps-approx "$CRPS_APPROX" --crps-seed "$CRPS_SEED"}
rc=$?
echo "[t49-score] method=$METHOD exit=$rc"
exit $rc
