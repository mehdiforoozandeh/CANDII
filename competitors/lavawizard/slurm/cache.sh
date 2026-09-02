#!/bin/bash
# CANDI_STORE -> one cache directory per stem. CPU only; the cost is store reads.
#
#   sbatch competitors/lavawizard/slurm/cache.sh                       # the three eval chromosomes
#   STAGE=shared sbatch --array=0 competitors/lavawizard/slurm/cache.sh   # the transferable scope
#
# A THIRD CALLER, 2026-09-01: `sigma.sh` caches its own training chromosomes inline rather than
# through this array — it wants one stem and is already holding the workspace. It passes the SOURCE
# regime here for the same reason this script does, because `train_columns` reads the training
# split and a self-pair regime would trip its §6.2 guard.
#
# TWO KINDS OF CACHE, because there are two stages (see _env.sh):
#
#   * one per EVAL chromosome, whole, with no training restriction. The genome stage fits position
#     tables there and §2 Rule 2 counts that as inference; prediction needs every bin regardless.
#   * one `shared` stem, the transferable stage's scope. Under eic.19 that is chr19 whole
#     (2,344,704 bins, 2.3 GiB of float32 over 267 tracks); under eic.pilot it is the contained
#     bins of eighteen chromosomes packed onto one 1,031,264-slot axis by `shared_layout` —
#     1,023,489 real, 0.75 % alignment slots that carry no data and are never trained on.
#
# 267 training tracks x n_bins float32, plus an int8 tercile row each: 2.0 GiB on chr21 and
# 10.6 GiB on chr1. The builder holds one biosample's block at a time, so the ask is set by the
# npy memmaps the OS writes back, not by the whole matrix. The shared stem under eic.pilot reads 40
# spans across 18 chromosomes per biosample, which is more store seeks than a whole chromosome and
# the reason its walltime is not shorter in proportion to its size.
#SBATCH --account=def-maxwl
#SBATCH --job-name=t53_cache
#SBATCH --output=slurm-logs/%x_%A_%a.out
#SBATCH --error=slurm-logs/%x_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
# fc30560 gave repeated Lustre OSError 108 on files its neighbours read fine (anchor run).
#SBATCH --exclude=fc30560
# THREE tasks, not 23: the list is the regime's eval_chroms (see _env.sh). STAGE=shared is ONE.
#SBATCH --array=0-2%3
# `$0` is sbatch's spool copy, not this file, so the path comes from the submit directory.
ENV="${SLURM_SUBMIT_DIR:-$PWD}/competitors/lavawizard/slurm/_env.sh"
[ -f "$ENV" ] || { echo "[error] no $ENV -- submit from the repo root" >&2; exit 2; }
source "$ENV"

if [ "${STAGE:-genome}" = "shared" ]; then
  echo "[cache] $SHARED_STEM (the transferable scope of $REGIME_NAME)  host=$(hostname)"
  srun python -u -m lavawizard.store_eic cache-shared --regime "$REGIME" --cache "$CACHE"
  exit $?
fi

C=${CHROMS[$SLURM_ARRAY_TASK_ID]:-}
[ -n "$C" ] || { echo "[error] array index $SLURM_ARRAY_TASK_ID names no chromosome; this regime has $NCHROM (${CHROMS[*]})" >&2; exit 2; }
echo "[cache] $C  host=$(hostname)"
srun python -u -m lavawizard.store_eic cache --regime "$REGIME" --chrom "$C" --cache "$CACHE"
