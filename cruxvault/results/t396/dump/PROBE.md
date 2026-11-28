# t396 dump probe (C7a)

Kit: `/project/def-maxwl/mforooz/onion_kit/SC7a` (git 2a91df30, dirty=0). Outputs of the probe: `/scratch/mforooz/sc396/probe/{model,truth}/ENCSR234YIU.pooled.uniform.base.npz`. Logs: `/scratch/mforooz/sc396/logs/`.

## Probe jobs

The model probe ran `dump_native.py model` on seed 0, `--regions all` (256 regions), `--shard 0/43` (one target). The truth probe ran `dump_native.py truth --shard 0/43`. Both ran on 8 CPUs and 32000M.

| job | mode | Elapsed (s) | TotalCPU (s) | MaxRSS | dump phase (s) | npz size |
|---|---|---|---|---|---|---|
| 6727795 | model, seed 0 | 1110 | 435 | 11.2 GiB (11782932K) | 1057 | 87.0 MB |
| 6727796 | truth | 113 | 93 | 14.4 GiB (15088740K) | 82 | 33.0 MB |

Source: `sacct -n -j <id> -o JobID,State,Elapsed,TotalCPU,MaxRSS` and the "timing by phase" line of each log.

## Gate (a)

The model probe wrote `max_abs_rel_diff 0`, i.e. the fresh `convert_nb` equals the stored `mu`. The gate (<= 1e-4) passed for this target. The per-region diagnostic `max_abs_rel_diff_region` was 0.747; it is not the gate.

## Projection

CPU time per target-region (model): 435 / 256 = 1.70 CPU-s. The t392 sanity job gave 2124 / (63 x 16) = 2.11 CPU-s.

- Model, 3 seeds: 3 x 43 x 435 s = 56,115 CPU-s = 15.6 CPU-h.
- Truth: 43 x 93 s = 3,999 CPU-s = 1.1 CPU-h.
- Total CPU time: 16.7 CPU-h. This is below the 80 CPU-h bar.

The jobs use about 5% of their allocated cores (435 CPU-s over 1110 s x 8 cores). The allocated core time is therefore larger: model 3 x 43 x 1057 s x 5 cores = 189 core-h; truth about 4 x (31 + 11 x 82) s x 8 cores = 8.3 core-h.

## Submitted

| job | what | array | resources |
|---|---|---|---|
| 6731315 | truth, 43 targets | 0-3%3 | 8 cpu, 32000M, 01:00:00 |
| 6731316 | model seed 0 | 0-3%3 | 5 cpu, 20000M, 06:00:00 |
| 6731317 | model seed 1 | 0-3%3 | 5 cpu, 20000M, 06:00:00 |
| 6731318 | model seed 2 | 0-3%3 | 5 cpu, 20000M, 06:00:00 |

The four arrays each cap at 3 live tasks, so at most 12 tasks import torch at once. A model shard holds at most 11 targets: 11 x 1057 s = 3.2 h, so the limit is 06:00:00. The model arrays use 5 CPUs and 20000M because the probe peaked at 11.2 GiB and used under 1 core on average. Progress check: each target prints a `model: wrote` line to `/scratch/mforooz/sc396/logs/sc_dump_model_s<k>_<job>_<task>.out`.
