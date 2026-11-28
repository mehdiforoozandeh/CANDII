# t396 dump: collect and gate (a) (C7b)

## Arrays on Fir

| set | path | files | bytes |
|---|---|---|---|
| truth | `/scratch/mforooz/sc396/truth/<target>.npz` | 43 | 1,086,825,562 |
| read-start model, seed 0 | `/scratch/mforooz/sc396/arrays/rs_halves_mix_avg_pearson_unet/seed0/<target>.npz` | 43 | 3,783,601,738 |
| read-start model, seed 1 | `.../seed1/` | 43 | 3,800,646,328 |
| read-start model, seed 2 | `.../seed2/` | 43 | 3,780,369,901 |

Source: `ls | wc -l` and `du -sb` on each folder (2026-11-28). The three seed folders hold the same 43 target names as the truth folder. The model files cover the read-start model only, so the model count is 3 x 43 = 129. The read-count reference needs no dump: the scoring reads its prediction store in place.

Each model file holds `mu_sub`, `n_sub` float32 [256, 16384, 2], `mu`, `n` float32 [256, 16384], `max_abs_rel_diff`, `max_abs_rel_diff_region` and `region_indices` (all 256 regions).

## Jobs

| job | what | tasks | state | TotalCPU (s) | allocated core-s | peak MaxRSS |
|---|---|---|---|---|---|---|
| 6731315 | truth | 4 | 4 COMPLETED | 1,371 | 12,512 | 22.8 GiB |
| 6731316 | model seed 0 | 4 | 4 COMPLETED | 13,145 | 90,520 | 19.5 GiB |
| 6731317 | model seed 1 | 4 | 4 COMPLETED | 18,553 | 109,140 | 19.5 GiB |
| 6731318 | model seed 2 | 4 | 4 COMPLETED | 15,707 | 105,445 | 19.5 GiB |

Source: `sacct -n -P -j <ids> -o JobID,TotalCPU,Elapsed,AllocCPUS,MaxRSS`. TotalCPU is summed over the batch steps; allocated core-s is Elapsed x AllocCPUS summed over the tasks.
Total CPU time: 1,371 + 13,145 + 18,553 + 15,707 = 48,776 s = 13.5 CPU-h. Allocated core time: 12,512 + 90,520 + 109,140 + 105,445 = 317,617 s = 88.2 core-h.
The model tasks peaked at 19.5 GiB against a 20000M (19.53 GiB) limit. A rerun of the model dump needs more memory.

## Gate (a)

`run.sh` reads `max_abs_rel_diff` of each model file and writes `gate.tsv` (target, seed, value, verdict at 1e-4). Result: 129 of 129 pass. The largest value is 2.29e-07 (ENCSR521AIV.br1.uniform.base, seed 2); 3 files have 0. `dump_native.py` writes no file for a target above the gate, so a missing file is a fail in `gate.tsv`; none is missing.
