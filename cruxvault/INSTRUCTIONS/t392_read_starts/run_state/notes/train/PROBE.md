# Training probe on the read-start store (C12a/C12b)

Job: Fir 5687075, kit RS4 (3704f023), node fc30161 (turin, 192 cores), 12 CPUs, 48000M, `--max-windows 2000`. Run folder: `/project/def-maxwl/mforooz/t392/runs/_probe/seed0`.

| quantity | probe | reference (rcp_halves_mix_avg_pearson_unet, seeds 0 / 1 / 2) |
|---|---|---|
| state | COMPLETED, CAPPED at 2,000 windows | DONE at 80,000 windows |
| parameters | 993,695 (run.json) | 993,615 |
| tables hash | 7a1a0325d00ddff3... (eic_v1_2) | same |
| staged data | 2,329 files, 73.18 GB, 108.4 s | 2,328 files 56.10 GB + average 7.45 GB, 37-46 s |
| wall at 2,000 windows | 1,979 s | 340 / 94 / 758 s |
| wall at 80,000 windows | not run | 20,907 / 3,581 / 23,453 s |
| peak RSS (batch step) | 49,142,600 K | 49,142,844 / 49,143,120 / 49,142,872 K |
| CPUTimeRAW | 30,000 s = 8.3 CPU-h | 163.0 CPU-h for 3 seeds |

All 4 jobs ran on the same node type (turin, 192 cores), so the spread between reference seeds (6.5x) is load from other jobs on a shared node. Peak memory equals the reference, so 48000M stays.

Seconds per window between log rows of the probe: 0.535, 1.874, 1.052, 0.469, 0.939. Over the first 2,000 windows the probe was 2.6x slower than the slowest reference seed (1,979 / 758).

Projection for 3 seeds, by 2 scalings of the reference wall:
- at the reference rate: 163 CPU-h (measured);
- at 2.6x the slowest reference seed: 23,453 s x 2.6 = 61,000 s = 16.9 h per seed; x 12 CPUs = 203 CPU-h per seed; x 3 = 610 CPU-h.
The second figure is above the 240 CPU-h report threshold of C12b and, with the 116.6 CPU-h spent, above the 600 CPU-h stop. One probe cannot separate node load from a real data-path cost (the store is 15% larger, and ChIP has 2 strand channels).
