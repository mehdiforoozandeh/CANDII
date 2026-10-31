# GPU job sizing and cluster choice

Rule (from the Fir GPU queue diagnosis of 2026-10-29, applied by the PI the same day):

1. Request what the job measurably uses. Read the wall time and MaxRSS of a short smoke run or of earlier runs of the same recipe (`sacct`, `seff`), then set the wall time from that measurement with a margin, not from the band maximum.
2. Request only the CPUs the job uses. Fir bills a GPU job by its largest resource (MAX_TRES billing), so 4 CPUs on one MIG slice are billed as 4 CPU-equivalents and drain fair-share faster than the GPU alone.
3. Before a large array, compare fair-share and queue length on Fir and Nibi (`sshare`, `squeue`), and submit to the cluster with headroom.

Diagnosis (2026-10-29): the def-maxwl_gpu fair-share on Fir was exhausted; the jobs requested 4 CPUs and were billed by them; and 12 h was requested against a median run near 86 min. The Nibi queue was about 10x shorter.

Applied:
- The arm-to-arm grid (h28, task t136) moved to Nibi at 2 CPUs, `--mem=10000M`, `--time=7:00:00` (PI 2026-10-29). Its smoke (300 steps) projected Darm1 < 1 h and Darm2 < 1.5 h at 12 000 steps.
- The pivot-study seed arrays on Fir (task t135) were cut from 12 h to 7 h (train) and 6 h (law) by the PI with `scontrol update`.
- Fir GPU nodes fc10713 and fc10715 stay excluded (seen 2026-10-25).
