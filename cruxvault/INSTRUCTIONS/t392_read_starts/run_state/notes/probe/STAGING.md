# t392 probe staging on Nibi (chunk C2)

Date: 2026-11-23. Cluster: Nibi. All cluster files live under `/scratch/mforooz/rs392/probe/`.

## Ground truth checked before submission

- Tar: `/nearline/def-maxwl/mforooz/scratch_archive_2026-11-17/onion-work__000.tar`, `stat -c %s` = 220,679,424,000 bytes.
- `lfs hsm_state` of the tar before the read: `(0x00000009) exists archived, archive_id:1`. A disk copy was present.
- Index: `/project/def-maxwl/mforooz/ONION/nearline_archive_2026-11-17/index/onion-work__000.txt`, 5,809 lines. All 20 members below appear in it.
- Scratch use (`diskusage_report`): 708 GiB of 1,024 GiB; 65K of 1000K files.

## Measurements

The member paths came from `provenance.json::halves.{pr1,pr2}.tagalign` under `/scratch/mforooz/onion/measurements/<id>/`, with the prefix `/scratch/mforooz/` removed.

| measurement | assay | lines_per_fragment | members | bytes (index) |
|---|---|---|---|---|
| ENCSR000AKT.pooled.uniform.base | H3K4me2 (small ChIP-seq) | 1 | 4 | 106,627,943 |
| ENCSR000ENE.pooled.uniform.base | DNase-seq | 1 | 4 | 385,542,185 |
| ENCSR000AMI.pooled.uniform.mapq0 | ChIP-seq control (largest) | 1 | 8 | 2,137,505,813 |
| ENCSR011CSR.pooled.uniform.base | H3K4me1 | 2 | 4 | 712,406,140 |

Total: 20 members, 3,342,082,081 bytes = 3.11 GiB (3,342,082,081 / 1,073,741,824).

The lines_per_fragment 2 measurement was picked by a scan of every measurement directory: 10 measurements had lines_per_fragment 2 and all their tagAligns in `onion-work__000`, and all 10 were ENCSR011CSR layers. I picked `ENCSR011CSR.pooled.uniform.base` because it is the pooled base layer, like the other three probes. The smallest candidate was `ENCSR011CSR.pooled.uniform.depth0.125` (89,050,194 bytes).

## Member list

Also on Nibi at `/scratch/mforooz/rs392/probe/members.txt`.

```
onion/work/ENCSR000AKT/br1.uniform/base/pr1.tagAlign.gz
onion/work/ENCSR000AKT/br2.uniform/base/pr1.tagAlign.gz
onion/work/ENCSR000AKT/br1.uniform/base/pr2.tagAlign.gz
onion/work/ENCSR000AKT/br2.uniform/base/pr2.tagAlign.gz
onion/work/ENCSR000ENE/br1.uniform/base/pr1.tagAlign.gz
onion/work/ENCSR000ENE/br2.uniform/base/pr1.tagAlign.gz
onion/work/ENCSR000ENE/br1.uniform/base/pr2.tagAlign.gz
onion/work/ENCSR000ENE/br2.uniform/base/pr2.tagAlign.gz
onion/work/ENCSR000AMI/br1.uniform/mapq0/pr1.tagAlign.gz
onion/work/ENCSR000AMI/br2.uniform/mapq0/pr1.tagAlign.gz
onion/work/ENCSR000AMI/br3.uniform/mapq0/pr1.tagAlign.gz
onion/work/ENCSR000AMI/br4.uniform/mapq0/pr1.tagAlign.gz
onion/work/ENCSR000AMI/br1.uniform/mapq0/pr2.tagAlign.gz
onion/work/ENCSR000AMI/br2.uniform/mapq0/pr2.tagAlign.gz
onion/work/ENCSR000AMI/br3.uniform/mapq0/pr2.tagAlign.gz
onion/work/ENCSR000AMI/br4.uniform/mapq0/pr2.tagAlign.gz
onion/work/ENCSR011CSR/br1.uniform/base/pr1.tagAlign.gz
onion/work/ENCSR011CSR/br2.uniform/base/pr1.tagAlign.gz
onion/work/ENCSR011CSR/br1.uniform/base/pr2.tagAlign.gz
onion/work/ENCSR011CSR/br2.uniform/base/pr2.tagAlign.gz
```

The extracted files land at `/scratch/mforooz/rs392/probe/work/onion/work/<experiment>/...`. C3 calls this prefix `--work-root /scratch/mforooz/rs392/probe/work/`.

## Job

Script: `/scratch/mforooz/rs392/probe/rs392_stage.sh`. Log: `/scratch/mforooz/rs392/probe/rs392_stage.<jobid>.out`.

```bash
#SBATCH --job-name=rs392_stage --account=def-maxwl --cpus-per-task=1
#SBATCH --mem=4000M --time=03:00:00 --gres=none
TAR=/nearline/def-maxwl/mforooz/scratch_archive_2026-11-17/onion-work__000.tar
OUT=/scratch/mforooz/rs392/probe/work
ls -l "$TAR"; head -c 512 "$TAR" > /dev/null   # prints NEARLINE_UNREADABLE and exits 2 on failure
mapfile -t MEMBERS < /scratch/mforooz/rs392/probe/members.txt
tar -xf "$TAR" -C "$OUT" "${MEMBERS[@]}"
find "$OUT" -name 'pr[12].tagAlign.gz' | wc -l
```

Submitted: `cd /scratch/mforooz/rs392/probe && sbatch rs392_stage.sh` gave job **28710841** (2026-11-23).

## Status

Job 28710841 ran on compute node c667 at 2026-11-23T20:53:39Z and FAILED (exit 2:0, 3 s, CPUTimeRAW 3).
A Nibi compute node cannot read `/nearline`. The log says: `ls: cannot access '/nearline/def-maxwl/mforooz/scratch_archive_2026-11-17/onion-work__000.tar': No such file or directory`, then `NEARLINE_UNREADABLE stat`.
Nothing was extracted; `/scratch/mforooz/rs392/probe/work` holds 0 bytes.

The fallback, a Nibi data-transfer node, was not found. The login node `l1.nibi.sharcnet` reads `/nearline`. The Nibi `ssh_known_hosts` names no DTN; it names only `robot.nibi.sharcnet` and `ic-robot.nibi.sharcnet`, i.e., automation nodes that need a registered CCDB key.
Finding: Nibi compute nodes do not mount `/nearline`. A job that reads a nearline tar must run on the login node, or the tar must first be copied to `/scratch`.

## Login-node run (foreman decision, 2026-11-23)

The foreman approved running the same `tar` command on the login node.
GNU tar 1.35 seeks over unwanted members in a regular-file archive, so the read is the headers plus the 20 members, not the full 220.7 GB.
Script: `/scratch/mforooz/rs392/probe/stage_login.sh` (the job's `tar` command, without the Slurm header). Log: `/scratch/mforooz/rs392/probe/stage_login.log`.

```bash
cd /scratch/mforooz/rs392/probe && nohup nice -n 19 bash stage_login.sh > stage_login.log 2>&1 < /dev/null &
```

- Host `l1.nibi.sharcnet`, start 2026-11-23T20:55:30Z, end 2026-11-23T20:58:53Z, `tar` exit 0.
- Wall time: 1795467533 - 1795467330 = 203 s.
- The seek held. At 122 s, `/proc/<pid>/io` showed rchar 2,257,269,804 bytes, and 2,244,133,756 bytes had landed. At 133 s the file position was 99,384,401,920 bytes into the tar.
- Extracted: 20 files; `du -sb` = 3,342,082,081 bytes. This equals the index sum.

## Test

`hpc run nibi 'find /scratch/mforooz/rs392/probe/work -name "pr[12].tagAlign.gz" | wc -l'` must print 20.
It printed 20 on 2026-11-23, after the login-node run.
