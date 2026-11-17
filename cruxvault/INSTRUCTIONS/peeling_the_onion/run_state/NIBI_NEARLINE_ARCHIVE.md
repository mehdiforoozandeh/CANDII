---
title: Nibi scratch folders archived to nearline (2026-11-17)
kind: recovery note
parent: "[[peeling the onion]]"
---

# Nibi scratch folders archived to nearline

The PI approved this on 2026-11-17 ("please archive all and note it somewhere so we can later recover them if needed"). Nibi `/scratch` refused every write at 4,324 GiB, over the 1,024 GiB soft quota. Four folders went to `/nearline` (tape storage) as tar files.

## What was archived

| Folder under `/scratch/mforooz/` | Size | Content | Tar name prefix |
|---|---|---|---|
| `t112_cf` | 1,590 GiB | September counterfactual arms: FASTQ arms, FASTQ cache, BAM arms, pseudoreplicates, products | `t112_cf`, `t112_cf-fastqarms`, `t112_cf-fastqarms-harvest` |
| `onion/work` | 1,105 GiB | Pipeline work folders of 457 built rows (tagAligns and intermediates) | `onion-work` |
| `onion/fastq` | 861 GiB | Staged ENCODE FASTQs of built rows | `onion-fastq` |
| `EIC_REPRO` | 242 GiB | EIC reproduction of August | `EIC_REPRO` |

## Where it is (Nibi)

- Tar files: `/nearline/def-maxwl/mforooz/scratch_archive_2026-11-17/<name>.tar` (23 files, about 200 GiB each).
- State and index: `/project/def-maxwl/mforooz/ONION/nearline_archive_2026-11-17/`
  - `plan.tsv`: one line per tar, with its planned size in MiB.
  - `lists/<name>.list`: the members of each tar, as paths relative to `/scratch/mforooz`.
  - `index/<name>.txt`: the full file listing of each tar (`tar -tvf`).
  - `status.tsv`: `OK` means that the tar was compared byte for byte with scratch (`tar -d`).
  - `archive.sh`: the script. A copy is beside this note (`nibi_nearline_archive.sh`).

## How to recover

1. Find the tar that holds a path: `grep -l '<path fragment>' /project/def-maxwl/mforooz/ONION/nearline_archive_2026-11-17/index/*.txt`.
2. Restore the whole tar: `tar -xf /nearline/def-maxwl/mforooz/scratch_archive_2026-11-17/<name>.tar -C /scratch/mforooz`.
3. Restore one folder only: add the path as given in the index, e.g. `tar -xf <tar> -C /scratch/mforooz onion/work/ENCSR000AKP`.

Run these on a Nibi login node; compute nodes do not mount `/nearline`.

## Limits

- A read from tape takes hours, up to about one day. `lfs hsm_state <file>` shows if a tar is on tape; `lfs hsm_restore <file>` asks for it early.
- A restore needs free space on `/scratch`.
- The staged FASTQs can also be downloaded again from ENCODE, which can be faster than a recall.

## State

See the Log in `plan.md` for the run state and for the date when the scratch copies were removed.
