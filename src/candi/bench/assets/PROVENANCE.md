# EIC scoring assets — provenance

Pinned for `t18`. These are the **exact bytes the ENCODE Imputation Challenge scored with**, taken
from the organizers' own repository, not rebuilt from upstream sources. That is the point: a
rebuilt GENCODE v29 bed would be *better* and would make our `mseprom`/`msegene` incomparable to
the published table. `EVAL_PLAN.md` D5 and D16.

**Source:** `https://github.com/ENCODE-DCC/imputation_challenge`, path `annot/hg38/`
**Fetched:** 2026-08-19, via the GitHub git-blob API (the contents API silently returns empty
content for files over 1 MB — `F5.hg38.enhancers.bed.gz` came back as 0 bytes the first time).

| file | bytes | lines | git blob sha1 | sha256 |
|---|---|---|---|---|
| `gencode.v29.genes.gtf.bed.gz` | 786,504 | 58,721 | `3d47da05775effafb26e86b7bdd40251b7c746fd` | `f6294b10c03d9ed45c9ffdc151f4a55cf05a4b0283b55624e87263950b19e560` |
| `F5.hg38.enhancers.bed.gz` | 1,677,678 | 63,285 | `a7b9118acf78e4aa57d909264ac72bca1fb32721` | `85562823f7a394aad2a1ede17dadcca63772b5aac20922dd802850ec3eb606ee` |
| `hg38.blacklist.bed.gz` | 310 | 38 | `57d8ea2d25dac99c68183d745b4156a36c52fe0f` | `7e3af7a6d572c8447ef8c0513830d23974a47a36a29b58646fab455903c6f6d2` |
| `gtf_to_bed.sh` | 137 | — | `b2d1480fac7e5cf26ca4dd084617ceca8c463b8b` | `b59d9682a95743618deac8233f8a24a13de4de8bc20fc21f7ff1813daee4bdae` |

`annotations.py::verify_assets()` re-checks every sha256 at import. The authoritative values live
in that module, not in this table; this table is for a human reading the directory.

## Formats, and why the column counts matter

`score_metrics.py` unpacks these by **positional split with a fixed arity**, so the column count is
load-bearing — a bed with one extra column raises `ValueError` inside the metric, not at load.

**`gencode.v29.genes.gtf.bed.gz` — 6 columns**, `chrom start end gene_id score strand`:

```
chr1	11869	14409	ENSG00000223972.5	.	+
```

`mseprom` and `msegene` both do `chrom_, start, end, _, _, strand = line.split()`.

**`F5.hg38.enhancers.bed.gz` — 12 columns** (bed12), of which only the first three are read:

```
chr10	100006233	100006603	chr10:100006233-100006603	35	.	100006509	100006510	0,0,0	2	218,34	0,336
```

`mseenh` does a 12-way unpack and discards nine of them.

**`hg38.blacklist.bed.gz` — 3 columns**, `chrom start end`.

## `gtf_to_bed.sh` — how the gene bed was made

```bash
zcat -f $GTF | sed 's/[\"\;]//g' | awk 'BEGIN{OFS="\t"} {print $1,$4,$5,$10,$6,$7}' | gzip -nc > $BED
```

Field 10 of a GENCODE GTF attribute string is `gene_id`. Note there is **no `feature == "gene"`
filter** in that pipeline — whatever rows were in the input GTF are what landed in the bed. The
58,721 lines are consistent with a GTF already subset to genes, but the script itself does not
guarantee it, so treat the row count as an observation rather than a definition.

## The blacklist here is NOT the ENCODE Exclusion list

This is the finding that most changes what the E-block must do.

| | regions | chromosomes | bp covered |
|---|---|---|---|
| `annot/hg38/hg38.blacklist.bed.gz` (this file) | **38** | 9 (chr1, 2, 3, 4, 5, 10, 16, 20, 21) | **17,040** |
| `cruxvault/results/t4/hg38-blacklist.v2.bed` (real ENCODE v2) | **636** | 24 | **227,162,400** |

A factor of **13,331** in excluded sequence, and 15 chromosomes with no entry at all. Whatever this
38-region file is, it is not Amemiya et al.'s exclusion list.

### …and on the scoring path actually used, it was never applied at all

`bw_to_npy.py::bw_to_dict` opens with:

```python
if bw_file.lower().endswith(('npy', 'npz')):
    return load_npy(bw_file)
```

**The blacklist branch is below that early return, inside the `bw`/`bigwig` case only.** The
organizers' own README instructs scorers to convert bigwigs to npy first, "for efficiency", and
`score.py` then reads npy. So on the path the challenge actually ran, no blacklist was applied to
the signal — not the real one, and not even this 38-region stub.

A second detail says the same thing from the other side. `blacklist_filter` does not mask bins, it
**deletes** them:

```python
for i, val in enumerate(result_per_chr):
    if i in blacklist[c]:
        continue
    else:
        bfilt_result_per_chr.append(val)
```

That compacts the array and shifts every downstream bin index. Every annotation coordinate in
`mseprom`/`msegene`/`mseenh` is computed as `int(start) // window_size` against the **unshifted**
axis, so a filtered array and the gene bed no longer refer to the same positions. The bigwig branch
is not merely unused — it is inconsistent with the metrics that consume its output.

**What the E-block does:** no blacklist on the signal, matching the npy path and therefore the
published numbers. The file is pinned anyway, because "we checked, and it is a 38-region stub that
the scoring path never reads" is a fact worth being able to re-verify, and because a future reader
who finds `--blacklist-file` in the organizers' CLI will otherwise assume it was used.

`cruxvault/wiki/encode-imputation-challenge.md` already records that the challenge applied the real
Exclusion list to its **peak calls**, not to its signal tracks. This is the mechanical reason that
statement is true. The P-block and B-block use t4's real v2 list where a blacklist is called for.

None of this is a bug to fix. It is a property of the benchmark, and `EVAL_PLAN.md` D16 says we
reproduce the benchmark's properties including the inconvenient ones.
