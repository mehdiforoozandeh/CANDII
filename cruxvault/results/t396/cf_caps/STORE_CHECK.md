# Store check of the 21 testbed rows (t396 iteration 2, chunk B3)

Checked 2026-11-28 on Fir, read only. For each of the 7 counterfactual-arm tracks, the rows `.base`, `.nodedup` and `.pe` were checked in both stores.

- Count store `/scratch/mforooz/onion/measurements/<id>/`: `counts.h5`, `pr1.h5`, `pr2.h5` present and non-empty.
- Read-start store `/scratch/mforooz/rs392/store/<id>/`: `starts.h5`, `starts_pr1.h5`, `starts_pr2.h5` present and non-empty.
- `gate_passed`: the top-level key of the read-start `provenance.json`. `gates` lists `outputs.<file>.gate.result` of the 3 start files (the start file rebuilds the count file bit for bit, `n_differing` 0).
- `n_lines`: from the read-start `provenance.json` (`outputs.<file>.n_lines`). The count-store `provenance.json` (`halves.pr1/pr2.n_lines`) gave the same half counts for all 21 rows.
- `lines_per_fragment`: from the read-start `provenance.json`.

Result: 21 of 21 rows have all 6 files and `gate_passed` true. No file is missing and no gate is false, so nothing was rebuilt.

| measurement_id | testbed pid | count files | start files | gate_passed | gates | n_lines | n_lines pr1 | n_lines pr2 | lines_per_fragment |
|---|---|---|---|---|---|---|---|---|---|
| ENCSR061UOM.pooled.testbed.base | C07M20__base__base | 3/3 | 3/3 | true | pass,pass,pass | 30,000,000 | 15,000,000 | 15,000,000 | 1 |
| ENCSR061UOM.pooled.testbed.nodedup | C07M20__dedup__off | 3/3 | 3/3 | true | pass,pass,pass | 30,000,000 | 15,000,000 | 15,000,000 | 1 |
| ENCSR061UOM.pooled.testbed.pe | C07M20__pe__pe | 3/3 | 3/3 | true | pass,pass,pass | 30,000,000 | 15,000,000 | 15,000,000 | 2 |
| ENCSR131DVD.pooled.testbed.base | C19M16__base__base | 3/3 | 3/3 | true | pass,pass,pass | 30,000,000 | 15,000,000 | 15,000,000 | 1 |
| ENCSR131DVD.pooled.testbed.nodedup | C19M16__dedup__off | 3/3 | 3/3 | true | pass,pass,pass | 30,000,000 | 15,000,000 | 15,000,000 | 1 |
| ENCSR131DVD.pooled.testbed.pe | C19M16__pe__pe | 3/3 | 3/3 | true | pass,pass,pass | 30,000,000 | 15,000,000 | 15,000,000 | 2 |
| ENCSR401WUV.pooled.testbed.base | C07M29__base__base | 3/3 | 3/3 | true | pass,pass,pass | 30,000,000 | 15,000,000 | 15,000,000 | 1 |
| ENCSR401WUV.pooled.testbed.nodedup | C07M29__dedup__off | 3/3 | 3/3 | true | pass,pass,pass | 30,000,000 | 15,000,000 | 15,000,000 | 1 |
| ENCSR401WUV.pooled.testbed.pe | C07M29__pe__pe | 3/3 | 3/3 | true | pass,pass,pass | 30,000,000 | 15,000,000 | 15,000,000 | 2 |
| ENCSR564OSQ.pooled.testbed.base | C40M17__base__base | 3/3 | 3/3 | true | pass,pass,pass | 30,000,000 | 15,000,000 | 15,000,000 | 1 |
| ENCSR564OSQ.pooled.testbed.nodedup | C40M17__dedup__off | 3/3 | 3/3 | true | pass,pass,pass | 30,000,000 | 15,000,000 | 15,000,000 | 1 |
| ENCSR564OSQ.pooled.testbed.pe | C40M17__pe__pe | 3/3 | 3/3 | true | pass,pass,pass | 30,000,000 | 15,000,000 | 15,000,000 | 2 |
| ENCSR643QXU.pooled.testbed.base | C40M18__base__base | 3/3 | 3/3 | true | pass,pass,pass | 30,000,000 | 15,000,000 | 15,000,000 | 1 |
| ENCSR643QXU.pooled.testbed.nodedup | C40M18__dedup__off | 3/3 | 3/3 | true | pass,pass,pass | 30,000,000 | 15,000,000 | 15,000,000 | 1 |
| ENCSR643QXU.pooled.testbed.pe | C40M18__pe__pe | 3/3 | 3/3 | true | pass,pass,pass | 30,000,000 | 15,000,000 | 15,000,000 | 2 |
| ENCSR882CQE.pooled.testbed.base | C19M22__base__base | 3/3 | 3/3 | true | pass,pass,pass | 30,000,000 | 15,000,000 | 15,000,000 | 1 |
| ENCSR882CQE.pooled.testbed.nodedup | C19M22__dedup__off | 3/3 | 3/3 | true | pass,pass,pass | 30,000,000 | 15,000,000 | 15,000,000 | 1 |
| ENCSR882CQE.pooled.testbed.pe | C19M22__pe__pe | 3/3 | 3/3 | true | pass,pass,pass | 30,000,000 | 15,000,000 | 15,000,000 | 2 |
| ENCSR903SKE.pooled.testbed.base | C12M02__base__base | 3/3 | 3/3 | true | pass,pass,pass | 50,000,000 | 25,000,000 | 25,000,000 | 1 |
| ENCSR903SKE.pooled.testbed.nodedup | C12M02__dedup__off | 3/3 | 3/3 | true | pass,pass,pass | 50,000,000 | 25,000,000 | 25,000,000 | 1 |
| ENCSR903SKE.pooled.testbed.pe | C12M02__pe__pe | 3/3 | 3/3 | true | pass,pass,pass | 50,000,000 | 25,000,000 | 25,000,000 | 2 |

Read totals before the subsample are in `totals.tsv`. The 7 `.base` totals were measured with `samtools idxstats` on the Nibi EIC_REPRO nodup BAMs. The 14 `.nodedup` and `.pe` totals are `NA`: the unsubsampled BAMs of the FASTQ-route arms are only in the Nibi nearline tar files, so `rows.tsv` has `q_origin = fitted` for all 14 pairs (the source rate is fitted, the target rate is measured).
