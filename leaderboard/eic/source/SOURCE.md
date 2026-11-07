# Where these four tables come from

Copied unchanged from Max Libbrecht's research repo (private):
`mlibbrecht/01_2026-10-19_research`, folder
`projects/03_2026-10-20_epi_imputation/sprints/04-2026-10-28_eic/output/`.
The tables were committed in `b27005e311e9be548608a927c57a34f7bebb4498` (2026-10-30) and are
unchanged at `51870703af225e32f6af4e9cb03294d89886674e` (2026-11-04), when they were copied here.

| file here | file there | what it is |
|---|---|---|
| `pe_vs_MOESM2_board.tsv` | `compare/pe_vs_MOESM2_board.tsv` | the 25 published entries scored against the original 2019 truth; reproduces Additional File 2 (MOESM2) position for position |
| `pq_vs_MOESM3_board.tsv` | `compare/pq_vs_MOESM3_board.tsv` | the same against the original truth after quantile normalization (rebuilt); reproduces MOESM3 position for position |
| `qn_vs_MOESM4_board.tsv` | `compare/qn_vs_MOESM4_board.tsv` | the same against the single-end reprocessed + normalized truth (Zenodo 14736610); reproduces MOESM4 value for value |
| `candi_placement.tsv` | `candi_placement.tsv` | each CANDI arm added alone as a 26th entry to each board, and re-ranked |

sha256 at copy time:

```
6f32eceaf6bc143548bdfb8d3dd70f89911b0bc9743375ffaa0f57c3a7896241  pe_vs_MOESM2_board.tsv
eac6de0eb8090a7eab63713542f769a69a536b3d53aafe025da89b6f4138c2f7  pq_vs_MOESM3_board.tsv
ca676404faad480b289174e308c28b9d24d82b859a6fd7c11b4c93c1f408dde5  qn_vs_MOESM4_board.tsv
4e88465465994b2946ccf370f903b1fef635f6f8cde606437ccdc041babf8c18  candi_placement.tsv
```

The method is in that sprint's `report.html` (source `scripts/report/report.md`): the scorer is
`scripts/vendor031/metrics9.py` / `scripts/score/fastglobal.py`, the ranking statistic
`scripts/vendor031/rank_eic.py`, the normalization `scripts/qnorm/apply_bp.py`. Per-row scores
live on Nibi under `/project/def-maxwl/maxwl/epi_imputation/s04/scores/<truth>/<entry>/`.

To add a new CANDI version: score its 51 blind predictions with Max's `scripts/score/` against
truth sets `pe`, `pq` and `qn`, place it with `scripts/compare/place_candi.py`, and append its
`published MOESM*` rows to `candi_placement.tsv` (same columns). Then name the arm in
`tools/eic_board.py` (`ARMS`).
