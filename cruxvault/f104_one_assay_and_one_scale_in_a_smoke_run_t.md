---
id: f104
type: finding
schema: 3
title: One assay and one scale in a smoke run; the cause is shown by matched…
parent: q59
confidence: moderate
why: One assay and one scale in a smoke run; the cause is shown by matched refits, but the remaining gap is not fully resolved
evidence: results/closeout_E4/E4_MIXED_DIAG.md
created: "2026-11-17T06:00:29"
updated: "2026-11-17T06:00:29"
---

# f104 — One assay and one scale in a smoke run; the cause is shown by matched…

Parent:: [[q59_do_the_nested_replicate_ladder_and_the_c]]

## Answer

Not for the small shares. On H3K36me3 at the finest scale the two methods use the same response (they agree within 2 to 3 percent on 12 identical pairs), but the mixed model puts cell type at about 0 while the ladder gives a share of about 0.08. The mixed model's fit is sound; cell type sits on the bound because replicate has the same partition as cell type in 15 of 19 cell types, so only their sum (0.0064) is identified, and its single residual variance fails because the per-track sampling variance ranges from 0.021 to 0.049 per bin. The ladder subtracts the lower rung of one side only; a two-sided subtraction lowers cell type over sampling from 0.00187 to 0.00144. The ladder's sampling and replicate rungs can be quoted; its cell-type and other-experiment rungs are upper values; the mixed model's small fractions cannot be quoted until it has a per-library residual variance.
