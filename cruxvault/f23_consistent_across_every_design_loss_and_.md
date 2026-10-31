---
id: f23
type: finding
schema: 3
title: Consistent across every design, loss and size tried, with the PI closing the…
parent: q7
confidence: moderate
why: Consistent across every design, loss and size tried, with the PI closing the scale branch on 2026-10-31; moderate because the work was exploratory and the size and loss checks were written after their runs.
evidence: results/h19/report.md, results/t128p3/m5_depthlaw_table.tsv, results/t136/DIGEST.md
created: "2026-10-31T12:58:16"
updated: "2026-10-31T12:58:16"
---

# f23 — Consistent across every design, loss and size tried, with the PI closing the…

Parent:: [[q7_why_do_the_covariate_conditioned_g_f_des]]

## Answer

Not the pseudocount, the loss, or the size of g or f at the widths tried: the miss is a failure to generalize. g fits trained base ↔ arm depth pairs but does not carry the rule to never-trained arm → arm pairs, in every design tried. Chaining two trained legs through the base meets the law, so the pivot through the base is the working fix. Arm-to-arm training on two pairs (Darm1) learns the scale but not the distribution.
