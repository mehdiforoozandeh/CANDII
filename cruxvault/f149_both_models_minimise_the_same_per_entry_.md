---
id: f149
type: finding
schema: 3
title: Both models minimise the same per-entry NB mean on one code path with the…
parent: h52
direction: against
confidence: weak
null: none
why: Both models minimise the same per-entry NB mean on one code path with the same schedule; the 3x smaller gradients are absorbed by AdamW; a code reading, no run.
vs_expect: "Not met: no uncompensated difference found."
evidence: results/t394/INVESTIGATION.md
created: "2026-11-25T00:25:44"
updated: "2026-11-25T00:25:44"
---

# f149 — Both models minimise the same per-entry NB mean on one code path with the…

Parent:: [[h52_weighting_each_read_once_in_the_start_lo]]
