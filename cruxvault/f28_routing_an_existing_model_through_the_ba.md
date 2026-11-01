---
id: f28
type: finding
schema: 3
title: Routing an existing model through the base already lowered held-out CRPS on…
parent: h30
direction: no-answer-yet
confidence: weak
null: not-ruled-out
why: Routing an existing model through the base already lowered held-out CRPS on never-trained arm-to-arm pairs for one model per track (0.545 against 0.662 direct, seed spread 0.008), but for the one model shared across tracks the gain was within noise (0.636 against 0.667, spread 0.064), and that shared case is what this claim tests; read after the runs, 3 seeds, old design D, before the planned 6-seed test.
evidence: results/t128p3/analyze_out.txt
vs_expect: "Shared model: gain 0.031 against a needed 2x spread of 0.128, so below the bar; per-track model: gain 0.117, far above 2x spread 0.016, but outside this claim's scope."
created: "2026-10-31T18:10:39"
updated: "2026-10-31T18:10:39"
---

# f28 — Routing an existing model through the base already lowered held-out CRPS on…

Parent:: [[h30_chaining_one_shared_count_space_correcti]]
