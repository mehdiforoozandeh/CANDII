---
id: f163
type: finding
schema: 3
title: Scored in read-start space after Richardson-Lucy deconvolution, the…
parent: h57
direction: no-answer-yet
confidence: weak
why: Scored in read-start space after Richardson-Lucy deconvolution, the read-count model tied the read-start model (0.588 vs 0.578, seed ranges 0.067 and 0.072), but the deconvolution failed its own validation gate (round trip on H3K36me3 and H3K9me3; conversion floor 0.133 > 0.10) and amplified count noise, so the start-space test is void; in count and 1 kb spaces the read-count model led beyond seed noise (0.543 vs 0.375; 0.394 vs 0.184)
vs_expect: expectation of a read-count lead in start space beyond seed range not met; test voided by gate (c)
evidence: results/t396/legB_spaces/LEG_B.md
created: "2026-11-28T00:09:47"
updated: "2026-11-28T00:09:47"
---

# f163 — Scored in read-start space after Richardson-Lucy deconvolution, the…

Parent:: [[h57_the_read_count_model_still_beats_the_rea]]
