---
id: f402
type: finding
schema: 3
title: The formula-based h assumed (rl+24)/25 bins per read; the store rule gives…
parent: q6
confidence: moderate
evidence: results/t390/hmap_span_fix/HMAP_SPAN_FIX.md
created: "2026-11-24T12:13:48"
updated: "2026-11-24T12:13:48"
---

# f402 — The formula-based h assumed (rl+24)/25 bins per read; the store rule gives…

Parent:: [[q6_how_closely_does_a_map_h_reproduce_the_l]]

## Answer

The formula-based h assumed (rl+24)/25 bins per read; the store rule gives (rl+25)/25. With the fix, the code reproduces the old outputs bit for bit when set back to 24 (40/40 formula arrays, 6,423/6,423 smoke values). On 8 real chr21 products, MSE against the true P fell on 7 of 8 for the bin variant (largest: DNase 0.536 -> 0.492) and rose on 7 of 8 for the frag variant, by at most 2.8%. Single readings, no noise floor; the old t125 grid was not rescored.
