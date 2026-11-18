---
id: f128
type: finding
schema: 3
title: Read from the full pytest log; 32 earlier failures were fixed without…
parent: q27
confidence: strong
why: Read from the full pytest log; 32 earlier failures were fixed without weakening a check; later commits change documents only.
evidence: results/closeout_records/TEST_GATES.md
created: "2026-11-18T16:26:09"
updated: "2026-11-18T16:26:09"
---

# f128 — Read from the full pytest log; 32 earlier failures were fixed without…

Parent:: [[q27_are_the_onion_run_s_estimators_scorers_a]]

## Answer

The full test suite on the umbrella branch passes (6,950 passed, 9 skipped, 0 failed) and the golden check is bit-exact (0 ULP) against origin/main, so both merge gates of a non-experiment pull request hold for the code state 6b2a2273.
