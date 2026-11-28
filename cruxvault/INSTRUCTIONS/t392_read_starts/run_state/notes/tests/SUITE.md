# Full test suite on Fir: new code vs reference code (C10b)

Date: 2026-11-24. Logs and failure lists: Fir `/project/def-maxwl/mforooz/t392/logs/`.

| kit | code | job | state | elapsed | CPUTimeRAW s | failure ids |
|---|---|---|---|---|---|---|
| RS2 | read-start umbrella at aa796683 | 5521259 | COMPLETED | 02:22:57 | 68,616 | 66 |
| RCP9 | reference recipe base | 5521260 | COMPLETED | 02:12:55 | 63,800 | 471 (75 failed + 396 errors) |

`comm -13 failures_RCP9 failures_RS2` is empty: no failure is new in the read-start code. All 66 failures of RS2 also fail on RCP9. 405 failure ids of RCP9 do not fail on RS2.
No test of the read-start code (`tests/test_t392_*`) is among the failures.
Spend: (68,616 + 63,800) / 3600 = 36.8 CPU-h.
