Rows enter via `python tools/leaderboard.py add` — one stamped score json per row, full provenance mandatory. Never write a row file by hand.

One row is one address (`plan/BENCHMARK_DESIGN.md` §1): method · regime · truth · panel · scope · metric. The first five are the path, the last is the row's contents:

    leaderboard/rows/<regime>/<truth>.<panel>.<scope>/<method>@<version>.json

`add` refuses a row that leaves any of those five unknown, so a row that cannot be addressed cannot reach the ranked table. Anchor rows — the 2019 submissions, which carry no regime because we never trained them — live under `leaderboard/anchor/` instead, and rows retired by §3.3 live under `leaderboard/void/`.
