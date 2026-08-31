Void rows — kept as a record, never printed as numbers.

`plan/BENCHMARK_DESIGN.md` §3.3: the old "Full genome" board scored all 23 chromosomes, chr19
included, and chr19 is the chromosome CANDI trained on. Those numbers are not portable to the
regime design and are not carried forward. The 2019-entrant rows are void for a second reason
(§6): they were scored by the vendored 001 scorer, which retires to a one-time recorded proof in
the vault and never scores a board row again.

The files stay here, byte-identical to how `add` stamped them, so the record of what was scored
survives in git. `tools/leaderboard.py` reads them, re-runs the row shape gate on each, and
compiles **only the method name, the version, the date and the board they came from** into the
payload. It drops every metric value on the way, on purpose: a void number rendered under a new
label would read as freshly computed.

Nothing is ever added here by hand. When a regime's retrain lands, the new row is stamped into
`leaderboard/rows/<regime>/<truth>.<panel>.<scope>/` by `add`, and the void entry stays where it
is.
