"""t125 part 1: the t118 count ladder g -> f with a depth offset (plan/ETA_OFFSET_AND_H_DESIGN.md).

eta_in = log(1 + X) - log d (d = source depth in millions of reads), and the target NB has
log mu' = log d' + eta_out. Modules: `offset` (the transform), `pairs` (rungs, the 432-row task
table, the depth-only law pairs).

Import convention (pinned for every t125 module and test): put `<repo>/tools/t118` AND
`<repo>/tools/t125` on `sys.path`, then `from ladder import ...`, `from eta import ...`,
`from hmap import ...`. No t118 file is edited; t118 code is reused by import.
"""
