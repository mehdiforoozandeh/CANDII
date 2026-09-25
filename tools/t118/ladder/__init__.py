"""t118 architecture ladder: g(C, C') -> theta, f_theta(X) -> X' per 25 bp bin.

Import convention (pinned for every module and test): put `<repo>/tools/t118` on `sys.path`, then
`import ladder` and `from ladder import base, data, pairs`. Modules: `pairs` (constants, the pair
and task tables), `data` (the corpus reader and the memory-mapped cache), `base` (the f-form
interface), `synth` (synthetic products for tests), `fforms/form_<rung>.py` (one file per rung).
"""
