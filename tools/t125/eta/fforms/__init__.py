"""t125 f-forms that differ from t118's: `form_a.py` (row-1 A with its spread set by level).

Each module exposes `build(space, stats) -> ladder.base.FForm`. The loader is `eta.model.load_form`,
which maps a t125 rung to a module through `eta.pairs.FORM_MODULE`; every other rung reuses its
`ladder.fforms` module unchanged.
"""
