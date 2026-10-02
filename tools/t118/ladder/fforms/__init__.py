"""The f-forms, one module per rung: `form_a.py` ... `form_d.py` (and `form_identity.py`, a test stub).

Each module exposes `build(space, stats) -> ladder.base.FForm`. Nothing registers them:
`ladder.base.load_form(rung, space, stats)` imports `ladder.fforms.form_<rung.lower()>` by name.
"""
