"""`competitors/` — everything that produces prediction tracks and is not CANDI (`RIVALS_PLAN.md` §3).

ONE RULE, ONE DIRECTION. `candi` never imports anything under here; the only thing that crosses
between the two worlds is the §4.1 prediction-track format on disk. Code under `competitors/` may
READ the corpus store (`candi.store`) to get its training data and the bin grid — a rival trained on
our data has to be handed our data somehow, and re-implementing the store's codecs and its
`floor(chr_len / 25)` grid in a second place is exactly the silent mismatch `bench.external`'s length
assertion exists to catch. Nothing here imports `candi.bench`, `candi.model` or `candi.train`.

This package is not installed: `pyproject.toml` packages `src/` only.
"""
