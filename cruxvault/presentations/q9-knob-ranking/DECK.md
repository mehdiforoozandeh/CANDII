# Deck — which processing settings change a track the most (exploratory, 2026-10-26)

Published: https://claude.ai/artifact/1T1NKMYo5taJjDowse9r2f
Source: `deck_src.html` + `build_deck.py` here; numbers from `results/t133/pairs.tsv`. Rebuild with
`python build_deck.py`, then republish `index.html` to the same URL.

1. Title — 192 base→arm comparisons, 121 arm products, 7 tracks.
2. Motivation — one setting moved per arm; read side vs control side.
3. The question — peak height and shape, kept apart.
4. Grid of the ten settings, their levels, how each was made.
5. Metrics — peak height (top-1% bins, log2), shape (1 − rank correlation, 1 kb), FRiP (own top 1%).
6. (a) Peak height, counts — depth leads; antibody proxy and read length next.
7. (b) Shape, counts — antibody proxy leads; re-run settings bunch at 0.35–0.45.
8. (c) Shape, −log10 p — control-side settings ≤ 0.17; control depth weakest.
9. (d) DNase — keeping duplicates: FRiP +0.53 (unchecked); paired-end +0.34.
10. (e) Caution — re-run settings sit between the quarter- and eighth-depth cuts on shape.
11. Discussion — clear ends, unresolved middle, doubts (regression to the mean, FRiP at low depth).
12. Options — A re-draw floor; B shared peak set; C check the DNase jump.
