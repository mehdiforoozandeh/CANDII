# Deck — which processing settings change a track the most (exploratory, 2026-10-26)

Published: https://claude.ai/artifact/1T1NKMYo5taJjDowse9r2f
Source: `deck_src.html` + `build_deck.py` here; numbers from `results/t133/pairs.tsv`. Rebuild with
`python build_deck.py`, then republish `index.html` to the same URL.

Every bar carries its base → arm setting.

1. Title — 192 base→arm comparisons, 121 arm products, 7 tracks.
2. Motivation — one setting moved per arm; read side vs control side.
3. The question — peak height and shape, kept apart.
4. Grid of the ten settings, their levels, how each was made.
5. Metrics — peak height (top-1% bins, log2), shape (1 − rank correlation, 1 kb), FRiP (own top 1%).
6. (a) Peak height, counts — depth leads; antibody proxy and read length next.
7. (b) Shape, counts — antibody proxy leads; re-run settings bunch at 0.35–0.45.
8. (c) Shape, −log10 p — control-side settings ≤ 0.17; control depth weakest.
9. (d) DNase — keeping duplicates: FRiP +0.53 (unchecked); paired-end +0.34 (signed bars, + = more reads in peaks).
10. (e) Why — schematic: a re-run arm differs from its base by the setting AND a new random 30M draw; a depth cut keeps a subset of the base's reads.
11. (e) Evidence — re-run settings (shape 0.35–0.45) score like keeping a quarter to an eighth of the reads.
12. Discussion — clear ends, unresolved middle, doubts (regression to the mean, FRiP at low depth).
13. Options — A re-draw floor; B shared peak set; C check the DNase jump.
