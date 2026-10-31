# The pivot translator, rounds 1 and 2: unified lab deck

Artifact: https://claude.ai/artifact/EAbUbS4xUAfNyWyU6wwogh (private). Version 1 was published on 2026-10-26. It uses 233 of 240 round-1 runs (7 agreement-setting runs were cancelled on 2026-10-25 and were not rerun) and 96 of 96 round-2 runs.

It joins [the phase-1 deck](../t128-encdec-sym/DECK.md) and [the phase-2 deck](../t128p2-encdec/DECK.md) into one story for a lab audience that does not know the design. The deck gives new plain names to every model and defines them before the results. A backup slide maps the old names to the new ones:

| this deck | earlier decks |
|---|---|
| direct translator | old D (t118 / t126 rung D) |
| two-step via the base | old D through the base |
| pivot translator | symmetric encoder-decoder |
| learned pivot | learned middle, form A |
| boxed pivot | form D |
| base-labelled pivot | fixed middle |
| anchored pivot | anchored middle |
| round 1 / round 2 | phase 1 / phase 2 |

Source: `index.html` beside this file. `build_deck.py` builds it from `deck_src.html`. Its numbers come from `cruxvault/results/t128/` and `cruxvault/results/t128p2/` (gitignored; the main checkout's copies by default), with a sha256 sources manifest. To rebuild, run `bash tools/t128/readings/run_all.sh`, then `bash tools/t128p2/readings/run_all.sh`, then `python cruxvault/presentations/t128-pivot/build_deck.py`. `crux deck --lint` and `crux deck --verify --strict` both pass.

The study is exploratory, so the deck gives readings, not verdicts.

## Arc (29 slides)

- **Background and problem (slides 1–6).** What a track is. The task as translation between processing settings. Training uses base ↔ arm pairs only, and the test is never-trained arm → arm pairs. The problem: the direct translator fits trained depth pairs and misses never-trained ones, but chaining it through the base works. The idea: build that detour into the model as a pivot.
- **Hypothesis (slide 7).** A translator that sends every track through a shared middle, with one network for the way in and the way out, carries its fit over to never-trained pairs. The boring explanation to rule out: the pivot copies its input.
- **Method (slides 8–13).** g sets the FiLM of f. The three routes. The cross loss and the round-trip loss, and the two extras. The four pivot variants. The grid of variants. CRPS with its shape and scale split, and the noise rule.
- **Results (slides 14–22), one finding per slide, one g per track.**
  1. The round-trip loss alone copies the input; the cross loss is needed.
  2. The pivots pass the swap check far from the identity.
  3. Main result: no pivot beats the direct translator on never-trained pairs outside noise; only the two-step via the base does.
  4. The anchored pivot's first half matches the direct translator's arm → base map; the gap is in the shared decoder.
  5. The loss extras (agreement, noisy middle, box) are within noise of the learned pivot.
  6. Training longer does not help.
  7. The pivots cost nothing on trained pairs or the held-out depth.
  8. The depth test is not met on never-trained depth pairs.
  9. Only the anchored pivot brings two arms' middles closer than their inputs.
- **Discussion (slides 23–26).** The nine findings, the reading against the hypothesis, five caveats, and four options for the next round (no ranking).
- **Backup (slides 27–29).** Names, one g across tracks, run counts.
