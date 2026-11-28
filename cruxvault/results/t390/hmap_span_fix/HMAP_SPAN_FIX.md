# t390 hmap_span_fix: SPAN_EXTRA 24 -> 25 in the t125 depth-offset code

## The change

The t125 formula h and its features convert 25 bp overlap counts to MACS2 tags and pileup with
`count * 25 / (rl + SPAN_EXTRA)` and `count * fraglen / (rl + SPAN_EXTRA)`. Two copies of the constant
were 24: `tools/t125/hmap/caller.py` (`SPAN_EXTRA = BIN - 1`) and `tools/t125/hmap/formula.py`
(`SPAN_EXTRA = 24`). `formula.SPAN_EXTRA` feeds `formula._tags` (the control and no-control window
lambdas), `formula.pileup`, and `data.formula_log1p_t` (the f-NB training column). `caller.SPAN_EXTRA`
and `caller.PILEUP_RULE` are read only by tests.

Both constants are now `BIN` = 25, the same fix as `tools/onion_eval/signal_from_counts.py` (commit
0f957f05; derivation in `../span_fix/SPAN_FIX.md`). A read of length rl touches (rl + 25)/25 bins on
average under the store count rule.

The counts that hmap reads follow the store rule. `hmap.data.HCorpus` reads the treatment counts with
`ladder.data.Corpus` (`counts25.npz` of the t112 products) and the control counts from the cache that
`hmap/controls.py` builds from `control_counts25.npz`. `tools/t112/bam_arm.py` writes both files with
`bin25.counts_from_tagalign`, i.e. +1 to bins start//25 .. end//25 inclusive.

Tests. `tests/test_t125_h_features.py` has a new test, `test_span_extra_matches_the_bin25_rule`. It
bins reads at all 25 start phases with `counts_from_tagalign` and asserts that the mean bins per read
equals (rl + SPAN_EXTRA)/25 for both constants, for rl in 25, 36, 50, 75, 76, 100, 101. With the
constants patched back to 24 the test fails (2.0 against 1.96 at rl = 25). The known answers that
hard-coded 24 now use 25: `test_caller_constants` and 13 values in `tests/test_t125_h_formula.py`
(`25/(101 + 24)` -> `25/(101 + 25)`, `180/125` -> `180/126`, and the others).

Test runs (laptop, conda env `candii`). Before the change, the 19 `tests/test_t125_*.py` files gave
690 passed, 2 skipped. After the change, those files plus the 52 `tests/test_onion_eval_*.py` files gave
1,494 passed, 7 skipped, 2 failed. Neither failure touches hmap; onion_eval does not import it.
`test_onion_eval_score_memory.py::test_outputs_identical_to_base` also fails on the committed tree
without this change (a stale `marginal_nb` / `position_average` digest pin).
`test_onion_eval_score.py::test_end_to_end_train_score_compare` failed only in the full-suite run (a
checkpoint with keys "w", "pad" loaded into the full Model) and passes when its file runs alone (26 passed on the new tree; also passes on the committed tree).

## The constant is the only change (bit-exact)

"old" is the tree at 0dd6ddc7. "new24" is the new tree with both constants set back to 24 by
`patch24/sitecustomize.py`, which runs in every python process. "new25" is the new tree as committed.

| comparison | what | compared | equal | file |
|---|---|---|---|---|
| old vs new24 | `ref_formula.py` arrays (5 synthetic products x 8 arrays) | 40 | 40 bit-exact | `synth_bitexact.tsv` |
| old vs new24 | `smoke.py h --steps 10` numeric values (scores, aggregates, losses, checkpoint tensors) | 6,423 | 6,423 | `smoke_bitexact.tsv` |
| old vs old (rerun) | same smoke, determinism control | 6,423 | 6,423 | same |
| old vs new25 | same smoke | 6,423 | 4,833 (1,590 differ) | same |

The smoke comparison skips timing, date and path keys. The 1,590 values that move are all in the
formula h (`h_formula_bin`, `h_formula_frag`), in the h trained on the formula column
(`h_formula_mlp_r5_push_fnb`: 6 checkpoint tensors, 3 loss values), and in the aggregates built from
them. The f model and the h models without the formula column do not move.

## What moves

### Synthetic tracks (`synth_moved.tsv`)

`ref_formula.py`: 20,000 bins per product, k = 0..63 on 207 bins for the enumerated arrays. The
pileup scales by (rl + 24)/(rl + 25): 0.99206 at rl = 101, 0.98684 at rl = 76, 0.98361 at rl = 36. The
window lambdas scale by the same factor, then lambda_bg (which does not scale) clips them. P moves
non-uniformly because of floor(pileup) and the lambda_bg clip.

| product | rl | mean P bin old -> new | mean P frag old -> new | mean lambda old -> new | bins of P bin that differ |
|---|---|---|---|---|---|
| ChIP, control, base | 101 | 17.3209 -> 17.177 | 15.8771 -> 15.74 | 2.18726 -> 2.1793 | 9,266 of 20,000 |
| ChIP, control, treatment scaled (q0.25) | 101 | 2.04137 -> 2.02794 | 1.76278 -> 1.75062 | 2.76133 -> 2.73941 | 20,000 |
| ChIP, ratio arm V = 0.56 | 101 | 11.5215 -> 11.4022 | 10.1742 -> 10.0749 | 2.33678 -> 2.32905 | 6,932 |
| ChIP, control rl 50.3 | 36 | 62.1272 -> 61.0457 | 57.7122 -> 56.7571 | 3.6429 -> 3.59452 | 19,997 |
| DNase, no control | 76 | 5.86031 -> 5.76351 | 4.72645 -> 4.67738 | 8.19259 -> 8.12017 | 14,360 |

### Smoke scores (`smoke_moved.tsv`, synthetic products, 70 aggregate rows)

The smoke products come from `ladder.synth`, so their "true P" is not a real caller output. The
direction of these changes says nothing about which constant is right. Selected rows:

| h | marks | metric | old | new |
|---|---|---|---|---|
| formula bin, alone | DNase | crps_all | 0.777069 | 0.723879 |
| formula bin, alone | DNase | scale_error_all | 0.43339 | 0.381885 |
| formula bin, alone | narrow | crps_all | 1.88015 | 1.86981 |
| formula frag, alone | narrow | crps_all | 1.36716 | 1.35635 |
| formula bin, chain | narrow | crps_all | 1.50241 | 1.4977 |
| formula frag, chain | narrow | crps_all | 1.91391 | 1.89846 |
| formula + MLP (f-NB trained), chain | narrow | crps_all | 1.89918 | 1.8839 |

The largest relative change is -11.9% (formula bin, DNase, scale_error_all). The other rows move by
at most 9.9%, and most by under 2%.

### Real products, chr21 (`real_chr21.tsv`)

`real_chr.py` ran `formula.P` with the constant at 24 and at 25 on 8 t112 products (1,868,399 bins
each) and compared each with the product's true P (`pval25.npz`). Fir job 5756066 ran it (4 CPUs,
2 min 46 s elapsed, 0.18 CPU-h allocated, peak RSS 0.94 GB). No noise floor exists for these
numbers: they are deterministic functions of one chromosome, and there is no replicate.

| product | assay | rl | variant | MSE 24 -> 25 | Pearson 24 -> 25 | mean P 24 -> 25 (true) | bins that differ |
|---|---|---|---|---|---|---|---|
| C07M20 base | H3K4me1 | 101 | bin | 0.139194 -> 0.137487 | 0.808815 -> 0.808430 | 0.300283 -> 0.300822 (0.289409) | 37.8% |
| C07M20 base | H3K4me1 | 101 | frag | 0.055934 -> 0.056164 | 0.902267 -> 0.902414 | 0.249310 -> 0.247233 | 38.2% |
| C07M20 crop 36 | H3K4me1 | 36 | bin | 0.598318 -> 0.607647 | 0.589086 -> 0.589851 | 0.384113 -> 0.389496 (0.275400) | 43.9% |
| C07M20 crop 36 | H3K4me1 | 36 | frag | 0.108339 -> 0.108048 | 0.805437 -> 0.805771 | 0.240274 -> 0.243865 | 43.9% |
| C07M20 ratio k2 | H3K4me1 | 101 | bin | 0.028032 -> 0.027817 | 0.765414 -> 0.764436 | 0.099096 -> 0.099972 (0.091380) | 71.4% |
| C07M20 ratio k2 | H3K4me1 | 101 | frag | 0.010628 -> 0.010641 | 0.873896 -> 0.874009 | 0.074292 -> 0.074163 | 71.4% |
| C07M20 ctldepth q0.25 | H3K4me1 | 101 | bin | 0.132491 -> 0.130798 | 0.807600 -> 0.807174 | 0.280653 -> 0.281208 (0.265147) | 41.8% |
| C07M20 ctldepth q0.25 | H3K4me1 | 101 | frag | 0.053235 -> 0.053372 | 0.898465 -> 0.898547 | 0.232237 -> 0.230344 | 42.1% |
| C07M20 no control | H3K4me1 | 101 | bin | 0.139174 -> 0.137003 | 0.789715 -> 0.789713 | 0.309375 -> 0.309174 (0.294643) | 17.3% |
| C07M20 no control | H3K4me1 | 101 | frag | 0.052153 -> 0.052480 | 0.892215 -> 0.892353 | 0.256062 -> 0.253098 | 17.9% |
| C19M22 base | H3K4me3 | 101 | bin | 0.282268 -> 0.277409 | 0.982988 -> 0.982966 | 0.376735 -> 0.376692 (0.370285) | 39.6% |
| C19M22 base | H3K4me3 | 101 | frag | 0.146885 -> 0.150969 | 0.990935 -> 0.990879 | 0.337694 -> 0.329404 | 41.1% |
| C12M02 base | DNase-seq | 76 | bin | 0.535693 -> 0.492499 | 0.961545 -> 0.962031 | 0.494223 -> 0.456063 (0.458008) | 32.3% |
| C12M02 base | DNase-seq | 76 | frag | 0.105867 -> 0.107013 | 0.989888 -> 0.989865 | 0.413472 -> 0.407390 | 27.0% |
| C40M17 base | H3K27me3 | 101 | bin | 0.099682 -> 0.092066 | 0.775069 -> 0.777789 | 0.287507 -> 0.284727 (0.276296) | 48.8% |
| C40M17 base | H3K27me3 | 101 | frag | 0.043736 -> 0.044636 | 0.871903 -> 0.871444 | 0.233777 -> 0.229256 | 49.2% |

The bin variant's MSE fell on 7 of 8 products. The largest fall was DNase C12M02 (-8.1%), whose mean P
moved from 0.494 to 0.456 against a true mean of 0.458. The frag variant's MSE rose on 7 of 8
products, by at most 2.8% (C19M22). The crop-36 bin variant rose by 1.6%. Pearson r moved by at most
0.0027. These are chr21 readings on one product per arm, without a noise floor. The t125 grid
results that used the old constant (Nibi, 2026-10) are not rescored here.

## Files

| file | what |
|---|---|
| `ref_formula.py` | the synthetic-track dump (`--span`, `--repo`) |
| `smoke_compare.py` | numeric comparison of two smoke work dirs |
| `tables.py` | the before/after tables |
| `real_chr.py` | the real-product chr21 comparison (ran on Fir) |
| `patch24/sitecustomize.py` | sets both constants back to 24 in every process (bit-exact check only) |
| `runall.sh` | rebuilds every TSV; no arguments; skip-if-exists; checks MANIFEST.tsv |
