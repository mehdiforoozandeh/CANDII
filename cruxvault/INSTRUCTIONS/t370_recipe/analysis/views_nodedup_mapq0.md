# Why the alignment-level views hurt the no-dedup plus MAPQ-0 pairs (chunk C9, 2026-11-21)

Question q45, third analysis of D19, requirement R12. No new compute: every number below was read or recomputed from stored score files on Fir. The recompute script is in the appendix.

## Question

Adding the alignment-level views (no duplicate removal, i.e. the `nodedup` view; no MAPQ filter, i.e. the `mapq0` view) to the U-Net's training lowers gap closed on the never-trained pairs that join these two views. On the superseded half-read floor scale the value falls from 1.77 (base view only) to 0.08 (base + alignment-level views). Why?

Terms used below. A **mapq + nodedup pair** is a correction pair of one experiment whose source is the `mapq0` view and whose target is the `nodedup` view, or the reverse (`pairs_per_region.npz`, `knob` = `mapq+nodedup`; 84 pairs on the 16-region screen, 42 in each direction). Per pair, the scorer stores four CRPS values (`tools/onion_eval/score.py`, `CORRECTION_ONLY_DEFINITION`; `tools/onion_eval/oracle_floor.py`, `pair_records`):

- m, the model on the pair (source pr1 half to target pr2 half);
- s, the model on the same-view pair of the target (target pr1 to target pr2, the target's covariates on both sides);
- d, depth-only scaling, i.e. the source pr1 counts times the depth ratio, with one fitted NB size;
- f, the half-read floor, i.e. the target's own pr1 half used as the mean;
- O, the oracle floor, i.e. 0.5 x mean |pr1 - pr2| of the target.

Gap closed on the oracle scale is (d - m) / (d - O); on the half-read scale it is (d - m) / (d - f). A model that copies the source half (scaled to the target depth) has m = d. A model that copies its input on the same-view pass has s = f.

## Evidence

Source: Fir `/project/def-maxwl/mforooz/ONION/runs_fir/<arm>/seed{0,1,2}/{p16,p16_sv}/` (`scores.json`, `pairs_per_region.npz`, `same_view_per_region.npz`), joined with the oracle reference that `oracle_floor.find_oracle` selects, code snapshot `/project/def-maxwl/mforooz/onion_kit/CM`. Values are pair means pooled per direction, then the seed mean with the seed range (max minus min) in parentheses. "Beyond 2x" means the difference of seed means is larger than twice the larger seed range (the D13 bar).

Seed counts. The U-Net alignment arm has 3 seeds in `p16` and 2 in `p16_sv` (seed 2 same-view scoring failed with "Disk quota exceeded", `ablations_closeout/ABLATIONS_READ.md`, section 5). I use `p16` (3 seeds) for m and `p16_sv` (2 seeds) for s. The cross-view values m, d, f, O are equal in `p16` and `p16_sv` for every U-Net seed I compared.

### 1. The 1.77 and the 0.08 reproduce, and they are the same effect as 0.377 against 0.018

Pooled over the 84 mapq + nodedup pairs, d = 0.5663, f = 0.5200, O = 0.3488, so d - f = 0.0463 and d - O = 0.2175. The half-read denominator is 4.7 times smaller, so the half-read scale inflates every value by about 4.7.

- U-Net base only, m per seed = 0.4834, 0.4746, 0.4952. Half-read gap closed = 1.79, 1.98, 1.53: mean 1.77 (0.45). Oracle scale: 0.377 (0.095).
- U-Net base + alignment-level, m per seed = 0.5657, 0.5532, 0.5686. Half-read gap closed = 0.01, 0.28, -0.05: mean 0.08 (0.33). Oracle scale: 0.018 (0.071).

These equal the numbers in q45 and in `closeout_A3/WHERE_VIEWS.md` (table 2) of the onion worktree. The difference on the oracle scale, 0.359, is beyond 2x (2 x 0.095 = 0.190).

### 2. With alignment-level views, the U-Net becomes a copier on these pairs (m = d and s = f)

| U-Net arm | m - d, source nodedup | m - d, target nodedup | s - f, source nodedup | s - f, target nodedup |
|---|---|---|---|---|
| all views (`rung4_base_heldout`) | -0.072 (0.012) | -0.046 (0.016) | -0.073 (0.008) | -0.021 (0.020) |
| base view only | -0.075 (0.030) | -0.089 (0.011) | -0.079 (0.018) | -0.073 (0.014) |
| base + depth thinning | -0.072 (0.044) | -0.087 (0.010) | -0.074 (0.023) | -0.074 (0.016) |
| base + read-level (2 seeds) | -0.064 (0.028) | -0.084 (0.001) | -0.074 (0.010) | -0.064 (0.006) |
| base + alignment-level | -0.007 (0.027) | -0.001 (0.012) | -0.009 (0.029), 2 seeds | +0.011 (0.002), 2 seeds |

- On the cross-view pairs, the alignment arm scores as depth-only scaling does (m - d near 0 in both directions). Every arm without alignment views beats depth-only by 0.064 to 0.089. Against base only, the shift is +0.068 (source nodedup; 2 x 0.030 = 0.060, beyond 2x) and +0.088 (target nodedup; beyond 2x).
- On the same-view pass of the same targets, the alignment arm scores as the target's own pr1 half does (s - f near 0). The other arms beat that half by 0.064 to 0.079. Against base only the shift is +0.070 (2 x 0.029 = 0.058, beyond 2x) and +0.084 (beyond 2x).
- So the model outputs a scaled copy of its input track whenever the target is a `mapq0` or `nodedup` view. Two independent measurements (cross-view and same-view) show it.

The copy is gated by the covariates. On the crop pairs, the same alignment-arm seeds still denoise: s = 0.580 against f = 0.707, and crop gap closed is 0.518 (0.017) against 0.532 (0.056) for base only (within). The all-views arm, which also trains on alignment views but at a lower share, sits between the two (gap closed 0.272 (0.049); it copies partly on the target-nodedup direction, s - f = -0.021). Training rows by view kind are base 678, depth 936, mapq 308, nodedup 312, se 38, crop 56 (`closeout A3` brief, section 4). So the alignment views are 620 / (678 + 620) = 48% of the rows of the alignment arm and 620 / 2,328 = 27% of the rows of the all-views arm. This is a dose-response pattern, from 3 arms only.

### 3. The model, not shared fragments in the metric, causes the drop

- If source and target halves shared fragments so that copying won at scoring, depth-only scaling (a pure scaled copy) would come close to O. It does not: d - O = 0.191 (source nodedup) and 0.244 (target nodedup), against 0.253 on crop pairs (d = 0.7229, O = 0.4697). The earlier review reached the same result (`correction_metric_review/REVIEW.md`, suspect (a)).
- d, f and O do not depend on the model. They are the same for every arm, so they cannot make one arm lower than another. Only m and s move, and both move toward a copy.
- The copy is the training optimum. The training sampler reads full views (`counts`) only, never the pr1 or pr2 halves (`tools/minicandi/dataset.py`, docstring of `train_views`). A `nodedup` or `mapq0` view differs from the base view by a few percent of the reads (log2 source/target total = 0.054 on these pairs, `REVIEW.md` section 2). So, in training, the best prediction of such a target is the input track itself with a tight NB. At scoring, source and target are disjoint halves, and the copy carries the source half's sampling noise. That gives m = d.

### 4. The U-Net copies fully; the conv models move the same way by much less

The relative same-view copy gain (`scores.json` `copy_leak.same_view_copy_gain_relative`, `p16_sv`, per seed):

| model | base only | base + alignment-level | all views |
|---|---|---|---|
| U-Net | 0.580, 0.596, 0.599 | 0.658, 0.653 | 0.634, 0.654, 0.652 |
| conv autoencoder | 0.045, 0.040, 0.035 | 0.080, 0.081, 0.088 | 0.043, 0.039, 0.041 |
| conv + attention | 0.056, 0.049, 0.051 | 0.095, 0.089, 0.088 | 0.053, 0.055, 0.055 |

Alignment views raise the copy gain on all three models (each beyond its seed range). The U-Net starts with a copy gain about 13 times that of the conv models. It has a full-resolution path from each track's input features to that track's head (`rungs/unet.py`, `head_skip`, plan fact 3); the conv models have none.

Gap closed on the 84 mapq + nodedup pairs, oracle scale, `p16_sv`:

| model | base only | base + alignment-level | difference |
|---|---|---|---|
| U-Net (`p16`, 3 seeds each) | 0.377 (0.095) | 0.018 (0.071) | -0.359, beyond 2x |
| conv autoencoder | 0.523 (0.018) | 0.461 (0.019) | -0.062, beyond 2x |
| conv + attention | 0.519 (0.043) | 0.439 (0.012) | -0.080, beyond 1x, not 2x |

On the conv models, m - d moves from -0.122 to -0.115 (source nodedup) and from -0.105 to -0.086 (target nodedup) on the conv autoencoder. They stay far from a copy. But the conv models pay elsewhere: the conv autoencoder's crop-pair gap closed falls from 0.740 (0.019) to 0.594 (0.036), beyond 2x. This loss is inside the version-3 correction term (`ABLATIONS_READ.md`, section 2.14: correction 0.615, -0.102).

### 5. The correction-only term does not see the drop

On the mapq + nodedup pairs, the correction-only term, 1 - (m - s) / (d - f), is 0.132 (0.100) for U-Net base only and 0.165 (0.021, 2 seeds) for the alignment arm: within. Both m and s rise by about the same amount, so the model's excess over its own same-view score does not change. The loss is a loss of denoising on these targets, not a loss of covariate correction.

## Answer

The alignment-level views teach the model to copy its input when the target is a no-dedup or MAPQ-0 view, because in training these views are full files that share about 96% of their reads with the source, so a sharp copy minimizes the training loss; at scoring the source and target are disjoint pseudoreplicate halves, so the copy carries the source half's sampling noise and scores exactly as depth-only scaling does. The evidence is that the U-Net alignment arm has m - d = -0.007 and -0.001 and s - f = -0.009 and +0.011 on these pairs, against about -0.08 for every arm without alignment views (beyond 2x the seed range), while it still denoises crop targets; the U-Net copies fully because its head skip is a strong copy path (copy gain 0.65 against 0.04 to 0.09 for the conv models), and the shared-fragment explanation is ruled out because d, f and O are model-independent and d stays 0.19 to 0.24 above O.

## What this means for the recipe

1. Do not train on the alignment-level views as full-view pairs. This holds for all three models: they raise the copy gain on all three, collapse the U-Net on mapq + nodedup pairs, and cost the conv autoencoder 0.146 crop-pair gap closed, which is inside the D1 correction term. This agrees with the ablation decision "keep depth views; no other view type earns a place" (`ABLATIONS_READ.md`, section 2.14).
2. If a later arm needs alignment views, draw the cross-view training pairs from the halves (source pr1, target pr2), so that training matches scoring and a copy is no longer optimal. That is a sampler change in `tools/minicandi/dataset.py`, not a metric change. I intend this as a proposal; no run tests it.
3. The version-3 correction term already excludes the mapq + nodedup pairs (`correction_gap_closed_oracle_no_nodedup_side`), so the D1 screen will not show this failure on the U-Net. Report the mapq + nodedup column and the copy gain beside every view or skip arm, so that a copier is visible.
4. The U-Net head skip is the copy path here. A skip-family arm (chunk C7) that conditions or weakens `head_skip` should lower the copy gain; the copy gain is a cheap check to read for those arms.

### What I did not check

- The 256-region confirm tier: the alignment and depth arms have no `p256_sv` score (only the base-only arms do). All values here are from the 16-region screen.
- Whether pr1 and pr2 halves of different views share reads or duplicate copies. That is a property of the table builders and the BAMs.
- Per-bin predictions. I read only per-pair CRPS sums, predicted totals and the copy-leak keys. The copy is inferred from m = d and s = f, not seen bin by bin.
- Whether the training sampler draws mapq0-to-nodedup pairs in training cells. The eval pairs are labeled never trained by the scorer.
- The U-Net alignment and read-level arms each lack one seed in `p16_sv`, and the dose-response pattern rests on 3 arms.

## Appendix: recompute script

Run on Fir from a login node with `PYTHONPATH=$K/src:$K/tools`, `K=/project/def-maxwl/mforooz/onion_kit/CM`, interpreter `/project/6014832/mforooz/EpiDenoise/candi_venv/bin/python`, arguments `<sub> <comma-separated arms>`. It reads only.

```python
import json, sys, numpy as np
from pathlib import Path
from onion_eval import oracle_floor as OF
R = Path('/project/def-maxwl/mforooz/ONION/runs_fir')
arms = sys.argv[2].split(','); sub = sys.argv[1]
def grp(knob, src, tgt):
    if knob != 'mapq+nodedup': return knob
    return 'nd_src' if src.endswith('.nodedup') else ('nd_tgt' if tgt.endswith('.nodedup') else 'other')
for a in arms:
    for k in (0, 1, 2):
        d = R/a/f'seed{k}'/sub; sp = d/'scores.json'
        if not sp.is_file(): print(a, k, 'MISSING'); continue
        s = json.loads(sp.read_text()); o = OF.load_oracle(OF.find_oracle(sp, s))
        recs, why = OF.pair_records(d, o)
        z = np.load(d/'pairs_per_region.npz', allow_pickle=True)
        st = {str(p): (str(z['source'][i]), str(z['target'][i])) for i, p in enumerate(z['pair_ids'])}
        sv = {}
        if (d/'same_view_per_region.npz').is_file():
            y = np.load(d/'same_view_per_region.npz', allow_pickle=True)
            sv = {str(t): float(v) for t, v in zip(y['target'], y['crps_model_same_view'])}
        G = {}
        for r in recs:
            if not (r['never_trained'] == 1 and r['subset_pair'] == 0): continue
            src, tgt = st[r['pair_id']]
            G.setdefault(grp(r['knob'], src, tgt), []).append(
                (r['crps_model'], sv.get(tgt, np.nan), r['crps_depth_only'], r['crps_floor'], r['oracle']))
        out = []
        for g in ('crop', 'crop+se', 'nd_src', 'nd_tgt'):
            if g not in G: continue
            m, sv_, dd, f, O = np.nanmean(np.array(G[g]), 0)
            out.append(f"{g}: m={m:.4f} s={sv_:.4f} d={dd:.4f} f={f:.4f} O={O:.4f} "
                       f"gcO={(dd-m)/(dd-O):.3f} co={1-(m-sv_)/(dd-f):.3f}")
        print(a, k, ' | '.join(out))
```

The pooled 84-pair values average the two directions with equal weight (42 pairs each).
