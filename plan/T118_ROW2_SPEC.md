# Row 2 of the transformation grid: the generator also reads the source signal

Written 2026-09-26 for the PI to decide from. Design authority stays `plan/T118_COUNTERFACTUAL_F.md`;
this file adds one row to its architecture ladder and changes no ruling. Terms: **g** is the generator,
a small network that outputs the parameters **theta** of a transformation **f**; **C, C'** are the
covariate vectors of the source and the target track; **x** is the source track after the log
transform (log(1 + counts), or log of the floored −log10 p); **X'** is the target track.

- **Row 1** (built, 576 runs finished): theta = g(C, C'). One theta per (source, target) pair, so one
  f for the whole genome. Verified in the code: `G` maps a 40-number covariate pair to `[B, n_theta]`,
  every form's `forward(x, theta)` takes `[B, n_theta]`, and `predict_chrom` applies one theta to a
  whole chromosome.
- **Row 2** (this spec): theta_i = g(x_i, C, C'). g also reads the signal, so it outputs one theta per
  genomic bin, and f applies theta_i at bin i. Row 2 is outside the pre-registered ladder.

## 1. The grid

| | A: affine in log space | B: monotone curve, 12 knots | C: 33-bin kernel, then B's curve | D: FiLM-modulated dilated CNN |
|---|---|---|---|---|
| **Row 1** — g reads (C, C') | done: 144 runs, law test, report | done | done | done |
| Row 1 status in −log10 p | no exploding pairs; best row-1 design in p space | explodes on a few upscaling pairs (lowest knot's σ) | explodes (same cause) | explodes (cause not found) |
| **Row 2** — g reads (x, C, C') | needs: `form_a2.py`; g reads x; per-bin theta plumbing | needs: `form_b2.py` | needs: `form_c2.py` (a kernel per bin) | needs: `form_d2.py` (FiLM per bin) |
| Row 2 shares | the same data, pairs, chromosome split, 4000-step budget, early stopping on chr22, seeds 0–2, scoring, law test, both twins, checks and bars | | | |

"Upscaling pair" is the status file's label for a sparse source (low depth, or DNase with dedup off)
mapped to the full-depth base.

## 2. How g reads x, and what each design becomes

**One shared rule for all four designs (recommended; decision (i) and (ii) in section 7).** g keeps
its row-1 shape, an MLP with two hidden layers of 64, and gains exactly one input number: the bin's
own transformed value x_i. Its input is [x_i, C, C'] (41 numbers, was 40). It is applied at every
input position, so its output is `[B, W, n_theta]` where W is the window length plus the form's halo
(row 1: `[B, n_theta]`). The first layer is computed as `x_i · w_0 + (cov · W_rest + b)`, so the
41-wide input is never materialised (design X's trick). g is never given C' − C, as before.

- **Identity at initialisation is kept exactly.** g's last layer has zero weight and bias
  `init_theta`, so theta_i = init_theta at every bin whatever x_i is, and every form's identity map
  applies. The same mechanism as row 1.
- **Parameters:** g gains 64 weights (one input column). f's own parameters are unchanged. Row 2 is
  therefore "row 1's capacity, plus the right to read x".
- **A useful property:** because g reads only the scalar x_i, theta_i is a function of the level x_i
  alone for fixed (C, C'). g can be evaluated once per distinct value of x on a chromosome and
  gathered, which makes prediction, validation and the averaged twin (section 3) cheap and exact.
  Counts have at most a few thousand distinct values per chromosome; −log10 p has more, still far
  below the bin count.
- **Input scale of x_i** is left raw (as design X did): counts x in about 0–10, p-space x in about
  −6.9 to 3. Routine engineering, reported with the run.

### A2 — affine per bin
- theta_i = (a_i, b_i, d_i); loc_i = a_i + b_i · x_i; disp_i = d_i.
- Lost: "one straight line" per pair. Since a, b and d now vary with x_i, x ↦ loc is any smooth
  function of x, and so is x ↦ disp. Kept: per-bin (no neighbours), magnitude only.
- Cost: theta `[B, L, 3]`; negligible.

### B2 — monotone curve per bin
- theta_i = (y0_i, s_i[11], d_i[12]): a full 12-knot curve chosen at each bin and evaluated at x_i.
- Lost: monotonicity in x, and the claim "same function class as QuantileMatching". The curve at bin i
  is monotone, but bins at different levels use different curves, so x ↦ loc(x) need not be. In row 2,
  **A2 and B2 span the same function class** (any smooth map of x); they differ in parameterisation
  (3 numbers per bin against 24) and in how the knot grid shapes the map near the floor. The
  comparison A2 against B2 therefore tests a parameterisation, not a function class. Section 7 (iii)
  offers a monotone variant.
- Cost: theta `[B, L, 24]`; the gather over knots is per bin as before.

### C2 — a different kernel at each bin
- theta_i = (w_i[33], y0_i, s_i[11], d_i[12]). The smoothed value at bin i is
  x_c[i] = Σ_k w_i[k] · x[i + k − 16] (an unfold of x into 33-bin patches, weighted per bin), then B's
  curve with the per-bin knots, evaluated at x_c[i]. g reads the raw centre value x_i, not x_c.
- Lost: "the same kernel at every position". The kernel now depends on the centre bin's level, so the
  model can smooth background and sharpen peaks, which row 1's C cannot. Kept: an unconstrained
  kernel, one-hot at init, so the map starts at the identity.
- Cost: theta `[B, L, 57]` and patches `[B, L, 33]`; at the training defaults (32 windows of 2048)
  about 9 MB per tensor. Prediction at 8 chunks of 65 536 bins: about 70 MB per tensor. Fits a 10 GB
  slice.

### D2 — FiLM per bin
- theta_i = (γ_i[4][32], β_i[4][32]) at every position of the haloed window, since a layer's
  modulation inside the halo reaches the centre outputs through later layers. Each layer computes
  h = GELU(γ_i ⊙ conv(h)_i + β_i). The trunk and the zero-initialised head are unchanged.
- Lost: FiLM as a global per-pair modulation. It becomes an input-dependent gate, so D2 is a CNN
  whose channels are gated by the bin's own level and the covariates. Kept: identity at init through
  the zero head, whatever γ, β and the trunk do.
- Cost: theta `[B, W, 256]`, about 70 MB in training and about 540 MB per prediction chunk batch;
  fits a 10 GB slice. Roughly 1.5× row-1 D's step time is the planning assumption.

## 3. Twins, law test, references, depth law

- **No-covariates twin.** Training is unchanged: at every step each window takes the (C, C') of a
  uniformly random training pair, so the covariates carry no information, but g still reads x.
  Prediction and validation use **one averaged map that still reads x**: theta_bar(x) = the mean over
  the run's training pairs j of g(x, C_j, C'_j). This lifts row 1's rule (one theta = the mean of g's
  theta over the training pairs) bin by bin. It removes exactly the covariate information and nothing
  else. It is exact and cheap through the distinct-value property of section 2 (n_pairs evaluations
  per distinct x, not per bin).
- **Labels-as-ids twin.** Unchanged: one fixed permutation of (C, C') across the training pairs, for
  training and for prediction on trained pairs. g reads its own x and the permuted covariates, so the
  wrong label still identifies the pair, and only the never-trained pairs separate it from the real g.
- **Law test, shuffle, swap.** Unchanged. They only substitute which product's covariates g receives;
  in row 2 the prediction reads the source's x as always.
- **Depth law.** Unchanged: predicted total count scale against the true depth ratio, on depth pairs.
  It is still to be checked by hand (open item from the row-1 handoff).
- **References** noSolution and QuantileMatching: unchanged, shown beside both rows.
- **Checks for a row-2 cell:** beats its own no-covariates twin; law test against both twins; beats the
  row-2 design below (B2 > A2, C2 > B2, D2 > C2); **new:** beats its row-1 cell (A2 > A, and so on),
  gain > 2 × the larger seed wobble, per mark class, space and metric, both versions of g. The main
  claim's rung choice on chr22 stays over row 1 only (decision (vii)).

## 4. Design X

Design X (`form_x.py`, 24 runs on two tracks) put an MLP inside f that reads [x_i, a 32-number pair
embedding from g], with A's slope fixed at 1. In row 2's terms it is A2 with a two-stage g and b ≡ 1:
the same function class as A2. It is kept as a reference column in the two-track tables only; it is
not rerun and does not become a ninth cell (decision (viii)). Its result — no exploding pairs in p
space on two tracks, worse than B–D in DNase counts — is the only evidence so far for section 5.

## 5. Will row 2 remove the p-space explosion? Predictions, per design

The mechanism in B and C (status file): the lowest knot sits at the floor x = log(1e-3), where only
0.1% of bins lie; its σ drifts to about 11 where the data ask for about 1.8; linear interpolation
spreads that σ over the bins between the floor and the median, where the log-normal mean
(median × exp(σ²/2)) makes the CRPS explode. Capping σ and report-only were rejected by the PI.

- **A2 — likely removes it.** Row-1 A never exploded because it has one σ per pair. A2's σ(x) is a
  smooth MLP output of the level; at the floor it is fitted by the floor bins present in every batch
  (about 65 per step), with no knot whose σ can drift unanchored. Design X (same class) had no
  explosion on two tracks. Risk: a smooth σ(x) can still be pulled up just above the floor.
- **B2 — probably reduces it, may not remove it.** The knot grid stays; the lowest knot's σ is still a
  parameter with almost no data at its level, but now it is a function of x_i, so bins just above the
  floor get their own d_0 rather than the floor's. Whether that helps depends on how sharply g can
  change d_0 between x = floor and x slightly above it.
- **C2 — as B2.** The curve part is B2's; the per-bin kernel does not touch the σ mechanism.
- **D2 — unknown.** D has no knots and explodes anyway; its cause is not found. D2 adds a per-bin
  gate, which neither targets nor rules out that cause. Read D2 as a probe, not a fix. Decision (ix)
  asks whether the D diagnosis runs first, in parallel, or not at all.

These are predictions. Only the runs decide, and two tracks of design X are not proof.

## 6. Run scale and compute

Measured on row 1 (status file): per-track train + score about 12 min on a 10 GB slice, cold cache;
across-track law test up to about 2 h on 8 CPU cores; aggregation 2–4 min; design X on 16 CPU cores
4–51 min per per-track run. Not measured separately: across-track training (the first plan estimated
35 min) and the per-track law test (estimated 10–25 min). Planning assumption for row 2: 1.5× row-1
time per task; the smoke below replaces it with a measurement.

| option | train tasks | slice-hours | law tasks | CPU job-hours (8 cores) | wall at 40 concurrent |
|---|---|---|---|---|---|
| smoke: A2–D2, H3K27ac track, counts, real g, seed 0 | 4 | about 1.5 | 4 | about 2 | about 1 h |
| pilot P3: two tracks (H3K27ac, DNase), seed 0, real + no-covariates twin | 32 | about 10 | 32 | about 13 | about 1 h + 1 h |
| pilot P2: two tracks, everything else full | 144 | about 43 | 144 | about 60 | about 2 h + 2 h |
| full match to row 1: 4 designs × 8 g's × 2 spaces × 3 models × 3 seeds | 576 | about 215 (504 per-track × 18 min + 72 across × 53 min) | 576 | about 430 (504 × 25 min + 72 × 3 h) | about 5.5 h + 11 h |

Then aggregation: one CPU job per design plus the grid, run one after another, minutes each.
CPU fallback if the 10 GB slice nodes are held (they were on 2026-09-25): per-track runs at about
30 min on 16 cores from the design X measurement, across-track unknown (assume 2 h): about 6300
core-hours for the full match, 10–14 h wall at 40 concurrent 16-core jobs. The 109 GB cache is reused;
nothing new is cached. Outputs go under `/project/def-maxwl/mforooz/t118/row2/`; the 576 row-1 runs
are read, never rerun. All jobs on Nibi; GPU jobs use only the 10 GB MIG slice.

## 7. Decisions for the PI

Each entry: the choice, the options with cost and what each teaches, and a recommendation.

Decided by the PI in chat on 2026-09-26: adopt the recommendation in every entry, (i)–(x).

**(i) What g reads of x.**
(a) The bin's own value x_i only. Cost: none beyond section 2. Teaches: whether a level-dependent
transformation helps, with the columns keeping their meaning (A/B per-bin, C/D see neighbours).
(b) x_i plus one fixed local summary (the mean of ±16 bins), two inputs. Cost: a fixed pooling; small.
Teaches: whether context in g helps, but gives A2/B2 a view of neighbours, so the columns blur.
(c) A learned window encoder (g becomes a small CNN). Cost: g's size and step time grow; the design
question "where does shape live, in g or in f" is no longer separable. Teaches: the most, but not
cell by cell. Recommendation: (a).

**(ii) One shared reads-x rule, or per form.**
(a) One rule (section 2) for all four. Teaches a clean 2 × 4 comparison. (b) Per form (for example
C2's g reads the smoothed x_c, D2's g reads a hidden layer). Costs a design choice per cell and makes
the rows unequal. Recommendation: (a).

**(iii) Monotonicity in B2.**
(a) The literal lift; monotonicity lost; A2 and B2 are one function class (section 2). Cost: none.
Teaches: parameterisation and floor behaviour, not function class.
(b) x reaches only the dispersion knots: loc is row 1's monotone curve, σ(x) becomes a per-bin
function of the level. Cost: none. Teaches: whether "σ as a smooth function of level" alone stops the
explosion, which is one of the candidate fixes on record; but this B2 is not "g reads x" in full.
(c) Both variants (adds 144 runs, about 55 slice-hours plus law). Recommendation: (a) for the grid,
and (b) as a named follow-up if A2 removes the explosion and B2 does not.

**(iv) The no-covariates twin's averaged map.**
(a) Exact: the mean over training pairs of the per-bin theta at each level (section 3). Cost: n_pairs
evaluations per distinct x value; small through the distinct-value property. Teaches the same thing
as row 1's twin. (b) Feed g the mean covariate vector of the training pairs. Cheaper, but it is the
map of an average covariate, not the average map, and it does not match row 1's definition.
Recommendation: (a).

**(v) Scale.**
(a) Smoke (4 runs, measures time and memory and checks the row-1 re-score is bit-exact), then the
full 576 with no stop, A2 and B2 first, as row 1 was run. (b) Pilot P2 first, then decide. Teaches
the explosion answer on the two tracks that had it about 4 h earlier; delays the grid by a decision.
(c) Pilot P3 first: cheapest signal on the explosion, no twins' seed wobble. Recommendation: (a); if
the slice nodes are held, (c) on CPU first.

**(vi) Status in the notebook.**
(a) Pre-register: four hypotheses (one per row-2 cell) and one grid hypothesis, with null and checks
through the usual agents, before compute is spent. Cost: a session with the PI. (b) Exploratory:
report only, like design X. Recommendation: (a).

**(vii) The main claim's rung choice.**
(a) Unchanged: chosen over row 1 on chr22; a row-2 choice is reported beside it. (b) Chosen over all
eight cells. Recommendation: (a), because row 2 is not pre-registered and the row-1 result stands.

**(viii) Design X.**
(a) Keep its 24 runs as a reference in the two-track tables; no new X runs. (b) Rerun X at full scale
as a ninth cell (144 runs, about 55 slice-hours). (c) Drop it. Recommendation: (a).

**(ix) The row-1 D explosion diagnosis.**
(a) Run it in parallel on CPU from the finished D checkpoints, outside this build. (b) Finish it
before any row-2 compute; delays the grid. (c) Skip; read D2 as the probe. Recommendation: (a).

**(x) The bar for "row 2 beats its row-1 cell".**
(a) The standing gain bar: gain > 2 × the larger seed wobble, per mark class, space and metric, both
versions of g. (b) Report only, no bar. Recommendation: (a).
