# t136 — the arm-to-arm training grid for h28 (a Decision for the PI)

Status: DRAFT, 2026-10-29. Nothing here is approved, ticked or launched. The PI approves the null
(`crux approve-null h28`) and sets every `TODO(PI)` threshold. The checks then go into h28's
`## Verifiables` in a later step. The grid runs on Fir only after that.

Node: `cruxvault/h28_training_g_on_arm_to_arm_pairs_lets_a_di.md` (parent q7). Task: t136.
Code that this design must match: chunk G6 (`tools/t136/pairs.py`, `tools/t136/train.py`,
`slurm/t136/{train,law}.sh`). Every number below comes from a file named beside it. "Spread" is
the seed spread, max − min over 3 seeds of the per-seed value (the t128p2 rule).

## 1. The claim and the null draft

Claim (h28): if g also trains on some arm → arm pairs, a direct g → f map meets the depth law on
arm → arm pairs it never trained on, with no pivot through the base.

Null draft (one line, family first; the PI approves or sharpens it):

> Leakage: held-out arm→arm depth ratios also occur among trained arm→arm pairs, so g recalls seen ratios, not a pair-general depth law.

Why this family. The cheapest way for a positive result to appear while the claim is false is
that the held-out pairs are near copies of trained pairs. The depth law depends only on the
depth ratio. If a held-out pair has a depth ratio that g saw on a trained arm → arm pair, g can
reuse that answer. The grid below holds one arm (Darm1) where this cannot happen and one arm
(Darm2) where it can for the histone tracks. h23 states a near relative of this null under the
Selection family (interpolation between trained ratios). The PI can pick either label.

## 2. The ratios — what is seen and what is not

The depth arms are 15M, 7.5M and 3.75M reads. The base is 30M reads for the 6 histone tracks and
50M reads for DNase C12M02 (`tests/fixtures/t112_meta/MANIFEST.tsv`, column `depth`). Every
other knob arm has the base depth.

| pair | histone ratio | DNase ratio | Dref | Darm1 | Darm2 |
|---|---|---|---|---|---|
| base ↔ 15M | 2 | 3.33 | trained | trained | trained |
| base ↔ 3.75M | 8 | 13.3 | trained | trained | trained |
| base ↔ 7.5M | 4 | 6.67 | held out | held out | held out |
| 15M ↔ 3.75M | 4 | 4 | never trained | **trained** (arm → arm) | **trained** |
| 7.5M ↔ 15M, 7.5M ↔ 3.75M | 2 | 2 | never trained | held out | held out |
| knob arm ↔ 15M | 2 | 3.33 | never trained | never trained | **trained** |
| knob arm ↔ 3.75M | 8 | 13.3 | never trained | never trained | **trained** |
| knob arm ↔ 7.5M | 4 | 6.67 | never trained | held out | held out |

Consequences:
- In Darm1, ratio 2 never occurs on a trained arm → arm pair. The only trained arm → arm ratio is
  4. Ratio 2 occurs only on the trained base ↔ 15M pair of the histone tracks, as in Dref.
- In Darm2, ratio 2 occurs on trained arm → arm pairs (knob arm ↔ 15M) for the 6 histone tracks,
  but NOT for DNase, whose knob arms sit at 50M.
- So the null predicts: Darm1 fits its trained ratio-4 pair and misses the held-out ratio-2
  pairs; Darm2 does better than Darm1 on ratio 2 for histone tracks only.

## 3. The arms

All arms use the phase-2 old D recipe (`tools/t128p3/pairs.py` `_s3b(…, "D", …)`): per-track g,
g hidden 64, f 32 channels, n_theta 256, loss NB, `max_steps` 12 000, the t118 cross-NLL
validation rule, space `counts`, holdout `(("depth", "7.5M"),)`, seeds 0, 1, 2. The across-track
g is dropped (PI 2026-10-29).

| arm | training pairs per track | runs |
|---|---|---|
| `Dref` | the base ↔ arm pairs minus the 7.5M pairs | exist, not rerun: Fir `/project/def-maxwl/mforooz/t128p2/runs/D_<g>_counts_real_s{0,1,2}` |
| `Dviabase` | none (Dref's two trained maps in series, arm → base → arm) | exist, not rerun: Fir `/project/def-maxwl/mforooz/t128p2/viabase/D_<g>_counts_real_s{0,1,2}` |
| `Darm1` | Dref's pairs + the 2 ordered pairs 15M → 3.75M, 3.75M → 15M | 21 new |
| `Darm2` | Dref's pairs + every ordered arm → arm pair whose source and target are both not held out | 21 new |

New runs: 2 arms × 7 tracks × 3 seeds = **42**. Task index `arm_i*21 + g_i*3 + seed`;
`run_name = f"{arm}_{g}_counts_real_s{seed}"`; arrays `Darm1` 0-20, `Darm2` 21-41; smoke rows =
the C19M16 seed-0 rows of each arm. Output root Fir `/project/def-maxwl/mforooz/t136`.

## 4. The pair tables (what G6 emits)

Computed on 2026-10-29 with `ladder.pairs` on `tests/fixtures/t112_meta/MANIFEST.tsv`
(`usable_products` drops `C12M02__mapq__0`, `C12M02__mapq__10`). G6's test asserts the same
numbers from the code; if the two differ, the code is right and this table is corrected.

| track | class | usable arms | base pairs after holdout | Darm1 extra | Darm2 extra | held-out arm → arm (touch 7.5M) | of which depth ↔ depth |
|---|---|---|---|---|---|---|---|
| C07M20, C19M16, C19M22 | narrow | 19 | 36 | 2 | 306 | 36 | 4 |
| C07M29, C40M17, C40M18 | broad | 19 | 36 | 2 | 306 | 36 | 4 |
| C12M02 | DNase | 7 | 12 | 2 | **30** | 12 | 4 |
| sum over 7 tracks | | | 228 | 14 | 1 866 | 228 | 28 |

Correction to the brief: DNase has 7 usable arms (6 not held out), so Darm2 adds 6 × 5 = 30
pairs, not 56. A histone track has 18 arms not held out, so 18 × 17 = 306.

- `extra_pairs(rows, g, "Darm1")`: the two `law_pairs(rows, g)` rows with `(arm, level)` of source
  and target = `("depth","15M")` and `("depth","3.75M")`, in `law_pairs` order.
- `extra_pairs(rows, g, "Darm2")`: every `law_pairs(rows, g)` row with `_held(p, HOLDOUT)` false.
- `train_pairs(rows, g, arm) = ladder.pairs.train_pairs(rows, g, HOLDOUT) + extra_pairs(rows, g, arm)`.
- `heldout_score_pairs(rows, g)`: every `law_pairs(rows, g)` row with the 7.5M product as source or
  target (36 per histone track, 12 for DNase). The base ↔ 7.5M pairs come from `score_trained`'s
  `heldout` records (2 per track).
- `config["extra_pairs"]` is written on every run (an empty list on every old path), so the
  readings can move Darm1's 15M ↔ 3.75M pair from the "never" group to "trained".

Note on the mix. Darm2's training set is 342 pairs per histone track, of which only 36 are base
pairs. If training samples pairs uniformly, base pairs drop from 100 % to about 11 % of the draws.
Check c2 (below) exists to catch a loss of the base fit from this.

## 5. Score sets and metrics

Every reading is per-track g, chr19 + chr21, pooled by the t118 rule (per seed: mean over a
track's pairs, then mean over tracks; the reading is the mean over seeds). Every CRPS has
`crps_oracle_scaled_all` and `scale_error_all` beside it.

- **Depth law** (the `depthlaw` records of `score.py law --split`): median |log2 error| of the
  predicted scale, by group. The groups follow `tools/t128p2/readings/analyze.py::m5_depthlaw`,
  with one change: Darm1 and Darm2 move 15M ↔ 3.75M from `never` to `trained` (read from
  `config["extra_pairs"]`).
  - `heldout_r2` = the 4 ordered depth pairs that touch 7.5M (7.5M ↔ 15M, 7.5M ↔ 3.75M; ratio 2).
    This is t128p2's `never75` group.
  - `trained_r4` = 15M ↔ 3.75M (ratio 4). Trained in Darm1 and Darm2; `never` in Dref.
  - `trained_base` = base ↔ 15M, base ↔ 3.75M. `heldout_base` = base ↔ 7.5M.
- **CRPS** on the held-out arm → arm pairs (every law pair that touches 7.5M, all knobs), against
  Dref direct and Dviabase on the same pairs.

### Reference values (phase 2, per-track g, 3 seeds)

From `cruxvault/results/t128p2/m5_depthlaw_table.tsv` and `m1_never_table.tsv`,
`m3_trained_table.tsv`, `m9_steps_table.tsv` (main checkout mirror):

| group | Dref direct | Dviabase |
|---|---|---|
| `heldout_r2` (`never75`) median abs log2 err | 0.195 (spread 0.061) | 0.125 (spread 0.022) |
| 15M ↔ 3.75M (`never`) median abs log2 err | 0.291 (spread 0.051) | 0.110 (spread 0.054) |
| `trained_base` median abs log2 err | 0.052 (spread 0.010) | — |
| `heldout_base` median abs log2 err | 0.100 (spread 0.017) | — |
| all never-trained law pairs, CRPS | 0.662 (oracle_scaled 0.606, scale_error 0.057; spread 0.043) | 0.545 (0.537, 0.008; spread 0.008) |
| never-trained pairs touching the depth knob, CRPS | 0.561 (0.499, 0.062; spread 0.044) | 0.456 (0.451, 0.005; spread 0.003) |
| trained base ↔ arm pairs, CRPS | 0.445 (0.441, 0.0035; spread 0.003) | — |

The vault and the brief quote "direct 0.29, via the base 0.12". Those numbers belong to the
15M ↔ 3.75M group, which Darm1 and Darm2 now TRAIN on. The held-out comparator for h28 is the
`never75` row (0.195 against 0.125). The two CRPS rows above pool more pairs than the held-out
set; the readings recompute Dref and Dviabase on exactly `heldout_score_pairs`.

### The headroom problem (read this first)

On `heldout_r2`, Dref direct (0.195, spread 0.061) and Dviabase (0.125, spread 0.022) are already
within noise of each other: the gap 0.070 is below 2 × 0.061 = 0.122. Two consequences:
1. A check of the form "Darm1 ≤ via-base + δ" is also met by Dref when δ ≥ 0.07. Such a check
   cannot tell the claim from "arm → arm training changed nothing".
2. A check that Darm1 beats Dref by more than 2 × spread needs Darm1 below about 0.07 (if its
   spread is near Dref's), which is below the via-base chain itself.

The CRPS on the held-out arm → arm pairs has more headroom (whole never set: gap 0.117 against
2 × 0.043 = 0.087), so the claim-directed CRPS check (c4) carries the discrimination that the
depth-law check (c1) cannot. Options for the PI (Decision **D1**):
- **A (recommended):** run the grid as designed; read c1 as a gap-closed fraction without a noise
  clause, and let c3 and c4 carry the discrimination. Cost: 42 runs, no new reference runs.
- **B:** keep A and add seeds 3, 4 to Dref, Darm1 and Darm2 (35 runs more) to narrow the spread.
  Note: a max − min spread over 5 seeds is not smaller in expectation than over 3.
- **C:** swap what is held out: train the ratio-2 arm → arm pairs and score the never-trained
  15M ↔ 3.75M pair, where Dref has headroom (0.291 against 0.110). This needs a Dref without the
  7.5M holdout (21 new reference runs plus via-base) and changes G6's code. Defer unless A reads
  as uninformative.

## 6. Proposed checks (crux-verifiables form; PI sets every threshold)

Each check names its kind and a world where it fails. `TODO(PI)` marks a threshold the PI sets;
the value after "recommended" is a proposal only.

- **c1 `lawheldout`** [claim-directed] — On `heldout_r2`, the gap closed by Darm1,
  `(Dref − Darm1) / (Dref − Dviabase)` on median |log2 err|, is at least `TODO(PI)`;
  recommended ≥ 0.5 (the h1/h2 gapclosed level, PI ruling 2026-09-16), with no noise clause
  (Decision D1). Fails if: arm → arm training on a ratio-4 pair leaves the ratio-2 held-out pairs
  where Dref left them.
- **c2 `basefit`** [outcome-neutral control] — In Darm1 and in Darm2, the `trained_base`
  median |log2 err| is within `TODO(PI)` of Dref's 0.052, and the trained base ↔ arm CRPS is
  within `TODO(PI)` of Dref's 0.445; recommended: within 2 × the larger seed spread for each.
  Fails if: the extra pairs crowd out the base pairs (Darm2 has about 11 % base draws), so the
  run is not the same model plus arm → arm pairs. A failure invalidates that arm's run; it does
  not refute h28.
- **c3 `ratiotransfer`** [claim-directed, DISCRIMINATES against the null] — In Darm1, the
  `heldout_r2` median |log2 err| (ratio 2, never trained as an arm → arm ratio) exceeds the
  `trained_r4` median |log2 err| (ratio 4, trained) by no more than `TODO(PI)`; recommended
  ≤ 2 × the larger of the two seed spreads. Fails if: g recalls the trained arm → arm ratio and
  does not carry the law to a ratio it has not seen on an arm → arm pair — the null's world.
- **c4 `crpsheldout`** [claim-directed] — On the held-out arm → arm pairs (every law pair that
  touches 7.5M, all knobs), Darm1's pooled `crps_all` closes at least `TODO(PI)` of the
  Dref → Dviabase interval, and the gain `Dref − Darm1` exceeds 2 × the larger seed spread;
  recommended ≥ 0.5, the h1/h2 rule. `crps_oracle_scaled_all` and `scale_error_all` are quoted
  beside it, and the gain must not come only from `scale_error` (`TODO(PI)`; recommended: the
  oracle_scaled part closes ≥ 0.25 of its own interval). Fails if: the depth-law scale moves but
  the predicted distribution on unseen arm → arm pairs does not improve.
- **c5 `runhealth`** [outcome-neutral control] — Every one of the 42 runs finishes; no
  `config["train_pairs"]` row holds a 7.5M pid; `len(config["extra_pairs"])` is 2 (Darm1) and
  306 or 30 (Darm2) per track; the share of runs per arm whose best step is in the last 10 % of
  the 12 000-step cap is at most `TODO(PI)`; recommended ≤ 2 of 21 (Dref: 0 of 21,
  `m9_steps_table.tsv`). Fails if: the harness trained the wrong pairs or a run was cut before it
  converged. A failure invalidates the run.

Combination rule: `all` over the claim-directed checks c1, c3, c4, read on Darm1 only, and only
when the controls c2 and c5 pass (a control failure marks the run invalid, not refuted). Why
`all`: each claim check fails in a world the others pass in — c1 alone passes when only the
scale moves, c4 alone when the distribution improves without the law, c3 alone when the
ratio-4 fit is poor too. Cost of `all`: three checks at 80 % power each give about 51 % joint
power, and the thresholds may not be loosened to make up for it.

Darm2 is read, not checked. The reading: Darm2 − Darm1 on `heldout_r2`, histone tracks against
DNase. Under the null, Darm2 gains on histone tracks (ratio 2 seen arm → arm) and not on DNase
(ratio 2 not seen). This is one DNase track × 3 seeds, so it is a reading, not a check.

## 7. Cost

- Phase-2 D, per-track g, ran a median 3 750 steps at 0.88 s per step: 0.88 h of training per run
  (`m9_steps_table.tsv`; best step 1 500–6 000, 0 of 21 at the cap). The brief's "about 2 h" is
  the per-run upper range, not the median.
- If Darm runs behave like Dref: 42 runs at `%12` on MIG slices ≈ 4 waves × about 1 h ≈ 4 h of
  training, plus `score.py trained` per run and the law array. Worst case (every run to the
  12 000 cap): about 2.9 h per run, about 12 h in total. The scoring time was not measured here.
- No via-base step: the reference via-base runs exist under `t128p2/viabase`.

## 8. What is deferred

- Counterfactual augmentation of CANDI's corpus (re-derived arms from BAM or FASTQ) — t140. It
  waits for this grid and for the t135 pivot seeds (Fir jobs 1211830, 1211840, 1211839, 1211842).
- Decision D1 option C (swap the holdout) — only if option A reads as uninformative.
- The across-track g (PI 2026-10-29).

## 9. What the PI decides

- **D1** — the headroom option: A (recommended), B or C.
- **D2** — the null line in §1: approve, sharpen, or relabel as Selection (as h23 does).
- **D3** — the `TODO(PI)` thresholds of c1–c5, and the combination rule `all` over c1, c3, c4.

## Checks as written (PI, 2026-10-29)

The PI approved the null in §1 as written (`crux approve-null h28`, Leakage family) and accepted
the checks c1–c5 of §6 in chat on 2026-10-29, each at its recommended value. The checks are now
in h28's `## Verifiables` with `rule: all`. §1–§9 above keep the draft history; their threshold
placeholders are resolved here, not edited in place.

Decisions:
- **D1** — option A. The grid runs as designed, with no extra seeds.
- **D2** — the null line in §1 is approved as Leakage.
- **D3** — the thresholds below, and the combination rule `all` over c1, c3, c4.

The resolved thresholds ("spread" is max − min over seeds 0, 1, 2, as in the header):

| check | kind | accepted threshold |
|---|---|---|
| c1 `lawheldout` | claim-directed | (Dref − Darm1) / (Dref − Dviabase) ≥ 0.5 on `heldout_r2` median abs log2 err; no noise clause. With 0.195 and 0.125, Darm1 ≤ 0.160. |
| c2 `basefit` | outcome-neutral | Darm1 and Darm2 each within 2 × the larger seed spread of Dref on `trained_base` median abs log2 err (0.052) and on trained base ↔ arm CRPS (0.445). |
| c3 `ratiotransfer` | claim-directed, discriminates against the null | Darm1 `heldout_r2` ≤ Darm1 `trained_r4` + 2 × the larger of the two seed spreads. |
| c4 `crpsheldout` | claim-directed | Darm1 closes ≥ 0.5 of the Dref → Dviabase CRPS interval on the held-out arm → arm pairs; the gain exceeds 2 × the larger seed spread; the oracle_scaled part closes ≥ 0.25 of its own interval. |
| c5 `runhealth` | outcome-neutral | All 42 runs finish; pair counts are correct (below); at most 2 of 21 runs per arm have their best step above 10 800 of 12 000. |

Combination: `all` over c1, c3, c4, read on Darm1 only, and only when c2 and c5 pass. A control
failure makes the run invalid, not refuted. Joint power is about 51 % (three checks at 80 %).
Darm2 is read, not checked.

Pair counts. Darm2 adds 306 ordered arm → arm pairs per histone track and **30** for DNase
C12M02 (6 arms not held out, 6 × 5), not 56. Darm1 adds 2 per track. §4 and §6 c5 already
state 30; the value 56 appears only in §4's correction note.

Cluster. The grid runs on **Nibi**, not Fir (PI 2026-10-29), at 2 CPUs, `--mem=10000M` and
`--time=7:00:00` per run. The Fir paths in §3 (output root `/project/def-maxwl/mforooz/t136`)
and the MIG-slice estimate in §7 are superseded for the new runs. The Dref and Dviabase
reference runs stay where §3 names them on Fir.
