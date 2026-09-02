# HANDOFF — finish `plan/BENCHMARK_DESIGN.md` end to end, and ship the leaderboard

**Written 2026-09-01** from a full read of `plan/BENCHMARK_DESIGN.md` (1,461 lines), `origin/main`
at `127ce7c`, the t77–t90 task files, and the live state of Fir and Nibi.

**Mission.** Execute every remaining item of `plan/BENCHMARK_DESIGN.md`, close every open task in
the taskhub, and end with the public leaderboard page rebuilt and deployed, carrying the new
address fields (method · regime · truth · panel · scope · metric) and real scores in every cell the
design says should hold one.

**This is a multi-day programme, not a single session.** It contains ~14 GPU training runs, ~36
prediction runs and ~104 scoring passes (≈4,400 CPU-h) on a shared SLURM cluster. Section 0.4 below
defines the durable state file and the resume contract. Treat every wake-up as "read the state
file, advance what is ready, write the state file".

---

## 0. Before anything — verify, do not assume

Every line in this handoff was true on 2026-09-01. **Re-check each of these yourself.** A stale
premise here costs days.

### 0.1 Local

```bash
cd /Users/mforooz/Desktop/research/libbrechteam@sfu/CANDII
git fetch origin && git log --oneline origin/main -1     # was 127ce7c
git branch -r --format='%(refname:short)' | while read b; do
  n=$(git rev-list --count origin/main..$b 2>/dev/null); [ "${n:-0}" -gt 0 ] && echo "$n $b"; done
```

Python: **do not** `conda activate` (it fails in an agent shell). Name the interpreter:

```bash
/Users/mforooz/miniforge3/envs/candii/bin/python -m pytest tests/ -q
```

Golden gate takes a mode and a path — bare `python tools/golden.py` is an IndexError:

```bash
/Users/mforooz/miniforge3/envs/candii/bin/python tools/golden.py save <scratchpad>/base.pt
/Users/mforooz/miniforge3/envs/candii/bin/python tools/golden.py check <scratchpad>/base.pt
```

Record that baseline **once at session start**, off clean `origin/main`, and check every code
commit against that same file. Recordings are gitignored and machine-local.

crux runs from **inside** `cruxvault/`, and is not on PATH:

```bash
cd cruxvault && /Users/mforooz/miniforge3/envs/candii/bin/python ~/.claude/skills/crux/scaffold/crux.py task list
```

`cruxvault/TASKHUB.md` is generated: take the union of `cruxvault/tasks/` on a merge, then
regenerate with `engine.refresh(Path("cruxvault"))`.

In **any worktree**, `export PYTHONPATH=$PWD/src` or pytest silently tests the main checkout.

### 0.2 Cluster

```bash
hpc status                 # fir up / nibi up; exit 69 means the tunnel is down — ask the PI to run `hpc up fir`
hpc run fir 'squeue -u $USER'
```

Never authenticate on the PI's behalf. Raw `ssh`/`rsync` needs `-o BatchMode=yes`; for rsync that
is `-e 'ssh -o BatchMode=yes'` (rsync's own `-o` is `--owner`).

### 0.3 THE FIRST BLOCKER — the Fir checkout is stale

`/project/def-maxwl/mforooz/CANDII_t78_code` sits at **`7b3de6a`** on branch
`implementation/t77-benchmark-design`. `origin/main` has since merged that PR **and** ~30 further
commits: the four rivals' `V_` selection loops, the ChromImpute sampler move, Lavawizard's
transferable stage, and the 450-window selection scope. **Any run launched from that checkout runs
the old code.** Bring it to `origin/main` and record the sha before the first launch. Every launcher
banner echoes its git sha — read it in the log and confirm it matches.

Other checkouts on Fir (`CANDII_t50`, `t51`, `t52`, `t53`, `t49`, `t62`, `t77`, `t78`, `t80`) are
dead per-task copies. Two venvs exist — `/project/def-maxwl/mforooz/candi_venv` and
`/project/def-maxwl/mforooz/EpiDenoise/candi_venv`; the t81 launcher uses the **second**. Pick one
and pin it everywhere. Cap parallel torch imports off the shared venv at `%12` — above that they
fail with partial-module ImportErrors.

### 0.4 The state file and the resume contract

Create and maintain `cruxvault/results/t81/PROGRAMME_STATE.md` (gitignored dir — also push a copy
of the *table* into a tracked memo when a phase closes). It carries one row per unit:

| unit | phase | slurm job | state | artifact path | number | date |

States: `blocked` · `queued` · `running` · `done` · `failed`. **Write it before you launch and
after every check.** On any resume: read it first, then `squeue`, then reconcile. Do not re-derive
the plan from scratch each session.

---

## 1. What is already DONE — do not redo any of this

- **t78 — the DNase units defect.** Closed and **promoted into the live store**. Adopted layer is
  MACS2 at base resolution on adapter-filtered, duplicate-kept alignments; median Pearson **0.9851**
  against the 2019 challenge over all 34 `T_`. Live store manifest sha is
  `c9a95e4e424d94496be7197ad4aa3d08cd9d7d31144bcf72f53ca50505a2fd83` — quote this, never
  `6c0e0c3e…`. Verified 2026-09-01: 40/40 `signal_rdns.h5` archives and 40/40
  `counts.h5.pre_t78p3` backups present under
  `/project/def-maxwl/mforooz/CANDI_STORE/eic/biosamples/*/`. **Rollback stays open** — §25 Step 7
  does not run until the retrains land.
- **t79 — regimes.** `configs/regime.eic_19.json`, `configs/regime.eic_pilot.json`; hg38 Pilot
  Regions at `configs/regions/encode_pilot_hg38.bed` (44 in, 44 out); BED-restricted sampler landed
  for CANDI and for all four rivals.
- **t80 — eval stack.** Three-number `V_` split, `held-out`/`genome-wide` split with the blanking
  rule, challenge ranker (`src/candi/bench/ranking.py`) as the only ranker. Pairing fixed:
  `V_` = 45 experiments, `B_` = 51, zero prompt-holds-target leaks.
- **t82 — board.** New address, truth toggle, anchor block, markers. Board is **locked**: both
  regimes' `frozen` hashes are `TODO-…`, so `add` refuses every row. See §7.1.
- **t83 — prediction writer.** `src/candi/bench/dump.py` exists, deflates, and its manifest names
  the checkpoint that wrote the root.
- **t84** Lavawizard vendored. **t85** `--early-stop-epochs` (default 3).
- **Rival launcher repair.** All four target the live regimes, select on `V_`, honour a BED.
  ChromImpute's `GenerateTrainData` moved off the eval chromosomes. Lavawizard gained a transferable
  stage. Avocado's joint fit moved to chr19; genome-factor fits 23 → 3.
- **Selection scope.** `configs/regions/eval_random450_seed890217.bed`, 450 seeded windows,
  5.334 % of the eval bins. A mid-training check now costs ~¼ epoch instead of 91 min.
- **CANDI on `eic.pilot` — the one finished retrain.** Job 57674899_1, early-stopped at epoch 11,
  **selected epoch 5, impute macro CRPS 0.5386** (n = 45 tracks, chr20+21+22, full coverage).
  Checkpoint: `/project/def-maxwl/mforooz/t81_checkpoints/t81_eic_pilot_s0.best.ckpt`.
  **Do not retrain it.** Re-verify the checkpoint's md5 before you use it.

---

## 2. Decisions to take on day one

The PI has authorised finishing the plan. These five still need an explicit choice recorded in the
state file before the runs they gate. Recommendations given; take them unless you find evidence
against.

| # | question | recommendation |
|---|---|---|
| **D1** | `marginal`, `knn1`, `knn5` are **regime-dependent** — their fit reads `panel.train_chroms` (`competitors/baselines/generate.py:207-241, 262-300`), so §12.2's "one run each" collapse is wrong for 3 of 5. | Run those three **twice**, `avg` and `avg-arcsinh` once. Programme becomes **18** method-regime units, not 15. Correct §12.2/§12.3 and add the identity assertion §12.2 claims exists for `avg` (it does not — searched). |
| **D2** | The **training-residual σ pass does not exist anywhere in the tree**; all four rivals' `score.sh` refuse rather than write a void table. | **Build it** (Phase 1, item A). Without it the pval arm — §7's head-to-head — has no rival entries at all. |
| **D3** | §12.5 item 2: does the truth toggle apply to `V_` as well as `B_`? | **No.** Keep it excluded, as §12.4 has it. It is +20 passes / ≈1,000 CPU-h and answers nothing §6 needs. Record the exclusion on the board. |
| **D4** | `eic.19` walltime and epoch budget. | Band that fits **25 epochs at ~57 min/epoch plus 9 checks at the 450-window cost**, with `--early-stop-epochs 3`. The pilot run stopped itself at 11, so a 16 h band is very likely enough — but size it from a fresh measured rate, not from `FC_FACTOR=0.45`. |
| **D5** | t78 §25 Step 7 — delete the 40 pre-promotion backups? | **No, not yet.** Keep them until every retrain has landed and been scored. Then ask the PI. |

`crux task accept` is the PI's signature — **never run it on your own judgement**, whatever else
this handoff authorises. `crux task done` is yours to run when the output resolves.

---

## 3. Phase 1 — code that blocks the runs

Six items. Each is ordinary `implementation` work: **its own branch off fresh `origin/main`, pushed
at creation, draft PR at the same moment**, and it must hold the gate — `pytest tests/ -q` green and
`tools/golden.py check` **0 ULP**. These are independent of each other and can run in parallel.

```bash
git fetch origin && git switch -c implementation/<taskid>-<slug> origin/main
git push -u origin implementation/<taskid>-<slug>
gh pr create --draft --fill
```

Open a crux task for each first (`crux task add … -c implementation --why …`). A branch is one whole
issue; children land on their parent's branch and get no PR of their own.

**A. The training-residual σ pass. (new task — the biggest item, and it gates Phase 4 scoring.)**
§7 rules "σ is fit on training residuals only — never on `V_`, never on `B_`". Today
`competitors/{avocado,edice,lavawizard}/fit_sigma.py` walk `stream_truth` over the **declared eval
pairs**, and `competitors/chromimpute/fit_sigma.py` squares the `V_` scores json. All four are void
under Rule 1 and their `score.sh` already refuse. Build one pass that predicts on **training**
tracks and fits σ² per assay from those residuals, and repoint all four (plus `avg-arcsinh`, which
is pval point-only and needs a σ). Keep the `fitted_on` provenance string — it is the only thing
that tells a leak-free table from a leaky one afterwards.

**B. Rival predict-stage `B_` guard.** Both live regimes declare **38** `eval_pairs` (26 `V_` +
12 `B_`). Every rival's predict stage walks all 38, so pointing one at the shipped regime spends
§5's once-only `B_` touch inside the development loop. `slurm/t81_train_candi.sh`'s `SELECT_ON=V`
derivation (lines ~185–232) is the reference fix — give every rival predict stage the same, plus a
separate, deliberate, once-only `B_` predict verb.

**C. CANDI launcher does not pass `--eval-regions`.** `src/candi/train.py` supports it (lines
1382–1494, and it runs **two** monitors — a narrow one for selection, a full one for the final
number). `slurm/t81_train_candi.sh` never passes it. Wire it to
`configs/regions/eval_random450_seed890217.bed` and assert the recorded `eval_scope.sha256`.

**D. Baselines (D1).** Correct the collapse to 2 + 3×2 = 8 units and add the missing identity
assertion for `avg` / `avg-arcsinh`.

**E. `t87`** — two same-named pairing tools collided in a merge: `tools/declare_eval_pairs.py` and
`tools/split_regime_by_panel.py` (whose own docstring still calls itself `declare_eval_pairs.py`).
Reconcile to one.

**F. `t88`** — a shipped-regime test reads gitignored `cruxvault/results/`, so it fails in every
fresh clone and every worktree. Fix the test, not the gitignore.

Also open but not blocking: **`t89`** (architecture README/diagram — partly landed) and **`t90`**
(below, which *is* time-critical).

---

## 4. Phase 2 — the runs

### 4.0 The merge-gate rule for this phase

`implementation`'s gate forbids moving a number; a retrain moves every number. The resolution is
already on record (t81's task file): **work that changes no code runs from `main` and opens no PR.**
Launch scripts and vendoring are ordinary `implementation` work with a PR; the *runs* themselves are
flag-only. `t81` carries no hypothesis refs and is **not** an experiment — no `exp/` lane, no null,
no verifiables, no `crux close`.

### 4.1 t90 FIRST — there is a purge clock

`/scratch/mforooz/t54_submissions_round2` holds the 23 entrant submissions: **960 GB, 23
directories, oldest file 2026-08-25**, so scratch's 60-day purge lands about **2026-10-24**. The
anchor scoring is ≈2,835 CPU-h and has not started. `/project` had 13 TiB free. **Move them now.**
The challenge *truth* is not at risk — 363 bigwigs, 254 GB, already on `/project` at
`DATA_EIC_SYNAPSE/`. Re-check both facts before acting.

### 4.2 Training — 13 runs remaining

| method | runs | notes |
|---|---|---|
| CANDI `eic.19` | 1 | **relaunch.** The existing `t81_eic_19_s0.best.ckpt` is the killed job's epoch 2, produced by `candi.eval.quick_eval`, which commit `2f56cb1` deleted. `cruxvault/results/t81/RESCORE_EIC19.md` ruled it not reusable, and whether the merged monitor would still pick epoch 2 **cannot be determined** — epochs 5 and 8 were never written to disk. |
| CANDI `eic.pilot` | **0** | done (§1). |
| CANDI `eic.19` seed 1 | 1 | **t86**, the noise floor. Ruled to run *after* the retrains. Two seeds of one method, same data. |
| Avocado | 2 | each = 1 joint fit on `train_chroms` + **3** per-chromosome genome-factor fits (§12.2 corrected; the blanking rule means only chr20/21/22 are ever predicted). |
| ChromImpute | 2 | sampler now fits on the regime's training loci. |
| eDICE | 2 | `run_eic.py` concatenates the training matrix into host RAM — size the memory ask. |
| Lavawizard | 2 | our **two-stage variant**, not the published Lavawizard; the 2019 submission stays unmodified in the anchor block. |
| `avg`, `avg-arcsinh` | 1 each | collapse holds. |
| `marginal`, `knn1`, `knn5` | 2 each | D1. |

Every trainable method selects its checkpoint on `V_`, on the 450-window scope, with the same
cadence. **The selection *key* is deliberately NOT uniform** (PI ruling 2026-09-01): CANDI selects
on count-arm `crps` (`monitor.py`'s `SELECTION_KEY`); Avocado, eDICE and Lavawizard select on
`pval:mse`. §5's uniform rule binds panel, instrument, scope and cadence — **not** the key. This
handicaps CANDI rather than favouring it; every rival row carries the marker.

ChromImpute and the naive baselines have nothing to select and carry the **"no selection"** marker
(t82 shipped it on all five baselines, not only the two kNNs).

Launcher entry points: `slurm/t81_train_candi.sh` (CANDI, `MODE=probe` then `MODE=full`),
`competitors/{avocado,lavawizard}/slurm/{_env,bin,cache,train,predict,score}.sh`,
`competitors/chromimpute/slurm/{stage,submit,score}.sh`,
`competitors/edice/slurm/eic_{train,score}.sh`, `competitors/baselines/generate.py`.

**Measure throughput on the thing you will run.** A sampled probe overestimated `--full-coverage`
by 2.2×, which mis-sized a 16 h job. And do not co-schedule two array tasks on one node — the
loader is CPU/IO bound and contention made a 5× rate error. `--array=0-1%1` serialises.

### 4.3 Phase 3 — σ refits

After training, before scoring: refit σ for all point-only methods on **training** residuals via
Phase 1 item A. Nothing distributional scores until this is done.

### 4.4 Phase 4 — predictions, ~36 runs

18 units × `V_` and `B_`. Nine kinds of unit predict **genome-wide** (CANDI ×2, eDICE ×2, `avg`,
`avg-arcsinh`, `marginal` ×2, `knn1` ×2, `knn5` ×2); six predict **chr20+21+22 only** (Avocado ×2,
ChromImpute ×2, Lavawizard ×2) because §4 blanks their `genome-wide` cell and **a blanked cell is
not computed**.

- Writer: `python -m candi.bench.dump --store <regime> --ckpt <best.ckpt> --arch-from <run.json>
  --out <pred_root> --method CANDI`. Rivals write the same §4.1 external contract.
- **`V_` predictions → scratch** (deletable after scoring). **`B_` predictions → `/project`**
  (PI ruling 2026-08-31): `B_` is predicted exactly **once**, from the `V_`-selected checkpoint, so
  the first set is the only legitimate copy that will ever exist. Re-scoring stored predictions is
  free and allowed; re-predicting `B_` is not.
- **Storage: §12.6's figures are per ARRAY, not per track.** CANDI writes **five** arrays
  (`mu`, `n`, `signal_mu`, `signal_sigma`, `peak_score`) = 2.37 GB per genome-wide track raw.
  §12.3's ≈434 GB total was summed on a one-array assumption and is low by about that factor; with
  18 units it is lower still. Compression on smooth float32 predictions is only ~1.27×, not the
  2.69× an early draft quoted (that blend included a sparse *truth* count layer, which no prediction
  has). **Record the real ratio off the first prediction root and rewrite §12.6 from it** — the
  section instructs exactly this. Fir had 13 TiB free on `/project` and 17 TiB on scratch.

### 4.5 Phase 5 — scoring, ~104 passes (≈4,400 CPU-h, all CPU)

| pass | count |
|---|---|
| `V_`, store truth | 18 |
| `B_`, store truth | 18 |
| `B_`, challenge truth — **pval arm only** | 18 |
| the 25 anchor entrants, both truths | 50 |

Scorer: `python -m candi.bench.external`. One pass ≈ 50 CPU-h for 45 tracks genome-wide on 4 cores;
held-out-only passes are 5.34 % of that. Both aggregations (`held-out`, `genome-wide`) and all three
`V_` numbers (§5.2) come out of **one** pass — no extra inference.

Under `truth: challenge` the count and peak arms are **greyed out** (no counts, no peak calls in
2019 data). Rivals never enter the count or peak arms at all.

### 4.6 Phase 6 — the noise floor (t86)

From the two `eic.19` seeds, measure the spread separately on the **`V_` breadth panel** (22 assays,
11 singletons) and on the **8-assay panels** — they do not have the same resolution.
`tools/seed_floor.py` takes the two scored jsons. Until it lands, rows go up **unranked**; §15
allows that explicitly. Do not quote `AGENTS.md` §7.2's 0.1195 beside a `candi.bench` number — it
was measured with a deleted instrument on a different population, and that substitution is the exact
failure §7.2 exists to prevent.

---

## 5. Quoting rules — these are not optional

- Quote the **noise floor with every number**, and never quote raw CRPS without its
  `oracle_scaled` / `scale_error` split (`AGENTS.md` §7.2). For DNase the split is mandatory: the
  paired agreement is **shape-only**, with a ~3× scale offset.
- Every DNase number prints the disclosed residuals beside it: `T_K562` is the worst DNase row at
  Pearson 0.4559 with a 66,660-count chr5 tower still entering `eic.pilot` training windows;
  `T_HAP-1` runs 47.8× and 120.6× above target at two loci; **`B_DND-41` is scored truth carrying a
  mitochondrial-NUMT artifact for which no agreement number can ever exist.**
- The DNase truth toggle carries a **badge**: the two truths share a generating process at
  r ≈ 0.99 in shape, so agreement there is not evidence of robustness. The scale disagreement is
  the one thing it still measures for DNase.
- Anchor block **non-independence**: `CUImpute1`, `CUWA` and `ICU` submitted byte-identical tracks
  for all 26 broad-mark experiments, and `ICU`'s H3K4me1 *is* `Avocado_p0`. Any "beats N methods"
  claim must be counted, never read off the table.
- Ranking resolution limit ≈ **0.005 correlation units**, and 5 of 24 adjacent pairs invert on ≥3 of
  the ten chromosome subsets. A placement separating two methods by less than that is not a
  placement.
- Never subtract `V_` (breadth) from `B_`. The **matched** number is the only legal subtraction.

Read `EVAL.md` for what `candi.bench` and `candi.monitor` measure and the keys they write, and
`AGENTS.md` §7 for the frozen pre-CANDII numbers. **Where a doc and the code disagree, the code is
right and the doc is the bug.**

---

## 6. What is deferred by ruling — do NOT build these

- `eic.gw→20,21,22` and the `merged.*` regimes — placeholders only.
- The **zero-shot** claim (imputation for an unseen cell type). §8: the board measures
  missing-mark transfer *within a seen cell type*. Zero-shot goes to the merged corpus later, and
  its field is CANDI, ChromImpute and the naive baselines only.
- `V_` under challenge truth (D3).

---

## 7. Phase 7 — the leaderboard, which is the deliverable

### 7.1 Unlock the board

`leaderboard/boards.json` carries, for **both** live regimes:

```
"frozen": { "store_manifest_hash": "TODO-eic.19-store-manifest",
            "regime_sha256":       "TODO-eic.19-regime-sha256" }
```

A board whose `frozen` hashes are TODO **refuses every `add`**. Freeze them by reading the rebuilt
store manifest on Fir and the tracked regime json **at the same moment the first row is scored** —
that pairing is the point of the gate. Note `eic.pilot` shares `eic.19`'s store manifest (the
regimes differ only in training loci) but has its own regime sha, because it carries the hg38 Pilot
Regions BED.

### 7.2 Stamp, build, check

```bash
python tools/leaderboard.py add <score.json> --board eic.19 --method Avocado \
    --truth store --panel V_breadth --scope held-out \
    --version <YYYY-MM-DD> --date <YYYY-MM-DD> --lineage rival \
    --position-class transductive --cell-class retrained \
    --scoring-sha <sha> --store-manifest-hash <hash>
python tools/leaderboard.py build     # -> _site/leaderboard.json + the static site
python tools/leaderboard.py check     # row gates + a deterministic double build, diffed bit-exact
```

`add` **computes nothing** — every number is copied from the score file's macro block and gated on
the way in: NaN refused, provenance mandatory, the frozen eval-set hash enforced, and the registry's
companion rules applied as refusals (count `crps` never without `crps_oracle_scaled` +
`scale_error`; pval `crps` never without `pit_ks` + `coverage_95`). Rows live at
`leaderboard/rows/<regime>/<truth>.<panel>.<scope>/<method>@<version>.json`. The five address fields
**are** the path, so a row whose address does not resolve has nowhere on disk to live.

`leaderboard/void/` holds the pre-t77 rows. They are void by §3.3 and stay void — **do not
resurrect them.**

### 7.3 Rows the board must end with

- **One CANDI row per regime** — the current best only. Version history is a separate figure, never
  extra rows: the ranker ranks *across methods* within a cell, so an extra CANDI version shifts
  every rival's rank.
- Four rivals × 2 regimes, five baselines (2 collapsed + 3 × 2), each in `V_ breadth`,
  `V_ matched`, and `B_`, under both truths where the arm exists, in both scopes where the
  `genome-wide` cell is not blanked.
- The **anchor block**: 25 entrants (23 submissions + `Average` + `Avocado_p0`) rescored through our
  scorer on our grid, under challenge truth, lifted **out** of the ranked table and labelled as an
  anchor we did not run.
- **One extra labelled figure, never a board row:** CANDI inside the 2019 field — the challenge
  ranker over `B_` under challenge truth with CANDI added to the 25-entrant field. It needs no new
  prediction and no new scoring, only one more run of the ranker. It prints the non-independence
  count with it.
- Markers: eDICE's footnote (it queries a target using that target's paired `T_` cell's embedding),
  the "no selection" marker, the `native heteroscedastic` vs `fitted flat σ` badge on every
  distributional cell, the per-cell in-sample-fraction badge, and the selection-key asymmetry marker
  on every rival row.

### 7.4 Deploy

`.github/workflows/leaderboard.yml` runs `check` then `build` and deploys `_site/` to GitHub Pages
on **every push to `main`** touching `leaderboard/**`, `tools/leaderboard.py`, or the workflow.
Scoring never runs in CI — it stays a manual Fir run; CI only compiles what was stamped. Setup note
of record: Settings → Pages → Source must be "GitHub Actions", and Pages makes the page public even
while the repo is private (the PI accepted this on 2026-08-27).

So "update the landing page" = **land the stamped rows on `main`**. Verify the deployed page
afterwards; do not declare it shipped off a green CI run alone.

---

## 8. Phase 8 — close the loop

For every task: an output that **resolves** — a vault file or a `[[wikilink]]`, never a bare commit
hash.

```bash
cd cruxvault && python ~/.claude/skills/crux/scaffold/crux.py task done t81 \
    --output "[Retrain results](results/t81/DELIVERABLE.md)"
```

Then `crux validate --check=tree,tasks`. **`crux task accept` is the PI's signature — do not run
it.** Tasks to close: t77, t81, t86, t87, t88, t89, t90, plus the new σ-pass task. t78/t79/t80/t82/
t83/t84/t85 are already done.

Write the evidence into `cruxvault/results/<tid>/` with a `FIR_PATH.txt` naming the run.
Checkpoints and logs stay on the cluster; only the small evidence comes down. Note
`cruxvault/results/` is gitignored and therefore **per-worktree** — evidence written in one worktree
is invisible from another, so copy it across rather than assuming a merge carried it.

Update `plan/BENCHMARK_DESIGN.md` as you go — it is a live design doc. Correct §12.2 (D1), §12.3 and
§12.6 (the per-array budget, from the measured ratio), and §12.5. Do not append to `AGENTS.md` §7;
it is frozen.

---

## 9. Working rules that will bite you

- **Push at creation, not at the end.** Work on one laptop is already lost. `origin` is the record;
  `firmerge` is a truck to Fir, not a home.
- **Merge `origin/main` into a long-lived branch as you go** — merge, never rebase.
- **Cut an agent's worktree from the branch you are working on, never from `origin/main`** — a
  worktree off `origin/main` lacks every file that exists only on the working branch, and anything
  that numbers itself from what already exists will collide silently.
- **Another session can move HEAD under you.** Scope every `git add`; prefer a worktree.
- **Before you tidy, look for what is not committed** — check every worktree and temp directory
  first, and rescue what is there.
- **Search every branch before you build something**, so two branches do not build the same tool.
  Two of this programme's worst hours went to a claim about the repo read off one branch (§12.8's
  "there is no track writer" — there was, on `t62`; and t84's "only Avocado is vendored" — three of
  four were already implemented, on `t60`).
- **Datasets never come to this laptop.** Cluster-to-cluster moves go through Globus.
- Before a subagent spawn, load the `dispatch` skill. Every brief names the ground truth that agent
  must verify for itself, and where.
- Name work by what it does, not by bare crux ids. Use self-describing metric names (EIC style,
  e.g. `mse1obs`), never `C1`/`M3`/`S14`.
- **Never quote a number out of a doc comment** — read the loop and the config.

---

## 10. Definition of done

1. Every task in §8 is `done` with a resolving output; `crux validate --check=tree,tasks` clean.
2. All 13 remaining training runs finished, each with a `V_`-selected checkpoint on `/project`.
3. σ tables refit on training residuals for every point-only method; no `score.sh` still refusing.
4. 36 prediction roots written — `V_` on scratch, `B_` on `/project`, `B_` predicted exactly once.
5. ~104 scoring passes complete, including the 25-entrant anchor block under both truths.
6. The noise floor measured on both panel shapes and printed beside every rank.
7. `leaderboard/boards.json` frozen hashes filled; every row stamped; `check` green; the deployed
   GitHub Pages board shows the new address fields and real numbers, with every badge and marker
   §7.3 lists.
8. `plan/BENCHMARK_DESIGN.md` §12 corrected from measurement, and §15 rewritten from "what remains
   is execution" to what actually happened.
9. A closing memo in `cruxvault/results/t81/` that a reader arriving cold can follow.

**Report faithfully.** If a run fails, say so with the output. If a step is skipped, say that. Do
not narrow the scope quietly — if something turns out to be blocked, finish everything else in full
and say explicitly what you left out and why.
