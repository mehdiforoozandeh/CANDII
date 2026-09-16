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

---

## What this programme found the handoff had wrong (2026-09-02/03)

Nothing above is edited. This is an append-only record of where this document's premises did not
survive contact with the code, the cluster and the artefacts — each item as **what the handoff said
→ what was true → where it is recorded.** No score is quoted anywhere in it: the target-clustered
noise floor is not yet measured (t86), so every figure below is a cost, a count, a size, a ratio or
a timing.

- **The challenge-truth path did not exist at all.** §4.5 and `leaderboard/boards.json` assumed the
  Synapse bigwigs "load into `candi.bench.external`". → There was **no bigwig reader anywhere** —
  not in `src/candi/bench/*`, not in `src/candi/store/*`, on no local or remote ref; `score_external`
  returned `per_track` + `macro` only, with no `panels` block, no `genome_wide` block and no
  `--held-out-chroms`; and anchor rows went through a retired `--placement-method` gate. → Built:
  `tools/challenge_bigwigs.py` (`truth-root` and `pred-root` verbs), `--truth-root` and
  `--held-out-chroms` on `candi.bench.external`, and the `panels` block. Recorded in
  `.orchestrate/plan.md` "Verified ground truth", `cruxvault/results/t81/ANCHOR_ROOTS.md`,
  `SCORES_ANCHORS.md`.

- **`tools/seed_floor.py` already existed.** §4.6 reads as if it were still to be written ("Until it
  lands, rows go up unranked"). → It existed — 195 lines with its own tests. What it lacked was a
  `--panel` option: it read only `macro`, never `panels`, so it could not produce the per-panel
  floor §4.6 itself asks for (the `V_` breadth panel against the 8-assay panels). The option was
  added rather than the tool. Recorded in `.orchestrate/plan.md` "Verified ground truth".

- **The finished `eic.pilot` run's json was lost, so the checkpoint alone was unusable.** §1 said
  "Do not retrain it. Re-verify the checkpoint's md5". → The md5 was fine and the checkpoint is a
  bare `state_dict` of 155 tensors that **holds no architecture**: job 57674899_1 was cancelled at
  its 16 h limit *during the final full-coverage check*, and `train.py` writes the run json only
  after that check, so there was nothing to rebuild the model with. → `t81_eic_pilot_s0.arch.json`
  was reconstructed from the probe run's `config.arch`, copied verbatim by script, and proved by a
  strict load: `missing=[] unexpected=[] params=2,353,661`, matching the training banner. Recorded
  in `cruxvault/results/t81/PILOT_ARCH.md`. **Lesson: size a band with the final check inside it.**

- **The anchor block is held-out only, so its costing was ~20× too high — and the measured cost is
  lower again.** §4.5 put the 50 anchor passes inside a ≈4,400 CPU-h total sized on genome-wide
  passes. → `leaderboard/boards.json` declares that block `truth=challenge`, `panel=B_`,
  `scope=held-out`; held-out is 5.34 % of genome-wide, which alone takes it to ≈150 CPU-h. Measured,
  it is far under even that: the 50 passes ran in **2–4 min each under challenge truth and 8–13 min
  under store truth**, the whole block finishing in about **40 min of wall clock**, because an
  entrant carries no σ and none of the distributional work runs. Recorded in
  `cruxvault/results/t81/SCORES_ANCHORS.md` §6 and §9.1, and in `BENCHMARK_DESIGN.md` §12.4.

- **t88 and t89 had already landed.** §3F lists t88 as Phase-1 work and §3 lists t89 as "also open".
  → t88's fix was already on `origin/main` (commit `a2a47b1`, arrived through the t77 merge; the
  shipped-regime test now sweeps `configs/`), and a fresh detached worktree at HEAD ran **1,294
  passed, 1 skipped**, the single skip being by design. t89's `src/candi/README.md`,
  `tools/arch_diagram.py` and `tests/test_arch_readme.py` existed and passed. Only the crux closes
  remained. Recorded in `.orchestrate/plan.md` "Verified ground truth",
  `cruxvault/results/t88/FRESH_WORKTREE_PYTEST.md`, `cruxvault/results/t89/DELIVERABLE.md`.

- **A regime file cannot carry the σ pass's self-pairs.** §3A's design (and the programme plan's
  pinned interface) had the derived σ regime hold `eval_pairs = [[T_x, T_x] …]`. →
  `candi.store.regime` **refuses** a pair whose target is its own source. The derived file therefore
  carries `eval_pairs: []` with `biosamples.eval` set to the drawn cells, and every method reaches
  the self-pairs through the store's **no-pairing path**. Three predict entry points read
  `regime["eval_pairs"]` directly and silently wrote nothing under that shape until the fallback was
  added. Recorded in `cruxvault/results/t81/W3_CI_SIGMA.md`, `SIGMA_AVGARCSINH.md`, and
  `BENCHMARK_DESIGN.md` §12.2.

- **`V_matched` needs the sibling `B_` pass; it does not come out of the `V_` pass.** §4.5 says all
  three `V_` numbers come out of one pass. → True of a joint pass, false once the passes are split
  by panel: `harness.panel_macros` measures the matched assay set from the `B_` rows **of the same
  pass**, and our `V_` and `B_` passes run off panel-derived regimes, so `panels.V_matched` came out
  **empty in every `store.V_.json`** and the matched board cell would have been blank for all 18
  units. → Fixed by a merge, not a re-score: `candi.bench.external fill-panels --v … --b …`
  recomputes `panels` over the union of the two files' `per_track` and records
  `provenance.panels_from`; measured `V_matched.n_experiments` **0 → 21** with `V_breadth`
  byte-identical either side. Recorded in `cruxvault/results/t81/FILL_PANELS.md`.

- **Three of the five naive baselines are regime-dependent, and `avg-arcsinh` had a live bug
  underneath that.** §3D's "correct the collapse to 2 + 3×2 = 8 units" was right: `similarity_table`
  and `fit_marginal` read `panel.train_chroms`, so `marginal`, `knn1` and `knn5` fit differently per
  regime. → The identity assertion, once it existed, caught what the argument could not:
  **`avg-arcsinh` was averaging the top five contributors by kNN similarity** rather than every
  eligible one, which made it regime-dependent too while its one number was being printed in both
  regime rows. The three-contributor test fixture hid it (`top_k(…, 5)` over three returns all
  three); the identity tests run on seven contributors now. The pilot trio also needed `regions`
  support before they could be fitted at all. No board row quotes the old path. Recorded in
  `cruxvault/results/t81/BASELINES.md`, `competitors/baselines/README.md`, and
  `BENCHMARK_DESIGN.md` §12.2.

- **`sbatch --export=ALL,VAR=a,b,c` truncates a comma-valued variable — and it cost a whole apply
  stage.** Nothing here warned of it. → `CI_CHROMS=chr20,chr21,chr22` passed inside `--export=…`
  arrived at the job as `CI_CHROMS=chr20` (measured in the running job's environment), so
  ChromImpute's first `V_` apply ran **chr20 only**; chr21 and chr22 were never converted. Those
  outputs were discarded and the chain re-run with the variables exported in the shell and
  `--export=ALL`. The trained predictors were unaffected. Recorded in
  `cruxvault/results/t81/W3_EARLY.md`, `W3_CHECK_2.md`.

- **The real critical path was the GPU fairshare queue, not any CPU-h total.** §4.5's "Fir's CPU
  allocation is not the scarce resource" held — every CPU scoring pass ran unimpeded. → But with
  this programme holding ~39 % of the account's GPU use, `sprio` put essentially the whole of a
  pending job's priority in fairshare (631,533 of 631,602), the scheduler offered **no start time
  at all**, and six GPU jobs — both CANDI `V_` predicts, both Avocado σ runs, both eDICE `B_` runs —
  sat `PENDING (Priority)` for hours. **The CANDI rows are the last on the board to land, and the
  reason is queueing, not compute.** Recorded in `cruxvault/results/t81/TRAIN_CANDI_EIC19.md` and
  the `pred.V_.CANDI.*` rows of `PROGRAMME_STATE.md`.

- **The baseline prediction roots are uncompressed.** This document inherited §12.6's "deflates"
  claim from t83. → True of CANDI, eDICE, Lavawizard and ChromImpute, which all call
  `savez_compressed`. **Not** of the naive baselines: `competitors/baselines/generate.py:458` calls
  `np.savez`, and the measured ratio is **1.000×**, with on-disk **1,250 bytes larger** than raw
  (the zip container header). That is the largest class of roots on `/project` — **744 G** across
  the eight baseline `B_` roots. eDICE, measured on a real root, deflates at **1.208×**; CANDI's
  ratio is still unmeasured and is deliberately left blank in §12.6 rather than estimated. Recorded
  in `cruxvault/results/t81/SCORES_BASELINES_B.md` §1.3, `PRED_B_EDICE.md` §4, and
  `BENCHMARK_DESIGN.md` §12.6.

**The pattern in all eleven.** Nine of them are a claim about an artefact taken from a description of
it — a section of this handoff, a launcher header, a doc comment, a manifest that records no sizes.
Every one was settled by opening the thing itself: the npz's key list, the job's own banner, the
regime loader's refusal, a strict `load_state_dict`. **Where a doc and the code disagree, the code is
right and the doc is the bug** — and that includes this document.

### Second batch (2026-09-03/05) — the CANDI runs themselves

Six more, in the same form. Two of the eleven above have since been overtaken by measurement, and
are left standing rather than edited: the seed floor **is** measured now, on all three panels, in
`plan/BENCHMARK_DESIGN.md` §12.9, and CANDI's compression ratio is no longer blank — §12.6 carries
1.177× on the `B_` root. The preamble's discipline still holds below: **nothing here is a score.**
Every figure is a defect, a device, a memory size, a count or a date.

- **The CANDI predict launcher had never run end to end, and three defects were stacked in it
  (2026-09-03).** §4.4 treats prediction as a solved step to be scheduled. → No CANDI predict had
  ever passed its own plan step, so nothing had ever exercised the path. **K14:**
  `slurm/t81_predict_candi.sh` read chromosome sizes from `CANDI_STORE/eic/genome/chrom_sizes.json`,
  which does not exist — the genome layer is `CANDI_STORE/genome`, the same bug K13 had already
  fixed in the scorer — so the chromosome list came out empty and the plan step exited 1 at start.
  **K16, defect 1:** with that fixed, the launcher still captured the plan heredoc's **stdout**
  positionally (`sed -n 1p/2p/3p`), and `harness.py:701` prints a `[bench] … declared eval pair(s)`
  banner to stdout with no `file=sys.stderr`, so every field arrived shifted by one line and the
  chromosome list became that banner. This predates K15 — the chunk-C script at `1633756` has the
  same capture — and K15 only made it fail loudly, because the sharded path counts the chromosomes
  and refuses a mismatch. It broke the unsharded GPU path too, and exporting `CHROMS` did not save
  it, because the launcher overwrote the exported value unconditionally. **K16, defect 2:**
  `tools/declare_eval_pairs.py` wrote the derived regime json with `write_text` and read it back at
  once, and every shard of an array wrote the **same** path, so one shard read a half-written file;
  the fix was `mkstemp` + `os.replace`. A **third** defect appeared the next day when a seed-1 shard
  still read zero bytes back from another node — on Lustre an atomic rename can be visible before
  its data is flushed — and the foreman added an `fsync` of the temp file before the replace. All 92
  tasks of the first CPU launch failed at plan time. Nothing was written, and the `B_` once-only
  budget stayed intact, because the marker guard sits after the shard-index check. Recorded in
  `cruxvault/results/t81/TRAIN_CANDI_EIC19.md` §"CPU-sharded launches" §5 and §"CPU-sharded relaunch
  (2026-09-03, K16)".

- **The GPU queue starved, so CANDI predicted on CPU — and two memory ceilings nobody had chosen bit
  in turn (2026-09-03/04).** §4.4 sizes CANDI's prediction runs in GPU-hours. → With this programme
  holding about 39 % of the account's GPU use, `sprio` put essentially all of a pending job's
  priority in fairshare and `squeue --start` gave no estimate at all; CANDI's predicts sat `PENDING`
  for nine hours. A read-only check found a real `--device cpu` path in `dump.py` and no CUDA-only
  code, and measured throughput said one unsharded CPU pass would need 71–143 h — over the 60 h band
  — while 23 per-chromosome shards of about 6 h each would fit. **K15** built that sharded mode: a
  per-shard dump, a merge job that verifies every track × chromosome before writing one manifest,
  thread count from `--cpus-per-task`, and the device recorded in the manifest. It worked, at
  1:15–1:35 h a shard. Two ceilings then failed. **16 GB is too small for a `chr1` shard at 51
  tracks:** `MaxRSS` is linear in chromosome length at about 0.0688 GiB/Mb + 0.86, so `chr2` peaked
  at 15.99 G and `chr1` needed ~18 G; three `B_` arrays — `eic_19`, `eic_pilot` and seed 1 — each
  lost their `chr1` task, and each needed an unsharded 32 GB repair into the same root, plus a fresh
  merge behind it. **32 GB is far too small for a CANDI score pass:**
  `slurm/t81_score_external.sh`'s header `--mem=32G` was inherited by all six CANDI genome-wide
  passes, and a CANDI track holds five `float32` arrays — about 4.8 GB live per track genome-wide,
  against ~1 GB for a one-array rival. Two passes died hours in; the other four all sat at exactly
  32.0 GiB, were duplicated at raised memory pre-emptively, and then died as predicted. The
  replacements ran at 128 GB (`V_`) and 256 GB (`B_`), with `--cpus-per-task` raised to 33 and 66 so
  that the 4000 MB-per-core rule kept them on `cpubase` instead of a large-memory pool six days out.
  The live peak turned out to be 34–41 GB, far under the 148 GiB projection: the `eic_19` `B_` pass
  ran at 256 GB and landed, while 128 GB duplicates covered the `eic_pilot` and seed-1 `B_` passes
  and their 256 GB backstops were cancelled. One trap for a reader: the score launcher prints
  `DONE … rc=137` after a kernel-killed python, so a grep for `DONE` reads an OOM as a success.
  Recorded in
  `cruxvault/results/t81/TRAIN_CANDI_EIC19.md` §"eic_19 B_ shard repair", §"eic_pilot B_ shard
  repair" and §"Score-pass OOMs and resubmissions".

- **Avocado's `V_` and `B_` roots are not the same numerical path (2026-09-03).** Nothing here warns
  that a device fallback changes the arithmetic. → The same starvation pushed Avocado's `B_`
  predicts to CPU, where they finished in 13–19 min a chromosome. But `predict.py:133` enables
  autocast only when the device is CUDA, so Avocado's `V_` roots were predicted on a MIG slice under
  **bf16** autocast and its `B_` roots on CPU in **fp32**. Same checkpoints, same inversion, same
  chromosomes, same declared panel — only the precision of the forward differs, and no reader can
  infer that from the score json. Avocado's `V_`↔`B_` gap therefore carries a precision term that no
  other method's gap carries, and the size of that term is **unmeasured**: a CPU `V_` re-run to
  measure it was not in scope. Both boards carry it as a device note from stamp-10. CANDI has no
  such term — `candi.bench` wraps eval in `no_autocast`, so it is fp32 on either device. Recorded in
  `cruxvault/results/t81/W3_AVOCADO.md` §26.

- **The PI ruled that CANDI's CPU prediction roots are canonical (2026-09-04).** §4.4 assumes one
  prediction root per unit and says nothing about a second. → Because the doomed GPU jobs still held
  the canonical `V_` path when the CPU route launched, the CPU `V_` passes wrote to a **sibling
  root**, `…/CANDI/<regime>/V_cpu`, while `B_` went to its canonical root, which no GPU job had ever
  claimed. That left two candidate `V_` roots on paper. The PI ruled that **the CPU roots are
  canonical** and are not to be renamed — the score passes were already reading them — so `V_cpu` is
  the only CANDI `V_` root there is. The CANDI caveats on both boards were rewritten to say so at
  stamp-20. Recorded in `.orchestrate/plan.md` (PI rulings, 00:13 PDT 2026-09-04) and in the CANDI
  caveats of `leaderboard/boards.json`.

- **Ten queued GPU jobs were doomed from the moment they were submitted, and were left in the queue
  on purpose (2026-09-03/05).** §4.4's model is submit and wait. → Every CANDI GPU predict queued
  before K16 carried the stdout-capture defect above — `57910740`, `57914202`, `57942777`,
  `57942786`, `57943014`, `57943025`, `57944437`, `57944438`, and two chained score jobs — and each
  would have burned a GPU allocation before failing, less legibly than the CPU route did, because
  the model loads first. SLURM removed the dependants itself as `DependencyNeverSatisfied`; the
  predicts needed a `scancel`, which is the PI's call and not an agent's, so they were left queued
  and the PI cancelled them. **The general lesson:** a queued job carries the code it was submitted
  against, so a defect found after submission is still sitting in the queue, and only a human
  decides whether it is cancelled or allowed to fail. Recorded in
  `cruxvault/results/t81/TRAIN_CANDI_EIC19.md` §"CPU-sharded launches" §7 and `.orchestrate/plan.md`.

- **The board carried no caveat about the D1 collapse until stamp-10 (2026-09-03).** §7.3 lists the
  badges and markers a row must carry, and does not name this one. → Under decision D1, `avg` and
  `avg-arcsinh` are fitted once and scored twice, so their `eic.pilot` rows are the **`eic_19` root
  scored at pilot addresses**. That is provable, and was proved by the identity assertion, but it is
  invisible to anyone reading the pilot board: `boards.json` said nothing about it, while the σ badge
  on the `avg-arcsinh` pilot rows names `sigma.eic_19.json` and `avg` fits no σ table at all. A
  factual caveat was added on `eic.pilot` at stamp-10 — a statement of what the row is, not a
  judgement of it. Recorded in `plan/BENCHMARK_DESIGN.md` §12.2 (D1) and the `eic.pilot` caveats in
  `leaderboard/boards.json`.

**The pattern in these six is not the pattern in the first eleven, and is worth naming separately.**
Only one of them is a stale description. The rest are the price of a path nobody had walked: a
launcher with three defects stacked in it, because nothing had ever reached the second one; two
memory ceilings inherited from headers nobody had revisited; a device fallback that changed the
arithmetic silently; and a queue holding jobs built against code that no longer existed. **A step
that has never run end to end is not a scheduled step. It is unbuilt work with a job id.** Run the
whole path once before queueing twenty of it, and size every band with the final check inside it.
