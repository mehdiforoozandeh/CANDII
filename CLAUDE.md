# CLAUDE.md

This file loads on **every turn**. The other docs do not. Keep it a front door, not a spec —
`tests/test_docs.py` holds it to 120 lines. Content that outgrows this belongs in one of the
files below, with a router line here pointing at it.

## What this is

CANDI — a self-supervised epigenome imputer/denoiser over **raw counts**, emitting a Negative
Binomial `(n̂, p̂)` per assay per 25 bp bin. ~2.35 M parameters. It is conditioned on four
experimental covariates on both the input and the output side, so it does zero-shot imputation
and denoising on cell types it has never seen. `README.md` is the five-minute version.

## Where things live

| you need | read |
|---|---|
| invariants, frozen construction order, common tasks | `AGENTS.md` §3, §4 |
| validation gates, failure signatures | `AGENTS.md` §5, §6 |
| pre-CANDII results, and the rules for quoting any number | `AGENTS.md` §7 |
| h5 schema, masking, the input contract | `DATA.md` |
| the CANDI_STORE corpus store — layout, codecs, manifest | `STORE.md` |
| what M1/M2/M3/S14 measure and the keys they write | `EVAL.md` |
| every open question, hypothesis, task and result | `cruxvault/` |

Where a doc and the code disagree, **the code is right and the doc is the bug.**

## How we work

The old repo failed by accumulation, not by being wrong. Everything below exists to keep this
one legible after a hundred more commits.

### Every piece of work is a crux task first

Science goes in the tree (`cruxvault/` questions and hypotheses). **Doing** goes in the taskhub.
A task may serve many hypotheses; a hypothesis may need no task at all. Load the `crux` skill
rather than driving the CLI from memory.

```bash
crux task add "add a Bernoulli peak head" -c implementation \
    --ref h7 --ref h9 --blocked-by None --why "unblocks the peak arm of q2"
```

### A branch exists because code must change

Every architecture choice here is already a CLI flag, and scale comes from the h5. **An
experiment that only changes flags needs no branch** — run it from `main` and record it in the
vault. Open a branch when the code does not exist yet.

```bash
git switch -c <lane>/<taskid>-<slug>       # e.g. exp/t12-peak-head
git push -u origin <lane>/<taskid>-<slug>  # AT CREATION, not when it is finished
```

The lane is `exp` when the engine has called the task an *experiment* (it does that
automatically for any task carrying hypothesis refs); otherwise it is the task's own category.
Push at creation is not a style preference: work that lives on one laptop is work that is
already lost.

`origin` is the record. `firmerge` is a truck to Fir for running things — **nothing ever lives
only there.**

### What lets a branch merge

**Non-experiment lanes** — the branch must not move a number:

```bash
pytest tests/ -q          # green
python tools/golden.py    # bit-exact against the pre-change tree
```

If a `docs/` or `implementation/` branch moves a number, it was mislabelled. Find the bug or
relabel the task.

**Experiment lanes — TODO, NOT YET DEFINED.** The gate is blocked on an imputation-methods
leaderboard that does not exist yet. Until it lands, an `exp/` branch does **not** merge on
anyone's judgement that a number looks better. Write down why the naive version is not enough:
the target-clustered noise floor on macro CRPS is **~0.09**, and a *seed change alone* moves
pooled CRPS by **0.1195** (`AGENTS.md` §7.2). Several recorded between-arm gaps are smaller
than that. Do not invent a gate to unblock yourself — raise it.

### Closing the loop

```bash
crux task done t12 --output "merged 9a3f1c2" --concluded h7:supported
crux task accept t12     # the PI's signature — never run this on your own judgement
```

An experiment's conclusion does **not** close a hypothesis. `crux close h7` is a separate,
PI-gated step after the verifiable boxes are ticked.

## Where numbers live

- **Before this repo** → `AGENTS.md` §7. Frozen. Never append to it. The runs behind it are in
  the read-only archive vault at `~/Desktop/research/libbrechteam@sfu/CANDI/cruxvault`.
- **From CANDII on** → `cruxvault/`. Never copied back into `AGENTS.md`.
- The quoting rules in `AGENTS.md` §7.2 govern **both**: quote the noise floor with every
  number, and never quote raw CRPS without its `oracle_scaled` / `scale_error` split.

## The vault

`cruxvault/` is tracked. Two directories inside it are **not**, and must stay that way:

- `cruxvault/raw/` — 332 MB of source PDFs. Present locally, never committed.
- `cruxvault/results/` — `results/<hid>/` holds the small evidence rsync'd down from Fir: the
  report crux links from `## Artifacts`, its figures, the scored JSON, and a `FIR_PATH.txt`
  naming the run. Checkpoints and logs stay on the cluster. **Not** a symlink — Fir is reached
  over SSH and is not mounted here, so a link would have no local path to point at.

Only the literature wiki was inherited from the old vault. No old hypothesis was migrated.
Routine lint is `crux validate --check=tree,tasks`; the full wiki lint needs a local `raw/`.

## Exceptions on record

- `docs/claude-md` (2026-08-18) predates the vault and is the only branch with no task id.
