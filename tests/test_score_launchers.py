"""K13 — the two scoring launchers, run for real against a stub `python`.

`slurm/t81_score_external.sh` and `slurm/t49_baselines_score.sh` are the only two things that ever
invoke `candi.bench.external` on a board row, and both of the defects fixed here cost REAL WALLTIME
before anything noticed:

* the generic launcher derived its genome-wide `--chroms` from the STORE (24 chromosomes, chrY
  included) while every §4.1 prediction root carries 23. `external.read_track_arrays` does not skip
  a chromosome a track lacks, it raises -- at the first TRACK read, after the first pair's truth has
  been streamed. Job `57911698` died of it hours in; `57853765` lived only because someone appended
  a second `--chroms` by hand and argparse keeps the last;
* the baselines launcher is the ONLY one that can address a collapsed root (`avg`, `avg-arcsinh`,
  generated once under one regime) against the OTHER board, because it is the only one carrying the
  `regime_independent` stamp gate -- and it had no `SIGMA` and no `TRUTH`, so the four `eic.pilot`
  units needing a sigma table or the challenge truth had nowhere to run that did not skip the gate.

Both are argument-shaping bugs, so both are testable without a cluster. Each test below RUNS the
launcher with a stub `python` on PATH that records what `candi.bench.external` would have been
called with. Everything between the stubs is the real script: the real chromosome intersection, the
real Rule 1 sigma gate, the real stamp gate. Only the venv, the kit and the scorer are fakes.

K14 adds section (C): `slurm/t81_predict_candi.sh` had the SAME sizes-path defect in the heredoc
that plans its run, and it is worse there -- the plan runs before the dump, so with `CHROMS` unset
EVERY submission exited 1 at PLAN and no root was written at all. The predict launcher cannot be
run whole here (checkpoint, arch json, GPU, a real store behind `open_source`), so that one program
is lifted out of the script text and run against the same fake store.

K16 adds section (C2), and changes a fixture rather than only adding checks: the stub `open_source`
now PRINTS the `[bench] N declared eval pair(s), M scoreable experiment(s)` banner the real one
prints on stdout. The K15 stub was silent, so every test here passed on a plan whose first line was
that banner -- and all 92 array tasks of 2026-09-03 read it as their chromosome list. A stub quieter
than the library it stands for cannot see a defect about what the library says out loud.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SLURM = REPO / "slurm"
GENERIC = SLURM / "t81_score_external.sh"
BASELINES = SLURM / "t49_baselines_score.sh"
#: K14 -- `slurm/t81_predict_candi.sh` carried the SAME chrom_sizes defect, in the heredoc that
#: plans its run. It is not a scorer, so only that one program is exercised below.
PREDICT = SLURM / "t81_predict_candi.sh"

#: The 23 every prediction root carries, in `layout.sort_chroms` order. Also what the eic CORPUS
#: holds data for: `manifest.json` -> `genome.n_bins` names exactly these (checked on Fir,
#: as does every biosample entry), and it is the map `CorpusStore.n_bins()` reads -- which is where
#: the eDICE and baselines writers get their §4.1 list, hence roots that agree.
ROOT_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
#: What the SHARED GENOME LAYER declares -- `CANDI_STORE/genome/chrom_sizes.json`, one level above
#: the corpus. One more than the corpus holds, and that one is the whole bug.
STORE_CHROMS = ROOT_CHROMS + ["chrY"]
HELD_OUT = "chr20,chr21,chr22"
#: The one prefix a sigma table must carry, on either launcher.
GOOD_FITTED_ON = "training-residuals: sigma.eic_19.json T_ self-pairs, 12 cells"
BAD_FITTED_ON = "eval-pairs: V_ residuals, 26 pairs"

#: Stands in for `python`. Runs the launchers' inline programs FOR REAL (they are the logic under
#: test) and records the one call that would have cost 50 CPU-h. Silent on stdout, because the
#: genome-wide branch captures stdout into a shell variable.
_FAKE_PY = '''
import json, os, sys
from pathlib import Path

sys.path.insert(0, os.environ["HARNESS_SRC"])
argv = sys.argv[1:]


def opt(name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


def record():
    d = Path(os.environ["SCORE_RECORD_DIR"])
    d.mkdir(parents=True, exist_ok=True)
    n = len(list(d.glob("*.json")))
    (d / ("%03d.json" % n)).write_text(json.dumps({"argv": argv[2:], "env": dict(os.environ)}))


if argv and argv[0] == "-":
    # `python - a b c <<EOF` -- the launchers' inline programs. Run them, do not fake them.
    if os.environ.get("FAKE_HARNESS"):
        # K15: the predict launcher's PLAN block counts declared tracks through
        # `harness.open_source`, which wants a real StoreDataset over real h5. The chromosome
        # derivation beside it is the logic under test and stays entirely real.
        import types
        _stub = types.ModuleType("candi.bench.harness")

        class _Src:
            def pairs(self, kind):
                return [("T_C1", os.environ.get("PANEL", "V_") + "C1")]

            def targets(self, pair, kind):
                return ["H3K4me3", "H3K27ac"]

            def close(self):
                pass

        def _open_source(**kw):
            # K16: THE STUB IS NOISY ON PURPOSE, because the real one is. `open_source` prints
            # `[bench] N declared eval pair(s), M scoreable experiment(s)` on STDOUT with
            # flush=True, and the plan is captured whole by PLAN="$( ... )". The K15 stub was
            # silent, so every test here passed while all 92 array tasks of 2026-09-03 read that
            # banner as line 1 of the plan. A silent stub cannot see this defect at all.
            print("[bench] 1 declared eval pair(s), 2 scoreable experiment(s)", flush=True)
            print("[bench] 0 declared pair(s) are NOT disjoint on (cell, assay)", flush=True)
            return _Src()

        _stub.open_source = _open_source
        sys.modules["candi.bench.harness"] = _stub
    src = sys.stdin.read()
    sys.argv = list(argv)
    exec(compile(src, "<launcher-heredoc>", "exec"), {"__name__": "__main__"})
    sys.exit(0)

if argv[:2] == ["-m", "candi.bench.external"]:
    record()
    out = opt("--out")
    if out:                                   # else the launcher's own rc=5 guard fires
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text("{}")
    sys.exit(0)

if argv[:2] == ["-m", "candi.bench.dump"]:
    # K15. Writes what a real dump writes, in the order it writes it: one npz per declared track
    # per chromosome it was GIVEN, then manifest.json last, naming only those chromosomes. That
    # last property is the whole reason the sharded run needs a merge step.
    record()
    out = Path(opt("--out"))
    chroms = [c for c in (opt("--chroms") or "").split(",") if c]
    tracks = json.loads(os.environ.get("FAKE_DUMP_TRACKS",
                                       '["T_C1__V_C1__H3K4me3", "T_C1__V_C1__H3K27ac"]'))
    for t in tracks:
        (out / t).mkdir(parents=True, exist_ok=True)
        for c in chroms:
            (out / t / ("%s.npz" % c)).write_bytes(b"")
    (out / "manifest.json").write_text(json.dumps({
        "method": opt("--method"), "version": opt("--version", ""),
        "generated_by": "candi.bench.dump", "date": "2026-09-03",
        "arms": ["count", "pval"], "declared_tracks": list(tracks),
        "notes": opt("--notes", ""), "ckpt_sha256": "ckptsha", "store_manifest_sha256": "storesha",
        "regime_id": "regimesha", "panel": "V", "code_sha": "codesha", "seed": 0,
        "chroms": list(chroms), "unknown": {},
    }, indent=2) + "\\n")
    sys.exit(0)

if argv and Path(argv[0]).name == "declare_eval_pairs.py":
    # `split` copies the store through and keeps the pairs whose TARGET sits this panel -- the one
    # property the launchers downstream of it read back out of the derived file.
    panel = opt("--panel")
    src_regime = json.loads(Path(opt("--regime")).read_text())
    pairs = [p for p in src_regime.get("eval_pairs", [])
             if str(p[1] if isinstance(p, list) else p["target"]).startswith(panel)]
    Path(opt("--out")).write_text(json.dumps(
        {"panel": panel, "store": src_regime.get("store"), "eval_pairs": pairs,
         "eval_chroms": src_regime.get("eval_chroms", [])}))
    sys.exit(0)

sys.exit("[fake py] unexpected program: %r" % (argv,))
'''

#: `_kit_pin.sh` imports `candi` and compares `candi.__file__` to `$KIT/src`. The fake kit has no
#: src, so the guard is stubbed to its one observable effect. `tests/test_slurm_kit_pin.py` is what
#: holds the real guard, and holds these two scripts to sourcing it.
_FAKE_KIT_PIN = 'export PYTHONPATH="$KIT/src"\necho "[kit-pin] stub"\n'


def _stub_env(tmp: Path) -> dict:
    """A PATH with the stub `python` on it, a fake venv, a fake kit, and a place to record calls."""
    stub = tmp / "stub"
    stub.mkdir(parents=True, exist_ok=True)
    py = stub / "python"
    py.write_text(f"#!{sys.executable}\n{_FAKE_PY}", encoding="utf-8")
    py.chmod(0o755)

    kit = tmp / "kit"
    (kit / "slurm").mkdir(parents=True, exist_ok=True)
    (kit / "slurm" / "_kit_pin.sh").write_text(_FAKE_KIT_PIN, encoding="utf-8")
    (kit / "tools").mkdir(exist_ok=True)

    venv = tmp / "venv"
    (venv / "bin").mkdir(parents=True, exist_ok=True)
    (venv / "bin" / "activate").write_text("", encoding="utf-8")

    env = dict(os.environ)
    env.update({
        "PATH": f"{stub}{os.pathsep}{env['PATH']}",
        "HARNESS_SRC": str(REPO / "src"),
        "SCORE_RECORD_DIR": str(tmp / "records"),
        "KIT": str(kit),
        "VENV": str(venv),
    })
    for leak in ("SCOPE", "SIGMA", "TRUTH", "TRUTH_ROOT", "CHROMS", "HELD_OUT",
                 "HELD_OUT_CHROMS", "EXTRA", "VARPOOL", "METHOD", "PANEL", "REGIME",
                 "PRED", "OUT", "CRPS_APPROX", "SLURM_ARRAY_TASK_ID",
                 # K15 -- the sharding and device knobs of slurm/t81_predict_candi.sh
                 "DEVICE", "SHARD_CHROMS", "SHARD_MERGE", "SHARD_ARRAY", "FAKE_HARNESS",
                 "FAKE_DUMP_TRACKS", "B_ONCE", "SEED", "CKPT", "CKPT_DIR", "ARCH_FROM",
                 "WORKSPACE", "VERSION", "BATCH_WINDOWS", "REGIME_NAME", "OMP_NUM_THREADS",
                 "MKL_NUM_THREADS", "SLURM_CPUS_PER_TASK", "SLURM_ARRAY_TASK_COUNT",
                 "SLURM_ARRAY_JOB_ID", "SLURM_JOB_ID"):
        env.pop(leak, None)          # the operator's own shell must not decide what is tested
    return env


def _store(tmp: Path, chroms=STORE_CHROMS, corpus_chroms=ROOT_CHROMS) -> Path:
    """A CANDI_STORE with a shared genome layer, and one corpus under it. Returns the CORPUS root.

    `layout.corpus_genome_dir` is `<corpus>/../genome`, so the corpus has to be a real sibling of
    `genome/` -- the launcher reads the sizes through that helper and a flat directory would not
    exercise it.

    `chroms` is what the SHARED layer declares; `corpus_chroms` is what the corpus holds DATA for,
    recorded the way the real store records it -- `manifest.json` -> `genome.n_bins`, which is what
    `CorpusStore.n_bins()` reads. On Fir the two differ by chrY, and that difference is the defect
    both launchers had. The `biosamples/` directory is empty on purpose: `CorpusStore` requires it
    to exist, and a manifest means nothing under it is ever opened.
    """
    genome = tmp / "CANDI_STORE" / "genome"
    genome.mkdir(parents=True, exist_ok=True)
    (genome / "chrom_sizes.json").write_text(
        json.dumps({c: 50_000_000 for c in chroms}), encoding="utf-8")
    corpus = tmp / "CANDI_STORE" / "eic"
    (corpus / "biosamples").mkdir(parents=True, exist_ok=True)
    (corpus / "manifest.json").write_text(json.dumps({
        "schema": 2, "corpus": "eic", "resolution": 25,
        "genome": {"build": "GRCh38", "n_bins": {c: 2_000_000 for c in corpus_chroms}},
    }), encoding="utf-8")
    return corpus


def _regime(path: Path, corpus: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"store": str(corpus), "eval_pairs": [["C1", "T1"]],
                                "eval_chroms": HELD_OUT.split(",")}), encoding="utf-8")
    return path


def _pred_root(root: Path, chroms=ROOT_CHROMS, *, in_manifest=True, extra_manifest=None) -> Path:
    """A §4.1 prediction root: manifest.json plus one track dir holding one npz per chromosome."""
    track = root / "C1_T1"
    track.mkdir(parents=True, exist_ok=True)
    for c in chroms:
        (track / f"{c}.npz").write_bytes(b"")
    man = {"method": "fake", "arms": ["pval"], "n_tracks": 1}
    if in_manifest:
        man["chroms"] = list(chroms)
    man.update(extra_manifest or {})
    (root / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    return root


def _sigma(path: Path, fitted_on: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"method": "fake", "fitted_on": fitted_on,
                                "sigma": {"H3K4me3": 1.0}}), encoding="utf-8")
    return path


def _run(script: Path, env: dict, tmp: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(script)], env=env, cwd=str(tmp),
                          capture_output=True, text=True)


def _calls(env: dict) -> list:
    d = Path(env["SCORE_RECORD_DIR"])
    return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))] if d.exists() else []


def _flag(argv: list, name: str):
    """The value of `--name`, or None when it is absent. LAST wins, as argparse does it."""
    hits = [i for i, a in enumerate(argv) if a == name]
    return argv[hits[-1] + 1] if hits else None


pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="no bash on this host")


# ---------------------------------------------------------------------------------------------
# text — the three files still parse, and still say the things the launchers are for
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("script", [GENERIC, BASELINES, PREDICT], ids=lambda p: p.name)
def test_the_launcher_parses(script: Path):
    """All carry inline heredocs, and a heredoc that does not close is invisible by eye.

    On `t81_predict_candi.sh` this is a live guard, not a formality: its heredoc sits inside a
    `"$( )"`, where bash pairs up quote characters in the BODY even though the delimiter is
    quoted. One apostrophe in a comment there is a parse error for the whole script.
    """
    r = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert r.returncode == 0, f"{script.name} does not parse:\n{r.stderr}"


def test_the_baselines_launcher_names_sigma_and_truth_at_all():
    """The reported defect was a `grep` with ZERO hits in all 196 lines."""
    text = BASELINES.read_text(encoding="utf-8")
    for name in ("SIGMA", "TRUTH", "--sigma-table", "--truth-root"):
        assert name in text, f"t49_baselines_score.sh still never mentions {name}"


def test_the_generic_launcher_hands_the_prediction_root_to_its_chromosome_program():
    """A genome-wide list derived from `$REGIME` alone is the chrY bug, restored.

    The comment lines are dropped: the header EXPLAINS the store-only version it replaced, and a
    check that read the prose could not tell the explanation from the code.
    """
    code = "\n".join(ln for ln in GENERIC.read_text(encoding="utf-8").splitlines()
                     if not ln.strip().startswith("#"))
    assert 'python - "$REGIME" "$PRED"' in code, (
        "the genome-wide chromosome program is not given $PRED, so it can only be reading the "
        "store -- which carries chrY and no prediction root does")
    assert "chrom_sizes.json" in code, "it must still intersect AGAINST the store's list"


# ---------------------------------------------------------------------------------------------
# (A) slurm/t81_score_external.sh -- score the chromosomes the ROOT carries
# ---------------------------------------------------------------------------------------------

def _generic_env(tmp: Path, *, root_chroms=ROOT_CHROMS, store_chroms=STORE_CHROMS,
                 in_manifest=True, **over) -> dict:
    env = _stub_env(tmp)
    corpus = _store(tmp, store_chroms)
    env.update({
        "REGIME": str(_regime(tmp / "regime.eic_19.B_.json", corpus)),
        "PRED": str(_pred_root(tmp / "pred" / "B_", root_chroms, in_manifest=in_manifest)),
        "OUT": str(tmp / "scores" / "store.B_.json"),
        "SCOPE": "genomewide",
    })
    env.update(over)
    return env


def test_genomewide_scores_the_23_the_root_carries_and_not_the_stores_24(tmp_path):
    env = _generic_env(tmp_path)
    r = _run(GENERIC, env, tmp_path)
    assert r.returncode == 0, f"exited {r.returncode}:\n{r.stdout}\n{r.stderr}"
    calls = _calls(env)
    assert len(calls) == 1, f"the scorer was called {len(calls)} times, not once"
    chroms = _flag(calls[0]["argv"], "--chroms").split(",")
    assert chroms == ROOT_CHROMS, f"--chroms is {chroms}"
    assert "chrY" not in chroms, (
        "the store's chrY reached --chroms; read_track_arrays raises on it hours into the pass")
    assert _flag(calls[0]["argv"], "--held-out-chroms") == HELD_OUT, (
        "--held-out-chroms must be unchanged -- without it the json carries no genome_wide block")


def test_the_chromosome_list_is_printed_in_the_banner(tmp_path):
    env = _generic_env(tmp_path)
    r = _run(GENERIC, env, tmp_path)
    assert r.returncode == 0, r.stderr
    assert f"[t81-score] chroms={','.join(ROOT_CHROMS)}" in r.stdout, (
        f"the banner does not name the scored chromosomes:\n{r.stdout}")


def test_a_root_whose_manifest_omits_chroms_falls_back_to_the_npz_filenames(tmp_path):
    """Not every writer records `chroms`. The npz files on disk are the same set by construction."""
    env = _generic_env(tmp_path, in_manifest=False)
    r = _run(GENERIC, env, tmp_path)
    assert r.returncode == 0, f"exited {r.returncode}:\n{r.stdout}\n{r.stderr}"
    assert _flag(_calls(env)[0]["argv"], "--chroms").split(",") == ROOT_CHROMS
    assert "*.npz" in r.stderr, f"the log does not say where the list came from:\n{r.stderr}"


def test_a_root_missing_a_held_out_chromosome_is_refused_before_anything_is_read(tmp_path):
    """chr21 absent means the RANKED `macro` block cannot be computed at all."""
    short = [c for c in ROOT_CHROMS if c != "chr21"]
    env = _generic_env(tmp_path, root_chroms=short)
    r = _run(GENERIC, env, tmp_path)
    assert r.returncode != 0, f"a root without chr21 was accepted:\n{r.stdout}"
    assert "chr21" in r.stderr, f"the refusal does not name the missing chromosome:\n{r.stderr}"
    assert not _calls(env), "the scorer was launched anyway"


def test_a_root_sharing_no_chromosome_with_the_store_is_refused(tmp_path):
    env = _generic_env(tmp_path, root_chroms=["chrZZ"])
    r = _run(GENERIC, env, tmp_path)
    assert r.returncode != 0, f"an empty intersection was accepted:\n{r.stdout}"
    assert "none in common" in r.stderr, r.stderr
    assert not _calls(env)


def test_heldout_scope_is_untouched_and_never_consults_the_root(tmp_path):
    """The held-out branch was never broken; this is what says the fix did not move it."""
    env = _generic_env(tmp_path, SCOPE="heldout")
    r = _run(GENERIC, env, tmp_path)
    assert r.returncode == 0, f"exited {r.returncode}:\n{r.stdout}\n{r.stderr}"
    argv = _calls(env)[0]["argv"]
    assert _flag(argv, "--chroms") == HELD_OUT
    assert "--held-out-chroms" not in argv, (
        "a held-out pass that also names --held-out-chroms would claim a genome_wide block")


def test_extra_still_lands_last_so_a_hand_written_chroms_wins(tmp_path):
    """EXTRA is the sanctioned seam and stays one: argparse keeps the LAST --chroms."""
    env = _generic_env(tmp_path, EXTRA="--chroms chr20 --allow-missing")
    r = _run(GENERIC, env, tmp_path)
    assert r.returncode == 0, f"exited {r.returncode}:\n{r.stdout}\n{r.stderr}"
    argv = _calls(env)[0]["argv"]
    assert _flag(argv, "--chroms") == "chr20", f"EXTRA did not land last: {argv}"
    assert "--allow-missing" in argv


def test_the_sigma_rule_still_refuses_an_eval_pair_table(tmp_path):
    env = _generic_env(tmp_path, SIGMA=str(_sigma(tmp_path / "sigma.json", BAD_FITTED_ON)))
    r = _run(GENERIC, env, tmp_path)
    assert r.returncode == 3, f"exit {r.returncode}, not 3:\n{r.stdout}\n{r.stderr}"
    assert not _calls(env)


# ---------------------------------------------------------------------------------------------
# (B) slurm/t49_baselines_score.sh -- SIGMA and TRUTH, behind the stamp gate
# ---------------------------------------------------------------------------------------------

#: `avg` is one of the two COLLAPSED methods: generated once under `eic_19`, scored against both
#: boards. Its pred root is addressed by the GENERATION regime and its score file by the BOARD's.
STAMP = {"regime_independent": {"identical": True, "asserted_against": "eic_pilot",
                                "chrom": "chr21"}}


def _baselines_env(tmp: Path, *, method="avg", board="eic_19", panel="B_",
                   stamped=False, **over) -> dict:
    env = _stub_env(tmp)
    pred_base = tmp / ("pred_B" if panel == "B_" else "pred_V")
    gen_regime = "eic_19"           # where the collapsed pair was generated, always
    _pred_root(pred_base / method / gen_regime / panel,
               extra_manifest=STAMP if stamped else None)
    env.update({
        "REGIME": str(_regime(tmp / "configs" / f"regime.{board}.json", _store(tmp))),
        "PANEL": panel,
        "METHOD": method,
        "COLLAPSE_REGIME": gen_regime,
        "V_PRED_ROOT": str(tmp / "pred_V"),
        "B_PRED_ROOT": str(tmp / "pred_B"),
        "SCORES_ROOT": str(tmp / "scores"),
    })
    env.update(over)
    return env


def test_the_store_pass_is_unchanged(tmp_path):
    """Default TRUTH, no SIGMA: the same file name and the same flags as before this change."""
    env = _baselines_env(tmp_path, panel="V_", method="knn1")
    r = _run(BASELINES, env, tmp_path)
    assert r.returncode == 0, f"exited {r.returncode}:\n{r.stdout}\n{r.stderr}"
    argv = _calls(env)[0]["argv"]
    assert _flag(argv, "--out").endswith("/knn1/eic_19/store.V_.json"), _flag(argv, "--out")
    assert "--truth-root" not in argv and "--sigma-table" not in argv


def test_truth_challenge_writes_the_challenge_file_over_the_held_out_three(tmp_path):
    truth = _pred_root(tmp_path / "truth_challenge" / "B_", HELD_OUT.split(","))
    env = _baselines_env(tmp_path, method="knn1", TRUTH="challenge", TRUTH_ROOT=str(truth))
    r = _run(BASELINES, env, tmp_path)
    assert r.returncode == 0, f"exited {r.returncode}:\n{r.stdout}\n{r.stderr}"
    argv = _calls(env)[0]["argv"]
    assert _flag(argv, "--truth-root") == str(truth), argv
    assert _flag(argv, "--chroms") == HELD_OUT, argv
    assert "--held-out-chroms" not in argv, (
        "a challenge pass must carry no genome_wide block -- the truth root holds chr20-22 only")
    assert _flag(argv, "--out").endswith("/knn1/eic_19/challenge.B_.json"), _flag(argv, "--out")
    assert "truth=challenge" in r.stdout, f"the banner does not say the truth:\n{r.stdout}"


def test_truth_challenge_refuses_a_genome_wide_pass(tmp_path):
    truth = _pred_root(tmp_path / "truth_challenge" / "B_", HELD_OUT.split(","))
    env = _baselines_env(tmp_path, method="knn1", TRUTH="challenge", TRUTH_ROOT=str(truth),
                         SCOPE="genomewide")
    r = _run(BASELINES, env, tmp_path)
    assert r.returncode == 2, f"exit {r.returncode}, not 2:\n{r.stdout}\n{r.stderr}"
    assert not _calls(env)


def test_truth_challenge_refuses_a_truth_root_that_is_not_there(tmp_path):
    env = _baselines_env(tmp_path, method="knn1", TRUTH="challenge",
                         TRUTH_ROOT=str(tmp_path / "nowhere"))
    r = _run(BASELINES, env, tmp_path)
    assert r.returncode == 2, f"exit {r.returncode}, not 2:\n{r.stdout}\n{r.stderr}"
    assert not _calls(env)


def test_an_unknown_truth_is_refused(tmp_path):
    env = _baselines_env(tmp_path, method="knn1", TRUTH="synapse")
    r = _run(BASELINES, env, tmp_path)
    assert r.returncode == 2, f"exit {r.returncode}, not 2:\n{r.stdout}\n{r.stderr}"


def test_a_sigma_table_fitted_on_the_eval_pairs_is_refused_with_exit_3(tmp_path):
    """Rule 1, and the same exit code the generic launcher uses for it."""
    env = _baselines_env(tmp_path, method="avg-arcsinh", stamped=True,
                         SIGMA=str(_sigma(tmp_path / "sigma.json", BAD_FITTED_ON)))
    r = _run(BASELINES, env, tmp_path)
    assert r.returncode == 3, f"exit {r.returncode}, not 3:\n{r.stdout}\n{r.stderr}"
    assert "fitted_on" in r.stderr, r.stderr
    assert not _calls(env), "the scorer ran with a table fitted on the answer sheet"


def test_a_training_residual_sigma_table_reaches_the_scorer(tmp_path):
    sig = _sigma(tmp_path / "sigma.json", GOOD_FITTED_ON)
    env = _baselines_env(tmp_path, method="avg-arcsinh", stamped=True, SIGMA=str(sig))
    r = _run(BASELINES, env, tmp_path)
    assert r.returncode == 0, f"exited {r.returncode}:\n{r.stdout}\n{r.stderr}"
    assert _flag(_calls(env)[0]["argv"], "--sigma-table") == str(sig)
    assert "sigma=" in r.stdout, f"the banner does not name the table:\n{r.stdout}"


def test_a_missing_sigma_table_is_refused(tmp_path):
    env = _baselines_env(tmp_path, method="avg-arcsinh", stamped=True,
                         SIGMA=str(tmp_path / "no_such_sigma.json"))
    r = _run(BASELINES, env, tmp_path)
    assert r.returncode == 2, f"exit {r.returncode}, not 2:\n{r.stdout}\n{r.stderr}"


def test_a_collapsed_root_without_the_stamp_is_still_refused_against_the_other_board(tmp_path):
    """The gate that makes this launcher the right home for SIGMA and TRUTH must still fire."""
    env = _baselines_env(tmp_path, method="avg", board="eic_pilot", stamped=False)
    r = _run(BASELINES, env, tmp_path)
    assert r.returncode == 5, f"exit {r.returncode}, not 5:\n{r.stdout}\n{r.stderr}"
    assert "regime_independent" in r.stderr, r.stderr
    assert not _calls(env)


@pytest.mark.parametrize("truth", ["store", "challenge"])
def test_the_stamp_gate_fires_before_sigma_and_truth_are_of_any_use(tmp_path, truth):
    """Adding the two flags must not open a way around the gate, under either truth."""
    env = _baselines_env(tmp_path, method="avg-arcsinh", board="eic_pilot", stamped=False,
                         TRUTH=truth,
                         TRUTH_ROOT=str(_pred_root(tmp_path / "truth" / "B_",
                                                   HELD_OUT.split(","))),
                         SIGMA=str(_sigma(tmp_path / "sigma.json", GOOD_FITTED_ON)))
    r = _run(BASELINES, env, tmp_path)
    assert r.returncode == 5, f"exit {r.returncode}, not 5:\n{r.stdout}\n{r.stderr}"
    assert not _calls(env)


def test_the_stamped_collapsed_pair_reaches_the_other_board_with_both_new_flags(tmp_path):
    """The four blocked `eic.pilot` units, as one run: the pilot BOARD, the eic_19 ROOT, a sigma
    table and the challenge truth -- and the stamp is what licenses the address."""
    truth = _pred_root(tmp_path / "truth_challenge" / "B_", HELD_OUT.split(","))
    sig = _sigma(tmp_path / "sigma.json", GOOD_FITTED_ON)
    env = _baselines_env(tmp_path, method="avg-arcsinh", board="eic_pilot", stamped=True,
                         TRUTH="challenge", TRUTH_ROOT=str(truth), SIGMA=str(sig))
    r = _run(BASELINES, env, tmp_path)
    assert r.returncode == 0, f"exited {r.returncode}:\n{r.stdout}\n{r.stderr}"
    argv = _calls(env)[0]["argv"]
    assert _flag(argv, "--out").endswith("/avg-arcsinh/eic_pilot/challenge.B_.json"), argv
    assert "/avg-arcsinh/eic_19/B_" in _flag(argv, "--pred"), (
        f"the root must still be addressed by the GENERATION regime: {_flag(argv, '--pred')}")
    assert _flag(argv, "--sigma-table") == str(sig)
    assert _flag(argv, "--truth-root") == str(truth)
    assert "collapse asserted against" in r.stdout, (
        f"the stamp gate did not report passing:\n{r.stdout}")


# ---------------------------------------------------------------------------------------------
# (C) slurm/t81_predict_candi.sh -- dump over the chromosomes the STORE holds data for
# ---------------------------------------------------------------------------------------------
# The same defect as (A), one launcher over: with `CHROMS` unset the plan read its sizes through
# `layout.chrom_sizes_path(store)`, which is `<store>/genome/chrom_sizes.json` -- but the genome
# layer is SHARED and sits one level ABOVE the corpus, so on Fir that path names a directory that
# has never existed and every default submission died at PLAN with exit 1 (2026-09-03).
#
# The whole launcher cannot be run here: it wants a checkpoint, an arch json, a GPU and a real
# store behind `open_source`. Its chromosome program CAN, so the program is lifted out of the
# script text and run for real -- against the same fake store the scorer tests use -- with only
# `open_source` stubbed, since the track COUNT is not what is under test.

def _plan_program(script: Path = PREDICT) -> str:
    """The launcher's PLAN heredoc, lifted verbatim from the script text.

    Lifted rather than copied so it cannot drift: a rewrite of the plan is tested as it is, and a
    rename of the heredoc marker fails here loudly instead of testing a stale copy.
    """
    lines = script.read_text(encoding="utf-8").splitlines()
    starts = [i for i, ln in enumerate(lines) if ln.startswith('PLAN="$(python - ')]
    assert len(starts) == 1, f"{script.name} has {len(starts)} PLAN heredocs, expected 1"
    body = lines[starts[0] + 1:]
    end = next(i for i, ln in enumerate(body) if ln.strip() == "PYEOF")
    return "\n".join(body[:end]) + "\n"


#: Runs the lifted program with `candi.bench.harness` replaced. `open_source` there needs a real
#: StoreDataset over real h5; the plan uses it ONLY to count declared tracks, so a stub that
#: answers one pair with one target leaves the chromosome derivation entirely real.
#:
#: K16 -- and the stub PRINTS, because the real `open_source` prints. Its `[bench] N declared eval
#: pair(s), M scoreable experiment(s)` banner goes to stdout with flush=True, and the launcher
#: captures this program whole with PLAN="$( ... )". The K15 stub was silent, which is exactly why
#: every test in this section passed on a plan that no array task could read.
_PLAN_RUNNER = '''
import sys, types

sys.path.insert(0, {src!r})

_stub = types.ModuleType("candi.bench.harness")


class _Src:
    def pairs(self, kind):
        return [("T_C1", "V_C1")]

    def targets(self, pair, kind):
        return ["H3K4me3"]

    def close(self):
        pass


def _open_source(**kw):
    print("[bench] 1 declared eval pair(s), 1 scoreable experiment(s)", flush=True)
    return _Src()


_stub.open_source = _open_source
sys.modules["candi.bench.harness"] = _stub

sys.argv = ["-", {derived!r}, {panel!r}, {chroms!r}]
exec(compile(open({program!r}, encoding="utf-8").read(), "<launcher-heredoc>", "exec"),
     {{"__name__": "__main__"}})
'''


def _plan(tmp: Path, *, chroms_arg: str = "", store_chroms=STORE_CHROMS,
          corpus_chroms=ROOT_CHROMS, panel: str = "V_") -> subprocess.CompletedProcess:
    corpus = _store(tmp, store_chroms, corpus_chroms)
    derived = tmp / f"regime.eic_19.{panel}.json"
    derived.write_text(json.dumps({"store": str(corpus),
                                   "eval_pairs": [["T_C1", f"{panel}C1"]]}), encoding="utf-8")
    program = tmp / "plan_program.py"
    program.write_text(_plan_program(), encoding="utf-8")
    runner = tmp / "run_plan.py"
    runner.write_text(_PLAN_RUNNER.format(src=str(REPO / "src"), derived=str(derived),
                                          panel=panel, chroms=chroms_arg,
                                          program=str(program)), encoding="utf-8")
    return subprocess.run([sys.executable, str(runner)], capture_output=True, text=True,
                          cwd=str(tmp))


def _plan_values(r: subprocess.CompletedProcess) -> dict:
    """The plan as the launcher now reads it: by `PLAN:` marker, never by line number.

    K16. `sed -n 1p` made every field hostage to anything the library chose to print first, and
    `open_source` prints. Parsing the same way bash does keeps this helper honest about it.
    """
    return dict(ln[len("PLAN:"):].split("=", 1)
                for ln in r.stdout.splitlines() if ln.startswith("PLAN:"))


def test_the_predict_launcher_no_longer_asks_for_the_corpus_own_genome_dir():
    """`chrom_sizes_path(<corpus>)` is `<corpus>/genome/...`, which no corpus has.

    Comment lines are dropped before the check: the block above the fix EXPLAINS the helper it
    replaced, and a check reading the prose could not tell the explanation from the code.
    """
    code = "\n".join(ln for ln in PREDICT.read_text(encoding="utf-8").splitlines()
                     if not ln.strip().startswith("#"))
    assert "chrom_sizes_path(" not in code, (
        "the plan still resolves its sizes through layout.chrom_sizes_path, which takes "
        "CANDI_STORE and is handed the CORPUS -- the path it builds does not exist")
    assert "corpus_genome_dir(" in code, (
        "the plan must reach the shared genome layer through layout.corpus_genome_dir")
    assert "chrom_sizes.json" in code, "it must still read the shared sizes file"
    assert "chroms ({len(chroms)})" in PREDICT.read_text(encoding="utf-8"), (
        "the `[t81-pred]   chroms (N): ...` line is how an operator checks the plan before the "
        "dump spends 14 GPU-h; it must survive")


def test_the_default_chromosomes_are_the_ones_the_corpus_holds_data_for(tmp_path):
    """24 declared by the shared layer, 23 held by the corpus, and the dump gets the 23.

    A chromosome the store has no data for cannot be dumped at all, and `bench.dump` finds that
    out per track rather than up front -- so leaving chrY in costs the walltime before it fails.
    """
    r = _plan(tmp_path)
    assert r.returncode == 0, f"exited {r.returncode}:\n{r.stdout}\n{r.stderr}"
    planned = _plan_values(r)["chroms"].split(",")
    assert planned == ROOT_CHROMS, f"the plan chose {planned}"
    assert "chrY" not in planned, (
        "the shared genome layer's chrY reached the dump; the corpus holds no chrY track")
    assert f"chroms ({len(ROOT_CHROMS)})" in r.stderr, (
        f"the plan does not print the list it chose:\n{r.stderr}")
    assert "chrY" in r.stderr, f"the dropped chromosome is not reported anywhere:\n{r.stderr}"


def test_an_explicit_chroms_is_passed_through_untouched(tmp_path):
    """The operator override is not the bug and must not acquire the intersection.

    `EXTRA="--chroms chr21"`-style hand-holding is how the two jobs that survived the defect
    survived it, and a run pinned to one chromosome must stay pinned to it.
    """
    r = _plan(tmp_path, chroms_arg="chr21,chr22")
    assert r.returncode == 0, f"exited {r.returncode}:\n{r.stdout}\n{r.stderr}"
    assert _plan_values(r)["chroms"] == "chr21,chr22", r.stdout
    assert "dropped from the default" not in r.stderr, (
        "the explicit path must not report a drop -- it did not derive a default")


# ---------------------------------------------------------------------------------------------
# (C2) the plan is the only thing on stdout, and bash reads it by marker (K16)
# ---------------------------------------------------------------------------------------------
# `harness.open_source` prints `[bench] N declared eval pair(s), M scoreable experiment(s)` on
# STDOUT with flush=True, and the launcher captures the plan whole with PLAN="$( ... )". So the
# banner became line 1 and `sed -n 1p` handed it to CHROMS: the sharded path exited 6 having
# counted 2 chromosomes in the sentence, the unsharded one passed the sentence to --chroms. All 92
# array tasks of 2026-09-03 died of it (jobs 57949917/57949921/57949968/57949996).
#
# `open_source` is not changed -- other callers and tests read that banner. The plan is what has to
# survive it, so the two checks are: the library keeps its stdout OUT of the plan, and bash reads
# fields by name rather than by position.

def test_library_chatter_stays_out_of_the_plan_and_lands_on_stderr(tmp_path):
    """The stub prints the banner the real `open_source` prints; only PLAN: lines may be captured."""
    r = _plan(tmp_path)
    assert r.returncode == 0, f"exited {r.returncode}:\n{r.stdout}\n{r.stderr}"
    stray = [ln for ln in r.stdout.splitlines() if ln and not ln.startswith("PLAN:")]
    assert not stray, f"the plan carries lines bash would misread as fields: {stray}"
    assert "[bench]" in r.stderr, (
        "the banner did not reach the job .err either -- redirecting it must not silence it")


def test_every_plan_field_is_named_so_a_stray_line_cannot_shift_one(tmp_path):
    """Three fields, each behind its own marker, in any order. Position is not a contract."""
    r = _plan(tmp_path)
    got = _plan_values(r)
    assert set(got) == {"chroms", "n_tracks", "want_time"}, got
    assert got["chroms"].split(",") == ROOT_CHROMS
    assert got["n_tracks"] == "1", got                     # the stub declares one target
    assert got["want_time"] == "00:18:00", got             # 0.2363 * 1 * 1.3 h
    # Comments dropped first: the block above the fix NAMES the `sed -n 1p` it replaced, and a
    # check reading the prose could not tell the explanation from the code.
    code = "\n".join(ln for ln in PREDICT.read_text(encoding="utf-8").splitlines()
                     if not ln.strip().startswith("#"))
    assert "sed -n 1p" not in code and "sed -n 2p" not in code and "sed -n 3p" not in code, (
        "the launcher still reads the plan by line number, so any library print shifts a field")


def test_a_corpus_sharing_no_chromosome_with_the_genome_layer_is_refused(tmp_path):
    """Empty is not "dump nothing", it is a store that does not go with this genome layer."""
    r = _plan(tmp_path, corpus_chroms=["chrZ1", "chrZ2"])
    assert r.returncode != 0, f"it planned a dump of nothing:\n{r.stdout}"
    assert "REFUSING" in (r.stdout + r.stderr), f"{r.stdout}\n{r.stderr}"


# ---------------------------------------------------------------------------------------------
# (D) slurm/t81_predict_candi.sh -- the CPU, chromosome-sharded route (K15)
# ---------------------------------------------------------------------------------------------
# Every CANDI predict job sat PENDING on the GPU partition with no start estimate while the CPU
# partitions started in minutes, and a whole 45-track genome-wide pass does not fit one CPU job
# (10.6 GPU-h measured -> 71-143 CPU-h estimated, over the 60 h band). So DEVICE=cpu SHARD_CHROMS=1
# splits the pass into one array task per chromosome, into ONE root -- which puts the manifest at
# risk, because `bench.dump` builds it from the chromosomes THIS invocation was given and the
# scorer reads `manifest.chroms` and scores exactly those.
#
# The whole launcher runs here, against the stub `python` of section (A/B): the real plan, the real
# shard mapping, the real B_ guard, the real merge. Only the venv, the kit, the checkpoint and
# `bench.dump` itself are fakes -- and the fake dump writes what a real one writes, in the order it
# writes it, so the one-chromosome manifest a shard leaves behind is real behaviour, not a prop.

#: What the fake dump declares. Two, so the merge's per-track completeness check has something to
#: be wrong about.
DUMP_TRACKS = ["T_C1__V_C1__H3K4me3", "T_C1__V_C1__H3K27ac"]


def _predict_env(tmp: Path, *, panel: str = "V_", corpus_chroms=ROOT_CHROMS, **over) -> dict:
    """The predict launcher against a fake kit, a fake checkpoint and the shared fake store."""
    env = _stub_env(tmp)
    kit = tmp / "kit"
    corpus = _store(tmp, STORE_CHROMS, corpus_chroms)
    (kit / "configs").mkdir(parents=True, exist_ok=True)
    # The SHIPPED regime declares both panels; `declare_eval_pairs.py split` is what narrows it,
    # and the launcher refuses a derived regime that targets the other one.
    (kit / "configs" / "regime.eic_19.json").write_text(json.dumps({
        "store": str(corpus), "eval_pairs": [["T_C1", "V_C1"], ["T_C2", "B_C2"]],
        "eval_chroms": HELD_OUT.split(",")}), encoding="utf-8")
    ck = tmp / "ckpts"
    ck.mkdir(parents=True, exist_ok=True)
    (ck / "t81_eic_19_s0.best.ckpt").write_bytes(b"")
    (ck / "t81_eic_19_s0.json").write_text("{}", encoding="utf-8")
    env.update({
        "REGIME_NAME": "eic_19",
        "PANEL": panel,
        "SEED": "0",
        "CKPT_DIR": str(ck),
        "OUT": str(tmp / "pred" / panel),
        "FAKE_HARNESS": "1",
        "FAKE_DUMP_TRACKS": json.dumps(DUMP_TRACKS),
        "SLURM_CPUS_PER_TASK": "8",
    })
    env.update(over)
    return env


def _shard(tmp: Path, env: dict, index: int, *, count: int = len(ROOT_CHROMS)):
    e = dict(env)
    e.update({"SHARD_CHROMS": "1", "SLURM_ARRAY_TASK_ID": str(index),
              "SLURM_ARRAY_TASK_COUNT": str(count)})
    return _run(PREDICT, e, tmp), e


def test_the_default_predict_path_names_no_device_and_writes_its_own_manifest(tmp_path):
    """Byte-for-byte the argv the GPU route always sent: no --device, no --notes, all 23 at once.

    `bench.dump` picks cuda-if-available on its own, and a launcher that started naming a device
    would pin every unsharded run to whatever the last CPU submit line happened to leave exported.
    """
    env = _predict_env(tmp_path)
    r = _run(PREDICT, env, tmp_path)
    assert r.returncode == 0, f"exited {r.returncode}:\n{r.stdout}\n{r.stderr}"
    argv = _calls(env)[0]["argv"]
    assert _flag(argv, "--chroms").split(",") == ROOT_CHROMS, argv
    assert "--device" not in argv, f"the default path pinned a device: {argv}"
    assert "--notes" not in argv, f"the default path started writing notes: {argv}"
    out = Path(env["OUT"])
    assert (out / "manifest.json").is_file(), "the unsharded root lost its manifest"
    assert not list(out.glob(".shard_manifest.*")), "the unsharded path built shard machinery"
    assert json.loads((out / "manifest.json").read_text())["chroms"] == ROOT_CHROMS


def test_the_unsharded_path_reads_its_plan_past_the_libraries_banner(tmp_path):
    """K16, end to end through the real bash: the whole launcher, with a NOISY `open_source`.

    The stub now prints the `[bench] ...` banner the real one prints. Before the fix that banner
    was line 1 of the plan and CHROMS became the sentence, so `bench.dump` was handed
    `--chroms "[bench] 1 declared eval pair(s), 2 scoreable experiment(s)"`. N_TRACKS and
    WANT_TIME were the two lines after it, shifted by one.
    """
    env = _predict_env(tmp_path)
    r = _run(PREDICT, env, tmp_path)
    assert r.returncode == 0, f"exited {r.returncode}:\n{r.stdout}\n{r.stderr}"
    argv = _calls(env)[0]["argv"]
    assert _flag(argv, "--chroms").split(",") == ROOT_CHROMS, argv
    assert "[bench]" not in _flag(argv, "--chroms"), (
        f"the library banner reached --chroms: {argv}")
    # 2 declared tracks -> 0.2363 * 2 * 1.3 h. Both fields, not just the one that crashed.
    assert "[t81-pred] tracks=2 recommended --time=00:36:00" in r.stdout, r.stdout
    assert "[bench]" in r.stderr, "the banner must still be readable in the job .err"


@pytest.mark.parametrize("index", [0, 5, 22])
def test_a_shard_reads_its_plan_past_the_libraries_banner(tmp_path, index):
    """The path that actually died: 12 shards, every one exiting 6 on a 2-word chromosome list.

    `IFS=, read -a` over `[bench] 1 declared eval pair(s), 2 scoreable experiment(s)` yields two
    entries, so the array-length guard refused an array of 23 against a plan of 2.
    """
    env = _predict_env(tmp_path, DEVICE="cpu")
    r, env = _shard(tmp_path, env, index)
    assert r.returncode == 0, f"exited {r.returncode}:\n{r.stdout}\n{r.stderr}"
    assert _flag(_calls(env)[0]["argv"], "--chroms") == ROOT_CHROMS[index]
    assert f"shard {index}/{len(ROOT_CHROMS)} -> {ROOT_CHROMS[index]}" in r.stdout, r.stdout


def test_device_reaches_the_dump_and_sets_the_thread_count_from_the_allocation(tmp_path):
    """`--device` is a real flag on `bench.dump`; OMP_NUM_THREADS is what torch reads at import.

    Unset, torch takes a thread per core on the NODE rather than per core in the ALLOCATION, which
    on a shared cpubase node is oversubscription.
    """
    env = _predict_env(tmp_path, DEVICE="cpu")
    r = _run(PREDICT, env, tmp_path)
    assert r.returncode == 0, f"exited {r.returncode}:\n{r.stdout}\n{r.stderr}"
    call = _calls(env)[0]
    assert _flag(call["argv"], "--device") == "cpu", call["argv"]
    assert _flag(call["argv"], "--notes") == "device=cpu", call["argv"]
    assert call["env"].get("OMP_NUM_THREADS") == "8", (
        f"OMP_NUM_THREADS is {call['env'].get('OMP_NUM_THREADS')}, not SLURM_CPUS_PER_TASK")
    assert call["env"].get("MKL_NUM_THREADS") == "8"


def test_a_gpu_run_never_pins_the_thread_count(tmp_path):
    """The knob is the CPU route only -- 4 OMP threads on the GPU path is a change nobody asked."""
    env = _predict_env(tmp_path)
    r = _run(PREDICT, env, tmp_path)
    assert r.returncode == 0, r.stderr
    assert "OMP_NUM_THREADS" not in _calls(env)[0]["env"]


@pytest.mark.parametrize("index", [0, 7, 22])
def test_a_shard_predicts_exactly_the_chromosome_its_index_names(tmp_path, index):
    """Index -> chromosome by POSITION in the planned list, so shard i and an unsharded run agree."""
    env = _predict_env(tmp_path, DEVICE="cpu")
    r, env = _shard(tmp_path, env, index)
    assert r.returncode == 0, f"exited {r.returncode}:\n{r.stdout}\n{r.stderr}"
    argv = _calls(env)[0]["argv"]
    assert _flag(argv, "--chroms") == ROOT_CHROMS[index], argv
    assert _flag(argv, "--out") == env["OUT"], "a shard must write into the SAME root"
    assert f"shard {index}/{len(ROOT_CHROMS)} -> {ROOT_CHROMS[index]}" in r.stdout, r.stdout


def test_a_shard_leaves_no_manifest_in_the_root(tmp_path):
    """A one-chromosome manifest left in the root is a genome-wide score over one chromosome.

    `slurm/t81_score_external.sh` reads `manifest.chroms` and scores exactly those, and the B_
    once-guard reads the manifest as proof the panel was spent. Neither may see a partial array.
    """
    env = _predict_env(tmp_path, DEVICE="cpu")
    r, env = _shard(tmp_path, env, 3)
    assert r.returncode == 0, f"exited {r.returncode}:\n{r.stdout}\n{r.stderr}"
    out = Path(env["OUT"])
    assert not (out / "manifest.json").exists(), "a shard left its one-chromosome manifest behind"
    kept = out / f".shard_manifest.{ROOT_CHROMS[3]}.json"
    assert kept.is_file(), f"the shard manifest was not kept as a template: {list(out.iterdir())}"
    assert (out / DUMP_TRACKS[0] / f"{ROOT_CHROMS[3]}.npz").is_file()


def test_the_sharded_root_holds_no_directory_that_is_not_a_declared_track(tmp_path):
    """`bench.external` lists every DIRECTORY under a prediction root and refuses the pass if one
    names no declared track. So the shard bookkeeping has to be FILES: a `.shards/` holding pen
    would have cost the whole score pass, hours in, for a root that was otherwise perfect.
    """
    env = _predict_env(tmp_path, DEVICE="cpu")
    for i in (0, 1):
        r, _ = _shard(tmp_path, env, i)
        assert r.returncode == 0, f"shard {i} exited {r.returncode}:\n{r.stdout}\n{r.stderr}"
    out = Path(env["OUT"])
    _fill_root(out, ROOT_CHROMS[2:])
    r, _ = _merged(tmp_path, env)
    assert r.returncode == 0, f"the merge exited {r.returncode}:\n{r.stdout}\n{r.stderr}"
    dirs = sorted(p.name for p in out.iterdir() if p.is_dir())
    assert dirs == sorted(DUMP_TRACKS), (
        f"{dirs} — bench.external refuses a root holding a directory that is not a declared track")


def test_an_array_shorter_than_the_planned_list_is_refused_before_any_dump(tmp_path):
    """23 chromosomes and 3 tasks leaves 20 unpredicted, found out only at the merge hours later."""
    env = _predict_env(tmp_path, DEVICE="cpu")
    r, env = _shard(tmp_path, env, 1, count=3)
    assert r.returncode == 6, f"exit {r.returncode}, not 6:\n{r.stdout}\n{r.stderr}"
    assert "3 tasks" in r.stderr and str(len(ROOT_CHROMS)) in r.stderr, r.stderr
    assert not _calls(env), "the dump ran anyway"


def test_an_index_past_the_end_of_the_planned_list_is_refused(tmp_path):
    env = _predict_env(tmp_path, DEVICE="cpu")
    r, env = _shard(tmp_path, env, len(ROOT_CHROMS), count=len(ROOT_CHROMS))
    assert r.returncode == 6, f"exit {r.returncode}, not 6:\n{r.stdout}\n{r.stderr}"
    assert not _calls(env)


def test_sharded_without_an_array_is_refused(tmp_path):
    env = _predict_env(tmp_path, DEVICE="cpu", SHARD_CHROMS="1")
    r = _run(PREDICT, env, tmp_path)
    assert r.returncode == 6, f"exit {r.returncode}, not 6:\n{r.stdout}\n{r.stderr}"
    assert "--array" in r.stderr, r.stderr
    assert not _calls(env)


def _fill_root(out: Path, chroms) -> None:
    """Stand in for the shards this test does not pay to run: the npz they would have written."""
    for t in DUMP_TRACKS:
        (out / t).mkdir(parents=True, exist_ok=True)
        for c in chroms:
            (out / t / f"{c}.npz").write_bytes(b"")


def _merged(tmp: Path, env: dict, *, array: str = "57999001"):
    e = dict(env)
    e.update({"SHARD_CHROMS": "1", "SHARD_MERGE": "1", "SHARD_ARRAY": array})
    e.pop("SLURM_ARRAY_TASK_ID", None)
    return _run(PREDICT, e, tmp), e


def test_the_merge_step_writes_one_manifest_naming_every_planned_chromosome(tmp_path):
    """The point of the whole exercise: 23 shards, then ONE manifest the scorer can read.

    Two shards run for real and the other 21 are filled in, because 23 real launcher runs buys
    nothing this does not already prove -- the merge reads the npz on disk, not the shard count.
    """
    env = _predict_env(tmp_path, DEVICE="cpu")
    for i in (0, 1):
        r, _ = _shard(tmp_path, env, i)
        assert r.returncode == 0, f"shard {i} exited {r.returncode}:\n{r.stdout}\n{r.stderr}"
    out = Path(env["OUT"])
    _fill_root(out, ROOT_CHROMS[2:])
    r, _ = _merged(tmp_path, env)
    assert r.returncode == 0, f"the merge exited {r.returncode}:\n{r.stdout}\n{r.stderr}"
    man = json.loads((out / "manifest.json").read_text())
    assert man["chroms"] == ROOT_CHROMS, man["chroms"]
    assert man["declared_tracks"] == DUMP_TRACKS, "the identity half must come from a real shard"
    assert man["ckpt_sha256"] == "ckptsha", "the merge retyped the provenance instead of copying it"
    assert man["notes"] == "device=cpu; sharded: true; array=57999001; shards=2", man["notes"]


def test_the_merge_step_refuses_a_root_missing_a_chromosome(tmp_path):
    """A manifest naming 23 over a root holding 22 is a panel scored with a hole in it (D2)."""
    env = _predict_env(tmp_path, DEVICE="cpu")
    r, _ = _shard(tmp_path, env, 0)
    assert r.returncode == 0, r.stderr
    out = Path(env["OUT"])
    _fill_root(out, [c for c in ROOT_CHROMS[1:] if c != "chr17"])
    r, _ = _merged(tmp_path, env)
    assert r.returncode == 5, f"exit {r.returncode}, not 5:\n{r.stdout}\n{r.stderr}"
    assert "chr17" in r.stderr, r.stderr
    assert not (out / "manifest.json").exists(), "an incomplete root was given a manifest"


def test_the_merge_step_refuses_a_root_no_shard_ever_finished(tmp_path):
    env = _predict_env(tmp_path, DEVICE="cpu")
    r, _ = _merged(tmp_path, env)
    assert r.returncode == 5, f"exit {r.returncode}, not 5:\n{r.stdout}\n{r.stderr}"
    assert ".shard_manifest" in r.stderr, r.stderr


def test_a_merge_without_the_array_it_assembles_is_refused(tmp_path):
    env = _predict_env(tmp_path, DEVICE="cpu", SHARD_CHROMS="1", SHARD_MERGE="1")
    r = _run(PREDICT, env, tmp_path)
    assert r.returncode == 6, f"exit {r.returncode}, not 6:\n{r.stdout}\n{r.stderr}"
    assert "SHARD_ARRAY" in r.stderr, r.stderr


def test_shard_array_is_refused_on_anything_that_predicts(tmp_path):
    """It presents another array's B_ marker, so it may only ride on a step that writes no bytes."""
    env = _predict_env(tmp_path, DEVICE="cpu", SHARD_ARRAY="57999001")
    r, env = _shard(tmp_path, env, 0)
    assert r.returncode == 6, f"exit {r.returncode}, not 6:\n{r.stdout}\n{r.stderr}"
    assert not _calls(env)


# --- the B_ once-guard, under sharding -------------------------------------------------------

def _b_shard(tmp: Path, env: dict, index: int, array: str):
    e = dict(env)
    e.update({"SHARD_CHROMS": "1", "SLURM_ARRAY_TASK_ID": str(index),
              "SLURM_ARRAY_TASK_COUNT": str(len(ROOT_CHROMS)),
              "SLURM_ARRAY_JOB_ID": array, "B_ONCE": "1"})
    return _run(PREDICT, e, tmp), e


def test_the_sharded_b_guard_lets_this_arrays_siblings_through_and_refuses_a_later_one(tmp_path):
    """23 tasks of ONE array must all predict; a second submission must not.

    The marker is what tells them apart -- every task of one array writes the same
    `.b_once.<array job id>` name -- and it is what keeps the sharded pass from having to leave a
    manifest in the root, which is how a half-finished array would read as a spent panel.
    """
    env = _predict_env(tmp_path, panel="B_", DEVICE="cpu")
    first, _ = _b_shard(tmp_path, env, 0, "57999100")
    assert first.returncode == 0, f"exited {first.returncode}:\n{first.stdout}\n{first.stderr}"
    out = Path(env["OUT"])
    assert (out / ".b_once.57999100").exists(), "the array never claimed the root"

    sibling, senv = _b_shard(tmp_path, env, 1, "57999100")
    assert sibling.returncode == 0, (
        f"a sibling of the claiming array was refused:\n{sibling.stdout}\n{sibling.stderr}")
    assert _flag(_calls(senv)[-1]["argv"], "--chroms") == ROOT_CHROMS[1]

    later, _ = _b_shard(tmp_path, env, 2, "57999200")
    assert later.returncode == 4, f"exit {later.returncode}, not 4:\n{later.stdout}\n{later.stderr}"
    assert "REFUSING" in later.stderr or "already holds a B_ pass" in later.stderr, later.stderr


def test_the_b_merge_step_is_spared_by_the_marker_of_the_array_it_assembles(tmp_path):
    """The merge writes no prediction, so it may present the claiming array's marker -- and must.

    Without this the once-only pass could never be finished: the shards leave the root with no
    manifest on purpose, and the step that writes one is a different job.
    """
    env = _predict_env(tmp_path, panel="B_", DEVICE="cpu")
    r, _ = _b_shard(tmp_path, env, 0, "57999400")
    assert r.returncode == 0, f"exited {r.returncode}:\n{r.stdout}\n{r.stderr}"
    out = Path(env["OUT"])
    _fill_root(out, ROOT_CHROMS[1:])
    merged, _ = _merged(tmp_path, dict(env, B_ONCE="1"), array="57999400")
    assert merged.returncode == 0, f"exited {merged.returncode}:\n{merged.stdout}\n{merged.stderr}"
    assert json.loads((out / "manifest.json").read_text())["chroms"] == ROOT_CHROMS

    # And a merge naming an array that never claimed this root is still a second B_ claim.
    (out / "manifest.json").unlink()
    stranger, _ = _merged(tmp_path, dict(env, B_ONCE="1"), array="57999500")
    assert stranger.returncode == 4, (
        f"exit {stranger.returncode}, not 4:\n{stranger.stdout}\n{stranger.stderr}")


def test_a_b_shard_without_b_once_is_still_refused(tmp_path):
    env = _predict_env(tmp_path, panel="B_", DEVICE="cpu")
    e = dict(env)
    e.update({"SHARD_CHROMS": "1", "SLURM_ARRAY_TASK_ID": "0",
              "SLURM_ARRAY_TASK_COUNT": str(len(ROOT_CHROMS)), "SLURM_ARRAY_JOB_ID": "57999300"})
    r = _run(PREDICT, e, tmp_path)
    assert r.returncode == 4, f"exit {r.returncode}, not 4:\n{r.stdout}\n{r.stderr}"
    assert not Path(e["OUT"]).exists() or not list(Path(e["OUT"]).glob(".b_once.*")), (
        "a refused submission claimed the root")


def test_the_unsharded_b_guard_still_refuses_a_root_that_carries_a_manifest(tmp_path):
    """The default route is untouched: a manifest in the root means B_ was spent, full stop."""
    env = _predict_env(tmp_path, panel="B_", DEVICE="")
    out = Path(env["OUT"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text("{}", encoding="utf-8")
    r = _run(PREDICT, dict(env, B_ONCE="1"), tmp_path)
    assert r.returncode == 4, f"exit {r.returncode}, not 4:\n{r.stdout}\n{r.stderr}"
    assert not _calls(env)
