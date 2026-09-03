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


if argv and argv[0] == "-":
    # `python - a b c <<EOF` -- the launchers' inline programs. Run them, do not fake them.
    src = sys.stdin.read()
    sys.argv = list(argv)
    exec(compile(src, "<launcher-heredoc>", "exec"), {"__name__": "__main__"})
    sys.exit(0)

if argv[:2] == ["-m", "candi.bench.external"]:
    d = Path(os.environ["SCORE_RECORD_DIR"])
    d.mkdir(parents=True, exist_ok=True)
    n = len(list(d.glob("*.json")))
    (d / ("%03d.json" % n)).write_text(json.dumps({"argv": argv[2:], "env": dict(os.environ)}))
    out = opt("--out")
    if out:                                   # else the launcher's own rc=5 guard fires
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text("{}")
    sys.exit(0)

if argv and Path(argv[0]).name == "declare_eval_pairs.py":
    Path(opt("--out")).write_text(json.dumps({"panel": opt("--panel"), "eval_pairs": []}))
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
                 "PRED", "OUT", "CRPS_APPROX", "SLURM_ARRAY_TASK_ID"):
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


_stub.open_source = lambda **kw: _Src()
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
    planned = r.stdout.splitlines()[0].split(",")
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
    assert r.stdout.splitlines()[0] == "chr21,chr22", r.stdout
    assert "dropped from the default" not in r.stderr, (
        "the explicit path must not report a drop -- it did not derive a default")


def test_a_corpus_sharing_no_chromosome_with_the_genome_layer_is_refused(tmp_path):
    """Empty is not "dump nothing", it is a store that does not go with this genome layer."""
    r = _plan(tmp_path, corpus_chroms=["chrZ1", "chrZ2"])
    assert r.returncode != 0, f"it planned a dump of nothing:\n{r.stdout}"
    assert "REFUSING" in (r.stdout + r.stderr), f"{r.stdout}\n{r.stderr}"
