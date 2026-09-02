"""t81 — the eDICE and ChromImpute launchers, checked as text.

These launchers only ever run on Fir, against a store and a jar that do not exist on a laptop, so
nothing here executes one. What a test CAN assert is the part that goes wrong silently: which root
a default points at, which panel a predict pass opens, and whether a score pass would accept a σ
table fit on the wrong residuals. Every one of those is a string in the file.

Three failures are being prevented, and all three have happened once already in this tree:

* a launcher defaulting to a RETIRED checkout (`CANDII_t78_code`) or a purged scratch path, so a
  board run silently uses code or a jar nobody pinned;
* a predict pass reading the shipped 38-pair regime, which opens `B_` — BENCHMARK_DESIGN.md §5
  rules `B_` is touched once, at the very end, and a second pass leaves nothing in the record to
  say it happened;
* a score pass accepting a σ table squared off `V_` eval-pair residuals, which Rule 1 forbids and
  §12.2 declared VOID. The old escape hatch was an env var (`SIGMA_RULE1_OVERRIDE`); it is gone,
  and this file is what keeps it gone.

`bash -n` is included because these scripts carry generated heredocs and a here-doc that does not
close is not visible by eye.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
EDICE = REPO / "competitors" / "edice" / "slurm"
CHROMIMPUTE = REPO / "competitors" / "chromimpute" / "slurm"

#: The pinned Fir roots. A default that is not one of these is a path nobody agreed to.
KIT = "/project/def-maxwl/mforooz/CANDII_main"
JAR = "/project/def-maxwl/mforooz/tools/ChromImpute.jar"
PRED_V = "/scratch/mforooz/t81_pred"
PRED_B = "/project/def-maxwl/mforooz/t81_pred_B"
SIGMA_ROOT = "/project/def-maxwl/mforooz/t81_sigma"
SCORES_ROOT = "/project/def-maxwl/mforooz/t81_scores"
CKPT_ROOT = "/project/def-maxwl/mforooz/t81_checkpoints"

#: The one prefix a σ table must carry. Pinned across every method's score stage.
SIGMA_FITTED_ON_PREFIX = "training-residuals:"

#: Retired names. Each was a live default once; each now names something that does not exist, or
#: exists and is wrong.
BANNED = (
    "CANDII_t78_code",        # a retired checkout, parked on a branch nobody tracks
    "SIGMA_RULE1_OVERRIDE",   # the escape hatch out of the Rule 1 refusal
    "fit_sigma.py",           # the V_-residual σ fitter; every table it wrote is void
    "t51_chromimpute/tool",   # the jar on scratch, which is purged
)

#: Scripts that submit or run a PREDICTION pass, and therefore choose a panel.
PREDICT_LAUNCHERS = (EDICE / "eic_score.sh", CHROMIMPUTE / "submit.sh")
#: Scripts that hand a σ table to `candi.bench.external`.
SCORE_LAUNCHERS = (EDICE / "eic_score.sh", CHROMIMPUTE / "score.sh")
#: Scripts that FIT a σ table.
SIGMA_LAUNCHERS = (EDICE / "sigma.sh", CHROMIMPUTE / "sigma.sh")


def _scripts() -> list[Path]:
    return sorted([*EDICE.glob("*.sh"), *CHROMIMPUTE.glob("*.sh")])


def _text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _code(p: Path) -> str:
    """The script with its comment lines dropped.

    These files carry long headers that quote the very flags and paths the tests below forbid --
    saying WHY something is not passed is half of what the header is for. A check that reads the
    comments cannot tell an explanation from an instruction.
    """
    return "\n".join(ln for ln in _text(p).splitlines() if not ln.strip().startswith("#"))


#: Launchers that import nothing from `candi`, each for a stated reason.
NO_CANDI = {
    "roadmap_gate.sh": "runs the reference Roadmap loop, which imports only competitors/edice",
    "example_check.sh": "runs the vendor jar on the vendor's example data; the wig check is stdlib",
}


def test_there_are_launchers_to_check():
    """A glob that silently matched nothing would make every test below vacuously pass."""
    names = {p.name for p in _scripts()}
    assert names >= {"eic_train.sh", "eic_score.sh", "roadmap_gate.sh", "sigma.sh",
                     "stage.sh", "submit.sh", "score.sh", "gw_probe.sh", "example_check.sh"}
    assert len(_scripts()) >= 9


@pytest.mark.skipif(shutil.which("bash") is None, reason="no bash on this host")
@pytest.mark.parametrize("script", _scripts(), ids=lambda p: f"{p.parent.parent.name}/{p.name}")
def test_every_launcher_parses(script: Path):
    r = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert r.returncode == 0, f"{script.name} does not parse:\n{r.stderr}"


@pytest.mark.parametrize("script", _scripts(), ids=lambda p: f"{p.parent.parent.name}/{p.name}")
def test_no_retired_name_survives(script: Path):
    text = _text(script)
    hits = [b for b in BANNED if b in text]
    assert not hits, (
        f"{script.name} still names {hits}. Each of those points at a retired checkout, a purged "
        f"path, or the Rule 1 escape hatch.")


@pytest.mark.parametrize("script", _scripts(), ids=lambda p: f"{p.parent.parent.name}/{p.name}")
def test_a_repo_default_is_the_pinned_checkout(script: Path):
    """`REPO`/`CI_REPO` selects which SCRIPTS run. There is one clone the programme uses."""
    text = _text(script)
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if ("REPO=${REPO:-" in stripped or "REPO=${CI_REPO:-" in stripped
                or 'CI_REPO:=' in stripped):
            assert KIT in stripped, f"{script.name}: {stripped!r} does not default to {KIT}"


@pytest.mark.parametrize("script", _scripts(), ids=lambda p: f"{p.parent.parent.name}/{p.name}")
def test_every_job_puts_candi_on_the_path_itself(script: Path):
    """The Fir venv's editable install resolves `candi` to a checkout that does not carry it.

    Two gates are exempt, and `NO_CANDI` says why for each: they run the reference eDICE loop and
    the vendor jar, neither of which imports anything from `candi`.
    """
    text = _text(script)
    if script.name in NO_CANDI:
        pytest.skip(f"{script.name} {NO_CANDI[script.name]}")
    assert "/src" in text and "PYTHONPATH" in text, (
        f"{script.name} never puts a checkout's src on PYTHONPATH, so `import candi` would come "
        f"from whatever the venv's .pth happens to name")


# ---------------------------------------------------------------------------------------------
# the panel guard — §5's "B_ is touched once"
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("script", PREDICT_LAUNCHERS, ids=lambda p: p.parent.parent.name)
def test_a_predict_pass_derives_a_single_panel_regime(script: Path):
    text = _code(script)
    assert "declare_eval_pairs.py" in text and "--panel" in text, (
        f"{script.name} does not derive a single-panel regime, so a predict pass would read the "
        f"shipped 38-pair config and open B_")
    assert "split" in text


@pytest.mark.parametrize("script", PREDICT_LAUNCHERS, ids=lambda p: p.parent.parent.name)
def test_the_b_panel_needs_the_once_only_verb(script: Path):
    text = _code(script)
    assert "B_ONCE" in text, f"{script.name} lets PANEL=B_ run without the once-only verb"


@pytest.mark.parametrize("script", PREDICT_LAUNCHERS, ids=lambda p: p.parent.parent.name)
def test_an_existing_b_root_is_refused_with_exit_4(script: Path):
    """B_ONCE=1 does not lift this: a root that already carries a manifest proves it is not once."""
    text = _code(script)
    assert "manifest.json" in text and "exit 4" in text, (
        f"{script.name} would overwrite an existing B_ prediction root instead of refusing")


@pytest.mark.parametrize("script", PREDICT_LAUNCHERS, ids=lambda p: p.parent.parent.name)
def test_the_panel_roots_are_the_pinned_ones(script: Path):
    text = _text(script)
    assert f"{PRED_V}/" in text, f"{script.name} does not default V_ predictions to {PRED_V}"
    assert f"{PRED_B}/" in text, f"{script.name} does not default B_ predictions to {PRED_B}"


def test_b_predictions_land_on_project_and_v_on_scratch():
    """B_ is written once and must outlive a scratch purge; V_ is re-runnable and need not."""
    assert PRED_B.startswith("/project/")
    assert PRED_V.startswith("/scratch/")


# ---------------------------------------------------------------------------------------------
# the Rule 1 σ guard
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("script", SCORE_LAUNCHERS, ids=lambda p: p.parent.parent.name)
def test_a_score_pass_checks_the_sigma_provenance(script: Path):
    text = _code(script)
    assert SIGMA_FITTED_ON_PREFIX in text, (
        f"{script.name} does not check the σ table's `fitted_on` prefix, so a table fit on V_ "
        f"eval-pair residuals would score a CRPS tier and look valid")
    assert "fitted_on" in text
    assert "exit 3" in text or "SystemExit(3)" in text, (
        f"{script.name} must refuse a wrong σ table with exit 3, not a warning")


@pytest.mark.parametrize("script", SCORE_LAUNCHERS, ids=lambda p: p.parent.parent.name)
def test_a_score_pass_writes_to_the_pinned_scores_path(script: Path):
    text = _text(script)
    assert SCORES_ROOT in text, f"{script.name} does not default its output under {SCORES_ROOT}"
    assert ".$PANEL.json" in text or '${PANEL}.json' in text or "$TRUTH.$PANEL.json" in text, (
        f"{script.name}'s scores path does not carry the truth and panel, so two views would "
        f"overwrite each other")


@pytest.mark.parametrize("script", SCORE_LAUNCHERS, ids=lambda p: p.parent.parent.name)
def test_a_score_pass_defaults_to_the_pinned_sigma_table(script: Path):
    assert f"{SIGMA_ROOT}/" in _text(script)


def test_only_edice_aggregates_genome_wide():
    """§4 prints eDICE's `genome-wide` cell and BLANKS ChromImpute's, and a blank is not computed."""
    assert "--held-out-chroms" in _code(EDICE / "eic_score.sh")
    assert "--held-out-chroms" not in _code(CHROMIMPUTE / "score.sh")


# ---------------------------------------------------------------------------------------------
# the σ stage
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("script", SIGMA_LAUNCHERS, ids=lambda p: p.parent.parent.name)
def test_the_sigma_stage_draws_the_pinned_training_regime(script: Path):
    text = _text(script)
    assert "sigma_training_regime.py" in text
    assert "--n-cells" in text and "12" in text
    assert "890217" in text, (
        f"{script.name} does not use the pinned draw seed, so its σ would be fit on a different "
        f"set of cells than every other method's")


@pytest.mark.parametrize("script", SIGMA_LAUNCHERS, ids=lambda p: p.parent.parent.name)
def test_the_sigma_stage_fits_with_the_shared_pass(script: Path):
    text = _code(script)
    assert "competitors.sigma_pass" in text, (
        f"{script.name} must fit with the shared pass; a per-method fitter is how the four σ "
        f"tables stopped meaning the same thing last time")
    assert "--eval-regions" in text, (
        f"{script.name} does not pass the regime's own BED, so a D32 fit could be taken on loci "
        f"the model never trained on")


@pytest.mark.parametrize("script,method", [(EDICE / "sigma.sh", "eDICE"),
                                           (CHROMIMPUTE / "sigma.sh", "ChromImpute")],
                         ids=["edice", "chromimpute"])
def test_the_sigma_stage_writes_the_pinned_paths(script: Path, method: str):
    text = _text(script)
    assert f"/scratch/mforooz/t81_sigma/{method}/" in text, (
        f"{script.name} does not write training-track predictions to the pinned train_pred root")
    assert "train_pred" in text
    assert f"{SIGMA_ROOT}/{method}/sigma_" in text
    assert f"--method {method}" in text


# ---------------------------------------------------------------------------------------------
# the selected checkpoint
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("script,method", [(EDICE / "eic_train.sh", "eDICE"),
                                           (CHROMIMPUTE / "submit.sh", "ChromImpute")],
                         ids=["edice", "chromimpute"])
def test_the_selected_checkpoint_is_copied_off_scratch(script: Path, method: str):
    text = _text(script)
    assert f"{CKPT_ROOT}/{method}/" in text, (
        f"{script.name} leaves the fitted parameters on scratch, which is purged 60 days after "
        f"the oldest file")


# ---------------------------------------------------------------------------------------------
# ChromImpute's own invariants — Rule 2, and the two limits that are measurements
# ---------------------------------------------------------------------------------------------


def test_the_training_grid_is_deleted_before_prepare_rewrites_it():
    """A grid left by an earlier regime would retarget `dist` and `gtd`, and do it silently."""
    text = _code(CHROMIMPUTE / "submit.sh")
    assert 'rm -f "$RUN/input/chrominfo.train.txt"' in text
    assert text.index('rm -f "$RUN/input/chrominfo.train.txt"') < text.index('"$HERE/prepare.py"')


def test_the_prepare_shard_count_stays_at_the_torch_import_cap():
    """Beyond twelve, concurrent torch imports off /project return half-read modules."""
    assert "NSHARD=${CI_NSHARD:-12}" in _text(CHROMIMPUTE / "submit.sh")


def test_the_fitted_stages_run_on_the_training_grid_and_apply_does_not():
    """Rule 2: transferable parameters are fit on training loci; per-position inference is not."""
    text = _text(CHROMIMPUTE / "stage.sh")
    assert 'CI ComputeGlobalDist -m "$ITEM" "$CONV" "$IN/inputinfofile.txt" "$TRAIN_INFO"' in text
    assert '"$IN/chrominfo.txt" "$IMP" "$S" "$M"' in text, (
        "the board's Apply must keep chrominfo.txt — the chromosomes actually being predicted")


def test_the_sigma_apply_is_kept_out_of_the_board_output_directory():
    """A training self-pair wig in OUTPUTIMPUTEDIR would be collected as though it were a target."""
    text = _text(CHROMIMPUTE / "stage.sh")
    assert "SIMP=$CI_RUN/SIGMAIMPUTEDIR" in text
    assert "sapply)" in text and "strain)" in text
    assert '"$TRAIN_INFO" "$SIMP" "$S" "$M"' in text


@pytest.mark.parametrize("script", sorted(CHROMIMPUTE.glob("*.sh")), ids=lambda p: p.name)
def test_the_jar_default_is_the_pinned_one(script: Path):
    text = _text(script)
    for line in text.splitlines():
        if line.strip().startswith("#"):
            continue
        if "CI_JAR:-" in line or "CI_JAR:=" in line:
            assert JAR in line, f"{script.name}: {line.strip()!r} does not default to {JAR}"
