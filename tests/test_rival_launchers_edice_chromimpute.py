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


def test_the_edice_b_once_guard_watches_both_prediction_roots():
    """One root guarded is no guard: SCOPE picks the root, so two scopes are two B_ predictions.

    `SCOPE=genomewide` appends `.genomewide` to the panel root. A guard that only tested the root
    THIS invocation would write lets `PANEL=B_ SCOPE=heldout` and `PANEL=B_ SCOPE=genomewide` each
    find their own root absent and each predict B_ — two predictions of the blind panel, both
    passing the once-guard, with nothing in the record to say so.
    """
    text = _code(EDICE / "eic_score.sh")
    loops = [ln for ln in text.splitlines()
             if "PRED_ROOT_B" in ln and ".genomewide" in ln and "for " in ln]
    assert loops, (
        "eic_score.sh's B_ once-guard does not iterate over both the panel root and its "
        ".genomewide sibling, so a second B_ prediction would pass it")
    assert "exit 4" in text


def test_the_edice_b_panel_is_genome_wide_only():
    """eDICE's one permitted B_ prediction is the genome-wide one, so held-out B_ is refused.

    §4 prints eDICE's genome-wide cell, and the genome-wide scoring pass carries the ranked
    held-out `macro` and `panels` in the same json through `--held-out-chroms`. So genome-wide is
    the strictly larger scope and a held-out B_ pass could only SPEND the one permitted B_
    prediction on the smaller of the two.
    """
    text = _code(EDICE / "eic_score.sh")
    refusals = [ln for ln in text.splitlines()
                if 'PANEL" = "B_"' in ln and 'SCOPE" != "genomewide"' in ln]
    assert refusals, (
        "eic_score.sh does not refuse SCOPE=heldout with PANEL=B_, so the one permitted B_ "
        "prediction could be spent on the smaller scope")
    assert "--held-out-chroms" in text, (
        "the refusal above is only honest if the genome-wide pass really does emit the held-out "
        "numbers too")


def test_a_held_out_score_never_overwrites_a_genome_wide_one():
    """Both scopes write ONE path, so held-out-after-genome-wide would drop the bigger block."""
    text = _code(EDICE / "eic_score.sh")
    assert "genome_wide" in text, (
        "eic_score.sh does not look at the existing scores json's `genome_wide` block, so a "
        "held-out pass would silently overwrite a genome-wide result at the same path")
    assert "exit 5" in text or "SystemExit(5)" in text, (
        "that overwrite must be a refusal with its own exit code, not a warning")


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


def _eval_pairs_assignments(text: str) -> list[str]:
    """Every line that ASSIGNS to an `eval_pairs` key — not the ones that merely read it."""
    out = []
    for line in text.splitlines():
        if "eval_pairs" not in line:
            continue
        tail = line.split("eval_pairs", 1)[1][:6]     # past the key, before its value
        if "=" in tail and "==" not in tail:
            out.append(line.strip())
    return out


@pytest.mark.parametrize("script", SIGMA_LAUNCHERS, ids=lambda p: p.parent.parent.name)
def test_a_sigma_regime_never_carries_a_self_pair(script: Path):
    """`regime.py` REFUSES `[[c, c]]`, so a stage that writes one writes a file that cannot load.

    `tools/sigma_training_regime.py` spells the self-pairing the one way the code allows: no pairs
    at all, with the drawn cells in `biosamples.eval`, which `bench.harness.StoreSource` then
    self-pairs on its documented no-pairing path. A launcher that narrows the derived regime must
    keep that shape — narrow `biosamples.eval`, and leave `eval_pairs` empty.
    """
    for line in _eval_pairs_assignments(_code(script)):
        rhs = line.split("=", 1)[1].strip()
        assert rhs.startswith("[]"), (
            f"{script.name} assigns eval_pairs a non-empty value: {line!r}. "
            f"candi.store.regime._parse_eval_pairs refuses a pair of a cell with itself, so the "
            f"regime this writes would not load and the fit would die on its first read.")


def test_the_chromimpute_sigma_stage_reads_the_drawn_cells_off_biosamples_eval():
    """The draw's `eval_pairs` is EMPTY, so counting cells from it counts zero.

    That is the failure this test exists for: an empty cell list makes every cell "dropped", the
    `len(kept) < 3` guard below it fires, and the stage refuses every legitimate run with a message
    about missing training data.
    """
    text = _code(CHROMIMPUTE / "sigma.sh")
    assert "biosamples" in text and '"eval"' in text, (
        "chromimpute/sigma.sh does not read the drawn cells out of `biosamples.eval`")
    assert 'for _, t in draw["eval_pairs"]' not in text, (
        "chromimpute/sigma.sh still counts cells off the draw's eval_pairs, which is empty on the "
        "shape tools/sigma_training_regime.py writes")
    assert "len(kept) < 3" in text, "the survivor guard this defect defeated is gone"


def _continued(text: str, needle: str) -> str:
    """The ONE command containing `needle`, with its backslash continuations joined into a line.

    A flag check that read the whole file cannot tell which command a flag is on, and these stages
    run two in a row that take the same flag names with different values.
    """
    lines = text.splitlines()
    hits = [k for k, ln in enumerate(lines) if needle in ln]
    assert len(hits) == 1, f"{needle} appears on {len(hits)} lines, not one: {hits}"
    i = hits[0]
    parts = [lines[i]]
    while parts[-1].rstrip().endswith("\\") and i + 1 < len(lines):
        i += 1
        parts.append(lines[i])
    return " ".join(p.rstrip().rstrip("\\").rstrip() for p in parts)


def test_the_chromimpute_sigma_stage_collects_a_region_grid_rather_than_refusing_it():
    """A `regions` regime's training grid is region pseudo-chromosomes, and collect.py maps it back.

    The stage used to refuse that grid outright — `collect.py` looked every wig chromosome up in
    the store's `genome.n_bins`, which carries no region name, so the `eic_pilot` σ table could not
    be collected at all. `collect.py --regime` is the map, and it takes the SOURCE regime:
    `prepare.region_scope` cuts the BED to `train_chroms`, and the derived σ regime empties that
    list, so handing in the derived file is refused by name.
    """
    text = _text(CHROMIMPUTE / "sigma.sh")
    assert "PILOT REGIME IS REFUSED" not in text, (
        "chromimpute/sigma.sh still carries the pilot refusal, which collect.py --regime lifted")
    assert "D32 region training grid" not in text, (
        "the in-script refusal of a region grid is still there")
    call = _continued(_code(CHROMIMPUTE / "sigma.sh"), "collect.py")
    assert "--regime $SRC_REGIME" in call, (
        f"collect.py is called without the SOURCE regime, so a region grid cannot be mapped back "
        f"onto the store: {call!r}")
    assert "$SIGMA_REGIME" not in call, (
        "collect.py is handed the DERIVED regime; region_scope cuts its BED to `train_chroms`, "
        "which tools/sigma_training_regime.py empties, and collect.py refuses it by name")
    assert "--notes" not in call, (
        "a --notes of ours replaces collect.py's own note, which is the only place a reader of a "
        "region-sparse root is told the arrays are NaN outside the regions")


def test_the_chromimpute_sigma_fit_does_not_pass_the_apply_grid_as_chromosomes():
    """`collect.py` writes `<real chrom>.npz`; the grid names are the wigs', not the store's.

    `sigma_pass --chroms` is checked against the σ regime's `eval_chroms` and would refuse a region
    pseudo-chromosome as held out. Left off, the fit takes those `eval_chroms` — which ARE the
    source regime's `train_chroms`, the same loci the grid was cut from.
    """
    fit = _continued(_code(CHROMIMPUTE / "sigma.sh"), "competitors.sigma_pass")
    assert "--chroms" not in fit, f"the fit still passes the Apply grid as chromosomes: {fit!r}"
    assert "--regime $SIGMA_REGIME" in fit, "the fit must read the DERIVED sigma regime"


def test_the_sigma_array_throttle_is_not_confused_with_the_torch_import_cap():
    """Two different numbers, two different reasons; conflating them moved the wrong one once."""
    text = _text(CHROMIMPUTE / "sigma.sh")
    assert "THROTTLE=${CI_THROTTLE:-10}" in text
    assert "torch" in text, (
        "chromimpute/sigma.sh's array throttle is a queue courtesy cap on java-only stages; the "
        "comment must say so, because the neighbouring CI_NSHARD=12 IS the torch-import cap and "
        "the two look alike")
    assert "NSHARD=${CI_NSHARD:-12}" in _text(CHROMIMPUTE / "submit.sh")


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
