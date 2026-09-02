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

import json
import os
import re
import shutil
import subprocess
import sys
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


# ---------------------------------------------------------------------------------------------
# the predict batch — a VRAM knob, and the σ stage does not fit the MIG slice without it
# ---------------------------------------------------------------------------------------------

#: eDICE stages that run `run_eic.py predict` and must therefore choose a batch size.
EDICE_PREDICT_STAGES = (EDICE / "sigma.sh", EDICE / "eic_score.sh")

#: What both stages pass. `run_eic.py`'s own default is 4096, which at the σ panel's 98 declared
#: tracks costs `4096 x 98 x 2048 x 4 B` = 3.06 GiB for one decoder activation and OOM'd the slice.
PREDICT_BATCH_DEFAULT = "1024"


@pytest.mark.parametrize("script", EDICE_PREDICT_STAGES, ids=lambda p: p.name)
def test_the_predict_batch_is_reachable_from_the_launch_line(script: Path):
    """Both eDICE predict stages take PREDICT_BATCH, and default it to the value that fits.

    The σ job died on a CUDA OOM inside the decoder ReLU with no operator-side fix: the stage's
    whole env surface reached neither `--batch-size` nor `--slab`, so the only way to lower the
    per-batch shape was to edit the file. This is that knob.
    """
    text = _code(script)
    assert f'PREDICT_BATCH="${{PREDICT_BATCH:-{PREDICT_BATCH_DEFAULT}}}"' in text, (
        f"{script.name} does not default PREDICT_BATCH to {PREDICT_BATCH_DEFAULT}, so the σ "
        f"panel's 98-track decoder activation is back over the MIG slice")


@pytest.mark.parametrize("script", EDICE_PREDICT_STAGES, ids=lambda p: p.name)
def test_the_predict_batch_reaches_the_predict_command(script: Path):
    """On the predict line itself — not merely somewhere in the file.

    `_continued` joins the one `run_eic.py predict` command, so a PREDICT_BATCH that were set and
    then never passed on would fail here rather than pass on the assignment alone.
    """
    cmd = _continued(_code(script), "run_eic.py predict")
    assert '--batch-size "$PREDICT_BATCH"' in cmd, (
        f"{script.name}'s predict command does not pass PREDICT_BATCH, so run_eic.py falls back "
        f"to its own 4096 default:\n  {cmd}")
    # `--chroms` is nargs="+", so its expansion must stay LAST in the ARGV: any flag written after
    # it is swallowed as another chromosome name and never reaches --batch-size. `|| exit $?` is
    # shell control rather than an argument, so the argv ends at the first `||`.
    argv = cmd.split("||")[0].rstrip()
    assert argv.endswith('"${CHROMS[@]}"'), (
        f"{script.name} writes an argument after the nargs=+ --chroms expansion, which would "
        f"swallow it as a chromosome name:\n  {argv}")


@pytest.mark.parametrize("script", EDICE_PREDICT_STAGES, ids=lambda p: p.name)
def test_the_predict_batch_is_printed_in_the_banner(script: Path):
    """The log must say which batch a root was written under, since the flag is now settable."""
    text = _code(script)
    assert any("echo" in ln and "predict_batch=$PREDICT_BATCH" in ln
               for ln in text.splitlines()), (
        f"{script.name} does not echo predict_batch, so a log cannot say which shape it ran")


@pytest.mark.parametrize("script", EDICE_PREDICT_STAGES, ids=lambda p: p.name)
def test_the_predict_batch_echo_is_not_inside_a_heredoc(script: Path):
    """A previous edit put a banner echo inside a python heredoc and broke the stage on a
    SyntaxError. `echo "..."` is not python, so it must sit in the shell."""
    inside, delim = set(), None
    lines = _text(script).splitlines()
    for i, ln in enumerate(lines, 1):
        if delim is None:
            m = re.search(r"<<-?\s*'?\"?([A-Za-z_][A-Za-z0-9_]*)'?\"?", ln)
            if m:
                delim = m.group(1)
        elif ln.strip() == delim:
            delim = None
        else:
            inside.add(i)
    offenders = [(i, lines[i - 1].strip()) for i in sorted(inside)
                 if "PREDICT_BATCH" in lines[i - 1] or "[banner]" in lines[i - 1]]
    assert not offenders, f"{script.name} has shell lines inside a heredoc body: {offenders}"


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


# ---------------------------------------------------------------------------------------------
# the missing-track hatch on the σ fit
# ---------------------------------------------------------------------------------------------
#
# A rare mark held by ONE training cell has an empty leave-one-out pool, so no method can predict
# it on a self-pair. Lavawizard's eic_pilot σ predict pass wrote 96 of the 98 declared tracks and
# skipped `T_IMR-90 H2AK9ac` and `T_trophoblast_cell H4K12ac` (Fir 57836027,
# cruxvault/results/t81/W3_LAVA_PILOT.md); `competitors/sigma_pass.py` then refused the partial
# root, because a σ pooled over whichever tracks finished reads as a σ over the whole training
# panel in every table downstream. `--allow-missing` fits on what is there and names the gap in
# `skipped_tracks`. Each launcher must be able to say it from the launch line, and must not say it
# by default.


@pytest.mark.parametrize("script", SIGMA_LAUNCHERS, ids=lambda p: p.parent.parent.name)
def test_the_sigma_stage_takes_allow_missing_from_the_environment(script: Path):
    text = _code(script)
    assert "SIGMA_ALLOW_MISSING:-0" in text, (
        f"{script.name} has no SIGMA_ALLOW_MISSING pass-through defaulting to 0, so a root "
        f"missing a rare mark's track can only be fit by editing the file")
    assert any("echo" in ln and "sigma_allow_missing=$SIGMA_ALLOW_MISSING" in ln
               for ln in text.splitlines()), (
        f"{script.name} does not echo sigma_allow_missing, so a log cannot say whether the table "
        f"it wrote was pooled over a partial panel")
    for phrase in ("skipped_tracks", "leave-one-out"):
        assert phrase in _text(script), (
            f"{script.name}'s header does not name `{phrase}` -- the operator has to be told what "
            f"to confirm before setting the flag, and where the gap is then recorded")


def test_the_edice_sigma_fit_appends_allow_missing_only_when_asked():
    """`EXTRA` is the fit's own argument array, and it must stay last on the fitter's argv."""
    text = _code(EDICE / "sigma.sh")
    hits = [ln.strip() for ln in text.splitlines() if "--allow-missing" in ln]
    assert hits == ['[ "$SIGMA_ALLOW_MISSING" = "1" ] && EXTRA+=(--allow-missing)'], (
        f"edice/sigma.sh does not append `--allow-missing` conditionally: {hits}")
    fit = _continued(text, "competitors.sigma_pass")
    assert fit.rstrip().endswith('"${EXTRA[@]}"'), (
        f"edice/sigma.sh's fit does not end on the EXTRA expansion, so the flag may not reach "
        f"the fitter at all:\n  {fit}")


def test_the_chromimpute_sigma_fit_decides_the_flag_when_it_writes_the_fit_script():
    """The fit is a separate job hours later, so the flag has to be baked into `sigma_fit.sh`.

    `$RUN/sigma_fit.sh` is written by an UNQUOTED heredoc, so `$ALLOW_MISSING_ARG` is expanded as
    the file is written and the file itself records which way the fit was asked to go. A runtime
    read inside the generated script would depend on what that job's environment happened to
    carry, and `--export=ALL` is not what submits it.
    """
    text = _code(CHROMIMPUTE / "sigma.sh")
    assert 'if [ "$SIGMA_ALLOW_MISSING" = "1" ]; then ALLOW_MISSING_ARG="--allow-missing"; fi' \
        in text, "chromimpute/sigma.sh does not resolve ALLOW_MISSING_ARG on the login node"
    assert "EXTRA=($ALLOW_MISSING_ARG)" in text, (
        "chromimpute/sigma.sh's generated fit script does not seed EXTRA from ALLOW_MISSING_ARG, "
        "so the flag never reaches the job that runs the fit")


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


# ---------------------------------------------------------------------------------------------
# `strain` writes what `sapply` reads — one item list, one classifier name
# ---------------------------------------------------------------------------------------------
#
# `sapply` task 5 of Fir job 57807330 died three times on the jar's own
#
#     java.lang.IllegalArgumentException: No previously trained classifiers for mark DNase-seq were
#     found available to load!    at ernst.ChromImpute.ChromImpute.executeApply(...:2344)
#
# and `afterok` then cancelled the fit — while `strain` (57807329) had reported 22/22 COMPLETED.
# `Train` had written the 33 `useattributes_T_SK-MEL-5_DNase-seq_<i>_<j>.txt.gz` masks, no
# `classifier_` file, and exit 0. Two things keep that from being a silent success: the two verbs
# must walk ONE list and spell the classifier name in ONE place, and each must check the set it
# writes or reads.


def _verb(code: str, verb: str) -> str:
    """One `case` branch of `stage.sh`, from `<verb>)` to its `;;`."""
    i = code.index(f"  {verb})")
    return code[i:code.index("\n    ;;", i)]


def test_the_two_sigma_verbs_read_one_shared_item_list():
    """A row `sapply` has and `strain` does not is a task asking for a set nobody trained."""
    stage = _code(CHROMIMPUTE / "stage.sh")
    assert re.search(r"strain\|sapply\)\s*LIST=\$CI_RUN/lists/sigma_items\.txt", stage), (
        "stage.sh does not resolve both sigma verbs to the one shared item list")
    for stale in ("lists/strain.txt", "lists/sapply.txt"):
        assert stale not in stage, (
            f"stage.sh still reads a per-verb {stale}; two files can drift, one cannot")

    sigma = _code(CHROMIMPUTE / "sigma.sh")
    assert '(run / "lists" / "sigma_items.txt").write_text' in sigma, (
        "chromimpute/sigma.sh does not write the shared item list stage.sh reads")
    for stale in ('"strain.txt").write_text', '"sapply.txt").write_text'):
        assert stale not in sigma, f"chromimpute/sigma.sh still writes a per-verb list ({stale})"
    assert 'wc -l < "$RUN/lists/sigma_items.txt"' in sigma, (
        "chromimpute/sigma.sh sizes an array off something other than the shared list, so the two "
        "arrays could have different lengths even with one list on disk")


def test_the_sigma_verbs_spell_the_classifier_set_name_in_one_place():
    """The writer and the reader cannot disagree about a name only one line spells."""
    stage = _code(CHROMIMPUTE / "stage.sh")
    spelled = [ln.strip() for ln in stage.splitlines() if "classifier_" in ln]
    assert spelled == ['CLASSIFIERS="$SPRED/classifier_${S}_${M}_"'], (
        f"stage.sh spells the classifier-set name on {len(spelled)} line(s): {spelled}")
    for verb in ("strain", "sapply"):
        assert '"$CLASSIFIERS"*' in _verb(stage, verb), (
            f"stage.sh's {verb} does not check the shared classifier-set name")


def test_strain_refuses_when_train_exits_zero_having_written_nothing():
    """22/22 COMPLETED with an empty predictor directory is the failure this catches one stage up."""
    strain = _verb(_code(CHROMIMPUTE / "stage.sh"), "strain")
    assert strain.index("CI Train") < strain.index('"$CLASSIFIERS"*'), (
        "strain must count the classifier files AFTER Train, not before")
    assert "exit 3" in strain, "strain does not fail when Train wrote no classifier"


def test_sapply_checks_the_classifier_set_before_it_starts_the_jar():
    """An absent set is deterministic, so it must not ride the transient-Lustre retry three times."""
    sapply = _verb(_code(CHROMIMPUTE / "stage.sh"), "sapply")
    assert sapply.index('"$CLASSIFIERS"*') < sapply.index("CI Apply"), (
        "sapply calls the jar before it checks that there is a classifier set to load — the jar's "
        "own error names only the mark, and RETRY repeats it three times")
    assert "exit 3" in sapply
    check = [ln.strip() for ln in sapply.splitlines() if '"$CLASSIFIERS"*' in ln]
    assert check and all(ln.startswith("NCLS=") for ln in check), (
        f"sapply's classifier check is not a plain test outside RETRY/CI: {check}")


def test_the_sigma_stage_drops_a_cell_train_can_fit_nothing_for():
    """`Train` holds the target's own track out, so a one-track cell has no features left.

    Measured, not reasoned: of the 22 items of Fir strain 57807329, the only two that left zero
    `classifier_` files were the two cells with a single row in `inputinfofile.txt`
    (`T_SK-MEL-5`, `T_skin_of_body`); every cell with two rows or more got a set for every mark.
    """
    sigma = _code(CHROMIMPUTE / "sigma.sh")
    assert 'run / "input" / "inputinfofile.txt"' in sigma, (
        "chromimpute/sigma.sh judges the cell panel off the manifest alone; the panel that decides "
        "whether Train has features is the one in inputinfofile.txt, which is what the jar reads")
    assert "len(tracks) < 2" in sigma, (
        "chromimpute/sigma.sh keeps a one-track cell, whose one item trains no classifier at all")


@pytest.mark.parametrize("script", sorted(CHROMIMPUTE.glob("*.sh")), ids=lambda p: p.name)
def test_the_jar_default_is_the_pinned_one(script: Path):
    text = _text(script)
    for line in text.splitlines():
        if line.strip().startswith("#"):
            continue
        if "CI_JAR:-" in line or "CI_JAR:=" in line:
            assert JAR in line, f"{script.name}: {line.strip()!r} does not default to {JAR}"


# ---------------------------------------------------------------------------------------------
# the sbatch --export hand-off — a comma-valued variable must reach the stage whole
# ---------------------------------------------------------------------------------------------
#
# `sbatch --export=<[ALL,]environment variables>` takes a COMMA-SEPARATED list of `NAME` /
# `NAME=VALUE` entries, so a value that itself contains a comma cannot ride in it.
# `--export=ALL,CI_CHROMS=chr20,chr21,chr22` hands the job `CI_CHROMS=chr20` and reads `chr21` /
# `chr22` as bare variable names to copy from the submitting environment; they do not exist, and
# sbatch reports nothing. Measured on Fir, job 57806189 (cruxvault/results/t81/W3_EARLY.md §1) --
# and that truncation is why wave 2's ChromImpute `convert` and `apply` ran on chr20 alone while
# `chrominfo.txt` named three chromosomes.
#
# The drivers export their variables and submit with `--export=ALL`. The tests below run both
# ChromImpute drivers against a stub `sbatch`, then put every recorded submission back through an
# emulator of that comma rule -- so the assertion is on the value the STAGE would read, and not on
# the shape of a command line.

CHROMS_THREE = "chr20,chr21,chr22"


def _sbatch_export_env(export_arg: str | None, caller_env: dict) -> dict:
    """The environment `sbatch --export=<export_arg>` hands the job, given the submitting env.

    Implements the one rule that matters here: the argument is split on commas FIRST, and only then
    is each token read as `NAME=VALUE` or as a bare `NAME` to copy from the caller. A value with a
    comma in it is therefore cut at the comma, and its tail is swallowed as variable names.
    """
    if export_arg is None:
        return dict(caller_env)                       # sbatch's own default is ALL
    out: dict = {}
    for token in export_arg.split(","):
        if token.upper() == "ALL":
            out.update(caller_env)
        elif token.upper() == "NONE":
            out.clear()
        elif "=" in token:
            name, value = token.split("=", 1)
            out[name] = value
        elif token in caller_env:
            out[token] = caller_env[token]
        # a bare name the caller's environment does not carry is dropped, silently
    return out


def _export_arg(argv: list) -> str | None:
    """The `--export` argument of one recorded sbatch call, or None if it passed none."""
    for k, a in enumerate(argv):
        if a.startswith("--export="):
            return a.split("=", 1)[1]
        if a == "--export":
            return argv[k + 1]
    return None


def test_the_export_emulator_reproduces_the_measured_truncation():
    """The guard needs a guard of its own: this is job 57806189's output, verbatim."""
    caller = {"CI_SHELLCH": CHROMS_THREE}
    seen = _sbatch_export_env(f"ALL,CI_LISTCH={CHROMS_THREE},CI_TAIL=xyz", caller)
    assert seen["CI_LISTCH"] == "chr20"               # the list form, TRUNCATED
    assert seen["CI_TAIL"] == "xyz"                   # so chr21 and chr22 were eaten as names
    assert seen["CI_SHELLCH"] == CHROMS_THREE         # the shell-export + ALL form survives whole
    assert _sbatch_export_env("ALL", caller)["CI_SHELLCH"] == CHROMS_THREE
    assert _sbatch_export_env(None, caller)["CI_SHELLCH"] == CHROMS_THREE


#: Stands in for the Fir venv python on the login node. The drivers call it for the panel split,
#: `prepare.py`, the sigma draw and the sigma stage's inline program -- every one of which reads a
#: CANDI store that does not exist on a laptop -- and then read the files those calls wrote. So the
#: stub writes those files. `-c` is the driver's OWN inline python and runs for real.
_FAKE_PY = r'''#!{python}
import json, os, sys
from pathlib import Path

argv = sys.argv[1:]


def opt(name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


if argv and argv[0] == "-c":
    os.execv(sys.executable, [sys.executable, *argv])

REGIME = {{
    "eval_chroms": ["chr20", "chr21", "chr22"],
    "train_chroms": ["chr18", "chr19"],
    "assays": ["H3K4me3", "H3K27ac"],
    "biosamples": {{"train": ["C1", "C2", "C3"], "eval": ["C1", "C2", "C3"]}},
    "eval_pairs": [["C1", "T1"], ["C2", "T2"]],
}}
ITEMS = [("C1", "H3K4me3"), ("C2", "H3K4me3"), ("C3", "H3K27ac")]

if argv and argv[0] == "-":
    sys.stdin.read()                       # the sigma stage's inline program, stubbed
    draw, out, run = Path(argv[1]), Path(argv[2]), Path(argv[3])
    lines = "".join("%s\t%s\n" % it for it in ITEMS)
    (run / "lists" / "sigma_items.txt").write_text(lines)   # the ONE list both verbs read
    (run / "input" / "targets_sigma.tsv").write_text(
        "".join("%s\t%s\t%s\n" % (c, c, m) for c, m in ITEMS))
    drawn = json.loads(draw.read_text())
    drawn["eval_pairs"] = []
    out.write_text(json.dumps(drawn))
    sys.exit(0)

prog = Path(argv[0]).name if argv else ""
if prog == "declare_eval_pairs.py":
    Path(opt("--out")).write_text(json.dumps(REGIME))
elif prog == "sigma_training_regime.py":
    Path(opt("--out")).write_text(json.dumps(dict(REGIME, eval_pairs=[])))
elif prog == "prepare.py":
    out = Path(opt("--out"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "inputinfofile.txt").write_text(
        "".join("%s\t%s\t%s_%s.bedgraph\n" % (c, m, c, m) for c, m in ITEMS))
    (out / "chrominfo.txt").write_text(
        "".join("%s\t50000000\n" % c for c in opt("--chroms", "").split(",") if c))
    (out / "chrominfo.train.txt").write_text("chr18\t80373285\nchr19\t58617616\n")
    targets = "".join("T%d\t%s\t%s\n" % (k, c, m) for k, (c, m) in enumerate(ITEMS))
    (out / "targets.tsv").write_text(targets)
    (out / "targets_pilot.tsv").write_text(targets)
else:
    sys.exit("[fake py] unexpected program: %r" % (argv,))
'''

#: Records what would have been submitted, AND the environment it would have been submitted in --
#: which is the half `--export=ALL` propagates. Prints a job id, because `--parsable` is captured.
_FAKE_SBATCH = r'''#!{python}
import json, os, sys
from pathlib import Path

d = Path(os.environ["SBATCH_RECORD_DIR"])
d.mkdir(parents=True, exist_ok=True)
n = len(list(d.glob("*.json")))
(d / ("%03d.json" % n)).write_text(json.dumps({{"argv": sys.argv[1:], "env": dict(os.environ)}}))
print(57800000 + n)
'''


@pytest.fixture(scope="module")
def chromimpute_submissions(tmp_path_factory):
    """Run both ChromImpute drivers with a stub `sbatch` and a stub python, and record the calls.

    Nothing real is submitted and nothing real is computed. What IS real is every driver line
    between those stubs -- including the one that decides what environment the stage jobs get.
    `sigma.sh` runs second and against the same run directory, because it reads the work lists and
    the training grid `submit.sh` just laid out.
    """
    if shutil.which("bash") is None:
        pytest.skip("no bash on this host")
    tmp = tmp_path_factory.mktemp("ci_submit")
    stub = tmp / "stub"
    stub.mkdir()
    for name, template in (("fake_py", _FAKE_PY), ("sbatch", _FAKE_SBATCH)):
        p = stub / name
        p.write_text(template.format(python=sys.executable), encoding="utf-8")
        p.chmod(0o755)

    src_regime = tmp / "regime.eic_19.json"
    src_regime.write_text("{}\n", encoding="utf-8")
    run, records = tmp / "run", tmp / "records"

    env = dict(os.environ)
    env.update({
        "PATH": f"{stub}{os.pathsep}{env['PATH']}",
        "SBATCH_RECORD_DIR": str(records),
        "CI_REPO": str(REPO),
        "CI_STORE": str(tmp / "store"),
        "CI_PY": str(stub / "fake_py"),
        "CI_JAR": str(tmp / "ChromImpute.jar"),
        "CI_REGIME": str(src_regime),
        "CI_CHROMS": CHROMS_THREE,
        "CI_ACCT": "def-test",
        "CI_PRED_ROOT": str(tmp / "pred"),
        "CI_CKPT_DIR": str(tmp / "ckpt"),
    })
    # The σ stage reads this one off the plain environment, so a value in the operator's own shell
    # would decide what the fixture generates. The default is what is under test here.
    env.pop("SIGMA_ALLOW_MISSING", None)
    sigma_env = {"CI_RUN": str(run),
                 "CI_SIGMA_OUT": str(tmp / "sigma.json"),
                 "CI_TRAIN_PRED": str(tmp / "train_pred")}
    out = {"run": run, "env": env, "sigma_env": sigma_env}
    for driver, args, extra in (
        ("submit.sh", [str(run)], {}),
        ("sigma.sh", [], sigma_env),
    ):
        before = len(list(records.glob("*.json"))) if records.exists() else 0
        r = subprocess.run(["bash", str(CHROMIMPUTE / driver), *args],
                           env={**env, **extra}, cwd=str(tmp),
                           capture_output=True, text=True)
        assert r.returncode == 0, f"{driver} exited {r.returncode}:\n{r.stdout}\n{r.stderr}"
        out[driver] = [json.loads(p.read_text())
                       for p in sorted(records.glob("*.json"))[before:]]
        assert out[driver], f"{driver} submitted nothing, so nothing below is checked"
    return out


#: What each driver's array stages are called, so a fixture that submitted half a chain is caught.
STAGES_SUBMITTED = {"submit.sh": {"prepare", "convert", "dist", "gtd", "train", "apply"},
                    "sigma.sh": {"strain", "sapply"}}


@pytest.mark.parametrize("driver", sorted(STAGES_SUBMITTED))
def test_a_stage_job_sees_the_whole_chromosome_list(chromimpute_submissions, driver):
    """The wave-2 defect, as a test: `CI_CHROMS` must arrive as three chromosomes, not one."""
    stage_calls = [c for c in chromimpute_submissions[driver]
                   if c["argv"] and c["argv"][-1].endswith("stage.sh")]
    assert stage_calls, f"{driver} submitted no array stage against stage.sh"
    for call in stage_calls:
        seen = _sbatch_export_env(_export_arg(call["argv"]), call["env"])
        assert seen.get("CI_CHROMS") == CHROMS_THREE, (
            f"{driver} hands stage {seen.get('CI_STAGE')!r} CI_CHROMS={seen.get('CI_CHROMS')!r}, "
            f"not {CHROMS_THREE!r}. sbatch splits an --export list on commas, so a comma-valued "
            f"variable cannot be passed in one -- export it and submit with --export=ALL. "
            f"--export was {_export_arg(call['argv'])!r}.")


@pytest.mark.parametrize("driver", sorted(STAGES_SUBMITTED))
def test_a_stage_job_sees_the_rest_of_its_environment_too(chromimpute_submissions, driver):
    """Fixing the comma must not drop what the list used to carry: same names, same values."""
    run = chromimpute_submissions["run"]
    stages = set()
    for call in chromimpute_submissions[driver]:
        if not call["argv"] or not call["argv"][-1].endswith("stage.sh"):
            continue
        seen = _sbatch_export_env(_export_arg(call["argv"]), call["env"])
        for var in sorted(_ci_vars_stage_reads()):
            assert seen.get(var), f"{driver}: the stage would read no {var}"
        assert seen["CI_RUN"] == str(run)
        assert Path(seen["CI_REGIME"]).is_file(), (
            f"{driver} hands the stage CI_REGIME={seen['CI_REGIME']!r}, which is not a file")
        assert seen["CI_MX"].endswith("M")
        stages.add(seen["CI_STAGE"])
    assert stages == STAGES_SUBMITTED[driver], (
        f"{driver} submitted stages {sorted(stages)}, not {sorted(STAGES_SUBMITTED[driver])}")


def test_the_login_node_prepare_call_gets_the_whole_chromosome_list(chromimpute_submissions):
    """`prepare.py` takes `--chroms` as an ordinary argument, so its grid proves the driver's own
    value was three chromosomes -- and `chrominfo.txt` is what `Apply` is then handed."""
    grid = (chromimpute_submissions["run"] / "input" / "chrominfo.txt").read_text().split()
    assert [g for g in grid if g.startswith("chr")] == CHROMS_THREE.split(",")


def test_the_generated_fit_script_refuses_a_partial_root_by_default(chromimpute_submissions):
    """The fixture ran the driver with SIGMA_ALLOW_MISSING unset, so the flag must not be there."""
    gen = (chromimpute_submissions["run"] / "sigma_fit.sh").read_text(encoding="utf-8")
    assert "competitors.sigma_pass" in gen, (
        "the generated sigma_fit.sh does not call the shared fitter at all")
    assert "--allow-missing" not in gen, (
        "the generated sigma_fit.sh passes --allow-missing with SIGMA_ALLOW_MISSING unset, so a "
        "half-written apply pass would be pooled into σ with nothing in the table saying so")
    assert "EXTRA=()" in gen, (
        f"the generated sigma_fit.sh does not seed an empty EXTRA:\n{gen}")


def test_the_generated_fit_script_carries_allow_missing_when_the_environment_sets_it(
        chromimpute_submissions, tmp_path):
    """Same driver, same run directory, one variable different -- and it reaches the fit args.

    Run against a COPY of the fixture's run directory, so the default-off artifact the test above
    reads is not rewritten out from under it.
    """
    run = tmp_path / "run"
    shutil.copytree(chromimpute_submissions["run"], run)
    env = {**chromimpute_submissions["env"], **chromimpute_submissions["sigma_env"],
           "CI_RUN": str(run),
           "SBATCH_RECORD_DIR": str(tmp_path / "records"),
           "SIGMA_ALLOW_MISSING": "1"}
    r = subprocess.run(["bash", str(CHROMIMPUTE / "sigma.sh")], env=env, cwd=str(tmp_path),
                       capture_output=True, text=True)
    assert r.returncode == 0, f"sigma.sh exited {r.returncode}:\n{r.stdout}\n{r.stderr}"
    assert "sigma_allow_missing=1" in r.stdout, (
        f"the driver does not say in its log that it took the flag:\n{r.stdout}")
    gen = (run / "sigma_fit.sh").read_text(encoding="utf-8")
    assert "EXTRA=(--allow-missing)" in gen, (
        f"SIGMA_ALLOW_MISSING=1 did not reach the generated fit script:\n{gen}")
    fit = _continued(gen, "competitors.sigma_pass")
    assert '"${EXTRA[@]}"' in fit, (
        f"the generated fit command does not expand EXTRA, so the flag is set and never passed:\n"
        f"  {fit}")


def _ci_vars_stage_reads() -> set:
    """The `CI_*` names `stage.sh` reads with a default or a `:?` requirement.

    Read off the RECEIVING side on purpose: a variable the stage learns to read has to be exported
    by both drivers, and this is what makes forgetting one a failure rather than an empty default.
    """
    names = set(re.findall(r':\s*"\$\{(CI_[A-Z_]+)[:?]', _text(CHROMIMPUTE / "stage.sh")))
    assert len(names) >= 8, f"stage.sh's variable block did not parse: {names}"
    return names


@pytest.mark.parametrize("driver", sorted(STAGES_SUBMITTED))
def test_every_variable_the_stage_reads_is_exported_by_the_driver(driver):
    text = _code(CHROMIMPUTE / driver)
    missing = [v for v in sorted(_ci_vars_stage_reads())
               if not re.search(rf"^\s*export .*\b{v}=", text, re.M)]
    assert not missing, (
        f"{driver} never exports {missing}, and it cannot pass them on the sbatch line either -- "
        f"an --export list is split on commas. Add them to the export block.")


@pytest.mark.parametrize("script", sorted(CHROMIMPUTE.glob("*.sh")), ids=lambda p: p.name)
def test_no_assignment_rides_on_an_sbatch_export_list(script: Path):
    """`--export=ALL` or nothing. An assignment on that line is the wave-2 defect's shape.

    Checked as a form and not per variable: today only `CI_CHROMS` holds a comma, but a pilot
    sigma grid is a comma list of region names and the next comma-valued variable would land the
    same way, silently.
    """
    for line in _code(script).splitlines():
        for hit in re.finditer(r"--export[= ](\S+)", line):
            arg = hit.group(1).strip("\"'")
            assert arg.upper() in {"ALL", "NONE"}, (
                f"{script.name} submits with --export={arg!r}. sbatch splits that list on commas, "
                f"so a comma-valued variable in it is truncated and its tail is read as variable "
                f"names. Export the variables and pass --export=ALL.")


# ---------------------------------------------------------------------------------------------
# the dispatch block, RUN — predict+score vs score-only, and the challenge truth
# ---------------------------------------------------------------------------------------------
#
# `eic_score.sh` does predict -> score in ONE job, so its once-only `B_` guard and its scorer could
# not be reached independently: the guard fired on `PANEL=B_` alone and sat BEFORE the
# skip-when-complete test, so the store pass's own manifest made every later `B_` invocation exit 4
# — including the challenge-truth score, which writes no prediction and so cannot spend the one
# `B_` shot. ChromImpute and Lavawizard were never blocked because they have a separate score-only
# `score.sh`.
#
# The tests above read the launcher as text. These RUN the dispatch half of it — the completeness
# test, the guard, the challenge chromosome resolution, the predict and the score — against a fake
# `$PRED` tree and a stub `python`, and assert on the argv `candi.bench.external` would have got.

#: Everything from this line to EOF is the dispatch, and it reads only variables the harness sets.
DISPATCH_MARK = "# === DISPATCH: predict then score, or score a root that is already there"

#: What the fake store carries, and so what a `SCOPE=genomewide` root must list to count complete.
#: The launcher compares the manifest's list to this one element by element, so the order matters.
GW_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
#: The held-out three: the derived regime's `eval_chroms`, and all the challenge truth root covers.
HELD_OUT_THREE = ["chr20", "chr21", "chr22"]


def _dispatch_block() -> str:
    text = _text(EDICE / "eic_score.sh")
    assert DISPATCH_MARK in text, (
        "eic_score.sh no longer carries the dispatch marker, so this harness would run nothing")
    return text[text.index(DISPATCH_MARK):]


#: Stands in for the venv python inside the dispatch block. `-c` and `-` are the launcher's OWN
#: inline programs and run for real — they are the code under test. `run_eic.py` and
#: `-m candi.bench.external` are recorded instead of run, and the predict stub writes the manifest
#: a real predict pass would leave behind, because the score step reads it back.
_EDICE_FAKE_PY = r'''#!{python}
import json, os, sys
from pathlib import Path

argv = sys.argv[1:]
if argv and argv[0] in ("-c", "-"):
    os.execv(sys.executable, [sys.executable, *argv])

with open(os.environ["EDICE_RECORD"], "a") as fh:
    fh.write(json.dumps(argv) + "\n")

if argv and argv[0].endswith("run_eic.py"):
    out = Path(argv[argv.index("--out") + 1])
    out.mkdir(parents=True, exist_ok=True)
    gw = "--chroms" in argv and argv[argv.index("--chroms") + 1] == "all"
    chroms = os.environ["EDICE_GW_CHROMS" if gw else "EDICE_HELD_CHROMS"].split(",")
    (out / "manifest.json").write_text(json.dumps({{"chroms": chroms, "n_tracks": 1}}))
sys.exit(0)
'''

#: The completeness test imports the store reader to learn what `genomewide` means. A laptop has no
#: store, so these two stand in for `candi.store` on the harness's PYTHONPATH — the launcher exports
#: its own PYTHONPATH above the dispatch marker, so nothing here shadows the real package.
_SHIM_READER = '''import os


class CorpusStore:
    def __init__(self, path):
        self.path = path

    def n_bins(self):
        return {c: 1 for c in os.environ["EDICE_GW_CHROMS"].split(",")}
'''
_SHIM_LAYOUT = '''import os


def sort_chroms(chroms):
    order = os.environ["EDICE_GW_CHROMS"].split(",")
    return [c for c in order if c in set(chroms)]
'''


def _complete_root(root: Path, chroms, n_tracks: int = 2) -> None:
    """A prediction root the launcher must call COMPLETE: manifest, n_tracks dirs, every npz."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(
        json.dumps({"chroms": list(chroms), "n_tracks": n_tracks}), encoding="utf-8")
    for k in range(n_tracks):
        d = root / f"C{k}_H3K4me3"
        d.mkdir(exist_ok=True)
        for c in chroms:
            (d / f"{c}.npz").write_text("", encoding="utf-8")


def _run_dispatch(tmp_path: Path, *, panel: str, scope: str, truth: str,
                  pred_chroms=None, other_b_root_manifest: bool = False,
                  truth_chroms=HELD_OUT_THREE, extra_env=None):
    """Run eic_score.sh's dispatch block against a fake world; return the process and its calls.

    `pred_chroms=None` leaves `$PRED` ABSENT; a list writes a COMPLETE root listing those
    chromosomes. `truth_chroms=None` writes a truth manifest with no `chroms` key, which is what
    the fallback exists for. No GPU, no store, no model, no network.
    """
    if shutil.which("bash") is None:
        pytest.skip("no bash on this host")

    stub = tmp_path / "stub"
    stub.mkdir()
    py = stub / "python"
    py.write_text(_EDICE_FAKE_PY.format(python=sys.executable), encoding="utf-8")
    py.chmod(0o755)

    store_pkg = tmp_path / "shim" / "candi" / "store"
    store_pkg.mkdir(parents=True)
    (store_pkg.parent / "__init__.py").write_text("", encoding="utf-8")
    (store_pkg / "__init__.py").write_text("", encoding="utf-8")
    (store_pkg / "reader.py").write_text(_SHIM_READER, encoding="utf-8")
    (store_pkg / "layout.py").write_text(_SHIM_LAYOUT, encoding="utf-8")

    # The roots, shaped as the launcher's own defaults shape them: a genome-wide root is a
    # `.genomewide` SIBLING of the panel root, which is what makes the two-root guard necessary.
    pred_root_b = tmp_path / "pred_B" / "B_"
    base = pred_root_b if panel == "B_" else tmp_path / "pred_V" / "V_"
    pred = base.with_name(base.name + ".genomewide") if scope == "genomewide" else base
    if pred_chroms is not None:
        _complete_root(pred, pred_chroms)
    if other_b_root_manifest:
        pred_root_b.mkdir(parents=True, exist_ok=True)
        (pred_root_b / "manifest.json").write_text(
            json.dumps({"chroms": HELD_OUT_THREE, "n_tracks": 2}), encoding="utf-8")

    panel_regime = tmp_path / f"regime.test.{panel}.json"
    panel_regime.write_text(
        json.dumps({"eval_chroms": HELD_OUT_THREE, "store": str(tmp_path / "store")}),
        encoding="utf-8")
    truth_root = tmp_path / "truth" / "B_"
    truth_root.mkdir(parents=True)
    manifest = {"kind": "truth"}
    if truth_chroms is not None:
        manifest["chroms"] = list(truth_chroms)
    (truth_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    record = tmp_path / "calls.jsonl"
    env = {
        "PATH": f"{stub}{os.pathsep}{os.environ['PATH']}",
        "PYTHONPATH": str(tmp_path / "shim"),
        "EDICE_RECORD": str(record),
        "EDICE_GW_CHROMS": ",".join(GW_CHROMS),
        "EDICE_HELD_CHROMS": ",".join(HELD_OUT_THREE),
        "PANEL": panel, "SCOPE": scope, "TRUTH": truth,
        "PRED": str(pred), "PRED_ROOT_B": str(pred_root_b),
        "PANEL_REGIME": str(panel_regime), "TRUTH_ROOT": str(truth_root),
        "SCORES": str(tmp_path / "scores" / f"{truth}.{panel}.json"),
        "SIGMA": str(tmp_path / "sigma.json"),
        "MODEL": str(tmp_path / "model.selected.pt"),
        "N_TARGETS": "31", "PREDICT_BATCH": "1024",
    }
    env.update(extra_env or {})

    runner = tmp_path / "dispatch.sh"
    runner.write_text("set -uo pipefail\n" + _dispatch_block(), encoding="utf-8")
    r = subprocess.run(["bash", str(runner)], env=env, cwd=str(tmp_path),
                       capture_output=True, text=True)
    calls = ([json.loads(ln) for ln in record.read_text().splitlines() if ln.strip()]
             if record.exists() else [])
    return r, calls


def _predict_calls(calls):
    return [c for c in calls if c and c[0].endswith("run_eic.py")]


def _score_calls(calls):
    return [c for c in calls if c[:2] == ["-m", "candi.bench.external"]]


def _flag(argv, name):
    return argv[argv.index(name) + 1] if name in argv else None


def test_a_complete_b_root_is_scored_against_the_challenge_truth_without_predicting(tmp_path):
    """The pass that was unreachable: TRUTH=challenge over the B_ root the store pass wrote.

    Held-out only — `--chroms` comes from the TRUTH root's manifest, not from the prediction's 23,
    and no `--held-out-chroms` means no `genome_wide` block. B_ONCE is deliberately NOT set: a pass
    that predicts nothing cannot spend the one B_ prediction and does not claim to.
    """
    r, calls = _run_dispatch(tmp_path, panel="B_", scope="genomewide", truth="challenge",
                             pred_chroms=GW_CHROMS)
    assert r.returncode == 0, f"exited {r.returncode}:\n{r.stdout}\n{r.stderr}"
    assert not _predict_calls(calls), (
        f"a COMPLETE root was re-predicted under TRUTH=challenge: {_predict_calls(calls)}")
    assert "mode=score-only" in r.stdout, r.stdout
    assert "truth_chroms=chr20,chr21,chr22" in r.stdout, r.stdout
    scored = _score_calls(calls)
    assert len(scored) == 1, f"{len(scored)} score passes, not 1: {calls}"
    argv = scored[0]
    assert _flag(argv, "--truth-root") == str(tmp_path / "truth" / "B_"), argv
    assert _flag(argv, "--chroms") == "chr20,chr21,chr22", (
        f"the challenge pass scores {_flag(argv, '--chroms')!r}, not the truth root's own "
        f"chromosomes — read_truth_root_arrays would be sent after npz files that do not exist")
    assert "--held-out-chroms" not in argv, (
        f"the challenge json would carry a `genome_wide` block: {argv}")


def test_a_complete_root_scored_against_the_store_still_splits_off_the_held_out_scope(tmp_path):
    """Same root, TRUTH=store: still score-only, and still the genome-wide superset json."""
    r, calls = _run_dispatch(tmp_path, panel="B_", scope="genomewide", truth="store",
                             pred_chroms=GW_CHROMS)
    assert r.returncode == 0, f"exited {r.returncode}:\n{r.stdout}\n{r.stderr}"
    assert not _predict_calls(calls), "a COMPLETE root was re-predicted under TRUTH=store"
    assert "mode=score-only" in r.stdout and "truth_chroms=-" in r.stdout, r.stdout
    argv = _score_calls(calls)[0]
    assert _flag(argv, "--chroms") == ",".join(GW_CHROMS), argv
    assert _flag(argv, "--held-out-chroms") == "chr20,chr21,chr22", argv
    assert "--truth-root" not in argv, argv


def test_a_b_prediction_without_the_once_only_verb_is_refused(tmp_path):
    """Root absent, so a prediction WOULD be written — and that is what B_ONCE=1 licenses."""
    r, calls = _run_dispatch(tmp_path, panel="B_", scope="genomewide", truth="store")
    assert r.returncode == 4, f"exited {r.returncode}, not 4:\n{r.stdout}\n{r.stderr}"
    assert not calls, f"the refusal came after work was done: {calls}"
    assert "B_ONCE=1" in r.stderr, r.stderr


def test_a_b_prediction_is_refused_when_the_OTHER_b_root_holds_a_manifest(tmp_path):
    """The two-root rule survives the reordering: `.../B_` refuses a `.../B_.genomewide` predict."""
    r, calls = _run_dispatch(tmp_path, panel="B_", scope="genomewide", truth="store",
                             other_b_root_manifest=True, extra_env={"B_ONCE": "1"})
    assert r.returncode == 4, f"exited {r.returncode}, not 4:\n{r.stdout}\n{r.stderr}"
    assert not _predict_calls(calls), "a second B_ prediction was made"
    assert "already holds a manifest.json" in r.stderr, r.stderr


def test_the_v_panel_predicts_and_scores_exactly_as_before(tmp_path):
    """No B_ONCE, a B_ root full of manifests, and V_ is untouched by any of it."""
    r, calls = _run_dispatch(tmp_path, panel="V_", scope="genomewide", truth="store",
                             other_b_root_manifest=True)
    assert r.returncode == 0, f"exited {r.returncode}:\n{r.stdout}\n{r.stderr}"
    assert "mode=predict+score" in r.stdout, r.stdout
    predicts = _predict_calls(calls)
    assert len(predicts) == 1, f"{len(predicts)} predict passes, not 1: {calls}"
    assert _flag(predicts[0], "--batch-size") == "1024", predicts[0]
    assert predicts[0][-2:] == ["--chroms", "all"], (
        f"--chroms is nargs=+, so anything written after it is eaten: {predicts[0]}")
    argv = _score_calls(calls)[0]
    assert _flag(argv, "--chroms") == ",".join(GW_CHROMS), argv
    assert _flag(argv, "--held-out-chroms") == "chr20,chr21,chr22", argv
    assert "--truth-root" not in argv, argv


def test_a_truth_manifest_with_no_chroms_falls_back_to_the_held_out_three(tmp_path):
    """The converter writes `chroms`; a root whose manifest lost it is still the held-out three."""
    r, calls = _run_dispatch(tmp_path, panel="B_", scope="genomewide", truth="challenge",
                             pred_chroms=GW_CHROMS, truth_chroms=None)
    assert r.returncode == 0, f"exited {r.returncode}:\n{r.stdout}\n{r.stderr}"
    assert "falling back to chr20,chr21,chr22" in r.stderr, r.stderr
    assert _flag(_score_calls(calls)[0], "--chroms") == "chr20,chr21,chr22"


def test_the_b_guard_is_reached_only_when_a_prediction_would_be_written():
    """As text, so the ordering that the harness above exercises cannot silently come apart."""
    text = _code(EDICE / "eic_score.sh")
    assert 'if [ "$PANEL" = "B_" ] && [ "$complete" = "no" ]; then' in text, (
        "eic_score.sh's B_ guard is not gated on the completeness test, so scoring a complete "
        "root would be refused again")
    assert text.index("complete=no") < text.index('[ "$complete" = "no" ]; then'), (
        "the guard runs before the completeness test that decides whether it applies")
    assert text.index('"${B_ONCE:-0}" != "1"') > text.index("complete=no"), (
        "B_ONCE is demanded before it is known whether this invocation would predict at all")


def test_the_banner_says_which_of_the_two_modes_the_job_is():
    text = _code(EDICE / "eic_score.sh")
    assert "mode=$MODE" in text, "eic_score.sh's banner does not say predict+score or score-only"
    assert "truth_chroms=$TRUTH_CHROMS" in text, (
        "eic_score.sh's banner does not say which chromosomes the truth covers")
