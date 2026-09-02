"""t81 -- what the Avocado and Lavawizard launchers must say, in the scripts themselves.

These are TEXT checks, in the style of `tests/test_slurm_kit_pin.py`, and for the same reason: the
real assertion lives on Fir, inside a job nobody can run from a laptop, so what a test here can add
is that the script would have made the right call had it run.

Five things are held. Each one is a mistake that already happened once, on this benchmark or the
one before it:

  * A predict stage that walks the SHIPPED regime spends the B_ touch. BENCHMARK_DESIGN.md §5 reads
    the blind panel once, at the end, from the selected checkpoint. Every predict launcher must
    derive a panel-only regime with the one shared tool, and must guard B_ behind `B_ONCE`.
  * A σ table fit on V_ eval-pair residuals scores the track it was fit on. §7 fits σ on training
    residuals only; §12.2 declares every other σ VOID. Every score launcher must check the
    `training-residuals:` prefix, and no launcher may reach for the old eval-pair fitter or the
    override that used to wave it through.
  * A default that names a dead checkout runs some other branch's code. `CANDII_t78_code` is parked
    on t77; the programme's checkout is `CANDII_main`.
  * An output path invented per script drifts between stages. Every root a launcher defaults to is
    one of the programme's pinned roots.
  * A σ chromosome the shared stem cannot transfer into loses the whole stage after the cache build
    (Fir 57833682). Lavawizard's `sigma.sh` must choose one on the stem's `UPSTREAM_HYPERPARAMS`
    row and refuse one off it -- the rule itself is tested in `tests/test_sigma_integration.py`.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
METHODS = {"avocado": "Avocado", "lavawizard": "Lavawizard"}
#: The stages every rival now has. `sigma.sh` is the one added for the training-residual σ pass.
STAGES = ("_env.sh", "train.sh", "predict.sh", "score.sh", "sigma.sh")


def _slurm(method: str) -> Path:
    return ROOT / "competitors" / method / "slurm"


def _read(method: str, stage: str) -> str:
    return (_slurm(method) / stage).read_text(encoding="utf-8")


def _every(stage: str):
    return [pytest.param(m, id=f"{m}/{stage}") for m in METHODS]


@pytest.mark.parametrize("method", list(METHODS))
def test_every_stage_exists(method: str):
    """A glob that silently matches nothing would make every test below vacuously pass."""
    missing = [s for s in STAGES if not (_slurm(method) / s).is_file()]
    assert not missing, f"competitors/{method}/slurm is missing {missing}"


# --------------------------------------------------------------------------- the panel axis


@pytest.mark.parametrize("method", _every("predict.sh"))
def test_predict_derives_the_panel_regime_with_the_shared_tool(method: str):
    """One tool decides what a panel is, or two methods can disagree about the same word."""
    t = _read(method, "predict.sh")
    assert "declare_eval_pairs.py" in t and "split" in t, (
        f"{method}/predict.sh does not name `tools/declare_eval_pairs.py split`, so whatever it "
        f"predicts is not this programme's definition of a panel")
    assert "PANEL_REGIME" in t, (
        f"{method}/predict.sh names the split tool but does not predict from its output")


@pytest.mark.parametrize("method", _every("predict.sh"))
def test_predict_guards_the_blind_panel_behind_b_once(method: str):
    """§5 spends B_ once. A warning in a header is not a guard; `exit 4` is."""
    t = _read(method, "predict.sh")
    assert "B_ONCE" in t, f"{method}/predict.sh has no B_ONCE guard"
    assert "exit 4" in t, (
        f"{method}/predict.sh names B_ONCE but never refuses -- the guard has no teeth")
    assert "manifest.json" in t, (
        f"{method}/predict.sh does not test for an existing root, so a second B_ pass would "
        f"overwrite the first")


@pytest.mark.parametrize("method", _every("_env.sh"))
def test_the_panel_default_is_the_open_one(method: str):
    """V_ is rerunnable and B_ is not, so the default has to be V_ and the set has to be closed."""
    t = _read(method, "_env.sh")
    assert 'PANEL="${PANEL:-V_}"' in t, f"{method}/_env.sh does not default PANEL to V_"
    assert "PANEL must be V_ or B_" in t, (
        f"{method}/_env.sh accepts a PANEL it never validates, so a typo would score nothing")


# --------------------------------------------------------------------------- Rule 1 on sigma


@pytest.mark.parametrize("method", _every("score.sh"))
def test_score_refuses_a_sigma_that_is_not_a_training_residual_one(method: str):
    """A void σ table and a valid one look identical from outside. The file has to say which."""
    t = _read(method, "score.sh")
    assert "SIGMA_FITTED_ON_PREFIX" in t, (
        f"{method}/score.sh does not check `fitted_on`, so it would score against a σ fit on the "
        f"very tracks it is scoring")
    assert "fitted_on" in t
    assert "exit 3" in t, f"{method}/score.sh reads `fitted_on` but does not refuse on it"
    assert 'SIGMA="${SIGMA:-$SIGMA_JSON}"' in t, (
        f"{method}/score.sh does not default SIGMA to the pinned σ path")


@pytest.mark.parametrize("method", _every("_env.sh"))
def test_the_prefix_is_written_once_per_method(method: str):
    t = _read(method, "_env.sh")
    assert 'SIGMA_FITTED_ON_PREFIX="training-residuals:"' in t, (
        f"{method}/_env.sh does not pin the σ contract prefix")


@pytest.mark.parametrize("method", _every("sigma.sh"))
def test_sigma_derives_a_training_regime_and_fits_with_the_shared_pass(method: str):
    """The pinned seed and cell count are part of the σ table's identity, not a preference."""
    t = _read(method, "sigma.sh")
    assert "tools/sigma_training_regime.py" in t, (
        f"{method}/sigma.sh does not derive a training regime, so its residuals are not training "
        f"residuals")
    assert "--n-cells 12" in t and "--seed 890217" in t, (
        f"{method}/sigma.sh does not use the pinned sample -- two methods would fit on different "
        f"cells and their CRPS would not compare")
    assert "competitors.sigma_pass" in t, f"{method}/sigma.sh does not call the shared σ fitter"
    assert '--method "$METHOD"' in t, f"{method}/sigma.sh does not stamp a method on its σ table"
    assert f"METHOD={METHODS[method]}" in _read(method, "_env.sh"), (
        f"{method}/_env.sh does not set METHOD to the board slug {METHODS[method]!r}, so the σ "
        f"table and every pinned root would be keyed on the wrong name")


@pytest.mark.parametrize("stage", ["score.sh", "sigma.sh", "predict.sh", "train.sh", "_env.sh"])
@pytest.mark.parametrize("method", list(METHODS))
def test_no_launcher_keeps_the_void_sigma_path_or_its_override(method: str, stage: str):
    """§12.2 voided every σ fit on eval pairs. The fitter and its escape hatch go together."""
    t = _read(method, stage)
    for dead in ("fit_sigma.py", "SIGMA_RULE1_OVERRIDE"):
        assert dead not in t, (
            f"competitors/{method}/slurm/{stage} still names `{dead}`, which BENCHMARK_DESIGN.md "
            f"§7 and §12.2 rule out")


# ------------------------------------------------------- the σ chromosome (Lavawizard only)
#
# Lavawizard is the only method whose transferable widths are keyed per chromosome, so it is the
# only launcher that has to choose. Avocado's σ stage may take any training chromosome and these
# checks would be false there.


def test_lavawizard_sigma_chooses_its_chromosome_with_the_chooser_not_the_first_entry():
    """The old default was `${_SIG_ALL%%,*}` -- chr1 under eic_pilot, and 175 != 225."""
    t = _read("lavawizard", "sigma.sh")
    assert "lavawizard.sigma_chrom" in t, (
        "lavawizard/sigma.sh does not call `python -m lavawizard.sigma_chrom`, so nothing holds "
        "its σ chromosome to the row the shared stem borrows")
    assert 'SIGMA_CHROMS:-${_SIG_ALL%%,*}' not in t, (
        "lavawizard/sigma.sh still defaults its σ chromosome to the FIRST training chromosome of "
        "the regime, which is chr1 under eic_pilot and off the stem's row")
    assert 'lavawizard.sigma_chrom --regime "$REGIME"' in t, (
        "the chooser must read the SOURCE regime -- the derived σ regime's `eval_chroms` are the "
        "training slice, so its row is not the one the stem borrows")
    assert '--chroms "${SIGMA_CHROMS:-}"' in t, (
        "lavawizard/sigma.sh does not pass SIGMA_CHROMS through the chooser, so an operator's "
        "override would not be validated at all")


def test_lavawizard_sigma_refuses_an_off_row_chromosome_before_the_gpu():
    """A header note is not a guard. `exit 3` before the cache loop is."""
    t = _read("lavawizard", "sigma.sh")
    assert "exit 3" in t.split("# 2. predict the self-pairs")[0], (
        "lavawizard/sigma.sh names the chooser but does not refuse on it before the cache build "
        "and the position-table fit")
    assert "SIGMA_CHROMS" in t, "lavawizard/sigma.sh no longer honours a SIGMA_CHROMS override"


def test_lavawizard_sigma_documents_the_rule_and_prints_the_row():
    """The choice has to be readable in the header and in the job log, not only in the module."""
    t = _read("lavawizard", "sigma.sh")
    for phrase in ("UPSTREAM_HYPERPARAMS", "shared_hparams_chrom", "block1.dense"):
        assert phrase in t, f"lavawizard/sigma.sh's header does not name `{phrase}`"
    assert "_SIG_ROW" in t and "[sigma] on the shared stem" in t, (
        "lavawizard/sigma.sh does not print the chosen chromosome's row, so a log cannot be read "
        "back to see which row the σ table was fit under")


# --------------------------------------------------------------------------- the pinned roots


@pytest.mark.parametrize("method", list(METHODS))
def test_the_checkout_default_is_the_programmes_own(method: str):
    """`CANDII_t78_code` is parked on the t77 branch; a job defaulting there runs other code."""
    t = _read(method, "_env.sh")
    assert 'REPO="${REPO:-/project/def-maxwl/mforooz/CANDII_main}"' in t, (
        f"{method}/_env.sh does not default REPO to CANDII_main")
    assert "CANDII_t78_code" not in t


@pytest.mark.parametrize("method", list(METHODS))
def test_every_default_output_root_is_a_pinned_one(method: str):
    """A root invented per script drifts between stages, and a drift is silent."""
    t = _read(method, "_env.sh")
    for pinned in (
        'PRED_V="/scratch/mforooz/t81_pred/$METHOD/$REGIME_NAME/V_"',
        'PRED_B="/project/def-maxwl/mforooz/t81_pred_B/$METHOD/$REGIME_NAME/B_"',
        "/scratch/mforooz/t81_sigma/$METHOD/$REGIME_NAME/train_pred",
        'SIGMA_JSON="/project/def-maxwl/mforooz/t81_sigma/$METHOD/sigma_$REGIME_NAME.json"',
        'SCORES_DIR="/project/def-maxwl/mforooz/t81_scores/$METHOD/$REGIME_NAME"',
        "/project/def-maxwl/mforooz/t81_checkpoints/$METHOD/$REGIME_NAME",
    ):
        assert pinned in t, f"{method}/_env.sh does not pin `{pinned}`"


@pytest.mark.parametrize("method", _every("score.sh"))
def test_score_writes_the_pinned_scores_name_and_names_its_scope(method: str):
    """`<truth>.<panel>.json` is what the board reads, and held-out is the only scope these two
    methods have -- §4 blanks their genome-wide cell, so the launcher says so rather than
    inheriting a default."""
    t = _read(method, "score.sh")
    assert '$SCORES_DIR/${TRUTH}.${PANEL}.json' in t, (
        f"{method}/score.sh does not write the pinned scores name")
    assert "--chroms" in t, f"{method}/score.sh does not pass --chroms, so its scope is inherited"


@pytest.mark.parametrize("method", _every("train.sh"))
def test_train_keeps_the_selected_checkpoint_off_scratch(method: str):
    """$WS and $RUNS are working areas; the board row outlives them."""
    t = _read(method, "train.sh")
    assert "CKPT_KEEP" in t and "cp -p" in t, (
        f"{method}/train.sh does not copy its selected checkpoint to the pinned checkpoint root")
