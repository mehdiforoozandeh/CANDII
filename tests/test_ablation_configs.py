"""Ten named ablations of miniCANDI (chunk AB): every config under
tools/minicandi/configs/ablations/<ablation>/rung{2,3,4}/<arm>.json is the architecture's base recipe
(configs/final/rung<r>/reference_equal.json) plus exactly the declared keys of its arm, and builds
through the registry within 10% of 1.0e6 parameters."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

from minicandi import registry as R  # noqa: E402

ABL = _ROOT / "tools/minicandi/configs/ablations"
FINAL = _ROOT / "tools/minicandi/configs/final"
#: EXPECTED[ablation][arm] = the keys that differ from reference_equal.json (simpler_than_base aside)
EXPECTED = {
    "control_input": {"control_output_only": {"control_input"}},
    "predict_control": {"predict_control_off": {"predict_control"}},
    "average_input": {"average_input": {"average_input", "average_h5"}},
    "two_step": {"two_step_native": {"two_step"},
                 "two_step_fixed": {"two_step", "base_depth", "base_ref_log2_depth"}},
    "dna_source": {"dna_gnlm": {"dna", "dna_embedding_dim", "dna_embedding_h5"},
                   "dna_flashzoi": {"dna", "dna_embedding_dim", "dna_embedding_h5"},
                   "dna_shift5mb": {"dna_shift_bp"}},
    "dna_off": {"dna_off": {"dna"}},
    "cov_embedding": {"cov_numeric_only": {"covariate_embedding"},
                      "cov_dropout": {"covariate_dropout"},
                      "cov_numeric_per_value": {"covariate_embedding", "param_budget"}},
    "depth_input": {"input_depth_fixed": {"input_depth"}},
    "cov_injection": {"inject_adaln_zero": {"covariate_injection"}},
    "dna_fusion": {"fusion_film": {"dna_fusion"}, "fusion_late": {"dna_fusion"},
                   "fusion_xattn": {"dna_fusion"}, "fusion_early": {"dna_fusion"}},
    "losses": {"loss_impute_x4": {"loss_weights"}, "loss_poisson_mean": {"loss"},
               "loss_nb_crps": {"loss"}, "loss_bin_equal": {"loss_bin_equal"}},
    # close-out item A2: the basic controls on the three architectures
    "scrambled_covariates": {"scrambled_covariates": {"scramble_covariates"}},
    "mask_mode": {"mask_drop": {"mask_mode"}},
    "cov_injection_form": {"inject_concat": {"covariate_injection"},
                           "inject_tokens": {"covariate_injection"}},
    # close-out A3: the U-Net counterfactual-view arms (configs/views/rung4) on the conv models
    "views": {"base_only": {"train_views"}, "base_depth": {"train_views"},
              "base_fastq": {"train_views"}, "base_alignment": {"train_views"},
              "all_same_view_only": {"same_view_prob"}},
    # close-out A8: the 8 named transforms and the pseudocount point of the sweep
    "count_transforms": {**{f"tf_{t}": {"count_transform"} for t in (
        "log1p", "asinh", "anscombe", "pearson", "local_ratio", "quantile", "raw", "soft_clip")},
        "offset_eps0.1": {"offset_eps"}, "offset_eps0.01": {"offset_eps"}},
}
UNET_VIEWS = _ROOT / "tools/minicandi/configs/views/rung4"
#: arms outside 10% of 1.0e6 by design, with their exact count: the DNA-fusion arms keep the base
#: trunk width (item A11), and U-Net `fusion_early` drops the DNA tower (base 993,599, -24.5%)
OUT_OF_BAND = {("dna_fusion", 4, "fusion_early"): 750_149}
ON_DISK = sorted(ABL.glob("*/rung*/*.json"))


def _key(p: Path) -> tuple[str, int, str]:
    return p.parts[-3], int(p.parent.name[4:]), p.stem


def test_every_config_on_disk_is_declared():
    for p in ON_DISK:
        ab, r, arm = _key(p)
        assert r in (2, 3, 4), p
        assert arm in EXPECTED.get(ab, {}), p


@pytest.mark.parametrize("path", ON_DISK, ids=lambda p: "/".join(p.parts[-3:]))
def test_config_is_reference_plus_declared_keys_and_in_band(path):
    ab, r, arm = _key(path)
    ref = json.loads((FINAL / f"rung{r}" / "reference_equal.json").read_text())
    cfg = json.loads(path.read_text())
    changed = {k for k in set(ref) | set(cfg) if cfg.get(k) != ref.get(k)} - {"simpler_than_base"}
    assert changed == EXPECTED[ab][arm]
    torch.manual_seed(0)
    n = sum(p.numel() for p in R.build_model(cfg).parameters() if p.requires_grad)
    if (ab, r, arm) in OUT_OF_BAND:     # the arm keeps the base width by rule; count stated
        assert n == OUT_OF_BAND[(ab, r, arm)], n
        return
    assert abs(n - 1.0e6) <= 1.0e5, n


def test_inject_tokens_is_not_applicable_on_the_conv_autoencoder():
    """A2: the conv autoencoder has no attention that reads covariate tokens, so the tokens arm
    exists only for conv + attention and U-Net."""
    from minicandi.components import NotApplicable
    assert not (ABL / "cov_injection_form/rung2/inject_tokens.json").exists()
    ref = json.loads((FINAL / "rung2" / "reference_equal.json").read_text())
    with pytest.raises(NotApplicable):
        R.build_model({**ref, "covariate_injection": "tokens"})


@pytest.mark.parametrize("path", sorted(ABL.glob("views/rung*/*.json")),
                         ids=lambda p: "/".join(p.parts[-2:]))
def test_view_arm_matches_the_unet_arm(path):
    """Each conv-model view arm sets the view keys exactly as the U-Net arm of the same name."""
    cfg = json.loads(path.read_text())
    unet = json.loads((UNET_VIEWS / f"views_{path.stem}.json").read_text())
    for k in ("train_views", "same_view_prob"):
        assert cfg.get(k) == unet.get(k), k
