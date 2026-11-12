"""miniCANDI batch contract (plan pinned interface C), the NB loss and the loss mask.

Every rung, the train loop, the dataset and the scorer import from here. Nothing in this file may
be renamed or reshaped without changing the pinned interface first.

Batch (dict of torch tensors; B windows, L bins, A = 36 tracks, K covariates):
  x_counts   int32   [B, L, A]   input counts, 0 where not given (and 0 on hidden bins)
  x_avail    bool    [B, A]      track given as input (False = missing or assay-masked)
  x_bin_mask bool    [B, L, A]   True = this bin of an available track is hidden from the input
  x_dna      uint8   [B, L*25]   codes A=0 C=1 G=2 T=3 N=4
  c_in       float32 [B, A, K]   input covariates C, 0 where x_avail is False
  c_out      float32 [B, A, K_META]  output covariates C' (never read-derived QC, D6)
  y_counts   int32   [B, L, A]   target counts
  y_avail    bool    [B, A]      a target exists
  y_kind     uint8   [B, A]      0 none, 1 imputation, 2 correction, 3 same-view reconstruction
  avg        float32 [B, L, A]   optional, leave-one-cell-out per-position average rate
  meta       list[dict]          cell, chrom, start_bin, measurement ids in and out:
                                 `inputs`, `targets` (lists of ids) and `measurements_in`,
                                 `measurements_out` (lists of length A, "" where none)
Optional keys added by M1 for two-step training (decision D15; an ADDITION to pinned interface C,
no pinned key changed), present only when config `two_step` is true:
  c_base     float32 [B, A, K_META]  C_base: the base view's covariates of every target's track
                                     (0 where the track has no target or no base view)
  y_base     int32   [B, L, A]       the base view's counts (thinned under base_depth = fixed)

Covariate columns hold RAW values (log2 depth in log2 reads, read length in bp, categorical ids
as floats); standardisation and embedding live in the network (`covariates.CovariateEmbedding`),
so the depth add-back can read column 0 (`log2_depth`) of `c_out` directly.

The NB loss reuses `tools/t118/ladder/model.py::nb_nll` (float64 inside, mean/size form, the
same clamps as the CAC testbed) by passing log(mu) and log(n).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_TOOLS = Path(__file__).resolve().parents[1]
for _p in (_TOOLS / "t118", _TOOLS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ladder.model import MU_MAX, MU_MIN, N_MAX, N_MIN  # noqa: E402
from ladder.model import nb_nll as _ladder_nb_nll  # noqa: E402

RES = 25
#: `CANDI_STORE/eic/manifest.json::assay_vocabulary` is written sorted (src/candi/store/manifest.py)
#: and equals `configs/panel.eic_full.json::assays`; the control follows at index 35.
ASSAYS = (
    "ATAC-seq", "DNase-seq", "H2AFZ", "H2AK5ac", "H2AK9ac", "H2BK120ac", "H2BK12ac", "H2BK15ac",
    "H2BK20ac", "H2BK5ac", "H3F3A", "H3K14ac", "H3K18ac", "H3K23ac", "H3K23me2", "H3K27ac",
    "H3K27me3", "H3K36me3", "H3K4ac", "H3K4me1", "H3K4me2", "H3K4me3", "H3K56ac", "H3K79me1",
    "H3K79me2", "H3K9ac", "H3K9me1", "H3K9me2", "H3K9me3", "H3T11ph", "H4K12ac", "H4K20me1",
    "H4K5ac", "H4K8ac", "H4K91ac",
)
CONTROL = "chipseq-control"
TRACKS = ASSAYS + (CONTROL,)
A = len(TRACKS)
CONTROL_INDEX = A - 1
TRACK_INDEX = {t: i for i, t in enumerate(TRACKS)}
assert A == 36 and CONTROL_INDEX == 35 and len(ASSAYS) == 35

BROAD_HISTONE = ("H3F3A", "H3K27me3", "H3K36me3", "H3K4me1", "H3K79me1", "H3K79me2", "H3K9me1",
                 "H3K9me2", "H3K9me3", "H4K20me1")
ASSAY_CLASSES = ("accessibility_dnase", "accessibility_atac", "broad_histone", "narrow_histone",
                 "control")


def assay_class(track: str) -> str:
    """Pinned interface B `assay_class` of a track name."""
    if track == "DNase-seq":
        return "accessibility_dnase"
    if track == "ATAC-seq":
        return "accessibility_atac"
    if track == CONTROL:
        return "control"
    if track in BROAD_HISTONE:
        return "broad_histone"
    if track in TRACK_INDEX:
        return "narrow_histone"
    raise KeyError(f"unknown track {track!r}")


#: pinned interface B, K_META = 11, in this order
META_COLUMNS = ("log2_depth", "run_type_pe", "read_length", "dedup_on", "mapq_threshold",
                "pipeline_id", "pipeline_version_id", "lab_id", "platform_id", "antibody_id",
                "donor_id")
#: D6 arm "QC on input only": appended to C only, never to C'
QC_COLUMNS = ("nsc", "rsc", "nrf", "pbc1", "pbc2", "frip", "frag_len")
K_META = len(META_COLUMNS)
K_QC = len(QC_COLUMNS)
assert K_META == 11 and K_QC == 7
LOG2_DEPTH_COL = 0

KIND_NONE, KIND_IMPUTE, KIND_CORRECT, KIND_SAME = 0, 1, 2, 3

DTYPES = {
    "x_counts": torch.int32, "x_avail": torch.bool, "x_bin_mask": torch.bool,
    "x_dna": torch.uint8, "c_in": torch.float32, "c_out": torch.float32,
    "y_counts": torch.int32, "y_avail": torch.bool, "y_kind": torch.uint8,
    "avg": torch.float32, "c_base": torch.float32, "y_base": torch.int32,
}
#: optional keys of pinned interface E (chunk XI), present only when their config key is set:
#: x_rna [B, L, 2], x_rna_avail [B], x_cell_vec [B, n_genes], x_cell_vec_avail [B], cell_id [B]
EXTRA_DTYPES = {"x_rna": torch.float32, "x_rna_avail": torch.bool, "x_cell_vec": torch.float32,
                "x_cell_vec_avail": torch.bool, "cell_id": torch.int64}
#: config keys that build an extra-input module in a rung; a rung sizes its trunk without them
EXTRA_KEYS = ("rna_track", "cell_vector", "cell_embedding")
REQUIRED = ("x_counts", "x_avail", "x_bin_mask", "x_dna", "c_in", "c_out", "y_counts",
            "y_avail", "y_kind", "meta")
CKPT_KEYS = ("state_dict", "config", "param_count", "windows_seen", "tables_sha256", "git_sha")


def k_input(config: dict | None) -> int:
    """K of `c_in`: K_META, plus the 7 QC columns when `config["qc_on_input"]` is true."""
    return K_META + (K_QC if (config or {}).get("qc_on_input") else 0)


def expected_shapes(B: int, L: int, K: int) -> dict:
    return {"x_counts": (B, L, A), "x_avail": (B, A), "x_bin_mask": (B, L, A),
            "x_dna": (B, L * RES), "c_in": (B, A, K), "c_out": (B, A, K_META),
            "y_counts": (B, L, A), "y_avail": (B, A), "y_kind": (B, A), "avg": (B, L, A),
            "c_base": (B, A, K_META), "y_base": (B, L, A)}


def validate_batch(batch: dict, K: int | None = None) -> tuple[int, int]:
    """Raise ValueError unless `batch` matches the contract exactly. Returns (B, L)."""
    missing = [k for k in REQUIRED if k not in batch]
    if missing:
        raise ValueError(f"batch lacks {missing}")
    B, L, a = batch["x_counts"].shape
    K = batch["c_in"].shape[-1] if K is None else K
    if K not in (K_META, K_META + K_QC):
        raise ValueError(f"c_in has K = {K}; want {K_META} or {K_META + K_QC}")
    shapes = expected_shapes(B, L, K)
    for key, dt in DTYPES.items():
        if key not in batch:
            continue
        t = batch[key]
        if tuple(t.shape) != shapes[key]:
            raise ValueError(f"{key}: shape {tuple(t.shape)}, want {shapes[key]}")
        if t.dtype != dt:
            raise ValueError(f"{key}: dtype {t.dtype}, want {dt}")
    if len(batch["meta"]) != B:
        raise ValueError(f"meta has {len(batch['meta'])} entries, want {B}")
    kind, y_avail, x_avail = batch["y_kind"], batch["y_avail"], batch["x_avail"]
    if ((kind > 0) != y_avail).any():
        raise ValueError("y_kind > 0 must equal y_avail")
    if (kind > KIND_SAME).any():
        raise ValueError("y_kind > 3")
    if ((kind == KIND_IMPUTE) & x_avail).any():
        raise ValueError("an imputation target (kind 1) has its track given as input")
    if (((kind == KIND_CORRECT) | (kind == KIND_SAME)) & ~x_avail).any():
        raise ValueError("a kind 2 or 3 target has no input track")
    if (batch["x_bin_mask"] & ~x_avail[:, None, :]).any():
        raise ValueError("x_bin_mask is set on a track that is not available")
    hidden_or_missing = batch["x_bin_mask"] | ~x_avail[:, None, :]
    if (batch["x_counts"][hidden_or_missing] != 0).any():
        raise ValueError("x_counts must be 0 where a track is missing or a bin is hidden")
    if (batch["c_in"][~x_avail] != 0).any():
        raise ValueError("c_in must be 0 where x_avail is False")
    if (batch["c_out"][~y_avail] != 0).any():
        raise ValueError("c_out must be 0 where y_avail is False")
    for key, dt in EXTRA_DTYPES.items():
        if key not in batch:
            continue
        t = batch[key]
        want = {"x_rna": (B, L, 2), "x_cell_vec": (B, t.shape[-1])}.get(key, (B,))
        if tuple(t.shape) != want or t.dtype != dt:
            raise ValueError(f"{key}: {t.dtype} {tuple(t.shape)}, want {dt} {want}")
    return B, L


def collate(windows: list[dict]) -> dict:
    """Stack single-window dicts (no B axis; `meta` a dict) into one contract batch."""
    out = {}
    for key in windows[0]:
        if key == "meta":
            out["meta"] = [w["meta"] for w in windows]
        else:
            out[key] = torch.stack([torch.as_tensor(w[key]) for w in windows])
    return out


def without_extras(cfg: dict | None) -> dict:
    """`cfg` without the extra-input keys: the config a rung's width search builds from, so an
    extra input never changes the trunk."""
    return {k: v for k, v in (cfg or {}).items() if k not in EXTRA_KEYS}


class CellFiLMEmbed(torch.nn.Module):
    """A covariate embedding followed by the cell-state FiLM of the current forward
    (`extra_inputs.apply_cell_state`); `MiniCandi.begin_extras` sets `film` per forward."""

    def __init__(self, base: torch.nn.Module):
        super().__init__()
        self.base = base
        self.film = None

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        from minicandi.extra_inputs import apply_cell_state
        e = self.base(c)
        if self.film is None:
            raise RuntimeError("cell-state embedding called before begin_extras(batch)")
        gamma, beta = self.film
        B = gamma.shape[0]
        return apply_cell_state(e.reshape(B, -1, e.shape[-1]), gamma, beta).reshape(e.shape)


class MiniCandi(torch.nn.Module):
    """Base class of every rung: `forward(batch) -> {"n": [B,L,A] > 0, "mu": [B,L,A] > 0}`.

    Extra inputs (Amendment 3, chunk XI). A rung calls `build_extras(cfg, d_dna, injectors)` as
    the LAST step of its construction, `begin_extras(batch)` first in `forward`, and passes its
    per-bin DNA features through `add_extra_track(dna, batch)`. With no key of `EXTRA_KEYS` set the
    three calls build nothing, draw nothing and return their input: the model is today's.
      rna_track       `extra_inputs.ExtraTrackStem(2, d_dna)` of `x_rna` (null row where
                      `x_rna_avail` is False), added to the DNA features of every bin. A rung that
                      has no DNA features (d_dna None, e.g. dna = off) raises NotApplicable.
      cell_vector / cell_embedding
                      one `extra_inputs.CellState(d_cov)`, whose (gamma, beta) FiLM the output of
                      the covariate embedding of every listed injector (input and output side):
                      each injector's `embed` is wrapped in `CellFiLMEmbed`.
    Both start as the identity (stem 0, gamma 1, beta 0), so at initialisation the model with the
    keys gives the output of the model without them under the same seed.
    """

    def forward(self, batch: dict) -> dict:  # pragma: no cover - abstract
        raise NotImplementedError

    def param_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def build_extras(self, cfg: dict, d_dna: int | None, injectors: tuple) -> None:
        from minicandi.components import NotApplicable
        from minicandi.extra_inputs import CellState, ExtraTrackStem, extra_kwargs
        kw = extra_kwargs(cfg)
        if kw["rna_track"] is not None:
            if d_dna is None:
                raise NotApplicable(f"{type(self).__module__}: rna_track is added to the per-bin "
                                    f"DNA features, and this configuration has none (dna = "
                                    f"{cfg.get('dna', 'own')!r})")
            self.rna_stem = ExtraTrackStem(d=int(d_dna), **kw["rna_track"])
        if kw["cell_state"] is not None:
            mods = [getattr(self, name) for name in injectors]
            d_cov = {m.embed.mlp[-1].out_features for m in mods}
            if len(d_cov) != 1:
                raise NotApplicable(f"cell state: the covariate embeddings differ in width {d_cov}")
            self.cell_state = CellState(d_cov.pop(), **kw["cell_state"])
            self._film_embeds = []
            for m in mods:
                m.embed = CellFiLMEmbed(m.embed)
                self._film_embeds.append(m.embed)

    def begin_extras(self, batch: dict) -> None:
        cs = getattr(self, "cell_state", None)
        if cs is not None:
            film = cs(batch)
            for e in self._film_embeds:
                e.film = film

    def add_extra_track(self, dna: torch.Tensor, batch: dict) -> torch.Tensor:
        stem = getattr(self, "rna_stem", None)
        if stem is None:
            return dna
        if "x_rna" not in batch:
            raise ValueError("rna_track is set but the batch has no `x_rna` (dataset config "
                             "rna_track / rna_h5_dir)")
        x = batch["x_rna"].to(dna.dtype)
        avail = batch.get("x_rna_avail")
        if avail is None:
            avail = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
        return dna + stem(x, avail).to(dna.dtype)


def nb_nll(n: torch.Tensor, mu: torch.Tensor, y: torch.Tensor,
           mask: torch.Tensor | None = None) -> torch.Tensor:
    """Mean NB negative log-likelihood of counts y under NB(mean mu, size n) over `mask`.

    p in the (n, p) form is n / (n + mu). mu is clamped to [1e-4, 1e6] and n to [1e-3, 1e4]
    (the CAC testbed's clamps). An empty mask gives a zero that still carries a graph.
    """
    per = _ladder_nb_nll(torch.log(mu.clamp_min(MU_MIN)), torch.log(n.clamp_min(N_MIN)),
                         y.to(mu.dtype))
    if mask is None:
        return per.mean()
    m = mask.to(torch.bool)
    if m.shape != per.shape:
        m = m.expand_as(per)
    k = int(m.sum())
    if k == 0:
        return (per * 0.0).sum()
    return per[m].sum() / k


def loss_mask(batch: dict) -> torch.Tensor:
    """bool [B, L, A]: kinds 1 and 2 on every bin; kind 3 only where x_bin_mask is True."""
    kind = batch["y_kind"]
    L = batch["y_counts"].shape[1]
    every = ((kind == KIND_IMPUTE) | (kind == KIND_CORRECT))[:, None, :].expand(-1, L, -1)
    same = (kind == KIND_SAME)[:, None, :] & batch["x_bin_mask"]
    return every | same


def make_ckpt(model: torch.nn.Module, config: dict, windows_seen: int, tables_sha256: str,
              git_sha: str) -> dict:
    """The pinned `ckpt.pt` dict; `torch.save(make_ckpt(...), path)`."""
    return {"state_dict": model.state_dict(), "config": dict(config),
            "param_count": int(sum(p.numel() for p in model.parameters())),
            "windows_seen": int(windows_seen), "tables_sha256": str(tables_sha256),
            "git_sha": str(git_sha)}


def check_ckpt(ck: dict) -> None:
    keys = tuple(ck.keys())
    if set(keys) != set(CKPT_KEYS):
        raise ValueError(f"ckpt keys {sorted(keys)}, want {sorted(CKPT_KEYS)}")


def wiring_check(model: torch.nn.Module, batch: dict, seed: int = 0) -> dict:
    """Copy-leak guardrail part 1 (a hard wiring check).

    Changes the counts under `x_bin_mask` and the counts of every track with `x_avail` False to
    random values, and reports the largest relative change of `n` and `mu`. The check passes at
    `max_rel_change <= 1e-6`. Run it with the model in eval mode.
    """
    g = torch.Generator().manual_seed(seed)
    hidden = batch["x_bin_mask"] | ~batch["x_avail"][:, None, :]
    pert = dict(batch)
    noise = torch.randint(1, 500, batch["x_counts"].shape, generator=g, dtype=torch.int32)
    pert["x_counts"] = torch.where(hidden, noise, batch["x_counts"])
    with torch.no_grad():
        a, b = model(batch), model(pert)
    worst = 0.0
    for key in ("n", "mu"):
        rel = ((a[key] - b[key]).abs() / a[key].abs().clamp_min(1e-12)).max().item()
        worst = max(worst, rel)
    return {"max_rel_change": worst, "n_perturbed": int(hidden.sum()),
            "pass": bool(worst <= 1e-6)}


__all__ = [
    "A", "ASSAYS", "ASSAY_CLASSES", "BROAD_HISTONE", "CKPT_KEYS", "CONTROL", "CONTROL_INDEX",
    "CellFiLMEmbed", "EXTRA_DTYPES", "EXTRA_KEYS", "without_extras",
    "K_META", "K_QC", "KIND_CORRECT", "KIND_IMPUTE", "KIND_NONE", "KIND_SAME", "LOG2_DEPTH_COL",
    "META_COLUMNS", "MU_MAX", "MU_MIN", "MiniCandi", "N_MAX", "N_MIN", "QC_COLUMNS", "RES",
    "TRACKS", "TRACK_INDEX", "assay_class", "check_ckpt", "collate", "expected_shapes",
    "k_input", "loss_mask", "make_ckpt", "nb_nll", "validate_batch", "wiring_check",
]
