"""The Lavawizard / Guacamole architecture in PyTorch.

Transcribed from `02_guacamole6_pretrain.py` (`Precamole`) and `03_guacamole6_train.py`
(`Guacamole`) at upstream commit `d638b204`. Every layer keeps its Keras name, because
`keras_weights.load_keras_h5` maps their `.h5` onto this module **by name** and a rename here is a
silently wrong parity check there.

Four details are load-bearing and none of them are PyTorch defaults:

1. **The block order is Dense → PReLU → BatchNorm → Dropout.** The activation comes *before* the
   normalisation. That is unusual, and it is what they trained, so it is what we reproduce.
2. **BatchNorm epsilon is 1e-3**, Keras's default, not torch's 1e-5. Keras `momentum=0.99` is
   torch `momentum=0.01` — the two libraries name the complementary quantity.
3. **PReLU has one alpha per unit** (Keras `shared_axes=None` over a `(batch, 2048)` input), so
   `num_parameters=2048`, not torch's default 1.
4. **The average is added to the output.** `y = Dense(1)(x) + average` — the cross-cell average is
   the base prediction and the network learns a correction on top of it. The same scalar also
   enters `dense_3` through `concat_last`, so it reaches the output by two paths; only the first
   is exactly additive.

The signal space is `arcsinh(-log10 p)` throughout. Inversion (`sinh`, then clip at 0) belongs to
whoever writes tracks, not here — see `emit.py`.

**One thing here is ours and not upstream's: the parameter split.** `Factors.GENOME_TABLES`,
`Guacamole.genome_parameters` and `Guacamole.transferable_parameters` cut the model in two along
BENCHMARK_DESIGN.md §2's line — position tables on one side, everything Rule 2 binds on the other.
Upstream never needed the cut because it fit one whole model per chromosome. Nothing about the
architecture, the forward pass or the Keras names changes; the methods only name a partition of
tensors that was always there. See `train.py` for what the two stages then do with it.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["KERAS_BN_EPS", "KERAS_BN_MOMENTUM", "DenseBlock", "Factors", "Precamole", "Guacamole"]

#: Keras 2.2.4 `BatchNormalization` defaults, read off their own `model_config` rather than assumed.
KERAS_BN_EPS = 1e-3
#: Keras `momentum=0.99` is the decay of the running estimate; torch's `momentum` is `1 - decay`.
KERAS_BN_MOMENTUM = 0.01


class DenseBlock(nn.Module):
    """`Dense → PReLU → BatchNorm → Dropout`, the unit repeated three times upstream.

    Three real call sites (`dense_1`, `dense_2`, `dense_3`), which is what earns it a class.
    """

    def __init__(self, in_features: int, out_features: int, dropout: float):
        super().__init__()
        self.dense = nn.Linear(in_features, out_features)
        self.ac = nn.PReLU(num_parameters=out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=KERAS_BN_EPS, momentum=KERAS_BN_MOMENTUM)
        self.dp = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dp(self.bn(self.ac(self.dense(x))))


class Factors(nn.Module):
    """The five embedding tables, flattened and concatenated in upstream's order.

    Kept as one module for both `Precamole` and `Guacamole` rather than duplicated into each. Two
    call sites is under the usual bar for a helper, but the concatenation ORDER is exactly what
    `keras_weights` relies on to slice `dense_1/kernel:0` correctly, and two copies of an order is
    two chances to get it wrong.
    """

    #: The three tables indexed by POSITION. BENCHMARK_DESIGN.md §2 counts fitting these on an eval
    #: chromosome as inference, not training, so they are the half a transfer throws away and
    #: refits. Everything else in the model — the cell table, the assay table and the dense network
    #: — is a *transferable* parameter and is bound by Rule 2. Named here rather than in `train.py`
    #: because the split is a fact about the architecture.
    GENOME_TABLES = ("genome_25bp_embedding", "genome_250bp_embedding", "genome_5kbp_embedding")

    def __init__(self, n_celltypes: int, n_assays: int, n_positions: int,
                 n_celltype_factors: int = 45, n_assay_factors: int = 65,
                 n_25bp_factors: int = 25, n_250bp_factors: int = 30, n_5kbp_factors: int = 60):
        super().__init__()
        self.n_positions = int(n_positions)
        self.celltype_embedding = nn.Embedding(n_celltypes, n_celltype_factors)
        self.assay_embedding = nn.Embedding(n_assays, n_assay_factors)
        # Upstream sizes the coarse tables `n // 10 + 1` and `n // 200 + 1` with integer division on
        # a float (`int(n_positions / 10)`). For every chromosome the two agree; reproduced as
        # integer division so the table sizes match their checkpoints exactly.
        self.genome_25bp_embedding = nn.Embedding(n_positions, n_25bp_factors)
        self.genome_250bp_embedding = nn.Embedding(n_positions // 10 + 1, n_250bp_factors)
        self.genome_5kbp_embedding = nn.Embedding(n_positions // 200 + 1, n_5kbp_factors)
        #: Width of the concatenation — `dense_1`'s input, 225 with the defaults.
        self.out_features = (n_celltype_factors + n_assay_factors
                             + n_25bp_factors + n_250bp_factors + n_5kbp_factors)

    def genome_tables(self):
        return [getattr(self, n) for n in self.GENOME_TABLES]

    def forward(self, celltype: Tensor, assay: Tensor, pos25: Tensor) -> Tensor:
        """`pos25` is the absolute 25 bp bin index; the coarse indices are derived, as upstream."""
        return torch.cat([
            self.celltype_embedding(celltype),
            self.assay_embedding(assay),
            self.genome_25bp_embedding(pos25),
            self.genome_250bp_embedding(torch.div(pos25, 10, rounding_mode="floor")),
            self.genome_5kbp_embedding(torch.div(pos25, 200, rounding_mode="floor")),
        ], dim=-1)


class Precamole(nn.Module):
    """Stage 1 — three-way classification over terciles of the arcsinh signal.

    The one-hot tercile of the average track is added to the three logits before the softmax, the
    same skip `Guacamole` uses on the scalar. Returns **logits**; upstream's softmax lives in the
    Keras loss and `nn.CrossEntropyLoss` wants logits anyway.
    """

    def __init__(self, n_celltypes: int, n_assays: int, n_positions: int, **factor_sizes):
        super().__init__()
        self.factors = Factors(n_celltypes, n_assays, n_positions, **factor_sizes)
        self.block1 = DenseBlock(self.factors.out_features, 2048, dropout=0.5)
        self.block2 = DenseBlock(2048, 2048, dropout=0.5)
        self.y_pred_dense = nn.Linear(2048, 3)

    def forward(self, celltype: Tensor, assay: Tensor, pos25: Tensor,
                average_onehot: Tensor) -> Tensor:
        x = self.block2(self.block1(self.factors(celltype, assay, pos25)))
        return self.y_pred_dense(x) + average_onehot


class Guacamole(nn.Module):
    """Stage 2 — scalar regression in `arcsinh(-log10 p)`.

    Upstream builds this by loading the stage-1 checkpoint, popping four layers (which leaves
    `dense_2_dp` as the trunk output) and stacking a third block that also takes the per-bin
    average and variance. `Model(inputs, outputs=y)` then rebuilds the graph from the new output
    backwards, which drops the stage-1 head. `from_precamole` does that transfer directly instead
    of re-deriving it from a saved file.
    """

    def __init__(self, n_celltypes: int, n_assays: int, n_positions: int, **factor_sizes):
        super().__init__()
        self.factors = Factors(n_celltypes, n_assays, n_positions, **factor_sizes)
        self.block1 = DenseBlock(self.factors.out_features, 2048, dropout=0.5)
        self.block2 = DenseBlock(2048, 2048, dropout=0.5)
        # `concat_last` puts variance first, then average, then the trunk: 2 + 2048 = 2050.
        self.block3 = DenseBlock(2048 + 2, 2048, dropout=0.7)
        self.y_pred = nn.Linear(2048, 1)

    def forward(self, celltype: Tensor, assay: Tensor, pos25: Tensor,
                average: Tensor, variance: Tensor) -> Tensor:
        """All of `celltype`, `assay`, `pos25` are `(B,)` long; `average`, `variance` are `(B,)`
        float. Returns `(B,)` in `arcsinh(-log10 p)`, un-clipped."""
        x = self.block2(self.block1(self.factors(celltype, assay, pos25)))
        x = torch.cat([variance.unsqueeze(-1), average.unsqueeze(-1), x], dim=-1)
        return self.y_pred(self.block3(x)).squeeze(-1) + average

    # -- the Rule 2 parameter split -------------------------------------------------------------
    #
    # Prefix of every tensor in `state_dict()` that belongs to a position table. Derived from
    # `Factors.GENOME_TABLES` rather than typed out, so "which half is which" is written down once
    # and cannot drift between the freeze, the load and the state filter.
    _GENOME_PREFIXES = tuple(f"factors.{n}." for n in Factors.GENOME_TABLES)

    def genome_parameters(self):
        return [p for t in self.factors.genome_tables() for p in t.parameters()]

    def transferable_parameters(self):
        """Cell factors, assay factors and the whole dense network — everything Rule 2 binds."""
        genome = {id(p) for p in self.genome_parameters()}
        return [p for p in self.parameters() if id(p) not in genome]

    def transferable_state(self, state=None) -> dict:
        """`state_dict()` minus the position tables — what one scope hands the next."""
        state = self.state_dict() if state is None else state
        return {k: v for k, v in state.items() if not k.startswith(self._GENOME_PREFIXES)}

    def freeze_transferable(self) -> "Guacamole":
        """Stop the transferable half learning. Chains, so a caller reads as one statement.

        BatchNorm's running estimates are NOT parameters and keep moving in `train()` mode, so the
        three blocks are put in `eval()` too: a frozen trunk whose normalisation statistics drift to
        the eval chromosome's would be fitting the trunk to that chromosome by another route.
        """
        for p in self.transferable_parameters():
            p.requires_grad_(False)
        for b in (self.block1, self.block2, self.block3):
            b.bn.eval()
        return self

    @torch.no_grad()
    def load_transferable(self, state: dict) -> "Guacamole":
        """Copy another scope's transferable half in, leaving the position tables untouched.

        Strict on its own half: every transferable tensor must be present and the right shape, so a
        checkpoint from a model with different factor widths — `dataset3.UPSTREAM_HYPERPARAMS`
        varies them by chromosome, and they set `dense_1`'s input width — fails here rather than
        producing a silently mis-shaped transfer.
        """
        incoming = self.transferable_state(state)
        want = set(self.transferable_state())
        missing, extra = sorted(want - set(incoming)), sorted(set(incoming) - want)
        if missing or extra:
            raise ValueError(f"the checkpoint's transferable half is not this model's: "
                             f"missing {missing}, unexpected {extra}")
        res = self.load_state_dict(incoming, strict=False)
        assert not res.unexpected_keys, res.unexpected_keys
        return self

    @torch.no_grad()
    def from_precamole(self, pre: Precamole) -> "Guacamole":
        """Copy the stage-1 trunk in, exactly as upstream's four `layers.pop()` calls do.

        The stage-1 head (`y_pred_dense`) is dropped, and `block3` / `y_pred` keep their fresh
        initialisation. Returns self, so it chains.
        """
        self.factors.load_state_dict(pre.factors.state_dict())
        self.block1.load_state_dict(pre.block1.state_dict())
        self.block2.load_state_dict(pre.block2.state_dict())
        return self
