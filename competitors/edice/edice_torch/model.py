"""eDICE, reimplemented in PyTorch.

Reference implementation: github.com/alex-hh/eDICE @ 5e4e3f2 (MIT, TensorFlow 2.11), read but
never run. Paper: Hawkins-Hooker et al., Nat Commun 14:4750 (2023).

The reference is the authority for semantics wherever it and the paper could be read differently.
The places that matter are marked `# FIDELITY:` below and are repeated in `../README.md`.

Shapes. One example is one 25 bp genomic bin. Within a bin the model sees a bag of observed
(cell, assay, value) triples -- the *supports* -- and is asked for the values at a set of
(cell, assay) *targets*.

    supports            (B, S)      float, already arcsinh-transformed
    support_cell_ids    (B, S)      long
    support_assay_ids   (B, S)      long
    target_cell_ids     (B, T)      long
    target_assay_ids    (B, T)      long
    -> prediction       (B, T)      float, in arcsinh space

The model is a two-sided factorisation. One tower treats CELLS as the nodes of a graph whose
features are assays; the mirror tower treats ASSAYS as nodes whose features are cells. Each tower
turns the bin's observations into one embedding per node, refines them with self-attention, reads
out the embeddings the targets ask for, and the decoder MLP consumes the concatenated pair.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "dense", "MultiHeadAttention", "CrossTransformerBlock", "SelfTransformerBlock",
    "SignalEmbedder", "CrossContextualSignalEmbedder", "CellAssayCrossFactoriser", "MASK_LOGIT",
]

#: FIDELITY: the reference adds `mask * -1e9` to the attention logits (models/attention.py:46), not
#: -inf. The difference is visible exactly when a query's keys are ALL masked -- with -1e9 the
#: softmax stays uniform and finite, with -inf it is NaN. A bin in which no cell carries an
#: observation is not hypothetical (the Roadmap h5 has all-zero rows), so this constant is load
#: bearing and is copied rather than modernised.
MASK_LOGIT = -1e9


def dense(in_features: int, out_features: int) -> nn.Linear:
    """A `tf.keras.layers.Dense` with Keras' defaults, not PyTorch's.

    FIDELITY: Keras initialises a Dense kernel with `glorot_uniform` and its bias with zeros;
    `nn.Linear` uses kaiming-uniform(a=sqrt(5)) for both. On the 512->2048->2048->1 decoder the two
    differ by ~1.7x in initial weight scale, which is not a detail you can wave away when the
    acceptance gate is "land within the published numbers". So we set Keras' scheme explicitly.
    """
    layer = nn.Linear(in_features, out_features)
    limit = math.sqrt(6.0 / (in_features + out_features))
    nn.init.uniform_(layer.weight, -limit, limit)
    nn.init.zeros_(layer.bias)
    return layer


def _mlp(d_in: int, d_output: int, d_hidden: int, n_hidden_layers: int = 1,
         dropout: float = 0.0) -> nn.Sequential:
    """`models/layers.py::output_mlp`, with the input width made explicit.

    Keras Dense infers `in_features`; PyTorch does not, so the caller passes it. Layer ORDER is the
    reference's: every hidden layer is Dense->ReLU->Dropout, and the final projection carries no
    activation and no dropout.
    """
    layers: list = []
    width = d_in
    for _ in range(n_hidden_layers):
        layers.append(dense(width, d_hidden))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        width = d_hidden
    layers.append(dense(width, d_output))
    return nn.Sequential(*layers)


class MultiHeadAttention(nn.Module):
    """`models/attention.py::MultiHeadAttention` -- the TF transformer-tutorial version.

    Four projections of width `d_model` (q, k, v, out); logits scaled by 1/sqrt(depth) where
    `depth = d_model // num_heads` -- i.e. the PER-HEAD width, applied after the head split, which
    is what the reference does and is the usual convention.
    """

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError(f"d_model {d_model} is not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        self.wq = dense(d_model, d_model)
        self.wk = dense(d_model, d_model)
        self.wv = dense(d_model, d_model)
        self.out = dense(d_model, d_model)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, length, _ = x.shape
        return x.view(b, length, self.num_heads, self.depth).transpose(1, 2)

    def forward(self, v: torch.Tensor, k: torch.Tensor, q: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """`mask` is the (B, n_keys) node mask: 1 where the key is to be EXCLUDED."""
        b = q.shape[0]
        q = self._split_heads(self.wq(q))
        k = self._split_heads(self.wk(k))
        v = self._split_heads(self.wv(v))

        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.depth))
        if mask is not None:
            if mask.dim() != 2:
                raise ValueError(f"node mask must be rank 2 (B, n_keys), got {tuple(mask.shape)}")
            logits = logits + mask[:, None, None, :] * MASK_LOGIT
        weights = torch.softmax(logits, dim=-1)

        attended = torch.matmul(weights, v)                       # (B, H, Lq, depth)
        attended = attended.transpose(1, 2).reshape(b, -1, self.d_model)
        return self.out(attended), weights


class SelfTransformerBlock(nn.Module):
    """`models/layers.py::NoLNTransformerBlock`.

    FIDELITY: there is NO layer norm. The Roadmap configuration in the reference's own training
    script pins `layer_norm_type=None` (scripts/train_roadmap.py:15), and the block it selects has
    none to disable. Residual-plus-FFN with nothing normalising in between is unusual for a 2023
    transformer, so it reads like an oversight -- it is not ours to correct.
    """

    def __init__(self, d_model: int, num_heads: int, dff: int, rate: float = 0.1,
                 ffn_dropout: float = 0.0) -> None:
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = _mlp(d_model, d_model, dff, n_hidden_layers=1, dropout=ffn_dropout)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn, self.attn_weights = self.mha(x, x, x, mask)
        out1 = x + self.dropout1(attn)
        return out1 + self.dropout2(self.ffn(out1))


class CrossTransformerBlock(nn.Module):
    """`models/layers.py::CrossTransformerBlock` -- filtered self-attention.

    The queries are the rows of `x` the targets name, so the block costs O(T x n_nodes) instead of
    O(n_nodes^2) and returns one refined embedding per target. Keys and values remain the FULL node
    set: a target cell attends to every cell in the panel, itself included.
    """

    def __init__(self, d_model: int, num_heads: int, dff: int, rate: float = 0.1,
                 ffn_dropout: float = 0.0) -> None:
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = _mlp(d_model, d_model, dff, n_hidden_layers=1, dropout=ffn_dropout)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x: torch.Tensor, query_ids: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # batched gather: (B, n_nodes, d) indexed by (B, T) -> (B, T, d)
        idx = query_ids.long().unsqueeze(-1).expand(-1, -1, x.shape[-1])
        x_q = torch.gather(x, 1, idx)
        attn, self.attn_weights = self.mha(x, x, x_q, mask)
        out1 = x_q + self.dropout1(attn)
        return out1 + self.dropout2(self.ffn(out1))


class SignalEmbedder(nn.Module):
    """`models/embedders.py::SignalEmbedder` -- the bin's observations, as one vector per node.

    Scatter the observed values into a dense (B, n_nodes, n_feats) matrix, divide each node's row by
    that node's observation COUNT, project to `embed_dim` with a ReLU Dense, and add a learned
    per-node global embedding. Also returns the mask of nodes that observed nothing.
    """

    def __init__(self, n_nodes: int, n_feats: int, embed_dim: int = 32,
                 add_global_embedding: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.n_feats = n_feats
        self.embed_dim = embed_dim
        self.nodewise_hidden = dense(n_feats, embed_dim)
        self.add_global_embedding = add_global_embedding
        if add_global_embedding:
            # FIDELITY: Keras' `embeddings_initializer='uniform'` is RandomUniform(-0.05, 0.05).
            # `nn.Embedding` would give N(0, 1) -- twenty times wider, and added straight onto a
            # ReLU output, so the tower would start dominated by the prior instead of the signal.
            self.global_embeddings = nn.Parameter(
                torch.empty(n_nodes, embed_dim).uniform_(-0.05, 0.05))
        self.dropout = nn.Dropout(dropout)

    def expand(self, values: torch.Tensor, node_ids: torch.Tensor,
               feat_ids: torch.Tensor) -> torch.Tensor:
        """`models/layers.py::InputExpander` -- scatter (B, S) into (B, n_nodes, n_feats)."""
        b = values.shape[0]
        flat = node_ids.long() * self.n_feats + feat_ids.long()
        out = values.new_zeros(b, self.n_nodes * self.n_feats)
        # scatter_ADD, matching tf.scatter_nd: a repeated (node, feat) pair sums. The reference
        # relies on that being unreachable (a track is one cell x one assay), and so do we -- but
        # summing is what it does, so summing is what we do.
        out.scatter_add_(1, flat, values)
        return out.view(b, self.n_nodes, self.n_feats)

    def forward(self, values: torch.Tensor, node_ids: torch.Tensor,
                feat_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = self.expand(values, node_ids, feat_ids)
        mask_mat = self.expand(torch.ones_like(values), node_ids, feat_ids)
        counts = mask_mat.sum(dim=-1, keepdim=True)                     # (B, n_nodes, 1)
        obs = obs * mask_mat
        # FIDELITY: tf.math.divide_no_nan -- an unobserved node's row is 0/0 = 0, not NaN.
        obs = torch.where(counts > 0, obs / counts.clamp(min=1.0), torch.zeros_like(obs))
        missing = (counts.squeeze(-1) == 0).to(obs.dtype)               # (B, n_nodes)

        embedded = F.relu(self.nodewise_hidden(obs))
        if self.add_global_embedding:
            embedded = embedded + self.global_embeddings
        return self.dropout(embedded), missing


class CrossContextualSignalEmbedder(nn.Module):
    """`models/embedders.py::CrossContextualSignalEmbedder` -- one tower.

    `n_attn_layers - 1` self-attention blocks over all nodes, then one cross-attention block that
    reads out only the queried nodes. The Roadmap configuration uses `n_attn_layers=1`, so in
    practice the tower is the embedder plus a single cross block.
    """

    def __init__(self, n_nodes: int, n_feats: int, embed_dim: int = 32, n_attn_layers: int = 1,
                 n_attn_heads: int = 4, intermediate_fc_dim: int = 128,
                 transformer_dropout: float = 0.1, intermediate_fc_dropout: float = 0.0,
                 embedding_dropout: float = 0.0, add_global_embedding: bool = True) -> None:
        super().__init__()
        if n_attn_layers < 1:
            raise ValueError("n_attn_layers must be >= 1")
        self.signal_embedder = SignalEmbedder(n_nodes, n_feats, embed_dim=embed_dim,
                                              dropout=embedding_dropout,
                                              add_global_embedding=add_global_embedding)
        self.self_blocks = nn.ModuleList([
            SelfTransformerBlock(embed_dim, n_attn_heads, intermediate_fc_dim,
                                 rate=transformer_dropout, ffn_dropout=intermediate_fc_dropout)
            for _ in range(n_attn_layers - 1)
        ])
        self.cross_block = CrossTransformerBlock(
            embed_dim, n_attn_heads, intermediate_fc_dim,
            rate=transformer_dropout, ffn_dropout=intermediate_fc_dropout)

    def forward(self, values: torch.Tensor, node_ids: torch.Tensor, feat_ids: torch.Tensor,
                query_ids: torch.Tensor) -> torch.Tensor:
        embedded, mask = self.signal_embedder(values, node_ids, feat_ids)
        for block in self.self_blocks:
            embedded = block(embedded, mask=mask)
        return self.cross_block(embedded, query_ids, mask=mask)


class CellAssayCrossFactoriser(nn.Module):
    """`models/predictors.py::CellAssayCrossFactoriser` -- the whole model.

    At the Roadmap settings (127 cells, 24 assays, embed_dim 256, decoder 2x2048) this comes to
    5,985,025 parameters, against the "~ 6M" of the paper's Supplementary Table 1. That count is a
    unit test, not a remark: it is the cheapest end-to-end check that the reimplementation has the
    same weights as the thing it claims to be.
    """

    def __init__(self, n_cells: int, n_assays: int, embed_dim: int = 256, n_attn_layers: int = 1,
                 n_attn_heads: int = 4, intermediate_fc_dim: int = 128, decoder_layers: int = 2,
                 decoder_hidden: int = 2048, decoder_dropout: float = 0.3,
                 transformer_dropout: float = 0.1, intermediate_fc_dropout: float = 0.0,
                 embedding_dropout: float = 0.0) -> None:
        super().__init__()
        self.n_cells = n_cells
        self.n_assays = n_assays
        tower = dict(embed_dim=embed_dim, n_attn_layers=n_attn_layers, n_attn_heads=n_attn_heads,
                     intermediate_fc_dim=intermediate_fc_dim,
                     transformer_dropout=transformer_dropout,
                     intermediate_fc_dropout=intermediate_fc_dropout,
                     embedding_dropout=embedding_dropout)
        # cells are nodes, assays are features -- and the mirror image.
        self.cell_embedder = CrossContextualSignalEmbedder(n_cells, n_assays, **tower)
        self.assay_embedder = CrossContextualSignalEmbedder(n_assays, n_cells, **tower)
        self.decoder = _mlp(2 * embed_dim, 1, decoder_hidden,
                            n_hidden_layers=decoder_layers, dropout=decoder_dropout)

    def forward(self, supports: torch.Tensor, support_cell_ids: torch.Tensor,
                support_assay_ids: torch.Tensor, target_cell_ids: torch.Tensor,
                target_assay_ids: torch.Tensor) -> torch.Tensor:
        cell_emb = self.cell_embedder(supports, support_cell_ids, support_assay_ids,
                                      target_cell_ids)
        assay_emb = self.assay_embedder(supports, support_assay_ids, support_cell_ids,
                                        target_assay_ids)
        return self.decoder(torch.cat([cell_emb, assay_emb], dim=-1)).squeeze(-1)
