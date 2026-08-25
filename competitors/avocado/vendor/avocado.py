#!/usr/bin/env python3
"""Avocado (Schreiber et al. 2020, Genome Biology 21:81) in PyTorch.

This is a re-implementation, not the authors' code.  The published
implementation (jmschrei/avocado) is Keras/TF1-era and does not install against
Fir's module stack; since the point of this experiment is a *retrained* Avocado
whose two copies differ only in their training data, an implementation we can
run twice matters more than byte-compatibility with the original.  The
architecture below follows the paper:

  cell-type factors     32 per cell type
  assay factors        256 per assay
  genomic factors       25 at 25 bp, 40 at 250 bp, 45 at 5 kbp resolution
  predictor            concat (398) -> 2048 ReLU -> 2048 ReLU -> 1

and the training target is the arcsinh-transformed signal, inverted with sinh at
prediction time.

Two things are worth stating because they shape everything downstream:

1. The genomic factors are *per-position free parameters*.  A position the model
   never saw has no representation, so Avocado cannot extrapolate to unseen
   loci: to predict genome-wide, factors must be fitted genome-wide.  The paper
   handles this by training the whole model on one chromosome and then, for
   every other chromosome, freezing the cell-type factors, assay factors and
   network and fitting only that chromosome's genomic factors.  We do the same
   (`freeze_shared`), which is also what makes the genome-wide fit parallel.

2. Every blind-test cell type has at least one train/validation experiment (1 to
   11 of them), so all 12 have a learnable cell-type embedding.  Nothing here
   ever sees a blind-test track.
"""
import math

import torch
import torch.nn as nn

N_CELL_FACTORS = 32
N_ASSAY_FACTORS = 256
N_G25, N_G250, N_G5K = 25, 40, 45
HIDDEN = 2048

G250_STRIDE = 10        # 250 bp / 25 bp
G5K_STRIDE = 200        # 5 kbp / 25 bp

N_GENOME_FACTORS = N_G25 + N_G250 + N_G5K            # 110
N_PAIR_FACTORS = N_CELL_FACTORS + N_ASSAY_FACTORS    # 288


class Avocado(nn.Module):
    def __init__(self, n_cells, n_assays, n_bins):
        super().__init__()
        self.n_bins = n_bins
        self.cell = nn.Embedding(n_cells, N_CELL_FACTORS)
        self.assay = nn.Embedding(n_assays, N_ASSAY_FACTORS)
        self.g25 = nn.Embedding(n_bins, N_G25)
        self.g250 = nn.Embedding(math.ceil(n_bins / G250_STRIDE), N_G250)
        self.g5k = nn.Embedding(math.ceil(n_bins / G5K_STRIDE), N_G5K)

        # The first layer is written as two matrices so that the (positions x
        # tracks x 398) input tensor is never materialised: it is a sum of a
        # per-position term and a per-track term.  Mathematically identical to
        # Linear(398, 2048) on the concatenation.
        self.w_genome = nn.Linear(N_GENOME_FACTORS, HIDDEN, bias=True)
        self.w_pair = nn.Linear(N_PAIR_FACTORS, HIDDEN, bias=False)
        self.fc2 = nn.Linear(HIDDEN, HIDDEN)
        self.fc3 = nn.Linear(HIDDEN, 1)

        for emb in (self.cell, self.assay, self.g25, self.g250, self.g5k):
            nn.init.uniform_(emb.weight, -0.05, 0.05)

    # -- parameter groups ---------------------------------------------------
    def genome_parameters(self):
        return list(self.g25.parameters()) + list(self.g250.parameters()) \
             + list(self.g5k.parameters())

    def shared_parameters(self):
        genome = {id(p) for p in self.genome_parameters()}
        return [p for p in self.parameters() if id(p) not in genome]

    def freeze_shared(self):
        """Cell-type factors, assay factors and the network stop learning.

        Used when transferring a model trained on one chromosome to another:
        only that chromosome's genomic factors are fitted.
        """
        for p in self.shared_parameters():
            p.requires_grad_(False)

    def replace_genome(self, n_bins, device):
        """Swap in a fresh, untrained set of genomic factors of a new length."""
        self.n_bins = n_bins
        self.g25 = nn.Embedding(n_bins, N_G25).to(device)
        self.g250 = nn.Embedding(math.ceil(n_bins / G250_STRIDE), N_G250).to(device)
        self.g5k = nn.Embedding(math.ceil(n_bins / G5K_STRIDE), N_G5K).to(device)
        for emb in (self.g25, self.g250, self.g5k):
            nn.init.uniform_(emb.weight, -0.05, 0.05)

    # -- forward ------------------------------------------------------------
    def genome_embedding(self, pos):
        return torch.cat([self.g25(pos),
                          self.g250(torch.div(pos, G250_STRIDE, rounding_mode="floor")),
                          self.g5k(torch.div(pos, G5K_STRIDE, rounding_mode="floor"))],
                         dim=-1)

    def pair_embedding(self, cell_idx, assay_idx):
        return torch.cat([self.cell(cell_idx), self.assay(assay_idx)], dim=-1)

    def forward(self, pos, cell_idx, assay_idx):
        """Predict arcsinh signal for every (position, track) in the outer product.

        pos        (B,)  int64 bin indices
        cell_idx   (T,)  int64
        assay_idx  (T,)  int64
        returns    (B, T)
        """
        g = self.w_genome(self.genome_embedding(pos))              # (B, H)
        p = self.w_pair(self.pair_embedding(cell_idx, assay_idx))  # (T, H)
        h = torch.relu(g.unsqueeze(1) + p.unsqueeze(0))            # (B, T, H)
        h = torch.relu(self.fc2(h))
        return self.fc3(h).squeeze(-1)


# ---------------------------------------------------------------------------
# the held-out entries used to monitor training
# ---------------------------------------------------------------------------
HOLDOUT_MOD = 50        # 2% of (position, track) entries


def holdout_mask(pos, n_tracks, device):
    """Deterministic 1-in-50 mask over the (position, track) grid.

    Held out by *entry*, not by position or by track: every genomic position and
    every experiment is still trained on, but 2% of the tensor's cells are never
    shown to the loss, which is the same shape of problem as imputation itself.
    Deterministic in the bin index, so it is identical across runs, datasets and
    restarts without storing a mask the size of the data.
    """
    j = torch.arange(n_tracks, device=device, dtype=torch.int64)
    h = (pos.unsqueeze(1) * 1000003 + j.unsqueeze(0) * 7919) % HOLDOUT_MOD
    return h == 0
