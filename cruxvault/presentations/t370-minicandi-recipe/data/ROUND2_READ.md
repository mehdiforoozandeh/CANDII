# Tuning round 2 and fair-chance round 1, read 2026-11-16 (Fir)

Source: Fir `/project/def-maxwl/mforooz/ONION/runs_fir/<run>/seed<k>/log.jsonl` (key `heldout_loss`, mean of the last 2 values) and `.../p16/scores.json`. Base scores for comparison: Nibi `/project/def-maxwl/mforooz/ONION/runs/rung<r>_base`. Pointwise MLP not run (GPU).

## Held-out loss (NB loss on held-out windows; lower is better)

| architecture | base, seeds 0/1/2 | margin | best trial (seed 0) | change vs base seed 0 | verdict |
|---|---|---|---|---|---|
| conv autoencoder | 0.9695 / 0.9667 / 0.9693 | 0.005 | 3 levels, 0.9544 | -0.0151 | winner: 3 levels (also past margin: lr 3e-3 0.9634, 1 block per level 0.9637) |
| conv + attention | 0.9578 / 0.9597 / 0.9640 | 0.005 | attention at 128 positions, 0.9491 | -0.0087 | winner: attention at 128 positions |
| U-Net | 0.9179 / 0.9164 / 0.9145 | 0.005 | lr 3e-3, 0.9109 | -0.0070 | winner: lr 3e-3 |
| axial transformer | 0.9312 / 0.9329 / 0.9301 | 0.005 | warmup 15%, 0.9285 | -0.0027 | no winner; base stays |
| Perceiver IO | 1.0604 / 1.0611 / 1.0719 | 0.0058 | unpatch MLP 128, 1.0578 | -0.0026 | no winner; base stays |

Other held-out reads (3 seeds, mean): 160,000 windows: conv autoencoder 0.9423 (base 0.9685), conv + attention 0.9325 (0.9605), U-Net 0.9022 (0.9163). Axial at matched compute (106,672 windows) 0.9206 (0.9314); axial lr 1e-3 0.9358. Perceiver lr 1e-3 1.0943, more latents 1.0636, decoder norm 1.1095 (seed 0), lr 3e-3 1.0817 (seed 0).

## Composite on the 16-region set (3 seeds, mean (range)); base from Nibi

| run | composite | specific variance | gap closed | base composite | base specific variance | base gap closed |
|---|---|---|---|---|---|---|
| conv autoencoder, 160k windows | 0.470 (0.041) | 0.358 | 0.551 | 0.502 (0.031) | 0.385 | 0.576 |
| conv + attention, 160k windows | 0.456 (0.047) | 0.353 | 0.518 | 0.493 (0.033) | 0.389 | 0.572 |
| U-Net, 160k windows | 0.433 (0.033) | 0.451 | 0.203 | 0.444 (0.030) | 0.440 | 0.298 |
| axial, matched compute | 0.447 (0.021) | 0.397 | 0.345 | 0.469 (0.030) | 0.390 | 0.447 |
| Perceiver, more latents | 0.285 (0.092) | 0.189 | 0.333 | 0.260 (0.052) | 0.154 | 0.291 |
| Perceiver, lr 1e-3 | 0.248 (0.068) | 0.054 | 0.231 | 0.260 (0.052) | 0.154 | 0.291 |

Axial lr 1e-3: seed 0 scored 0.487; the metrics jobs of seeds 1 and 2 failed on a Lustre error on Fir scratch (jobs 4246925, 4246928).

## Readings

1. Held-out loss ranks the architectures U-Net < axial < conv + attention < conv autoencoder < Perceiver IO. The composite ranks the two conv models first. The two measures disagree.
2. Twice the windows lowered held-out loss in all three architectures tested, so no architecture had converged at 80,000 windows. The composite did not rise with it.
3. Every Fir run has a lower gap closed than its Nibi base. A cluster or code-snapshot offset is not excluded. The 15 held-out reference runs are now being scored on Fir (jobs 4298888 to 4298962) to give a same-cluster base.
4. Perceiver IO stays last under every trial; no recipe or shape trial passed the margin.
