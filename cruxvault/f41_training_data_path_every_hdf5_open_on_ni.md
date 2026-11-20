---
id: f41
type: finding
schema: 3
title: "Training data path: every HDF5 open on Nibi's NFS /scratch takes about 0.21…"
parent: q27
confidence: moderate
evidence: results/t213/SUBMITTED.md
created: "2026-11-14T02:18:32"
updated: "2026-11-14T02:18:32"
---

# f41 — Training data path: every HDF5 open on Nibi's NFS /scratch takes about 0.21…

Parent:: [[q27_are_the_onion_run_s_estimators_scorers_a]]

## Answer

Training data path: every HDF5 open on Nibi's NFS /scratch takes about 0.21 s, which made rung-1 training about 55 times slower on Nibi than on Fir (2.58 against 0.0466 s per window); staging the 2,328 training count files (56.1 GB) to the node's local disk in 138 to 170 s fixes it, with bit-identical batches and losses (1.5533499717712402, 1.0269404649734497, 1.131638526916504 at steps 1, 50, 100); measured with staging: rung 1 0.0423 s per window on a GPU slice, rung 2 0.0669 on CPU (2.7 times Fir, cause not found), rung 5 0.1011. The scrambled-covariates arm is wired: it permuted 19 of 23 covariate rows in one batch; it equals the base arm only at initialization (zero NB head, identity FiLM), and the losses split from step 2.
