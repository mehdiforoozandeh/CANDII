"""Build NUMBERS.tsv: every number in report.pdf, where it appears, its source file and the source line.

Each entry names a needle string; the script finds the needle in the source file and records the
line number and the line. It fails if a needle is missing, so a rerun is the number check.
Run: /Users/mforooz/miniforge3/bin/python make_numbers.py   (from this folder)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
V = os.path.abspath(os.path.join(HERE, "..", ".."))  # cruxvault/
PAPER = "INSTRUCTIONS/peeling_the_onion/onion paper.md"
PLAN = "INSTRUCTIONS/peeling_the_onion/run_state/plan.md"
WD = "results/t170/withheld_design.md"
DS = "results/t183/DATASET_EIC_V1.md"
LR = "results/t305/LAYERS_REAL.md"
LE = "results/eta_space/LAYERS_ETA.md"
TH = "results/eta_space/THEORY.md"
BL = "results/t307/BASELINES.md"
AG = "results/t308/AG2_VS_BASELINES.md"
OF = "results/oracle_floor/ORACLE_FLOOR.md"
LAD = "results/metrics_screening/scores_ladder.md"
R4 = "results/metrics_screening/scores_rung4_arms.md"
R1 = "results/metrics_screening/scores_rung1_arms.md"
SEL = "results/metrics_screening/scores_selection_arms.md"
FC = "results/ladder_audit/FAIR_CHANCE.md"
COV = "results/ladder_audit/COVERAGE.md"
CEN = "results/t156/CENSUS.md"
T4 = "presentations/onion-update-max-2026-11-15/figs/data/rung4_table_nibi_2026-11-15.tsv"

E = [
    # status box
    ("16 of 256 regions", "status", PAPER, "A screening score runs pairs on 16 nested regions"),
    ("equal windows not equal compute (80,000 windows; FLOPs differ 58.7-fold)", "status; 6", FC, "7.81e15 / 1.33e14 = 58.7"),
    ("+0.101 (range 0.052)", "status; 6", R4, "| correction gap closed, oracle floor (all 102 pairs) | 0.2984 (0.0519) | 0.3997 (0.0220) | +0.1024 | +0.0790 | +0.1227 | +0.1014"),
    ("10 of 38 fully tested; 7 partial; 21 untested", "status; 6; Fig 6", COV, "Fully tested (3 scored seeds at the scope he asked for): 10"),
    ("38 knobs", "status; 6; Fig 6", COV, "Knobs the PI named (rows 1 to 39 without row 12): 38."),
    ("output-side depth arm 1 seed (0.462)", "status; 6; Fig 5", T4, "rung4_hide_depth_output\t2\tdone\t0.461613"),
    ("13:10 UTC 2026-11-15", "status", PLAN, "NIBI OUTAGE at 13:10 UTC"),
    # section 1
    ("8 designed views", "1; Fig 1", PAPER, "Three FASTQ-level views imitate other sequencing runs"),
    ("0.50, 0.25, 0.125 (Fig 1)", "Fig 1", PAPER, "three depth thinnings (`depth0.50`, `depth0.25`, `depth0.125`)"),
    ("778 half pairs", "1", LR, "Added: 778 pseudoreplicate pairs"),
    ("219 isogenic pairs; 2,911 cell-type pairs", "1", LR, "2,911 other_cell_type, 778 pseudoreplicate, 219 isogenic"),
    ("100% paired-end; 94% one lab", "1", PAPER, "The blind set is 100% paired-end, 100% HiSeq 4000 and 94% from one lab"),
    # section 2
    ("composite weights 0.5 / 0.25 / 0.25", "2", LAD, "Composite version 2 = 0.5 x specific variance + 0.25 x gap closed"),
    ("2 seed ranges over 3 seeds", "2", PAPER, "higher by more than 2 seed spreads over at least 3 seeds"),
    ("84 of 102 with a duplicate-keeping side", "2", OF, "The other 84 headline pairs (`mapq+nodedup`) have a side that keeps duplicates"),
    ("18 clean pairs", "2; Fig 5", OF, "18 pairs (knob `crop` 14, `crop+se` 4), 6 cells, 6 of the 11 assays"),
    # section 3
    ("Var(Y) = D E[r] + D^2 Var(r)", "3", TH, "Var(Y) = E[Var(Y | eta)] + Var(E[Y | eta]) = D E[r] + D^2 Var(r)."),
    ("76% to 97%", "3; Fig 2 caption", LR, "Over all regions it is 76% (DNase-seq) to 97% (ChIP-seq control)"),
    ("0.1% to 6%", "3", LR, "is small: 0.1% to 6% of the three layers"),
    ("cell type 10% to 23%", "3", LR, "The cell-type layer (mixed with donor and experiment) is 23% for DNase-seq, 20% for H3K27ac and H3K4me3, and 10% for the broad marks"),
    ("at most 10%; 8% to 63%", "3", LE, "changed by at most 10%"),
    ("8% to 63%", "3", LE, "fell by 8% to 63% from full to 1/4 depth"),
    ("0.24 vs 0.07 (H3K27ac)", "3", LE, "it is 0.12 for DNase-seq and 0.24 for H3K27ac in eta space, against 0.04 and 0.07 on log1p"),
    ("0.38 [0.23, 0.54]; -0.002", "3", LE, "0.38 [0.23, 0.54] of its variance, where log1p gave -0.002"),
    ("8 to 12 isogenic pairs", "3", LE, "These assays have 8 to 12 isogenic pairs"),
    ("256 regions of 409.6 kb", "3", PAPER, "so a region is 409,600 bp"),
    ("63 imputation items", "3", PAPER, "The set has 63 imputation items"),
    ("1,480 correction pairs", "3; 5", PAPER, "The set has 1,480 correction pairs over 11 correction-scorable assays"),
    ("102 headline pairs", "3; 5", OF, "All headline pairs: never trained, not a subset pair, 102 pairs"),
    ("0.258 [0.192, 0.373]", "3; Fig 3b", OF, "| position average | full | 0.5234 | 0.0490 | 2.864 | 0.258 [0.192, 0.373]"),
    ("0.167 [0.094, 0.219]", "Fig 3b", OF, "| half-read floor | full | 0.5326 | 0.0397 | 1 | 0.167 [0.094, 0.219]"),
    ("-0.286 [-0.986, -0.123]", "Fig 3b", OF, "| marginal NB | full | 0.6228 | -0.0504 | -2.401 | -0.286 [-0.986, -0.123]"),
    ("0 (depth-only scaling)", "Fig 3b", OF, "| depth-only scaling | full | 0.5724 | 0 | 0 | 0 [0, 0]"),
    ("11 tracks in 9 cells", "3; Fig 3a", AG, "Board, fair set (`unseen_by_both`, 11 tracks, 9 cells"),
    ("0.438", "3; Fig 3a", AG, "| ag2_matched | 0.438 | 0.549 | 0.417 |"),
    ("0.222", "3; Fig 3a", AG, "| position_average | 0.222 | 0.415 | 0.386 |"),
    ("0.121", "Fig 3a", AG, "| marginal_nb | 0.121 | 0.084 | 0.123 |"),
    ("0.066", "Fig 3a", AG, "| closest5_oracle | 0.066 | 0.371 | 0.256 |"),
    ("0 (training-track mean)", "Fig 3a", AG, "| training_mean_signal | 0 (by definition) | 0.366 | 0.251 |"),
    ("0.333 [0.276, 0.382]", "3; Fig 3a", AG, "| pseudoreplicate_half (noise floor) | 0.333 [0.276, 0.382] |"),
    # section 4
    ("check PASS", "4", DS, "`check: PASS`, exit 0. Every rule line is 0"),
    ("2,740 measurements", "4", DS, "2,740 measurement directories read"),
    ("249 of 267; 18 DNase-seq", "4", PAPER, "| training | 249 | 267 | 18, all DNase-seq |"),
    ("42 of 45", "4", PAPER, "| validation | 42 | 45 |"),
    ("37 of 40", "4", PAPER, "| control | 37 | 40 |"),
    ("DNase-seq not imputation-scorable", "4", PAPER, "DNase-seq is not imputation-scorable, because only 1 validation track has halves and the rule needs 3"),
    # section 5 counterfactual views
    ("0.182 to 0.234 (target deeper)", "5", BL, "the depth-increasing pairs of the three compound depth knobs have the three largest gaps (0.182 to 0.234)"),
    ("0.035 [0.021, 0.055] (pure depth thinning)", "5", BL, "The pure subset (depth-thinning) pairs are not near 0: 0.035 [0.021, 0.055]"),
    ("position average 0.235 (screening)", "5", LAD, "| baseline position_average | 1 | 0.1190 | 0.5563 | 0.2349"),
    ("0.291 (Perceiver IO)", "5", LAD, "| rung6_base | 3 | 0.1541 (0.0641) | 0.5421 (0.0107) | 0.2907 (0.0439)"),
    ("0.576 (conv autoencoder)", "5", LAD, "| rung2_base | 3 | 0.3850 (0.0297) | 0.4754 (0.0027) | 0.5761 (0.0125)"),
    ("crop 0.582 (range 0.034)", "5", "results/oracle_floor/per_knob_p16.tsv", "rung1_base\tarm\tnever_trained_non_subset_oracle\tcrop\t3\t3\t0.582153\t0.0340144"),
    ("mapq+nodedup 0.420 (range 0.017)", "5", "results/oracle_floor/per_knob_p16.tsv", "rung1_base\tarm\tnever_trained_non_subset_oracle\tmapq+nodedup\t3\t3\t0.420349\t0.0169706"),
    ("no per-knob rows for U-Net or conv autoencoder", "5", "results/oracle_floor/per_knob_p16.tsv", "rung1_base\tarm"),
    # section 6
    ("about 1.0 M parameters; 80,000 windows", "6", FC, "All six counts in `run.json` fall in 993,599 to 1,042,514"),
    ("0.502 (0.031)", "6; Fig 4", LAD, "| rung2_base | 3 | 0.3850 (0.0297) | 0.4754 (0.0027) | 0.5761 (0.0125) | 0.7187 (0.0123) | 6.5296 (0.2156) | 0.5021 (0.0309)"),
    ("0.493 (0.033)", "6; Fig 4", LAD, "| rung3_base | 3 | 0.3885 (0.0128) | 0.4772 (0.0026) | 0.5718 (0.0105) | 0.6789 (0.0286) | 6.4174 (0.1930) | 0.4927 (0.0326)"),
    ("0.444 to 0.469 (axial 0.469)", "6; Fig 4", LAD, "| rung5_base | 3 | 0.3901 (0.0094) | 0.5010 (0.0082) | 0.4474 (0.0490) | 0.5793 (0.0146) | 4.6741 (0.5848) | 0.4690 (0.0297)"),
    ("0.444 (pointwise MLP)", "6; Fig 4", LAD, "| rung1_base | 3 | 0.3431 (0.0029) | 0.4971 (0.0034) | 0.4743 (0.0230) | 0.5930 (0.0309) | 5.3386 (0.4062) | 0.4439 (0.0086)"),
    ("0.261 (0.052)", "6; Fig 4", LAD, "| rung6_base | 3 | 0.1541 (0.0641) | 0.5421 (0.0107) | 0.2907 (0.0439) | 0.4445 (0.0427) | 4.0279 (0.4294) | 0.2605 (0.0516)"),
    ("U-Net 0.444 (0.030); 0.440 (0.013); 0.298 (0.052); clean 0.519 (0.053)", "6; Fig 4; Fig 5", LAD, "| rung4_base | 3 | 0.4397 (0.0134) | 0.5309 (0.0114) | 0.2984 (0.0519) | 0.5185 (0.0530) | 2.9784 (0.5925) | 0.4444 (0.0297)"),
    ("0.029 (0.013)", "6; Fig 5", R4, "| rung4_scrambled_covariates | 3 | -0.0291 [0.0134] yes"),
    ("input side: 0.382 (0.009); 0.401 (0.020); 0.215 (0.087); 0.601 (0.036); 0.511 (0.164)", "6; Fig 5 caption", T4, "rung4_hide_depth_input\t0\tdone\t0.387045"),
    ("both sides + transform: 0.260 (0.019); -0.099 (0.211)", "6; Fig 5", T4, "rung4_hard_offset\t0\tdone\t0.250926"),
    ("uncertainty base 0.600 (0.100)", "6; Fig 5", R4, "Per seed, base 0.5409, 0.6411, 0.6179 (mean 0.5999, spread 0.1002)"),
    ("both-sides transform hides depth in the input transform", "6; Fig 5", "presentations/onion-update-max-2026-11-15/make_numbers.py", "tools/minicandi/covariates.py lines 230-271: hide_depth_covariate True also hides t_in"),
    ("+0.009 (range 0.029)", "6", SEL, "| rg_terr_r1024_w20k | 3 | +0.0071 [0.0194] no | -0.0062 [0.0081] no | +0.0252 [0.0318] no | +0.0286 [0.0484] no | +0.2360 [0.3721] no | +0.0092 [0.0289] no"),
    ("42 trial runs over five architectures; held-out loss only", "6", "results/ladder_tuning/LEAD_PLAN.md", "**CPU, rungs 2 to 6 (42 runs, 436.0 CPU-h):**"),
    ("selection on held-out loss only", "6", "results/ladder_tuning/LEAD_PLAN.md", "1. **Selection on held-out loss only.**"),
    ("36 runs of a first round = 21 (7 configs x 3 seeds, jobs 27446721-27446846) + 5 held-out (27446847-51) + 10 held-out seeds 1,2 (27447277-86)", "6", PLAN, "FAIR-CHANCE ROUND 1 SUBMITTED from kit FC"),
    ("36 (part 2: held-out curves seed 0)", "6", PLAN, "held-out curve runs rung{2..6}_base_heldout seed 0: jobs 27446847 to 27446851"),
    ("36 (part 3: held-out seeds 1, 2)", "6", "results/ladder_tuning/LEAD_PLAN.md", "jobs 27447277 to 27447286"),
    ("pointwise MLP trials wait for GPU", "6", "results/ladder_tuning/LEAD_PLAN.md", "Rung 1 runs only on a GPU, and the Nibi GPU queue is blocked."),
    # section 7
    ("160,000 windows", "7", PLAN, "rung2_windows160k 27446793/99/805"),
]


def main():
    out = ["number\twhere_in_report\tsource_path\tsource_line\tsource_excerpt"]
    bad = 0
    for num, where, path, needle in E:
        if path.endswith("make_numbers.py"):
            out.append(f"{num}\t{where}\tNibi ls/find, 2026-11-15 (this session)\t-\t{needle}")
            continue
        full = os.path.join(V, path)
        hit = None
        with open(full) as fh:
            for i, line in enumerate(fh, 1):
                if needle in line:
                    hit = (i, line.strip())
                    break
        if hit is None:
            print("MISSING:", num, path, needle[:70]); bad += 1
            continue
        ex = hit[1].replace("\t", " | ")
        if len(ex) > 220:
            k = ex.find(needle.replace("\t", " | ")[:40])
            ex = ex[max(0, k - 20):k + 200]
        out.append(f"{num}\t{where}\tcruxvault/{path}\t{hit[0]}\t{ex}")
    with open(os.path.join(HERE, "NUMBERS.tsv"), "w") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"{len(out) - 1} rows, {bad} missing")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
