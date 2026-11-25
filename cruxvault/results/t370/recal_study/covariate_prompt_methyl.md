# Covariate-prompt test

8 imputed targets in held-out cells, 32 regions, 3 seeds. One output covariate changed per arm; the rest kept.

Expected directions, stated before the run:

- depth_up: total_ratio in [1.6, 2.5] (2x depth -> ~2x counts); shape_r > 0.98
- depth_down: total_ratio in [0.4, 0.625] (0.5x depth); shape_r > 0.98
- run_type_flip: SE->PE at the same read depth: total_ratio < 1 (half the fragments); PE->SE: total_ratio > 1
- read_length: 36->100 bp: total_ratio >= 1 (more reads map); 100->36: total_ratio <= 1
- dedup_flip: dedup on->off: total_ratio > 1 (duplicates kept); off->on: < 1
- mapq_flip: MAPQ 30->0: total_ratio > 1 (multi-mapped reads kept); 0->30: < 1
- lab_other: another seen lab: |total_ratio - 1| < 0.2 (no direction)
- lab_unknown: unknown lab: |total_ratio - 1| < 0.2; n_ratio < 1 (less confident)
- platform_other: another seen platform: |total_ratio - 1| < 0.2 (no direction)
- platform_unknown: unknown platform: |total_ratio - 1| < 0.2; n_ratio < 1
- antibody_other: another seen antibody (ChIP only): |total_ratio - 1| < 0.2 (no direction)
- antibody_unknown: unknown antibody (ChIP only): |total_ratio - 1| < 0.2; n_ratio < 1

Per target and arm: total ratio, shape r, n ratio (median over bins), mean over seeds [min, max]; met = seeds meeting the stated rule.

| assay | target | arm | total ratio | shape r | n ratio | met |
|---|---|---|---|---|---|---|
| H3K27me3 | ENCSR000DQU.pooled.uniform.base | depth_up | 2.001 [1.977, 2.017] | 0.995 [0.994, 0.995] | 1.049 [1.013, 1.084] | 3/3 |
| H3K27me3 | ENCSR000DQU.pooled.uniform.base | depth_down | 0.512 [0.508, 0.519] | 0.995 [0.995, 0.996] | 0.939 [0.843, 1.040] | 3/3 |
| H3K27me3 | ENCSR000DQU.pooled.uniform.base | run_type_flip | 0.959 [0.938, 0.991] | 0.991 [0.985, 0.994] | 0.991 [0.834, 1.197] | 3/3 |
| H3K27me3 | ENCSR000DQU.pooled.uniform.base | read_length | 1.257 [1.044, 1.524] | 0.946 [0.890, 0.987] | 1.173 [1.009, 1.334] | 3/3 |
| H3K27me3 | ENCSR000DQU.pooled.uniform.base | dedup_flip | 0.996 [0.994, 0.997] | 0.999 [0.998, 0.999] | 0.603 [0.400, 0.720] | 0/3 |
| H3K27me3 | ENCSR000DQU.pooled.uniform.base | mapq_flip | 1.008 [0.949, 1.060] | 0.774 [0.741, 0.829] | 1.145 [0.968, 1.295] | 2/3 |
| H3K27me3 | ENCSR000DQU.pooled.uniform.base | lab_other | 0.943 [0.834, 1.016] | 0.980 [0.974, 0.987] | 1.374 [1.321, 1.417] | 3/3 |
| H3K27me3 | ENCSR000DQU.pooled.uniform.base | lab_unknown | 1.057 [1.012, 1.102] | 0.981 [0.976, 0.983] | 0.656 [0.410, 0.825] | 3/3 |
| H3K27me3 | ENCSR000DQU.pooled.uniform.base | platform_other | 0.950 [0.919, 0.969] | 0.960 [0.913, 0.990] | 0.967 [0.560, 1.678] | 3/3 |
| H3K27me3 | ENCSR000DQU.pooled.uniform.base | platform_unknown | 1.025 [0.938, 1.115] | 0.973 [0.961, 0.996] | 0.509 [0.261, 0.677] | 3/3 |
| H3K27me3 | ENCSR000DQU.pooled.uniform.base | antibody_other | 1.062 [1.031, 1.085] | 0.933 [0.907, 0.986] | 0.611 [0.548, 0.645] | 3/3 |
| H3K27me3 | ENCSR000DQU.pooled.uniform.base | antibody_unknown | 0.930 [0.905, 0.961] | 0.965 [0.951, 0.983] | 1.314 [1.154, 1.602] | 0/3 |
| H3K36me3 | ENCSR038OIN.pooled.uniform.base | depth_up | 1.989 [1.980, 2.002] | 0.994 [0.993, 0.995] | 1.170 [1.027, 1.307] | 3/3 |
| H3K36me3 | ENCSR038OIN.pooled.uniform.base | depth_down | 0.507 [0.503, 0.512] | 0.994 [0.993, 0.996] | 0.845 [0.722, 0.981] | 3/3 |
| H3K36me3 | ENCSR038OIN.pooled.uniform.base | run_type_flip | 1.003 [0.992, 1.020] | 0.997 [0.996, 0.999] | 0.962 [0.645, 1.334] | 2/3 |
| H3K36me3 | ENCSR038OIN.pooled.uniform.base | read_length | 1.255 [1.119, 1.460] | 0.971 [0.958, 0.988] | 1.103 [0.984, 1.184] | 3/3 |
| H3K36me3 | ENCSR038OIN.pooled.uniform.base | dedup_flip | 0.992 [0.977, 1.009] | 0.998 [0.997, 0.998] | 0.241 [0.219, 0.276] | 1/3 |
| H3K36me3 | ENCSR038OIN.pooled.uniform.base | mapq_flip | 0.928 [0.899, 0.964] | 0.891 [0.854, 0.916] | 1.237 [1.109, 1.395] | 0/3 |
| H3K36me3 | ENCSR038OIN.pooled.uniform.base | lab_other | 1.122 [1.056, 1.250] | 0.929 [0.901, 0.976] | 1.189 [0.672, 2.082] | 2/3 |
| H3K36me3 | ENCSR038OIN.pooled.uniform.base | lab_unknown | 1.044 [1.011, 1.087] | 0.971 [0.917, 0.997] | 1.194 [0.660, 1.593] | 1/3 |
| H3K36me3 | ENCSR038OIN.pooled.uniform.base | platform_other | 0.964 [0.881, 1.007] | 0.947 [0.885, 0.979] | 2.414 [2.057, 2.870] | 3/3 |
| H3K36me3 | ENCSR038OIN.pooled.uniform.base | platform_unknown | 1.010 [0.919, 1.096] | 0.955 [0.891, 0.994] | 1.295 [0.707, 2.442] | 2/3 |
| H3K36me3 | ENCSR038OIN.pooled.uniform.base | antibody_other | 0.960 [0.924, 0.995] | 0.956 [0.941, 0.966] | 1.811 [1.407, 2.435] | 3/3 |
| H3K36me3 | ENCSR038OIN.pooled.uniform.base | antibody_unknown | 0.943 [0.895, 0.977] | 0.911 [0.899, 0.931] | 2.261 [1.451, 2.678] | 0/3 |
| H3K4me1 | ENCSR254NLC.pooled.uniform.base | depth_up | 1.999 [1.980, 2.015] | 0.995 [0.994, 0.997] | 1.054 [1.016, 1.096] | 3/3 |
| H3K4me1 | ENCSR254NLC.pooled.uniform.base | depth_down | 0.500 [0.499, 0.503] | 0.995 [0.994, 0.996] | 0.920 [0.908, 0.935] | 3/3 |
| H3K4me1 | ENCSR254NLC.pooled.uniform.base | run_type_flip | 0.993 [0.989, 0.997] | 0.998 [0.996, 1.000] | 0.968 [0.936, 1.019] | 3/3 |
| H3K4me1 | ENCSR254NLC.pooled.uniform.base | read_length | 1.115 [0.966, 1.244] | 0.983 [0.977, 0.989] | 1.106 [0.976, 1.252] | 2/3 |
| H3K4me1 | ENCSR254NLC.pooled.uniform.base | dedup_flip | 0.987 [0.975, 0.995] | 0.998 [0.997, 0.999] | 0.655 [0.522, 0.782] | 0/3 |
| H3K4me1 | ENCSR254NLC.pooled.uniform.base | mapq_flip | 1.026 [1.016, 1.032] | 0.869 [0.815, 0.916] | 0.998 [0.954, 1.048] | 3/3 |
| H3K4me1 | ENCSR254NLC.pooled.uniform.base | lab_other | 0.999 [0.894, 1.081] | 0.936 [0.907, 0.972] | 0.832 [0.640, 0.970] | 3/3 |
| H3K4me1 | ENCSR254NLC.pooled.uniform.base | lab_unknown | 1.033 [1.008, 1.046] | 0.981 [0.948, 1.000] | 0.695 [0.297, 1.058] | 2/3 |
| H3K4me1 | ENCSR254NLC.pooled.uniform.base | platform_other | 0.966 [0.945, 0.984] | 0.983 [0.971, 0.990] | 1.522 [1.492, 1.577] | 3/3 |
| H3K4me1 | ENCSR254NLC.pooled.uniform.base | platform_unknown | 1.010 [1.002, 1.023] | 0.981 [0.962, 0.991] | 0.688 [0.562, 0.826] | 3/3 |
| H3K4me1 | ENCSR254NLC.pooled.uniform.base | antibody_other | 0.999 [0.920, 1.063] | 0.961 [0.955, 0.966] | 1.271 [0.507, 1.776] | 3/3 |
| H3K4me1 | ENCSR254NLC.pooled.uniform.base | antibody_unknown | 0.927 [0.840, 1.006] | 0.912 [0.904, 0.926] | 1.764 [1.091, 2.129] | 0/3 |
| H3K4me2 | ENCSR730POX.pooled.uniform.base | depth_up | 2.007 [1.989, 2.039] | 0.993 [0.992, 0.995] | 1.191 [1.076, 1.274] | 3/3 |
| H3K4me2 | ENCSR730POX.pooled.uniform.base | depth_down | 0.497 [0.489, 0.503] | 0.991 [0.990, 0.992] | 0.797 [0.755, 0.850] | 3/3 |
| H3K4me2 | ENCSR730POX.pooled.uniform.base | run_type_flip | 1.028 [0.985, 1.072] | 0.997 [0.996, 0.998] | 0.721 [0.483, 0.961] | 1/3 |
| H3K4me2 | ENCSR730POX.pooled.uniform.base | read_length | 1.369 [1.284, 1.472] | 0.969 [0.955, 0.983] | 0.940 [0.686, 1.285] | 3/3 |
| H3K4me2 | ENCSR730POX.pooled.uniform.base | dedup_flip | 1.063 [0.978, 1.155] | 0.992 [0.988, 0.997] | 0.070 [0.040, 0.114] | 2/3 |
| H3K4me2 | ENCSR730POX.pooled.uniform.base | mapq_flip | 1.003 [0.965, 1.045] | 0.802 [0.753, 0.857] | 1.510 [1.042, 1.915] | 1/3 |
| H3K4me2 | ENCSR730POX.pooled.uniform.base | lab_other | 1.029 [1.019, 1.042] | 0.974 [0.968, 0.978] | 0.921 [0.747, 1.043] | 3/3 |
| H3K4me2 | ENCSR730POX.pooled.uniform.base | lab_unknown | 1.089 [1.002, 1.165] | 0.954 [0.870, 0.998] | 0.844 [0.728, 1.051] | 2/3 |
| H3K4me2 | ENCSR730POX.pooled.uniform.base | platform_other | 1.008 [0.969, 1.076] | 0.990 [0.986, 0.997] | 0.693 [0.443, 1.030] | 3/3 |
| H3K4me2 | ENCSR730POX.pooled.uniform.base | platform_unknown | 1.034 [0.990, 1.085] | 0.980 [0.963, 0.997] | 0.294 [0.122, 0.482] | 3/3 |
| H3K4me2 | ENCSR730POX.pooled.uniform.base | antibody_other | 1.038 [0.994, 1.060] | 0.989 [0.974, 1.000] | 0.877 [0.648, 1.110] | 3/3 |
| H3K4me2 | ENCSR730POX.pooled.uniform.base | antibody_unknown | 0.986 [0.970, 1.007] | 0.997 [0.996, 0.998] | 1.165 [0.991, 1.431] | 1/3 |
| H3K4me3 | ENCSR234YIU.pooled.uniform.base | depth_up | 2.029 [1.999, 2.070] | 0.986 [0.983, 0.990] | 1.170 [1.051, 1.332] | 3/3 |
| H3K4me3 | ENCSR234YIU.pooled.uniform.base | depth_down | 0.496 [0.478, 0.510] | 0.987 [0.982, 0.991] | 0.827 [0.749, 0.951] | 3/3 |
| H3K4me3 | ENCSR234YIU.pooled.uniform.base | run_type_flip | 1.001 [0.973, 1.040] | 0.992 [0.985, 0.998] | 0.717 [0.396, 0.943] | 2/3 |
| H3K4me3 | ENCSR234YIU.pooled.uniform.base | read_length | 1.569 [1.461, 1.677] | 0.937 [0.916, 0.952] | 2.507 [0.805, 3.759] | 3/3 |
| H3K4me3 | ENCSR234YIU.pooled.uniform.base | dedup_flip | 0.981 [0.971, 0.991] | 0.995 [0.992, 0.997] | 0.407 [0.276, 0.486] | 0/3 |
| H3K4me3 | ENCSR234YIU.pooled.uniform.base | mapq_flip | 0.979 [0.912, 1.061] | 0.882 [0.859, 0.903] | 1.042 [0.782, 1.214] | 1/3 |
| H3K4me3 | ENCSR234YIU.pooled.uniform.base | lab_other | 1.236 [1.056, 1.379] | 0.917 [0.853, 0.972] | 2.499 [1.511, 3.061] | 1/3 |
| H3K4me3 | ENCSR234YIU.pooled.uniform.base | lab_unknown | 1.078 [1.002, 1.209] | 0.996 [0.993, 0.999] | 1.062 [0.676, 1.569] | 2/3 |
| H3K4me3 | ENCSR234YIU.pooled.uniform.base | platform_other | 0.881 [0.854, 0.930] | 0.936 [0.868, 0.978] | 0.526 [0.407, 0.596] | 3/3 |
| H3K4me3 | ENCSR234YIU.pooled.uniform.base | platform_unknown | 0.857 [0.795, 0.919] | 0.957 [0.944, 0.971] | 0.179 [0.108, 0.306] | 2/3 |
| H3K4me3 | ENCSR234YIU.pooled.uniform.base | antibody_other | 0.948 [0.835, 1.059] | 0.965 [0.940, 0.991] | 1.264 [0.610, 2.132] | 3/3 |
| H3K4me3 | ENCSR234YIU.pooled.uniform.base | antibody_unknown | 0.958 [0.928, 0.990] | 0.974 [0.962, 0.987] | 1.339 [1.150, 1.628] | 0/3 |
| H3K79me2 | ENCSR000APD.pooled.uniform.base | depth_up | 2.013 [1.983, 2.035] | 0.995 [0.994, 0.996] | 1.006 [0.921, 1.064] | 3/3 |
| H3K79me2 | ENCSR000APD.pooled.uniform.base | depth_down | 0.496 [0.487, 0.505] | 0.994 [0.993, 0.995] | 1.030 [0.886, 1.225] | 3/3 |
| H3K79me2 | ENCSR000APD.pooled.uniform.base | run_type_flip | 1.000 [0.979, 1.036] | 0.993 [0.987, 0.998] | 0.903 [0.755, 1.037] | 2/3 |
| H3K79me2 | ENCSR000APD.pooled.uniform.base | read_length | 1.379 [1.226, 1.541] | 0.978 [0.967, 0.994] | 1.538 [0.912, 2.187] | 3/3 |
| H3K79me2 | ENCSR000APD.pooled.uniform.base | dedup_flip | 1.025 [0.921, 1.165] | 0.998 [0.998, 0.999] | 0.350 [0.197, 0.519] | 1/3 |
| H3K79me2 | ENCSR000APD.pooled.uniform.base | mapq_flip | 0.969 [0.925, 1.052] | 0.902 [0.852, 0.935] | 1.518 [1.024, 1.913] | 1/3 |
| H3K79me2 | ENCSR000APD.pooled.uniform.base | lab_other | 1.007 [0.916, 1.090] | 0.977 [0.950, 0.995] | 0.767 [0.339, 1.019] | 3/3 |
| H3K79me2 | ENCSR000APD.pooled.uniform.base | lab_unknown | 1.086 [0.963, 1.260] | 0.985 [0.964, 0.996] | 0.985 [0.736, 1.244] | 2/3 |
| H3K79me2 | ENCSR000APD.pooled.uniform.base | platform_other | 0.980 [0.808, 1.166] | 0.961 [0.931, 0.977] | 3.907 [2.309, 4.894] | 3/3 |
| H3K79me2 | ENCSR000APD.pooled.uniform.base | antibody_other | 1.004 [0.946, 1.109] | 0.971 [0.935, 0.993] | 1.490 [0.627, 3.007] | 3/3 |
| H3K79me2 | ENCSR000APD.pooled.uniform.base | antibody_unknown | 0.978 [0.889, 1.096] | 0.990 [0.987, 0.991] | 1.496 [1.367, 1.640] | 0/3 |
| H3K9me3 | ENCSR133QBG.pooled.uniform.base | depth_up | 1.996 [1.986, 2.002] | 0.995 [0.995, 0.996] | 0.958 [0.905, 1.023] | 3/3 |
| H3K9me3 | ENCSR133QBG.pooled.uniform.base | depth_down | 0.503 [0.502, 0.506] | 0.995 [0.995, 0.996] | 1.028 [0.960, 1.081] | 3/3 |
| H3K9me3 | ENCSR133QBG.pooled.uniform.base | run_type_flip | 1.022 [0.993, 1.036] | 0.996 [0.995, 0.997] | 0.836 [0.702, 0.919] | 1/3 |
| H3K9me3 | ENCSR133QBG.pooled.uniform.base | read_length | 0.672 [0.638, 0.699] | 0.946 [0.946, 0.947] | 0.619 [0.487, 0.694] | 3/3 |
| H3K9me3 | ENCSR133QBG.pooled.uniform.base | dedup_flip | 1.005 [0.999, 1.009] | 0.998 [0.998, 0.999] | 0.799 [0.684, 0.883] | 2/3 |
| H3K9me3 | ENCSR133QBG.pooled.uniform.base | mapq_flip | 0.961 [0.945, 0.985] | 0.836 [0.811, 0.881] | 1.202 [0.949, 1.661] | 0/3 |
| H3K9me3 | ENCSR133QBG.pooled.uniform.base | lab_other | 0.841 [0.649, 1.058] | 0.891 [0.747, 0.971] | 0.713 [0.308, 1.034] | 2/3 |
| H3K9me3 | ENCSR133QBG.pooled.uniform.base | lab_unknown | 0.904 [0.750, 1.033] | 0.871 [0.668, 0.983] | 0.700 [0.212, 1.059] | 1/3 |
| H3K9me3 | ENCSR133QBG.pooled.uniform.base | platform_other | 0.896 [0.857, 0.952] | 0.918 [0.795, 0.982] | 0.377 [0.207, 0.482] | 3/3 |
| H3K9me3 | ENCSR133QBG.pooled.uniform.base | platform_unknown | 0.730 [0.646, 0.830] | 0.715 [0.685, 0.760] | 0.089 [0.068, 0.120] | 1/3 |
| H3K9me3 | ENCSR133QBG.pooled.uniform.base | antibody_other | 1.023 [1.014, 1.040] | 0.983 [0.969, 0.996] | 1.130 [0.942, 1.272] | 3/3 |
| H3K9me3 | ENCSR133QBG.pooled.uniform.base | antibody_unknown | 1.031 [1.018, 1.042] | 0.928 [0.894, 0.980] | 0.944 [0.761, 1.241] | 2/3 |
| H4K20me1 | ENCSR000ARC.pooled.uniform.base | depth_up | 2.003 [1.983, 2.031] | 0.993 [0.992, 0.994] | 1.062 [1.032, 1.078] | 3/3 |
| H4K20me1 | ENCSR000ARC.pooled.uniform.base | depth_down | 0.498 [0.488, 0.508] | 0.992 [0.992, 0.992] | 0.951 [0.911, 0.977] | 3/3 |
| H4K20me1 | ENCSR000ARC.pooled.uniform.base | run_type_flip | 1.010 [0.972, 1.050] | 0.991 [0.984, 0.998] | 1.490 [0.840, 2.449] | 1/3 |
| H4K20me1 | ENCSR000ARC.pooled.uniform.base | read_length | 1.160 [1.074, 1.253] | 0.963 [0.961, 0.966] | 1.619 [1.315, 1.798] | 3/3 |
| H4K20me1 | ENCSR000ARC.pooled.uniform.base | dedup_flip | 1.012 [0.931, 1.080] | 0.994 [0.992, 0.997] | 0.306 [0.195, 0.412] | 2/3 |
| H4K20me1 | ENCSR000ARC.pooled.uniform.base | mapq_flip | 0.974 [0.945, 1.031] | 0.809 [0.744, 0.868] | 1.873 [1.377, 2.414] | 1/3 |
| H4K20me1 | ENCSR000ARC.pooled.uniform.base | lab_other | 0.975 [0.960, 0.986] | 0.959 [0.948, 0.972] | 0.955 [0.582, 1.316] | 3/3 |
| H4K20me1 | ENCSR000ARC.pooled.uniform.base | lab_unknown | 0.964 [0.916, 0.990] | 0.972 [0.955, 0.984] | 1.357 [0.900, 1.682] | 1/3 |
| H4K20me1 | ENCSR000ARC.pooled.uniform.base | platform_other | 0.900 [0.834, 0.999] | 0.951 [0.918, 0.978] | 3.516 [1.623, 4.510] | 3/3 |
| H4K20me1 | ENCSR000ARC.pooled.uniform.base | platform_unknown | 0.966 [0.934, 0.988] | 0.974 [0.956, 0.984] | 0.583 [0.409, 0.884] | 3/3 |
| H4K20me1 | ENCSR000ARC.pooled.uniform.base | antibody_other | 0.968 [0.887, 1.037] | 0.977 [0.973, 0.980] | 3.834 [1.301, 7.679] | 3/3 |
| H4K20me1 | ENCSR000ARC.pooled.uniform.base | antibody_unknown | 0.933 [0.915, 0.967] | 0.975 [0.963, 0.985] | 3.400 [0.853, 5.874] | 1/3 |

Summary per arm: targets where every seed met the rule.

| arm | met / targets |
|---|---|
| antibody_other | 8 / 8 |
| antibody_unknown | 0 / 8 |
| dedup_flip | 0 / 8 |
| depth_down | 8 / 8 |
| depth_up | 8 / 8 |
| lab_other | 5 / 8 |
| lab_unknown | 1 / 8 |
| mapq_flip | 1 / 8 |
| platform_other | 8 / 8 |
| platform_unknown | 4 / 7 |
| read_length | 7 / 8 |
| run_type_flip | 2 / 8 |
