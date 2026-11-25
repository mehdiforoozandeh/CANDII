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
| DNase-seq | ENCSR080PZL.pooled.uniform.base | depth_up | 1.986 [1.958, 2.017] | 0.994 [0.993, 0.994] | 1.095 [1.020, 1.240] | 3/3 |
| DNase-seq | ENCSR080PZL.pooled.uniform.base | depth_down | 0.514 [0.500, 0.527] | 0.993 [0.992, 0.995] | 0.857 [0.825, 0.908] | 3/3 |
| DNase-seq | ENCSR080PZL.pooled.uniform.base | run_type_flip | 1.007 [0.985, 1.042] | 0.992 [0.988, 0.995] | 0.644 [0.567, 0.757] | 2/3 |
| DNase-seq | ENCSR080PZL.pooled.uniform.base | read_length | 1.130 [1.088, 1.201] | 0.994 [0.990, 0.997] | 0.986 [0.905, 1.038] | 3/3 |
| DNase-seq | ENCSR080PZL.pooled.uniform.base | dedup_flip | 0.989 [0.979, 0.997] | 0.998 [0.995, 1.000] | 0.720 [0.614, 0.806] | 0/3 |
| DNase-seq | ENCSR080PZL.pooled.uniform.base | mapq_flip | 1.027 [0.985, 1.072] | 0.960 [0.931, 0.976] | 1.010 [0.910, 1.186] | 2/3 |
| DNase-seq | ENCSR080PZL.pooled.uniform.base | lab_unknown | 0.979 [0.927, 1.048] | 0.975 [0.959, 0.983] | 1.033 [0.723, 1.229] | 1/3 |
| DNase-seq | ENCSR080PZL.pooled.uniform.base | platform_other | 1.068 [1.035, 1.112] | 0.995 [0.993, 0.995] | 0.558 [0.349, 0.758] | 3/3 |
| DNase-seq | ENCSR080PZL.pooled.uniform.base | platform_unknown | 1.063 [1.001, 1.159] | 0.991 [0.982, 0.998] | 0.686 [0.280, 0.967] | 3/3 |
| H2AFZ | ENCSR124DYB.pooled.uniform.base | depth_up | 1.992 [1.974, 2.020] | 0.993 [0.992, 0.994] | 1.176 [1.098, 1.251] | 3/3 |
| H2AFZ | ENCSR124DYB.pooled.uniform.base | depth_down | 0.509 [0.504, 0.516] | 0.993 [0.992, 0.994] | 0.815 [0.711, 0.889] | 3/3 |
| H2AFZ | ENCSR124DYB.pooled.uniform.base | run_type_flip | 0.978 [0.947, 1.018] | 0.998 [0.997, 0.999] | 1.218 [0.695, 2.061] | 2/3 |
| H2AFZ | ENCSR124DYB.pooled.uniform.base | read_length | 1.172 [1.018, 1.321] | 0.986 [0.973, 0.999] | 1.195 [0.863, 1.712] | 3/3 |
| H2AFZ | ENCSR124DYB.pooled.uniform.base | dedup_flip | 0.971 [0.957, 0.986] | 0.997 [0.996, 0.999] | 0.211 [0.153, 0.257] | 0/3 |
| H2AFZ | ENCSR124DYB.pooled.uniform.base | mapq_flip | 0.926 [0.861, 1.017] | 0.921 [0.884, 0.975] | 1.640 [1.246, 2.423] | 1/3 |
| H2AFZ | ENCSR124DYB.pooled.uniform.base | lab_other | 0.921 [0.831, 0.975] | 0.976 [0.968, 0.992] | 1.500 [1.157, 2.138] | 3/3 |
| H2AFZ | ENCSR124DYB.pooled.uniform.base | lab_unknown | 1.000 [0.987, 1.014] | 0.989 [0.971, 0.999] | 1.209 [0.827, 1.599] | 1/3 |
| H2AFZ | ENCSR124DYB.pooled.uniform.base | platform_other | 0.884 [0.757, 0.963] | 0.971 [0.945, 0.990] | 2.606 [1.161, 5.201] | 2/3 |
| H2AFZ | ENCSR124DYB.pooled.uniform.base | platform_unknown | 0.874 [0.745, 1.011] | 0.976 [0.955, 0.991] | 1.530 [0.386, 3.698] | 1/3 |
| H2AFZ | ENCSR124DYB.pooled.uniform.base | antibody_other | 0.913 [0.844, 0.948] | 0.970 [0.955, 0.995] | 5.151 [1.762, 11.036] | 3/3 |
| H2AFZ | ENCSR124DYB.pooled.uniform.base | antibody_unknown | 0.802 [0.755, 0.843] | 0.929 [0.916, 0.939] | 9.302 [2.676, 21.951] | 0/3 |
| H2AK5ac | ENCSR703KXH.pooled.uniform.base | depth_up | 2.022 [1.954, 2.116] | 0.993 [0.991, 0.995] | 1.075 [1.009, 1.115] | 3/3 |
| H2AK5ac | ENCSR703KXH.pooled.uniform.base | depth_down | 0.503 [0.487, 0.517] | 0.992 [0.988, 0.995] | 0.853 [0.816, 0.893] | 3/3 |
| H2AK5ac | ENCSR703KXH.pooled.uniform.base | run_type_flip | 1.004 [0.957, 1.045] | 0.998 [0.998, 0.999] | 0.849 [0.520, 1.078] | 1/3 |
| H2AK5ac | ENCSR703KXH.pooled.uniform.base | read_length | 1.173 [0.993, 1.501] | 0.978 [0.957, 0.990] | 0.953 [0.700, 1.083] | 2/3 |
| H2AK5ac | ENCSR703KXH.pooled.uniform.base | dedup_flip | 1.009 [0.987, 1.027] | 0.999 [0.999, 1.000] | 0.593 [0.522, 0.714] | 2/3 |
| H2AK5ac | ENCSR703KXH.pooled.uniform.base | mapq_flip | 1.027 [1.000, 1.079] | 0.911 [0.890, 0.926] | 1.037 [0.941, 1.126] | 3/3 |
| H2AK5ac | ENCSR703KXH.pooled.uniform.base | lab_unknown | 1.073 [0.921, 1.233] | 0.989 [0.976, 0.999] | 0.662 [0.418, 0.908] | 2/3 |
| H2AK5ac | ENCSR703KXH.pooled.uniform.base | platform_other | 1.009 [0.954, 1.069] | 0.990 [0.981, 0.995] | 0.902 [0.879, 0.947] | 3/3 |
| H2AK5ac | ENCSR703KXH.pooled.uniform.base | platform_unknown | 1.140 [1.034, 1.295] | 0.989 [0.979, 0.994] | 0.570 [0.313, 0.712] | 2/3 |
| H2AK5ac | ENCSR703KXH.pooled.uniform.base | antibody_other | 1.020 [0.942, 1.073] | 0.997 [0.996, 0.998] | 0.829 [0.779, 0.913] | 3/3 |
| H2AK5ac | ENCSR703KXH.pooled.uniform.base | antibody_unknown | 0.815 [0.677, 0.901] | 0.985 [0.978, 0.995] | 0.961 [0.869, 1.136] | 1/3 |
| H2BK12ac | ENCSR523ENE.pooled.uniform.base | depth_up | 1.967 [1.962, 1.974] | 0.994 [0.993, 0.995] | 1.033 [0.975, 1.134] | 3/3 |
| H2BK12ac | ENCSR523ENE.pooled.uniform.base | depth_down | 0.510 [0.503, 0.515] | 0.994 [0.993, 0.995] | 0.954 [0.920, 1.018] | 3/3 |
| H2BK12ac | ENCSR523ENE.pooled.uniform.base | run_type_flip | 0.938 [0.915, 0.951] | 0.999 [0.999, 1.000] | 0.991 [0.885, 1.171] | 3/3 |
| H2BK12ac | ENCSR523ENE.pooled.uniform.base | read_length | 1.200 [0.907, 1.614] | 0.990 [0.983, 0.997] | 1.249 [0.882, 1.500] | 2/3 |
| H2BK12ac | ENCSR523ENE.pooled.uniform.base | dedup_flip | 0.997 [0.973, 1.032] | 1.000 [0.999, 1.000] | 0.680 [0.552, 0.767] | 1/3 |
| H2BK12ac | ENCSR523ENE.pooled.uniform.base | mapq_flip | 0.920 [0.901, 0.932] | 0.985 [0.982, 0.987] | 1.142 [1.124, 1.161] | 0/3 |
| H2BK12ac | ENCSR523ENE.pooled.uniform.base | lab_unknown | 1.026 [0.916, 1.123] | 0.998 [0.996, 1.000] | 1.054 [0.969, 1.156] | 1/3 |
| H2BK12ac | ENCSR523ENE.pooled.uniform.base | platform_other | 0.939 [0.882, 0.987] | 0.996 [0.995, 0.997] | 1.356 [0.901, 1.647] | 3/3 |
| H2BK12ac | ENCSR523ENE.pooled.uniform.base | platform_unknown | 1.002 [0.970, 1.024] | 0.999 [0.998, 1.000] | 0.717 [0.488, 0.941] | 3/3 |
| H2BK12ac | ENCSR523ENE.pooled.uniform.base | antibody_other | 0.991 [0.948, 1.028] | 0.999 [0.999, 1.000] | 1.188 [0.927, 1.706] | 3/3 |
| H2BK12ac | ENCSR523ENE.pooled.uniform.base | antibody_unknown | 0.840 [0.706, 0.955] | 0.996 [0.994, 0.998] | 1.604 [1.094, 1.885] | 0/3 |
| H2BK15ac | ENCSR123JZO.pooled.uniform.base | depth_up | 1.986 [1.956, 2.002] | 0.995 [0.994, 0.996] | 1.217 [1.155, 1.264] | 3/3 |
| H2BK15ac | ENCSR123JZO.pooled.uniform.base | depth_down | 0.508 [0.504, 0.511] | 0.995 [0.994, 0.996] | 0.767 [0.682, 0.813] | 3/3 |
| H2BK15ac | ENCSR123JZO.pooled.uniform.base | run_type_flip | 0.951 [0.929, 0.965] | 0.995 [0.993, 0.999] | 0.641 [0.358, 1.171] | 3/3 |
| H2BK15ac | ENCSR123JZO.pooled.uniform.base | read_length | 1.161 [0.984, 1.361] | 0.984 [0.982, 0.987] | 0.871 [0.536, 1.201] | 2/3 |
| H2BK15ac | ENCSR123JZO.pooled.uniform.base | dedup_flip | 1.011 [0.963, 1.055] | 0.998 [0.998, 0.998] | 0.064 [0.046, 0.088] | 2/3 |
| H2BK15ac | ENCSR123JZO.pooled.uniform.base | mapq_flip | 0.981 [0.944, 1.051] | 0.869 [0.854, 0.894] | 1.169 [1.008, 1.371] | 1/3 |
| H2BK15ac | ENCSR123JZO.pooled.uniform.base | lab_unknown | 0.952 [0.878, 1.024] | 0.966 [0.903, 0.999] | 0.879 [0.710, 1.210] | 2/3 |
| H2BK15ac | ENCSR123JZO.pooled.uniform.base | platform_other | 0.985 [0.916, 1.032] | 0.946 [0.939, 0.955] | 2.919 [1.914, 3.585] | 3/3 |
| H2BK15ac | ENCSR123JZO.pooled.uniform.base | platform_unknown | 1.045 [1.008, 1.081] | 0.977 [0.949, 0.992] | 1.157 [0.655, 2.026] | 2/3 |
| H2BK15ac | ENCSR123JZO.pooled.uniform.base | antibody_other | 0.988 [0.885, 1.040] | 0.949 [0.940, 0.953] | 2.511 [1.778, 3.574] | 3/3 |
| H2BK15ac | ENCSR123JZO.pooled.uniform.base | antibody_unknown | 0.935 [0.884, 0.980] | 0.938 [0.934, 0.940] | 2.553 [1.814, 3.102] | 0/3 |
| H2BK5ac | ENCSR213TIR.pooled.uniform.base | depth_up | 1.978 [1.964, 1.998] | 0.994 [0.993, 0.995] | 1.181 [1.090, 1.274] | 3/3 |
| H2BK5ac | ENCSR213TIR.pooled.uniform.base | depth_down | 0.512 [0.506, 0.516] | 0.994 [0.993, 0.995] | 0.806 [0.734, 0.903] | 3/3 |
| H2BK5ac | ENCSR213TIR.pooled.uniform.base | run_type_flip | 0.986 [0.950, 1.023] | 0.999 [0.998, 0.999] | 0.687 [0.219, 0.928] | 2/3 |
| H2BK5ac | ENCSR213TIR.pooled.uniform.base | read_length | 1.310 [1.273, 1.345] | 0.984 [0.980, 0.987] | 0.661 [0.259, 1.029] | 3/3 |
| H2BK5ac | ENCSR213TIR.pooled.uniform.base | dedup_flip | 1.033 [1.007, 1.047] | 0.996 [0.995, 0.997] | 0.062 [0.051, 0.074] | 3/3 |
| H2BK5ac | ENCSR213TIR.pooled.uniform.base | mapq_flip | 0.976 [0.954, 1.006] | 0.887 [0.849, 0.910] | 1.191 [0.945, 1.406] | 1/3 |
| H2BK5ac | ENCSR213TIR.pooled.uniform.base | lab_unknown | 0.987 [0.964, 1.026] | 0.984 [0.955, 1.000] | 0.800 [0.400, 1.004] | 2/3 |
| H2BK5ac | ENCSR213TIR.pooled.uniform.base | platform_other | 1.016 [0.975, 1.059] | 0.997 [0.995, 0.999] | 1.266 [1.036, 1.507] | 3/3 |
| H2BK5ac | ENCSR213TIR.pooled.uniform.base | platform_unknown | 1.019 [0.987, 1.041] | 0.992 [0.987, 0.997] | 0.631 [0.340, 1.001] | 2/3 |
| H2BK5ac | ENCSR213TIR.pooled.uniform.base | antibody_other | 0.992 [0.966, 1.034] | 0.997 [0.995, 0.999] | 1.425 [1.028, 1.667] | 3/3 |
| H2BK5ac | ENCSR213TIR.pooled.uniform.base | antibody_unknown | 0.958 [0.934, 0.985] | 0.995 [0.991, 0.999] | 1.488 [1.032, 2.190] | 0/3 |
| H3K14ac | ENCSR687EBF.pooled.uniform.base | depth_up | 1.997 [1.967, 2.045] | 0.994 [0.993, 0.994] | 1.239 [1.131, 1.441] | 3/3 |
| H3K14ac | ENCSR687EBF.pooled.uniform.base | depth_down | 0.505 [0.488, 0.517] | 0.995 [0.994, 0.996] | 0.876 [0.874, 0.880] | 3/3 |
| H3K14ac | ENCSR687EBF.pooled.uniform.base | run_type_flip | 0.969 [0.926, 1.003] | 0.999 [0.998, 0.999] | 0.755 [0.181, 1.207] | 2/3 |
| H3K14ac | ENCSR687EBF.pooled.uniform.base | read_length | 1.196 [1.123, 1.276] | 0.990 [0.986, 0.995] | 0.536 [0.159, 0.864] | 3/3 |
| H3K14ac | ENCSR687EBF.pooled.uniform.base | dedup_flip | 0.988 [0.983, 0.998] | 0.998 [0.998, 0.998] | 0.073 [0.050, 0.087] | 0/3 |
| H3K14ac | ENCSR687EBF.pooled.uniform.base | mapq_flip | 0.979 [0.915, 1.014] | 0.893 [0.868, 0.910] | 1.006 [0.811, 1.110] | 2/3 |
| H3K14ac | ENCSR687EBF.pooled.uniform.base | lab_unknown | 1.013 [0.978, 1.034] | 0.981 [0.945, 0.999] | 1.157 [0.562, 1.801] | 1/3 |
| H3K14ac | ENCSR687EBF.pooled.uniform.base | platform_other | 1.066 [0.982, 1.118] | 0.976 [0.960, 0.997] | 1.658 [0.809, 2.420] | 3/3 |
| H3K14ac | ENCSR687EBF.pooled.uniform.base | platform_unknown | 1.059 [1.015, 1.081] | 0.986 [0.971, 0.998] | 0.471 [0.413, 0.535] | 3/3 |
| H3K14ac | ENCSR687EBF.pooled.uniform.base | antibody_other | 0.984 [0.939, 1.042] | 0.998 [0.996, 0.999] | 1.074 [0.691, 1.352] | 3/3 |
| H3K14ac | ENCSR687EBF.pooled.uniform.base | antibody_unknown | 0.959 [0.943, 0.967] | 0.962 [0.953, 0.978] | 1.906 [1.014, 3.034] | 0/3 |
| H3K18ac | ENCSR604THF.pooled.uniform.base | depth_up | 1.973 [1.931, 1.999] | 0.995 [0.994, 0.995] | 1.229 [1.121, 1.312] | 3/3 |
| H3K18ac | ENCSR604THF.pooled.uniform.base | depth_down | 0.514 [0.503, 0.522] | 0.995 [0.994, 0.995] | 0.846 [0.769, 0.998] | 3/3 |
| H3K18ac | ENCSR604THF.pooled.uniform.base | run_type_flip | 0.940 [0.860, 1.001] | 0.999 [0.998, 1.000] | 0.982 [0.526, 1.335] | 2/3 |
| H3K18ac | ENCSR604THF.pooled.uniform.base | read_length | 1.182 [1.009, 1.360] | 0.991 [0.987, 0.998] | 0.855 [0.716, 0.939] | 3/3 |
| H3K18ac | ENCSR604THF.pooled.uniform.base | dedup_flip | 1.003 [0.989, 1.031] | 0.998 [0.998, 0.999] | 0.163 [0.088, 0.264] | 1/3 |
| H3K18ac | ENCSR604THF.pooled.uniform.base | mapq_flip | 1.010 [0.893, 1.106] | 0.926 [0.921, 0.933] | 1.096 [0.948, 1.251] | 2/3 |
| H3K18ac | ENCSR604THF.pooled.uniform.base | lab_unknown | 1.003 [0.939, 1.125] | 0.987 [0.964, 1.000] | 0.752 [0.220, 1.148] | 2/3 |
| H3K18ac | ENCSR604THF.pooled.uniform.base | platform_other | 1.010 [0.985, 1.037] | 0.988 [0.981, 0.993] | 1.439 [1.269, 1.703] | 3/3 |
| H3K18ac | ENCSR604THF.pooled.uniform.base | platform_unknown | 1.023 [0.974, 1.082] | 0.992 [0.983, 0.999] | 0.624 [0.597, 0.652] | 3/3 |
| H3K18ac | ENCSR604THF.pooled.uniform.base | antibody_other | 0.924 [0.884, 0.986] | 0.976 [0.947, 0.993] | 1.158 [1.101, 1.211] | 3/3 |
| H3K18ac | ENCSR604THF.pooled.uniform.base | antibody_unknown | 0.917 [0.884, 0.957] | 0.968 [0.962, 0.980] | 1.567 [1.054, 2.363] | 0/3 |

Summary per arm: targets where every seed met the rule.

| arm | met / targets |
|---|---|
| antibody_other | 7 / 7 |
| antibody_unknown | 0 / 7 |
| dedup_flip | 1 / 8 |
| depth_down | 8 / 8 |
| depth_up | 8 / 8 |
| lab_other | 1 / 1 |
| lab_unknown | 0 / 8 |
| mapq_flip | 1 / 8 |
| platform_other | 7 / 8 |
| platform_unknown | 4 / 8 |
| read_length | 5 / 8 |
| run_type_flip | 2 / 8 |
