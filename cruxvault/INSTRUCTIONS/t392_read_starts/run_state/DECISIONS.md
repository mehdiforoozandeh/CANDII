# Decisions in force (t392)

| id | date | decision | source (verbatim or near) | provenance | supersedes |
|---|---|---|---|---|---|
| D1 | 2026-11-23 | Staged: 16 regions, then stop; 256 only on PI's word | "The run scores on 16 regions, then stops." H | PI | |
| D2 | 2026-11-23 | Score both models against overlap-count truth; read-start predictions converted by box convolution, NB moment-matched | H settled decisions | PI | |
| D3 | 2026-11-23 | Reference = halves_mix_avg_pearson seeds 0-2, reused | H | PI | |
| D4 | 2026-11-23 | ChIP stranded, DNase unstranded | H | PI | |
| D5 | 2026-11-23 | All communication through foreman local_b5c55e32 | H | PI | |
| D6 | 2026-11-23 | Run memory in run_state/, not .longrun/ (t370's); plan file here too | session prompt | foreman | longrun default |
| D7 | 2026-11-23 | Ledger written directly from the two short docs, no reader agent | agent | agent | |
| D8 | 2026-11-23 | Merge origin/exp/t370-minicandi-recipe into umbrella after PR 65 lands; deviance scoring after that merge; report input-check results (esp. figures differing from handoff) as soon as done | foreman message 2026-11-23 | foreman | |
| D9 | 2026-11-23 | Gate is exact at read level: each read's span placed from its own start rebuilds eic_v1_2 with 0 differing bins on all 8,220 files. Predictions convert with each measurement's measured average span kernel. Report conversion error of observed binned starts vs overlap truth as the floor. | PI via foreman: "Read-level gate + floor": yes | PI | INSTRUCTIONS §3 item 2 gate wording |
| D10 | 2026-11-23 | Build site: Fir, after PI-started transfer of 6 tars + testbed readsets (pending foreman confirm). | PI: "which ever gives us the results ASAP." | PI | |
| D11 | 2026-11-23 | ATAC-seq: one channel like DNase; store keeps both strands for all assays | agent; sent to foreman, no objection yet | agent proposal | |
| D12 | 2026-11-23 | 600 CPU-h cap counts CPUTimeRAW (cores x hours), same measure as reference figures; check spend before deviance pass | agent; told foreman | agent | |
| D13 | 2026-11-23 | Never git add .longrun/; the glossary hook writes there uncommitted | plan review | agent | |
| D14 | 2026-11-23 | New read-start dataset class accepts average_h5; 1 bp ResDataset keeps refusal. Not a deviation. | foreman: "new class, yes ... not a deviation" | foreman (author of the instruction) | |
| D15 | 2026-11-23 | Chromosome grid per measurement from the eic_v1_2 h5 keys (chrY fix); rerun the same 4-measurement probe on kit RS3; any differing bin stops again; on pass, build on Fir once tars land | PI via foreman: "yes" | PI | |
| D16 | 2026-11-23 | Read starts in and out: inputs, average input and targets are read starts; only the output is converted for scoring | PI via foreman: "yes" | PI | |
| D17 | 2026-11-23 | Comparison stays at 1x windows on both sides, though the PI picked the 2x-windows full combination as final | foreman relay | PI | |
| D18 | 2026-11-23 | Interim notes in run_state/notes/<x>/; Nibi /nearline reads on the login node under nice; --testbed-root swap; store chunks (min(4096,n_bins),2) | plan Log amendments | foreman/agent | plan §results paths |
| D19 | 2026-11-24 | Store build time requests: 00:30:00 per recount task, 04:00:00 for the average job (288 CPU-h requested; used 79.0) | agent, over the 300 CPU-h chunk cap | agent | notes/build/BUILD.md |
| D20 | 2026-11-24 | Own unpacked tagAligns are deleted after the gate passes (tars kept); Fir pytest basetemp is node-local ($SLURM_TMPDIR) | foreman: file-count warning, Mehdi's yes on pytest folders | foreman | slurm/t392/rs_tests.sh |
| D21 | 2026-11-24 | Training: rs_train.sh passes extra args (kit RS4); seed 0 first, seeds 1-2 released only if seed 0 projects <= ~80 CPU-h; read at ~1 h. Seeds 1-2 sat held (not resubmitted) after the foreman's --hold correction; released 09:40Z on a 68 CPU-h projection | PI via foreman | PI | notes/train/jobs.tsv |
