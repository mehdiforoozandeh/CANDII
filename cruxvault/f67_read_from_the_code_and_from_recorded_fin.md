---
id: f67
type: finding
schema: 3
title: Read from the code and from recorded findings, with line citations checked;…
parent: q34
confidence: moderate
why: Read from the code and from recorded findings, with line citations checked; moderate because no experiment on the onion model has tested the hard offset or the dispersion across depth yet.
evidence: results/eta_space/MODEL_DESIGN.md
created: "2026-11-14T16:38:01"
updated: "2026-11-14T16:38:01"
---

# f67 — Read from the code and from recorded findings, with line citations checked;…

Parent:: [[q34_what_does_the_eta_formulation_change_in_]]

## Answer

The rung-1 model already uses the eta formulation for the mean: count_transform 'offset' is the default (tools/minicandi/configs/rung1/base.json line 9), the input is log2((y + 1)/s) with size factor s = 2^(log2 depth - 24), and the head emits eta with mean = s x 2^eta (tools/minicandi/components.py, OFFSET_ADD_BACK and add_back). The model also reads log2 depth as a soft covariate (column 0 of C and C', tools/minicandi/contract.py LOG2_DEPTH_COL = 0) on the input side and before the head, and the NB size is a per-bin output, so both eta and the dispersion can depend on depth. 'Use eta throughout' is therefore already done for the mean; the open choices are a hard offset (depth hidden from the encoders), a dispersion tied across depth, and eta-space readouts in the evaluation. The earlier offset study (t125, 432 runs; f17) found the depth law held in 0 of 72 settings and the offset helped CRPS in 6 of 54 cells, all in the simplest design, so it gives no reason to expect a general accuracy gain from a hard offset. Pure thinning obeys proportionality within 0.71% (q7), while non-subset pairs depart by about 7% in median (half-against-half floor 0.099 [0.072, 0.135] log2 on the depth-law metric), and rung 1 seed 0 scored 0.1712 on 16 regions against the floor's 0.1705, so the depth-law metric on these pairs cannot separate a model that bends the law from a truth that is not proportional. Not run: any of the four proposed experiments (no code change, no job).
