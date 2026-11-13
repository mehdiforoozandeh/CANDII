---
title: onion region budget and selection
kind: open questions
parent: "[[peeling the onion]]"
siblings: ["[[long-horizon decisions]]", "[[long-horizon notepad]]", "[[onion eval protocol]]", "[[onion data plan]]", "[[miniCANDI design notes]]", "[[miniCANDI architecture ablations]]"]
created: 2026-11-13
status: questions added by the PI on 2026-11-13. Open. No answer is recorded here yet.
---

# onion region budget and selection

The PI added these questions to the goals of the analysis on 2026-11-13. The PI's words are quoted. The other text is a draft by the agent and is marked.

## The PI's questions

> how many training regions do we need? (it's likely related to the scaling phase and our model capacity and how fast we want our feedback loop to be during prototype optimization)

> how should we select the training regions? randomly? stratified for ccres? stratified by chromhmm annotations? stratified by something else? is there a better more defined way of selecting the best training regions if our training budget is limited (e.g. prototype optimization phase)

> i'm not sure if 256 regions is enough

> answering these might need research into the literature and web and some experiments with actual data

## ==Reading of the questions (agent draft)==

==The questions cover two different sets of regions. The agent keeps them apart.==

1. ==**Training regions.** How many windows of the genome a miniCANDI run trains on, and how those windows are chosen. The PI's two questions are about this set.==
2. ==**Evaluation regions.** The fixed regions on which every arm is scored. The remark "256 regions" is about this set. On 2026-11-13 the evaluation set of `eic_v1` holds 256 regions of 16,384 bins of 25 bp, i.e. 256 x 16,384 x 25 bp = 104.9 Mb, drawn at random with a seed from 6,724 candidate blocks (source: `cruxvault/results/t283/REGIONS.md`).==

## ==Sub-questions (agent draft)==

==Q1. Training budget.==
- ==How does each headline metric change with the number of training windows, at a fixed model size? Where does the curve flatten for a model of about 2 M parameters?==
- ==How does that point move with model capacity, i.e. is the small model limited by data or by capacity?==
- ==Which smallest budget still ranks arms in the same order as the full budget? That budget sets the speed of the feedback loop in the prototype phase.==

==Q2. Training selection.==
- ==Does a stratified draw beat a uniform random draw at an equal budget? Candidate strata: candidate cis-regulatory elements (cCREs), ChromHMM states, signal variance across cell types, gene density, mappability.==
- ==Does the gain of a stratified draw hold on the regions that the strata under-sample, e.g. quiescent regions and broad domains? A draw that favors active elements can hurt H3K9me3 and H3K27me3.==
- ==Is there a principled selection rule for a small budget, e.g. coverage of the space of cell-type-specific signal, or selection by the loss of a cheap proxy model? Any rule that reads validation or blind tracks is a leak and is excluded.==

==Q3. Evaluation regions.==
- ==Is the seed spread of the composite score small enough at 256 regions? The plan already has this check after ladder rung 1: if 2 x the seed spread of the composite exceeds 0.05, the number of scored regions or seeds is raised.==
- ==How wide are the confidence intervals per assay and per variance layer at 256 regions, and how do they shrink at 512 and 1,024?==
- ==Should the evaluation regions be stratified, so that rare region classes get enough bins for a per-stratum metric?==

## ==How the agent will answer (intent, not a result)==

1. ==Literature and web: how Enformer, Borzoi, AlphaGenome 1 and 2, Avocado, eDICE, the EIC entries, scooby and corgi choose the count and the selection of their training regions, and what is published on stratified sampling by regulatory annotation.==
2. ==Code: what the current sampler does, and which budgets the ladder configs use.==
3. ==Experiments on `eic_v1`: a budget curve and a selection comparison at an equal budget, each with 3 seeds, scored on the fixed evaluation set. The plan amendment states the arms, the cost and the rule that settles them.==

## PI rulings of 2026-11-13

- **Seen positions.** No change to the protocol now. Training windows may fall inside the evaluation regions. One experiment arm excludes the evaluation regions from training and is scored on the same set, so the gap between seen and unseen positions is measured. The PI rules after he sees it.
- **Annotations.** The selection experiment uses leak-free strata only. cCRE and ChromHMM annotations are not used now. The PI asked to keep the idea for later; see the next section.
- **Evaluation regions.** The count may go from 256 to 512 on the evidence of the interval study. A step to 1,024 comes back to the PI with the interval curve.

## Ideas kept for later

==**Training regions stratified by cCRE class or by ChromHMM state.** The idea is to draw training windows so that each regulatory class or chromatin state gets a chosen share of the budget, with loss weights that keep the objective unbiased. It is not used now, because both annotation sets may be built from data that includes EIC validation or blind cells, which makes them a leak for the EIC comparison. Routes that would make it usable later:==

1. ==Use it as a labelled side arm, never for a headline or a default: one cCRE arm and one ChromHMM arm, about 6 runs and one annotation fetch.==
2. ==Rebuild the annotation without the held-out cells: a ChromHMM model trained only on training cells, or a cCRE-like set called only from training tracks. The peak-frequency strata of the current experiment are a first step in this direction.==
3. ==Use it on the extended dataset, where the split is by cell type and the annotation's source cells can be checked against the split.==

==Things to settle before it runs: the registry version and license of the cCRE set, the list of cells behind the full-stack ChromHMM model, and the genome share of each class, which sets the loss weights.==

## Answers

==None yet. Results go to `cruxvault/results/` and are cited here.==
