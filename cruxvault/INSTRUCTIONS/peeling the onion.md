(nov 9 2026)

the way i see it, there are different axes and sources of variation in the data that we are working with. below i'll try to explain them one by one. knowing this helps us in how we design CANDI, how we evaluate it, and to know what's possible and what's not possible.

# sources of variation
> Background: [[onion variation and objectives]] · caveats on claims in this doc: [[onion caveats]]

1. data-generation variation
	- these are different sources of variation and artifacts that get introduced into our data coming from what happens in the lab from getting the sample from the donor up to sequencing it and getting the raw sequence data
	- good resource - especially regarding definition of different replicates
		- every type of replicate mentioned here except Pseudoreplicate which are related to data-processing and not data-generation
		- https://www.encodeproject.org/data-standards/terms/
	- depending on the data-generation process and the type of experiment, this category can be broken down to more granular sources of artifact variation
		- e.g. lab, sequencing technology, antigen, etc.
	- these are also the type of artifact/variations that we want to be able to model away from the data and make it robust to it (and measure and show that robustness)

2. data-processing variation
	- starting from raw sequencing data to QC, alignment, peak calling, etc. these are different sources of variation that end up in the final data format that CANDI sees (both during training and inference)

	- depending on the data-processing pipelines, this category can be broken down to more granular sources of variation or artifact

3. biological variation 
	this is the type of variation that we are most interested in capturing and predicting. 
	1. cell-type specific variation
		- e.g. differential expression and genome regulation -- some parts of genome are active in some celltypes and not active in others
		- obviously the most granular form of this happens at single cell level. but in our current data we are dealing with bulk (celltypes and tissue types and cell lines).
		
	2. position specific variation
		- obviously, different positions of the genome have different roles and are in different epigenetic states.
		
	3. assay specific variation
		- obviously, different assays are measuring different things and have different patterns and values. 
		
	4. population-level variation
		- the forth dimension to the biological variation ladder is population level variation
		- same assay, same genomic position, same celltype in the same organ have variation in expression when you look at it at different individuals.
		- this is the hardest to measure and to predict -- it can be broken down into more granular sources of variance: age, sex, genetic ancestry, and sequence/allelic variantions. 

---
# objectives
> Background: [[onion variation and objectives]]

in the above breakdown of variation, i'd call type 1 and 2 (data generation and processing) as "Artifacts variation". 

With CANDI, our goal is:
1. biological variation: be able to impute across different axes of biological variation
	1. main priority: similar to other imputation methods, we want to impute missing experiments from the available ones i.e. across the <Cell, Assay, Position> axes.
	2. nice to have upon data availability: population level prediction will be predicting individual epigenomes -- would need data for it. if readily available we can consider it. have a look at eDICE paper and this paper: https://www.biorxiv.org/content/10.64898/2026.02.01.702969v1.full.pdf
2. artifacts variation: be able to model-away/remove/correct artifacts (type 1 and 2 of variation) i.e. given the covariates (of both data generation and processing), we want to be able to go from signal obtained from one covariate set A to the same signal as if it was obtained from another covariate set B. 
3. uncertainty: obviously our modelling of both biological variation and artifact variation will be imperfect. same for how much we can model the unmodeled inherent biological variation.  so our third goal is for CANDI to be able to output calibrated aleatoric uncertainty. aleatoric uncertainty is whatever remained and unmodeled variance there is the data. we want candi to predict that by predicting probability distributions instead of point values.

---
# eval
> Headline metrics and background: [[onion eval protocol]]

- one of our core goals is to establish how we can measure and evaluate how much of the above objectives we are achieving. i.e. a detailed break down of different sources of variation that we are modelling (bio + artifact -- broken down even more based on the above explanation) and also how calibrated our confidence measurements are?
	- this means that we want our evaluation protocol to be basically an accurate and fair decomposition of how much variance there is and peeling the onion i.e. dividing the total variance by its different components and finally what can't be modelled. 
	- then we want to show how much of the variation  is being captured by CANDI (and other methods) and how much of the remaining variation is it putting into its predicted aleatoric uncertainty and how calibrated it is. 
- we want a balanced set of metrics and measurements that cover all of the things that we should measure and evaluate.  
	- e.g. we can measure how much variance we are predicting in different subsets of the genome: e.g.. promoters, enhancers, gene bodies, foreground, background, heterochromatin, etc. 
	- same for the amount of variance caused by different artifact sources: depth, sequencing platform, antigen efficiency, etc. 
	- we will also have the same measurements for different assays
	- and most importantly, we must be able measure the amount of cell type specific and non cell type specific variations that we can capture and predict. 
- to measure artifact variation well, we might need different types of replicates (pseudoreplicates, isogenic, etc.) -- some of these can be processed by us i.e. those related to data-processing variation and those related to generation must be obtained from ENCODE portal (or any other portal) upon availability. 
- basically in our data collection (explained below), we must look for types of data (augmentations, replicates, etc.) that can help us peel the onion. e.g. pseudoreplicates help establish the poisson-error component of the variation. or biological vs technical replicates can help us identify different layers of variation. -- in other words, we need any kind of data views that we can collect to define the boundary of detectable variation coming different source -- e.g. distinguish celltype specific-signal from artifact-related noise.

so we need to formalize this into our evaluation system: clearly define a new metric set reflecting the above formulation and definition of the task CANDI is trying to address. let's make a structured and well defined evaluation benchmark based on this. we want to score other methods and baselines against the same eval set. 

 **interesting question:** do different types of variation captured from different sources depend on resolution? for example, do we see most of the data-processing artifacts at very granular resolutions? or do we see cell-type specific variations at let's say 500bp resolution mostly? this is an interesting thing to think about and investigate if possible. the answer to this question can guide us towards a more informed decision on the resolution of epigenomic data that CANDI works on. 

reference: a good reference for this kind of eval is Max's "Explained and unexplained variance" results that are also on the github pages leaderboard. we want to expand on that and formalize it. 

---
# data
> Background: [[onion data plan]]

we have two datasets right now:

- eic dataset (published by ENCODE Imputation Challenge)
- EED (extended ENCODE portal Dataset): a manually collected dataset for CANDI from ENCODE portal aiming for more training data

The two above datasets were collected when I was actually thinking about this problem with only three dimensions. One dimension being assays, the other being cell types, the other being genomic positions. This is the official formulation of this problem and data tensor By previous imputation methods. 

But now after more than two years of working on this problem, I can claim that the initial formulation of this problem in those three axes only was reductionistic, and there is actually another very important dimension to this problem.

I'd name the **4th dimension** "**measurement**".

this axis is basically the experimental covariates and conditions (sources 1 and 2 of variation) that exist in the dataset and are not biologically interesting but problematic - distorting our observations of the biological truth. 

in this new definition of dataset, we must define the data instance differently. 
- each data "instance" is an array of a particular measurement over the whole genome. one thread in this 4 dimensional manifold. e.g. in cell c, assay a, with experimental metadata m.  this will give us back the 3D tensor (CAM:celltype-assay-measurement) but this time the 3rd dimension is the measurement. 
 
in the 2D slice matrix of Cell-assay, for each entry we can have many possible options: these are different form of replicates, or augmented views, or duplicates of same cell-assay pairs - let's think of the criteria for adding something to the dataset is for it to help us identify a new source of variation. this is exactly what i meant by what i explained above once about needing different data views to help with breaking down the contribution of each source of variation to the total variation and then measuring how much of it candi can recover. we can also use the same data views as augmentations of the data and use it for training. 

so we need to start preparing a third dataset based on the above definitions and goals.
sources for this dataset can be  ENCODE, ChIP-Atlas, also EpiATLAS by IHEC: https://ihec-epigenomes.org/epiatlas/data/

in the train-test split, we must aim for a balanced split along all three dimensions (CAM:celltype-assay-measurement) -- i.e. an imbalanced split along the M axis, ruined encode imputation challenge. 

We also want to show in the paper that we are addressing the issue that plagued the EIC paper. So we want to show that CANDI can solve that by taking into account this 4th dimension (measurements). therefore, we need to collect new data for training CANDI. the requirement here is that we need to collect data that can help us train a version of CANDI with this new definition but on exact EIC's experiments (hard checked by accession -- what we are working with should exactly match paper's published accessions) - we want to be able to fairly compare CANDI's results with published EIC predictions and results. 

so to recap:
- we are collecting a new extended dataset
- we are retiring EED
- we are updating our EIC dataset -- to train the new CANDI

---
# miniCANDI
> Background: [[miniCANDI design notes]] · ablation background: [[miniCANDI architecture ablations]]

we want to redesign CANDI from first principles based on the above formulation of the problem - our first step towards that is miniCANDI - look at /principles scaling vs prototype. 
- miniCANDI is the prototype and we move on to CANDI when we are sure that miniCANDI prototype is well defined and working and we are ready to scale (miniCANDI -scale-> CANDI)

optimizing miniCANDI is faster and easier -- feedback loops are faster 

here are architecture ablations for miniCANDI:
[[miniCANDI architecture ablations]]

We have gathered a lot of insights and results and lessons with our covariate artifact correction (CAC) experiments. we have a formulation of:

g(c,c') -> parameters of function f
f(X) -> X'

we also showed that this model generalizes to unseen C,C' pairs if we first go from (X, C) to (X_base, C_base) and from (X_base, C_base) to (X', C') -- i.e. a two step CAC. 

our CAC experiments were done in an isolated test bed where we decoupled the two tasks of CAC and imputation. now we must redefine and redesign minicandi to do both masked assay imputation and CAC. 

minicandi:
- takes as input: 
	- X: epigenomic signals - for a context length W, for one cell type, all assays available
		- size: (B, W/resolution, num_all_assays) -> a subset of num_all_assays are either masked or missing
		- ablation: experiment with adding chip controls (upon availability for those celltypes) as one assay
	- S: DNA sequence for the context length W:
		- size: (B, W, onehot-encoded)
	- C: covariates of input experiments (measurement dimension)
		- size: (B, num_total_assays, num_all_covariates)
	- C': covariates of desired output experiments (measurement dimension)
		- size: (B, num_total_assays, num_all_covariates)
	- optional/ablation: 
		- average baseline: for each assay type, we can get its average activity along the genome (training set only) - average baseline per-assay type 
		- size: (B, W/resolution, num_all_assays)
- outputs: 
	- /hat{X'}: imputed/predicted full epigenomic signals with C' covariates
		- parameters of NB i.e. two parameters per position
			- size: 2 tensors each (B, W/resolution, num_all_assays)
	- ablation: also predict the chip control
	- based on CAC findings, we can use the exact formula to go from predicted counts to signal p-value and even binary peaks -- so we don't need dedicated decoders for signal p-value and peaks. we can optionally generate them from our predicted NB
		- if we are also predicting chip control, we'd be relying on our own NB predictions of both signal and control 
- objective(s):
	- core loss: NB NLL on the output predicted NB distribution
		- on masked assays (imputation)
		- on available assays (reconstruction/denoising)
	- ablations: any other loss/regularization/etc. that can help with our three main objectives: 1) imputation 2) covariate artifact correction 3) calibrated uncertainty estimation
- important notes:
	- how we embed and fuse different modalities i.e. epigenomic count data, dna sequence, and covariate information is an important an non trivial part of the whole architecture design. these are targets for ablation i.e. covariate embedding and covariate injection, DNA embedding and injection, etc.
	- for DNA embeddings we have three options:
		- our own DNA tower trained from scratch
		- using Flashzoi embeddings
		- using a genomic language model embeddings e.g. nucleotide transformer
	- as an ablation, the count input data to the model can be depth controlled

later on at scaling, we want to scale on three axes and observe the effects:
	- data: more celltypes and assays
	- context: bigger context window
	- model size: bigger model with more parameters