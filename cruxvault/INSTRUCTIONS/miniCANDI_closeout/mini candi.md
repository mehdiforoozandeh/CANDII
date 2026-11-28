
based on the results of the peeling the onion sprint and all the designs and ablations that went into miniCANDI, some are still remaining.

one is that even though unet had the best imputation quality, its correction capabilities were lower than the other models (conv and conv+atten)

there are other things about the minicandi results that need some deeper investigation

what if we can make a variant of the unet that gives us the best of both worlds results i.e. breaking between the pareto of imputation quality and correction.

what if it was the best at both?

that's what we want to find.

the optimal architecture that can deliver all three:

1. best imputation
    
2. best correction
    
3. best uncertainty calibration
    

for all three, we have metrics.

we might think of different ways to try to see if we can improve all three together - not compromising the correction capability.

1. one thing that i was thinking about was alternative ways to do covariate correction. for example having CAC (same f, g formulation) before the input to CANDI (preprocessing layer of the counts) and after the output (post processing of predicted NB) -- that's just a suggested ablation


2. another way to do CAC, is to play with where we are injecting? count conv and decov towers? attention layers? all?

3. two step CAC through base was shown to be helpful. does that improve our performance?


also in the peeling the onion docs i mentioned that one very interesting ablation for me is to have chip control as both input and output assay and also train candi to be able to predict chip controls. -- in cells where we don't have chip control, for all post processing of the predicted counts (into signal p-value or log Fold change over control), having a control can be very helpful. i definitely want to see if we can get this variant working.

moreover, we showed that currently we are beating naive baselines. however we haven't shown yet that we can beat other imputation methods in EIC based on this new minicandi results. to do that, we need to also compare minicandi against other imputation methods on our imputation metrics (+some of EIC imputation metrics). for this, we might need to process the predicted count data from miniCANDI to signal pvalues (which are comparable with the EIC)

an advantage that minicandi over other EIC methods is that minicandi gets to see other CAC related counterfactual views of the training data. hoping to address the main EIC issue of covariate-dependent batch effect across training and test set. our goal is to make miniCANDI robust to this covariate shift and also be a good imputation method (+ caliberated uncertainty)

i repeat, your goal for this session is to define and finalize the best recipe for minicandi one that's likely the best based on our objectives and data and problem formulation.