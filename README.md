Modified from
--
https://github.com/compbiomed-unito/MUSE-XAE

Main changes
--
 - it accepts any number of variables (i.e. neurons in the input layer) rather that just the 96 trinucleotide SBS
 - it accepts non-count variables, so loss function is not necessarily Poisson, and normalization is done differently (across-samples standardization) to account for negative values, or not done at all
 - data augmentation is done a priori by resampling coefficients from their CI80%
 - a single and unique resampling is used at each epoch
 - no EarlyStopping (instead I recover the epoch with lowest validation loss)
 - no mapping of signatures to COSMIC
 - it's optional to have the non-negative weights constraint in the decoder (kernel_constraint=NonNeg())
 - each sample's exposure to each signature is obtained
 - it's possible to change the batch size, the % of samples used for validation, and the dimensions (neurons) of the 1st encoder layer (the 2nd and 3rd are fractions thereof)
 - parallelization of number of signatures, and all of the previous point, is done externally with Nextflow
 
