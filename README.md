Modified from
--
https://github.com/compbiomed-unito/MUSE-XAE

Main changes
--
 - It accepts any number of variables (i.e. neurons in the input layer) rather that just the 96 trinucleotide SBS
 - It accepts non-count variables (so loss function is not necessarily Poisson-based), and normalization is done differently (across-samples standardization) to account for negative values, or not done at all
 - Data augmentation is done a priori by resampling coefficients from their confidence intervals
 - At each independent iteration there is a different seed
 - In the main training (the one to obtain the signature weights) a single and unique resampling is used at each epoch (i.e. actually there is only 1 epoch and an "infinitely-augmented" dataset); I recover the "epoch" with lowest validation loss, so in practice it has the same result as with EarlyStopping but taking longer time because it will go through all the specified "epochs". In the "refit" step (renamed "encoder_prediction()", i.e. the one to obtain the exposures) settings are mostly like in MUSE-XAE
 - No mapping of signatures to COSMIC
 - It's optional to have the non-negative weights constraint in the decoder (kernel_constraint=NonNeg()) -- In my data the model is fitted better without this constraint
 - It's possible to change the batch size, the % of samples used for validation, and the dimensions (neurons) of the 1st encoder layer (the 2nd and 3rd are fractions thereof) -- The default ones for MUSE-XAE work well though
 - Parallelization of number of signatures, and all of the previous point, is done externally with Nextflow
 