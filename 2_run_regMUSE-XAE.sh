#!/bin/bash

# create or update the .py executable, based on the .ipynb
conda activate my_base
jupytext --set-formats ipynb,py regMUSE_XAE.ipynb --sync

# run pipe
conda activate nextflow

export NXF_DEFAULT_DSL=1

mkdir -p log/

nextflow -log $PWD/log/nextflow.log run pipe.nf --filename_real_data "original_coeff.tsv" \
												--filename_permuted_data_training "perm_coeff_iter*_training.tsv" \
												--filename_permuted_data_validation "perm_coeff_validation.tsv" \
												--minutes 5 \
												--memGB 4 \
												-resume
