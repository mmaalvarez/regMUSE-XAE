#!/usr/bin/env nextflow

// variables (channels)
n_signatures = Channel.from( [ '2', '3' ] ) //, '4', '5', '6', '7', '8', '9', '10', '11', '12' ] )
epochs = Channel.from( [ '1000' ] )
batch_size = Channel.from( [ '32' ] ) //, '64', '128' ] )
l1_size = Channel.from( [ '64' ] ) //, '128', '256' ] )
validation_perc = Channel.from( [ '10' ] ) //, '20', '30' ] )
normalization = Channel.from( [ 'True' ] ) //, 'False' ] )

process run_autoencoder {

    publishDir "$PWD/res/", mode: 'copy'

    time = { (params.minutes + 5*(task.attempt-1)).min }
    memory = { (params.memGB + 4*(task.attempt-1)).GB }

    input:
    // fixed paths
    val filename_real_data from params.filename_real_data
    val filename_permuted_data_training from params.filename_permuted_data_training
    val filename_permuted_data_validation from params.filename_permuted_data_validation
    // combine channels
    set n_signatures,epochs,batch_size,l1_size,validation_perc,normalization from n_signatures.combine(epochs).combine(batch_size).combine(l1_size).combine(validation_perc).combine(normalization)

    output:
    path 'nFeatures_*__nSignatures_*__nEpochs_*__batchSize_*__l1Size_*__validationPerc_*__normalization_*__seed_*/*'

    """
    #!/usr/bin/env bash

    conda activate musexae

    python $PWD/regMUSE_XAE.py \
        --filename_real_data "${filename_real_data}" \
        --filename_permuted_data_training "${filename_permuted_data_training}" \
        --filename_permuted_data_validation "${filename_permuted_data_validation}" \
        --n_signatures ${n_signatures} \
        --epochs ${epochs} \
        --batch_size ${batch_size} \
        --l1_size ${l1_size} \
        --validation_perc ${validation_perc} \
        --normalization ${normalization}
    """
}
