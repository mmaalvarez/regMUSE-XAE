#!/usr/bin/env nextflow

// variables (channels)
n_signatures = Channel.from( [ '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35' ] )
n_iters = Channel.from( [ '10' ] )
epochs = Channel.from( [ '1000' , '2000' ] ) //, '3000', '4000', '5000', '6000' ] )
batch_size = Channel.from( [ '64' ] )
l1_size = Channel.from( [ '128' ] )
n_encoder_layers = Channel.from( [ '2', '3', '4' ] )
validation_perc = Channel.from( [ '10', '20' ] ) //, '30' ] )
normalization = Channel.from( [ 'no' ] ) //,'yes' 
allow_negative_weights = Channel.from( [ 'yes' ] ) //, 'no' ] ) 

process run_autoencoder {

    publishDir "$PWD/res/", mode: 'copy', pattern: 'nFeatures_*__nSignatures_*__nIters_*__nEpochs_*__batchSize_*__l1Size_*__validationPerc_*__normalization_*__allow_negative_weights_*__seed_*/*'

    time = { (params.minutes + 15*(task.attempt-1)).min }
    memory = { (params.memGB + 5*(task.attempt-1)).GB }

    input:
    // fixed paths
    path inputDir from params.inputDir
    val filename_real_data from params.filename_real_data
    val filename_permuted_data_training from params.filename_permuted_data_training
    val filename_permuted_data_validation from params.filename_permuted_data_validation
    // combine channels
    set n_signatures,epochs,n_iters,batch_size,l1_size,n_encoder_layers,validation_perc,normalization,allow_negative_weights from n_signatures.combine(epochs).combine(n_iters).combine(batch_size).combine(l1_size).combine(n_encoder_layers).combine(validation_perc).combine(normalization).combine(allow_negative_weights)

    output:
    path 'nFeatures_*__nSignatures_*__nIters_*__nEpochs_*__batchSize_*__l1Size_*__validationPerc_*__normalization_*__allow_negative_weights_*__seed_*/*'
    file 'best_model_losses_epoch_alliters.tsv' into best_model_losses_epoch_alliters

    """
    #!/usr/bin/env bash

    conda activate musexae

    python $PWD/regMUSE_XAE.py --filename_real_data "${filename_real_data}" \
                               --filename_permuted_data_training "${filename_permuted_data_training}" \
                               --filename_permuted_data_validation "${filename_permuted_data_validation}" \
                               --n_signatures ${n_signatures} \
                               --iters ${n_iters} \
                               --epochs ${epochs} \
                               --batch_size ${batch_size} \
                               --l1_size ${l1_size} \
                               --n_encoder_layers ${n_encoder_layers} \
                               --validation_perc ${validation_perc} \
                               --normalization ${normalization} \
                               --allow_negative_weights ${allow_negative_weights} \
                               --inputDir ${inputDir}
    """
}

best_model_losses_epoch_alliters
    .collectFile(name: 'res/all_best_model_losses_epoch_alliters.tsv', keepHeader: true)
    .println { "Finished! Results saved in res/" }
