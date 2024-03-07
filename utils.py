import pandas as pd
import numpy as np
import tensorflow as tf
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import sys
from glob import glob
import datetime
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.regularizers import OrthogonalRegularizer,L2
from tensorflow.keras.callbacks import Callback,ModelCheckpoint
from tensorflow.keras.layers import Input,Dense,BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers.legacy import Adam
tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


def standardize_coefficients(data, n_samples):
    
    # transpose
    data = data.T
    
    mean_coeff = pd.concat([data.mean(axis=1)]*n_samples,axis=1)
    mean_coeff.columns=data.columns
    sd_coeff = pd.concat([data.std(axis=1)]*n_samples,axis=1)
    sd_coeff.columns=data.columns
    standardized_coefficients = (data-mean_coeff) / sd_coeff
    
    # transpose
    standardized_coefficients = standardized_coefficients.T

    return np.array(standardized_coefficients, dtype='float64')


def load_dataset(filename_real_data, filename_dataset_permuted_training, filename_dataset_permuted_validation, epochs, validation_perc, normalization, seed):

    ## load real data (no resamplings etc, all samples)
    real_data_df = pd.read_csv(f'./datasets/{filename_real_data}', sep='\t')

    # store sample names column, renamed as "Sample"
    real_data_sample_names = real_data_df.drop(real_data_df.columns[1:], axis=1).rename(columns={'Sample': 'Sample'})

    # store feature names column, renamed as "Feature"
    feature_names = pd.DataFrame(list(real_data_df.columns)[1:]).rename(columns={0: 'Feature'})
    
    n_samples_total = len(real_data_sample_names)

    real_data_df = real_data_df.set_index('Sample')
    
    if normalization == True:
        # standardize coefficients
        real_data_df = standardize_coefficients(real_data_df, n_samples_total)

    ## load all permuted tables (up to n=epochs) and store in a dict (separately training and validation sets); each pair will be used for a different epoch

    # for TRAINING there are many permuted tables in ./datasets/...
    dataset_permuted_training_list_ALL = glob(f'./datasets/validation_perc_{validation_perc}/{filename_dataset_permuted_training}')
    
    # ...keep only up to n epochs
    dataset_permuted_training_list = []
    for permuted_df in dataset_permuted_training_list_ALL:
        if f'iter{epochs+1}' in permuted_df:
            break
        else:
            dataset_permuted_training_list.append(permuted_df)
            
    # make sure that there are as many training files as epochs
    n_training_files = len(dataset_permuted_training_list)
    try:
        assert n_training_files == epochs
    except AssertionError: 
        print(f'Error: There are {n_training_files} resampled training files but more epochs ({epochs}) specified! There should be 1 resampled training file per epoch. Exiting...\n')
        sys.exit(1)
            
    # load training tables
    training_dfs_dict = dict()
    
    for i,dataset_permuted_training in enumerate(dataset_permuted_training_list):
        
        # load resampled table
        training_df = pd.read_csv(dataset_permuted_training, sep='\t').set_index('Sample')
        
        if i == 0:
            # how many features and training samples (should be ~= total n_samples × 'training_fraction' at 1_parse_input.R)
            n_features_training, n_samples_training = len(training_df.columns), len(training_df.index)
        
        if normalization == True:
            # standardize coefficients
            training_df = standardize_coefficients(training_df, n_samples_training)
        
        # store
        training_dfs_dict[i] = training_df


    ## for VALIDATION there is a single permuted table
    dataset_permuted_validation = glob(f'./datasets/validation_perc_{validation_perc}/{filename_dataset_permuted_validation}')

    try:
        assert len(dataset_permuted_validation) == 1
        dataset_permuted_validation = dataset_permuted_validation[0]
    except AssertionError: 
        print("Error: There is more than 1 (or none) validation tables!\n")
        sys.exit(1)

    # load validation table
    validation_df = pd.read_csv(dataset_permuted_validation, sep='\t').set_index('Sample')

    # how many features (should be same as for training) and validation samples (should be ~= total n_samples × (1 - 'training_fraction' at 1_parse_input.R))
    n_features_validation, n_samples_validation = len(validation_df.columns), len(validation_df.index)

    try:
        assert n_features_validation == n_features_training
    except AssertionError:
        print("ERROR: n_features are different between training and validation sets! Exiting...\n")
        sys.exit(1)

    if normalization == True:
        # standardize coefficients
        validation_df = standardize_coefficients(validation_df, n_samples_validation)

    ## append (the only) validation_df to each entry (epoch) of training dictionary
    training_validation_dfs_dict = dict()

    n_permuted_table_pairs = list(training_dfs_dict.keys())

    for i in n_permuted_table_pairs:
        training_validation_dfs_dict[i] = [training_dfs_dict[i], validation_df]
          
    ## also choose a seed, if there is not any specific one requested
    if seed == None:
        seed = int(datetime.datetime.now().timestamp())

    return real_data_df, real_data_sample_names, feature_names, n_features_training, training_validation_dfs_dict, seed


class minimum_volume(Constraint):
    def __init__(self, dim=15, beta=0.001):
        self.beta = beta
        self.dim = dim
    
    def __call__(self, weights):
        w_matrix = K.dot(weights, K.transpose(weights))
        det = tf.linalg.det(w_matrix + K.eye(self.dim))
        log_det = K.log(det)/K.log(10.0)
        return self.beta * log_det

    def get_config(self):
        return {'dim': self.dim,
                'beta': float(self.beta)}

    
def MUSE_XAE(input_dim,l_1,n_signatures,beta=0.001,activation='softplus',reg='min_vol'):

    # hybrid autoencoder due to non linear encoder and linear decoder

    if reg=='min_vol': 
        regularizer=minimum_volume(beta=beta,dim=n_signatures)
    elif reg=='ortogonal': 
        regularizer=OrthogonalRegularizer(beta)
    elif reg=='L2' : 
        regularizer=L2(beta)

    encoder_input=Input(shape=(input_dim,))
    
    latent_1 = Dense(l_1,activation=activation,name='encoder_layer_1')(encoder_input)
    latent_1 = BatchNormalization()(latent_1)
    latent_1 = Dense(l_1/2,activation=activation,name='encoder_layer_2')(latent_1)
    latent_1 = BatchNormalization()(latent_1)
    latent_1 = Dense(l_1/4,activation=activation,name='encoder_layer_3')(latent_1)
    latent_1 = BatchNormalization()(latent_1)

    n_signatures = Dense(n_signatures,activation='softplus',name='latent_space')(latent_1)

    decoder = Dense(input_dim,activation='linear',name='decoder_layer',use_bias=False,kernel_regularizer=regularizer)(n_signatures)
    
    ## encoder model: map the input layer to its encoded representation (i.e. to the latent space)
    # this will be used after training
    encoder_model = Model(encoder_input,n_signatures)
    
    ## autoencoder model: map the input layer to its reconstruction (i.e. to the output layer)
    hybrid_dae = Model(encoder_input,decoder)
    
    return hybrid_dae,encoder_model


class DataSwitchCallback(Callback):
    
    ## custom callback to change the permuted training and validation data pair to use in each epoch
    
    def __init__(self, data_dict):
        super(DataSwitchCallback, self).__init__()
        # initialize variables at first epoch only
        self.data_dict = data_dict # store training_validation_dfs_dict
        self.epoch_count = 0
        
    # when starting an epoch...
    def on_epoch_begin(self, epoch, logs=None):
        # ...find out which permuted data is to be used in this epoch
        data_index = self.epoch_count % len(self.data_dict)
        # ...select that permuted data (train and validation)
        self.model.train_data, self.model.val_data = self.data_dict[data_index]
        # ...print log message
        print(f"Using data pair #{data_index+1} for epoch {self.epoch_count + 1}")
        # ...and add another epoch to the count
        self.epoch_count += 1
        
        
def train_model(training_validation_dfs_dict, input_dim, feature_names, n_signatures, epochs, batch_size, l1_size, loss, activation, seed, output_folder_name):
        
    autoencoder,encoder = MUSE_XAE(input_dim=input_dim, 
                                   l_1=l1_size,
                                   n_signatures=n_signatures, 
                                   activation=activation)

    autoencoder.compile(optimizer=Adam(), loss=loss, metrics=['mse'])

    autoencoder.summary()

    # Add a reference to the training and validation data in the autoencoder model; initialized with the first pair (i.e. first epoch)
    autoencoder.train_data = training_validation_dfs_dict[0][0]
    autoencoder.val_data = training_validation_dfs_dict[0][1]

    ## by default Keras returns the final model from the final training step, regardless if the final model gives the minimum loss on validation data
    # so, if you want to use a model with the lowest loss on validation data, you need to use the ModelCheckpoint callback with parameter save_best_only. After training is done, you need to load saved model. It will be the best model in terms of loss on validation data
    # To do this, create also another instance, this time of the 'ModelCheckpoint' callback
    save_best_model = ModelCheckpoint(f'{output_folder_name}/best_model_weights',
                                      monitor='val_loss', 
                                      save_best_only=True, 
                                      save_weights_only=True)

    ## Create an instance of the 'DataSwitchCallback' callback with the data dictionary
    data_switcher = DataSwitchCallback(training_validation_dfs_dict)
  
    ## Run the training for the 'autoencoder' model
    # (which simultaneously trains the 'encoder' model)
    
    history = {'loss': [], 'val_loss': []}
    
    print(f'Running {epochs} epochs with {n_signatures} mutational signatures...\n')

    tf.random.set_seed(seed)
    
    for epoch in range(epochs):
        # use a different train_data + val_data pair at each iteration (i.e. epoch)
        epoch_it = autoencoder.fit(autoencoder.train_data, autoencoder.train_data, 
                                   shuffle = True,
                                   epochs=1, # run only once each epoch
                                   batch_size=batch_size, 
                                   verbose=False,
                                   validation_data=(autoencoder.val_data, autoencoder.val_data),
                                   callbacks=[data_switcher, save_best_model]) # switch datasets at each epoch change
        # keep track of losses
        history['loss'].append(epoch_it.history['loss'][0])
        history['val_loss'].append(epoch_it.history['val_loss'][0])

    ## evaluate final loss
    training_loss = autoencoder.evaluate(autoencoder.train_data, autoencoder.train_data)[0]
    validation_loss = autoencoder.evaluate(autoencoder.val_data, autoencoder.val_data)[0]
    print(f'\nFinal training loss: {training_loss.round(2)}')
    print(f'Final validation loss: {validation_loss.round(2)}\n')

    # visualize training performance
    history_df = pd.DataFrame(history)
    ax = history_df.plot()
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Autoencoder loss')
    loss_plot = ax.get_figure()

    ## evaluate best model's loss
    
    # load the weights of the best model (i.e. epoch with minimum validation loss); this automatically also updates the weights of the shared layers in the 'encoder' Model
    autoencoder.load_weights(f'{output_folder_name}/best_model_weights')
    
    # now evaluate loss thereof
    best_model_training_loss = autoencoder.evaluate(autoencoder.train_data, autoencoder.train_data)[0]
    best_model_validation_loss = autoencoder.evaluate(autoencoder.val_data, autoencoder.val_data)[0]
    min_val_loss_epoch = history['val_loss'].index(min(history['val_loss'])) + 1
    
    print(f"The model from epoch {min_val_loss_epoch} has the lowest validation loss, and therefore it will be used to obtain the final signatures:")
    print(f'Best model\'s training loss: {best_model_training_loss.round(2)}')
    print(f'Best model\'s validation loss: {best_model_validation_loss.round(2)}\n')

    ## load signature weights (the weights of the decoder layer -i.e. the last before the output layer)
    signature_weights = pd.DataFrame(autoencoder.layers[-1].get_weights()[0].T).add_prefix('ae')
    # append feature names column
    signature_weights = pd.concat([feature_names, signature_weights], axis=1)
    
    return autoencoder,encoder,loss_plot,signature_weights


def encoder_prediction(encoder, real_df, real_data_sample_names, n_signatures):

    encoded_real_df = encoder.predict_on_batch(real_df)

    # rename columns ('signatures'), and convert to pandas df
    encoded_real_df = pd.DataFrame(encoded_real_df, columns = range(0, n_signatures)).add_prefix('ae')

    # check nodes activity to ensure that the model is learning a distribution of feature activations, and not zeroing out features
    sum_node_activity = encoded_real_df.sum(axis=0).sort_values(ascending=False)
    sum_node_activity_mean = sum_node_activity.mean()
    print(sum_node_activity)

    # append sample names column
    encoded_real_df = pd.concat([real_data_sample_names, encoded_real_df], axis=1)

    return encoded_real_df
