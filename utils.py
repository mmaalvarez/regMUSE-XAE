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
from lap import lapjv
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from tensorflow.keras import backend as K
from tensorflow.keras import activations,regularizers,constraints
from tensorflow.keras.constraints import NonNeg,Constraint
from tensorflow.keras.regularizers import OrthogonalRegularizer,L2
from tensorflow.keras.callbacks import Callback,ModelCheckpoint,EarlyStopping
from tensorflow.keras.layers import Input,Dense,BatchNormalization
from tensorflow.keras.models import Model,load_model
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


def load_dataset(filename_real_data, filename_permuted_data_training, filename_permuted_data_validation, epochs, validation_perc, normalization, inputDir, seed):

    ## load real data (no resamplings etc, all samples)
    real_data_df = pd.read_csv(f'./datasets/{filename_real_data}', sep='\t')

    # store sample names column, renamed as "Sample"
    real_data_sample_names = real_data_df.drop(real_data_df.columns[1:], axis=1).rename(columns={'Sample': 'Sample'})

    # store feature names column, renamed as "Feature"
    feature_names = pd.DataFrame(list(real_data_df.columns)[1:]).rename(columns={0: 'Feature'})
    
    n_samples_total = len(real_data_sample_names)

    real_data_df = real_data_df.set_index('Sample')
    
    if normalization == 'yes':
        # standardize coefficients
        real_data_df = standardize_coefficients(real_data_df, n_samples_total)

    ## load all permuted tables (up to n=epochs) and store in a dict (separately training and validation sets); each pair will be used for a different epoch

    # for TRAINING there are many permuted tables in ./datasets/...
    permuted_data_training_list_ALL = glob(f'./datasets/validation_perc_{validation_perc}/{filename_permuted_data_training}')
    
    # ...keep only up to n epochs
    permuted_data_training_list = []
    for permuted_df in permuted_data_training_list_ALL:
        if f'iter{epochs+1}' in permuted_df:
            break
        else:
            permuted_data_training_list.append(permuted_df)
            
    # make sure that there are as many training files as epochs
    n_training_files = len(permuted_data_training_list)
    try:
        assert n_training_files == epochs
    except AssertionError: 
        print(f'Error: There are {n_training_files} resampled training files but more epochs ({epochs}) specified! There should be 1 resampled training file per epoch. Exiting...\n')
        sys.exit(1)
            
    # load training tables
    training_dfs_dict = dict()
    
    for i,permuted_data_training in enumerate(permuted_data_training_list):
        
        # load resampled table
        training_df = pd.read_csv(permuted_data_training, sep='\t').set_index('Sample')
        
        if i == 0:
            # how many features and training samples (should be ~= total n_samples × 'training_fraction' at 1_parse_input.R)
            n_features_training, n_samples_training = len(training_df.columns), len(training_df.index)
        
        if normalization == 'yes':
            # standardize coefficients
            training_df = standardize_coefficients(training_df, n_samples_training)
        
        # store
        training_dfs_dict[i] = training_df


    ## for VALIDATION there is a single permuted table
    permuted_data_validation = glob(f'./datasets/validation_perc_{validation_perc}/{filename_permuted_data_validation}')

    try:
        assert len(permuted_data_validation) == 1
        permuted_data_validation = permuted_data_validation[0]
    except AssertionError: 
        print("Error: There is more than 1 (or none) validation tables!\n")
        sys.exit(1)

    # load validation table
    validation_df = pd.read_csv(permuted_data_validation, sep='\t').set_index('Sample')

    # how many features (should be same as for training) and validation samples (should be ~= total n_samples × (1 - 'training_fraction' at 1_parse_input.R))
    n_features_validation, n_samples_validation = len(validation_df.columns), len(validation_df.index)

    try:
        assert n_features_validation == n_features_training
    except AssertionError:
        print("ERROR: n_features are different between training and validation sets! Exiting...\n")
        sys.exit(1)

    if normalization == 'yes':
        # standardize coefficients
        validation_df = standardize_coefficients(validation_df, n_samples_validation)

    ## append (the only) validation_df to each entry (epoch) of training dictionary
    training_validation_dfs_dict = dict()

    n_permuted_table_pairs = list(training_dfs_dict.keys())

    for i in n_permuted_table_pairs:
        training_validation_dfs_dict[i] = [training_dfs_dict[i], validation_df]
          
    ## also choose a seed, if there is not any specific one requested
    if seed == None:
        seed = int(str(int(datetime.datetime.now().timestamp() * 1e10))[11:])

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

    
def MUSE_XAE(input_dim,l_1,n_encoder_layers=3,n_signatures,allow_negative_weights,beta=0.001,activation='softplus',reg='min_vol',refit=False):

    # hybrid autoencoder due to non linear encoder and linear decoder

    if reg=='min_vol': 
        regularizer=minimum_volume(beta=beta,dim=n_signatures)
    elif reg=='ortogonal': 
        regularizer=OrthogonalRegularizer(beta)
    elif reg=='L2' : 
        regularizer=L2(beta)

    ## when training the encoder using a fixed optimal signature in the decoder, in order to just get the sample exposures, use relu
    if refit==True:
        activation='relu'
        
    encoder_input=Input(shape=(input_dim,))
    
    latent_1 = Dense(l_1,activation=activation,name='encoder_layer_1')(encoder_input)
    latent_1 = BatchNormalization()(latent_1)

    if n_encoder_layers == 2:
        latent_1 = Dense(l_1/2,activation=activation,name='encoder_layer_2')(latent_1)
        latent_1 = BatchNormalization()(latent_1)

        if n_encoder_layers == 3:
            latent_1 = Dense(l_1/4,activation=activation,name='encoder_layer_3')(latent_1)
            latent_1 = BatchNormalization()(latent_1)

            if n_encoder_layers == 4:
                latent_1 = Dense(l_1/8,activation=activation,name='encoder_layer_4')(latent_1)
                latent_1 = BatchNormalization()(latent_1)

    if refit==True: 
        signatures = Dense(n_signatures, activation='relu', activity_regularizer=regularizers.l1(1e-3), name='latent_space')(latent_1)
    else: 
        signatures = Dense(n_signatures, activation=activation, name='latent_space')(latent_1)    

    if allow_negative_weights == 'yes':
        # negative weights allowed in decoder layer
        decoder = Dense(input_dim,activation='linear',name='decoder_layer',use_bias=False,kernel_regularizer=regularizer)(signatures)
    else:
        # apply non-negative constraint to decoder layer weights
        decoder = Dense(input_dim,activation='linear',name='decoder_layer',use_bias=False,kernel_constraint=NonNeg(),kernel_regularizer=regularizer)(signatures)
       
    ## encoder model: map the input layer to its encoded representation (i.e. to the latent space)
    encoder_model = Model(encoder_input,signatures)
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
        
        
def train_model(training_validation_dfs_dict, input_dim, feature_names, n_signatures, epochs, batch_size, l1_size, n_encoder_layers, loss, activation, allow_negative_weights, seed, output_folder_name, iter):
        
    autoencoder,encoder = MUSE_XAE(input_dim=input_dim, 
                                   l_1=l1_size,
                                   n_encoder_layers=n_encoder_layers,
                                   n_signatures=n_signatures,
                                   allow_negative_weights=allow_negative_weights,
                                   activation=activation)

    autoencoder.compile(optimizer=Adam(), loss=loss, metrics=['mse'])

    autoencoder.summary()

    # Add a reference to the training and validation data in the autoencoder model; initialized with the first pair (i.e. first epoch)
    autoencoder.train_data = training_validation_dfs_dict[0][0]
    autoencoder.val_data = training_validation_dfs_dict[0][1]

    ## by default Keras returns the final model from the final training step, regardless if the final model gives the minimum loss on validation data
    # so, if you want to use a model with the lowest loss on validation data, you need to use the ModelCheckpoint callback with parameter save_best_only. After training is done, you need to load saved model. It will be the best model in terms of loss on validation data
    # To do this, create also another instance, this time of the 'ModelCheckpoint' callback
    save_best_model = ModelCheckpoint(f'{output_folder_name}/best_model_weights_iter{iter+1}.h5',
                                      monitor='val_loss', 
                                      save_best_only=True, 
                                      save_weights_only=True)

    ## Create an instance of the 'DataSwitchCallback' callback with the data dictionary
    data_switcher = DataSwitchCallback(training_validation_dfs_dict)
  
    ## Run the training for the 'autoencoder' model
    
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
    
    # load the weights of the best model (i.e. epoch with minimum validation loss)
    autoencoder.load_weights(f'{output_folder_name}/best_model_weights_iter{iter+1}.h5')
    
    # now evaluate loss thereof
    best_model_training_loss = autoencoder.evaluate(autoencoder.train_data, autoencoder.train_data)[0]
    best_model_validation_loss = autoencoder.evaluate(autoencoder.val_data, autoencoder.val_data)[0]
    min_val_loss_epoch = history['val_loss'].index(min(history['val_loss'])) + 1
    
    best_model_losses_epoch_i = pd.DataFrame({'output_folder_name': [output_folder_name],
                                              'iter': [iter+1],
                                              'seed': [seed],
                                              'best_model_training_loss': [best_model_training_loss],
                                              'best_model_validation_loss': [best_model_validation_loss],
                                              'min_val_loss_epoch': [min_val_loss_epoch]})
    
    print(f"At iter {iter+1}, the model from epoch {min_val_loss_epoch} has the lowest validation loss, and therefore it will be used to obtain the final signatures:")
    print(f'At iter {iter+1}, the best model\'s training loss: {best_model_training_loss.round(2)}')
    print(f'At iter {iter+1}, the best model\'s validation loss: {best_model_validation_loss.round(2)}\n')

    ## load signature weights (the weights of the decoder layer -i.e. the last before the output layer)
    signature_weights = pd.DataFrame(autoencoder.layers[-1].get_weights()[0].T).add_prefix('ae')
    # append feature names column
    signature_weights = pd.concat([feature_names, signature_weights], axis=1)
    
    return autoencoder,loss_plot,best_model_losses_epoch_i,signature_weights


class KMeans_with_matching:
    def __init__(self, X, n_clusters, max_iter, random=False):
        
        self.X = X
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n = X.shape[0]

        if self.n_clusters > 1:
            model = KMeans(n_clusters=n_clusters, init='random').fit(self.X)
            self.C = np.asarray(model.cluster_centers_)
        else:
            self.C = self.X[np.random.choice(self.n, size=1), :]

        self.C_prev = np.copy(self.C)
        self.C_history = [np.copy(self.C)]    

    def fit_predict(self):
        
        if self.n_clusters == 1:
            cost = np.zeros((self.n, self.n))
            for j in range(self.n):
                cost[0,j] = 1 - cosine_similarity(self.X[j,:].reshape(1,-1), self.C[0,:].reshape(1,-1))
            for i in range(1, self.n):
                cost[i,:] = cost[0,:]
            _, _, colsol = lapjv(cost)
            self.partition = np.mod(colsol, self.n_clusters)
            self.C[0,:] = self.X[np.where(self.partition == 0)].mean(axis=0)
            return pd.DataFrame(self.C).T, self.partition

        for k_iter in range(self.max_iter):
            cost = np.zeros((self.n, self.n))
            for i in range(self.n_clusters): 
                for j in range(self.n):
                    cost[i,j]=1-cosine_similarity(self.X[j,:].reshape(1,-1),self.C[i,:].reshape(1,-1))
            for i in range(self.n_clusters, self.n):
                cost[i,:] = cost[np.mod(i,self.n_clusters),:]
            _,_,colsol = lapjv(cost)
            self.partition = np.mod(colsol, self.n_clusters)
            for i in range(self.n_clusters):
                self.C[i,:] = self.X[np.where(self.partition == i)].mean(axis=0)
            self.C_history.append(np.copy(self.C))
            if np.array_equal(self.C, self.C_prev):
                break
            else:
                self.C_prev = np.copy(self.C)
            
        return pd.DataFrame(self.C).T,self.partition

    
def get_consensus_signatures(n_signatures, extractions, feature_names_col):
    
    all_extractions_dfs = pd.concat([pd.DataFrame(df) for df in extractions], axis=1).T
    X_all = np.asarray(all_extractions_dfs)
    clustering_model = KMeans_with_matching(X=X_all, n_clusters=n_signatures, max_iter=100)
    consensus_sig, cluster_labels = clustering_model.fit_predict()
    consensus_sig = pd.concat([feature_names_col, consensus_sig], axis=1)
    if n_signatures==1:
        means_lst=[1]
        min_sil=1
        mean_sil=1
    else:
        sample_silhouette_values = sklearn.metrics.silhouette_samples(all_extractions_dfs, cluster_labels, metric='cosine')
        means_lst = []

        for label in range(len(set(cluster_labels))):
            means_lst.append(sample_silhouette_values[np.array(cluster_labels) == label].mean())
        min_sil=np.min(means_lst)
        mean_sil=np.mean(means_lst)
    
    return min_sil, mean_sil, consensus_sig, means_lst


def encoder_prediction(real_data_df, real_data_sample_names, S, input_dim, l1_size, n_encoder_layers, n_signatures, batch_size, allow_negative_weights, seed, output_folder_name):

    real_data = np.array(real_data_df)
    
    autoencoder,encoder = MUSE_XAE(input_dim=input_dim, 
                                   l_1=l1_size,
                                   n_encoder_layers=n_encoder_layers,
                                   n_signatures=n_signatures,
                                   allow_negative_weights=allow_negative_weights,
                                   refit=True)
    
    # fix the optimal signatures in the decoder, make them untrainable
    S = S.drop('Feature', axis=1)
    autoencoder.layers[-1].set_weights([np.array(S.T)])
    autoencoder.layers[-1].trainable=False 

    early_stopping = EarlyStopping(monitor='loss',patience=100) # here we do use the EarlyStopping
    save_best_model = ModelCheckpoint(f'{output_folder_name}/best_model_refit.h5', 
                                      monitor='loss', 
                                      save_best_only=True, 
                                      verbose=False)
    autoencoder.compile(optimizer = Adam(learning_rate=0.001), # specify learning rate
                        loss = 'mse',
                        metrics = ['mse','kullback_leibler_divergence'])

    ## Run the training for the 'autoencoder' model (which simultaneously trains the 'encoder' model)
    print(f'Obtaining sample exposures for the consensus {n_signatures} mutational signatures...\n')
    tf.random.set_seed(seed)
    history = autoencoder.fit(real_data, real_data,
                              epochs=10000, # hardcoded
                              batch_size=batch_size,
                              verbose=False,
                              validation_data=(real_data, real_data),
                              callbacks=[early_stopping,save_best_model])
    
    # load the weights of the best model
    best_autoencoder = load_model(f'{output_folder_name}/best_model_refit.h5', custom_objects={"minimum_volume":minimum_volume(beta=0.001, dim=int(len(S.T)))})
    best_encoder = Model(inputs=best_autoencoder.input, outputs=best_autoencoder.get_layer('latent_space').output)    
    
    # get sample exposures
    encoded_real_df = best_encoder.predict(real_data)
    
    # rename columns ('signatures'), and convert to pandas df
    encoded_real_df = pd.DataFrame(encoded_real_df, columns = range(0, n_signatures)).add_prefix('ae')

    # append sample names column
    encoded_real_df = pd.concat([real_data_sample_names, encoded_real_df], axis=1)
    
    return best_encoder, encoded_real_df
