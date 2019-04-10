"""
This module contains functions that create different autoencoders
"""

import keras
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras import backend as K
from keras import metrics, optimizers
from keras.callbacks import Callback, TensorBoard
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Dense, Lambda, Layer, Activation, Concatenate

def make_variational_layer(input_layer, size, batch_normalize_intermediaries,
    relu_intermediaries, sampling_function):
        # variational layer for hidden dim
        z_mean_component = Dense(size, kernel_initializer='glorot_uniform')
        z_mean_dense_linear = z_mean_component(input_layer)

        if batch_normalize_intermediaries:
            z_mean_dense_batchnorm = BatchNormalization()(z_mean_dense_linear)
        else: z_mean_dense_batchnorm = z_mean_dense_linear

        if relu_intermediaries:
            z_mean_encoded = Activation('relu')(z_mean_dense_batchnorm)
        else:
            z_mean_encoded = z_mean_dense_batchnorm

        z_log_var_dense_linear = Dense(size, kernel_initializer='glorot_uniform')(input_layer)

        if batch_normalize_intermediaries:
            z_log_var_dense_batchnorm = BatchNormalization()(z_log_var_dense_linear)
        else:
            z_log_var_dense_batchnorm = z_log_var_dense_linear

        if relu_intermediaries:
            z_log_var_encoded = Activation('relu')(z_log_var_dense_batchnorm)
        else:
            z_log_var_encoded = z_log_var_dense_batchnorm

        # return the encoded and randomly sampled z vector (hidden layer 1)
        z = Lambda(sampling_function, output_shape=(size, ))([z_mean_encoded, z_log_var_encoded])
        return z, z_mean_component

def stacked_vae(x_train, x_val, hidden_dims=[300], latent_dim=100, initial_beta_val=0, learning_rate=.0005,
    epsilon_std=1., kappa=1., epochs=50, batch_size=50, batch_normalize_inputs=True,
    batch_normalize_intermediaries=True, batch_normalize_embedding=True,
    relu_intermediaries=True, relu_embedding=True, max_beta_val=1):
    """
    This is a deep, or stacked, vae.
    `hidden_dims` denotes the size of each successive hidden layer,
    until `latend_dim` which is the middle layer.
    """
    # Function for reparameterization trick to make model differentiable
    def sampling(args):
        
        import tensorflow as tf
        # Function with args required for Keras Lambda function
        z_mean, z_log_var = args

        # Draw epsilon of the same shape from a standard normal distribution
        epsilon = K.random_normal(shape=tf.shape(z_mean), mean=0.,
                                  stddev=epsilon_std)
        
        # The latent vector is non-deterministic and differentiable
        # in respect to z_mean and z_log_var
        z = z_mean + K.exp(z_log_var / 2) * epsilon
        return z


    class CustomVariationalLayer(Layer):
        """
        Define a custom layer that learns and performs the training
        This function is borrowed from:
        https://github.com/fchollet/keras/blob/master/examples/variational_autoencoder.py
        """
        def __init__(self, **kwargs):
            # https://keras.io/layers/writing-your-own-keras-layers/
            self.is_placeholder = True
            super(CustomVariationalLayer, self).__init__(**kwargs)

        def vae_loss(self, x_input, x_decoded):
            reconstruction_loss = original_dim * metrics.binary_crossentropy(x_input, x_decoded)
            kl_loss = - 0.5 * K.sum(1 + l_log_var_dense_linear - K.square(l_mean_dense_linear) -
                                    K.exp(l_log_var_dense_linear), axis=-1)
            return K.mean(reconstruction_loss + (K.get_value(beta) * kl_loss))

        def call(self, inputs):
            x = inputs[0]
            x_decoded = inputs[1]
            loss = self.vae_loss(x, x_decoded)
            self.add_loss(loss, inputs=inputs)
            # We won't actually use the output.
            return x

    class LadderCallback(Callback):
        """
        This class implements ladder autoecoders:
            https://arxiv.org/abs/1602.02282

        A callback on each epoch end, increments beta by kappa
        """
        def __init__(self, beta, kappa, max_val=1):
            self.beta = beta
            self.kappa = kappa
            self.max_val = max_val
        def on_epoch_end(self, *args, **kwargs):
            if K.get_value(self.beta) <= self.max_val:
                K.set_value(self.beta, K.get_value(self.beta) + self.kappa)


    # Init beta value
    beta = K.variable(initial_beta_val)
    
    # Input place holder for RNAseq data with specific input size
    original_dim = x_train.shape[1]

    # Input place holder for RNAseq data with specific input size
    rnaseq_input = Input(shape=(original_dim, ))

    if batch_normalize_inputs:
        batchnorm_input = BatchNormalization()(rnaseq_input)
    else:
        batchnorm_input = rnaseq_input

    prev = batchnorm_input
    encoder_target = batchnorm_input
    if hidden_dims:
        for hidden_dim in hidden_dims:
            z, z_mean_component = make_variational_layer(prev, hidden_dim,
                batch_normalize_intermediaries, relu_intermediaries,
                sampling)
            prev = z
            # the encoder part to have a path that doesn't do sampling or ReLU'ing
            encoder_target = z_mean_component(encoder_target)
    else:
        z = prev

    # variational layer for latent dim
    l_mean_component = Dense(latent_dim, kernel_initializer='glorot_uniform')
    l_mean_dense_linear = l_mean_component(z)

    if batch_normalize_embedding:
        l_mean_dense_batchnorm = BatchNormalization()(l_mean_dense_linear)
    else:
        l_mean_dense_batchnorm = l_mean_dense_linear

    if relu_embedding:
        l_mean_encoded = Activation('relu')(l_mean_dense_batchnorm)
    else:
        l_mean_encoded = l_mean_dense_batchnorm

    l_log_var_dense_linear = Dense(latent_dim, kernel_initializer='glorot_uniform')(z)

    if batch_normalize_embedding:
        l_log_var_dense_batchnorm = BatchNormalization()(l_log_var_dense_linear)
    else:
        l_log_var_dense_batchnorm = l_log_var_dense_linear

    if relu_embedding:
        l_log_var_encoded = Activation('relu')(l_log_var_dense_batchnorm)
    else:
        l_log_var_encoded = l_log_var_dense_batchnorm

    l = Lambda(sampling, output_shape=(latent_dim,))([l_mean_encoded, l_log_var_encoded])

    # the encoder part's l to come from the path that only considers mean
    encoder_target = l_mean_component(encoder_target)
    if batch_normalize_embedding:
        encoder_target = BatchNormalization()(encoder_target)
    if relu_embedding:
        encoder_target = Activation('relu')(encoder_target)

    # decoder latent->hidden
    prev = l
    if hidden_dims:
        for hidden_dim in reversed(hidden_dims):
            h = Dense(hidden_dim, kernel_initializer='glorot_uniform', activation='relu')(prev)
            prev = h
    else:
        h = Dense(latent_dim, kernel_initializer='glorot_uniform', activation='relu')(prev)
    reconstruction = Dense(original_dim, kernel_initializer='glorot_uniform', activation='sigmoid')(h)

    adam = optimizers.Adam(lr=learning_rate)
    vae_layer = CustomVariationalLayer()([rnaseq_input, reconstruction])
    vae = Model(rnaseq_input, vae_layer)
    vae.compile(optimizer=adam, loss=None, loss_weights=[beta])

    # Train the model
    K.get_session().run(tf.global_variables_initializer())
    hist = vae.fit(np.array(x_train),
                   shuffle=True,
                   epochs=epochs,
                   verbose=0,
                   batch_size=batch_size,
                   validation_data=(np.array(x_val), None),
                   callbacks=[LadderCallback(beta, kappa, max_beta_val)])
    
    # non-sampling encoder
    encoder = Model(rnaseq_input, encoder_target)

    # sampling encoder
    sampling_encoder = Model(rnaseq_input, l)

    # Also, create a decoder model
    encoded_input = Input(shape=(latent_dim,))
    prev = encoded_input
    if hidden_dims:
        for i in reversed(range(len(hidden_dims)+1)):
            prev = vae.layers[-(i+2)](prev)
    decoder = Model(encoded_input, prev)

    return hist, vae, encoder, sampling_encoder, decoder


def deep_vae(x_train, x_val, hidden_dims=[300], latent_dim=100, initial_beta_val=0, learning_rate=.0005,
    epsilon_std=1., kappa=1., epochs=50, batch_size=50,
    batch_normalize_inputs=True, batch_normalize_embedding=False,
    relu_embedding=False, max_beta_val=1):
    """
    This is a deep, not stacked, vae.
    `hidden_dims` denotes the size of each successive hidden layer,
    until `latend_dim` which is the middle layer, which will be mean/variance of a normal distribution.
    """
    # Function for reparameterization trick to make model differentiable
    def sampling(args):
        import tensorflow as tf
        # Function with args required for Keras Lambda function
        z_mean, z_log_var = args

        # Draw epsilon of the same shape from a standard normal distribution
        epsilon = K.random_normal(shape=tf.shape(z_mean), mean=0.,
                                  stddev=epsilon_std)
        # The latent vector is non-deterministic and differentiable
        # in respect to z_mean and z_log_var
        z = z_mean + K.exp(z_log_var / 2) * epsilon
        return z


    class CustomVariationalLayer(Layer):
        """
        Define a custom layer that learns and performs the training
        This function is borrowed from:
        https://github.com/fchollet/keras/blob/master/examples/variational_autoencoder.py
        """
        def __init__(self, **kwargs):
            # https://keras.io/layers/writing-your-own-keras-layers/
            self.is_placeholder = True
            super(CustomVariationalLayer, self).__init__(**kwargs)

        def vae_loss(self, x_input, x_decoded):
            reconstruction_loss = original_dim * metrics.binary_crossentropy(x_input, x_decoded)
            kl_loss = - 0.5 * K.sum(1 + l_log_var_encoded - K.square(l_mean_encoded) -
                                    K.exp(l_log_var_encoded), axis=-1)
            return K.mean(reconstruction_loss + (K.get_value(beta) * kl_loss))

        def call(self, inputs):
            x = inputs[0]
            x_decoded = inputs[1]
            loss = self.vae_loss(x, x_decoded)
            self.add_loss(loss, inputs=inputs)
            # We won't actually use the output.
            return x

    class LadderCallback(Callback):
        """
        This class implements ladder autoecoders:
            https://arxiv.org/abs/1602.02282

        A callback on each epoch end, increments beta by kappa
        """
        def __init__(self, beta, kappa, max_val=1):
            self.beta = beta
            self.kappa = kappa
            self.max_val = max_val
        def on_epoch_end(self, *args, **kwargs):
            if K.get_value(self.beta) <= self.max_val:
                K.set_value(self.beta, K.get_value(self.beta) + self.kappa)


    # Init beta value
    beta = K.variable(initial_beta_val)

    # Input place holder for RNAseq data with specific input size
    original_dim = x_train.shape[1]

    # Input place holder for RNAseq data with specific input size
    rnaseq_input = Input(shape=(original_dim, ))

    if batch_normalize_inputs:
        batchnorm_input = BatchNormalization()(rnaseq_input)
    else:
        batchnorm_input = rnaseq_input

    prev = batchnorm_input
    if hidden_dims:
        for hidden_dim in hidden_dims:
            z = Dense(hidden_dim, activation='relu')(prev)
            prev = z
    else:
        z = prev

    # variational layer for latent dim
    l_mean_component = Dense(latent_dim, kernel_initializer='glorot_uniform')
    l_mean_dense_linear = l_mean_component(z)

    if batch_normalize_embedding:
        l_mean_dense_batchnorm = BatchNormalization()(l_mean_dense_linear)
    else:
        l_mean_dense_batchnorm = l_mean_dense_linear

    if relu_embedding:
        l_mean_encoded = Activation('relu')(l_mean_dense_batchnorm)
    else:
        l_mean_encoded = l_mean_dense_batchnorm

    l_log_var_dense_linear = Dense(latent_dim, kernel_initializer='glorot_uniform')(z)

    if batch_normalize_embedding:
        l_log_var_dense_batchnorm = BatchNormalization()(l_log_var_dense_linear)
    else:
        l_log_var_dense_batchnorm = l_log_var_dense_linear

    if relu_embedding:
        l_log_var_encoded = Activation('relu')(l_log_var_dense_batchnorm)
    else:
        l_log_var_encoded = l_log_var_dense_batchnorm

    l = Lambda(sampling, output_shape=(latent_dim,))([l_mean_encoded, l_log_var_encoded])

    # the encoder part's l to come from the path that only considers mean
    encoder_target = l_mean_component(z)
    if batch_normalize_embedding:
        encoder_target = BatchNormalization()(encoder_target)
    if relu_embedding:
        encoder_target = Activation('relu')(encoder_target)

    # decoder latent->hidden
    if hidden_dims:
        prev = l
        for hidden_dim in reversed(hidden_dims):
            h = Dense(hidden_dim, kernel_initializer='glorot_uniform', activation='relu')(prev)
            prev = h
    else:
        h = l

    reconstruction = Dense(original_dim, kernel_initializer='glorot_uniform', activation='sigmoid')(h)

    adam = optimizers.Adam(lr=learning_rate)
    vae_layer = CustomVariationalLayer()([rnaseq_input, reconstruction])
    vae = Model(rnaseq_input, vae_layer)
    vae.compile(optimizer=adam, loss=None, loss_weights=[beta])

    # Train the model
    K.get_session().run(tf.global_variables_initializer())
    hist = vae.fit(np.array(x_train),
                   shuffle=True,
                   epochs=epochs,
                   verbose=0,
                   batch_size=batch_size,
                   validation_data=(np.array(x_val), None),
                   callbacks=[LadderCallback(beta, kappa, max_beta_val)])

    # non-sampling encoder
    encoder = Model(rnaseq_input, encoder_target)

    # sampling encoder
    sampling_encoder = Model(rnaseq_input, l)

    # Also, create a decoder model
    encoded_input = Input(shape=(latent_dim,))
    prev = encoded_input
    if hidden_dims:
        for i in reversed(range(len(hidden_dims)+1)):
            prev = vae.layers[-(i+2)](prev)
    decoder = Model(encoded_input, prev)

    return hist, vae, encoder, sampling_encoder, decoder
