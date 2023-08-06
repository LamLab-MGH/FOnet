import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Add, Activation
from tensorflow.keras.layers import TimeDistributed, Bidirectional, LSTM
from tensorflow.keras.layers import Flatten, Dropout
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.optimizers import Adam
import keras
import numpy as np


def fo_spike_detector(
    N_samples=256, N_channels=1, f_base=32, k_base=32, n_conv_layers=3,
    final_layer='sigmoid', dropout_rate=0.2):

    #  Create model
    input_shape = (N_samples, N_channels)
    X_input = Input(input_shape)
    x = Conv1D(
            filters=f_base, kernel_size=k_base,
            name="Conv1", padding="same")(X_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=4, strides=None)(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)

    for i in range(n_conv_layers-1):
        x = Conv1D(
                filters=f_base*(i+2), kernel_size=int(k_base/(2*(i+1))) + 1,
                kernel_initializer='glorot_uniform', name="Conv2" + str(i + 2),
                padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(pool_size=4, strides=None)(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x) 
    
    x = Flatten()(x)
    x = Dense(
            int(N_samples/(4*n_conv_layers)), name='FC',
            kernel_initializer='glorot_uniform')(x)
    x = Activation('relu')(x)
    
    x = Dense(1, name='LR', kernel_initializer='glorot_uniform')(x)
    x = Activation('sigmoid')(x)

    model = Model(inputs = X_input, outputs=x, name='fo_spike_detector')
    
    return model


class FOnet:
    
    def __init__(self, weights_path='./model_weights/FOnet.h5'):
        
        self.model = fo_spike_detector()
        if weights_path is not None:
            self.model.load_weights(weights_path)
        
    def predict(self, x, a=None, batch_size=1024):
        """
            x: EEG-FO input of dimensions (n_examples, n_channels, 256)
            a: Clean channels of dimensions (n_examples, n_channels, 1) 
                a = 1 where the channel is clean and = 0 when channel is noisy
        """
        
        n_examples = np.shape(x)[0]
        n_channels = np.shape(x)[1]
        yhat_channel = np.zeros((n_examples, n_channels))
        
        if x.ndim < 4:
            x = np.expand_dims(x, axis = 3)
        
        if a is None:
            a = np.ones((n_examples, n_channels))
        
        for i in range(n_channels):
            yhat_channel[:, i] = np.squeeze(
                self.model.predict(x[:, i, :],
                                   batch_size=batch_size,
                                   verbose = 1))
            
        yhat = np.zeros((n_examples, ))
        for i in range(n_channels):
            yhat = yhat + yhat_channel[:, i]*a[:, i]
        
        yhat = yhat/(np.sum(a, axis = 1))
                    
        return yhat