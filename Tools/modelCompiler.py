import tensorflow as tf
import numpy as np
from tensorflow.keras import regularizers

def compileModel4():
    
    model = tf.keras.models.Sequential() #we are using a feed forward model.
    model.add(tf.keras.layers.Dense(24, activation= 'relu')) #this is the first dense layer using the recified linear activation function

    model.add(tf.keras.layers.Dense(128, activation= 'relu', kernel_regularizer=regularizers.l2(0.0001)))

    model.add(tf.keras.layers.Dense(32, activation= 'relu', kernel_regularizer=regularizers.l2(0.0001)))

    model.add(tf.keras.layers.Dense(1, activation= 'sigmoid')) # output layer using nodes because there are 2 outputs
          
    model.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'])
    
    return model

def compileHeftyModel():
    model = tf.keras.models.Sequential() #we are using a feed forward model.
    model.add(tf.keras.layers.Dense(201, activation= 'relu')) #this is the first dense layer using the recified linear activation function

    model.add(tf.keras.layers.Dense(512, activation= 'relu', kernel_regularizer=regularizers.l2(0.0001)))

    model.add(tf.keras.layers.Dense(1024, activation= 'relu', kernel_regularizer=regularizers.l2(0.0001)))

    model.add(tf.keras.layers.Dense(512, activation= 'relu', kernel_regularizer=regularizers.l2(0.0001))) # output layer using nodes because there are 2 outputs

    model.add(tf.keras.layers.Dense(256, activation= 'relu', kernel_regularizer=regularizers.l2(0.0001))) # output layer using nodes because there are 2 outputs

    model.add(tf.keras.layers.Dense(128, activation= 'relu', kernel_regularizer=regularizers.l2(0.0001))) # output layer using nodes because there are 2 outputs

    model.add(tf.keras.layers.Dense(32, activation= 'relu', kernel_regularizer=regularizers.l2(0.0001))) # output layer using nodes because there are 2 outputs

    model.add(tf.keras.layers.Dense(1, activation= 'sigmoid')) # output layer using nodes because there are 2 outputs
          
    model.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'])
    return model

def compileHughHeftyModel():
    #this model is similar to the heftyModel but the training data is 'richer'

    model = tf.keras.models.Sequential() #we are using a feed forward model.
    model.add(tf.keras.layers.Dense(199, activation= 'relu')) #this is the first dense layer using the recified linear activation function

    model.add(tf.keras.layers.Dense(512, activation= 'relu', kernel_regularizer=regularizers.l2(0.0001)))

    model.add(tf.keras.layers.Dense(1024, activation= 'relu', kernel_regularizer=regularizers.l2(0.0001)))

    model.add(tf.keras.layers.Dense(512, activation= 'relu', kernel_regularizer=regularizers.l2(0.0001))) # output layer using nodes because there are 2 outputs

    model.add(tf.keras.layers.Dense(256, activation= 'relu', kernel_regularizer=regularizers.l2(0.0001))) # output layer using nodes because there are 2 outputs

    model.add(tf.keras.layers.Dense(128, activation= 'relu', kernel_regularizer=regularizers.l2(0.0001))) # output layer using nodes because there are 2 outputs

    model.add(tf.keras.layers.Dense(32, activation= 'relu', kernel_regularizer=regularizers.l2(0.0001))) # output layer using nodes because there are 2 outputs

    model.add(tf.keras.layers.Dense(1, activation= 'sigmoid')) # output layer using nodes because there are 2 outputs
          
    model.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'])

    return model


