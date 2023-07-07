#!/usr/bin/env python3
"""
Module to build, train, and validate a modified LeNet-5
neural network model in keras
"""

import tensorflow.keras as K


def lenet5(X):
    """
    Builds, trains, and validates a modified LeNet-5
    neural network model in keras

    parameters:
        X [K.Input of shape (m, 28, 28, 1)]:
            contains the input images for the network
            m: number of images

    The model should consist of the following layers in order:
        - Convolutional layer with 6 kernels of shape 5x5 with same padding
        - Max pooling layer with kernels of shape 2x2 with 2x2 strides
        - Convolutional layer with 16 kernels of shape 5x5 with valid padding
        - Max pooling layer with kernels of shape 2x2 with 2x2 strides
        - Fully connected layer with 120 nodes
        - Fully connected layer with 84 nodes
        - Fully connected softmax output layer with 10 nodes

    All layers requiring initialization should initialize their kernels with
    the he_normal initialization method
    All hidden layers requiring activation
    should use the relu activation function

    returns:
        a K.Model compiled to use Adam optimization
        (with default hyperparameters) and accuracy metrics
    """
    init = K.initializers.he_normal(seed=None)

    model = K.models.Sequential()

    model.add(K.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='same',
                              activation='relu', kernel_initializer=init,
                              input_shape=(28, 28, 1)))

    model.add(K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(K.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='valid',
                              activation='relu', kernel_initializer=init))

    model.add(K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(K.layers.Flatten())

    model.add(K.layers.Dense(units=120, activation='relu',
                             kernel_initializer=init))

    model.add(K.layers.Dense(units=84, activation='relu',
                             kernel_initializer=init))

    model.add(K.layers.Dense(units=10, activation='softmax',
                             kernel_initializer=init))

    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
