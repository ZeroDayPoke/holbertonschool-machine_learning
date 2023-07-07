#!/usr/bin/env python3
"""
Module to build, train, and validate a modified LeNet-5
neural network model in keras
"""
import tensorflow.keras as K


def lenet5(X):
    """
    Function that builds a modified version of the LeNet-5
    architecture using keras
    Arguments:
     - X is a K.Input of shape (m, 28, 28, 1) containing the input images
        for the network
        * m is the number of images
    Returns:
        A K.Model compiled to use Adam optimization (with default
        hyperparameters) and accuracy metrics
    """
    init = K.initializers.he_normal(seed=None)

    conv1 = K.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='same',
                            activation='relu', kernel_initializer=init)(X)
    pool1 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)

    conv2 = K.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='valid',
                            activation='relu', kernel_initializer=init)(pool1)
    pool2 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)

    flatten = K.layers.Flatten()(pool2)

    fc1 = K.layers.Dense(units=120, activation='relu',
                         kernel_initializer=init)(flatten)
    fc2 = K.layers.Dense(units=84, activation='relu',
                         kernel_initializer=init)(fc1)

    logits = K.layers.Dense(units=10, kernel_initializer=init)(fc2)
    y_pred = K.layers.Activation('softmax')(logits)

    model = K.models.Model(inputs=X, outputs=y_pred)
    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
