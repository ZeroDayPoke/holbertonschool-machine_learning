#!/usr/bin/env python3
"""DCNN - Projection Block"""
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """Projection Block"""
    F11, F3, F12 = filters

    X = K.layers.Conv2D(F11, (1, 1),
                        strides=s,
                        kernel_initializer='he_normal')(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(F3, (3, 3), padding='same',
                        kernel_initializer='he_normal')(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(F12, (1, 1), kernel_initializer='he_normal')(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    A_shortcut = K.layers.Conv2D(F12, (1, 1), strides=s,
                                 kernel_initializer='he_normal')(A_prev)
    A_shortcut = K.layers.BatchNormalization(axis=3)(A_shortcut)

    X = K.layers.Add()([X, A_shortcut])
    X = K.layers.Activation('relu')(X)

    return X
