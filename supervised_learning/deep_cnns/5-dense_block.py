#!/usr/bin/env python3
"""DCNN - DenseNet-121"""

import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """Builds a dense block for a DenseNet"""
    for i in range(layers):
        # Batch Normalization -> ReLU -> 1x1 Convolution (Bottleneck layer)
        X1 = K.layers.BatchNormalization()(X)
        X1 = K.layers.ReLU()(X1)
        X1 = K.layers.Conv2D(4 * growth_rate,
                             (1, 1),
                             kernel_initializer='he_normal')(X1)

        # Batch Normalization -> ReLU -> 3x3 Convolution
        X1 = K.layers.BatchNormalization()(X1)
        X1 = K.layers.ReLU()(X1)
        X1 = K.layers.Conv2D(growth_rate,
                             (3, 3), padding='same',
                             kernel_initializer='he_normal')(X1)

        # Concatenate X1 and the input tensor
        X = K.layers.Concatenate()([X, X1])

        nb_filters += growth_rate

    return X, nb_filters
