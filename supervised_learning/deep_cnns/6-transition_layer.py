#!/usr/bin/env python3
"""DCNN - DenseNet-121"""

import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """Builds a transition layer for a DenseNet"""
    # Batch Normalization -> ReLU -> 1x1 Convolution
    X = K.layers.BatchNormalization()(X)
    X = K.layers.ReLU()(X)
    X = K.layers.Conv2D(int(nb_filters * compression), (1, 1), kernel_initializer='he_normal')(X)
    
    # Average Pooling
    X = K.layers.AveragePooling2D((2, 2))(X)
    
    return X, int(nb_filters * compression)
