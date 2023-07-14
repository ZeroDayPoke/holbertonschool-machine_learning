#!/usr/bin/env python3
"""DCNN module"""

import tensorflow.keras as K

def inception_block(A_prev, filters):
    """Builds Block inception"""
    F1, F3R, F3, F5R, F5, FPP = filters

    # 1x1 convolution
    conv_1x1 = K.layers.Conv2D(filters=F1, kernel_size=(1, 1), activation='relu')(A_prev)

    # 3x3 convolution
    conv_3x3_reduce = K.layers.Conv2D(filters=F3R, kernel_size=(1, 1), activation='relu')(A_prev)
    conv_3x3 = K.layers.Conv2D(filters=F3, kernel_size=(3, 3), padding='same', activation='relu')(conv_3x3_reduce)

    # 5x5 convolution
    conv_5x5_reduce = K.layers.Conv2D(filters=F5R, kernel_size=(1, 1), activation='relu')(A_prev)
    conv_5x5 = K.layers.Conv2D(filters=F5, kernel_size=(5, 5), padding='same', activation='relu')(conv_5x5_reduce)

    # pooling
    pool_proj = K.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')(A_prev)
    pool_1x1 = K.layers.Conv2D(filters=FPP, kernel_size=(1, 1), activation='relu')(pool_proj)

    # concatenate
    output = K.layers.concatenate([conv_1x1, conv_3x3, conv_5x5, pool_1x1], axis=-1)

    return output
