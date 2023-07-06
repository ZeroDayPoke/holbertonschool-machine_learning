#!/usr/bin/env python3
"""Function that sets up Adam optimization for a keras model with"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """Function that sets up Adam optimization for a keras model with"""
    opt = K.optimizers.Adam(lr=alpha, beta_1=beta1, beta_2=beta2)
    network.compile(optimizer=opt,
                    loss='categorical_crossentropy', metrics=['accuracy'])
