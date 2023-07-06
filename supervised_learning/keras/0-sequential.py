#!/usr/bin/env python3
"""Keras module"""
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2


def build_model(nx, layers, activations, lambtha, keep_prob):
    """builds a neural network with the Keras library"""
    model = Sequential()
    for i in range(len(layers)):
        if i == 0:
            model.add(
                Dense(
                    layers[i],
                    input_dim=nx,
                    activation=activations[i],
                    kernel_regularizer=l2(lambtha)))
        else:
            model.add(
                Dense(
                    layers[i],
                    activation=activations[i],
                    kernel_regularizer=l2(lambtha)))
        if i < len(layers) - 1:
            model.add(Dropout(1 - keep_prob))
    return model
