#!/usr/bin/env python3
"""Predict"""


def predict(network, data, verbose=False):
    """Function that makes a prediction using a neural network"""
    if verbose:
        print("Predicting...")
    return network.predict(data)
