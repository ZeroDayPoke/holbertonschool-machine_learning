#!/usr/bin/env python3
"""Test Model Module"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """Function that tests a neural network"""
    if verbose:
        print("Testing model...")
    return network.evaluate(x=data, y=labels, verbose=verbose)
