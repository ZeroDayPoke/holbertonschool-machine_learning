#!/usr/bin/env python3
"""Test"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """Function that tests a neural network"""
    if verbose:
        print("Testing model...")
    loss, accuracy = network.evaluate(data, labels, verbose=verbose)
    return loss, accuracy
