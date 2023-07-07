#!/usr/bin/env python3
"""Config"""
import tensorflow.keras as K


def save_config(network, filename):
    """Function that saves a model’s configuration"""
    config = network.to_json()
    with open(filename, 'w') as f:
        f.write(config)


def load_config(filename):
    """Function that loads a model with a specific configuration"""
    with open(filename, 'r') as f:
        config = f.read()
    network = K.models.model_from_json(config)
    return network
