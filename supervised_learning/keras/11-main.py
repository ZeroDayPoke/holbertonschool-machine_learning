#!/usr/bin/env python3
"""
Main file
"""

# Force Seed - fix for Keras
import tensorflow.keras as K
import tensorflow as tf
import numpy as np
import random
import os
SEED = 0

os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.set_random_seed(SEED)
session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.backend.set_session(sess)

# Imports
model = __import__('9-model')
config = __import__('11-config')

if __name__ == '__main__':
    network = model.load_model('network1.h5')
    config.save_config(network, 'config1.json')
    del network

    network2 = config.load_config('config1.json')
    network2.summary()
    print(network2.get_weights())
