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
build_model = __import__('1-input').build_model
optimize_model = __import__('2-optimize').optimize_model
one_hot = __import__('3-one_hot').one_hot
train_model = __import__('4-train').train_model


if __name__ == '__main__':
    datasets = np.load('../data/MNIST.npz')
    X_train = datasets['X_train']
    X_train = X_train.reshape(X_train.shape[0], -1)
    Y_train = datasets['Y_train']
    Y_train_oh = one_hot(Y_train)

    lambtha = 0.0001
    keep_prob = 0.95
    network = build_model(
        784, [
            256, 256, 10], [
            'relu', 'relu', 'softmax'], lambtha, keep_prob)
    alpha = 0.001
    beta1 = 0.9
    beta2 = 0.999
    optimize_model(network, alpha, beta1, beta2)
    batch_size = 64
    epochs = 5
    train_model(network, X_train, Y_train_oh, batch_size, epochs)
