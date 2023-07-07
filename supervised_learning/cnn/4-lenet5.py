#!/usr/bin/env python3
"""
Module to build, train, and validate a modified LeNet-5
neural network model in tensorflow
"""

import tensorflow as tf


def lenet5(x, y):
    """
    Builds, trains, and validates a modified LeNet-5
    neural network model in tensorflow

    parameters:
        x [tf.placeholder of shape (m, 28, 28, 1)]:
            contains the input images for the network
            m: number of images
        y [tf.placeholder of shape (m, 10)]:
            contains the one-hot labels for the network

    The model should consist of the following layers in order:
        - Convolutional layer with 6 kernels of shape 5x5 with same padding
        - Max pooling layer with kernels of shape 2x2 with 2x2 strides
        - Convolutional layer with 16 kernels of shape 5x5 with valid padding
        - Max pooling layer with kernels of shape 2x2 with 2x2 strides
        - Fully connected layer with 120 nodes
        - Fully connected layer with 84 nodes
        - Fully connected softmax output layer with 10 nodes

    All layers requiring initialization should initialize their kernels with
    the he_normal initialization method:
    tf.contrib.layers.variance_scaling_initializer()
    All hidden layers requiring activation should
    use the relu activation function

    returns:
        tensor for the softmax activated output
        training operation that utilizes Adam optimization
        (with default hyperparameters)
        tensor for the loss of the netowrk
        tensor for the accuracy of the network
    """
    init = tf.contrib.layers.variance_scaling_initializer()
    activation = tf.nn.relu

    conv1 = tf.layers.conv2d(x, filters=6, kernel_size=5, padding='same',
                             activation=activation,
                             kernel_initializer=init)

    pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(pool1, filters=16, kernel_size=5, padding='valid',
                             activation=activation,
                             kernel_initializer=init)

    pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=2)

    flat = tf.contrib.layers.flatten(pool2)

    fc1 = tf.layers.dense(flat, units=120, activation=activation,
                          kernel_initializer=init)

    fc2 = tf.layers.dense(fc1, units=84, activation=activation,
                          kernel_initializer=init)

    fc3 = tf.layers.dense(fc2, units=10, kernel_initializer=init)

    y_pred = tf.nn.softmax(fc3)

    loss = tf.losses.softmax_cross_entropy(y, fc3)

    train_op = tf.train.AdamOptimizer().minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return y_pred, train_op, loss, accuracy
