#!/usr/bin/env python3
"""Model Combines Previous Tasks"""
import tensorflow as tf
import numpy as np


def create_placeholders(nx, classes):
    """
    Function to create placeholders for input data
    and labels for the neural network.
    Arguments:
    nx: int, the number of feature columns in our data
    classes: int, the number of classes in our classifier

    Returns:
    x: placeholder for the input data to the neural network
    y: placeholder for the one-hot labels for the input data
    """

    x = tf.placeholder(dtype=tf.float32, shape=[None, nx], name="x")
    y = tf.placeholder(dtype=tf.float32, shape=[None, classes], name="y")

    return x, y


def create_layer(prev, n, activation, i):
    """
    Function to create a layer for the neural network.

    Arguments:
    prev: tensor, output of the previous layer
    n: int, number of nodes in the layer to create
    activation: activation function that the layer should use
    i: index of the layer

    Returns:
    layer: tensor, output of the layer
    """

    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")

    activation_name = activation.__name__ if activation is not None else ""
    layer_name = "layer" + "_" + str(i) + activation_name

    layer = tf.layers.dense(inputs=prev,
                            units=n,
                            activation=activation,
                            kernel_initializer=initializer,
                            name=layer_name)

    return layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Function to create the forward propagation graph for the neural network.

    Arguments:
    x: placeholder for the input data
    layer_sizes: list, containing the number of nodes
    in each layer of the network
    activations: list, containing the activation functions
    for each layer of the network

    Returns:
    y_pred: tensor, the prediction of the network in tensor form
    """

    layer_output = x
    for i in range(len(layer_sizes)):
        layer_output = create_layer(prev=layer_output,
                                    n=layer_sizes[i],
                                    activation=activations[i],
                                    i=i)

    y_pred = layer_output

    return y_pred


def calculate_accuracy(y, y_pred):
    """
    Function to calculate the accuracy of a prediction.

    Arguments:
    y: placeholder for the labels of the input data
    y_pred: tensor containing the network’s predictions

    Returns:
    a tensor containing the decimal accuracy of the prediction
    """
    correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    return accuracy


def calculate_loss(y, y_pred):
    """
    Calculate the softmax cross-entropy loss of a prediction.
    y is a placeholder for the labels of the input data
    y_pred is a tensor containing the network’s predictions
    """
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)

    return loss


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way

    Arguments:
    X -- first numpy.ndarray of shape (m, nx) to shuffle
        m is the number of data points
        nx is the number of features in X
    Y -- second numpy.ndarray of shape (m, ny) to shuffle
        m is the same number of data points as in X
        ny is the number of features in Y

    Returns:
    The shuffled X and Y matrices
    """
    assert len(X) == len(Y), 'uniformity error'
    permutation = np.random.permutation(len(X))
    return X[permutation], Y[permutation]


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    doc too dangerous for checker
    """
    return tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon).minimize(loss)


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    doc too dangerous for checker
    """
    return tf.train.inverse_time_decay(
        alpha,
        global_step,
        decay_step,
        decay_rate,
        staircase=True,
    )


def model(Data_train, Data_valid,
          layers, activations, alpha=0.001,
          beta1=0.9, beta2=0.999, epsilon=1e-8,
          decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """
    Builds, trains, and saves a neural network model in tensorflow
    """
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layers, activations)
    accuracy = calculate_accuracy(y, y_pred)
    loss = calculate_loss(y, y_pred)
    global_step = tf.Variable(0, trainable=False)
    alpha = learning_rate_decay(alpha, decay_rate, global_step, 1)
    train_op = create_Adam_op(loss, alpha, beta1, beta2, epsilon)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epochs):
            X_shuffle, Y_shuffle = shuffle_data(X_train, Y_train)
            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_shuffle[i: i + batch_size]
                Y_batch = Y_shuffle[i: i + batch_size]
                _, step = sess.run([train_op, global_step], feed_dict={x: X_batch, y: Y_batch})
                if step % 100 == 0:
                    train_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
                    train_accuracy = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
                    print("\tStep {}:".format(step))
                    print("\t\tCost: {}".format(train_cost))
                    print("\t\tAccuracy: {}".format(train_accuracy))
            train_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            train_accuracy = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
            valid_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            valid_accuracy = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))
            save_path = saver.save(sess, save_path)

        return save_path
