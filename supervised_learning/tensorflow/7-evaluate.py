#!/usr/bin/env python3
"""Tensorflow module"""

import tensorflow as tf


def evaluate(X, Y, save_path):
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        saver = tf.train.import_meta_graph(save_path + '.meta')
        saver.restore(sess, save_path)

        # Get the tensors from the graph's collection
        y_pred = tf.get_collection('y_pred')[0]
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection('accuracy')[0]

        # Evaluate the network
        feed_dict = {'X:0': X, 'Y:0': Y}
        pred, acc, cost = sess.run([y_pred, accuracy, loss],
                                   feed_dict=feed_dict)

    return pred, acc, cost
