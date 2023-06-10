#!/usr/bin/env python3
"""Tensorflow module"""

import tensorflow as tf


def evaluate(X, Y, save_path):
    """Tests a neural network"""
    sess = tf.Session()
    saved = tf.train.import_meta_graph(save_path + '.meta')
    saved.restore(sess, save_path)
    graph = tf.get_default_graph()
    x = graph.get_collection("x")[0]
    y = graph.get_collection("y")[0]
    y_pred = graph.get_collection("y_pred")[0]
    accuracy = graph.get_collection("accuracy")[0]
    loss = graph.get_collection("loss")[0]
    return tuple(sess.run([y_pred, accuracy, loss], feed_dict={x: X, y: Y}))
