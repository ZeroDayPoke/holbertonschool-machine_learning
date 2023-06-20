#!/usr/bin/env python3
"""Mini-Batch"""
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train,
                     X_valid, Y_valid,
                     batch_size=32, epochs=5,
                     load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    Trains a loaded neural network model using mini-batch gradient descent.

    Args:
        X_train (np.ndarray): shape (m, 784) containing the training data.
        Y_train (np.ndarray): shape (m, 10) containing the training labels.
        X_valid (np.ndarray): shape (m, 784) containing the validation data.
        Y_valid (np.ndarray): shape (m, 10) containing the validation labels.
        batch_size (int): number of data points in a batch.
        epochs (int): number of times the training should pass through the whole dataset.
        load_path (str): path from which to load the model.
        save_path (str): path to where the model should be saved after training.

    Returns:
        str: the path where the model was saved.
    """

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + ".meta")
        saver.restore(sess, load_path)

        graph = tf.get_default_graph()
        x = graph.get_collection("x")[0]
        y = graph.get_collection("y")[0]
        accuracy = graph.get_collection("accuracy")[0]
        loss = graph.get_collection("loss")[0]
        train_op = graph.get_collection("train_op")

        m = X_train.shape[0]
        steps = m // batch_size
        if steps * batch_size < m:
            steps += 1

        for epoch in range(epochs + 1):
            X_shuff, Y_shuff = shuffle_data(X_train, Y_train)

            train_cost, train_accuracy = sess.run(
                [loss, accuracy], feed_dict={x: X_train, y: Y_train})
            valid_cost, valid_accuracy = sess.run(
                [loss, accuracy], feed_dict={x: X_valid, y: Y_valid})

            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))

            if epoch < epochs:
                for step in range(steps):
                    start = step * batch_size
                    end = step * batch_size + batch_size
                    if end > m:
                        end = m
                    X_batch = X_shuff[start:end]
                    Y_batch = Y_shuff[start:end]
                    sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})

                    if step != 0 and (step + 1) % 100 == 0:
                        mb_cost, mb_accuracy = sess.run(
                            [loss, accuracy],
                            feed_dict={x: X_batch, y: Y_batch})
                        print("\tStep {}:".format(step + 1))
                        print("\t\tCost: {}".format(mb_cost))
                        print("\t\tAccuracy: {}".format(mb_accuracy))

        save_path = saver.save(sess, save_path)

    return save_path
