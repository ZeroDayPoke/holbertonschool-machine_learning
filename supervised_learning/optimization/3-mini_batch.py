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
    Trains a loaded neural network model using mini-batch gradient descent
    Blog post on mini-batch gradient descent: zerodaypoke.com
    """

    # Create a session and import the meta graph
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + ".meta")
        saver.restore(sess, load_path)

        # Get the tensors
        graph = tf.get_default_graph()
        x = graph.get_collection("x")[0]
        y = graph.get_collection("y")[0]
        accuracy = graph.get_collection("accuracy")[0]
        loss = graph.get_collection("loss")[0]
        train_op = graph.get_collection("train_op")

        # Get the number of steps
        m = X_train.shape[0]
        steps = m // batch_size
        if steps * batch_size < m:
            steps += 1

        # Calculate and print cost and accuracy for 0th epoch
        train_cost, train_accuracy = 0., 0.
        for step in range(steps):
            start = step * batch_size
            end = (step + 1) * batch_size if (step + 1) * \
                batch_size <= m else m
            X_batch = X_train[start:end]
            Y_batch = Y_train[start:end]
            batch_cost, batch_accuracy = sess.run(
                [loss, accuracy], feed_dict={x: X_batch, y: Y_batch})
            train_cost += batch_cost / steps
            train_accuracy += batch_accuracy / steps

        valid_cost, valid_accuracy = sess.run(
            [loss, accuracy], feed_dict={x: X_valid, y: Y_valid})

        print("After 0 epochs:")
        print("\tTraining Cost: {}".format(train_cost))
        print("\tTraining Accuracy: {}".format(train_accuracy))
        print("\tValidation Cost: {}".format(valid_cost))
        print("\tValidation Accuracy: {}".format(valid_accuracy))

        for epoch in range(epochs):
            X_train, Y_train = shuffle_data(X_train, Y_train)
            epoch_cost = 0.
            epoch_accuracy = 0.
            for step in range(steps):
                start = step * batch_size
                end = (step + 1) * batch_size if (step + 1) * \
                    batch_size <= m else m
                X_batch = X_train[start:end]
                Y_batch = Y_train[start:end]
                _, batch_cost, batch_accuracy = sess.run(
                    [train_op, loss, accuracy],
                    feed_dict={x: X_batch, y: Y_batch})
                epoch_cost += batch_cost / steps
                epoch_accuracy += batch_accuracy / steps
                if step != 0 and (step + 1) % 100 == 0:
                    print("\tStep {}:".format(step + 1))
                    print("\t\tCost: {}".format(batch_cost))
                    print("\t\tAccuracy: {}".format(batch_accuracy))

            valid_cost, valid_accuracy = sess.run(
                [loss, accuracy], feed_dict={x: X_valid, y: Y_valid})
            print("After {} epochs:".format(epoch + 1))
            print("\tTraining Cost: {}".format(epoch_cost))
            print("\tTraining Accuracy: {}".format(epoch_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))

        save_path = saver.save(sess, save_path)
    return save_path
