#!/usr/bin/env python3
"""Mini-Batch"""
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train,
                     X_valid, Y_valid,
                     batch_size=32, epochs=5,
                     load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """trains a loaded model using mini-batch gradient descent"""
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("{}.meta".format(load_path))
        saver.restore(sess, load_path)

        graph = tf.get_default_graph()

        x = graph.get_tensor_by_name("x:0")
        y = graph.get_tensor_by_name("y:0")
        accuracy = graph.get_tensor_by_name("accuracy:0")
        loss = graph.get_tensor_by_name("loss:0")
        train_op = graph.get_operation_by_name("train_op")

        m = X_train.shape[0]
        step_count = 0

        for epoch in range(epochs):
            X_train, Y_train = shuffle_data(X_train, Y_train)
            train_cost, train_accuracy = sess.run(
                [loss, accuracy], feed_dict={x: X_train, y: Y_train})
            valid_cost, valid_accuracy = sess.run(
                [loss, accuracy], feed_dict={x: X_valid, y: Y_valid})

            print('After {} epochs:'.format(epoch))
            print('\tTraining Cost: {}'.format(train_cost))
            print('\tTraining Accuracy: {}'.format(train_accuracy))
            print('\tValidation Cost: {}'.format(valid_cost))
            print('\tValidation Accuracy: {}'.format(valid_accuracy))

            for i in range(0, m, batch_size):
                start = i
                end = i + batch_size

                if end > m:
                    end = m

                X_batch = X_train[start:end]
                Y_batch = Y_train[start:end]

                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})

                if step_count % 100 == 0:
                    step_cost, step_accuracy = sess.run(
                        [loss, accuracy], feed_dict={x: X_batch, y: Y_batch})
                    print('\tStep {}:'.format(step_count))
                    print('\t\tCost: {}'.format(step_cost))
                    print('\t\tAccuracy: {}'.format(step_accuracy))

                step_count += 1

        save_path = saver.save(sess, save_path)

    return save_path
