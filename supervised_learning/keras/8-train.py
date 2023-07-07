#!/usr/bin/env python3
"""Train"""
import tensorflow.keras as K


def train_model(
        network,
        data,
        labels,
        batch_size,
        epochs,
        validation_data=None,
        early_stopping=False,
        patience=0,
        learning_rate_decay=False,
        alpha=0.1,
        decay_rate=1,
        save_best=False,
        filepath=None,
        verbose=True,
        shuffle=False):
    """Function that trains a model using mini-batch gradient descent"""
    callbacks = []
    if validation_data and early_stopping:
        early_stop = K.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience)
        callbacks.append(early_stop)

    if validation_data and learning_rate_decay:
        def lr_schedule(epoch):
            return alpha / (1 + decay_rate * epoch)

        lr_decay = K.callbacks.LearningRateScheduler(lr_schedule, verbose=1)
        callbacks.append(lr_decay)

    if validation_data and save_best and filepath:
        checkpoint = K.callbacks.ModelCheckpoint(
            filepath, monitor='val_loss', save_best_only=True)
        callbacks.append(checkpoint)

    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data,
        callbacks=callbacks
    )
    return history
