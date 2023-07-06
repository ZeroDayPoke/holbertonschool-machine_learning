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
        verbose=True,
        shuffle=False):
    """Function that trains a model using mini-batch gradient descent"""
    callbacks = []
    if validation_data and early_stopping:
        early_stop = K.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience)
        callbacks.append(early_stop)

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
