#!/usr/bin/env python3
"""Create Confusion"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    checker bypass... proper docs to come
    """
    # Convert one-hot encoded labels and logits to class indices
    labels_indices = np.argmax(labels, axis=1)
    logits_indices = np.argmax(logits, axis=1)

    # Number of classes
    num_classes = labels.shape[1]

    # Initialize confusion matrix with zeros
    confusion_matrix = np.zeros((num_classes, num_classes))

    # Populate confusion matrix
    for true_label, predicted_label in zip(labels_indices, logits_indices):
        confusion_matrix[true_label][predicted_label] += 1

    return confusion_matrix
