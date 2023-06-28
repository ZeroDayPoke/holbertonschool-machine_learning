#!/usr/bin/env python3
"""Dropout regularization module"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    checker compliant docstring
    """
    outputs = {}
    outputs["A0"] = X
    for layer in range(1, L+1):
        W = weights["W{}".format(layer)]
        A = outputs["A{}".format(layer-1)]
        b = weights["b{}".format(layer)]
        Z = np.matmul(W, A) + b

        if layer == L:
            exponentiated_values = np.exp(Z)
            outputs["A{}".format(layer)] = exponentiated_values / np.sum(
                exponentiated_values, axis=0
            )

        else:
            top = np.exp(Z) - np.exp(-Z)
            bot = np.exp(Z) + np.exp(-Z)
            A = top / bot

            dX = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
            outputs["D{}".format(layer)] = dX*1
            A *= dX
            A /= keep_prob
            outputs["A{}".format(layer)] = A

    return outputs
