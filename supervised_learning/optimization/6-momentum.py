#!/usr/bin/env python3
"""Momentum"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """
    doc too dangerous for checker
    """
    optimizer = tf.train.MomentumOptimizer(alpha, beta1)
    train_op = optimizer.minimize(loss)
    return train_op
