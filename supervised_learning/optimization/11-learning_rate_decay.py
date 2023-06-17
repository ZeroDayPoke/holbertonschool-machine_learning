#!/usr/bin/env python3
"""Learning Rate Decay"""
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    doc too dangerous for checker
    """
    return alpha / (1 + decay_rate * np.floor(global_step / decay_step))
