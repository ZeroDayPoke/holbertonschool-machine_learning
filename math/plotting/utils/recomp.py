#!/usr/bin/env python3
"""WSL probs; gotta recompress"""
import numpy as np
data = np.load('pca/data.npy')
labels = np.load('pca/labels.npy')
np.savez('pca.npz', data=data, labels=labels)
