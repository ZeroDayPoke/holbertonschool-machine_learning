#!/usr/bin/env python3
""" Yolo class """

from tensorflow.keras.models import load_model
import numpy as np

class Yolo:
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """_summary_

        Args:
            model_path (_type_): _description_
            classes_path (_type_): _description_
            class_t (_type_): _description_
            nms_t (_type_): _description_
            anchors (_type_): _description_
        """
        self.model = load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
