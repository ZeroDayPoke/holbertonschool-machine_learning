#!/usr/bin/env python3
"""Object Detection - 1. Yolo"""

import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np


class Yolo:
    """
    Yolo v3 algorithm to perform object detection.

    Attributes:
        model (Keras model): The Darknet Keras model.
        class_names (list): A list of the class names for the model.
        class_t (float): The box score threshold for the initial filtering.
        nms_t (float): The IOU threshold for non-max suppression.
        anchors (numpy.ndarray): The anchor boxes.
    """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        self.model = tf.keras.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        """
        _summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        _summary_

        Args:
            outputs (_type_): _description_
            image_size (_type_): _description_

        Returns:
            _type_: _description_
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            # Box coordinates
            box = output[..., :4]
            for j in range(anchor_boxes):
                anchor = self.anchors[i, j]
                box[..., j, 0:2] = self.sigmoid(box[..., j, 0:2])
                box[..., j, 2:4] = np.exp(box[..., j, 2:4]) * anchor

            # Box confidence
            confidence = self.sigmoid(output[..., 4:5])

            # Box class probabilities
            class_probs = self.sigmoid(output[..., 5:])

            boxes.append(box)
            box_confidences.append(confidence)
            box_class_probs.append(class_probs)

        for i, box in enumerate(boxes):
            grid_height, grid_width, anchor_boxes, _ = box.shape

            # Create a grid
            c = np.zeros((grid_height, grid_width, anchor_boxes, 1), dtype=int)
            idx_y = np.arange(grid_height).reshape(grid_height, 1, 1, 1)
            idx_x = np.arange(grid_width).reshape(1, grid_width, 1, 1)
            cx = c + idx_x
            cy = c + idx_y

            # Combine cx and cy
            cxy = np.concatenate((cx, cy), axis=-1)

            # Set the center of the bounding boxes
            box[..., :2] = (box[..., :2] + cxy) / (grid_width, grid_height)

            # Convert (center_x, center_y, width, height) --> (x1, y1, x2, y2)
            box[..., 0] = box[..., 0] - box[..., 2] / 2
            box[..., 1] = box[..., 1] - box[..., 3] / 2
            box[..., 2] = box[..., 0] + box[..., 2]
            box[..., 3] = box[..., 1] + box[..., 3]

            boxes[i] = box

        return (boxes, box_confidences, box_class_probs)
