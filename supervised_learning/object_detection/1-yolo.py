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

    def process_outputs(self, outputs, image_size):
        """
        Process Outputs.

        Args:
            outputs (list of numpy.ndarray): List of arrays containing the predictions
                                            from the Darknet model for a single image.
            image_size (numpy.ndarray): Array containing the image’s original size 
                                        [image_height, image_width].

        Returns:
            tuple: (boxes, box_confidences, box_class_probs)
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            # Get the meshgrid for the height and width
            cx = np.tile(np.arange(grid_width), (grid_height, 1)).reshape(grid_height, grid_width, 1)
            cy = np.tile(np.arange(grid_height), (grid_width, 1)).T.reshape(grid_height, grid_width, 1)
            
            cx = cx[..., np.newaxis].repeat(anchor_boxes, axis=-1)
            cy = cy[..., np.newaxis].repeat(anchor_boxes, axis=-1)

            # Box coordinates
            tx = output[..., 0:1]
            ty = output[..., 1:2]
            tw = output[..., 2:3]
            th = output[..., 3:4]

            pw = self.anchors[i, :, 0].reshape(1, 1, anchor_boxes, 1)
            ph = self.anchors[i, :, 1].reshape(1, 1, anchor_boxes, 1)

            bx = self.sigmoid(tx) + cx
            by = self.sigmoid(ty) + cy
            bw = pw * np.exp(tw)
            bh = ph * np.exp(th)

            # Normalize
            bx /= grid_width
            by /= grid_height
            bw /= float(self.model.input_shape[1])
            bh /= float(self.model.input_shape[2])

            # Convert (center_x, center_y, width, height) --> (x1, y1, x2, y2)
            x1 = (bx - bw / 2) * image_size[1]
            y1 = (by - bh / 2) * image_size[0]
            x2 = (bx + bw / 2) * image_size[1]
            y2 = (by + bh / 2) * image_size[0]

            # Constrain boxes within the image dimensions
            x1 = np.clip(x1, 0, image_size[1])
            y1 = np.clip(y1, 0, image_size[0])
            x2 = np.clip(x2, 0, image_size[1])
            y2 = np.clip(y2, 0, image_size[0])

            box = np.concatenate((x1, y1, x2, y2), axis=-1)
            boxes.append(box)

            # Box confidence
            confidence = self.sigmoid(output[..., 4:5])
            box_confidences.append(confidence)

            # Box class probabilities
            class_probs = self.sigmoid(output[..., 5:])
            box_class_probs.append(class_probs)

        return (boxes, box_confidences, box_class_probs)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
