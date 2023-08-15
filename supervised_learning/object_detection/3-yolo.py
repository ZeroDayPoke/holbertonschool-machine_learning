#!/usr/bin/env python3
"""Object Detection - 3. Yolo"""

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
        """Documentation"""
        boxes = []
        box_confidences = []
        box_class_probs = []

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            # Box coordinates
            tx = output[..., 0:1]
            ty = output[..., 1:2]
            tw = output[..., 2:3]
            th = output[..., 3:4]

            box_confidence = 1 / (1 + np.exp(-output[..., 4:5]))
            box_class_prob = 1 / (1 + np.exp(-output[..., 5:]))

            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_prob)

            for cy in range(grid_height):
                for cx in range(grid_width):
                    for b in range(anchor_boxes):
                        pw, ph = self.anchors[i][b]

                        bx = (1 / (1 + np.exp(-tx[cy, cx, b]))) + cx
                        by = (1 / (1 + np.exp(-ty[cy, cx, b]))) + cy
                        bw = pw * np.exp(tw[cy, cx, b])
                        bh = ph * np.exp(th[cy, cx, b])

                        bx /= grid_width
                        by /= grid_height
                        bw /= int(self.model.input.shape[1])
                        bh /= int(self.model.input.shape[2])

                        x1 = (bx - (bw / 2)) * image_size[1]
                        y1 = (by - (bh / 2)) * image_size[0]
                        x2 = (bx + (bw / 2)) * image_size[1]
                        y2 = (by + (bh / 2)) * image_size[0]

                        tx[cy, cx, b] = x1
                        ty[cy, cx, b] = y1
                        tw[cy, cx, b] = x2
                        th[cy, cx, b] = y2

            boxes.append(np.concatenate((tx, ty, tw, th), axis=-1))

        return (boxes, box_confidences, box_class_probs)

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter box outputs."""
        box_scores = []
        for i in range(len(boxes)):
            box_scores.append(box_confidences[i] * box_class_probs[i])

        # Getting the box classes and their respective scores
        box_classes = [np.argmax(box_score, axis=-1)
                       for box_score in box_scores]
        box_class_scores = [np.max(box_score, axis=-1)
                            for box_score in box_scores]

        # Filtering the boxes
        prediction_mask = [box_class_score >= self.class_t
                           for box_class_score in box_class_scores]

        # Using the mask to filter out boxes, their classes and scores
        filtered_boxes = [box[mask] for
                          box, mask in zip(boxes, prediction_mask)]
        box_classes = [box_class[mask]
                       for box_class, mask
                       in zip(box_classes, prediction_mask)]
        box_scores = [box_class_score[mask]
                      for box_class_score, mask
                      in zip(box_class_scores, prediction_mask)]

        # Flatten the filtered boxes, classes and scores
        filtered_boxes = np.concatenate(filtered_boxes)
        box_classes = np.concatenate(box_classes)
        box_scores = np.concatenate(box_scores)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Apply non-max suppression on bounding boxes."""

        # Sort by box scores
        sorted_indices = np.argsort(box_scores)[::-1]
        filtered_boxes = filtered_boxes[sorted_indices]
        box_classes = box_classes[sorted_indices]
        box_scores = box_scores[sorted_indices]

        n = len(filtered_boxes)
        keep = []

        while filtered_boxes.shape[0]:
            # The box with highest score
            i = 0
            keep.append(i)

            # Compute IoU of this box with rest
            iou = self._iou(filtered_boxes[i], filtered_boxes[1:])

            # Find where overlap is below threshold
            remove = np.where(iou > self.nms_t)[0] + 1

            # Remove indices of overlapping boxes
            filtered_boxes = np.delete(filtered_boxes, remove, axis=0)
            box_classes = np.delete(box_classes, remove, axis=0)
            box_scores = np.delete(box_scores, remove, axis=0)

        # Use keep to filter the boxes, scores, and classes
        box_predictions = filtered_boxes[keep]
        predicted_box_classes = box_classes[keep]
        predicted_box_scores = box_scores[keep]

        return box_predictions, predicted_box_classes, predicted_box_scores

    def _iou(self, boxA, boxB_list):
        """Compute the Intersection Over Union (IoU) of two boxes."""

        # Determine the (x, y)-coordinates of the intersection rectangle
        xA = np.maximum(boxA[0], boxB_list[:, 0])
        yA = np.maximum(boxA[1], boxB_list[:, 1])
        xB = np.minimum(boxA[2], boxB_list[:, 2])
        yB = np.minimum(boxA[3], boxB_list[:, 3])

        # Compute the area of intersection rectangle
        interArea = np.maximum(0, xB - xA + 1) * np.maximum(0, yB - yA + 1)

        # Compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB_list[:, 2] -
                    boxB_list[:, 0] + 1) * (boxB_list[:, 3]
                                            - boxB_list[:, 1] + 1)

        # Compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area
        iou = interArea / (boxAArea + boxBArea - interArea)

        return iou
