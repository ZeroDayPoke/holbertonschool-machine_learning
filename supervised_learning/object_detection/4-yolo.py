#!/usr/bin/env python3
"""Object Detection - 4. Yolo"""

import os
import cv2
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
        """Performs non-max suppression"""
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        unique_classes = set(box_classes)

        for c in unique_classes:
            idxs = np.where(box_classes == c)
            class_boxes = filtered_boxes[idxs]
            class_box_scores = box_scores[idxs]

            # Sort boxes by score
            sorted_idxs = np.argsort(class_box_scores)[::-1]
            class_boxes = class_boxes[sorted_idxs]
            class_box_scores = class_box_scores[sorted_idxs]

            while len(class_boxes) > 0:
                # Take the box with the highest score
                box_predictions.append(class_boxes[0])
                predicted_box_classes.append(c)
                predicted_box_scores.append(class_box_scores[0])

                # If only one box left, break
                if len(class_boxes) == 1:
                    break

                # Calculate IOU for the remaining boxes with the top box
                iou = self.intersection_over_union(class_boxes[0],
                                                   class_boxes[1:])
                iou_mask = iou < self.nms_t

                # Update class_boxes and class_box_scores using the iou_mask
                class_boxes = class_boxes[1:][iou_mask]
                class_box_scores = class_box_scores[1:][iou_mask]

        # Convert results into numpy arrays
        box_predictions = np.array(box_predictions)
        predicted_box_classes = np.array(predicted_box_classes)
        predicted_box_scores = np.array(predicted_box_scores)

        return box_predictions, predicted_box_classes, predicted_box_scores

    def intersection_over_union(self, box1, boxes):
        """Calculates intersection over union"""
        x1 = np.maximum(box1[0], boxes[:, 0])
        y1 = np.maximum(box1[1], boxes[:, 1])
        x2 = np.minimum(box1[2], boxes[:, 2])
        y2 = np.minimum(box1[3], boxes[:, 3])

        intersection_area = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        union_area = box1_area + boxes_area - intersection_area

        return intersection_area / union_area

    @staticmethod
    def load_images(folder_path):
        """Loads images from a folder."""
        image_files = [os.path.join(folder_path, f)
                       for f in os.listdir(folder_path)
                       if f.endswith(('.jpg', '.jpeg', '.png'))]

        images = [cv2.imread(image_file) for image_file in image_files]

        return images, image_files
